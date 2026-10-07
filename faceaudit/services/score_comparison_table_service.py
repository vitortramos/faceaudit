"""Tabela das comparações que produziram os escores.

As instruções de scoring (`SIMILARITY_SCORING`, `PAIR_MINING`,
`TYPICALITY_PAIR_SCORING`, `SCORE_CALIBRATION`) entregavam, na interface,
apenas a distribuição agregada: o perito via a forma das curvas e as
contagens, mas não conseguia saber QUAL escore saiu de QUAL par de faces.
Num confronto de caso isso é a informação central do exame.

Dois regimes, porque o volume é incomparável:

- **dataset de caso** (`metadata["is_casework"]`): são poucas comparações, e
  todas interessam. A tabela traz todas, ordenadas do escore mais alto para
  o mais baixo, que é a ordem em que se lê um confronto.
- **fora de caso**: um conjunto de calibração tem milhões de pares, e
  listá-los seria inútil além de impossível. O que importa na auditoria são
  os **extremos**, que é onde o sistema erra: os genuínos de menor escore
  (mesma pessoa que o sistema separou) e os impostores de maior escore
  (pessoas diferentes que o sistema aproximou).

Duas fontes, com a mesma saída:

- o payload em memória, que carrega o nome do arquivo de cada face;
- o parquet em disco, que guarda identidade e `face_hash` mas **não** o nome
  do arquivo. Ler do disco é o caminho normal (um dataset de milhões de
  pares não fica hidratado), então a tabela identifica a face pelo hash
  abreviado nesse caso, e o campo `has_file_names` avisa quem exibe.

A varredura do parquet é por lote, com duas pilhas limitadas a `k`
elementos: percorrer 5 milhões de linhas não aloca 5 milhões de objetos nem
constrói DataFrame intermediário.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

# Rótulos dos regimes. Expostos como constantes porque a interface decide
# texto de cabeçalho a partir deles.
MODO_CASO = "caso"
MODO_EXTREMOS = "extremos"

# Classe de cada linha. `CLASSE_CASO` é a comparação de caso, cujo ground
# truth é desconhecido por definição — não é "nem genuíno nem impostor" por
# falta de dado, é a pergunta que o exame responde.
# Num conjunto de caso a classe é SEMPRE `CLASSE_CASO`, mesmo que o campo
# `same_person` dos pares traga um resíduo: `SimilarityScoreSample` infere
# ground truth comparando as strings de identidade, e em casework elas são
# convencionais ("questionada", "padrão"), o que marcaria todo par como
# impostor. Exibir isso seria afirmar a conclusão do exame no lugar dele.
CLASSE_GENUINO = "genuino"
CLASSE_IMPOSTOR = "impostor"
CLASSE_CASO = "caso"

# Quantos extremos de cada classe fora do modo caso. 50 cabe na tela com
# rolagem curta e já mostra a cauda que interessa.
EXTREMOS_POR_CLASSE_PADRAO = 50

# Teto do modo caso. Um dataset de caso tem tipicamente dezenas de pares; o
# teto existe só para que um dataset marcado como caso por engano não
# despeje milhões de linhas na interface.
LIMITE_CASO_PADRAO = 5_000

# Tamanho do lote de leitura do parquet.
_LINHAS_POR_LOTE = 65_536

# Caracteres do hash mostrados quando o nome do arquivo não está disponível.
_PREFIXO_HASH = 10

# Ressalva acrescentada à nota quando a origem é o parquet de escores, que
# não guarda nome de arquivo. `attach_face_images` a retira quando recupera
# os nomes do dataset.
_RESSALVA_SEM_NOME = (
    " As faces aparecem pelo início do hash porque o nome do arquivo não é "
    "gravado no parquet de escores."
)


@dataclass(frozen=True)
class ComparisonRow:
    """Uma comparação: as duas faces, o escore e a classe.

    Os caminhos de imagem são opcionais porque o parquet de escores não os
    guarda — quem quiser as miniaturas chama `attach_face_images` com os
    parquets dos datasets de origem.
    """

    probe_identity: str
    probe_face: str
    reference_identity: str
    reference_face: str
    score: float
    same_person: Optional[bool]
    log_lr: Optional[float]
    classe: str
    probe_hash: str = ""
    reference_hash: str = ""
    probe_image_path: Optional[str] = None
    reference_image_path: Optional[str] = None

    @property
    def classe_label(self) -> str:
        if self.classe == CLASSE_GENUINO:
            return "Mesma pessoa"
        if self.classe == CLASSE_IMPOSTOR:
            return "Pessoas diferentes"
        return "Caso"


@dataclass(frozen=True)
class ComparisonTable:
    """As linhas a exibir mais o que explica a seleção delas."""

    rows: Tuple[ComparisonRow, ...]
    mode: str
    total_pairs: int
    n_genuines: int
    n_impostors: int
    truncated: bool
    has_log_lr: bool
    has_file_names: bool
    note: str
    #: True quando ao menos uma linha tem imagem localizada, e portanto a
    #: tabela pode ser exibida com as duas faces lado a lado.
    has_images: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.rows


# ── Construção a partir do payload em memória ────────────────────────────

def build_from_dataset(
    dataset: Any,
    *,
    extremos_por_classe: int = EXTREMOS_POR_CLASSE_PADRAO,
    limite_caso: int = LIMITE_CASO_PADRAO,
) -> ComparisonTable:
    """Monta a tabela a partir de um `SimilarityScoreDataset` hidratado."""
    items: Sequence[Any] = list(getattr(dataset, "items", None) or [])
    metadata = getattr(dataset, "metadata", None) or {}
    is_casework = bool(metadata.get("is_casework", False))

    if not items:
        return _tabela_vazia(is_casework)

    if is_casework:
        selecionadas = items[:limite_caso]
        linhas = [_linha_do_item(s, de_caso=True) for s in selecionadas]
        linhas.sort(key=lambda r: r.score, reverse=True)
        return _montar(
            linhas,
            mode=MODO_CASO,
            total_pairs=len(items),
            truncated=len(items) > len(selecionadas),
            extremos_por_classe=extremos_por_classe,
        )

    contador = itertools.count()
    menores_genuinos: List[tuple] = []   # max-heap por -score: guarda os k menores
    maiores_impostores: List[tuple] = []  # min-heap por score: guarda os k maiores
    n_gen = n_imp = 0

    for s in items:
        mesma = getattr(s, "same_person", None)
        try:
            score = float(getattr(s, "raw_score"))
        except (TypeError, ValueError):
            continue
        if mesma is True:
            n_gen += 1
            _empilhar(menores_genuinos, (-score, next(contador), s), extremos_por_classe)
        elif mesma is False:
            n_imp += 1
            _empilhar(maiores_impostores, (score, next(contador), s), extremos_por_classe)

    genuinos = sorted(
        (_linha_do_item(t[2]) for t in menores_genuinos), key=lambda r: r.score,
    )
    impostores = sorted(
        (_linha_do_item(t[2]) for t in maiores_impostores),
        key=lambda r: r.score, reverse=True,
    )
    return _montar(
        genuinos + impostores,
        mode=MODO_EXTREMOS,
        total_pairs=len(items),
        truncated=(n_gen > len(genuinos)) or (n_imp > len(impostores)),
        extremos_por_classe=extremos_por_classe,
        n_genuines=n_gen,
        n_impostors=n_imp,
    )


# ── Construção a partir do parquet ───────────────────────────────────────

def build_from_parquet(
    parquet_path: Path | str,
    *,
    extremos_por_classe: int = EXTREMOS_POR_CLASSE_PADRAO,
    limite_caso: int = LIMITE_CASO_PADRAO,
) -> Optional[ComparisonTable]:
    """Monta a tabela lendo o parquet por lote.

    Devolve `None` quando o arquivo não existe ou não tem o schema de
    escores — quem chama decide o que mostrar, e inventar uma tabela vazia
    aqui faria o caso "arquivo ausente" parecer "nenhuma comparação".
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None

    caminho = Path(parquet_path)
    if not caminho.is_file():
        return None

    try:
        arquivo = pq.ParquetFile(caminho)
    except Exception:
        return None

    nomes = set(arquivo.schema_arrow.names)
    if "raw_score" not in nomes or "same_person" not in nomes:
        return None

    colunas = [
        c for c in (
            "probe_identity", "reference_identity",
            "probe_face_hash", "reference_face_hash",
            "same_person", "raw_score", "log_lr",
        ) if c in nomes
    ]
    tem_log_lr = "log_lr" in nomes
    total = int(arquivo.metadata.num_rows)
    is_casework = _parquet_e_de_caso(arquivo)

    if total == 0:
        return _tabela_vazia(is_casework, has_log_lr=tem_log_lr, has_file_names=False)

    if is_casework:
        linhas: List[ComparisonRow] = []
        for lote in arquivo.iter_batches(
            batch_size=_LINHAS_POR_LOTE, columns=colunas,
        ):
            for registro in lote.to_pylist():
                linhas.append(_linha_do_registro(registro, de_caso=True))
                if len(linhas) >= limite_caso:
                    break
            if len(linhas) >= limite_caso:
                break
        linhas.sort(key=lambda r: r.score, reverse=True)
        return _montar(
            linhas, mode=MODO_CASO, total_pairs=total,
            truncated=total > len(linhas),
            extremos_por_classe=extremos_por_classe,
            has_log_lr=tem_log_lr, has_file_names=False,
        )

    contador = itertools.count()
    menores_genuinos: List[tuple] = []
    maiores_impostores: List[tuple] = []
    n_gen = n_imp = 0

    # A varredura é feita em numpy, não linha a linha em Python. Converter
    # 5 milhões de linhas em dicionários para depois descartar 99,998 %
    # delas levava mais de dez segundos na thread da interface; aqui cada
    # lote vira dois vetores, os candidatos saem por `argpartition`, e só
    # as poucas linhas que podem entrar na tabela são materializadas.
    for lote in arquivo.iter_batches(batch_size=_LINHAS_POR_LOTE, columns=colunas):
        for mesma, pilha, sinal in (
            (True, menores_genuinos, -1.0),
            (False, maiores_impostores, 1.0),
        ):
            indices, escores, total_classe = _candidatos_do_lote(
                lote, mesma=mesma, quantidade=extremos_por_classe,
            )
            if mesma:
                n_gen += total_classe
            else:
                n_imp += total_classe
            if len(indices) == 0:
                continue
            registros = _materializar(lote, indices)
            for escore, registro in zip(escores, registros):
                _empilhar(
                    pilha, (sinal * float(escore), next(contador), registro),
                    extremos_por_classe,
                )

    genuinos = sorted(
        (_linha_do_registro(t[2]) for t in menores_genuinos), key=lambda r: r.score,
    )
    impostores = sorted(
        (_linha_do_registro(t[2]) for t in maiores_impostores),
        key=lambda r: r.score, reverse=True,
    )
    return _montar(
        genuinos + impostores,
        mode=MODO_EXTREMOS, total_pairs=total,
        truncated=(n_gen > len(genuinos)) or (n_imp > len(impostores)),
        extremos_por_classe=extremos_por_classe,
        n_genuines=n_gen, n_impostors=n_imp,
        has_log_lr=tem_log_lr, has_file_names=False,
    )


# ── Despacho ─────────────────────────────────────────────────────────────

def build_comparison_table(
    *,
    dataset: Any = None,
    parquet_path: Path | str | None = None,
    extremos_por_classe: int = EXTREMOS_POR_CLASSE_PADRAO,
    limite_caso: int = LIMITE_CASO_PADRAO,
) -> Optional[ComparisonTable]:
    """Prefere o payload hidratado, porque só ele traz o nome do arquivo."""
    if dataset is not None and getattr(dataset, "items", None) is not None:
        return build_from_dataset(
            dataset,
            extremos_por_classe=extremos_por_classe,
            limite_caso=limite_caso,
        )
    if parquet_path is not None:
        return build_from_parquet(
            parquet_path,
            extremos_por_classe=extremos_por_classe,
            limite_caso=limite_caso,
        )
    return None


# ── Interno ──────────────────────────────────────────────────────────────

def _candidatos_do_lote(lote: Any, *, mesma: bool, quantidade: int):
    """Índices do lote que podem entrar na tabela, mais a contagem total.

    Devolve `(indices, escores, total_da_classe)`. Só os `quantidade`
    melhores do lote podem sobreviver à pilha global, então não há motivo
    para materializar os outros.
    """
    import numpy as np
    import pyarrow.compute as pc

    mascara = pc.equal(lote.column("same_person"), mesma)
    mascara = pc.fill_null(mascara, False).to_numpy(zero_copy_only=False)
    mascara = np.asarray(mascara, dtype=bool)
    total = int(mascara.sum())
    if total == 0 or quantidade <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0), total

    escores = np.asarray(
        lote.column("raw_score").to_numpy(zero_copy_only=False), dtype=np.float64,
    )
    indices = np.flatnonzero(mascara)
    valores = escores[indices]
    # Genuíno interessa pelo menor escore; impostor, pelo maior.
    ordenaveis = valores if mesma else -valores
    if total > quantidade:
        corte = np.argpartition(ordenaveis, quantidade - 1)[:quantidade]
        indices = indices[corte]
        valores = valores[corte]
    return indices, valores, total


def _materializar(lote: Any, indices) -> List[dict]:
    """As poucas linhas selecionadas, como dicionários."""
    import pyarrow as pa

    return lote.take(pa.array(indices)).to_pylist()


def _empilhar(pilha: List[tuple], entrada: tuple, capacidade: int) -> None:
    """Mantém a pilha com no máximo `capacidade` elementos.

    A chave já vem com o sinal certo: a raiz do min-heap é sempre o pior
    candidato a permanecer, então descartá-la preserva os melhores sem
    guardar o conjunto inteiro.
    """
    if capacidade <= 0:
        return
    if len(pilha) < capacidade:
        heapq.heappush(pilha, entrada)
    else:
        heapq.heappushpop(pilha, entrada)


def _classe_de(mesma: Optional[bool]) -> str:
    if mesma is True:
        return CLASSE_GENUINO
    if mesma is False:
        return CLASSE_IMPOSTOR
    return CLASSE_CASO


def _abreviar_hash(valor: Any) -> str:
    texto = str(valor or "")
    return texto[:_PREFIXO_HASH] if texto else "—"


def _linha_do_item(sample: Any, *, de_caso: bool = False) -> ComparisonRow:
    probe = getattr(sample, "probe", None)
    referencia = getattr(sample, "reference", None)
    metadata = getattr(sample, "metadata", None)
    log_lr = metadata.get("log_lr") if isinstance(metadata, dict) else None
    mesma = None if de_caso else getattr(sample, "same_person", None)
    # O payload hidratado já traz o caminho da imagem em cada face; é o
    # único caminho em que as miniaturas aparecem sem consultar o dataset.
    return ComparisonRow(
        probe_identity=str(getattr(probe, "identity", None) or "—"),
        probe_face=_rotulo_da_face(probe),
        reference_identity=str(getattr(referencia, "identity", None) or "—"),
        reference_face=_rotulo_da_face(referencia),
        score=float(getattr(sample, "raw_score", 0.0) or 0.0),
        same_person=mesma,
        log_lr=float(log_lr) if log_lr is not None else None,
        classe=_classe_de(mesma),
        probe_hash=str(getattr(probe, "face_hash", "") or ""),
        reference_hash=str(getattr(referencia, "face_hash", "") or ""),
        probe_image_path=_caminho_da_face(probe),
        reference_image_path=_caminho_da_face(referencia),
    )


def _caminho_da_face(info: Any) -> Optional[str]:
    """Caminho da imagem quando o payload o carrega."""
    caminho = getattr(info, "image_path", None)
    return str(caminho) if caminho else None


def _rotulo_da_face(info: Any) -> str:
    """Nome do arquivo quando existe; hash abreviado como alternativa."""
    nome = getattr(info, "file_name", None)
    if nome:
        return str(nome)
    caminho = getattr(info, "image_path", None)
    if caminho:
        return Path(str(caminho)).name
    return _abreviar_hash(getattr(info, "face_hash", None))


def _linha_do_registro(registro: dict, *, de_caso: bool = False) -> ComparisonRow:
    mesma = None if de_caso else registro.get("same_person")
    log_lr = registro.get("log_lr")
    # NaN no parquet marca "este par não foi calibrado"; exibir "nan" na
    # coluna seria pior que deixá-la vazia.
    if log_lr is not None and log_lr != log_lr:
        log_lr = None
    return ComparisonRow(
        probe_identity=str(registro.get("probe_identity") or "—"),
        probe_face=_abreviar_hash(registro.get("probe_face_hash")),
        reference_identity=str(registro.get("reference_identity") or "—"),
        reference_face=_abreviar_hash(registro.get("reference_face_hash")),
        score=float(registro.get("raw_score") or 0.0),
        same_person=mesma,
        log_lr=float(log_lr) if log_lr is not None else None,
        classe=_classe_de(mesma),
        probe_hash=str(registro.get("probe_face_hash") or ""),
        reference_hash=str(registro.get("reference_face_hash") or ""),
    )


def _parquet_e_de_caso(arquivo: Any) -> bool:
    """Lê a marca `faceaudit:is_casework` do schema.

    Não cai para a inferência `todos os same_person são nulos`: parquets
    antigos sem a marca ficam no regime de extremos, que é o seguro — um
    dataset de calibração tratado como caso tentaria listar tudo.
    """
    try:
        metadata = arquivo.schema_arrow.metadata or {}
        return metadata.get(b"faceaudit:is_casework") == b"true"
    except Exception:
        return False


def attach_face_images(
    table: ComparisonTable,
    dataset_parquets: Sequence[Path | str],
) -> ComparisonTable:
    """Preenche a imagem e o nome de arquivo de cada face da tabela.

    Lido do parquet, o escore só tem identidade e `face_hash`; a imagem
    pertence ao dataset. Esta função faz a ponte, e de passagem troca o
    hash abreviado pelo nome do arquivo real, que é o que serve ao perito.

    Face não encontrada fica sem imagem, não sem linha: dataset movido ou
    não informado é caso normal, e esconder a comparação por causa da
    miniatura seria esconder o resultado por causa da moldura.
    """
    if table.is_empty:
        return table

    from faceaudit.services.face_image_locator_service import locate_faces

    pendentes = {
        h
        for linha in table.rows
        for h, caminho in (
            (linha.probe_hash, linha.probe_image_path),
            (linha.reference_hash, linha.reference_image_path),
        )
        if h and not caminho
    }
    if not pendentes:
        return table

    achados = locate_faces(dataset_parquets, pendentes)
    if not achados:
        return table

    novas = []
    for linha in table.rows:
        mudancas = {}
        for lado in ("probe", "reference"):
            if getattr(linha, f"{lado}_image_path"):
                continue
            registro = achados.get(getattr(linha, f"{lado}_hash"))
            if registro is None:
                continue
            mudancas[f"{lado}_image_path"] = str(registro.image_path)
            mudancas[f"{lado}_face"] = registro.file_name
        novas.append(replace(linha, **mudancas) if mudancas else linha)

    novas_tuple = tuple(novas)
    tem_imagem = any(
        r.probe_image_path or r.reference_image_path for r in novas_tuple
    )
    tem_nomes = table.has_file_names or bool(achados)
    nota = table.note
    if tem_nomes and not table.has_file_names:
        # A ressalva sobre o hash deixa de valer quando o nome do arquivo
        # foi recuperado do dataset.
        nota = nota.replace(_RESSALVA_SEM_NOME, "")
    return replace(
        table, rows=novas_tuple, has_images=tem_imagem,
        has_file_names=tem_nomes, note=nota,
    )


def _tabela_vazia(
    is_casework: bool,
    *,
    has_log_lr: bool = False,
    has_file_names: bool = True,
) -> ComparisonTable:
    return ComparisonTable(
        rows=(),
        mode=MODO_CASO if is_casework else MODO_EXTREMOS,
        total_pairs=0, n_genuines=0, n_impostors=0,
        truncated=False, has_log_lr=has_log_lr, has_file_names=has_file_names,
        note="Nenhuma comparação neste conjunto.",
    )


def _montar(
    linhas: Sequence[ComparisonRow],
    *,
    mode: str,
    total_pairs: int,
    truncated: bool,
    extremos_por_classe: int,
    n_genuines: Optional[int] = None,
    n_impostors: Optional[int] = None,
    has_log_lr: Optional[bool] = None,
    has_file_names: bool = True,
) -> ComparisonTable:
    linhas = tuple(linhas)
    if n_genuines is None:
        n_genuines = sum(1 for r in linhas if r.classe == CLASSE_GENUINO)
    if n_impostors is None:
        n_impostors = sum(1 for r in linhas if r.classe == CLASSE_IMPOSTOR)
    if has_log_lr is None:
        has_log_lr = any(r.log_lr is not None for r in linhas)
    return ComparisonTable(
        rows=linhas, mode=mode, total_pairs=total_pairs,
        n_genuines=n_genuines, n_impostors=n_impostors,
        truncated=truncated, has_log_lr=has_log_lr,
        has_file_names=has_file_names,
        has_images=any(
            r.probe_image_path or r.reference_image_path for r in linhas
        ),
        note=_nota(
            linhas, mode=mode, total_pairs=total_pairs, truncated=truncated,
            extremos_por_classe=extremos_por_classe,
            has_file_names=has_file_names,
        ),
    )


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural


def _milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _nota(
    linhas: Sequence[ComparisonRow],
    *,
    mode: str,
    total_pairs: int,
    truncated: bool,
    extremos_por_classe: int,
    has_file_names: bool,
) -> str:
    """Frase que explica por que estas linhas e não outras.

    Sem ela, o modo de extremos passa por tabela completa e o perito lê
    uma amostra enviesada como se fosse o conjunto.
    """
    if mode == MODO_CASO:
        if truncated:
            texto = (
                f"Conjunto de caso com {_milhar(total_pairs)} comparações, "
                f"das quais a tabela mostra as {_milhar(len(linhas))} de "
                f"maior escore."
            )
        else:
            texto = (
                f"Conjunto de caso: {_milhar(total_pairs)} "
                f"{_plural(total_pairs, 'comparação', 'comparações')}, "
                f"em ordem decrescente de escore."
            )
    else:
        gen = sum(1 for r in linhas if r.classe == CLASSE_GENUINO)
        imp = sum(1 for r in linhas if r.classe == CLASSE_IMPOSTOR)
        texto = (
            f"Conjunto de referência com {_milhar(total_pairs)} comparações. "
            f"A tabela mostra os extremos, que é onde o sistema erra: "
            f"{_milhar(gen)} {_plural(gen, 'genuíno', 'genuínos')} de menor "
            f"escore e {_milhar(imp)} {_plural(imp, 'impostor', 'impostores')} "
            f"de maior escore, até {_milhar(extremos_por_classe)} de cada."
        )
    if not has_file_names:
        texto += _RESSALVA_SEM_NOME
    return texto
