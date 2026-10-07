"""Onde está a imagem de uma face, a partir do seu `face_hash`.

O parquet de escores guarda identidade e `face_hash` de cada lado do par, e
**não** guarda caminho de imagem — o que é correto, porque a imagem
pertence ao dataset e não ao escore. Mas uma tabela de comparações sem as
duas faces lado a lado obriga o perito a conferir nome de arquivo contra a
pasta, o que é justamente o trabalho que a tabela existe para poupar.

A ponte é o `face_hash`: ele é o mesmo no parquet de escores
(`probe_face_hash` / `reference_face_hash`) e no parquet do dataset, onde a
coluna se chama `hash` e vem acompanhada de `image_path`, `file_name` e
`identity`.

A varredura lê só as quatro colunas, por lote, e descarta a linha que não
interessa antes de materializá-la: um dataset de referência tem dezenas de
milhares de faces e a tabela mostra cem. Para quando encontra tudo o que
foi pedido.

A resolução do caminho passa por `resolve_image_path`, o mesmo ponto que o
resto do sistema usa, porque o dataset pode ser relativo e nesse caso o
caminho gravado só faz sentido ancorado na pasta do parquet mais o
`path_root` da metadata. Reimplementar isso aqui daria caminho quebrado em
dataset portátil.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.project_folder import ProjectFolder

# Colunas procuradas no parquet do dataset. `hash` é o nome da coluna do
# face_hash lá; no parquet de escores o mesmo valor aparece como
# `probe_face_hash` / `reference_face_hash`.
_COLUNA_HASH = "hash"
_COLUNAS_DESEJADAS = (_COLUNA_HASH, "image_path", "file_name", "identity")

_LINHAS_POR_LOTE = 65_536


@dataclass(frozen=True)
class FaceImageRecord:
    """O que o dataset sabe sobre uma face, além do embedding."""

    image_path: Path
    file_name: str
    identity: Optional[str]


def locate_faces(
    dataset_parquets: Sequence[Path | str],
    hashes: Iterable[str],
) -> Dict[str, FaceImageRecord]:
    """Mapeia `face_hash` → imagem, procurando nos parquets informados.

    Hash não encontrado simplesmente não aparece no resultado: dataset
    movido, apagado ou não informado é caso normal, e quem exibe tem de
    poder mostrar a linha sem a miniatura em vez de falhar.
    """
    procurados = {h for h in hashes if h}
    if not procurados or not dataset_parquets:
        return {}

    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {}

    encontrados: Dict[str, FaceImageRecord] = {}
    for bruto in dataset_parquets:
        if not procurados:
            break
        caminho = Path(bruto)
        if not caminho.is_file():
            continue
        try:
            arquivo = pq.ParquetFile(caminho)
        except Exception:
            continue

        nomes = set(arquivo.schema_arrow.names)
        if _COLUNA_HASH not in nomes or "image_path" not in nomes:
            continue
        colunas = [c for c in _COLUNAS_DESEJADAS if c in nomes]
        modo, raiz = _ancoragem(arquivo)

        try:
            for lote in arquivo.iter_batches(
                batch_size=_LINHAS_POR_LOTE, columns=colunas,
            ):
                achados_no_lote = _filtrar_lote(lote, procurados)
                for registro in achados_no_lote:
                    h = registro[_COLUNA_HASH]
                    encontrados[h] = _registro(
                        registro, parquet_dir=caminho.parent,
                        path_mode=modo, path_root=raiz,
                    )
                    procurados.discard(h)
                if not procurados:
                    break
        except Exception:
            continue

    return encontrados


def find_dataset_parquet(
    project_root: Path | str,
    resource_id: str,
) -> Optional[Path]:
    """Parquet do dataset de um recurso de projeto, pelo nome do arquivo.

    O `resource_id` de um recurso de projeto é o nome do arquivo sem
    extensão, então a busca é por nome. É recursiva porque a pasta de
    datasets tem subpastas (`primary/`, saídas de instrução por
    estratégia) e a convenção de onde cada um cai mudou ao longo do tempo.
    """
    if not resource_id:
        return None
    base = Path(project_root) / ProjectFolder.DATASETS.value
    if not base.is_dir():
        return None
    alvo = f"{resource_id}.parquet"
    direto = base / "primary" / alvo
    if direto.is_file():
        return direto
    try:
        for candidato in base.rglob(alvo):
            if candidato.is_file():
                return candidato
    except OSError:
        return None
    return None


# ── Interno ──────────────────────────────────────────────────────────────

def _ancoragem(arquivo) -> tuple:
    """`(path_mode, path_root)` declarados na metadata do parquet.

    Parquet sem a marca é tratado como absoluto, que é o comportamento
    herdado dos parquets anteriores ao campo.
    """
    try:
        meta = arquivo.schema_arrow.metadata or {}
        bruto = meta.get(b"faceaudit:path_mode")
        modo = PathMode(bruto.decode()) if bruto else PathMode.ABSOLUTE
    except Exception:
        modo = PathMode.ABSOLUTE
    try:
        raiz_bruta = (arquivo.schema_arrow.metadata or {}).get(
            b"faceaudit:path_root"
        )
        raiz = raiz_bruta.decode() if raiz_bruta else None
    except Exception:
        raiz = None
    if raiz:
        from faceaudit.services.image_path_resolver import normalize_stored_path
        raiz = normalize_stored_path(raiz)
    return modo, raiz


def _filtrar_lote(lote, procurados: set) -> List[dict]:
    """As linhas do lote cujos hashes estão sendo procurados.

    O filtro é feito sobre a coluna de hash antes de converter qualquer
    linha em dicionário — é a diferença entre materializar as poucas
    linhas que interessam e materializar o dataset inteiro.
    """
    import numpy as np
    import pyarrow as pa

    coluna = lote.column(_COLUNA_HASH).to_pylist()
    indices = [i for i, h in enumerate(coluna) if h in procurados]
    if not indices:
        return []
    return lote.take(pa.array(np.asarray(indices, dtype=np.int64))).to_pylist()


def _registro(
    linha: dict,
    *,
    parquet_dir: Path,
    path_mode: PathMode,
    path_root: Optional[str],
) -> FaceImageRecord:
    from faceaudit.services.image_path_resolver import resolve_image_path

    caminho = resolve_image_path(
        linha.get("image_path") or "",
        parquet_dir=parquet_dir,
        path_mode=path_mode,
        path_root=path_root,
    )
    nome = linha.get("file_name") or caminho.name
    return FaceImageRecord(
        image_path=caminho,
        file_name=str(nome),
        identity=linha.get("identity"),
    )
