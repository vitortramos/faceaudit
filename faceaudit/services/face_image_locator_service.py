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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.models.aggregation_provenance import (
    COLUNA_COUNT, COLUNA_HASHES, COLUNA_PATHS, AggregationProvenance,
)
from faceaudit.models import aggregation_provenance

# Colunas procuradas no parquet do dataset. `hash` é o nome da coluna do
# face_hash lá; no parquet de escores o mesmo valor aparece como
# `probe_face_hash` / `reference_face_hash`.
# `FaceDataset` grava o hash da face em `hash`; `EmbeddingDataset`, que é o
# que sai da agregação e de `PIPELINE_APPLICATION`, grava em `face_hash`.
# Aceitar os dois é o que permite achar o nome de um template agregado, que
# só existe no segundo.
_COLUNAS_DE_HASH = ("hash", "face_hash")
_OUTRAS_COLUNAS = (
    "image_path", "file_name", "identity", "bbox", "keypoints",
    COLUNA_COUNT, COLUNA_HASHES, COLUNA_PATHS,
)

_LINHAS_POR_LOTE = 65_536


@dataclass(frozen=True)
class FaceImageRecord:
    """O que o dataset sabe sobre uma face, além do embedding.

    A geometria vem no sistema de coordenadas da imagem apontada por
    `image_path`, que é o invariante do dataset — por isso dá para
    desenhá-la direto sobre a imagem, sem conversão.
    """

    image_path: Path
    file_name: str
    identity: Optional[str]
    bbox: Optional[Tuple[float, ...]] = None
    keypoints: Optional[Tuple[Tuple[float, float], ...]] = None
    # Preenchido só em template agregado: de quais faces ele foi feito.
    provenance: Optional[AggregationProvenance] = None


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
        coluna_hash = next((c for c in _COLUNAS_DE_HASH if c in nomes), None)
        if coluna_hash is None:
            continue
        colunas = [coluna_hash] + [c for c in _OUTRAS_COLUNAS if c in nomes]
        modo, raiz = _ancoragem(arquivo)

        try:
            for lote in arquivo.iter_batches(
                batch_size=_LINHAS_POR_LOTE, columns=colunas,
            ):
                achados_no_lote = _filtrar_lote(lote, procurados, coluna_hash)
                for registro in achados_no_lote:
                    h = registro[coluna_hash]
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


def _filtrar_lote(lote, procurados: set, coluna_hash: str = "hash") -> List[dict]:
    """As linhas do lote cujos hashes estão sendo procurados.

    O filtro é feito sobre a coluna de hash antes de converter qualquer
    linha em dicionário — é a diferença entre materializar as poucas
    linhas que interessam e materializar o dataset inteiro.
    """
    import numpy as np
    import pyarrow as pa

    coluna = lote.column(coluna_hash).to_pylist()
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

    bruto = linha.get("image_path") or ""
    # Template agregado não tem imagem: `image_path` sai vazio de
    # propósito, porque a média de N faces não corresponde a foto nenhuma.
    # Resolver o vazio devolveria a pasta do parquet, que existe, e a linha
    # passaria por face com imagem.
    caminho = (
        resolve_image_path(
            bruto, parquet_dir=parquet_dir,
            path_mode=path_mode, path_root=path_root,
        )
        if bruto else Path("")
    )
    nome = linha.get("file_name") or (caminho.name if bruto else "")
    return FaceImageRecord(
        image_path=caminho,
        file_name=str(nome),
        identity=linha.get("identity"),
        bbox=_caixa(linha.get("bbox")),
        keypoints=_pontos(linha.get("keypoints")),
        provenance=aggregation_provenance.from_row(linha),
    )


def _caixa(bruto) -> Optional[Tuple[float, ...]]:
    """`(x1, y1, x2, y2)`, ou nada quando o tamanho não bate.

    Meia caixa desenharia um retângulo errado sobre a face, o que é pior
    que não desenhar nada.
    """
    if bruto is None:
        return None
    try:
        valores = [float(v) for v in bruto]
    except (TypeError, ValueError):
        return None
    return tuple(valores[:4]) if len(valores) >= 4 else None


def _pontos(bruto) -> Optional[Tuple[Tuple[float, float], ...]]:
    """Os cinco pares, a partir da lista achatada gravada no parquet."""
    if bruto is None:
        return None
    try:
        valores = [float(v) for v in bruto]
    except (TypeError, ValueError):
        return None
    if len(valores) != 10:
        return None
    return tuple(
        (valores[i], valores[i + 1]) for i in range(0, 10, 2)
    )
