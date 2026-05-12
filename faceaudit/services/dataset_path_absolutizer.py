"""Absolutiza paths de imagem ao copiar um parquet pra outra pasta.

Motivação: quando um dataset com `path_mode=RELATIVE` é copiado da
pasta original (onde `(parquet_dir / path_root)` aponta corretamente
pras imagens) pra dentro de um projeto FaceAudit
(`<projeto>/datasets/<id>.parquet`), o `parquet_dir` muda mas o
`path_root` continua o mesmo — então
`image_path_resolver.resolve_image_path` passa a apontar pra
`<projeto>/datasets/<path_root>/imagem.jpg` (lugar errado: as imagens
estão na pasta original).

Solução simples (opção 1 do plano de import): antes de copiar,
absolutiza os paths — reescreve `image_path` ancorando em
`(source_parquet_dir / path_root)` e regrava o parquet com
`path_mode=ABSOLUTE` (sem `path_root`). Datasets já em modo ABSOLUTE
passam batidos (cópia byte-a-byte via `shutil.copy2`).

Trade-off conhecido: o projeto deixa de ser autocontido até o usuário
rodar "Selar projeto" (que copia as imagens pra dentro do `.fa` e
reescreve paths como relativos). Comprado em troca de import barato.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Union

import pyarrow as pa
import pyarrow.parquet as pq

from faceaudit.enums.path_mode import PathMode
from faceaudit.services.image_path_resolver import normalize_stored_path


_PATH_MODE_KEY = "faceaudit:path_mode"
_PATH_ROOT_KEY = "faceaudit:path_root"
_IMAGE_PATH_COLUMN = "image_path"


def absolutize_parquet_paths(
    source: Union[str, Path],
    destination: Union[str, Path],
) -> bool:
    """Copia `source` pra `destination`, absolutizando paths se RELATIVE.

    Args:
        source: caminho absoluto do parquet original.
        destination: onde gravar a versão absolutizada (ou cópia direta).
            Diretórios intermediários são criados sob demanda.

    Returns:
        True se houve reescrita de paths (parquet original era RELATIVE).
        False se foi cópia byte-a-byte (parquet já ABSOLUTE ou sem
        metadata `path_mode`).

    Raises:
        FileNotFoundError: se `source` não existir.
    """
    src = Path(source)
    dst = Path(destination)
    if not src.exists():
        raise FileNotFoundError(f"Parquet de origem não encontrado: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    meta = _read_schema_metadata(src)
    path_mode = _parse_path_mode(meta.get(_PATH_MODE_KEY))

    if path_mode == PathMode.ABSOLUTE:
        # Nada a reescrever — cópia direta preservando mtime/permissões.
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return False

    # path_mode == RELATIVE → carrega, reescreve `image_path`, regrava metadata.
    parquet_dir = src.parent.resolve()
    path_root_raw = meta.get(_PATH_ROOT_KEY) or "."
    anchor = (parquet_dir / normalize_stored_path(path_root_raw)).resolve()

    table = pq.read_table(src)
    if _IMAGE_PATH_COLUMN in table.column_names:
        table = _rewrite_image_path_column(table, anchor)

    # Atualiza metadata: declara ABSOLUTE e remove `path_root` (irrelevante
    # em ABSOLUTE e ruído pra leitores futuros).
    new_meta = {k: v for k, v in meta.items() if k != _PATH_ROOT_KEY}
    new_meta[_PATH_MODE_KEY] = PathMode.ABSOLUTE.value
    encoded = {k.encode("utf-8"): v.encode("utf-8") for k, v in new_meta.items()}
    table = table.replace_schema_metadata(encoded)

    pq.write_table(table, dst)
    return True


# ── Internals ────────────────────────────────────────────────────────────────


def _read_schema_metadata(path: Path) -> Dict[str, str]:
    """Lê a schema metadata do parquet decodificada como dict de strings.

    Retorna dict vazio se o parquet não tem metadata.
    """
    pq_file = pq.ParquetFile(path)
    raw = pq_file.schema_arrow.metadata
    if not raw:
        return {}
    return {
        k.decode("utf-8"): v.decode("utf-8")
        for k, v in raw.items()
    }


def _parse_path_mode(raw: object) -> PathMode:
    """Converte o valor de metadata em PathMode, com fallback ABSOLUTE.

    Compatível com parquets legados (sem `path_mode` registrado): assume
    ABSOLUTE — mesmo default do `FaceDataset._from_new_format`.
    """
    if not raw:
        return PathMode.ABSOLUTE
    try:
        return PathMode(str(raw))
    except ValueError:
        return PathMode.ABSOLUTE


def _rewrite_image_path_column(table: pa.Table, anchor: Path) -> pa.Table:
    """Substitui a coluna `image_path` por versão absolutizada.

    Mantém `None` e strings vazias intactas (sem cabeças relativas a
    absolutizar). Strings já absolutas são preservadas (defensivo:
    parquet declarou RELATIVE mas algum face vazou um path absoluto).
    """
    original = table.column(_IMAGE_PATH_COLUMN).to_pylist()
    absolutized = []
    for raw in original:
        if raw is None or raw == "":
            absolutized.append(raw)
            continue
        normalized = normalize_stored_path(str(raw))
        p = Path(normalized)
        if p.is_absolute():
            absolutized.append(normalized)
        else:
            absolutized.append((anchor / p).resolve().as_posix())

    idx = table.column_names.index(_IMAGE_PATH_COLUMN)
    new_array = pa.array(absolutized, type=table.schema.field(idx).type)
    return table.set_column(idx, table.schema.field(idx), new_array)
