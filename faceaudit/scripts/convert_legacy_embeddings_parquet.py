"""Converte parquet legado de embeddings (formato WebFace4M) pro formato
canônico FaceAudit.

Diferente de `migrate_legacy_parquet`:
- **Não sobrescreve** — recebe SRC e DST distintos.
- **Streaming** via `ParquetWriter` (vs. carregar tudo via FaceDataset),
  necessário pra arquivos grandes (>5 GB) que não cabem em memória.
- **Especializado**: parquets WebFace4M trazem 3 colunas de embedding
  redundantes em float64. Mantém só `embedding_v2` (sepaelv2), faz cast
  pra float32, e dropa `embedding` (alias) e `embedding_v4` (sepaelv4).
- **Dropa colunas vazias**: as colunas null-typed (ipd, fiqa_score,
  yaw, pitch, roll, det_score, gender, ethnicity, etc.) são removidas.
- **Multiple row groups**: source vem com row group único (12 GB);
  output sai com 50k linhas por row group, permitindo leitura paginada.

Uso:
    python -m faceaudit.scripts.convert_legacy_embeddings_parquet SRC DST
    python -m faceaudit.scripts.convert_legacy_embeddings_parquet SRC DST \\
        --name "WebFace4M_1M" --description "Subset 1M WebFace4M"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pyarrow as pa
import pyarrow.parquet as pq

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.column_catalog import (
    DatasetType, get_default_specs, specs_to_json,
)


# Colunas que sobrevivem na saída — todas as outras são dropadas.
# `embedding_v2` é o sepaelv2 (AdaFace canônico do projeto).
KEEP_COLS = [
    "embedding_v2", "image_path", "identity", "origin",
    "dataset_name", "dataset_description",
]
ROW_GROUP_SIZE = 50_000
BATCH_SIZE = 10_000


def main(argv=None) -> int:
    # Console Windows usa cp1252 por padrão, que não tem mapeamento pra
    # vários símbolos usados no log (→, em-dash, etc.). Reconfigura
    # stdout/stderr pra UTF-8 com `replace` antes de qualquer print —
    # sem isso, o primeiro `→` causa UnicodeEncodeError + abort.
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("src", help="Parquet de origem (legacy)")
    parser.add_argument("dst", help="Parquet de destino (novo formato)")
    parser.add_argument("--name", help="Nome do dataset (default: stem do dst)")
    parser.add_argument("--description", default="", help="Descrição do dataset")
    parser.add_argument(
        "--force", action="store_true",
        help="Sobrescrever DST se já existir",
    )
    args = parser.parse_args(argv)

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    if not src.exists():
        print(f"✗ SRC não encontrado: {src}", file=sys.stderr)
        return 1
    if dst.exists() and not args.force:
        print(
            f"✗ DST já existe (use --force pra sobrescrever): {dst}",
            file=sys.stderr,
        )
        return 1

    name = args.name or dst.stem
    description = args.description

    pf = pq.ParquetFile(src)
    src_md = pf.metadata
    print(f"SRC: {src}")
    print(f"DST: {dst}")
    print(
        f"Linhas: {src_md.num_rows:,}  |  Row groups: {src_md.num_row_groups}  |  "
        f"Colunas no SRC: {src_md.num_columns}"
    )
    src_cols = {f.name for f in pf.schema_arrow}
    missing = [c for c in KEEP_COLS if c not in src_cols]
    if missing:
        print(
            f"✗ Colunas obrigatórias ausentes no SRC: {missing}",
            file=sys.stderr,
        )
        return 1
    print(f"Mantendo: {KEEP_COLS}")
    print(f"Dropando: {sorted(src_cols - set(KEEP_COLS))}")
    print()

    # ── Pass 1: identidade → samples_per_identity ─────────────────
    # Necessário pra metadata `samples_per_identity`. Lê só a coluna
    # `identity` — barata.
    print("Pass 1/2: contando samples por identidade…")
    t0 = time.time()
    samples_per_id: Dict[str, int] = {}
    n_imgs_seen = 0
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["identity"]):
        for idn in batch.column("identity").to_pylist():
            if idn is None:
                continue
            key = str(idn)
            samples_per_id[key] = samples_per_id.get(key, 0) + 1
        n_imgs_seen += batch.num_rows
    n_imgs = sum(samples_per_id.values())
    n_ids = len(samples_per_id)
    print(
        f"  {n_imgs_seen:,} faces lidas, {n_ids:,} identidades únicas — "
        f"{time.time() - t0:.1f}s"
    )
    print()

    # ── Schema de saída + metadata FaceAudit ─────────────────────
    out_schema = pa.schema([
        pa.field("embedding_v2", pa.list_(pa.float32())),
        pa.field("image_path", pa.string()),
        pa.field("identity", pa.string()),
        pa.field("origin", pa.string()),
        pa.field("dataset_name", pa.string()),
        pa.field("dataset_description", pa.string()),
    ])

    # column_specs: defaults filtrados pelas colunas que vamos persistir.
    defaults = get_default_specs(DatasetType.FACE)
    keep_set = set(KEEP_COLS)
    specs_to_persist = {n: s for n, s in defaults.items() if n in keep_set}

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta_payload = {
        "faceaudit:dataset_type": DatasetType.FACE.value,
        "faceaudit:dataset_name": name,
        "faceaudit:dataset_description": description,
        "faceaudit:path_mode": PathMode.ABSOLUTE.value,
        "faceaudit:column_specs": specs_to_json(specs_to_persist),
        # column_coverage vazio — não computado em modo streaming.
        # `QualityMetricAvailabilityService` cai no probing colunar
        # quando não há essa metadata, então isso não quebra nada.
        "faceaudit:column_coverage": "{}",
        "faceaudit:samples_per_identity": json.dumps(
            samples_per_id, ensure_ascii=False, separators=(",", ":")
        ),
        "faceaudit:n_imgs": str(n_imgs),
        "faceaudit:n_ids": str(n_ids),
        "faceaudit:last_modified": now_iso,
    }
    schema_with_meta = out_schema.with_metadata({
        k.encode("utf-8"): v.encode("utf-8")
        for k, v in meta_payload.items()
    })

    # ── Pass 2: stream + cast + write ────────────────────────────
    print("Pass 2/2: convertendo embeddings (float64 → float32) e gravando…")
    t0 = time.time()
    n_written = 0
    writer = pq.ParquetWriter(
        dst, schema_with_meta, compression="snappy",
    )
    try:
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=KEEP_COLS):
            # Cast list<float64> → list<float32>. Compute padrão usa
            # alocação nova → o batch original pode ser descartado.
            emb_f32 = batch.column("embedding_v2").cast(
                pa.list_(pa.float32())
            )

            new_batch = pa.RecordBatch.from_arrays(
                [
                    emb_f32,
                    batch.column("image_path"),
                    batch.column("identity"),
                    batch.column("origin"),
                    batch.column("dataset_name"),
                    batch.column("dataset_description"),
                ],
                schema=out_schema,
            )
            tbl = pa.Table.from_batches([new_batch], schema=out_schema)
            writer.write_table(tbl, row_group_size=ROW_GROUP_SIZE)

            n_written += new_batch.num_rows
            # Progresso a cada 100k linhas
            if n_written % 100_000 < BATCH_SIZE or n_written == src_md.num_rows:
                el = time.time() - t0
                pct = 100 * n_written / src_md.num_rows
                rate = n_written / max(el, 0.001)
                print(
                    f"  {n_written:,}/{src_md.num_rows:,} "
                    f"({pct:.1f}%) — {el:.1f}s — {rate:,.0f} faces/s"
                )
    finally:
        writer.close()

    size_in = src.stat().st_size / (1024 * 1024)
    size_out = dst.stat().st_size / (1024 * 1024)
    ratio = size_out / size_in if size_in else 0.0
    print()
    print(f"✅ Conversão concluída em {time.time() - t0:.1f}s")
    print(f"   {size_in:,.1f} MB  →  {size_out:,.1f} MB  ({ratio*100:.1f}% do original)")
    print(f"   {n_written:,} faces, {n_ids:,} identidades, {ROW_GROUP_SIZE:,} linhas/row group")
    return 0


if __name__ == "__main__":
    sys.exit(main())
