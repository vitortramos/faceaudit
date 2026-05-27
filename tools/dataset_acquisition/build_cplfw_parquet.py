#!/usr/bin/env python3
"""Constrói um EmbeddingDataset parquet do CPLFW pronto pro FaceAudit.

Combina:
    - imagens do CPLFW (`images/` raw 250×250 ou `aligned images/` 224×224)
    - rótulos de demographics.csv (output do `join_cplfw_lfw_attributes.py`)
    - embeddings + pose + FIQA do FFService (forensicface + sepaelv2 default)

Default: usa `images/` com `FFService.process_face` (detect + align via
ArcFace 5-point). Mais lento (~50-100ms por imagem com CoreML), mas
garantidamente correto. A pasta `aligned images/` do CPLFW vem em
224×224 com convenção de alinhamento PRÓPRIA do grupo (Beihang); não
bate exatamente com ArcFace, então embeddings extraídos dela podem
ficar levemente off vs `process_face` no raw.

Use `--use-cplfw-aligned` SÓ se você está disposto a aceitar essa
diferença em troca de velocidade (pula detector det_10g).

Output parquet (compatível com `EmbeddingDataset.from_parquet`):

    Colunas canônicas:
        embedding_v2 (lista 512 float)   ← sepaelv2 default
        identity, gender, ethnicity      ← do CSV demographics
        image_path, file_name, hash
        yaw, pitch, roll                 ← do detector ArcFace
        fiqa_score, gender_est, age_est
        bbox, keypoints                  (do detector)

    Colunas extras (atributos Columbia):
        gender_score, race_label, race_score, race_scores_json,
        age_band, age_scores_json, match_kind, is_matched

    73 atributos brutos (snake_case): male, asian, white, black, baby,
        child, youth, ... (mesma lista do demographics.csv)

`hash` é SHA-256 do arquivo de imagem (não do alinhado). É a chave
canônica `face_hash` usada pelo SCORE_VECTORIZATION e pelas regras
de pareamento simétrico.

Uso típico:
    python tools/dataset_acquisition/build_cplfw_parquet.py \\
        --demographics ~/Documents/bases/cplfw/metadata/demographics.csv \\
        --out ~/Documents/bases/cplfw.parquet

Teste rápido (100 imagens):
    python tools/dataset_acquisition/build_cplfw_parquet.py \\
        --demographics ~/Documents/bases/cplfw/metadata/demographics.csv \\
        --out /tmp/cplfw_sample.parquet \\
        --limit 100

Resume após interrupção (continua de onde parou):
    python tools/dataset_acquisition/build_cplfw_parquet.py \\
        --demographics ~/Documents/bases/cplfw/metadata/demographics.csv \\
        --out ~/Documents/bases/cplfw.parquet \\
        --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Lazy import — só carrega o FFService (com ONNX runtime) quando vamos
# de fato extrair embeddings. Permite que `--help` funcione sem ter
# forensicface instalado. O FFService internamente aplica o
# workaround do CoreML (`RequireStaticInputShapes=1`) quando
# `use_gpu=True`, então este script não precisa mais fazer nada.
def _build_ff_service(model: str, use_gpu: bool):
    from faceaudit.services.ff_service import FFService

    return FFService(models=model, use_gpu=use_gpu, extended=True)


def _sha256_of_file(path: Path, chunk: int = 1 << 16) -> str:
    """SHA-256 hex digest do arquivo. Usa `file_digest` (Python 3.11+)
    quando disponível pra evitar buffer Python overhead em arquivos
    grandes; cai pro loop manual em Python mais antigo."""
    if hasattr(hashlib, "file_digest"):
        with path.open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _load_demographics(csv_path: Path) -> List[dict]:
    """Carrega o CSV demographics.csv como lista de dicts."""
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    print(f"[build] {len(rows)} entradas em {csv_path.name}")
    return rows


def _load_resume_hashes(out_path: Path) -> Set[str]:
    """Lê hashes já presentes no parquet de destino (modo --resume).
    Retorna conjunto vazio se o arquivo não existir."""
    if not out_path.exists():
        return set()
    try:
        t = pq.read_table(out_path, columns=["hash"])
        return set(t["hash"].to_pylist())
    except Exception as exc:
        print(f"[build] Aviso: falha ao ler '{out_path}' pra resume "
              f"({exc}); processarei tudo do zero.", file=sys.stderr)
        return set()


def _build_face_info(
    img_path: Path,
    row: dict,
    file_hash: str,
):
    """Constrói FaceInfo populado com rótulos do CSV demographics.

    Os 73 atributos brutos do Columbia vão em `extra_data` e serão
    flatten direto pelo `to_flat_dict()` — fica como coluna por sample
    no parquet.
    """
    from faceaudit.models.face_info import FaceInfo

    # Race do CSV vai pra `ethnicity` canônico
    ethnicity = row.get("race_label", "").strip() or None
    # Gender do CSV ("male"/"female"/"unknown") vai pra `gender` canônico
    gender = row.get("gender", "").strip() or None
    if gender in ("", "unknown"):
        gender = None
    if ethnicity in ("", "unknown"):
        ethnicity = None

    # Atributos brutos + metadados não-canônicos vão em extra_data.
    # Pulamos colunas que JÁ são canônicas pra não duplicar.
    canonical_keys = {
        "image_path", "identity", "image_num",
        "gender", "race_label",  # já viram canônicos acima
    }
    extra: Dict[str, object] = {}
    for k, v in row.items():
        if k in canonical_keys:
            continue
        if v == "" or v is None:
            continue
        # tenta converter pra float quando aplicável (scores numéricos)
        try:
            extra[k] = float(v)
        except (TypeError, ValueError):
            extra[k] = v

    fi = FaceInfo(
        image_path=str(img_path),
        identity=row.get("identity") or None,
        gender=gender,
        ethnicity=ethnicity,
        hash=file_hash,
        origin="CPLFW",
        extra_data=extra,
    )
    return fi


def _process_one(
    ff_service,
    img_path: Path,
    row: dict,
    use_aligned: bool,
) -> Optional[dict]:
    """Pipeline completo de UMA imagem: hash → FaceInfo → FFService →
    flat dict pronto pro parquet."""
    if not img_path.exists():
        print(f"[build] Aviso: imagem ausente {img_path}", file=sys.stderr)
        return None

    file_hash = _sha256_of_file(img_path)
    face_info = _build_face_info(img_path, row, file_hash)

    if use_aligned:
        face_info = ff_service.process_face_aligned(face_info)
    else:
        face_info = ff_service.process_face(face_info)

    # Falha silenciosa: process_face devolve face_info inalterado quando
    # detector não acha face. Detecta esse caso e marca erro.
    if face_info.ff_info is None:
        face_info.error = True
        face_info.error_stage = "ff"
        face_info.error_message = "nenhuma face detectada"

    return face_info.to_flat_dict()


def _flush_rows_to_parquet(rows: List[dict], out_path: Path, append: bool) -> None:
    """Escreve `rows` num parquet — append se já existe, senão cria.

    Estratégia simples: lê o existente, concatena, regrava. Pra
    11k imagens isso é rápido. Não otimiza pra ingestão contínua de
    milhões de linhas.
    """
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if append and out_path.exists():
        df_old = pd.read_parquet(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demographics", required=True,
                    help="CSV gerado por join_cplfw_lfw_attributes.py.")
    ap.add_argument("--out", required=True,
                    help="Parquet de saída.")
    ap.add_argument("--model", default="sepaelv2",
                    help="Modelo Sepael a carregar (default: sepaelv2).")
    ap.add_argument("--use-cplfw-aligned", action="store_true",
                    help="Usa pasta `aligned images/` 224×224 do CPLFW "
                         "(pula detector — mais rápido, mas alignment "
                         "não bate exatamente com ArcFace).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Processa só as primeiras N imagens (0 = todas).")
    ap.add_argument("--gpu", action="store_true",
                    help="use_gpu=True (CoreML no Mac, CUDA no Linux).")
    ap.add_argument("--resume", action="store_true",
                    help="Pula imagens cujo hash já está no parquet alvo.")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="Persiste o parquet a cada N imagens (default 500). "
                         "Reduz pra valores menores em runs longos pra "
                         "minimizar perda em caso de interrupção.")
    args = ap.parse_args()

    csv_path = Path(args.demographics).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not csv_path.exists():
        print(f"[build] CSV não encontrado: {csv_path}", file=sys.stderr)
        return 1

    demo_rows = _load_demographics(csv_path)
    if args.limit > 0:
        demo_rows = demo_rows[: args.limit]
        print(f"[build] --limit {args.limit} — processando subset.")

    # Filtro de resume: pula imagens cujo hash já está no parquet existente.
    # Calculamos hash antes de chamar o FFService pra evitar o custo do
    # detector + encoder em imagens já processadas.
    resume_hashes: Set[str] = set()
    if args.resume:
        resume_hashes = _load_resume_hashes(out_path)
        if resume_hashes:
            print(f"[build] Resume ativo — {len(resume_hashes)} hashes "
                  "já presentes serão pulados.")

    print(f"[build] Carregando FFService (modelo={args.model}, "
          f"gpu={args.gpu})…")
    ff_service = _build_ff_service(args.model, args.gpu)
    print(f"[build] FFService pronto.")

    rows_buffer: List[dict] = []
    n_processed = 0
    n_skipped = 0
    n_errors = 0
    t_start = time.time()
    append_mode = args.resume and out_path.exists()

    for i, row in enumerate(demo_rows, 1):
        img_path = Path(row["image_path"])
        if not img_path.exists():
            n_skipped += 1
            continue

        if resume_hashes:
            try:
                h = _sha256_of_file(img_path)
            except Exception:
                h = None
            if h and h in resume_hashes:
                n_skipped += 1
                continue

        try:
            result = _process_one(ff_service, img_path, row, args.use_cplfw_aligned)
        except Exception as exc:
            print(f"[build] Erro em {img_path.name}: {exc}", file=sys.stderr)
            n_errors += 1
            continue

        if result is None:
            n_skipped += 1
            continue

        rows_buffer.append(result)
        n_processed += 1

        if n_processed % 50 == 0:
            elapsed = time.time() - t_start
            rate = n_processed / max(elapsed, 0.001)
            remaining = (len(demo_rows) - i) / max(rate, 0.001)
            print(f"[build] {n_processed:5d}/{len(demo_rows)} "
                  f"({rate:.1f} img/s, ETA {remaining/60:.1f} min)")

        if len(rows_buffer) >= args.flush_every:
            _flush_rows_to_parquet(rows_buffer, out_path, append=append_mode)
            rows_buffer.clear()
            append_mode = True  # depois do primeiro flush, sempre append

    if rows_buffer:
        _flush_rows_to_parquet(rows_buffer, out_path, append=append_mode)

    elapsed = time.time() - t_start
    print(f"\n[build] Concluído em {elapsed/60:.1f} min")
    print(f"  Processadas: {n_processed}")
    print(f"  Puladas:     {n_skipped}")
    print(f"  Erros:       {n_errors}")
    print(f"  Output:      {out_path}")

    # Verificação final do parquet
    if out_path.exists():
        try:
            final = pq.read_metadata(out_path)
            print(f"  Total no parquet: {final.num_rows} linhas, "
                  f"{final.num_columns} colunas")
        except Exception:
            pass

    return 0 if n_errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
