"""ImagePipelineBatchService — aplica pipeline de imagens em todo o
dataset, produzindo dataset derivado enxuto (PR I5).

Output:
    <target_dir>/
    ├── <dataset_name>__<pipeline_name>.parquet  ← dataset derivado
    ├── images/
    │   ├── <file>.png
    │   └── ...
    └── manifest.json                             ← anchor forense

Decisões consolidadas (PR I0):
  - Output enxuto: novo FaceDataset com `image_path`/`hash`/`file_name`/
    `identity` apenas. Sem ff_info/ofiq_info/serfiq_info — perito roda
    Enrich depois quando precisar.
  - Naming: PRESERVE com `.png` forçado (sempre PNG, lossless — pipelines
    lossy ficam evidentes pelo conteúdo, não pela serialização).
  - Idempotência via `skip_existing=True`: PNG já existe + hash bate → pula.
  - Determinismo: `seed_jitter = face.image_path` propaga pro pipeline.
  - Audit log: `AuditOperation.COPY_IMAGES` no parquet original.

Service stateless. Construa uma instância por chamada ou compartilhe.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.image_processing.batch_processing_result import (
    BatchProcessingResult,
)
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.pipelines.image_processing_pipeline import (
    ImageProcessingPipeline,
)
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.hash_service import HashService


_IMAGES_SUBDIR = "images"
_MANIFEST_NAME = "manifest.json"


class ImagePipelineBatchService:
    """Aplica pipeline em todo dataset → cria dataset derivado +
    manifest. Stateless."""

    def run(
        self,
        dataset: FaceDataset,
        pipeline_config: ImageProcessingPipelineConfig,
        target_dir: Path,
        *,
        ff_service=None,
        skip_existing: bool = True,
        output_filename: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> BatchProcessingResult:
        """Aplica pipeline em todo dataset.

        Args:
            dataset: dataset de entrada (não é mutado).
            pipeline_config: configuração do pipeline a aplicar.
            target_dir: pasta destino. Será criada se não existir.
                Imagens em `target_dir/images/`, parquet em
                `target_dir/<output_filename>.parquet` (ou
                `target_dir/<name>__<pipeline>.parquet` quando
                `output_filename` é None), manifest em
                `target_dir/manifest.json`.
            ff_service: necessário se `input_source=ALIGNED_FACE` ou
                algum step usa FFService (ex: CropMode.FACE_DETECTED).
            skip_existing: idempotência — PNGs já existentes com mesmo
                hash são pulados.
            output_filename: stem do parquet de saída (sem extensão).
                Default = `<dataset>__<pipeline>` (compat). Falhas vão
                pra `<stem>__errors.parquet`.
            progress_callback: chamado por face com `(idx, total, msg)`.

        Returns:
            `BatchProcessingResult` agregando totais.
        """
        target_dir = Path(target_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        images_dir = target_dir / _IMAGES_SUBDIR
        images_dir.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        # Constrói pipeline uma vez (factory + processors)
        pipeline = ImageProcessingPipeline(
            pipeline_config, ff_service=ff_service,
        )

        new_faces: List[FaceInfo] = []
        failures: List[Tuple[str, str]] = []
        n_processed = 0
        n_succeeded = 0
        n_skipped = 0
        used_names: set[str] = set()
        hash_service = HashService()

        for idx, face in enumerate(dataset.faces):
            n_processed += 1
            if progress_callback:
                progress_callback(idx, len(dataset), face.image_path or "")

            try:
                result = self._process_one(
                    face, dataset, pipeline,
                    pipeline_config.input_source,
                    images_dir, used_names, skip_existing,
                    ff_service, hash_service,
                )
            except Exception as e:
                failures.append((face.image_path or "<empty>", str(e)))
                continue

            if result["status"] == "skipped":
                n_skipped += 1
            else:
                n_succeeded += 1

            if result["face"] is not None:
                new_faces.append(result["face"])
                used_names.add(result["rel_name"])

        # Cria FaceDataset enxuto (sem ff_info/ofiq_info/serfiq_info)
        output_stem = (
            output_filename.strip() if output_filename
            else self._build_derived_name(
                dataset.name, pipeline_config.name,
            )
        )
        # is_aligned propaga quando o pipeline foi configurado com
        # input_source=ALIGNED_FACE: o batch service aplica
        # detect+align nas imagens fontes antes dos steps, então o
        # output é sempre faces alinhadas 112×112. Sem essa marca, o
        # FFService re-detectaria a face dentro do crop alinhado no
        # próximo enrich (degrada qualidade do embedding).
        derived_is_aligned = (
            pipeline_config.input_source == ImageInputSource.ALIGNED_FACE
        )
        new_dataset = FaceDataset(
            name=output_stem,
            description=(
                f"Derivado de '{dataset.name}' via pipeline "
                f"'{pipeline_config.name}'."
            ),
            faces=new_faces,
            path_mode=PathMode.RELATIVE,
            path_root=_IMAGES_SUBDIR,
            is_aligned=derived_is_aligned,
        )

        # Salva parquet
        parquet_path = target_dir / f"{output_stem}.parquet"
        new_dataset.to_parquet(parquet_path)

        # Salva parquet de erros quando houver — mesmo schema do
        # dataset original (image_path/identity/file_name/hash) +
        # coluna `error_message`. Permite ao perito abrir esse arquivo
        # como dataset normal pra inspecionar/exportar/reanálisar as
        # falhas.
        error_parquet_path: Optional[Path] = None
        if failures:
            error_parquet_path = target_dir / f"{output_stem}__errors.parquet"
            self._save_error_parquet(
                dataset, failures, error_parquet_path, output_stem,
            )

        duration = time.time() - start_time

        # Manifest forense
        manifest = {
            "pipeline_name": pipeline_config.name,
            "pipeline_seed": pipeline_config.seed,
            "input_source": pipeline_config.input_source.value,
            "step_types": [s.step_type.value for s in pipeline_config.steps],
            "n_steps": len(pipeline_config.steps),
            "n_processed": n_processed,
            "n_succeeded": n_succeeded,
            "n_skipped": n_skipped,
            "n_failed": len(failures),
            "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "duration_seconds": round(duration, 3),
            "source_dataset": dataset.name,
            "operator": AuditLogService.current_operator(),
            "pipeline_config": pipeline_config.to_dict(),
        }
        manifest_path = target_dir / _MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return BatchProcessingResult(
            target_dir=target_dir,
            parquet_path=parquet_path,
            pipeline_name=pipeline_config.name,
            n_processed=n_processed,
            n_succeeded=n_succeeded,
            n_skipped=n_skipped,
            n_failed=len(failures),
            failures=tuple(failures),
            duration_seconds=duration,
            error_parquet_path=error_parquet_path,
        )

    # ── Per-face ───────────────────────────────────────────────────────────

    def _process_one(
        self,
        face: FaceInfo,
        dataset: FaceDataset,
        pipeline: ImageProcessingPipeline,
        input_source: ImageInputSource,
        images_dir: Path,
        used_names: set[str],
        skip_existing: bool,
        ff_service,
        hash_service: HashService,
    ) -> dict:
        """Processa uma face. Retorna dict com `status` (success/skipped)
        + `face` (FaceInfo novo ou None) + `rel_name`."""
        # Resolve src
        src_path = dataset.resolve_image_path(face)
        if not src_path.exists():
            raise RuntimeError(f"arquivo fonte não encontrado: {src_path}")

        # Carrega imagem
        bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"falha ao decodificar: {src_path.name}")

        # Se ALIGNED_FACE, alinha primeiro
        if input_source == ImageInputSource.ALIGNED_FACE:
            if ff_service is None:
                raise RuntimeError(
                    "input_source=ALIGNED_FACE requer FFService"
                )
            aligned = ff_service.align_face_from_array(bgr)
            if aligned is None:
                raise RuntimeError("nenhuma face detectada")
            bgr = aligned

        # Aplica pipeline (com seed_jitter pra determinismo)
        jitter = str(face.image_path or "")
        processed = pipeline.process_image(bgr, seed_jitter=jitter)

        # Determina filename — sempre .png
        rel_name = self._build_dest_name(face, src_path, used_names)
        dst = images_dir / rel_name

        # Idempotência: PNG já existe?
        if skip_existing and dst.exists():
            existing_hash = hash_service.calculate_file_hash(dst)
            new_face = self._clone_face_minimal(
                face, rel_name, existing_hash,
            )
            return {"status": "skipped", "face": new_face, "rel_name": rel_name}

        # Grava PNG (lossless)
        dst.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst), processed)
        if not ok:
            raise RuntimeError("cv2.imwrite retornou False")

        # Hash do PNG resultante
        file_hash = hash_service.calculate_file_hash(dst)

        new_face = self._clone_face_minimal(face, rel_name, file_hash)
        return {"status": "success", "face": new_face, "rel_name": rel_name}

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_dest_name(
        face: FaceInfo, src_path: Path, used_names: set,
    ) -> str:
        """PRESERVE com `.png` forçado. Em colisão (a.png já existe na
        rodada atual), sufixa `_1`, `_2`, ..."""
        base = face.file_name or src_path.name
        stem = Path(base).stem
        candidate = f"{stem}.png"
        if candidate not in used_names:
            return candidate
        i = 1
        while True:
            candidate = f"{stem}_{i}.png"
            if candidate not in used_names:
                return candidate
            i += 1

    @staticmethod
    def _clone_face_minimal(
        face: FaceInfo, new_image_path: str, new_hash: Optional[str],
    ) -> FaceInfo:
        """Cria FaceInfo enxuto: só image_path/file_name/hash/identity.

        Descarta ff_info, ofiq_info, serfiq_info, embedding_*, etc. —
        decisão #4 do plano (perito enriquece depois).
        """
        return FaceInfo(
            image_path=new_image_path,
            file_name=Path(new_image_path).name,
            hash=new_hash,
            identity=face.identity,
            origin=face.origin,
            ethnicity=face.ethnicity,
            gender=face.gender,
        )

    @staticmethod
    def _build_derived_name(dataset_name: str, pipeline_name: str) -> str:
        """`<dataset>__<pipeline>` — sanitizado pra safe filename."""
        bad = '\\/:*?"<>| '
        ds = dataset_name or "dataset"
        pn = pipeline_name or "pipeline"
        for ch in bad:
            ds = ds.replace(ch, "_")
            pn = pn.replace(ch, "_")
        return f"{ds}__{pn}"

    @staticmethod
    def _save_error_parquet(
        source_dataset: FaceDataset,
        failures: List[Tuple[str, str]],
        error_parquet_path: Path,
        output_stem: str,
    ) -> None:
        """Salva parquet com 1 linha por face que falhou + coluna
        `error_message`. Schema mínimo (image_path/identity/file_name/
        hash/error_message) — perito abre como dataset normal pra
        ordenar/filtrar/inspecionar e gerar lista pra reanálise.

        Os paths apontam pra IMAGEM-FONTE (não pro PNG, que não
        chegou a ser gerado), com `path_mode=ABSOLUTE` herdado do
        dataset original quando aplicável.
        """
        # Indexa o dataset-fonte por image_path pra resgatar identity/
        # file_name/hash de cada falha.
        by_path = {f.image_path: f for f in source_dataset.faces}
        error_faces: List[FaceInfo] = []
        # Vai guardar a mensagem pra cada índice — injetada via
        # column_specs do FaceDataset.to_parquet não é trivial, então
        # depois do save patchamos o parquet adicionando a coluna.
        messages: List[str] = []
        for img_path, msg in failures:
            src = by_path.get(img_path)
            if src is None:
                # Falha de path desconhecido (ex: <empty>) — cria
                # face placeholder pra preservar a linha.
                error_faces.append(FaceInfo(image_path=img_path))
            else:
                # Clone enxuto preservando metadata da fonte.
                error_faces.append(FaceInfo(
                    image_path=src.image_path,
                    file_name=src.file_name,
                    identity=src.identity,
                    hash=src.hash,
                    origin=src.origin,
                    ethnicity=src.ethnicity,
                    gender=src.gender,
                ))
            messages.append(msg)

        error_ds = FaceDataset(
            name=f"{output_stem}__errors",
            description=(
                f"Falhas do pipeline em '{source_dataset.name}'. "
                f"Coluna `error_message` traz a razão de cada falha."
            ),
            faces=error_faces,
            # Paths apontam pra fonte (absolutos do dataset original)
            # — preserva path_mode/path_root pra que se abra correto.
            path_mode=source_dataset.path_mode,
            path_root=source_dataset.path_root,
        )
        error_ds.to_parquet(error_parquet_path)

        # Injeta coluna `error_message` no parquet recém-escrito.
        # Não dá pra passar via FaceInfo (modelo não tem o campo),
        # então re-escreve com a coluna extra preservando a metadata.
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pq.read_table(error_parquet_path)
            # Garante mesmo length (defensivo).
            if table.num_rows == len(messages):
                meta = table.schema.metadata or {}
                table = table.append_column(
                    "error_message", pa.array(messages, type=pa.string()),
                )
                table = table.replace_schema_metadata(meta)
                pq.write_table(table, error_parquet_path)
        except Exception:
            # Best-effort — sem a coluna, o parquet de erros ainda
            # lista os paths das falhas; o motivo pode ser inferido
            # do manifest.json (que registra os totals).
            pass
