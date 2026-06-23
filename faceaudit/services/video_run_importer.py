# faceaudit/services/video_run_importer.py
"""Importador de um run de extração de vídeo → `FaceDataset`.

Fonte ÚNICA de verdade para promover a proveniência por face ao parquet:
lê o manifesto + sidecars de um run produzido, valida os hashes dos
recortes, e monta um `FaceDataset` RELATIVE (apontando para os crops do
próprio run) com a projeção escalar em `extra_data`. É usado tanto pelo
modo produtor→importação manual quanto pelo ponta-a-ponta (Fase 6).

Invariante (Fase 6): o dataset final só inclui faces com
`human_review.status` em `accepted_statuses` (default `("accepted",)`);
a revisão (Fase 3) flipa pending→accepted/rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.video.face_provenance import FaceProvenanceScalarProjection
from faceaudit.services.hash_service import HashService
from faceaudit.models.column_catalog import (
    ColumnSpec, ColumnRole, ColumnSource, ColumnDType,
    QualityDirection, QualitySubtype,
)

SELECTION_WARNING = (
    "Dataset criado por seleção condicionada a similaridade biométrica com "
    "uma âncora. Não representa amostra populacional neutra."
)

_CATEGORY_VIDEO = "Vídeo (extração)"


def build_video_column_specs() -> Dict[str, ColumnSpec]:
    """`ColumnSpec` para as colunas `video_*`/proveniência promovidas ao
    parquet — sem isso o Schema Report da UI fica sem label/categoria."""
    def meta_str(name, label):
        return ColumnSpec(name=name, label=label, role=ColumnRole.METADATA,
                          source=ColumnSource.DERIVED, dtype=ColumnDType.STR,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO)

    def meta_num(name, label, dtype):
        return ColumnSpec(name=name, label=label, role=ColumnRole.METADATA,
                          source=ColumnSource.DERIVED, dtype=dtype,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO)

    def qual(name, label, scale=None):
        return ColumnSpec(name=name, label=label, role=ColumnRole.QUALITY,
                          source=ColumnSource.DERIVED, dtype=ColumnDType.FLOAT,
                          editable=False, visible_default=False, category=_CATEGORY_VIDEO,
                          metric_name=name, direction=QualityDirection.HIGHER_IS_BETTER,
                          scale=scale, quality_subtype=QualitySubtype.AUX)

    specs = {
        "target_label": meta_str("target_label", "Alvo (rótulo)"),
        "similarity_to_anchor": qual("similarity_to_anchor", "Similaridade c/ âncora", (-1.0, 1.0)),
        "source_video_sha256": meta_str("source_video_sha256", "SHA-256 do vídeo"),
        "frame_index": meta_num("frame_index", "Índice do quadro", ColumnDType.INT),
        "timestamp_s_from_fps": meta_num("timestamp_s_from_fps", "Timestamp (s)", ColumnDType.FLOAT),
        "video_extraction_det_score": qual("video_extraction_det_score", "Det. (triagem vídeo)", (0.0, 1.0)),
        "video_extraction_fiqa_score": qual("video_extraction_fiqa_score", "FIQA (triagem vídeo)"),
        "video_det_score_decision": meta_str("video_det_score_decision", "Decisão det (vídeo)"),
        "video_fiqa_score_decision": meta_str("video_fiqa_score_decision", "Decisão FIQA (vídeo)"),
        "video_det_score_policy": meta_str("video_det_score_policy", "Política det (vídeo)"),
        "video_fiqa_score_policy": meta_str("video_fiqa_score_policy", "Política FIQA (vídeo)"),
        "video_min_det_score": meta_num("video_min_det_score", "Min det (vídeo)", ColumnDType.FLOAT),
        "video_min_fiqa_score": meta_num("video_min_fiqa_score", "Min FIQA (vídeo)", ColumnDType.FLOAT),
        "anchor_embedding_fingerprint": meta_str("anchor_embedding_fingerprint", "Fingerprint da âncora"),
    }
    return specs


class VideoRunImporter:
    def __init__(self, hash_service: Optional[HashService] = None):
        self._hash = hash_service or HashService()

    def import_run(
        self,
        run_dir,
        *,
        dataset_name: Optional[str] = None,
        accepted_statuses: Tuple[str, ...] = ("accepted",),
        validate_hashes: bool = True,
        logger: Optional[Callable[[str], None]] = None,
        enrich: bool = False,
        pipeline: Any = None,
        ingestion_config: Any = None,
        require_review: bool = True,
    ) -> FaceDataset:
        run_dir = Path(run_dir)
        meta_dir = run_dir / "_meta"
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"Pasta _meta/ não encontrada em {run_dir}")
        manifest = self._load_manifest(meta_dir)

        # Invariante END_TO_END: o parquet só é finalizado após o commit da
        # revisão (review_status completed ou skipped-override).
        if enrich and require_review:
            rs = manifest.get("review_status", "not_started")
            if rs not in ("completed", "skipped"):
                raise RuntimeError(
                    f"enrich bloqueado: revisão não concluída (review_status={rs!r}). "
                    f"Conclua a revisão (ou marque skipped como override) antes do dataset final."
                )

        faces: List[FaceInfo] = []
        n_skipped_status = n_hash_mismatch = n_missing = 0
        for sidecar in sorted(meta_dir.glob("*.json")):
            nm = sidecar.name
            if nm.startswith("_run_") or nm.startswith("_decisions_") or nm.startswith("_frames_"):
                continue
            try:
                d = json.loads(sidecar.read_text(encoding="utf-8"))
            except Exception as e:
                if logger:
                    logger(f"sidecar ilegível {nm}: {e}")
                continue
            status = (d.get("human_review") or {}).get("status", "pending")
            if status not in accepted_statuses:
                n_skipped_status += 1
                continue

            label = (d.get("identity") or {}).get("label")
            image_file = (d.get("artifact") or {}).get("image_file")
            if not label or not image_file:
                continue
            abs_path = run_dir / label / image_file
            if not abs_path.exists():
                n_missing += 1
                if logger:
                    logger(f"recorte ausente {label}/{image_file}")
                continue
            expected = (d.get("artifact") or {}).get("image_sha256")
            if validate_hashes and expected:
                actual = self._hash.calculate_file_hash(abs_path)
                if actual and actual != expected:
                    n_hash_mismatch += 1
                    if logger:
                        logger(f"hash divergente {label}/{image_file}")
                    continue

            extra = FaceProvenanceScalarProjection.from_sidecar_dict(d).to_extra_data()
            fi = FaceInfo(image_path=f"{label}/{image_file}", identity=label, hash=expected)
            fi.extra_data = extra
            faces.append(fi)

        name = dataset_name or run_dir.name
        ds = FaceDataset(name=name, faces=faces, path_mode=PathMode.RELATIVE, path_root=".")
        ds._parquet_dir = run_dir.resolve()
        ds.column_specs = build_video_column_specs()
        ds.provenance_metadata = self._build_provenance_metadata(manifest)

        if logger:
            logger(f"importadas {len(faces)} face(s) | status-pulados={n_skipped_status} "
                   f"| hash-divergentes={n_hash_mismatch} | ausentes={n_missing}")

        if enrich:
            if pipeline is None:
                raise ValueError("enrich=True requer um pipeline (FeatureExtractionPipeline).")
            cfg = ingestion_config or self._default_ingestion_config(ds.name, run_dir)
            pipeline.enrich_existing_dataset(ds, cfg)
            self._finalize_manifest(meta_dir, run_dir, ds, now=None)

        return ds

    # ---- helpers ----
    @staticmethod
    def _default_ingestion_config(name: str, run_dir: Path) -> Any:
        """DatasetIngestionConfig mínimo p/ enrich (FF apenas; input_path é
        ignorado pelo enrich, mas o DTO o exige)."""
        from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
        return DatasetIngestionConfig(
            input_path=run_dir, dataset_name=name,
            compute_ff=True, compute_ofiq=False, compute_serfiq=False,
        )

    def _finalize_manifest(self, meta_dir: Path, run_dir: Path, ds: FaceDataset, now=None) -> None:
        """Após o enrich, grava no manifesto dataset_status=enriched +
        final_dataset_path/sha256/n_faces (fecha a trilha run→parquet)."""
        manifests = sorted(meta_dir.glob("_run_*.json"))
        if not manifests:
            return
        parquet_path = Path(ds._parquet_dir or run_dir) / f"{ds.name}.parquet"
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        manifest["dataset_status"] = "enriched"
        manifest["final_dataset"] = {
            "path": parquet_path.name,
            "sha256": self._hash.calculate_file_hash(parquet_path) if parquet_path.exists() else None,
            "n_faces": len(ds),
        }
        manifests[0].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def audit_metadata(self, run_dir) -> Dict[str, Any]:
        """Metadata para o audit log VIDEO_EXTRACT (emitido pela UI na Fase 5)."""
        manifest = self._load_manifest(Path(run_dir) / "_meta")
        summary = manifest.get("summary") or {}
        review = manifest.get("review") or {}
        return {
            "run_id": manifest.get("run_id"),
            "video_sha256": (manifest.get("source_video") or {}).get("sha256"),
            "n_frames_sampled": summary.get("n_frames_sampled"),
            "n_accepted_auto": summary.get("n_accepted_auto"),
            "n_accepted_after_review": review.get("n_accepted"),
            "n_rejected_human": review.get("n_rejected"),
            "review_skipped": review.get("skipped"),
        }

    @staticmethod
    def _load_manifest(meta_dir: Path) -> Dict[str, Any]:
        candidates = sorted(meta_dir.glob("_run_*.json"))
        if not candidates:
            return {}
        try:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def _build_provenance_metadata(manifest: Dict[str, Any]) -> Dict[str, str]:
        targets = manifest.get("targets") or []
        model = manifest.get("recognition_model")
        threshold = targets[0].get("similarity_threshold") if targets else None
        run_id = manifest.get("run_id")
        meta = {
            "dataset_origin_type": "VIDEO_TARGET_EXTRACTION",
            "selection_conditioned_on_embedding": "true",
            "selection_warning": SELECTION_WARNING,
        }
        if model:
            meta["selection_anchor_model"] = str(model)
        if threshold is not None:
            meta["selection_threshold"] = str(threshold)
        if run_id:
            meta["source_run_id"] = str(run_id)
        return meta
