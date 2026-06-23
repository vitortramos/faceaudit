# faceaudit/models/video/face_provenance.py
"""Proveniência por face extraída de vídeo.

Duas projeções com papéis distintos:

- `FaceProvenance` (FULL): tudo que vai para o sidecar `_meta/<crop>.json`
  (autodescritivo; inclui keypoints, matriz de alinhamento, pose, etc.).
- `FaceProvenanceScalarProjection`: APENAS os escalares seguros para
  promover ao parquet via `FaceInfo.extra_data` — com **namespace**
  `video_extraction_*` para não colidir com os scores autoritativos que
  o `enrich` recalcula. Nunca carrega keypoints/matrizes/embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SIDECAR_SCHEMA = "faceaudit.video_extraction.face_sidecar"
SIDECAR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FaceProvenanceScalarProjection:
    """Whitelist de escalares promovidos ao parquet (via `extra_data`)."""

    target_label: Optional[str] = None
    similarity_to_anchor: Optional[float] = None
    source_video_sha256: Optional[str] = None
    frame_index: Optional[int] = None
    timestamp_s_from_fps: Optional[float] = None
    video_extraction_det_score: Optional[float] = None
    video_extraction_fiqa_score: Optional[float] = None
    video_det_score_decision: Optional[str] = None
    video_fiqa_score_decision: Optional[str] = None
    video_det_score_policy: Optional[str] = None
    video_fiqa_score_policy: Optional[str] = None
    video_min_det_score: Optional[float] = None
    video_min_fiqa_score: Optional[float] = None
    anchor_embedding_fingerprint: Optional[str] = None

    def to_extra_data(self) -> Dict[str, Any]:
        """Dict pronto para `FaceInfo.extra_data` (só chaves não-None)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class FaceProvenance:
    """Registro completo por face salva — origem do sidecar JSON."""

    run_id: str
    run_manifest: str
    identity_label: str

    # artefato (recorte salvo)
    image_file: str
    image_sha256: Optional[str] = None
    image_width_px: Optional[int] = None
    image_height_px: Optional[int] = None
    crop_margin_factor: Optional[float] = None
    color_space: str = "BGR"
    saved_at_utc: Optional[str] = None

    target_index: int = 0

    # vídeo de origem
    source_video_filename: Optional[str] = None
    source_video_sha256: Optional[str] = None

    # quadro
    frame_native_index: Optional[int] = None
    timestamp_s_from_fps: Optional[float] = None
    timestamp_s_capture_prop: Optional[float] = None
    decoded_frame_sha256: Optional[str] = None
    decoded_frame_hash_algorithm: Optional[str] = None
    decoded_frame_color_space: Optional[str] = None
    decoder: Optional[str] = None
    opencv_version: Optional[str] = None

    # detecção
    bbox_xyxy: Optional[List[float]] = None
    det_score: Optional[float] = None
    keypoints_5pt: Optional[List[List[float]]] = None
    align_matrix_2x3: Optional[List[List[float]]] = None
    align_matrix_available: bool = False
    pose: Optional[Dict[str, float]] = None
    gender: Optional[str] = None
    age: Optional[float] = None

    # re-identificação
    reid_method: str = "embedding"
    similarity_to_anchor: Optional[float] = None
    similarity_threshold: Optional[float] = None
    assigned_by: Optional[str] = None            # embedding | tiebreak_track | track_recovery
    track_id: Optional[int] = None
    recognition_model: Optional[str] = None
    anchor_embedding_fingerprint: Optional[str] = None
    margin_to_second_best: Optional[float] = None

    # gates de qualidade
    min_det_score: Optional[float] = None
    det_score_policy: Optional[str] = None
    det_score_decision: Optional[str] = None     # passed | tag_only | route_to_review | hard_reject
    fiqa_score: Optional[float] = None
    min_fiqa_score: Optional[float] = None
    fiqa_score_policy: Optional[str] = None
    fiqa_score_decision: Optional[str] = None

    # decisão + revisão
    auto_decision: str = "accepted"              # accepted | conflict
    review_reasons: List[str] = field(default_factory=list)
    human_review_status: str = "pending"         # pending | accepted | rejected
    human_review_operator: Optional[str] = None
    human_review_timestamp_utc: Optional[str] = None
    human_review_reason: Optional[str] = None

    def to_sidecar_dict(self) -> Dict[str, Any]:
        return {
            "schema": SIDECAR_SCHEMA,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_manifest": self.run_manifest,
            "identity": {"label": self.identity_label, "target_index": self.target_index},
            "artifact": {
                "image_file": self.image_file,
                "image_sha256": self.image_sha256,
                "image_size_px": [self.image_width_px, self.image_height_px],
                "crop_margin_factor": self.crop_margin_factor,
                "color_space": self.color_space,
                "saved_at_utc": self.saved_at_utc,
            },
            "source_video": {
                "filename": self.source_video_filename,
                "sha256": self.source_video_sha256,
            },
            "frame": {
                "native_index": self.frame_native_index,
                "timestamp_s_from_fps": self.timestamp_s_from_fps,
                "timestamp_s_capture_prop": self.timestamp_s_capture_prop,
                "decoded_frame_sha256": self.decoded_frame_sha256,
                "decoded_frame_hash_algorithm": self.decoded_frame_hash_algorithm,
                "decoded_frame_color_space": self.decoded_frame_color_space,
                "decoder": self.decoder,
                "opencv_version": self.opencv_version,
            },
            "detection": {
                "bbox_xyxy": self.bbox_xyxy,
                "det_score": self.det_score,
                "keypoints_5pt": self.keypoints_5pt,
                "align_matrix_2x3": self.align_matrix_2x3,
                "align_matrix_available": self.align_matrix_available,
                "pose": self.pose,
                "gender": self.gender,
                "age": self.age,
            },
            "reidentification": {
                "method": self.reid_method,
                "similarity_to_anchor": self.similarity_to_anchor,
                "threshold": self.similarity_threshold,
                "decision": self.auto_decision,
                "assigned_by": self.assigned_by,
                "track_id": self.track_id,
                "recognition_model": self.recognition_model,
                "anchor_embedding_fingerprint": self.anchor_embedding_fingerprint,
                "margin_to_second_best": self.margin_to_second_best,
            },
            "quality_gates": {
                "det_score": self.det_score,
                "min_det_score": self.min_det_score,
                "det_score_policy": self.det_score_policy,
                "det_score_decision": self.det_score_decision,
                "fiqa_score": self.fiqa_score,
                "min_fiqa_score": self.min_fiqa_score,
                "fiqa_score_policy": self.fiqa_score_policy,
                "fiqa_score_decision": self.fiqa_score_decision,
            },
            "auto_decision": self.auto_decision,
            "review_reasons": list(self.review_reasons),
            "human_review": {
                "status": self.human_review_status,
                "operator": self.human_review_operator,
                "timestamp_utc": self.human_review_timestamp_utc,
                "reason": self.human_review_reason,
            },
        }

    def scalar_projection(self) -> FaceProvenanceScalarProjection:
        return FaceProvenanceScalarProjection(
            target_label=self.identity_label,
            similarity_to_anchor=self.similarity_to_anchor,
            source_video_sha256=self.source_video_sha256,
            frame_index=self.frame_native_index,
            timestamp_s_from_fps=self.timestamp_s_from_fps,
            video_extraction_det_score=self.det_score,
            video_extraction_fiqa_score=self.fiqa_score,
            video_det_score_decision=self.det_score_decision,
            video_fiqa_score_decision=self.fiqa_score_decision,
            video_det_score_policy=self.det_score_policy,
            video_fiqa_score_policy=self.fiqa_score_policy,
            video_min_det_score=self.min_det_score,
            video_min_fiqa_score=self.min_fiqa_score,
            anchor_embedding_fingerprint=self.anchor_embedding_fingerprint,
        )

    def sidecar_filename(self) -> str:
        """`<crop_stem>.json` — mesmo stem do PNG."""
        stem = self.image_file.rsplit(".", 1)[0] if "." in self.image_file else self.image_file
        return f"{stem}.json"
