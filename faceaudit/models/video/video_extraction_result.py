# faceaudit/models/video/video_extraction_result.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from faceaudit.models.video.face_provenance import FaceProvenance


@dataclass
class VideoExtractionResult:
    """Resultado de um run de extração (objeto de RUNTIME).

    Aponta para a pasta do run e os artefatos de proveniência; carrega
    a lista de `FaceProvenance` das faces aceitas (modo produtor) que
    depois alimenta o importador / a grade de revisão.
    """

    run_dir: str
    run_id: str
    extraction_status: str               # completed | cancelled | failed
    manifest_path: str
    decisions_path: str
    frames_path: str

    n_frames_sampled: int = 0
    n_detections: int = 0
    n_accepted_auto: int = 0
    n_rejected_auto: int = 0
    n_no_detection_frames: int = 0

    target_labels: List[str] = field(default_factory=list)
    faces: List[FaceProvenance] = field(default_factory=list)
