# faceaudit/models/video/video_extraction_config.py

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.quality_gate_policy import QualityGatePolicy
from faceaudit.enums.video_extraction_mode import VideoExtractionMode


@dataclass
class VideoExtractionConfig(JSONSerializableMixin):
    """Parâmetros de uma extração de faces de vídeo (um run).

    A taxa é expressa em **FPS-alvo** (humano); `every_n_frames` é
    derivado de `target_fps` e do fps nominal do vídeo quando None.
    Os gates de qualidade default são `TAG_ONLY` (só registram); os
    limiares `min_*` começam None e são calibrados pela distribuição
    empírica na prévia.
    """

    video_path: str
    output_root: str

    # --- Amostragem temporal ---
    target_fps: float = 2.0
    every_n_frames: Optional[int] = None     # None → derivar de target_fps + fps_nominal
    start_s: float = 0.0
    end_s: Optional[float] = None            # None → até o fim

    # --- Recorte / identidade ---
    margin: float = 2.0                      # fator de expansão da bbox (extend_bbox)
    similarity_threshold: float = 0.30       # limiar de IDENTIDADE (cosseno); calibrar na prévia
    select_single_face_by: str = "size"      # "size" | "centrality" (desempate da âncora)

    # --- Gates de qualidade (separados da identidade) ---
    min_det_score: Optional[float] = None
    min_fiqa_score: Optional[float] = None
    det_score_policy: QualityGatePolicy = QualityGatePolicy.TAG_ONLY
    fiqa_score_policy: QualityGatePolicy = QualityGatePolicy.TAG_ONLY

    # --- Saída / modo ---
    mode: VideoExtractionMode = VideoExtractionMode.PRODUCER
    recognition_model: Optional[str] = None  # modelo de re-id; None → default do FFService
    compute_ff: bool = True                  # enrich ponta-a-ponta
    compute_ofiq: bool = False
    compute_serfiq: bool = False
    save_annotated_anchor: bool = True
    save_annotated_accepted: bool = False
    save_decoded_frame_hash: bool = True     # default ON só para aceitos
    run_tag: Optional[str] = None            # sufixo do run/parquet; None → derivado

    def to_manifest_dict(self) -> dict:
        """Projeção dos parâmetros para o manifesto (enums como string)."""
        return {
            "target_fps": self.target_fps,
            "every_n_frames": self.every_n_frames,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "margin": self.margin,
            "similarity_threshold": self.similarity_threshold,
            "select_single_face_by": self.select_single_face_by,
            "min_det_score": self.min_det_score,
            "min_fiqa_score": self.min_fiqa_score,
            "det_score_policy": self.det_score_policy.to_str(),
            "fiqa_score_policy": self.fiqa_score_policy.to_str(),
            "mode": self.mode.to_str(),
            "recognition_model": self.recognition_model,
            "compute_ff": self.compute_ff,
            "compute_ofiq": self.compute_ofiq,
            "compute_serfiq": self.compute_serfiq,
            "save_annotated_anchor": self.save_annotated_anchor,
            "save_annotated_accepted": self.save_annotated_accepted,
            "save_decoded_frame_hash": self.save_decoded_frame_hash,
            "run_tag": self.run_tag,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VideoExtractionConfig":
        d = dict(data or {})
        if "det_score_policy" in d:
            d["det_score_policy"] = QualityGatePolicy.from_str(d["det_score_policy"])
        if "fiqa_score_policy" in d:
            d["fiqa_score_policy"] = QualityGatePolicy.from_str(d["fiqa_score_policy"])
        if "mode" in d:
            d["mode"] = VideoExtractionMode.from_str(d["mode"])
        valid = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)
