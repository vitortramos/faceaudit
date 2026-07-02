# faceaudit/models/video/video_extraction_config.py

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Optional, Tuple

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

    # --- Multivídeo (Fase 3) ---
    # Busca o(s) alvo(s) em VÁRIOS vídeos numa run só. `video_path` (1º vídeo)
    # é mantido para compat; `resolved_video_paths()` devolve a lista efetiva.
    # `case_name` nomeia a run quando há >1 vídeo (default = stem do 1º).
    video_paths: List[str] = field(default_factory=list)
    case_name: Optional[str] = None
    # Janela de tempo POR vídeo (paralela a `video_paths`). Cada item:
    # [start_s, end_s] (end_s None = até o fim) ou None = vídeo inteiro. Quando
    # ausente/None, cai na janela global (start_s/end_s).
    video_windows: List[Optional[List[float]]] = field(default_factory=list)

    # --- Amostragem temporal ---
    target_fps: float = 2.0
    every_n_frames: Optional[int] = None     # None → derivar de target_fps + fps_nominal
    start_s: float = 0.0
    end_s: Optional[float] = None            # None → até o fim

    # --- Recorte / identidade ---
    margin: float = 2.0                      # fator de expansão da bbox (extend_bbox)
    similarity_threshold: float = 0.30       # limiar de IDENTIDADE (cosseno); calibrar na prévia
    select_single_face_by: str = "size"      # "size" | "centrality" (desempate da âncora)
    det_size: int = 640                      # tamanho do detector; ↑ p/ faces pequenas (vigilância: 960)
    det_thresh: float = 0.5                  # limiar do detector; ↓ p/ faces de baixa confiança

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

    # --- Proveniência (reprodutibilidade; NÃO afeta a extração) ---
    # Lista COMPLETA das imagens de referência carregadas na janela — inclusive
    # as que não viraram âncora de nenhum alvo. Só alimenta o manifesto; fica
    # FORA de `to_manifest_dict()`/run_tag (não muda a saída da extração nem o
    # run_id).
    reference_images: List[str] = field(default_factory=list)

    def resolved_video_paths(self) -> List[str]:
        """Lista efetiva de vídeos a percorrer (multivídeo); cai no `video_path`
        único quando `video_paths` está vazio (compat)."""
        return list(self.video_paths) if self.video_paths else [self.video_path]

    def window_for(self, index: int) -> Tuple[float, Optional[float]]:
        """Janela (start_s, end_s) do vídeo `index`: usa `video_windows[index]`
        se definida (end_s None = até o fim), senão a janela global."""
        if 0 <= index < len(self.video_windows):
            w = self.video_windows[index]
            if w:
                start = float(w[0]) if w[0] is not None else 0.0
                end = None if (len(w) < 2 or w[1] is None) else float(w[1])
                return start, end
        return self.start_s, self.end_s

    def to_manifest_dict(self) -> dict:
        """Projeção dos parâmetros para o manifesto (enums como string)."""
        return {
            "case_name": self.case_name,
            "video_windows": self.video_windows,
            "target_fps": self.target_fps,
            "every_n_frames": self.every_n_frames,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "margin": self.margin,
            "similarity_threshold": self.similarity_threshold,
            "select_single_face_by": self.select_single_face_by,
            "det_size": self.det_size,
            "det_thresh": self.det_thresh,
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
