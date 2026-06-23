# faceaudit/models/video/video_info.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VideoInfo:
    """Propriedades de um arquivo de vídeo (resultado de `probe_video`).

    Usado pela UI (scrubber: total de quadros, fps) e gravado no
    manifesto do run para reprodutibilidade/custódia.
    """

    path: str
    filename: str
    sha256: Optional[str]
    file_size_bytes: int
    fps_reported: float
    frame_count_reported: int
    duration_reported_s: float
    resolution: Tuple[int, int]          # (largura, altura)
    opencv_version: str
    decoder_backend: Optional[str] = None

    def to_manifest_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "fps_reported": self.fps_reported,
            "frame_count_reported": self.frame_count_reported,
            "duration_reported_s": self.duration_reported_s,
            "resolution": list(self.resolution),
            "opencv_version": self.opencv_version,
            "decoder_backend": self.decoder_backend,
        }
