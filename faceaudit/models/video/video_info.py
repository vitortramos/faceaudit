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
    # Rotação declarada no contêiner, em graus horários. O OpenCV a ignora
    # ao decodificar — vídeo de celular em retrato vem deitado —, então o
    # FaceAudit a aplica a cada quadro lido. Registrada aqui porque vai ao
    # manifesto do run: sem isso ninguém reconstrói depois por que os
    # recortes estão de pé e o arquivo, aberto num player, aparece deitado.
    container_rotation: int = 0

    @property
    def resolution_effective(self) -> Tuple[int, int]:
        """(largura, altura) dos quadros COMO O SISTEMA OS VÊ.

        Girar 90° ou 270° troca os dois. `resolution` continua sendo o que o
        decodificador reporta — as duas convivem porque confundi-las é que
        geraria engano: uma descreve o arquivo, a outra o que foi detectado.
        """
        largura, altura = self.resolution
        if self.container_rotation in (90, 270):
            return (altura, largura)
        return (largura, altura)

    def to_manifest_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "fps_reported": self.fps_reported,
            "frame_count_reported": self.frame_count_reported,
            "duration_reported_s": self.duration_reported_s,
            "resolution": list(self.resolution),
            "container_rotation": self.container_rotation,
            "resolution_effective": list(self.resolution_effective),
            "opencv_version": self.opencv_version,
            "decoder_backend": self.decoder_backend,
        }
