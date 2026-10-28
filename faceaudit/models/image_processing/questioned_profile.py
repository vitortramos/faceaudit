"""Perfil da imagem questionada que um pipeline de imagens reproduz.

Fica gravado no próprio pipeline: quem abrir o JSON (ou o manifesto do
dataset adaptado) sabe QUAL imagem serviu de alvo — pelo SHA-256, não só
pelo nome — e quais eram as medidas dela. A imagem em si não é copiada
para o pipeline; o caminho é só uma conveniência para reabrir no editor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)


@dataclass(frozen=True)
class QuestionedProfile:
    file_name: str
    source_path: str
    sha256: str
    measured_at: str
    face: Optional[ImageFaceGeometry]
    # "detector" quando a face foi localizada pelo detector; None quando
    # não foi encontrada (as medidas são então da imagem inteira).
    face_source: Optional[str]
    measures: Any  # DegradationMeasures

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_name": self.file_name,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "measured_at": self.measured_at,
            "face": None if self.face is None else self.face.to_dict(),
            "face_source": self.face_source,
            "measures": None if self.measures is None else self.measures.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["QuestionedProfile"]:
        if not data:
            return None
        from faceaudit.services.image_degradation_profile_service import (
            DegradationMeasures,
        )

        return cls(
            file_name=str(data.get("file_name", "")),
            source_path=str(data.get("source_path", "")),
            sha256=str(data.get("sha256", "")),
            measured_at=str(data.get("measured_at", "")),
            face=ImageFaceGeometry.from_dict(data.get("face")),
            face_source=data.get("face_source"),
            measures=DegradationMeasures.from_dict(data.get("measures")),
        )
