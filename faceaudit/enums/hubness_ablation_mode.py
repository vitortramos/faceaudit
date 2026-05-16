# faceaudit/enums/hubness_ablation_mode.py

from __future__ import annotations
from enum import Enum


class HubnessAblationMode(Enum):
    """
    Define o critério usado pela instrução HUBNESS_ABLATION para recortar
    progressivamente o dataset em cenários.

    - BY_QUANTITY: prioriza as N melhores fotos por identidade ordenadas
      por qualidade (FIQA). Cenários: 1, 2, 3 fotos por ID + dataset total.
    - BY_POSE: prioriza as N fotos mais frontais por identidade
      (menor abs(yaw)+abs(pitch)). Cenários: 1, 2, 3 fotos por ID + total.
    """

    BY_QUANTITY = "by_quantity"
    BY_POSE = "by_pose"

    @property
    def label_pt(self) -> str:
        return {
            "by_quantity": "Por quantidade (top-N por qualidade)",
            "by_pose":     "Por pose (top-N mais frontais)",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        return self.value

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> HubnessAblationMode:
        """Reconstrói o enum a partir de string com fallback para BY_QUANTITY."""
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.BY_QUANTITY
