# faceaudit/enums/pipeline_mode.py

from __future__ import annotations
from enum import Enum


class PipelineMode(Enum):
    """
    Define a topologia do pipeline de transformação de embeddings.

    - LINEAR:
        Um único fluxo de dados. Usado quando não há adaptação de domínio.
        (Ex: Centering → Whitening → L2)

    - DUAL:
        Dois fluxos (Source e Target). Usado em cenários com adaptação de domínio.
        (Ex: CORAL, LDA supervisionado, etc.)
    """

    LINEAR = "linear"
    DUAL = "dual"

    @property
    def label_pt(self) -> str:
        return {
            "linear": "Linear (sem adaptação)",
            "dual":   "Dual (source + target, com adaptação)",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        """Retorna o identificador textual do modo."""
        return self.value

    def to_str(self) -> str:
        """Retorna o valor persistível para JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> PipelineMode:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.LINEAR