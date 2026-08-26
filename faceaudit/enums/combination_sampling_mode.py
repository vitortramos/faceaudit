# faceaudit/enums/combination_sampling_mode.py

from __future__ import annotations
from enum import Enum


class CombinationSamplingMode(Enum):
    """
    Como escolher QUAIS combinações gerar quando C(N,k) excede o teto de
    templates por identidade (`max_synthetic_templates_per_id`).

    Só tem efeito em `AggregationMethod.COMBINATORIAL` e apenas quando o total
    de combinações possíveis passa do teto — abaixo dele todas são geradas e a
    cobertura é perfeitamente uniforme por construção.

    - BALANCED_COVERAGE: escolhe, a cada passo, as k faces menos usadas até
      então. Cada face acaba participando de ⌊teto·k/N⌋ ou ⌈teto·k/N⌉ templates
      — amplitude 1 — e nenhuma fica de fora.
    - UNIFORM: sorteia combinações com probabilidade igual entre todas as
      possíveis. É uma amostra não-enviesada do espaço combinatório, mas o
      sorteio é uniforme sobre COMBINAÇÕES, não sobre FACES: a cobertura sai
      desigual e uma foto de referência pode não entrar em template nenhum.

    Medido em n=60, k=5, teto=100 (C=5.461.512): BALANCED_COVERAGE dá cobertura
    por face entre 8 e 9; UNIFORM, entre 4 e 13. Em n=10, k=3, teto=8, UNIFORM
    deixou uma face fora de todos os templates.
    """

    BALANCED_COVERAGE = "balanced_coverage"
    UNIFORM = "uniform"

    @property
    def label(self) -> str:
        return {
            "balanced_coverage": "Cobertura balanceada",
            "uniform":           "Sorteio uniforme",
        }.get(self.value, self.value)

    @property
    def label_pt(self) -> str:
        return self.label

    @property
    def description(self) -> str:
        return {
            "balanced_coverage": (
                "Cada face participa de aproximadamente o mesmo número de "
                "templates, e nenhuma fica de fora."
            ),
            "uniform": (
                "Amostra não-enviesada do espaço de combinações. A cobertura "
                "por face sai desigual — uma foto pode não entrar em template "
                "nenhum."
            ),
        }.get(self.value, "")

    def __str__(self) -> str:
        return self.value

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> CombinationSamplingMode:
        """Reconstrói o enum a partir de string, com fallback para
        BALANCED_COVERAGE. Idempotente para membro já hidratado."""
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower() if value else "")
        except (ValueError, AttributeError):
            return cls.BALANCED_COVERAGE
