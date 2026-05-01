# faceaudit/enums/plda_variant.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class PLDAVariant(Enum):
    """
    Define a variante matemática do PLDA usada pelo `PLDASimilarityModel`.

    - SIMPLE (padrão): formulação simplificada baseada em SpeechBrain (Ioffe-style),
      com subspace de baixo rank `F` entre-classes e covariância residual `Σ`.
      Suporta os modos FULL_RANK e LOW_RANK via `latent_dim`.

    - TWO_COVARIANCE: formulação de duas covariâncias completas (Brümmer & de
      Villiers 2010), x = y + ε com y ~ N(μ, B), ε ~ N(0, W). Treino é em
      forma fechada (sem EM iterativo), scoring por log-LR direto. `latent_dim`
      e `plda_mode` são ignorados — B e W são sempre full-rank D×D.
      Recomendada na literatura forense (FoCal toolkit) por estabilidade
      numérica e calibração natural.
    """

    SIMPLE = (
        "SIMPLE",
        "Simples (SpeechBrain)",
        "Formulação simplificada com subspace F entre-classes (rank configurável) "
        "e covariância residual Σ. Treinada via EM. Suporta FULL_RANK e LOW_RANK.",
    )

    TWO_COVARIANCE = (
        "TWO_COVARIANCE",
        "Duas Covariâncias (Brümmer)",
        "Modelo de duas covariâncias completas (B entre-classes, W intra-classe). "
        "Treino em forma fechada — sem EM iterativo. Não usa `latent_dim` nem "
        "`plda_mode`. Padrão na literatura forense (Brümmer & de Villiers 2010).",
    )

    def __new__(cls, value: str, label: str, description: str) -> "PLDAVariant":
        # Definir _value_ em __new__ é crítico para que o enum seja serializável
        # via pickle e reconstruído por value=str (mesmo padrão de PLDAMode).
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "PLDAVariant":
        if not value:
            return cls.SIMPLE

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for m in cls:
                if m.value == val_norm:
                    return m
            return cls.SIMPLE

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(m.value, m.label) for m in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(m.value, m.label, m.description) for m in cls]
