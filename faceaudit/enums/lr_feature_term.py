# faceaudit/enums/lr_feature_term.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class LRFeatureTerm(Enum):
    """
    Termo atômico da vetorização de features de calibração LR.

    Cada arquitetura em `LRCalibrationArchitecture` é declarada como uma
    sequência ordenada destes termos. O `ArchitectureFeatureBuilder` despacha
    cada termo para sua função de construção (`build_fn`) e sua derivada
    parcial em relação a s (`derivative_wrt_base_fn`) registradas em
    `LRFeatureTermSpec` (no `LR_FEATURE_TERM_SPECS`).

    Termos `*_RAW` consomem r_F/r_P sem centragem.
    Termos `*_CENTERED_MEAN` consomem r̃_F = r_F − r̄_F, r̃_P = r_P − r̄_P.

    Centragem é decidida em `LRArchitectureSpec.centering_mode`, que por sua
    vez seleciona qual variante de termo aparece na tupla `feature_terms`.
    """

    BASE_SCORE = (
        "BASE_SCORE",
        "Cosseno bruto s",
        "Score base — cosseno entre embeddings (ou PLDA).",
    )

    HINGE_POSITIVE = (
        "HINGE_POSITIVE",
        "h⁺(s) = max(s − s₀, 0)",
        "Hinge positiva: ativa apenas onde s > s₀.",
    )

    HINGE_NEGATIVE = (
        "HINGE_NEGATIVE",
        "h⁻(s) = max(s₀ − s, 0)",
        "Hinge negativa: ativa apenas onde s < s₀.",
    )

    HINGE_POSITIVE_X_RARITY_PROBE_RAW = (
        "HINGE_POSITIVE_X_RARITY_PROBE_RAW",
        "h⁺·r_F (não centrada)",
        "Interação h⁺(s)·r_F, sem centragem.",
    )

    HINGE_POSITIVE_X_RARITY_REFERENCE_RAW = (
        "HINGE_POSITIVE_X_RARITY_REFERENCE_RAW",
        "h⁺·r_P (não centrada)",
        "Interação h⁺(s)·r_P, sem centragem.",
    )

    HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN = (
        "HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN",
        "h⁺·(r_F − r̄_F)",
        "Interação h⁺(s)·r̃_F com centragem pela média de r_F em D_cal.",
    )

    HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN = (
        "HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN",
        "h⁺·(r_P − r̄_P)",
        "Interação h⁺(s)·r̃_P com centragem pela média de r_P em D_cal.",
    )

    HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN = (
        "HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN",
        "h⁻·(r_F − r̄_F)",
        "Interação h⁻(s)·r̃_F com centragem.",
    )

    HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN = (
        "HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN",
        "h⁻·(r_P − r̄_P)",
        "Interação h⁻(s)·r̃_P com centragem.",
    )

    LINEAR_X_RARITY_PROBE_RAW = (
        "LINEAR_X_RARITY_PROBE_RAW",
        "(s − s₀)·r_F",
        "Interação linear bilateral (s − s₀)·r_F sem centragem.",
    )

    LINEAR_X_RARITY_REFERENCE_RAW = (
        "LINEAR_X_RARITY_REFERENCE_RAW",
        "(s − s₀)·r_P",
        "Interação linear bilateral (s − s₀)·r_P sem centragem.",
    )

    LINEAR_X_RARITY_PROBE_CENTERED_MEAN = (
        "LINEAR_X_RARITY_PROBE_CENTERED_MEAN",
        "(s − s₀)·(r_F − r̄_F)",
        "Interação linear bilateral (s − s₀)·r̃_F com centragem.",
    )

    LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN = (
        "LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN",
        "(s − s₀)·(r_P − r̄_P)",
        "Interação linear bilateral (s − s₀)·r̃_P com centragem.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "LRFeatureTerm":
        if isinstance(value, cls):
            return value
        if not value:
            raise ValueError("LRFeatureTerm.from_str: valor vazio")
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"LRFeatureTerm desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @property
    def consumes_rarity_probe(self) -> bool:
        """True se o termo usa r_F (raridade da face probe/frontal)."""
        return "RARITY_PROBE" in self.value

    @property
    def consumes_rarity_reference(self) -> bool:
        """True se o termo usa r_P (raridade da face reference/profile)."""
        return "RARITY_REFERENCE" in self.value

    @property
    def consumes_hinge_positive(self) -> bool:
        """True se o termo combina h⁺ (e portanto requer s₀)."""
        return self in (
            LRFeatureTerm.HINGE_POSITIVE,
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW,
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_RAW,
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN,
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        )

    @property
    def consumes_hinge_negative(self) -> bool:
        """True se o termo combina h⁻ (e portanto requer s₀)."""
        return self in (
            LRFeatureTerm.HINGE_NEGATIVE,
            LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN,
            LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        )

    @property
    def consumes_linear_delta(self) -> bool:
        """True se o termo usa (s − s₀) sem hinge (e portanto requer s₀)."""
        return self in (
            LRFeatureTerm.LINEAR_X_RARITY_PROBE_RAW,
            LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_RAW,
            LRFeatureTerm.LINEAR_X_RARITY_PROBE_CENTERED_MEAN,
            LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN,
        )

    @property
    def requires_s0(self) -> bool:
        """True se o termo precisa de s₀ para ser calculado."""
        return (
            self.consumes_hinge_positive
            or self.consumes_hinge_negative
            or self.consumes_linear_delta
        )

    @property
    def is_centered(self) -> bool:
        """True se o termo aplica centragem r̃ = r − r̄."""
        return "CENTERED_MEAN" in self.value

    @property
    def has_kink_at_s0(self) -> bool:
        """True para hinges (descontinuidade da derivada em s = s₀)."""
        return self.consumes_hinge_positive or self.consumes_hinge_negative
