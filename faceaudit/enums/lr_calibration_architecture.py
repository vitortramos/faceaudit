# faceaudit/enums/lr_calibration_architecture.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class LRCalibrationArchitecture(Enum):
    """
    Arquiteturas de vetorização de features de calibração LR.

    Cada membro representa uma combinação declarada de termos atômicos
    (`LRFeatureTerm`) que serão entregues ao calibrador `LogisticRegression`
    como `score_vector_matrix`. A escolha da arquitetura substitui o
    encadeamento manual (~13 instruções) de `SCORE_INTERACTION_FEATURE` +
    `SCORE_VECTORIZATION` previsto na API low-level.

    A descrição matemática completa de cada arquitetura — termos, centragem,
    exigência de s₀, de tipicidade e de papéis ordenados nos pares — vive
    em `LRArchitectureSpec` no registry `LR_ARCHITECTURE_SPECS`. Este enum
    carrega apenas o identificador estável + rótulos de UI.
    """

    COSINE_ONLY = (
        "COSINE_ONLY",
        "Cosseno puro",
        "Baseline univariado [s]. Sem hinge, sem raridade. Útil como "
        "controle e referência mínima de comparação.",
    )

    HINGE_ONLY = (
        "HINGE_ONLY",
        "Hinge positivo [s, h⁺]",
        "Cosseno + hinge positivo h⁺(s) = max(s − s₀, 0). Controle "
        "estrutural simples sem raridade.",
    )

    TWO_SIDED_HINGE_ONLY = (
        "TWO_SIDED_HINGE_ONLY",
        "Hinge bilateral [s, h⁺, h⁻]",
        "Cosseno + ambas as hinges h⁺ e h⁻, ambas puramente derivadas "
        "de s (sem raridade). Controle estrutural da arquitetura "
        "bilateral cheia. Anteriormente chamada de BILATERAL_SIMPLE.",
    )

    ONE_SIDED_NON_CENTERED = (
        "ONE_SIDED_NON_CENTERED",
        "One-sided não-centrada [s, h⁺·r_F, h⁺·r_P]",
        "Raridade modula evidência positiva sem centragem. Coeficientes "
        "absorvem o efeito médio da raridade. Exige dois escores de "
        "tipicidade (probe + reference).",
    )

    ONE_SIDED_CENTERED = (
        "ONE_SIDED_CENTERED",
        "One-sided centrada [s, h⁺·r̃_F, h⁺·r̃_P]",
        "One-sided com centragem MEAN: r̃ = r − r̄. Inspirada no "
        "princípio hierárquico de Tukey (parcialmente). Coeficientes "
        "refletem modulação por desvio da raridade típica.",
    )

    LINEAR_NON_CENTERED = (
        "LINEAR_NON_CENTERED",
        "Linear não-centrada [s, (s−s₀)·r_F, (s−s₀)·r_P]",
        "Raridade modula em todo o range de s, bilateral em torno de "
        "s₀ sem hinge. Sem centragem.",
    )

    LINEAR_CENTERED = (
        "LINEAR_CENTERED",
        "Linear centrada [s, (s−s₀)·r̃_F, (s−s₀)·r̃_P]",
        "Linear com centragem MEAN. Coeficientes refletem modulação "
        "por desvio da raridade típica em todo o range.",
    )

    HIERARCHICAL_CENTERED = (
        "HIERARCHICAL_CENTERED",
        "Hierárquica centrada [s, h⁺, h⁺·r̃_F, h⁺·r̃_P]",
        "Inclui h⁺ como feature explícita + interações centradas. "
        "Estrutura inspirada em Tukey: efeito principal da hinge "
        "estimado independentemente do efeito de modulação por "
        "raridade. Mais parcimoniosa que a bilateral cheia.",
    )

    BILATERAL_FULL_CENTERED = (
        "BILATERAL_FULL_CENTERED",
        "Bilateral cheia centrada (D=7)",
        "Arquitetura mais expressiva: [s, h⁺, h⁻, h⁺·r̃_F, h⁺·r̃_P, "
        "h⁻·r̃_F, h⁻·r̃_P]. Raridade modula em ambos os lados de s₀. "
        "Centragem MEAN aplicada às interações.",
    )

    # ─── Família ONE_FACE ────────────────────────────────────────────────
    # Variantes das 6 arquiteturas com tipicidade que usam apenas UMA face
    # (em vez de probe+reference). Úteis em casework forense onde só uma
    # face é questionada, ou quando a tipicidade de um dos lados é
    # considerada ruidosa demais (e.g. M_P em cross-pose com base escassa).
    ONE_SIDED_NON_CENTERED_ONE_FACE = (
        "ONE_SIDED_NON_CENTERED_ONE_FACE",
        "One-sided não-centrada one-face [s, h⁺·r]",
        "Variante one-face de ONE_SIDED_NON_CENTERED. Raridade da face "
        "única modula evidência positiva sem centragem. Usa apenas r "
        "(uma face), não r_F e r_P.",
    )
    ONE_SIDED_CENTERED_ONE_FACE = (
        "ONE_SIDED_CENTERED_ONE_FACE",
        "One-sided centrada one-face [s, h⁺·r̃]",
        "Variante one-face de ONE_SIDED_CENTERED. Centra raridade por "
        "r̃ = r − r̄. Apenas uma face contribui com tipicidade.",
    )
    LINEAR_NON_CENTERED_ONE_FACE = (
        "LINEAR_NON_CENTERED_ONE_FACE",
        "Linear não-centrada one-face [s, (s−s₀)·r]",
        "Variante one-face de LINEAR_NON_CENTERED. Raridade da face "
        "única modula em todo o range de s, sem hinge.",
    )
    LINEAR_CENTERED_ONE_FACE = (
        "LINEAR_CENTERED_ONE_FACE",
        "Linear centrada one-face [s, (s−s₀)·r̃]",
        "Variante one-face de LINEAR_CENTERED. Linear com centragem MEAN, "
        "apenas uma face.",
    )
    HIERARCHICAL_CENTERED_ONE_FACE = (
        "HIERARCHICAL_CENTERED_ONE_FACE",
        "Hierárquica centrada one-face [s, h⁺, h⁺·r̃]",
        "Variante one-face de HIERARCHICAL_CENTERED. Inclui h⁺ como "
        "feature explícita + 1 interação centrada (face única).",
    )
    BILATERAL_FULL_CENTERED_ONE_FACE = (
        "BILATERAL_FULL_CENTERED_ONE_FACE",
        "Bilateral cheia centrada one-face (D=5)",
        "Variante one-face de BILATERAL_FULL_CENTERED: [s, h⁺, h⁻, "
        "h⁺·r̃, h⁻·r̃]. Raridade da face única modula ambos os lados de s₀.",
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
    def from_str(cls, value: str) -> "LRCalibrationArchitecture":
        if not value:
            return cls.COSINE_ONLY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        # Backward-compat: BILATERAL_SIMPLE foi renomeada para TWO_SIDED_HINGE_ONLY
        if norm == "BILATERAL_SIMPLE":
            return cls.TWO_SIDED_HINGE_ONLY
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.COSINE_ONLY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
