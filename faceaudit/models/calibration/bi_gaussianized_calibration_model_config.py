# faceaudit/models/calibration/bi_gaussianized_calibration_model_config.py

"""DTO de configuração do calibrador Bi-Gaussianized (Morrison 2024).

Mapeia scores para um sistema bi-gaussiano perfeitamente calibrado com σ²
alvo derivado de Cllr ou EER do treino. Quatro decisões:

  - `target_metric`     — Cllr (numérico, ~10ms) ou EER (analítico, exato).
  - `initial_calibrator`— calibrador inicial usado pra derivar Cllr/EER.
                          NONE pra scores já em espaço de ln(Λ) (ex: PLDA);
                          LOGREG quando rápido; KDE quando há outliers
                          (variante recomendada pelo paper §3.6).
  - `inversion_method`  — interp1d (rápido) ou Brent root-finding (preciso)
                          para inverter a CDF da mistura de duas gaussianas.
  - `sigma_shrinkage_factor` — encolhimento opcional do σ alvo (§3.5),
                               puxa LRs grandes pra perto de 1.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="BiGaussianizedCalibrationModelConfig")


class BiGaussTargetMetric(Enum):
    """Métrica usada pra derivar σ² do sistema bi-gaussiano alvo."""
    CLLR = "CLLR"   # via fit ln(Cllr) = b·(exp(-c·σ²) - 1) → Eq. (8)
    EER  = "EER"    # σ = -2·Φ⁻¹(EER) → Eq. (10), analítico

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussTargetMetric":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.CLLR
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.CLLR


class BiGaussInitialCalibrator(Enum):
    """Calibrador inicial aplicado antes do passo de bi-gaussianização."""
    NONE   = "NONE"     # scores já em espaço de ln(Λ) (ex: PLDA log-LR)
    LOGREG = "LOGREG"   # rápido; assume linearidade no espaço dos scores
    KDE    = "KDE"      # mais robusto a outliers; §3.6 do paper recomenda

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussInitialCalibrator":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.NONE
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.NONE


class BiGaussInversionMethod(Enum):
    """Método pra inverter a CDF da mistura 0.5·N(+σ²/2,σ²) + 0.5·N(-σ²/2,σ²)."""
    INTERP1D    = "INTERP1D"     # interpolação linear sobre grid pré-computado
    BRENT_ROOT  = "BRENT_ROOT"   # scipy.optimize.brentq por ponto; preciso, lento

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value) -> "BiGaussInversionMethod":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.INTERP1D
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.INTERP1D


@dataclass
class BiGaussianizedCalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração Bi-Gaussianized.

    Bi-Gaussianized Calibration (Morrison 2024, Law, Probability and Risk):
    score → ln(Λ) via mapeamento monotônico não-linear pra distribuições alvo
    `Hp ~ N(+σ²/2, σ²)` e `Hd ~ N(-σ²/2, σ²)`, com σ² calculado de modo que
    o Cllr (ou EER) do sistema alvo bata com o do treino.
    """

    target_metric: BiGaussTargetMetric = field(
        default=BiGaussTargetMetric.CLLR,
        metadata={
            "help": (
                "<html><b>Métrica usada pra derivar σ²</b> do sistema bi-gaussiano alvo."
                "<ul>"
                "<li><b>CLLR</b> — numérico via fit ln(Cllr) = b·(exp(-c·σ²)-1). "
                "Usa toda a informação dos scores, recomendado quando há ≥ 50 fontes.</li>"
                "<li><b>EER</b> — analítico σ = -2·Φ⁻¹(EER). Mais ruidoso com poucas "
                "same-source samples (discretiza em passos 1/N).</li>"
                "</ul></html>"
            )
        },
    )

    initial_calibrator: BiGaussInitialCalibrator = field(
        default=BiGaussInitialCalibrator.NONE,
        metadata={
            "help": (
                "<html><b>Calibrador inicial</b> aplicado antes da bi-gaussianização."
                "<ul>"
                "<li><b>NONE</b> — scores já vivem em espaço de ln(Λ) (caso PLDA "
                "log-LR). Cllr/EER calculados direto dos scores crus.</li>"
                "<li><b>LOGREG</b> — calibração linear inicial. Rápida.</li>"
                "<li><b>KDE</b> — calibração inicial via KDE. Menos outliers, ~15× "
                "mais lento que LogReg. Recomendado pelo paper §3.6.</li>"
                "</ul>"
                "Só afeta o passo de derivação do σ² alvo; o mapeamento final "
                "não-linear sempre usa a CDF empírica."
                "</html>"
            )
        },
    )

    inversion_method: BiGaussInversionMethod = field(
        default=BiGaussInversionMethod.INTERP1D,
        metadata={
            "help": (
                "<html><b>Método pra inverter a CDF da mistura alvo.</b>"
                "<ul>"
                "<li><b>INTERP1D</b> — grid pré-computado + interpolação linear. "
                "O(N) total. Preciso o bastante para uso prático.</li>"
                "<li><b>BRENT_ROOT</b> — root-finding por ponto via Brent. "
                "Mais preciso nas caudas, ~50-100× mais lento.</li>"
                "</ul></html>"
            )
        },
    )

    sigma_shrinkage_factor: float = field(
        default=1.0,
        metadata={
            "help": (
                "Fator multiplicativo aplicado ao σ alvo (§3.5 do paper). "
                "1.0 = sem encolhimento; 0.9 = 90% do σ calculado, "
                "puxa LRs grandes pra perto de 1 (útil quando há pouco treino "
                "e preocupa-se com extrapolação)."
            )
        },
    )

    min_log_lr: float = field(
        default=-20.0,
        metadata={"help": "Limite inferior do log-LR calibrado (clipping)."}
    )

    max_log_lr: float = field(
        default=20.0,
        metadata={"help": "Limite superior do log-LR calibrado (clipping)."}
    )

    interpolation_points: int = field(
        default=10000,
        metadata={
            "help": (
                "Quantidade de pontos do grid usado pra construir G(x) (CDF "
                "empírica) e F⁻¹ (inversa da mistura). Só aplica se "
                "inversion_method=INTERP1D."
            )
        },
    )

    brent_xtol: float = field(
        default=1e-6,
        metadata={"help": "Tolerância absoluta do Brent (só aplica se BRENT_ROOT)."}
    )

    brent_search_range: float = field(
        default=50.0,
        metadata={
            "help": (
                "Range simétrico [-R, +R] em ln(Λ) usado como bracket inicial "
                "do Brent. Tem que conter as caudas; default 50 cobre LR de até e^50."
            )
        },
    )

    initial_logreg_C: float = field(
        default=1.0,
        metadata={"help": "C da LogReg inicial (só aplica se initial_calibrator=LOGREG)."}
    )

    initial_kde_bandwidth: float = field(
        default=0.0,
        metadata={
            "help": (
                "Bandwidth do KDE inicial. 0.0 = automático via Silverman "
                "(padrão do paper). Só aplica se initial_calibrator=KDE."
            )
        },
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente para amostragens internas."}
    )

    def __post_init__(self) -> None:
        # Normaliza enums vindos de string (caso from_dict externo)
        if not isinstance(self.target_metric, BiGaussTargetMetric):
            self.target_metric = BiGaussTargetMetric.from_str(self.target_metric)
        if not isinstance(self.initial_calibrator, BiGaussInitialCalibrator):
            self.initial_calibrator = BiGaussInitialCalibrator.from_str(self.initial_calibrator)
        if not isinstance(self.inversion_method, BiGaussInversionMethod):
            self.inversion_method = BiGaussInversionMethod.from_str(self.inversion_method)

        if not (0.0 < self.sigma_shrinkage_factor <= 1.0):
            raise ValueError("sigma_shrinkage_factor deve estar em (0, 1].")
        if self.min_log_lr >= self.max_log_lr:
            raise ValueError("min_log_lr deve ser menor que max_log_lr.")
        if self.interpolation_points < 100:
            raise ValueError("interpolation_points deve ser pelo menos 100.")
        if self.brent_xtol <= 0:
            raise ValueError("brent_xtol deve ser positivo.")
        if self.brent_search_range <= 0:
            raise ValueError("brent_search_range deve ser positivo.")
        if self.initial_logreg_C <= 0:
            raise ValueError("initial_logreg_C deve ser positivo.")
        if self.initial_kde_bandwidth < 0:
            raise ValueError("initial_kde_bandwidth não pode ser negativo.")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        payload = dict(data)
        if "target_metric" in payload:
            payload["target_metric"] = BiGaussTargetMetric.from_str(
                payload["target_metric"]
            )
        if "initial_calibrator" in payload:
            payload["initial_calibrator"] = BiGaussInitialCalibrator.from_str(
                payload["initial_calibrator"]
            )
        if "inversion_method" in payload:
            payload["inversion_method"] = BiGaussInversionMethod.from_str(
                payload["inversion_method"]
            )
        return cls(**payload)
