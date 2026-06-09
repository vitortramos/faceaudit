# faceaudit/models/typicality/movmf_typicality_model_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Type, TypeVar

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_percentile_basis import TypicalityPercentileBasis
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="MoVMFTypicalityModelConfig")

CenteringStr = Literal["none", "frechet", "euclidean_legacy"]


@dataclass
class MoVMFTypicalityModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do `MoVMFTypicalityModel` na instrução unificada
    `TYPICALITY_MODEL_TRAINING`.

    **Backend EXPERIMENTAL**: PR 4 endurece a implementação (validação de
    norma L2, balanceamento por identidade, cross-fit OOD, canonicalização
    de componentes). A UI exibe badge `EXPERIMENTAL` ao lado do dropdown
    de tipo até a validação científica ser concluída
    (ver `docs/typicality_models.md`).
    """

    n_components: int = field(
        default=1,
        metadata={"help": (
            "<html><b>Número de componentes da mistura</b> (K). "
            "Hiperparâmetro científico — a escolha deve ser justificada "
            "externamente, não ajustada sobre o caso examinado. PR 4 "
            "endurece com diagnóstico de componentes degenerados; uma "
            "iteração futura poderá ofererecer seleção automática por "
            "BIC / held-out por identidade.</html>"
        )},
    )

    n_init: int = field(
        default=1,
        metadata={"help": (
            "Número de inicializações aleatórias do EM. Mantém a melhor "
            "log-likelihood entre tentativas. Relevante apenas quando "
            "<code>n_components &gt; 1</code>."
        )},
    )

    centering: CenteringStr = field(
        default="none",
        metadata={"help": (
            "<html><b>Centralização pré-fit</b>.<br/>"
            "<b>none</b> (default): nenhuma centralização. Preserva a "
            "métrica esférica original.<br/>"
            "<b>frechet</b>: aplica reflexão de Householder alinhando "
            "a média de Fréchet ao polo norte. Transformação <i>ortogonal</i> "
            "— preserva cossenos, distâncias angulares e likelihood "
            "da mistura vMF com parâmetros livres. <b>Não mitiga hubness</b> "
            "(hubness é função de proximidade angular, invariante à "
            "orientação). Útil como canonicalização de orientação para "
            "diagnóstico/reprodutibilidade.<br/>"
            "<b>euclidean_legacy</b>: média euclidiana renormalizada. "
            "Não preserva a métrica esférica — disponível por "
            "retrocompatibilidade, emite warning.</html>"
        )},
    )

    convergence_patience: int = field(
        default=3,
        metadata={"help": "Iterações consecutivas abaixo de `tol` para convergir."},
    )

    max_iter: int = field(
        default=200,
        metadata={"help": "Iterações máximas do EM por inicialização."},
    )

    tol: float = field(
        default=1e-5,
        metadata={"help": "Tolerância relativa de log-likelihood para convergência."},
    )

    random_state: int = field(
        default=42,
        metadata={"help": "Semente base para reprodutibilidade."},
    )

    min_component_weight: float = field(
        default=1e-6,
        metadata={"help": (
            "Peso mínimo de componente antes de descartar como degenerado."
        )},
    )

    min_kappa: float = field(
        default=1e-6,
        metadata={"help": "Limite inferior do parâmetro κ por componente."},
    )

    max_kappa: float = field(
        default=1e4,
        metadata={"help": "Limite superior do parâmetro κ por componente."},
    )

    ood_quantile: float = field(
        default=0.001,
        metadata={"help": (
            "Quantil de log-likelihood usado como threshold OOD. "
            "0 desativa OOD detection."
        )},
    )

    population_weighting_mode: PopulationWeightingMode = field(
        default=PopulationWeightingMode.IDENTITY_CENTROIDS,
        metadata={"help": (
            "Como ponderar a população de referência (PR 4 endurece "
            "INVERSE_IDENTITY_FREQUENCY)."
        )},
    )

    unit_norm_policy: UnitNormPolicy = field(
        default=UnitNormPolicy.ERROR,
        metadata={"help": (
            "Política de norma L2 dos embeddings (default ERROR — "
            "rastreabilidade pericial)."
        )},
    )

    unit_norm_tolerance: float = field(
        default=1e-4,
        metadata={"help": "Tolerância da política UnitNormPolicy."},
    )

    percentile_basis_score_type: TypicalityPercentileBasis = field(
        default=TypicalityPercentileBasis.LOG_DENSITY_ESTIMATE,
        metadata={"help": (
            "Base do score TYPICALITY_PERCENTILE. MoVMF default: "
            "<code>LOG_DENSITY_ESTIMATE</code> (log-likelihood da "
            "mistura)."
        )},
    )

    reference_score_calibration_mode: ReferenceScoreCalibrationMode = field(
        default=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        metadata={"help": (
            "<html><b>CROSS_FIT_IDENTITY</b> recomendado (PR 4 "
            "implementa). <b>IN_SAMPLE_LEGACY</b> reproduz o "
            "comportamento de oráculo histórico para auditoria.</html>"
        )},
    )

    cv_n_folds: int = field(
        default=5,
        metadata={"help": (
            "Número de folds para CROSS_FIT_IDENTITY. Ignorado em "
            "IN_SAMPLE_LEGACY."
        )},
    )

    population_label: Optional[str] = field(
        default=None,
        metadata={"help": "Rótulo opcional da população (entra no fingerprint)."},
    )

    embedding_column: Optional[str] = field(
        default=None,
        metadata={"help": "Override opcional da coluna de embedding."},
    )

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError("n_components deve ser >= 1.")
        if self.n_init < 1:
            raise ValueError("n_init deve ser >= 1.")
        if self.centering not in ("none", "frechet", "euclidean_legacy"):
            raise ValueError(
                f"centering inválido: {self.centering!r}."
            )
        if self.convergence_patience < 1:
            raise ValueError("convergence_patience deve ser >= 1.")
        if self.max_iter < 1:
            raise ValueError("max_iter deve ser >= 1.")
        if not (0.0 <= self.ood_quantile < 0.5):
            raise ValueError("ood_quantile deve estar em [0, 0.5).")
        if self.unit_norm_tolerance <= 0.0:
            raise ValueError("unit_norm_tolerance deve ser > 0.")
        if self.cv_n_folds < 0:
            raise ValueError("cv_n_folds deve ser >= 0.")
        if (
            self.reference_score_calibration_mode
            == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
            and self.cv_n_folds < 2
        ):
            raise ValueError(
                "cv_n_folds deve ser >= 2 quando "
                "reference_score_calibration_mode = CROSS_FIT_IDENTITY."
            )

    # ─────────────────────── visibilidade UI ────────────────────────

    REACTIVE_FIELDS = ("n_components", "reference_score_calibration_mode")

    def get_hidden_fields(self):
        hidden = set()
        if self.n_components <= 1:
            # n_init só relevante para mistura > 1 componente
            hidden.add("n_init")
        if (
            self.reference_score_calibration_mode
            != ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
        ):
            hidden.add("cv_n_folds")
        return hidden

    # ─────────────────────── deserialização ─────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        d = dict(data)
        if "population_weighting_mode" in d and d["population_weighting_mode"]:
            d["population_weighting_mode"] = PopulationWeightingMode.from_str(
                d["population_weighting_mode"]
            )
        if "unit_norm_policy" in d and d["unit_norm_policy"]:
            d["unit_norm_policy"] = UnitNormPolicy.from_str(d["unit_norm_policy"])
        if "percentile_basis_score_type" in d and d["percentile_basis_score_type"]:
            d["percentile_basis_score_type"] = TypicalityPercentileBasis.from_str(
                d["percentile_basis_score_type"]
            )
        if (
            "reference_score_calibration_mode" in d
            and d["reference_score_calibration_mode"]
        ):
            d["reference_score_calibration_mode"] = ReferenceScoreCalibrationMode.from_str(
                d["reference_score_calibration_mode"]
            )
        return cls(**d)
