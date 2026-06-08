# faceaudit/models/typicality/knn_typicality_model_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_percentile_basis import TypicalityPercentileBasis
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="KNNTypicalityModelConfig")


@dataclass
class KNNTypicalityModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do `KNNTypicalityModel` na instrução unificada
    `TYPICALITY_MODEL_TRAINING`.

    Substitui os campos que viviam dentro de `KNNTypicalityModelTrainingInstruction`
    (legado, deletado em PR 2).
    """

    k: int = field(
        default=50,
        metadata={"help": (
            "Número de vizinhos do k-NN esférico. Heurística defensável: "
            "<code>k ≈ √N</code>. Valores muito pequenos amplificam ruído "
            "local; muito grandes diluem o sinal local."
        )},
    )

    ood_quantile: float = field(
        default=0.01,
        metadata={"help": (
            "Quantil de T_k usado como corte de fora-de-distribuição. "
            "0 desativa OOD detection."
        )},
    )

    population_weighting_mode: PopulationWeightingMode = field(
        default=PopulationWeightingMode.IDENTITY_CENTROIDS,
        metadata={"help": (
            "<html><b>Como ponderar a referência populacional.</b><br/>"
            "<b>IDENTITY_CENTROIDS</b> (default forense): uma observação "
            "por pessoa (centroide L2-normalizado). Robusto a "
            "desbalanceamento de coleta.<br/>"
            "<b>FACE_LEVEL_RAW</b>: cada foto é uma observação. "
            "Identidades com mais fotos pesam mais.<br/>"
            "<b>INVERSE_IDENTITY_FREQUENCY</b>: cada foto entra com peso "
            "1/n_faces(identidade). PR 4 implementa.</html>"
        )},
    )

    unit_norm_policy: UnitNormPolicy = field(
        default=UnitNormPolicy.ERROR,
        metadata={"help": (
            "<html><b>Política de norma L2 dos embeddings de entrada.</b><br/>"
            "<b>ERROR</b> (default forense): rejeita embeddings com "
            "<code>|‖x‖ − 1| > unit_norm_tolerance</code>.<br/>"
            "<b>NORMALIZE_WITH_WARNING</b>: normaliza com warning.<br/>"
            "<b>NORMALIZE_SILENTLY</b>: normaliza silenciosamente "
            "(comportamento histórico; usado pela migração V2.14).</html>"
        )},
    )

    unit_norm_tolerance: float = field(
        default=1e-4,
        metadata={"help": (
            "Tolerância da política UnitNormPolicy. Embeddings com "
            "<code>|‖x‖ − 1| > tolerance</code> disparam a política "
            "configurada."
        )},
    )

    percentile_basis_score_type: TypicalityPercentileBasis = field(
        default=TypicalityPercentileBasis.MEAN_KNN_SIMILARITY,
        metadata={"help": (
            "<html><b>Base do score TYPICALITY_PERCENTILE.</b><br/>"
            "<b>MEAN_KNN_SIMILARITY</b> (default histórico): rank "
            "empírico de T_k.<br/>"
            "<b>LOG_DENSITY_ESTIMATE</b>: rank empírico da log-densidade "
            "estimada (L-Q esférica).</html>"
        )},
    )

    reference_score_calibration_mode: ReferenceScoreCalibrationMode = field(
        default=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
        metadata={"help": (
            "<html><b>Como construir a CDF de referência e o threshold "
            "OOD.</b><br/>"
            "<b>CROSS_FIT_IDENTITY</b> (recomendado, PR 4): K-fold por "
            "identidade; CDF construída com scores OOF; refit final em "
            "toda a população.<br/>"
            "<b>IN_SAMPLE_LEGACY</b> (default migratório): LOO de imagem; "
            "comportamento histórico do k-NN.</html>"
        )},
    )

    cv_n_folds: int = field(
        default=0,
        metadata={"help": (
            "Número de folds para CROSS_FIT_IDENTITY. Ignorado em "
            "IN_SAMPLE_LEGACY. Default 0 (sentinel: serviço escolhe "
            "5 quando ativo)."
        )},
    )

    population_label: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Rótulo opcional da população de referência (ex.: "
            "'frontal_brasil_2025'). Persistido nos fingerprints."
        )},
    )

    embedding_column: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Override opcional da coluna de embedding. Quando None, usa "
            "a coluna definida pela estratégia."
        )},
    )

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("KNNTypicalityModelConfig.k deve ser >= 1.")
        if not (0.0 <= self.ood_quantile < 0.5):
            raise ValueError(
                "KNNTypicalityModelConfig.ood_quantile deve estar em [0, 0.5)."
            )
        if self.unit_norm_tolerance <= 0.0:
            raise ValueError(
                "KNNTypicalityModelConfig.unit_norm_tolerance deve ser > 0."
            )
        if self.cv_n_folds < 0:
            raise ValueError(
                "KNNTypicalityModelConfig.cv_n_folds deve ser >= 0."
            )
        if (
            self.reference_score_calibration_mode
            == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
            and self.cv_n_folds == 0
        ):
            # Sentinel: 5 folds default quando CROSS_FIT ativo.
            object.__setattr__(self, "cv_n_folds", 5)

    # ─────────────────────── visibilidade UI ────────────────────────

    REACTIVE_FIELDS = ("reference_score_calibration_mode",)

    def get_hidden_fields(self):
        """`cv_n_folds` só faz sentido em CROSS_FIT_IDENTITY."""
        hidden = set()
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
