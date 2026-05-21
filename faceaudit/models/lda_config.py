from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.lda_estimator_type import LdaEstimatorType
from faceaudit.enums.lda_grid_search_metric import LdaGridSearchMetric

T = TypeVar("T", bound="LDAConfig")


@dataclass
class LDAConfig(JSONSerializableMixin):
    """
    Configuração para a etapa de Linear Discriminant Analysis (LDA).

    Suporta três opções de regularização (combináveis):
      - Ridge clássico (`regularization`).
      - Shrinkage explícito (`shrinkage` α: Sw ← (1-α)·Sw + α·I).
      - Estimadores robustos (`estimator`: STANDARD / LEDOIT_WOLF / OAS).

    Opcionalmente, busca o melhor ε via holdout interno por identidade
    quando `use_grid_search=True`.
    """

    n_components: Optional[int] = field(
        default=None,
        metadata={"help": "Número de eixos LDA a manter. None = max possível "
                          "(n_classes - 1, limite teórico do Fisher LDA)."}
    )
    regularization: float = field(
        default=1e-4,
        metadata={"help": "Termo ridge ε·I somado a Sw antes da inversão. "
                          "Ignorado quando use_grid_search=True (usa reg_grid)."}
    )
    shrinkage: float = field(
        default=0.0,
        metadata={"help": "Shrinkage explícito α ∈ [0,1]: Sw ← (1-α)·Sw + α·I. "
                          "0 = desligado; 1 = identidade. Combinável com qualquer estimator."}
    )
    estimator: LdaEstimatorType = field(
        default=LdaEstimatorType.STANDARD,
        metadata={"help": "Estimador de Sw: STANDARD (soma de outer products), "
                          "LEDOIT_WOLF / OAS (sklearn.covariance com shrinkage automático)."}
    )
    use_grid_search: bool = field(
        default=False,
        metadata={"help": "Quando True, busca o melhor ε via holdout interno por "
                          "identidade. Ignora `regularization` (usa `reg_grid`)."}
    )
    reg_grid: List[float] = field(
        default_factory=lambda: [1e-6, 1e-4, 1e-2, 1e-1, 1.0],
        metadata={"help": "Grade de valores de ε avaliados quando use_grid_search=True."}
    )
    grid_search_holdout_ratio: float = field(
        default=0.2,
        metadata={"help": "Fração de identidades reservadas para validação no grid "
                          "search. Default 0.2 → 80/20."}
    )
    grid_search_metric: LdaGridSearchMetric = field(
        default=LdaGridSearchMetric.FISHER_RATIO,
        metadata={"help": "Métrica de seleção: FISHER_RATIO (trace(Sb_proj)/trace(Sw_proj)) "
                          "ou COSINE_AUC (AUC entre pares genuíno/impostor projetados)."}
    )

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization deve ser > 0")
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage deve estar em [0, 1]")
        if self.use_grid_search:
            if not self.reg_grid or any(r <= 0 for r in self.reg_grid):
                raise ValueError(
                    "reg_grid não pode ser vazio e todos os valores devem ser > 0 "
                    "quando use_grid_search=True."
                )
            if not 0.05 <= self.grid_search_holdout_ratio <= 0.5:
                raise ValueError(
                    "grid_search_holdout_ratio deve estar em [0.05, 0.5]."
                )

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()

        processed = data.copy()

        if "estimator" in processed and processed["estimator"]:
            est = processed["estimator"]
            if isinstance(est, str):
                processed["estimator"] = LdaEstimatorType.from_str(est)

        if "grid_search_metric" in processed and processed["grid_search_metric"]:
            metric = processed["grid_search_metric"]
            if isinstance(metric, str):
                processed["grid_search_metric"] = LdaGridSearchMetric.from_str(metric)

        if "reg_grid" in processed and processed["reg_grid"] is not None:
            raw = processed["reg_grid"]
            if isinstance(raw, str):
                processed["reg_grid"] = [
                    float(v.strip()) for v in raw.split(",") if v.strip()
                ]
            else:
                processed["reg_grid"] = [float(v) for v in raw]

        return cls(**processed)
