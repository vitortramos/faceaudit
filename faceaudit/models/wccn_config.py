from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.wccn_estimator_type import WccnEstimatorType
from faceaudit.enums.wccn_grid_search_metric import WccnGridSearchMetric

T = TypeVar("T", bound="WCCNConfig")


@dataclass
class WCCNConfig(JSONSerializableMixin):
    """
    Configuração da etapa de Within-Class Covariance Normalization (WCCN).

    Suporta três opções de regularização (combináveis):
      - Ridge clássico (`regularization`).
      - Shrinkage explícito (`shrinkage` α: W ← (1-α)·W + α·I).
      - Estimadores robustos (`estimator`: STANDARD / LEDOIT_WOLF / OAS).

    Opcionalmente, busca o melhor ε via holdout interno por identidade
    quando `use_grid_search=True`.
    """

    regularization: float = field(
        default=1e-6,
        metadata={"help": "Termo ridge ε·I somado a W antes da inversão. "
                          "Ignorado quando use_grid_search=True (use reg_grid)."}
    )
    shrinkage: float = field(
        default=0.0,
        metadata={"help": "Shrinkage explícito α ∈ [0,1]: W ← (1-α)·W + α·I. "
                          "0 = desligado; 1 = identidade. Combinável com ridge."}
    )
    estimator: WccnEstimatorType = field(
        default=WccnEstimatorType.STANDARD,
        metadata={"help": "Estimador de W: STANDARD (média de outer products), "
                          "LEDOIT_WOLF / OAS (sklearn.covariance com shrinkage automático)."}
    )
    use_grid_search: bool = field(
        default=False,
        metadata={"help": "Quando True, busca o melhor ε via holdout interno por "
                          "identidade. Ignora `regularization` (usa `reg_grid`)."}
    )
    reg_grid: List[float] = field(
        default_factory=lambda: [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0],
        metadata={"help": "Grade de valores de ε avaliados quando use_grid_search=True. "
                          "Usada apenas se use_grid_search=True."}
    )
    grid_search_holdout_ratio: float = field(
        default=0.2,
        metadata={"help": "Fração de identidades reservadas para validação no grid "
                          "search. Default 0.2 → 80/20."}
    )
    grid_search_metric: WccnGridSearchMetric = field(
        default=WccnGridSearchMetric.FISHER_RATIO,
        metadata={"help": "Métrica de seleção: FISHER_RATIO (trace(Sb)/trace(Sw)) "
                          "ou COSINE_AUC (AUC entre pares genuíno/impostor do holdout)."}
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
                processed["estimator"] = WccnEstimatorType.from_str(est)

        if "grid_search_metric" in processed and processed["grid_search_metric"]:
            metric = processed["grid_search_metric"]
            if isinstance(metric, str):
                processed["grid_search_metric"] = WccnGridSearchMetric.from_str(metric)

        if "reg_grid" in processed and processed["reg_grid"] is not None:
            raw = processed["reg_grid"]
            if isinstance(raw, str):
                processed["reg_grid"] = [
                    float(v.strip()) for v in raw.split(",") if v.strip()
                ]
            else:
                processed["reg_grid"] = [float(v) for v in raw]

        return cls(**processed)
