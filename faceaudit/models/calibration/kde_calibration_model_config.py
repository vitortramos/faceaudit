# faceaudit/models/calibration/kde_calibration_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="KDECalibrationModelConfig")


@dataclass
class KDECalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração KDE.

    Este calibrador modela separadamente:
    - a densidade dos scores genuínos
    - a densidade dos scores impostores

    e transforma score bruto em LR/log-LR calibrado.

    A configuração aqui é propositalmente enxuta e focada no calibrador em si,
    sem misturar:
    - mineração de pares
    - comparação entre embeddings
    - estratégia de similaridade
    """

    bandwidth_genuine: float = field(
        default=0.1,
        metadata={"help": "Bandwidth do KDE para scores genuínos."}
    )

    bandwidth_impostor: float = field(
        default=0.1,
        metadata={"help": "Bandwidth do KDE para scores impostores."}
    )

    use_gridsearch: bool = field(
        default=True,
        metadata={"help": "Indica se o calibrador deve otimizar bandwidth via GridSearch."}
    )

    max_samples_for_gridsearch: int = field(
        default=10000,
        metadata={"help": "Máximo de amostras usadas no GridSearch do bandwidth."}
    )

    max_stored_samples: int = field(
        default=50000,
        metadata={"help": "Máximo de amostras armazenadas por classe para ajuste final do KDE."}
    )

    fisher_eps: float = field(
        default=1e-6,
        metadata={"help": "Epsilon numérico para clipping antes da transformação de Fisher."}
    )

    min_bandwidth_abs: float = field(
        default=1e-3,
        metadata={"help": "Bandwidth mínimo absoluto no espaço transformado."}
    )

    max_grid_expansions: int = field(
        default=2,
        metadata={"help": "Número máximo de expansões do grid se o melhor bandwidth cair na borda."}
    )

    min_log_lr: float = field(
        default=-20.0,
        metadata={"help": "Limite inferior do log-LR calibrado."}
    )

    max_log_lr: float = field(
        default=20.0,
        metadata={"help": "Limite superior do log-LR calibrado."}
    )

    interpolation_points: int = field(
        default=10000,
        metadata={"help": "Quantidade de pontos usados para construir a curva interpolada score -> LR."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente usada em amostragens internas quando aplicável."}
    )

    kernel: str = field(
        default="gaussian",
        metadata={"help": "Kernel do sklearn.neighbors.KernelDensity."}
    )

    def __post_init__(self) -> None:
        if self.bandwidth_genuine <= 0:
            raise ValueError("bandwidth_genuine deve ser maior que zero.")

        if self.bandwidth_impostor <= 0:
            raise ValueError("bandwidth_impostor deve ser maior que zero.")

        if self.max_samples_for_gridsearch <= 0:
            raise ValueError("max_samples_for_gridsearch deve ser maior que zero.")

        if self.max_stored_samples <= 0:
            raise ValueError("max_stored_samples deve ser maior que zero.")

        if self.fisher_eps <= 0:
            raise ValueError("fisher_eps deve ser maior que zero.")

        if self.min_bandwidth_abs <= 0:
            raise ValueError("min_bandwidth_abs deve ser maior que zero.")

        if self.max_grid_expansions < 0:
            raise ValueError("max_grid_expansions não pode ser negativo.")

        if self.interpolation_points < 100:
            raise ValueError("interpolation_points deve ser pelo menos 100.")

        if self.min_log_lr >= self.max_log_lr:
            raise ValueError("min_log_lr deve ser menor que max_log_lr.")

        if not self.kernel:
            raise ValueError("kernel não pode ser vazio.")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO de configuração a partir de dicionário.
        """
        if not data:
            return cls()

        return cls(**data)