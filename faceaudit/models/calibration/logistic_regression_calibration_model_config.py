# faceaudit/models/calibration/logistic_regression_calibration_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="LogisticRegressionCalibrationModelConfig")


@dataclass
class LogisticRegressionCalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração por Regressão Logística.

    Esta configuração representa apenas parâmetros do calibrador, sem misturar:
    - modelo de similaridade
    - pipeline de transformação
    - estratégia de execução
    - mineração de pares

    Nesta primeira versão, o único hiperparâmetro exposto é o C da regressão logística.
    """

    C: float = field(
        default=1.0,
        metadata={"help": "Parâmetro de regularização inversa da Regressão Logística."}
    )

    def __post_init__(self) -> None:
        """
        Validação defensiva mínima do DTO.
        """
        if self.C <= 0:
            raise ValueError("C deve ser maior que zero.")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO de configuração a partir de dicionário.
        """
        if not data:
            return cls()

        return cls(**data)