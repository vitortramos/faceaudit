# faceaudit/models/similarity/plda_similarity_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.plda_mode import PLDAMode


T = TypeVar("T", bound="PLDASimilarityModelConfig")


@dataclass
class PLDASimilarityModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de similaridade PLDA na nova arquitetura.

    Esta configuração representa apenas parâmetros do modelo de similaridade,
    sem misturar:
    - calibração
    - pipeline de transformação
    - estratégias antigas de treino
    - decisões de execução da strategy engine

    Regras semânticas:
    - FULL_RANK: usa o maior rank permitido pelos dados; latent_dim é ignorado
    - LOW_RANK: usa latent_dim como limite do espaço latente
    """

    em_iterations: int = field(
        default=20,
        metadata={"help": "Número de iterações do algoritmo EM no treino do PLDA."}
    )

    plda_mode: PLDAMode = field(
        default=PLDAMode.LOW_RANK,
        metadata={"help": "Modo de dimensionalidade latente do PLDA."}
    )

    latent_dim: int = field(
        default=128,
        metadata={"help": "Dimensão latente usada quando plda_mode=LOW_RANK."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente aleatória para reprodutibilidade, quando aplicável."}
    )

    def __post_init__(self) -> None:
        if self.em_iterations <= 0:
            raise ValueError("em_iterations deve ser maior que zero.")

        if self.latent_dim <= 0:
            raise ValueError("latent_dim deve ser maior que zero.")

    @property
    def uses_full_rank(self) -> bool:
        return self.plda_mode == PLDAMode.FULL_RANK

    @property
    def uses_low_rank(self) -> bool:
        return self.plda_mode == PLDAMode.LOW_RANK

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, convertendo corretamente o enum PLDAMode.
        """
        if not data:
            return cls()

        processed_data = data.copy()

        if "plda_mode" in processed_data and processed_data["plda_mode"]:
            processed_data["plda_mode"] = PLDAMode.from_str(
                processed_data["plda_mode"]
            )

        return cls(**processed_data)