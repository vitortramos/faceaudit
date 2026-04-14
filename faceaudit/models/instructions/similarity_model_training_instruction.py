# faceaudit/models/instructions/similarity_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.similarity_model_type import SimilarityModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="SimilarityModelTrainingInstruction")


@dataclass
class SimilarityModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de similaridade.

    Esta instrução descreve:
    - qual família de modelo de similaridade será usada
    - qual recurso de entrada será usado no treinamento/preparação
    - qual recurso de configuração do modelo será usado
    - qual recurso de saída representará o modelo treinado/preparado

    Observação:
    O termo "training" aqui é usado em sentido amplo. Alguns modelos podem exigir
    treino real (ex: PLDA), enquanto outros podem apenas ser preparados/instanciados
    (ex: cosine similarity).
    """

    similarity_model_type: SimilarityModelType = field(
        default=SimilarityModelType.COSINE,
        metadata={"help": "Tipo de modelo de similaridade a ser usado."}
    )

    input_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset de entrada usado pelo modelo de similaridade."}
    )

    similarity_model_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de configuração do modelo de similaridade."}
    )

    trained_similarity_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o modelo de similaridade treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SIMILARITY_MODEL_TRAINING

        if not self.name:
            self.name = "Similarity Model Training"

        self._sync_declared_io()

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos específicos
        desta instrução.
        """
        inputs = []
        outputs = []

        if self.input_dataset_resource_id:
            inputs.append(self.input_dataset_resource_id)

        if self.similarity_model_config_resource_id:
            inputs.append(self.similarity_model_config_resource_id)

        if self.trained_similarity_model_resource_id:
            outputs.append(self.trained_similarity_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @property
    def uses_cosine(self) -> bool:
        return self.similarity_model_type == SimilarityModelType.COSINE

    @property
    def uses_plda(self) -> bool:
        return self.similarity_model_type == SimilarityModelType.PLDA

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - similarity_model_type
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if (
            "similarity_model_type" in processed_data
            and processed_data["similarity_model_type"]
        ):
            processed_data["similarity_model_type"] = SimilarityModelType.from_str(
                processed_data["similarity_model_type"]
            )

        return cls(**processed_data)