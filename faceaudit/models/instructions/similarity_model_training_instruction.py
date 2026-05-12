# faceaudit/models/instructions/similarity_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.similarity_model_type import SimilarityModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)


T = TypeVar("T", bound="SimilarityModelTrainingInstruction")


@dataclass
class SimilarityModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de similaridade.

    Esta instrução descreve:
    - qual família de modelo de similaridade será usada
    - qual recurso de entrada (dataset) será usado no treinamento/preparação
    - qual config de hyperparâmetros (embutido na instrução)
    - qual recurso de saída representará o modelo treinado/preparado

    Observação:
    O termo "training" aqui é usado em sentido amplo. Cosine não exige treino —
    é apenas instanciação de uma função de similaridade. PLDA exige treino real
    com hyperparâmetros descritos em `similarity_model_config`.
    """

    similarity_model_type: SimilarityModelType = field(
        default=SimilarityModelType.COSINE,
        metadata={"help": "Tipo de modelo de similaridade a ser usado."}
    )

    input_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset de entrada usado pelo modelo de similaridade."}
    )

    similarity_model_config: Optional[PLDASimilarityModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do modelo (PLDA). None para Cosine."}
    )

    trained_similarity_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o modelo de similaridade treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SIMILARITY_MODEL_TRAINING

        if not self.name:
            self.name = "Similarity Model Training"

        # Auto-instancia o config quando o tipo o exige e ele ainda não foi
        # preenchido. Para Cosine, mantém None (não tem hyperparâmetros).
        if self.uses_plda and self.similarity_model_config is None:
            self.similarity_model_config = PLDASimilarityModelConfig()

        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        triples = SimilarityModelType.choices_with_description()
        return [
            ChoiceFieldSpec(
                field_name="similarity_model_type",
                label="Tipo do Modelo",
                choices=tuple(value for value, _label, _desc in triples),
                enum_type=SimilarityModelType,
                descriptions={value: desc for value, _label, desc in triples},
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TS = (StrategyResourceType.TRAINED_SIMILARITY_MODEL,)
        specs: list[ResourceFieldSpec] = []
        # Dataset só aparece para PLDA — COSINE é stateless e não precisa de
        # treino.
        if self.uses_plda:
            specs.append(ResourceFieldSpec("input_dataset_resource_id", "Dataset", DS))
        specs.append(ResourceFieldSpec(
            "trained_similarity_model_resource_id",
            "Modelo Treinado",
            TS,
            is_output=True,
        ))
        return specs

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos específicos
        desta instrução. A config do modelo é embutida no DTO e não conta como input.

        Cosine não exige dataset de entrada (stateless), então só PLDA inclui
        input_dataset_resource_id no gate de dependência.
        """
        inputs = []
        outputs = []

        if self.uses_plda and self.input_dataset_resource_id:
            inputs.append(self.input_dataset_resource_id)

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
        - similarity_model_config
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

        if "similarity_model_config" in processed_data and isinstance(
            processed_data["similarity_model_config"], dict
        ):
            processed_data["similarity_model_config"] = PLDASimilarityModelConfig.from_dict(
                processed_data["similarity_model_config"]
            )

        # Retrocompat: campo removido
        processed_data.pop("similarity_model_config_resource_id", None)

        return cls(**processed_data)
