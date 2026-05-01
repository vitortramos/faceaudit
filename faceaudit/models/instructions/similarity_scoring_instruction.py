# faceaudit/models/instructions/similarity_scoring_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)


T = TypeVar("T", bound="SimilarityScoringInstruction")


@dataclass
class SimilarityScoringInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para geração de scores de similaridade.

    Esta instrução descreve:
    - qual modelo de similaridade treinado será usado
    - quais datasets serão comparados
    - qual recurso de saída conterá o SimilarityScoreDataset

    Semântica:
    - se apenas probe_dataset_resource_id for informado, a instrução executa
      auto-comparação dentro do próprio dataset (upper triangle, sem diagonal)
    - se probe_dataset_resource_id e reference_dataset_resource_id forem informados,
      a instrução executa comparação cruzada entre os dois datasets
    """

    trained_similarity_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso que referencia o modelo de similaridade treinado."}
    )

    probe_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do dataset probe."}
    )

    reference_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do dataset reference. Opcional para auto-comparação."}
    )

    output_similarity_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que conterá o SimilarityScoreDataset."}
    )

    output_similarity_scores_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do SimilarityScoreDataset gerado."}
    )

    store_metadata_per_sample: bool = field(
        default=False,
        metadata={"help": "Indica se metadados adicionais devem ser armazenados em cada amostra."}
    )

    is_casework: bool = field(
        default=False,
        metadata={
            "help": (
                "Se marcado, os pares são tratados como caso forense (sem "
                "ground truth). Os samples produzidos têm same_person=None "
                "(não impostor por default), o output do dataset é marcado "
                "com metadata['is_casework']=True, e a UI exibe 'Pares de "
                "caso' em vez de 'Genuínos / Impostores'. STRATEGY_EVALUATION "
                "downstream rejeita este tipo de input por falta de ground "
                "truth para calcular EER/Cllr/AUC."
            )
        }
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SIMILARITY_SCORING

        if not self.name:
            self.name = "Similarity Scoring"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET, StrategyResourceType.INTERMEDIATE_DATASET)
        TS = (StrategyResourceType.TRAINED_SIMILARITY_MODEL,)
        OUT = (StrategyResourceType.SIMILARITY_SCORES,)
        return [
            ResourceFieldSpec("trained_similarity_model_resource_id", "Modelo de Similaridade", TS),
            ResourceFieldSpec("probe_dataset_resource_id", "Dataset 1", DS),
            ResourceFieldSpec("reference_dataset_resource_id", "Dataset 2 (opcional)", DS),
            ResourceFieldSpec("output_similarity_scores_resource_id", "Scores de Saída", OUT, is_output=True),
        ]

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                "is_casework",
                "Modo Casework (sem ground truth)",
                "bool",
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        """
        Texto explicativo do modo de comparação, exibido no topo da seção
        Configuração na UI. Esclarece o que acontece em auto-comparação vs
        comparação cruzada e que um embedding nunca é comparado consigo mesmo.
        """
        return (
            "Modo de comparação:\n"
            "• Apenas Dataset 1 → AUTO-COMPARAÇÃO dentro do dataset. "
            "Cada par (i, j) com i < j é comparado uma única vez. "
            "Um embedding nunca é comparado com ele mesmo.\n"
            "• Dataset 1 + Dataset 2 → COMPARAÇÃO CRUZADA. "
            "Cada amostra de Dataset 1 é comparada com cada amostra de Dataset 2 "
            "(N₁ × N₂ pares)."
        )

    @property
    def uses_self_comparison(self) -> bool:
        return bool(self.probe_dataset_resource_id) and not bool(
            self.reference_dataset_resource_id
        )

    @property
    def uses_cross_comparison(self) -> bool:
        return bool(self.probe_dataset_resource_id) and bool(
            self.reference_dataset_resource_id
        )

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.trained_similarity_model_resource_id:
            inputs.append(self.trained_similarity_model_resource_id)

        if self.probe_dataset_resource_id:
            inputs.append(self.probe_dataset_resource_id)

        if self.reference_dataset_resource_id:
            inputs.append(self.reference_dataset_resource_id)

        if self.output_similarity_scores_resource_id:
            outputs.append(self.output_similarity_scores_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        return cls(**processed_data)