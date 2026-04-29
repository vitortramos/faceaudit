# faceaudit/models/instructions/strategy_instruction_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="StrategyInstructionDefinition")


@dataclass(frozen=True)
class ResourceFieldSpec:
    """Especificação de um campo de recurso para renderização no config panel."""
    field_name: str
    label: str
    accepted_types: Tuple[StrategyResourceType, ...]
    is_output: bool = False


@dataclass(frozen=True)
class ChoiceFieldSpec:
    """Especificação de um campo de escolha (enum) para renderização no config panel.

    Campos:
        field_name:   nome do atributo do DTO.
        label:        rótulo exibido ao lado do combo.
        choices:      tupla de strings (uma por opção do combo). Os valores
                      devem ser conversíveis pelo `enum_type.from_str` quando
                      este existir.
        enum_type:    classe do enum, usada para conversão segura via from_str
                      ao gravar o valor de volta no DTO.
        descriptions: opcional. Mapa choice_value → descrição textual longa
                      exibida em uma linha de ajuda contextual abaixo do combo,
                      atualizada conforme a seleção do usuário.
    """
    field_name: str
    label: str
    choices: Tuple[str, ...]
    enum_type: Optional[type] = None
    descriptions: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ValueFieldSpec:
    """
    Especificação de um campo primitivo (int, float, str, bool, lista) editável
    no config panel, fora do `parameters` dict.

    value_type:
      - "int", "float", "str", "bool" — tipos primitivos diretos
      - "list_str" — lista de strings (entrada via texto separado por vírgula)
      - "list_int" — lista de inteiros (entrada via texto separado por vírgula)
    """
    field_name: str
    label: str
    value_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decimals: Optional[int] = None
    placeholder: Optional[str] = None


@dataclass
class StrategyInstructionDefinition(JSONSerializableMixin):
    """
    Representa a definição declarativa de uma instrução da estratégia.
    """

    instruction_id: str = field(
        metadata={"help": "Identificador único da instrução dentro da estratégia."}
    )
    name: str = field(
        metadata={"help": "Nome amigável da instrução."}
    )
    instruction_type: StrategyInstructionType = field(
        metadata={"help": "Tipo da instrução."}
    )

    input_resource_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de recursos de entrada consumidos pela instrução."}
    )
    output_resource_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de identificadores de recursos produzidos pela instrução."}
    )

    parameters: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Parâmetros específicos da instrução."}
    )

    validation_rule_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de identificadores das regras de validação associadas."}
    )

    description: Optional[str] = field(
        default=None,
        metadata={"help": "Descrição opcional da finalidade da instrução."}
    )

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        """Retorna campos de escolha (enums) para renderização no config panel. Override nos DTOs concretos."""
        return []

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        """Retorna specs para renderização no config panel. Override nos DTOs concretos."""
        return []

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        """
        Retorna campos primitivos (int, float, str, bool, listas) editáveis
        no config panel, declarados como atributos diretos do DTO (fora do
        `parameters` dict). Override nos DTOs concretos.
        """
        return []

    def get_configuration_help(self) -> Optional[str]:
        """
        Retorna texto explicativo opcional exibido no topo da seção
        Configuração da UI. Útil para esclarecer modos/regras semânticas
        que não cabem em tooltips de campo (ex: auto vs cross-comparison
        em SIMILARITY_SCORING). Override nos DTOs concretos quando
        relevante. None = sem ajuda contextual.
        """
        return None

    def build_validation_rules(self) -> list:
        """
        Retorna regras de validação dinâmicas baseadas no estado atual da instrução.
        Override nos DTOs concretos para regras semânticas específicas.
        """
        return []

    def get_output_resource_placeholders(self) -> list:
        """
        Retorna StrategyResource placeholders para cada campo de output preenchido.
        payload=None, artifact_ref=None — preenchidos ao executar a instrução.
        """
        from faceaudit.models.strategy.strategy_resource import StrategyResource

        placeholders = []
        for spec in self.get_resource_field_specs():
            if not spec.is_output:
                continue
            resource_id = getattr(self, spec.field_name, None)
            if not resource_id:
                continue
            resource_type = (
                spec.accepted_types[0]
                if spec.accepted_types
                else StrategyResourceType.UNKNOWN
            )
            placeholders.append(
                StrategyResource(
                    resource_id=resource_id,
                    name=resource_id,
                    resource_type=resource_type,
                )
            )
        return placeholders

    @property
    def has_inputs(self) -> bool:
        return bool(self.input_resource_ids)

    @property
    def has_outputs(self) -> bool:
        return bool(self.output_resource_ids)

    @property
    def has_validation_rules(self) -> bool:
        return bool(self.validation_rule_ids)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói a instrução a partir de um dicionário, realizando dispatch
        para o DTO especializado correto com base em instruction_type.
        """
        if not data:
            return None

        processed_data = data.copy()

        raw_type = processed_data.get("instruction_type")
        if raw_type:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(raw_type)

        instruction_type = processed_data.get("instruction_type")

        if instruction_type == StrategyInstructionType.PIPELINE_TRAINING:
            from faceaudit.models.instructions.pipeline_training_instruction import (
                PipelineTrainingInstruction,
            )
            return PipelineTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.PIPELINE_APPLICATION:
            from faceaudit.models.instructions.pipeline_application_instruction import (
                PipelineApplicationInstruction,
            )
            return PipelineApplicationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.DATASET_SPLIT:
            from faceaudit.models.instructions.dataset_split_instruction import (
                DatasetSplitInstruction,
            )
            return DatasetSplitInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.PAIRED_DATASET_SPLIT:
            from faceaudit.models.instructions.paired_dataset_split_instruction import (
                PairedDatasetSplitInstruction,
            )
            return PairedDatasetSplitInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.DATASET_MERGE:
            from faceaudit.models.instructions.dataset_merge_instruction import (
                DatasetMergeInstruction,
            )
            return DatasetMergeInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SIMILARITY_MODEL_TRAINING:
            from faceaudit.models.instructions.similarity_model_training_instruction import (
                SimilarityModelTrainingInstruction,
            )
            return SimilarityModelTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SIMILARITY_SCORING:
            from faceaudit.models.instructions.similarity_scoring_instruction import (
                SimilarityScoringInstruction,
            )
            return SimilarityScoringInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.PAIR_MINING:
            from faceaudit.models.instructions.pair_mining_instruction import (
                PairMiningInstruction,
            )
            return PairMiningInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.CALIBRATION_MODEL_TRAINING:
            from faceaudit.models.instructions.calibration_model_training_instruction import (
                CalibrationModelTrainingInstruction,
            )
            return CalibrationModelTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SCORE_CALIBRATION:
            from faceaudit.models.instructions.score_calibration_instruction import (
                ScoreCalibrationInstruction,
            )
            return ScoreCalibrationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.STRATEGY_EVALUATION:
            from faceaudit.models.instructions.strategy_evaluation_instruction import (
                StrategyEvaluationInstruction,
            )
            return StrategyEvaluationInstruction.from_dict(processed_data)

        return cls(**processed_data)