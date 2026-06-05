# faceaudit/models/instructions/strategy_instruction_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="StrategyInstructionDefinition")


@dataclass(frozen=True)
class ResourceFieldSpec:
    """Especificação de um campo de recurso para renderização no config panel.

    Campos:
        field_name: nome do atributo do DTO (ex: 'dataset_resource_id').
        label: rótulo exibido no formulário.
        accepted_types: tipos de recurso permitidos no dropdown.
        is_output: True se este field é uma saída produzida pela instrução
            (renderiza como linha de output em vez de dropdown).
        triggers_rebuild: se True, mudanças no dropdown disparam um rebuild
            completo do form. Útil quando outros field specs dependem do
            recurso selecionado (ex: DATASET_BALANCING.quality_metric usa
            `dynamic_choices` que inspeciona o dataset para filtrar as
            métricas disponíveis). Default False (sem rebuild).
    """
    field_name: str
    label: str
    accepted_types: Tuple[StrategyResourceType, ...]
    is_output: bool = False
    triggers_rebuild: bool = False


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
        dynamic_choices: opcional. Callable[[List[StrategyResource]], Optional[Tuple[str, ...]]]
                      executado pelo painel a cada rebuild do form. Quando
                      retorna uma tupla, substitui `choices` (típico: filtrar
                      opções de acordo com o estado atual de outras seleções
                      do DTO ou recursos disponíveis no projeto). Retornar None
                      preserva `choices`. O painel sempre passa a lista de
                      recursos disponíveis como argumento.
    """
    field_name: str
    label: str
    choices: Tuple[str, ...]
    enum_type: Optional[type] = None
    descriptions: Optional[Dict[str, str]] = None
    dynamic_choices: Optional[Any] = None


@dataclass(frozen=True)
class ValueFieldSpec:
    """
    Especificação de um campo primitivo (int, float, str, bool, lista) editável
    no config panel, fora do `parameters` dict.

    value_type:
      - "int", "float", "str", "bool" — tipos primitivos diretos
      - "list_str" — lista de strings (entrada via texto separado por vírgula)
      - "list_int" — lista de inteiros (entrada via texto separado por vírgula)

    triggers_rebuild:
      Se True, o config panel deve aplicar o valor ao DTO e reconstruir o
      formulário imediatamente após o usuário alterar o widget. Útil para
      campos cujo valor altera os specs subsequentes (ex: bool que habilita
      ou esconde campos de output condicionais).
    """
    field_name: str
    label: str
    value_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    decimals: Optional[int] = None
    placeholder: Optional[str] = None
    triggers_rebuild: bool = False


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

    validation_rule_configs: List[ValidationRuleConfig] = field(
        default_factory=list,
        metadata={"help": (
            "Lista de regras de validação associadas à instrução. Cada "
            "config carrega o `rule_type` (enum) + `params` (dict de "
            "construção). As regras são instanciadas em runtime pelo "
            "`ValidationRuleFactory` e executadas pelo estágio 2 do "
            "pipeline de validação. Populadas automaticamente por "
            "`__post_init__` em DTOs que conhecem seus defaults; podem "
            "ser editadas/removidas pelo usuário na UI."
        )}
    )

    validation_rules_user_managed: bool = field(
        default=False,
        metadata={"help": (
            "Sinaliza que o usuário assumiu o controle das regras de "
            "validação desta instrução. Quando True, `_ensure_default_"
            "validation_rules` não re-popula defaults — preserva exclusões "
            "explícitas entre reloads de projeto. É marcado True quando "
            "o usuário deleta uma regra na UI; volta a False quando ele "
            "clica em 'Restaurar padrões'."
        )}
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

    def prune_unused_fields(self) -> None:
        """
        Limpa campos do DTO que não são usados na configuração atual.

        Default no-op. Override em DTOs cujos campos relevantes variam com
        um choice field (ex: PipelineApplicationInstruction tem campos
        específicos pra LINEAR e outros pra DUAL; trocar de modo deixa o
        conjunto não usado pendurado, e o executor reclama de campos
        obrigatórios vazios enquanto há lixo do modo oposto preenchido).

        Chamado apenas quando o usuário clica "Aplicar alterações" — NÃO
        no `_sync_declared_io` (que roda a cada interação) pra evitar
        que digitação intermediária perca dados.
        """
        pass

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
        return bool(self.validation_rule_configs)

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
            resolved_type = StrategyInstructionType.from_str(raw_type)
            processed_data["instruction_type"] = resolved_type
            # Defensa contra instruction_type extinto: `from_str` faz
            # fallback pra PIPELINE_TRAINING quando não reconhece (ex:
            # `SCORE_FUSION` removido). Detecta o fallback comparando
            # o valor original com o resolvido — se for um fallback de
            # tipo desconhecido, retorna None pra que o caller filtre
            # essa instrução em vez de fazer dispatch errado.
            if not isinstance(raw_type, StrategyInstructionType):
                norm = (
                    str(raw_type).strip().upper()
                    .replace("-", "_").replace(" ", "_")
                )
                known_values = {t.value for t in StrategyInstructionType}
                known_names = {t.name for t in StrategyInstructionType}
                if norm not in known_values and norm not in known_names:
                    print(
                        f"[StrategyInstructionDefinition] AVISO: tipo "
                        f"de instrução extinto/desconhecido '{raw_type}' "
                        f"— instrução ignorada na carga da definition."
                    )
                    return None

        # Retrocompat: campo extinto `validation_rule_ids: List[str]` é
        # descartado em silêncio — defaults vão ser repopulados no
        # `__post_init__` das instruções que sabem suas regras.
        processed_data.pop("validation_rule_ids", None)

        # Hidrata configs persistidos (lista de dicts → lista de DTOs).
        raw_configs = processed_data.get("validation_rule_configs")
        if isinstance(raw_configs, list):
            hydrated = []
            for entry in raw_configs:
                if isinstance(entry, ValidationRuleConfig):
                    hydrated.append(entry)
                elif isinstance(entry, dict):
                    cfg = ValidationRuleConfig.from_dict(entry)
                    if cfg is not None:
                        hydrated.append(cfg)
            processed_data["validation_rule_configs"] = hydrated

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

        if instruction_type == StrategyInstructionType.SCORE_VECTORIZATION:
            from faceaudit.models.instructions.score_vectorization_instruction import (
                ScoreVectorizationInstruction,
            )
            return ScoreVectorizationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.STRATEGY_EVALUATION:
            from faceaudit.models.instructions.strategy_evaluation_instruction import (
                StrategyEvaluationInstruction,
            )
            return StrategyEvaluationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.CASEWORK_VISUALIZATION:
            from faceaudit.models.instructions.casework_visualization_instruction import (
                CaseworkVisualizationInstruction,
            )
            return CaseworkVisualizationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.DATASET_BALANCING:
            from faceaudit.models.instructions.dataset_balancing_instruction import (
                DatasetBalancingInstruction,
            )
            return DatasetBalancingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.HUBNESS_ANALYSIS:
            from faceaudit.models.instructions.hubness_analysis_instruction import (
                HubnessAnalysisInstruction,
            )
            return HubnessAnalysisInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.HUBNESS_ABLATION:
            from faceaudit.models.instructions.hubness_ablation_instruction import (
                HubnessAblationInstruction,
            )
            return HubnessAblationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.KNN_TYPICALITY_MODEL_TRAINING:
            from faceaudit.models.instructions.knn_typicality_model_training_instruction import (
                KNNTypicalityModelTrainingInstruction,
            )
            return KNNTypicalityModelTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.TYPICALITY_PAIR_SCORING:
            from faceaudit.models.instructions.typicality_pair_scoring_instruction import (
                TypicalityPairScoringInstruction,
            )
            return TypicalityPairScoringInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SCORE_INTERACTION_FEATURE:
            from faceaudit.models.instructions.score_interaction_feature_instruction import (
                ScoreInteractionFeatureInstruction,
            )
            return ScoreInteractionFeatureInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION:
            from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
                LRArchitectureVectorizationInstruction,
            )
            return LRArchitectureVectorizationInstruction.from_dict(processed_data)

        return cls(**processed_data)