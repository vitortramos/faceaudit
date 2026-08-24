# faceaudit/factories/strategy_executor_factory.py

from __future__ import annotations

from typing import Dict, Optional

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.executors.instruction_executor_registry import InstructionExecutorRegistry
from faceaudit.executors.instructions.calibration_model_training_instruction_executor import (
    CalibrationModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.dataset_merge_instruction_executor import (
    DatasetMergeInstructionExecutor,
)
from faceaudit.executors.instructions.dataset_split_instruction_executor import (
    DatasetSplitInstructionExecutor,
)
from faceaudit.executors.instructions.paired_dataset_split_instruction_executor import (
    PairedDatasetSplitInstructionExecutor,
)
from faceaudit.executors.instructions.pair_mining_instruction_executor import (
    PairMiningInstructionExecutor,
)
from faceaudit.executors.instructions.pipeline_application_instruction_executor import (
    PipelineApplicationInstructionExecutor,
)
from faceaudit.executors.instructions.pipeline_training_instruction_executor import (
    PipelineTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.similarity_model_training_instruction_executor import (
    SimilarityModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.score_calibration_instruction_executor import (
    ScoreCalibrationInstructionExecutor,
)
from faceaudit.executors.instructions.score_vectorization_instruction_executor import (
    ScoreVectorizationInstructionExecutor,
)
from faceaudit.executors.instructions.strategy_evaluation_instruction_executor import (
    StrategyEvaluationInstructionExecutor,
)
from faceaudit.executors.instructions.casework_visualization_instruction_executor import (
    CaseworkVisualizationInstructionExecutor,
)
from faceaudit.executors.instructions.dataset_balancing_instruction_executor import (
    DatasetBalancingInstructionExecutor,
)
from faceaudit.executors.instructions.hubness_analysis_instruction_executor import (
    HubnessAnalysisInstructionExecutor,
)
from faceaudit.executors.instructions.hubness_ablation_instruction_executor import (
    HubnessAblationInstructionExecutor,
)
from faceaudit.executors.instructions.similarity_scoring_instruction_executor import (
    SimilarityScoringInstructionExecutor,
)
from faceaudit.executors.instructions.typicality_model_training_instruction_executor import (
    TypicalityModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.typicality_pair_scoring_instruction_executor import (
    TypicalityPairScoringInstructionExecutor,
)
from faceaudit.executors.instructions.score_interaction_feature_instruction_executor import (
    ScoreInteractionFeatureInstructionExecutor,
)
from faceaudit.executors.instructions.lr_architecture_vectorization_instruction_executor import (
    LRArchitectureVectorizationInstructionExecutor,
)
from faceaudit.executors.instructions.embedding_aggregation_instruction_executor import (
    EmbeddingAggregationInstructionExecutor,
)
from faceaudit.executors.strategy_executor import StrategyExecutor
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol
from faceaudit.validation_rules.strategy_embedding_consistency_rule import (
    StrategyEmbeddingConsistencyRule,
)


class StrategyExecutorFactory:
    """
    Factory responsável por montar um StrategyExecutor da nova arquitetura.

    Esta factory cria:
    - o contexto de execução concreto
    - o registry de executores de instrução
    - o StrategyExecutor final

    Objetivo:
    manter a montagem da nova strategy engine isolada da arquitetura antiga.
    """

    @staticmethod
    def build_instruction_executor_registry(
        comparison_service=None,
    ) -> InstructionExecutorRegistry:
        """
        Monta e retorna o registry padrão da nova strategy engine.

        Args:
            comparison_service: instância opcional de ComparisonService a ser
                injetada nos executores que dela dependem (HUBNESS_ANALYSIS,
                HUBNESS_ABLATION). Se None, esses executores instanciam um
                ComparisonService default on-demand.
        """
        registry = InstructionExecutorRegistry()

        registry.register(
            StrategyInstructionType.PIPELINE_TRAINING,
            PipelineTrainingInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.PIPELINE_APPLICATION,
            PipelineApplicationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.DATASET_SPLIT,
            DatasetSplitInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.PAIRED_DATASET_SPLIT,
            PairedDatasetSplitInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.DATASET_MERGE,
            DatasetMergeInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.PAIR_MINING,
            PairMiningInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.SIMILARITY_MODEL_TRAINING,
            SimilarityModelTrainingInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.SIMILARITY_SCORING,
            SimilarityScoringInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.CALIBRATION_MODEL_TRAINING,
            CalibrationModelTrainingInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.SCORE_CALIBRATION,
            ScoreCalibrationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.SCORE_VECTORIZATION,
            ScoreVectorizationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.STRATEGY_EVALUATION,
            StrategyEvaluationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.CASEWORK_VISUALIZATION,
            CaseworkVisualizationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.DATASET_BALANCING,
            DatasetBalancingInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.HUBNESS_ANALYSIS,
            HubnessAnalysisInstructionExecutor(comparison_service=comparison_service),
        )

        registry.register(
            StrategyInstructionType.HUBNESS_ABLATION,
            HubnessAblationInstructionExecutor(comparison_service=comparison_service),
        )

        registry.register(
            StrategyInstructionType.TYPICALITY_MODEL_TRAINING,
            TypicalityModelTrainingInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.TYPICALITY_PAIR_SCORING,
            TypicalityPairScoringInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.SCORE_INTERACTION_FEATURE,
            ScoreInteractionFeatureInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
            LRArchitectureVectorizationInstructionExecutor(),
        )

        registry.register(
            StrategyInstructionType.EMBEDDING_AGGREGATION,
            EmbeddingAggregationInstructionExecutor(),
        )

        return registry

    @staticmethod
    def build_default_validation_rules() -> Dict[str, ValidationRuleProtocol]:
        """
        Retorna o conjunto default de regras "globais" que devem estar
        sempre disponíveis no contexto, independente da estratégia.

        Atualmente:
        - StrategyEmbeddingConsistencyRule: garante que todos os datasets
          da estratégia usam a mesma fonte de embedding (mesmo
          `model_name` no column_specs).

        Regras por instrução (Fisher, MinCalibrationPairs, etc.) NÃO
        entram aqui — elas são instanciadas pelo próprio DTO em
        `build_validation_rules()` ou referenciadas via
        `validation_rule_ids` na estratégia.
        """
        rule = StrategyEmbeddingConsistencyRule()
        return {rule.rule_id: rule}

    @staticmethod
    def build(
        validation_rules: Optional[Dict[str, ValidationRuleProtocol]] = None,
        artifact_storage: Optional[ArtifactStorageProtocol] = None,
        comparison_service=None,
    ) -> StrategyExecutor:
        """
        Monta e retorna um StrategyExecutor pronto para uso.

        Args:
            validation_rules: Mapa rule_id → ValidationRuleProtocol para validação
                              da estratégia e de instruções individuais.
            artifact_storage: Storage do projeto. Quando fornecido, os recursos
                              declarados na estratégia que possuírem artifact_ref
                              mas payload=None são hidratados automaticamente antes
                              da execução (carregamento de parquet, JSON, etc.).
                              Obrigatório quando a estratégia é carregada de JSON.
            comparison_service: Instância opcional de ComparisonService a ser
                              injetada nos executores que dela dependem
                              (HUBNESS_ANALYSIS, HUBNESS_ABLATION). Quando None,
                              esses executores instanciam um ComparisonService
                              default on-demand.
        """
        registry = StrategyExecutorFactory.build_instruction_executor_registry(
            comparison_service=comparison_service,
        )

        # Mescla as regras default (sempre disponíveis) com as eventualmente
        # passadas pelo caller. As do caller têm prioridade em caso de
        # conflito de rule_id (permite override). As default ficam marcadas
        # como always-on (rodam mesmo se não declaradas pela estratégia).
        default_rules = StrategyExecutorFactory.build_default_validation_rules()
        always_on_ids = list(default_rules.keys())

        merged_rules: Dict[str, ValidationRuleProtocol] = dict(default_rules)
        if validation_rules:
            merged_rules.update(validation_rules)

        return StrategyExecutor(
            context_factory=StrategyExecutionContext,
            instruction_executor_registry=registry,
            validation_rules=merged_rules,
            artifact_storage=artifact_storage,
            always_on_rule_ids=always_on_ids,
        )