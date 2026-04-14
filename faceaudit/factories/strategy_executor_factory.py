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
from faceaudit.executors.instructions.similarity_scoring_instruction_executor import (
    SimilarityScoringInstructionExecutor,
)
from faceaudit.executors.strategy_executor import StrategyExecutor
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol


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
    def build_instruction_executor_registry() -> InstructionExecutorRegistry:
        """
        Monta e retorna o registry padrão da nova strategy engine.
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

        return registry

    @staticmethod
    def build(
        validation_rules: Optional[Dict[str, ValidationRuleProtocol]] = None,
        artifact_storage: Optional[ArtifactStorageProtocol] = None,
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
        """
        registry = StrategyExecutorFactory.build_instruction_executor_registry()

        return StrategyExecutor(
            context_factory=StrategyExecutionContext,
            instruction_executor_registry=registry,
            validation_rules=validation_rules or {},
            artifact_storage=artifact_storage,
        )