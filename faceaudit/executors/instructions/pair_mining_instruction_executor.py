# faceaudit/executors/instructions/pair_mining_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_comparison_dataset import EmbeddingComparisonDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.pair_mining_instruction import PairMiningInstruction
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.comparison_mining_service import ComparisonMiningService


class PairMiningInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de curadoria de pares (PAIR_MINING).

    Delega a lógica de mineração ao ComparisonMiningService, produzindo dois
    EmbeddingComparisonDataset curados: impostores e genuínos.
    """

    def __init__(self) -> None:
        self._mining_service = ComparisonMiningService()

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, PairMiningInstruction):
            return self._failure_result(
                "O executor PairMiningInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando curadoria de pares: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            dataset_resource = context.get_resource(instruction.comparison_dataset_resource_id)
            dataset: EmbeddingComparisonDataset = dataset_resource.payload

            ds_impostors, ds_genuines = self._mining_service.mine_pairs(
                dataset=dataset,
                config=instruction.mining_config,
            )

            impostors_id = instruction.output_impostors_resource_id
            genuines_id = instruction.output_genuines_resource_id

            context.set_resource(StrategyResource(
                resource_id=impostors_id,
                name=impostors_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=ds_impostors,
                artifact_ref=None,
            ))
            context.set_resource(StrategyResource(
                resource_id=genuines_id,
                name=genuines_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=ds_genuines,
                artifact_ref=None,
            ))

            self._log(
                context,
                f"✅ Mineração concluída. "
                f"Impostores: {len(ds_impostors.items)} pares | "
                f"Genuínos: {len(ds_genuines.items)} pares."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[impostors_id, genuines_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_dataset_resource_id": instruction.comparison_dataset_resource_id,
                    "output_impostors_resource_id": impostors_id,
                    "output_genuines_resource_id": genuines_id,
                    "impostors_count": len(ds_impostors.items),
                    "genuines_count": len(ds_genuines.items),
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar curadoria de pares "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, PairMiningInstruction):
            result.add_error(
                "A instrução informada não é do tipo PairMiningInstruction."
            )
            return result

        if not instruction.comparison_dataset_resource_id:
            result.add_error(
                "A instrução de pair mining deve definir 'comparison_dataset_resource_id'."
            )
        elif not context.has_resource(instruction.comparison_dataset_resource_id):
            result.add_error(
                f"O recurso de dataset não foi encontrado no contexto: "
                f"'{instruction.comparison_dataset_resource_id}'."
            )
        else:
            dataset = context.get_resource(instruction.comparison_dataset_resource_id).payload
            if not isinstance(dataset, EmbeddingComparisonDataset):
                result.add_error(
                    f"O recurso '{instruction.comparison_dataset_resource_id}' deve ser um "
                    f"EmbeddingComparisonDataset, mas é {type(dataset).__name__}."
                )

        if instruction.mining_config is None:
            result.add_error(
                "A instrução de pair mining deve definir 'mining_config'."
            )

        if not instruction.output_impostors_resource_id:
            result.add_error(
                "A instrução de pair mining deve definir 'output_impostors_resource_id'."
            )

        if not instruction.output_genuines_resource_id:
            result.add_error(
                "A instrução de pair mining deve definir 'output_genuines_resource_id'."
            )

        if (
            instruction.output_impostors_resource_id
            and instruction.output_genuines_resource_id
            and instruction.output_impostors_resource_id == instruction.output_genuines_resource_id
        ):
            result.add_error(
                "Os recursos de saída de impostores e genuínos devem ser diferentes."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
