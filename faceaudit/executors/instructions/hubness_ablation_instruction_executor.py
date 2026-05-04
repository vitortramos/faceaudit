# faceaudit/executors/instructions/hubness_ablation_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.hubness_ablation_mode import HubnessAblationMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.hubness_ablation_instruction import (
    HubnessAblationInstruction,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class HubnessAblationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução HUBNESS_ABLATION.

    Recorta progressivamente o dataset (1, 2, 3 fotos por ID + total)
    seguindo o critério escolhido (BY_QUANTITY ou BY_POSE) e mede o
    hubness em cada cenário, produzindo um AblationStudyInfo persistido
    como pickle.

    O ComparisonService é injetado pela factory.
    """

    def __init__(self, comparison_service=None):
        self._comparison_service = comparison_service

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, HubnessAblationInstruction):
            return self._failure_result(
                "Executor HubnessAblationInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando estudo de ablação: {instruction.name} "
            f"({instruction.instruction_id}) | mode={instruction.ablation_mode.value}"
        )

        try:
            ds_resource = context.get_resource(instruction.dataset_resource_id)
            dataset: EmbeddingDataset = ds_resource.payload

            service = self._build_service(instruction.k_neighbors)
            log_cb = lambda msg: self._log(context, msg)

            if instruction.ablation_mode == HubnessAblationMode.BY_QUANTITY:
                study = service.compute_ablation_by_quantity(
                    dataset,
                    batch_size=instruction.batch_size,
                    logger_callback=log_cb,
                )
            else:  # BY_POSE
                study = service.compute_ablation_by_pose(
                    dataset,
                    batch_size=instruction.batch_size,
                    logger_callback=log_cb,
                )

            output_id = instruction.output_ablation_resource_id
            output = StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.HUBNESS_ABLATION_RESULT,
                payload=study,
                artifact_ref=None,
            )
            context.set_resource(output)

            scenarios_summary = ", ".join(
                f"{s.scenario_name} (max Nk={s.max_nk})"
                for s in study.scenarios
            )
            self._log(
                context,
                f"✅ Ablação concluída ({len(study.scenarios)} cenários). {scenarios_summary}"
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "ablation_mode": instruction.ablation_mode.value,
                    "k_neighbors": int(study.k_neighbors),
                    "n_scenarios": len(study.scenarios),
                    "output_resource_id": output_id,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro na ablação {instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _build_service(self, k_neighbors: int):
        from faceaudit.services.hubness_analysis_service import HubnessAnalysisService
        comp = self._comparison_service or self._build_default_comparison_service()
        return HubnessAnalysisService(comparison_service=comp, k_neighbors=k_neighbors)

    @staticmethod
    def _build_default_comparison_service():
        from faceaudit.services.comparison_service import ComparisonService
        return ComparisonService()

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, HubnessAblationInstruction):
            result.add_error("Instrução não é HubnessAblationInstruction.")
            return result

        if not instruction.dataset_resource_id:
            result.add_error("HUBNESS_ABLATION exige 'dataset_resource_id'.")
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de entrada não encontrado: '{instruction.dataset_resource_id}'."
            )

        if not instruction.output_ablation_resource_id:
            result.add_error("HUBNESS_ABLATION exige 'output_ablation_resource_id'.")

        if instruction.k_neighbors < 1:
            result.add_error("k_neighbors deve ser ≥ 1.")

        if instruction.batch_size < 1:
            result.add_error("batch_size deve ser ≥ 1.")

        if (
            instruction.dataset_resource_id
            and context.has_resource(instruction.dataset_resource_id)
        ):
            payload = context.get_resource(instruction.dataset_resource_id).payload
            if payload is None:
                result.add_error("Payload do dataset não pôde ser hidratado.")
            elif not isinstance(payload, EmbeddingDataset):
                result.add_error(
                    f"HUBNESS_ABLATION exige EmbeddingDataset. Recebido: "
                    f"{type(payload).__name__}."
                )
            else:
                # Para BY_QUANTITY precisa de quality_score; para BY_POSE precisa de yaw/pitch.
                if instruction.ablation_mode == HubnessAblationMode.BY_QUANTITY:
                    sample = payload.items[0] if payload.items else None
                    if sample is not None and not getattr(sample, "quality_score", None):
                        result.add_warning(
                            "Modo BY_QUANTITY ordena por quality_score (FIQA), mas "
                            "as amostras parecem não ter essa métrica preenchida — "
                            "todos os cenários terão a mesma ordem (fallback aleatório)."
                        )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result
