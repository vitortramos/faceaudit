# faceaudit/executors/instructions/dataset_balancing_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.instructions.dataset_balancing_instruction import (
    DatasetBalancingInstruction,
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


class DatasetBalancingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de balanceamento por identidade.

    Suporta tanto `FaceDataset` (lista de `FaceInfo`) quanto `EmbeddingDataset`
    (lista de `EmbeddingInfo`). A lógica de cap/floor é compartilhada — o
    que muda é a função extratora de identidade e de score de qualidade.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, DatasetBalancingInstruction):
            return self._failure_result(
                "O executor DatasetBalancingInstructionExecutor recebeu uma "
                "instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando balanceamento: {instruction.name} "
            f"({instruction.instruction_id}) — modo={instruction.balancing_mode.to_str()}, "
            f"critério={instruction.selection_criterion.to_str()}"
        )

        try:
            input_resource = context.get_resource(instruction.dataset_resource_id)
            input_dataset = input_resource.payload

            # Delega a lógica de balanceamento pro service
            # compartilhado — mesma implementação usada pelo botão
            # "Balancear" da toolbar do Dataset Manager.
            from faceaudit.services.dataset_balancing_service import (
                DatasetBalancingService,
            )
            service = DatasetBalancingService()
            result = service.balance(
                dataset=input_dataset,
                mode=instruction.balancing_mode,
                target_n=instruction.target_n,
                selection=instruction.selection_criterion,
                random_seed=instruction.random_seed,
                quality_metric=instruction.quality_metric,
                output_name=instruction.output_dataset_name,
                logger=lambda msg: self._log(context, msg),
            )

            output_resource_id = instruction.output_dataset_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=result.dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ Balanceamento concluído. "
                f"Entrada: {result.input_size} samples / "
                f"{result.input_n_ids} IDs → "
                f"Saída: {result.output_size} samples / "
                f"{result.output_n_ids} IDs (N={result.effective_n}). "
                f"Recurso gerado: '{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "balancing_mode": instruction.balancing_mode.to_str(),
                    "selection_criterion": instruction.selection_criterion.to_str(),
                    "quality_metric": (
                        instruction.quality_metric.name
                        if instruction.uses_quality_selection else None
                    ),
                    "effective_n": result.effective_n,
                    "input_size": result.input_size,
                    "input_n_ids": result.input_n_ids,
                    "output_size": result.output_size,
                    "output_n_ids": result.output_n_ids,
                    "discarded_identities": result.discarded_identities,
                    "quality_fallback_count": result.quality_fallback_count,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro no balanceamento {instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    # ── Validação específica do executor ─────────────────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, DatasetBalancingInstruction):
            result.add_error(
                "A instrução informada não é do tipo DatasetBalancingInstruction."
            )
            return result

        if not instruction.dataset_resource_id:
            result.add_error(
                "DATASET_BALANCING exige 'dataset_resource_id'."
            )
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de entrada não encontrado: "
                f"'{instruction.dataset_resource_id}'."
            )

        if not instruction.output_dataset_resource_id:
            result.add_error(
                "DATASET_BALANCING exige 'output_dataset_resource_id'."
            )

        if (
            instruction.output_dataset_resource_id
            and instruction.output_dataset_resource_id == instruction.dataset_resource_id
        ):
            result.add_warning(
                "output_dataset_resource_id coincide com a entrada — "
                "o contexto irá sobrescrever o recurso original."
            )

        if not instruction.uses_auto_min and instruction.target_n < 1:
            result.add_error(
                "target_n deve ser ≥ 1 nos modos CAP_ONLY e FLOOR_AND_CAP."
            )

        # Valida tipo do payload (apenas se o recurso estiver disponível).
        if (
            instruction.dataset_resource_id
            and context.has_resource(instruction.dataset_resource_id)
        ):
            payload = context.get_resource(
                instruction.dataset_resource_id
            ).payload

            if payload is None:
                result.add_error(
                    "Payload do dataset de entrada não pôde ser hidratado."
                )
            elif not isinstance(payload, (FaceDataset, EmbeddingDataset)):
                result.add_error(
                    f"Tipo de dataset não suportado para balanceamento: "
                    f"{type(payload).__name__}. Esperado FaceDataset ou "
                    f"EmbeddingDataset."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
