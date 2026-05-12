# faceaudit/executors/instructions/pipeline_application_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType

from faceaudit.executors.instructions.strategy_instruction_executor_base import StrategyInstructionExecutorBase
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.pipeline_application_instruction import (
    PipelineApplicationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.pipelines.embedding_transformation_pipeline import (
    EmbeddingTransformationPipeline,
)
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.pipeline_trajectory_service import PipelineTrajectoryService




class PipelineApplicationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de aplicação de pipeline.

    Fluxo:
    1. Valida a instrução e os recursos necessários.
    2. Resolve pipeline treinado e datasets no contexto.
    3. Executa a aplicação do pipeline.
    4. Registra os datasets transformados como novos recursos no contexto.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, PipelineApplicationInstruction):
            return self._failure_result(
                "O executor PipelineApplicationInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de aplicação de pipeline: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            if instruction.pipeline_mode == PipelineMode.LINEAR:
                result = self._execute_linear(instruction=instruction, context=context)
            else:
                result = self._execute_dual(instruction=instruction, context=context)

            self._log(
                context,
                f"✅ Aplicação de pipeline concluída para a instrução "
                f"{instruction.instruction_id}."
            )

            return result

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de aplicação de pipeline "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, PipelineApplicationInstruction):
            result.add_error(
                "A instrução informada não é do tipo PipelineApplicationInstruction."
            )
            return result

        if not instruction.trained_pipeline_resource_id:
            result.add_error(
                "A instrução de aplicação de pipeline deve definir "
                "'trained_pipeline_resource_id'."
            )
        elif not context.has_resource(instruction.trained_pipeline_resource_id):
            result.add_error(
                f"O recurso de pipeline treinado não foi encontrado no contexto: "
                f"'{instruction.trained_pipeline_resource_id}'."
            )
        else:
            trained_pipeline_resource = context.get_resource(
                instruction.trained_pipeline_resource_id
            )
            if trained_pipeline_resource is None or not isinstance(
                trained_pipeline_resource.payload, EmbeddingTransformationPipeline
            ):
                result.add_error(
                    f"O recurso '{instruction.trained_pipeline_resource_id}' não contém "
                    "um EmbeddingTransformationPipeline válido."
                )

        if instruction.pipeline_mode == PipelineMode.LINEAR:
            self._validate_linear_instruction(
                instruction=instruction,
                context=context,
                result=result,
            )
        elif instruction.pipeline_mode == PipelineMode.DUAL:
            self._validate_dual_instruction(
                instruction=instruction,
                context=context,
                result=result,
            )
        else:
            result.add_error(f"PipelineMode não suportado: {instruction.pipeline_mode}")

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _validate_linear_instruction(
        self,
        instruction: PipelineApplicationInstruction,
        context: StrategyExecutionContextProtocol,
        result: ValidationResult,
    ) -> None:
        if not instruction.dataset_resource_id:
            result.add_error(
                "Aplicação linear exige 'dataset_resource_id'."
            )
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"O recurso de dataset da aplicação linear não foi encontrado: "
                f"'{instruction.dataset_resource_id}'."
            )

        if not instruction.transformed_dataset_resource_id:
            result.add_error(
                "Aplicação linear exige 'transformed_dataset_resource_id'."
            )

    def _validate_dual_instruction(
        self,
        instruction: PipelineApplicationInstruction,
        context: StrategyExecutionContextProtocol,
        result: ValidationResult,
    ) -> None:
        required_dataset_ids = [
            ("source_dataset_resource_id", instruction.source_dataset_resource_id),
            ("target_dataset_resource_id", instruction.target_dataset_resource_id),
        ]
        for label, resource_id in required_dataset_ids:
            if not resource_id:
                result.add_error(f"Aplicação dual exige '{label}'.")
            elif not context.has_resource(resource_id):
                result.add_error(
                    f"O recurso '{resource_id}' não foi encontrado no contexto."
                )

        if not instruction.transformed_source_dataset_resource_id:
            result.add_error(
                "Aplicação dual exige 'transformed_source_dataset_resource_id'."
            )

        if not instruction.transformed_target_dataset_resource_id:
            result.add_error(
                "Aplicação dual exige 'transformed_target_dataset_resource_id'."
            )

    def _execute_linear(
        self,
        instruction: PipelineApplicationInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        trained_pipeline_resource = context.get_resource(
            instruction.trained_pipeline_resource_id
        )
        dataset_resource = context.get_resource(instruction.dataset_resource_id)

        pipeline = trained_pipeline_resource.payload
        dataset = dataset_resource.payload
        application_policy = (
            instruction.application_policy or PipelineApplicationPolicy.default()
        )

        transformed_dataset = pipeline.transform(
            dataset=dataset,
            application_policy=application_policy,
        )

        output_resource_id = instruction.transformed_dataset_resource_id
        transformed_resource = StrategyResource(
            resource_id=output_resource_id,
            name=output_resource_id,
            resource_type=StrategyResourceType.DATASET,
            payload=transformed_dataset,
            artifact_ref=None,
        )
        context.set_resource(transformed_resource)

        self._log(
            context,
            f"✅ Dataset transformado registrado como recurso: '{output_resource_id}'."
        )

        produced_ids = [output_resource_id]

        # Trajetória 3D opcional — controlada pela política de aplicação.
        trajectory_id = self._maybe_persist_trajectory(
            instruction=instruction,
            context=context,
            pipeline=pipeline,
            source_dataset=dataset,
            target_dataset=None,
            policy=application_policy,
        )
        if trajectory_id:
            produced_ids.append(trajectory_id)

        return InstructionExecutionResult.ok(
            produced_resource_ids=produced_ids,
            metadata={
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "pipeline_mode": instruction.pipeline_mode.to_str(),
                "execution_metrics": pipeline.get_execution_metrics(),
            },
        )

    def _execute_dual(
        self,
        instruction: PipelineApplicationInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        trained_pipeline_resource = context.get_resource(
            instruction.trained_pipeline_resource_id
        )
        source_dataset_resource = context.get_resource(
            instruction.source_dataset_resource_id
        )
        target_dataset_resource = context.get_resource(
            instruction.target_dataset_resource_id
        )

        pipeline = trained_pipeline_resource.payload
        source_dataset = source_dataset_resource.payload
        target_dataset = target_dataset_resource.payload
        application_policy = (
            instruction.application_policy or PipelineApplicationPolicy.default()
        )

        transformed_source_dataset = pipeline.transform_source(
            dataset=source_dataset,
            application_policy=application_policy,
        )
        transformed_target_dataset = pipeline.transform_target(
            dataset=target_dataset,
            application_policy=application_policy,
        )

        source_output_id = instruction.transformed_source_dataset_resource_id
        target_output_id = instruction.transformed_target_dataset_resource_id

        transformed_source_resource = StrategyResource(
            resource_id=source_output_id,
            name=source_output_id,
            resource_type=StrategyResourceType.DATASET,
            payload=transformed_source_dataset,
            artifact_ref=None,
        )
        transformed_target_resource = StrategyResource(
            resource_id=target_output_id,
            name=target_output_id,
            resource_type=StrategyResourceType.DATASET,
            payload=transformed_target_dataset,
            artifact_ref=None,
        )

        context.set_resource(transformed_source_resource)
        context.set_resource(transformed_target_resource)

        self._log(
            context,
            f"✅ Datasets transformados registrados como recursos: "
            f"'{source_output_id}' e '{target_output_id}'."
        )

        produced_ids = [source_output_id, target_output_id]

        # Trajetória 3D opcional — controlada pela política de aplicação.
        # No modo DUAL, computa para ambos os ramos simultaneamente.
        trajectory_id = self._maybe_persist_trajectory(
            instruction=instruction,
            context=context,
            pipeline=pipeline,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            policy=application_policy,
        )
        if trajectory_id:
            produced_ids.append(trajectory_id)

        return InstructionExecutionResult.ok(
            produced_resource_ids=produced_ids,
            metadata={
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "pipeline_mode": instruction.pipeline_mode.to_str(),
                "execution_metrics": pipeline.get_execution_metrics(),
            },
        )

    def _maybe_persist_trajectory(
        self,
        instruction: PipelineApplicationInstruction,
        context: StrategyExecutionContextProtocol,
        pipeline: EmbeddingTransformationPipeline,
        source_dataset,
        target_dataset,
        policy: PipelineApplicationPolicy,
    ):
        """
        Computa a TransformationTrajectoryData via PipelineTrajectoryService
        quando policy.save_trajectory_data=True e registra como StrategyResource
        no contexto. Retorna o resource_id do recurso produzido (ou None).

        A persistência em disco é delegada ao controller (ele itera os
        produced_resource_ids e chama persist_produced_resource para cada).
        """
        if not getattr(policy, "save_trajectory_data", False):
            return None

        # Early-skip: PCA 3D exige n_components <= min(n_samples, n_features).
        # Com poucos pares (caso típico: casework com 1 probe + 1 reference),
        # a redução para 3 dimensões é matematicamente impossível e a
        # visualização não traria informação útil mesmo se possível.
        # Mínimo 4 amostras = 3 (PCA 3D) + 1 de folga numérica.
        MIN_SAMPLES_FOR_3D = 4
        n_total = len(source_dataset) if source_dataset is not None else 0
        if target_dataset is not None:
            n_total += len(target_dataset)
        if n_total < MIN_SAMPLES_FOR_3D:
            self._log(
                context,
                f"ℹ️ Trajetória 3D não computada — apenas {n_total} amostra(s) "
                f"disponíveis (mínimo {MIN_SAMPLES_FOR_3D} para PCA 3D). "
                "Visualização trajetorial não faz sentido em casework de "
                "poucos pares; outputs de dataset transformado registrados "
                "normalmente."
            )
            return None

        trajectory_id = (
            instruction.trajectory_data_resource_id
            or f"{instruction.instruction_id}_trajectory"
        )

        try:
            trajectory_data = PipelineTrajectoryService().compute_trajectory(
                pipeline=pipeline,
                source_dataset=source_dataset,
                target_dataset=target_dataset,
            )
        except Exception as exc:
            # Falha no cálculo da trajetória não deve abortar a aplicação do
            # pipeline — apenas registra o erro e segue.
            self._log(
                context,
                f"⚠️ Falha ao calcular trajetória 3D para "
                f"'{instruction.instruction_id}': {exc}"
            )
            return None

        trajectory_resource = StrategyResource(
            resource_id=trajectory_id,
            name=f"Trajetória 3D ({instruction.name})",
            resource_type=StrategyResourceType.TRAJECTORY_DATA,
            payload=trajectory_data,
            artifact_ref=None,
        )
        context.set_resource(trajectory_resource)

        self._log(
            context,
            f"✅ Trajetória 3D registrada como recurso: '{trajectory_id}' "
            f"({len(trajectory_data.steps)} steps)."
        )

        return trajectory_id