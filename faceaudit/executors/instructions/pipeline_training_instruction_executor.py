# faceaudit/executors/instructions/pipeline_training_instruction_executor.py

from __future__ import annotations

from typing import Optional

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType

from faceaudit.executors.instructions.strategy_instruction_executor_base import StrategyInstructionExecutorBase
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.pipeline_training_instruction import (
    PipelineTrainingInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy
from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.pipelines.embedding_transformation_pipeline import (
    EmbeddingTransformationPipeline,
)
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)




class PipelineTrainingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de treinamento de pipeline.

    Fluxo:
    1. Valida a instrução e os recursos necessários.
    2. Resolve configs e datasets no contexto.
    3. Instancia o EmbeddingTransformationPipeline.
    4. Executa fit() ou fit_dual().
    5. Registra o pipeline treinado como novo recurso no contexto.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, PipelineTrainingInstruction):
            return self._failure_result(
                "O executor PipelineTrainingInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de treino de pipeline: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            if instruction.pipeline_mode == PipelineMode.LINEAR:
                pipeline = self._execute_linear(instruction=instruction, context=context)
            else:
                pipeline = self._execute_dual(instruction=instruction, context=context)

            output_resource_id = instruction.trained_pipeline_resource_id
            trained_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.TRAINED_PIPELINE,
                payload=pipeline,
                artifact_ref=None,
            )
            context.set_resource(trained_resource)

            self._log(
                context,
                f"✅ Pipeline treinado registrado como recurso: '{output_resource_id}'."
            )

            produced_ids = [output_resource_id]

            if instruction.apply_to_training_datasets:
                produced_ids.extend(
                    self._apply_to_training_datasets(
                        instruction=instruction,
                        context=context,
                        pipeline=pipeline,
                    )
                )

            return InstructionExecutionResult.ok(
                produced_resource_ids=produced_ids,
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "pipeline_mode": instruction.pipeline_mode.to_str(),
                    "training_metrics": pipeline.get_training_metrics(),
                    "applied_to_training_datasets": bool(
                        instruction.apply_to_training_datasets
                    ),
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de treino de pipeline "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, PipelineTrainingInstruction):
            result.add_error(
                "A instrução informada não é do tipo PipelineTrainingInstruction."
            )
            return result

        if not instruction.trained_pipeline_resource_id:
            result.add_error(
                "A instrução de treino de pipeline deve definir "
                "'trained_pipeline_resource_id'."
            )

        if instruction.apply_to_training_datasets:
            if instruction.pipeline_mode == PipelineMode.LINEAR:
                if not instruction.output_transformed_dataset_resource_id:
                    result.add_error(
                        "Com 'apply_to_training_datasets' marcado, o pipeline "
                        "linear exige 'output_transformed_dataset_resource_id'."
                    )
            elif instruction.pipeline_mode == PipelineMode.DUAL:
                if not instruction.output_transformed_source_dataset_resource_id:
                    result.add_error(
                        "Com 'apply_to_training_datasets' marcado, o pipeline "
                        "dual exige 'output_transformed_source_dataset_resource_id'."
                    )
                if not instruction.output_transformed_target_dataset_resource_id:
                    result.add_error(
                        "Com 'apply_to_training_datasets' marcado, o pipeline "
                        "dual exige 'output_transformed_target_dataset_resource_id'."
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
        instruction: PipelineTrainingInstruction,
        context: StrategyExecutionContextProtocol,
        result: ValidationResult,
    ) -> None:
        if not instruction.pipeline_config_resource_id:
            result.add_error(
                "Pipeline linear exige 'pipeline_config_resource_id'."
            )
        elif not context.has_resource(instruction.pipeline_config_resource_id):
            result.add_error(
                f"Recurso de config do pipeline linear não encontrado no contexto: "
                f"'{instruction.pipeline_config_resource_id}'."
            )
        else:
            config_resource = context.get_resource(instruction.pipeline_config_resource_id)
            if config_resource is None or not isinstance(
                config_resource.payload, TransformationPipelineConfig
            ):
                result.add_error(
                    f"O recurso '{instruction.pipeline_config_resource_id}' não contém "
                    "um TransformationPipelineConfig válido."
                )

        if not instruction.dataset_resource_id:
            result.add_error(
                "Pipeline linear exige 'dataset_resource_id'."
            )
        elif not context.has_resource(instruction.dataset_resource_id):
            result.add_error(
                f"Recurso de dataset do treino linear não encontrado no contexto: "
                f"'{instruction.dataset_resource_id}'."
            )

    def _validate_dual_instruction(
        self,
        instruction: PipelineTrainingInstruction,
        context: StrategyExecutionContextProtocol,
        result: ValidationResult,
    ) -> None:
        required_config_ids = [
            ("source_pipeline_config_resource_id", instruction.source_pipeline_config_resource_id),
            ("target_pipeline_config_resource_id", instruction.target_pipeline_config_resource_id),
        ]
        for label, resource_id in required_config_ids:
            if not resource_id:
                result.add_error(f"Pipeline dual exige '{label}'.")
            elif not context.has_resource(resource_id):
                result.add_error(
                    f"Recurso '{resource_id}' não encontrado no contexto."
                )
            else:
                resource = context.get_resource(resource_id)
                if resource is None or not isinstance(
                    resource.payload, TransformationPipelineConfig
                ):
                    result.add_error(
                        f"O recurso '{resource_id}' não contém um "
                        "TransformationPipelineConfig válido."
                    )

        required_dataset_ids = [
            ("source_dataset_resource_id", instruction.source_dataset_resource_id),
            ("target_dataset_resource_id", instruction.target_dataset_resource_id),
        ]
        for label, resource_id in required_dataset_ids:
            if not resource_id:
                result.add_error(f"Pipeline dual exige '{label}'.")
            elif not context.has_resource(resource_id):
                result.add_error(
                    f"Recurso '{resource_id}' não encontrado no contexto."
                )

    def _execute_linear(
        self,
        instruction: PipelineTrainingInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> EmbeddingTransformationPipeline:
        config_resource = context.get_resource(instruction.pipeline_config_resource_id)
        dataset_resource = context.get_resource(instruction.dataset_resource_id)

        config = config_resource.payload
        dataset = dataset_resource.payload

        training_policy = instruction.training_policy or PipelineTrainingPolicy.default_linear()

        pipeline = EmbeddingTransformationPipeline(
            mode=PipelineMode.LINEAR,
            config=config,
        )
        pipeline.fit(dataset=dataset, training_policy=training_policy)

        return pipeline

    def _execute_dual(
        self,
        instruction: PipelineTrainingInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> EmbeddingTransformationPipeline:
        source_config_resource = context.get_resource(
            instruction.source_pipeline_config_resource_id
        )
        target_config_resource = context.get_resource(
            instruction.target_pipeline_config_resource_id
        )
        source_dataset_resource = context.get_resource(
            instruction.source_dataset_resource_id
        )
        target_dataset_resource = context.get_resource(
            instruction.target_dataset_resource_id
        )

        source_config = source_config_resource.payload
        target_config = target_config_resource.payload
        source_dataset = source_dataset_resource.payload
        target_dataset = target_dataset_resource.payload

        training_policy = instruction.training_policy or PipelineTrainingPolicy.default_dual()

        pipeline = EmbeddingTransformationPipeline(
            mode=PipelineMode.DUAL,
            source_config=source_config,
            target_config=target_config,
        )
        pipeline.fit_dual(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            training_policy=training_policy,
        )

        return pipeline

    def _apply_to_training_datasets(
        self,
        instruction: PipelineTrainingInstruction,
        context: StrategyExecutionContextProtocol,
        pipeline: EmbeddingTransformationPipeline,
    ) -> list:
        """
        Aplica o pipeline recém-treinado nos datasets de treino e registra os
        recursos transformados no contexto. Retorna a lista de resource_ids
        produzidos para serem incluídos em produced_resource_ids.

        Reusa a mesma lógica de PIPELINE_APPLICATION (transform / transform_source /
        transform_target). Política de aplicação é a default — sem checkpoints
        nem trajetória, já que o usuário pode rodar PIPELINE_APPLICATION
        explícito depois se quiser esses artefatos.
        """
        application_policy = PipelineApplicationPolicy.default()
        produced_ids: list = []

        if instruction.pipeline_mode == PipelineMode.LINEAR:
            dataset = context.get_resource(instruction.dataset_resource_id).payload

            self._log(
                context,
                f"[{instruction.name}] 🔄 Aplicando pipeline ao dataset de treino…"
            )
            transformed = pipeline.transform(
                dataset=dataset,
                application_policy=application_policy,
            )

            output_id = instruction.output_transformed_dataset_resource_id
            context.set_resource(StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.DATASET,
                payload=transformed,
                artifact_ref=None,
            ))
            produced_ids.append(output_id)
            self._log(
                context,
                f"✅ Dataset transformado registrado como recurso: '{output_id}'."
            )

        else:  # DUAL
            source_dataset = context.get_resource(
                instruction.source_dataset_resource_id
            ).payload
            target_dataset = context.get_resource(
                instruction.target_dataset_resource_id
            ).payload

            self._log(
                context,
                f"[{instruction.name}] 🔄 Aplicando pipeline aos datasets Source/Target…"
            )
            transformed_source = pipeline.transform_source(
                dataset=source_dataset,
                application_policy=application_policy,
            )
            transformed_target = pipeline.transform_target(
                dataset=target_dataset,
                application_policy=application_policy,
            )

            source_id = instruction.output_transformed_source_dataset_resource_id
            target_id = instruction.output_transformed_target_dataset_resource_id

            context.set_resource(StrategyResource(
                resource_id=source_id,
                name=source_id,
                resource_type=StrategyResourceType.DATASET,
                payload=transformed_source,
                artifact_ref=None,
            ))
            context.set_resource(StrategyResource(
                resource_id=target_id,
                name=target_id,
                resource_type=StrategyResourceType.DATASET,
                payload=transformed_target,
                artifact_ref=None,
            ))
            produced_ids.extend([source_id, target_id])
            self._log(
                context,
                f"✅ Datasets transformados registrados como recursos: "
                f"'{source_id}' e '{target_id}'."
            )

        return produced_ids