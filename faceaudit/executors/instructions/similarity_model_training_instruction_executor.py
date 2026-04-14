# faceaudit/executors/instructions/similarity_model_training_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.similarity_model_training_instruction import (
    SimilarityModelTrainingInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.cosine_similarity_model import CosineSimilarityModel
from faceaudit.models.similarity.plda_similarity_model import PLDASimilarityModel
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class SimilarityModelTrainingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de treinamento/preparação de modelo de similaridade.

    Famílias atualmente suportadas:
    - COSINE
    - PLDA
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, SimilarityModelTrainingInstruction):
            return self._failure_result(
                "O executor SimilarityModelTrainingInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de treino de modelo de similaridade: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            dataset_resource = context.get_resource(instruction.input_dataset_resource_id)
            dataset = dataset_resource.payload

            if instruction.uses_cosine:
                trained_model = self._prepare_cosine_similarity_model(
                    dataset=dataset,
                    context=context,
                )

            elif instruction.uses_plda:
                config_resource = context.get_resource(
                    instruction.similarity_model_config_resource_id
                )
                config_payload = config_resource.payload

                trained_model = self._train_plda_similarity_model(
                    dataset=dataset,
                    config_payload=config_payload,
                    context=context,
                )

            else:
                return self._failure_result(
                    f"Tipo de modelo de similaridade não suportado: "
                    f"{instruction.similarity_model_type}"
                )

            output_resource_id = instruction.trained_similarity_model_resource_id
            trained_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.TRAINED_SIMILARITY_MODEL,
                payload=trained_model,
                artifact_ref=None,
            )
            context.set_resource(trained_resource)

            self._log(
                context,
                f"✅ Modelo de similaridade registrado como recurso: "
                f"'{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "similarity_model_type": instruction.similarity_model_type.to_str(),
                    "input_dataset_resource_id": instruction.input_dataset_resource_id,
                    "similarity_model_config_resource_id": instruction.similarity_model_config_resource_id,
                    "trained_similarity_model_resource_id": output_resource_id,
                    "trained_model_type": type(trained_model).__name__,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de treino de modelo de similaridade "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, SimilarityModelTrainingInstruction):
            result.add_error(
                "A instrução informada não é do tipo SimilarityModelTrainingInstruction."
            )
            return result

        if not instruction.input_dataset_resource_id:
            result.add_error(
                "A instrução deve definir 'input_dataset_resource_id'."
            )
        elif not context.has_resource(instruction.input_dataset_resource_id):
            result.add_error(
                f"O recurso de entrada não foi encontrado no contexto: "
                f"'{instruction.input_dataset_resource_id}'."
            )
        else:
            dataset_resource = context.get_resource(instruction.input_dataset_resource_id)
            if dataset_resource is None or not isinstance(
                dataset_resource.payload, EmbeddingDataset
            ):
                result.add_error(
                    f"O recurso '{instruction.input_dataset_resource_id}' não contém "
                    "um EmbeddingDataset válido."
                )

        if not instruction.trained_similarity_model_resource_id:
            result.add_error(
                "A instrução deve definir 'trained_similarity_model_resource_id'."
            )

        if (
            instruction.trained_similarity_model_resource_id
            and instruction.trained_similarity_model_resource_id
            == instruction.input_dataset_resource_id
        ):
            result.add_warning(
                "trained_similarity_model_resource_id coincide com o recurso de dataset "
                "de entrada. O contexto irá sobrescrever esse recurso."
            )

        if instruction.uses_cosine:
            if instruction.similarity_model_config_resource_id and not context.has_resource(
                instruction.similarity_model_config_resource_id
            ):
                result.add_error(
                    f"O recurso de configuração opcional de cosine não foi encontrado "
                    f"no contexto: '{instruction.similarity_model_config_resource_id}'."
                )

        elif instruction.uses_plda:
            if not instruction.similarity_model_config_resource_id:
                result.add_error(
                    "PLDA exige 'similarity_model_config_resource_id'."
                )
            elif not context.has_resource(instruction.similarity_model_config_resource_id):
                result.add_error(
                    f"O recurso de configuração do modelo PLDA não foi encontrado no "
                    f"contexto: '{instruction.similarity_model_config_resource_id}'."
                )
            else:
                config_resource = context.get_resource(
                    instruction.similarity_model_config_resource_id
                )
                if config_resource is None or not isinstance(
                    config_resource.payload, PLDASimilarityModelConfig
                ):
                    result.add_error(
                        f"O recurso '{instruction.similarity_model_config_resource_id}' "
                        "não contém um PLDASimilarityModelConfig válido."
                    )

            if (
                instruction.trained_similarity_model_resource_id
                and instruction.trained_similarity_model_resource_id
                == instruction.similarity_model_config_resource_id
            ):
                result.add_warning(
                    "trained_similarity_model_resource_id coincide com o recurso de "
                    "configuração do modelo. O contexto irá sobrescrever esse recurso."
                )

        else:
            result.add_error(
                f"Tipo de modelo de similaridade não suportado: "
                f"{instruction.similarity_model_type}"
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _prepare_cosine_similarity_model(
        self,
        dataset: EmbeddingDataset,
        context: StrategyExecutionContextProtocol,
    ) -> CosineSimilarityModel:
        """
        Instancia e prepara um CosineSimilarityModel, encaminhando os logs
        para o contexto de execução.
        """
        model = CosineSimilarityModel()
        model.fit(dataset=dataset, logger_callback=context.log)
        return model

    def _train_plda_similarity_model(
        self,
        dataset: EmbeddingDataset,
        config_payload: PLDASimilarityModelConfig,
        context: StrategyExecutionContextProtocol,
    ) -> PLDASimilarityModel:
        """
        Instancia e treina um PLDASimilarityModel a partir do config informado,
        encaminhando os logs para o contexto de execução.
        """
        model = PLDASimilarityModel(config=config_payload)
        model.fit(dataset=dataset, logger_callback=context.log)
        return model