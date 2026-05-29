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
            if instruction.uses_cosine:
                # Cosine é stateless — não exige dataset.
                trained_model = self._prepare_cosine_similarity_model(
                    dataset=None,
                    context=context,
                )

            elif instruction.uses_plda:
                dataset_resource = context.get_resource(instruction.input_dataset_resource_id)
                dataset = dataset_resource.payload

                # Modo cross-condition: segundo dataset opcional.
                secondary_dataset = None
                secondary_id = instruction.secondary_dataset_resource_id
                if secondary_id:
                    secondary_resource = context.get_resource(secondary_id)
                    secondary_dataset = secondary_resource.payload
                    # Provenance warning — DATASETS_EMBEDDING_COMPAT garante
                    # dim/coluna iguais mas não que os datasets vieram do
                    # mesmo pipeline. Hash de pipeline é trabalho futuro.
                    self._log(
                        context,
                        "⚠ PLDA cross-condition: assume que '"
                        f"{instruction.input_dataset_resource_id}' e "
                        f"'{secondary_id}' foram produzidos pela MESMA cadeia de "
                        "transformação (mesmo pipeline DUAL). Espaços vetoriais "
                        "incompatíveis (CORAL/Whitening fitados separadamente) "
                        "produzirão resultados sem sentido sem disparar erro "
                        "automático — verifique manualmente."
                    )

                # Config embutida no DTO da instrução (não mais via recurso externo).
                trained_model = self._train_plda_similarity_model(
                    dataset=dataset,
                    config_payload=instruction.similarity_model_config,
                    context=context,
                    secondary_dataset=secondary_dataset,
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

            execution_metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "similarity_model_type": instruction.similarity_model_type.to_str(),
                "input_dataset_resource_id": instruction.input_dataset_resource_id,
                "secondary_dataset_resource_id": instruction.secondary_dataset_resource_id,
                "trained_similarity_model_resource_id": output_resource_id,
                "trained_model_type": type(trained_model).__name__,
            }

            # Flags cross-condition para auditoria/UI.
            if (
                instruction.uses_plda
                and instruction.secondary_dataset_resource_id
                and isinstance(trained_model, PLDASimilarityModel)
                and trained_model.two_cov_state is not None
            ):
                state = trained_model.two_cov_state
                execution_metadata.update({
                    "cross_condition": getattr(state, "cross_condition", False),
                    "n_ids_condition_a": getattr(state, "n_ids_condition_a", None),
                    "n_ids_condition_b": getattr(state, "n_ids_condition_b", None),
                    "n_common_identities": getattr(state, "n_common_identities", None),
                    "cross_condition_within_mode": getattr(
                        state, "cross_condition_within_mode", None
                    ),
                })

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata=execution_metadata,
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

        # Dataset só é obrigatório para PLDA. Cosine é stateless — não exige
        # treino e portanto não precisa de dataset de entrada.
        if instruction.uses_plda:
            if not instruction.input_dataset_resource_id:
                result.add_error(
                    "PLDA exige 'input_dataset_resource_id'."
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
            # Cosine não tem hyperparâmetros — nada a validar.
            pass

        elif instruction.uses_plda:
            # Config embutida no DTO da instrução.
            if instruction.similarity_model_config is None:
                result.add_error(
                    "PLDA exige um 'similarity_model_config' embutido na instrução."
                )
            elif not isinstance(
                instruction.similarity_model_config, PLDASimilarityModelConfig
            ):
                result.add_error(
                    "O 'similarity_model_config' embutido não é um "
                    "PLDASimilarityModelConfig válido."
                )
            else:
                # Validações do modo cross-condition.
                secondary_id = instruction.secondary_dataset_resource_id
                if secondary_id:
                    config = instruction.similarity_model_config
                    if not config.uses_any_two_covariance_variant:
                        result.add_error(
                            "secondary_dataset_resource_id (modo cross-condition) "
                            "requer variante PLDA TWO_COVARIANCE ou "
                            "SHRINKAGE_TWO_COVARIANCE — variante atual: "
                            f"{config.plda_variant.value}."
                        )
                    elif not context.has_resource(secondary_id):
                        result.add_error(
                            f"O dataset secundário não foi encontrado no contexto: "
                            f"'{secondary_id}'."
                        )
                    else:
                        secondary_resource = context.get_resource(secondary_id)
                        if secondary_resource is None or not isinstance(
                            secondary_resource.payload, EmbeddingDataset
                        ):
                            result.add_error(
                                f"O dataset secundário '{secondary_id}' não contém "
                                "um EmbeddingDataset válido."
                            )
                        elif (
                            instruction.input_dataset_resource_id
                            and context.has_resource(instruction.input_dataset_resource_id)
                        ):
                            # Sanidade: dimensão de embedding compatível
                            # entre primário e secundário (defesa em
                            # profundidade — a regra plugável DATASETS_EMBEDDING_COMPAT
                            # já verifica, mas mantemos hard-check aqui).
                            primary_resource = context.get_resource(
                                instruction.input_dataset_resource_id
                            )
                            if (
                                primary_resource is not None
                                and isinstance(primary_resource.payload, EmbeddingDataset)
                            ):
                                primary_dim = (
                                    primary_resource.payload.matrix.shape[1]
                                    if len(primary_resource.payload) > 0 else None
                                )
                                secondary_dim = (
                                    secondary_resource.payload.matrix.shape[1]
                                    if len(secondary_resource.payload) > 0 else None
                                )
                                if (
                                    primary_dim is not None
                                    and secondary_dim is not None
                                    and primary_dim != secondary_dim
                                ):
                                    result.add_error(
                                        f"Dimensões de embedding incompatíveis: "
                                        f"primário D={primary_dim} vs "
                                        f"secundário D={secondary_dim}."
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
        dataset,
        context: StrategyExecutionContextProtocol,
    ) -> CosineSimilarityModel:
        """
        Instancia o CosineSimilarityModel. Cosine é stateless — não exige
        dataset nem fit. Se um dataset é fornecido, fit() registra metadados
        opcionais (input_dimension, n_samples_seen) para rastreabilidade.
        """
        model = CosineSimilarityModel()
        if dataset is not None and len(dataset) > 0:
            try:
                model.fit(dataset=dataset, logger_callback=context.log)
            except Exception:
                # Metadados são best-effort — falha no fit não invalida o modelo.
                pass
        else:
            context.log(f"[{model.name}] ✅ Modelo cosseno pronto (stateless).")
        return model

    def _train_plda_similarity_model(
        self,
        dataset: EmbeddingDataset,
        config_payload: PLDASimilarityModelConfig,
        context: StrategyExecutionContextProtocol,
        secondary_dataset: EmbeddingDataset = None,
    ) -> PLDASimilarityModel:
        """
        Instancia e treina um PLDASimilarityModel a partir do config informado,
        encaminhando os logs para o contexto de execução.

        Quando `secondary_dataset` é fornecido, dispara o estimador
        cross-condition (modo `BALANCED_IDENTITY` por default — configurável
        via `config.cross_condition_within_mode`).
        """
        model = PLDASimilarityModel(config=config_payload)
        model.fit(
            dataset=dataset,
            logger_callback=context.log,
            secondary_dataset=secondary_dataset,
        )
        return model