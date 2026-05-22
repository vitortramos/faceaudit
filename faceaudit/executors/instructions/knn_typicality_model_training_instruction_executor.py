# faceaudit/executors/instructions/knn_typicality_model_training_instruction_executor.py

from __future__ import annotations

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.knn_typicality_model_training_instruction import (
    KNNTypicalityModelTrainingInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.knn_typicality_model import KNNTypicalityModel


class KNNTypicalityModelTrainingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução KNN_TYPICALITY_MODEL_TRAINING.

    Lê o dataset de referência da subpopulação (já estratificada pelo usuário),
    extrai a matriz de embeddings da coluna indicada, e ajusta um
    KNNTypicalityModel. O recurso de saída carrega o modelo treinado (que
    inclui a referência inteira + CDF empírica LOO de T_k).
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, KNNTypicalityModelTrainingInstruction):
            return self._failure_result(
                "Executor recebeu instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando treino de modelo de tipicidade (kNN): "
            f"{instruction.name} ({instruction.instruction_id})",
        )

        try:
            input_resource = context.get_resource(instruction.input_dataset_resource_id)
            payload = input_resource.payload

            embeddings = self._extract_embeddings(
                payload, instruction.embedding_column
            )

            self._log(
                context,
                f"📦 Referência: N={embeddings.shape[0]}, D={embeddings.shape[1]}, "
                f"k={instruction.k}.",
            )

            model = KNNTypicalityModel(
                k=instruction.k,
                ood_quantile=instruction.ood_quantile,
            )
            model.fit(embeddings, population_label=instruction.population_label)

            summary = model.summary()
            self._log(
                context,
                f"✅ Modelo treinado. T_k mediano LOO={summary['loo_T_k_median']:.4f}, "
                f"threshold OOD={summary['ood_threshold']:.4f}.",
            )

            output_resource_id = instruction.trained_typicality_model_resource_id
            trained_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.TRAINED_TYPICALITY_MODEL,
                payload=model,
                artifact_ref=None,
            )
            context.set_resource(trained_resource)

            metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "input_dataset_resource_id": instruction.input_dataset_resource_id,
                "embedding_column": instruction.embedding_column,
                "k": instruction.k,
                "ood_quantile": instruction.ood_quantile,
                "population_label": instruction.population_label,
                "trained_typicality_model_resource_id": output_resource_id,
                "n_reference": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "ood_threshold": float(summary["ood_threshold"]),
                "loo_T_k_median": float(summary["loo_T_k_median"]),
                "loo_T_k_min": float(summary["loo_T_k_min"]),
                "loo_T_k_max": float(summary["loo_T_k_max"]),
            }

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata=metadata,
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao treinar modelo de tipicidade "
                f"{instruction.instruction_id}: {e}",
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, KNNTypicalityModelTrainingInstruction):
            result.add_error(
                "A instrução informada não é do tipo "
                "KNNTypicalityModelTrainingInstruction."
            )
            return result

        if not instruction.input_dataset_resource_id:
            result.add_error("A instrução deve definir 'input_dataset_resource_id'.")
        elif not context.has_resource(instruction.input_dataset_resource_id):
            result.add_error(
                f"Recurso de dataset '{instruction.input_dataset_resource_id}' "
                f"não encontrado no contexto."
            )
        else:
            ds_resource = context.get_resource(instruction.input_dataset_resource_id)
            if ds_resource is None or ds_resource.payload is None:
                result.add_error(
                    "Falha ao recuperar o dataset (payload nulo). "
                    "Verifique se o artefato existe em disco e foi hidratado."
                )
            else:
                try:
                    self._extract_embeddings(
                        ds_resource.payload, instruction.embedding_column
                    )
                except Exception as e:
                    result.add_error(
                        f"Coluna de embedding '{instruction.embedding_column}' "
                        f"inválida no dataset: {e}"
                    )

        if instruction.k < 1:
            result.add_error("'k' deve ser >= 1.")
        if not (0.0 <= instruction.ood_quantile < 0.5):
            result.add_error("'ood_quantile' deve estar em [0, 0.5).")

        if not instruction.trained_typicality_model_resource_id:
            result.add_error(
                "A instrução deve definir 'trained_typicality_model_resource_id'."
            )

        if (
            instruction.trained_typicality_model_resource_id
            and instruction.trained_typicality_model_resource_id
            == instruction.input_dataset_resource_id
        ):
            result.add_warning(
                "trained_typicality_model_resource_id coincide com o recurso "
                "de entrada — o contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    # ─────────────────────────── helpers ────────────────────────────

    @staticmethod
    def _extract_embeddings(payload, embedding_column: str) -> np.ndarray:
        """
        Extrai matriz (N, D) de embeddings do payload do dataset.

        Aceita:
        - EmbeddingDataset: usa o `matrix` property (vstack dos items).
        - FaceDataset com a coluna requerida: tenta converter via
          embedding_column lookup nos items.
        """
        if payload is None:
            raise ValueError("Payload do dataset é None.")

        # Caminho rápido: EmbeddingDataset já tem matriz pronta.
        if isinstance(payload, EmbeddingDataset):
            m = payload.matrix
            if m.size == 0:
                raise ValueError("EmbeddingDataset vazio.")
            if m.ndim != 2:
                raise ValueError(f"matrix shape inesperada: {m.shape}")
            return m

        # Fallback: dataset genérico (FaceDataset) com a coluna requerida.
        items = getattr(payload, "items", None)
        if items is None:
            raise ValueError(
                f"Payload {type(payload).__name__} não expõe 'items' nem 'matrix'."
            )

        vectors = []
        for it in items:
            v = getattr(it, embedding_column, None)
            if v is None:
                ff = getattr(it, "ff", None)
                if ff is not None:
                    v = getattr(ff, embedding_column, None)
            if v is None:
                raise ValueError(
                    f"Item sem atributo/coluna '{embedding_column}'."
                )
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.size == 0:
                raise ValueError(f"Embedding vazio em item.")
            vectors.append(arr)

        if not vectors:
            raise ValueError("Dataset não tem items.")

        m = np.vstack(vectors)
        return m
