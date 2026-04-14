# faceaudit/executors/instructions/similarity_scoring_instruction_executor.py

from __future__ import annotations

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.similarity_scoring_instruction import (
    SimilarityScoringInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.similarity_model_protocol import SimilarityModelProtocol
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class SimilarityScoringInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de geração de scores de similaridade.

    Modos suportados:
    - auto-comparação dentro de um único EmbeddingDataset
      (triângulo superior, sem diagonal)
    - comparação cruzada entre probe_dataset e reference_dataset

    O resultado sempre é um SimilarityScoreDataset.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, SimilarityScoringInstruction):
            return self._failure_result(
                "O executor SimilarityScoringInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de scoring de similaridade: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            model_resource = context.get_resource(
                instruction.trained_similarity_model_resource_id
            )
            similarity_model = model_resource.payload

            probe_resource = context.get_resource(instruction.probe_dataset_resource_id)
            probe_dataset = probe_resource.payload

            if instruction.uses_self_comparison:
                score_dataset = self._score_self_comparison(
                    model=similarity_model,
                    dataset=probe_dataset,
                    instruction=instruction,
                    context=context,
                )
            elif instruction.uses_cross_comparison:
                reference_resource = context.get_resource(
                    instruction.reference_dataset_resource_id
                )
                reference_dataset = reference_resource.payload

                score_dataset = self._score_cross_comparison(
                    model=similarity_model,
                    probe_dataset=probe_dataset,
                    reference_dataset=reference_dataset,
                    instruction=instruction,
                    context=context,
                )
            else:
                return self._failure_result(
                    "A instrução de similarity scoring deve definir probe_dataset_resource_id "
                    "e opcionalmente reference_dataset_resource_id."
                )

            output_resource_id = instruction.output_similarity_scores_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.INTERMEDIATE_DATASET,
                payload=score_dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ SimilarityScoreDataset registrado como recurso: '{output_resource_id}'."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "trained_similarity_model_resource_id": instruction.trained_similarity_model_resource_id,
                    "probe_dataset_resource_id": instruction.probe_dataset_resource_id,
                    "reference_dataset_resource_id": instruction.reference_dataset_resource_id,
                    "output_similarity_scores_resource_id": output_resource_id,
                    "n_samples": len(score_dataset),
                    "n_genuine": score_dataset.total_genuines,
                    "n_impostor": score_dataset.total_impostors,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de scoring de similaridade "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, SimilarityScoringInstruction):
            result.add_error(
                "A instrução informada não é do tipo SimilarityScoringInstruction."
            )
            return result

        if not instruction.trained_similarity_model_resource_id:
            result.add_error(
                "A instrução deve definir 'trained_similarity_model_resource_id'."
            )
        elif not context.has_resource(instruction.trained_similarity_model_resource_id):
            result.add_error(
                f"O recurso de modelo de similaridade não foi encontrado no contexto: "
                f"'{instruction.trained_similarity_model_resource_id}'."
            )
        else:
            model_resource = context.get_resource(
                instruction.trained_similarity_model_resource_id
            )
            if model_resource is None or not isinstance(
                model_resource.payload, SimilarityModelProtocol
            ):
                result.add_error(
                    f"O recurso '{instruction.trained_similarity_model_resource_id}' "
                    "não contém um SimilarityModelProtocol válido."
                )

        if not instruction.probe_dataset_resource_id:
            result.add_error(
                "A instrução deve definir 'probe_dataset_resource_id'."
            )
        elif not context.has_resource(instruction.probe_dataset_resource_id):
            result.add_error(
                f"O recurso de probe dataset não foi encontrado no contexto: "
                f"'{instruction.probe_dataset_resource_id}'."
            )
        else:
            probe_resource = context.get_resource(instruction.probe_dataset_resource_id)
            if probe_resource is None or not isinstance(
                probe_resource.payload, EmbeddingDataset
            ):
                result.add_error(
                    f"O recurso '{instruction.probe_dataset_resource_id}' não contém "
                    "um EmbeddingDataset válido."
                )

        if instruction.reference_dataset_resource_id:
            if not context.has_resource(instruction.reference_dataset_resource_id):
                result.add_error(
                    f"O recurso de reference dataset não foi encontrado no contexto: "
                    f"'{instruction.reference_dataset_resource_id}'."
                )
            else:
                reference_resource = context.get_resource(
                    instruction.reference_dataset_resource_id
                )
                if reference_resource is None or not isinstance(
                    reference_resource.payload, EmbeddingDataset
                ):
                    result.add_error(
                        f"O recurso '{instruction.reference_dataset_resource_id}' "
                        "não contém um EmbeddingDataset válido."
                    )

        if not instruction.output_similarity_scores_resource_id:
            result.add_error(
                "A instrução deve definir 'output_similarity_scores_resource_id'."
            )

        if (
            instruction.output_similarity_scores_resource_id
            and instruction.output_similarity_scores_resource_id
            == instruction.trained_similarity_model_resource_id
        ):
            result.add_warning(
                "output_similarity_scores_resource_id coincide com o recurso do modelo. "
                "O contexto irá sobrescrever esse recurso."
            )

        if (
            instruction.output_similarity_scores_resource_id
            and instruction.output_similarity_scores_resource_id
            == instruction.probe_dataset_resource_id
        ):
            result.add_warning(
                "output_similarity_scores_resource_id coincide com o probe dataset. "
                "O contexto irá sobrescrever esse recurso."
            )

        if (
            instruction.reference_dataset_resource_id
            and instruction.output_similarity_scores_resource_id
            == instruction.reference_dataset_resource_id
        ):
            result.add_warning(
                "output_similarity_scores_resource_id coincide com o reference dataset. "
                "O contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _score_self_comparison(
        self,
        model: SimilarityModelProtocol,
        dataset: EmbeddingDataset,
        instruction: SimilarityScoringInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> SimilarityScoreDataset:
        output_name = (
            instruction.output_similarity_scores_name
            or f"{dataset.name}_SimilarityScores"
        )

        result = SimilarityScoreDataset(
            name=output_name,
            description=f"Auto-comparação de similaridade para {dataset.name}",
            metadata={
                "mode": "self_comparison",
                "probe_dataset_name": dataset.name,
                "similarity_model_name": getattr(model, "name", type(model).__name__),
            },
        )

        n = len(dataset)
        self._log(
            context,
            f"📐 Gerando scores por auto-comparação "
            f"(N={n}, pares={(n * (n - 1)) // 2})..."
        )

        for i in range(n):
            probe = dataset[i]
            for j in range(i + 1, n):
                reference = dataset[j]
                raw_score = float(model.score_pair(probe, reference))

                metadata = {}
                if instruction.store_metadata_per_sample:
                    metadata = {
                        "probe_index": i,
                        "reference_index": j,
                    }

                result.add(
                    SimilarityScoreSample(
                        probe=probe,
                        reference=reference,
                        raw_score=raw_score,
                        metadata=metadata,
                    )
                )

        return result

    def _score_cross_comparison(
        self,
        model: SimilarityModelProtocol,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
        instruction: SimilarityScoringInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> SimilarityScoreDataset:
        output_name = (
            instruction.output_similarity_scores_name
            or f"{probe_dataset.name}_{reference_dataset.name}_SimilarityScores"
        )

        result = SimilarityScoreDataset(
            name=output_name,
            description=(
                f"Comparação cruzada de similaridade entre "
                f"{probe_dataset.name} e {reference_dataset.name}"
            ),
            metadata={
                "mode": "cross_comparison",
                "probe_dataset_name": probe_dataset.name,
                "reference_dataset_name": reference_dataset.name,
                "similarity_model_name": getattr(model, "name", type(model).__name__),
            },
        )

        n_probe = len(probe_dataset)
        n_ref = len(reference_dataset)
        self._log(
            context,
            f"📐 Gerando scores por comparação cruzada "
            f"(Probe={n_probe}, Reference={n_ref}, pares={n_probe * n_ref})..."
        )

        for i, probe in enumerate(probe_dataset):
            for j, reference in enumerate(reference_dataset):
                raw_score = float(model.score_pair(probe, reference))

                metadata = {}
                if instruction.store_metadata_per_sample:
                    metadata = {
                        "probe_index": i,
                        "reference_index": j,
                    }

                result.add(
                    SimilarityScoreSample(
                        probe=probe,
                        reference=reference,
                        raw_score=raw_score,
                        metadata=metadata,
                    )
                )

        return result