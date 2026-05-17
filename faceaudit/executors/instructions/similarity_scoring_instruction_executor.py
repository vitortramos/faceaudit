# faceaudit/executors/instructions/similarity_scoring_instruction_executor.py

from __future__ import annotations

from typing import Optional

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

            # Rastreabilidade do modelo de similaridade que produziu
            # estes scores. Consumido por `SCORE_VECTORIZATION` pra
            # registrar a procedência ordenada de cada coluna, e
            # eventualmente pela regra de consistência que detecta
            # ordem de sistemas trocada entre treino do calibrador
            # multivariado e a aplicação.
            score_dataset.metadata["similarity_model_id"] = (
                instruction.trained_similarity_model_resource_id
            )

            # Modo casework: marca o dataset como sem ground truth e zera
            # `same_person` em todos os samples (sobrescreve qualquer
            # inferência feita pelo __post_init__ a partir das identidades
            # convencionais como "questionada"/"padrão").
            if instruction.is_casework:
                score_dataset.metadata["is_casework"] = True
                for s in score_dataset.items:
                    s.same_person = None
                # Invalida cache de contagens (já zeradas, mas seguro).
                score_dataset.__dict__.pop("_cached_n_gen", None)
                score_dataset.__dict__.pop("_cached_n_imp", None)
                self._log(
                    context,
                    f"🔍 Modo casework — {len(score_dataset.items):,} pares "
                    "sem ground truth gerados.".replace(",", ".")
                )

            output_resource_id = instruction.output_similarity_scores_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
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
                    "is_casework": bool(instruction.is_casework),
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
            if model_resource is None:
                result.add_error(
                    f"O recurso '{instruction.trained_similarity_model_resource_id}' "
                    "está registrado mas é nulo no contexto."
                )
            elif model_resource.payload is None:
                # Distinguir o caso "load do disco falhou" do caso "tipo errado".
                ref_path = (
                    model_resource.artifact_ref.path
                    if model_resource.artifact_ref is not None
                    else None
                )
                if ref_path:
                    result.add_error(
                        f"O recurso '{instruction.trained_similarity_model_resource_id}' "
                        f"tem artifact_ref ('{ref_path}') mas o payload está vazio — "
                        "falha ao desserializar o modelo do disco. "
                        "Re-execute a instrução SIMILARITY_MODEL_TRAINING que produz este "
                        "modelo (ou veja o log do console para detalhes da exceção)."
                    )
                else:
                    result.add_error(
                        f"O recurso '{instruction.trained_similarity_model_resource_id}' "
                        "ainda não foi produzido. Execute a instrução "
                        "SIMILARITY_MODEL_TRAINING que produz este modelo antes."
                    )
            elif not isinstance(model_resource.payload, SimilarityModelProtocol):
                result.add_error(
                    f"O recurso '{instruction.trained_similarity_model_resource_id}' "
                    f"contém um payload do tipo {type(model_resource.payload).__name__}, "
                    "que não satisfaz SimilarityModelProtocol."
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

        # ── Compatibilidade de dimensões: modelo vs datasets ───────────────
        # Evita o erro "operands could not be broadcast together with shapes
        # (N, dim_dataset) (dim_modelo,)" que ocorre lá dentro do score_matrix
        # quando o usuário aplica um PLDA treinado em 512D sobre um dataset
        # de 256D (ou vice-versa). Falha cedo aqui com mensagem explícita.
        model_dim = self._get_model_input_dimension(instruction, context)
        if model_dim is not None:
            for ds_field, ds_id in (
                ("probe_dataset_resource_id", instruction.probe_dataset_resource_id),
                ("reference_dataset_resource_id", instruction.reference_dataset_resource_id),
            ):
                if not ds_id or not context.has_resource(ds_id):
                    continue
                ds_resource = context.get_resource(ds_id)
                if ds_resource is None or not isinstance(
                    ds_resource.payload, EmbeddingDataset
                ):
                    continue
                ds_dim = self._get_dataset_dim(ds_resource.payload)
                if ds_dim is not None and ds_dim != model_dim:
                    result.add_error(
                        f"Incompatibilidade de dimensões: o modelo de "
                        f"similaridade foi treinado com embeddings de "
                        f"dimensão {model_dim}, mas o recurso '{ds_id}' "
                        f"({ds_field}) contém embeddings de dimensão "
                        f"{ds_dim}. Aplique antes um pipeline que reduza/"
                        f"projete o dataset para {model_dim} dimensões, "
                        f"ou treine o modelo de similaridade com embeddings "
                        f"da dimensão correta."
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
        n_pairs = (n * (n - 1)) // 2
        self._log(
            context,
            f"📐 Auto-comparação: N={n}, pares={n_pairs:,}".replace(",", "."),
        )

        if n < 2:
            return result

        # Vetoriza o cálculo via score_matrix em vez de N² chamadas a score_pair —
        # ganho de ordens de grandeza para PLDA/Cosine.
        self._log(context, "🔢 Computando matriz de scores (vetorizado)…")
        self._check_dim_compatibility(model, dataset)
        score_mat = model.score_matrix(dataset.matrix, dataset.matrix)
        self._log(context, f"✅ Matriz {score_mat.shape} pronta. Empacotando pares…")

        progress_every = max(1, n_pairs // 20)  # ~5% steps
        emitted = 0
        for i in range(n):
            probe = dataset[i]
            for j in range(i + 1, n):
                reference = dataset[j]
                raw_score = float(score_mat[i, j])

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
                emitted += 1
                if emitted % progress_every == 0:
                    pct = 100.0 * emitted / n_pairs
                    self._log(
                        context,
                        f"  … {emitted:,}/{n_pairs:,} pares ({pct:.0f}%)".replace(",", "."),
                    )

        self._log(context, f"✅ {emitted:,} pares empacotados.".replace(",", "."))
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
        n_pairs = n_probe * n_ref
        self._log(
            context,
            f"📐 Comparação cruzada: Probe={n_probe}, Reference={n_ref}, "
            f"pares={n_pairs:,}".replace(",", "."),
        )

        if n_probe == 0 or n_ref == 0:
            return result

        # Vetoriza via score_matrix em vez de N_probe × N_ref chamadas a score_pair.
        self._log(context, "🔢 Computando matriz de scores (vetorizado)…")
        # Validação de defesa em runtime: se a dim do dataset não bate com a
        # do modelo, falha aqui com mensagem clara em vez de propagar o
        # cryptic "operands could not be broadcast together" do numpy.
        self._check_dim_compatibility(model, probe_dataset, reference_dataset)
        score_mat = model.score_matrix(probe_dataset.matrix, reference_dataset.matrix)
        self._log(context, f"✅ Matriz {score_mat.shape} pronta. Empacotando pares…")

        progress_every = max(1, n_pairs // 20)
        emitted = 0
        for i, probe in enumerate(probe_dataset):
            for j, reference in enumerate(reference_dataset):
                raw_score = float(score_mat[i, j])

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
                emitted += 1
                if emitted % progress_every == 0:
                    pct = 100.0 * emitted / n_pairs
                    self._log(
                        context,
                        f"  … {emitted:,}/{n_pairs:,} pares ({pct:.0f}%)".replace(",", "."),
                    )

        self._log(context, f"✅ {emitted:,} pares empacotados.".replace(",", "."))
        return result
    # =========================================================================
    # HELPERS DE COMPATIBILIDADE DE DIMENSÕES
    # =========================================================================

    @staticmethod
    def _get_model_input_dimension(
        instruction: SimilarityScoringInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> Optional[int]:
        """
        Lê a dimensão de entrada do modelo de similaridade. Tenta primeiro a
        property unificada `input_dimension` (Cosine + PLDA), com fallback
        para `trained_dim` (atributo legado do PLDA).
        """
        rid = instruction.trained_similarity_model_resource_id
        if not rid or not context.has_resource(rid):
            return None
        resource = context.get_resource(rid)
        if resource is None or resource.payload is None:
            return None
        model = resource.payload
        dim = getattr(model, "input_dimension", None)
        if dim is None:
            dim = getattr(model, "trained_dim", None)
        return int(dim) if dim is not None else None

    @staticmethod
    def _get_dataset_dim(dataset: EmbeddingDataset) -> Optional[int]:
        """
        Retorna a dimensão dos embeddings do dataset sem materializar a
        matriz inteira (que `dataset.matrix` aloca). Pega a forma do
        primeiro item; assume homogeneidade entre samples (pré-condição
        para qualquer scoring vetorizado).
        """
        items = getattr(dataset, "items", None) or []
        if not items:
            return None
        emb = getattr(items[0], "embedding", None)
        if emb is None:
            return None
        try:
            return int(emb.shape[0])
        except Exception:
            return None

    def _check_dim_compatibility(
        self,
        model,
        *datasets: EmbeddingDataset,
    ) -> None:
        """
        Levanta ValueError com mensagem clara se a dimensão dos datasets
        não bate com a dim de entrada do modelo. Usado como defesa em
        runtime no execute() — a validação semântica ANTES da execução
        já é feita em _validate_instruction_specifics, mas esta camada
        protege também os caminhos onde a UI não passa pela validação
        (ex: re-execução forçada com Executar habilitado sempre).
        """
        model_dim = getattr(model, "input_dimension", None)
        if model_dim is None:
            model_dim = getattr(model, "trained_dim", None)
        if model_dim is None:
            return
        for ds in datasets:
            ds_dim = self._get_dataset_dim(ds)
            if ds_dim is None or ds_dim == model_dim:
                continue
            raise ValueError(
                f"Incompatibilidade de dimensões: modelo treinado com {model_dim} "
                f"dimensões, mas dataset '{getattr(ds, 'name', '?')}' tem "
                f"embeddings de {ds_dim} dimensões."
            )
