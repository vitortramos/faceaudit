# faceaudit/executors/instructions/typicality_pair_scoring_instruction_executor.py

from __future__ import annotations

from typing import Optional

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.knn_typicality_model_training_instruction_executor import (
    KNNTypicalityModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.instructions.typicality_pair_scoring_instruction import (
    TypicalityPairScoringInstruction,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.knn_typicality_model import KNNTypicalityModel


class TypicalityPairScoringInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Aplica modelo(s) KNNTypicalityModel treinado(s) às faces dos datasets
    de entrada e produz um SimilarityScoreDataset onde cada `raw_score`
    é a combinação das tipicidades individuais (probe e reference).

    Combinadores simétricos (MEAN/MIN/MAX/PRODUCT/ABS_DIFF) preservam
    `LR(A,B) = LR(B,A)`. Combinadores unilaterais (REFERENCE_ONLY,
    PROBE_ONLY) quebram simetria propositalmente e alimentam o
    `SCORE_INTERACTION_FEATURE` downstream — emitem WARNING em
    self-comparação (papéis probe/reference são convencionais nesse
    caso).

    Dois modos de modelo:

    1. Modelo único (mesma população dos dois lados): `model_probe`
       é aplicado em ambos os datasets.

    2. Dois modelos (cross-população): `model_probe` aplicado no probe,
       `model_reference` aplicado no reference. Usado quando probe e
       reference vivem em populações distintas (ex.: cross-pose).

    A ordem de enumeração de pares é IDÊNTICA à de SIMILARITY_SCORING:
    - Self:  triangular superior (i < j).
    - Cross: produto cartesiano (i, j) com i ∈ probe, j ∈ ref.

    Isso garante que o output deste executor possa ser combinado com o
    output de SIMILARITY_SCORING pelo SCORE_VECTORIZATION pair-a-pair.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, TypicalityPairScoringInstruction):
            return self._failure_result(
                "Executor recebeu instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando scoring de tipicidade por par: "
            f"{instruction.name} ({instruction.instruction_id})",
        )

        try:
            embedding_column = self._resolve_embedding_column(context)
            model_probe = self._fetch_model(
                instruction.typicality_model_probe_resource_id,
                context,
                role="probe",
            )

            probe_dataset = self._fetch_dataset(
                instruction.probe_dataset_resource_id, context, "probe",
            )
            reference_dataset = (
                self._fetch_dataset(
                    instruction.reference_dataset_resource_id,
                    context,
                    "reference",
                )
                if instruction.reference_dataset_resource_id
                else None
            )

            # Modelo do reference: usa o próprio se definido, senão o do probe.
            # `effective_ref_model` é o que será aplicado nas faces do
            # reference_dataset.
            model_reference: Optional[KNNTypicalityModel] = None
            if (
                reference_dataset is not None
                and instruction.typicality_model_reference_resource_id
            ):
                model_reference = self._fetch_model(
                    instruction.typicality_model_reference_resource_id,
                    context,
                    role="reference",
                )
                self._log(
                    context,
                    f"🔀 Cross-população: aplicando modelo do reference "
                    f"'{instruction.typicality_model_reference_resource_id}' "
                    f"nas faces do reference_dataset.",
                )

            # Tipicidade individual por face — uma chamada batch por dataset.
            probe_scores = self._compute_per_face_scores(
                model_probe, probe_dataset, embedding_column,
                instruction.score_type,
            )
            self._log(
                context,
                f"📦 Tipicidades probe computadas: N={len(probe_scores)}",
            )

            if reference_dataset is not None:
                effective_ref_model = model_reference or model_probe
                ref_scores = self._compute_per_face_scores(
                    effective_ref_model, reference_dataset, embedding_column,
                    instruction.score_type,
                )
                self._log(
                    context,
                    f"📦 Tipicidades reference computadas: N={len(ref_scores)}",
                )
                result = self._enumerate_cross(
                    probe_dataset, reference_dataset,
                    probe_scores, ref_scores,
                    instruction, context,
                )
            else:
                result = self._enumerate_self(
                    probe_dataset, probe_scores, instruction, context,
                )

            # Modo casework: zera same_person em todos os samples e marca
            # o dataset. Mesmo padrão do SimilarityScoringExecutor pra que
            # downstream (SCORE_VECTORIZATION, ScoreCalibration) trate
            # consistente com o cosseno pareado.
            if instruction.is_casework:
                result.metadata["is_casework"] = True
                for s in result.items:
                    s.same_person = None
                result.__dict__.pop("_cached_n_gen", None)
                result.__dict__.pop("_cached_n_imp", None)
                self._log(
                    context,
                    f"🔍 Modo casework — {len(result.items):,} pares sem "
                    "ground truth.".replace(",", "."),
                )

            output_id = instruction.output_scores_resource_id
            scores_resource = StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
                payload=result,
                artifact_ref=None,
            )
            context.set_resource(scores_resource)

            self._log(
                context,
                f"✅ Scores de tipicidade por par registrados como '{output_id}' "
                f"({len(result.items):,} pares).".replace(",", "."),
            )

            metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "typicality_model_probe_resource_id":
                    instruction.typicality_model_probe_resource_id,
                "typicality_model_reference_resource_id":
                    instruction.typicality_model_reference_resource_id,
                "probe_dataset_resource_id": instruction.probe_dataset_resource_id,
                "reference_dataset_resource_id":
                    instruction.reference_dataset_resource_id,
                "score_type": instruction.score_type.to_str(),
                "combiner": instruction.combiner.to_str(),
                "uses_dual_typicality_models": (model_reference is not None),
                "output_scores_resource_id": output_id,
                "n_pairs": len(result.items),
                "is_casework": instruction.is_casework,
            }

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata=metadata,
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao computar scores de tipicidade "
                f"{instruction.instruction_id}: {e}",
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, TypicalityPairScoringInstruction):
            result.add_error(
                "A instrução informada não é do tipo TypicalityPairScoringInstruction."
            )
            return result

        # Modelo do probe (obrigatório)
        probe_model_id = instruction.typicality_model_probe_resource_id
        if not probe_model_id:
            result.add_error(
                "A instrução deve definir 'typicality_model_probe_resource_id'."
            )
        elif not context.has_resource(probe_model_id):
            result.add_error(
                f"Modelo de tipicidade do probe '{probe_model_id}' "
                f"não encontrado no contexto."
            )

        # Modelo do reference (opcional). Só faz sentido em cross-comparação.
        ref_model_id = instruction.typicality_model_reference_resource_id
        if ref_model_id:
            if not instruction.reference_dataset_resource_id:
                result.add_warning(
                    "'typicality_model_reference_resource_id' está definido mas "
                    "'reference_dataset_resource_id' não — o campo será ignorado "
                    "em auto-comparação."
                )
            elif not context.has_resource(ref_model_id):
                result.add_error(
                    f"Modelo de tipicidade do reference '{ref_model_id}' "
                    f"não encontrado no contexto."
                )

        if not instruction.probe_dataset_resource_id:
            result.add_error("A instrução deve definir 'probe_dataset_resource_id'.")
        elif not context.has_resource(instruction.probe_dataset_resource_id):
            result.add_error(
                f"Dataset probe '{instruction.probe_dataset_resource_id}' "
                f"não encontrado no contexto."
            )

        if (
            instruction.reference_dataset_resource_id
            and not context.has_resource(instruction.reference_dataset_resource_id)
        ):
            result.add_error(
                f"Dataset reference '{instruction.reference_dataset_resource_id}' "
                f"não encontrado no contexto."
            )

        if not instruction.output_scores_resource_id:
            result.add_error("A instrução deve definir 'output_scores_resource_id'.")

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ─────────────────────────── helpers ────────────────────────────

    @staticmethod
    def _resolve_embedding_column(
        context: StrategyExecutionContextProtocol,
    ) -> str:
        """Lê a coluna de embedding da estratégia (via contexto)."""
        col = getattr(context, "embedding_column", None)
        if not col:
            raise ValueError(
                "A estratégia não definiu 'embedding_column'. Configure "
                "a coluna de embedding no nível da estratégia antes de "
                "executar a instrução."
            )
        return str(col)

    @staticmethod
    def _fetch_model(
        resource_id: str,
        context: StrategyExecutionContextProtocol,
        *,
        role: str,
    ) -> KNNTypicalityModel:
        """Resolve um modelo de tipicidade do contexto. `role` é só pra mensagens."""
        res = context.get_resource(resource_id)
        if res is None or res.payload is None:
            raise ValueError(
                f"Recurso de modelo de tipicidade '{resource_id}' "
                f"({role}) ausente ou sem payload."
            )
        payload = res.payload
        if not isinstance(payload, KNNTypicalityModel):
            raise TypeError(
                f"Recurso '{resource_id}' ({role}) não contém um "
                f"KNNTypicalityModel (tem {type(payload).__name__})."
            )
        if not payload.is_fitted():
            raise RuntimeError(
                f"KNNTypicalityModel '{resource_id}' ({role}) não está treinado."
            )
        return payload

    @staticmethod
    def _fetch_dataset(
        resource_id: str,
        context: StrategyExecutionContextProtocol,
        role: str,
    ) -> EmbeddingDataset:
        res = context.get_resource(resource_id)
        payload = res.payload
        if not isinstance(payload, EmbeddingDataset):
            raise TypeError(
                f"Recurso '{resource_id}' ({role}) não contém um "
                f"EmbeddingDataset (tem {type(payload).__name__})."
            )
        return payload

    @staticmethod
    def _compute_per_face_scores(
        model: KNNTypicalityModel,
        dataset: EmbeddingDataset,
        embedding_column: str,
        score_type: TypicalityScoreType,
    ) -> np.ndarray:
        """Vetoriza tipicidade individual de todas as faces do dataset
        aplicando o `model` informado. EmbeddingDataset garante embeddings
        2D via .matrix; embedding_column é validado pela regra
        EMBEDDING_DIM_MATCHES_MODEL."""
        matrix, _ = (
            KNNTypicalityModelTrainingInstructionExecutor
            ._extract_embeddings_and_identities(
                dataset, embedding_column, need_identities=False,
            )
        )

        if score_type is TypicalityScoreType.MEAN_KNN_SIMILARITY:
            infos = model.get_typicality_infos(matrix, include_top_k=False)
            return np.array([i.mean_knn_similarity for i in infos], dtype=np.float64)

        if score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
            # Percentile vem em [0, 100]; normalizo pra [0, 1] pra ficar
            # numa escala parecida com cosseno (melhor pro LogReg sem precisar
            # padronizar).
            return model.get_percentiles(matrix) / 100.0

        if score_type is TypicalityScoreType.LOG_DENSITY_KNN:
            return model.score_samples(matrix)

        raise ValueError(f"score_type não suportado: {score_type}")

    @staticmethod
    def _combine_typicalities(
        combiner: TypicalityCombiner,
        t_probe: float,
        t_reference: float,
    ) -> float:
        """Despacha simétricos para `combiner.combine()` e unilaterais
        para leitura direta do papel correspondente.
        `REFERENCE_ONLY` → tipicidade do REFERENCE; `PROBE_ONLY` → do PROBE.
        """
        if combiner is TypicalityCombiner.REFERENCE_ONLY:
            return float(t_reference)
        if combiner is TypicalityCombiner.PROBE_ONLY:
            return float(t_probe)
        return float(combiner.combine(t_probe, t_reference))

    def _enumerate_self(
        self,
        dataset: EmbeddingDataset,
        face_scores: np.ndarray,
        instruction: TypicalityPairScoringInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> SimilarityScoreDataset:
        name = (
            instruction.output_scores_name
            or f"{dataset.name}_TypicalityPairScores"
        )
        out = SimilarityScoreDataset(
            name=name,
            description=(
                f"Tipicidade pareada (auto-comparação) em {dataset.name}. "
                f"score_type={instruction.score_type.to_str()}, "
                f"combiner={instruction.combiner.to_str()}."
            ),
            metadata={
                "mode": "self_comparison",
                "score_kind": "typicality_pair",
                "score_type": instruction.score_type.to_str(),
                "combiner": instruction.combiner.to_str(),
                "probe_dataset_name": dataset.name,
                "is_casework": instruction.is_casework,
                # PR 2: campos contratuais consumidos pelo
                # validate_state_compatibility da nova instrução
                # LR_ARCHITECTURE_VECTORIZATION. model_fingerprint fica
                # None na v1 (state validate trata None como "sem
                # informação", não falha).
                "model_resource_id": instruction.typicality_model_probe_resource_id,
                "model_fingerprint": None,
            },
        )

        n = len(dataset)
        n_pairs = (n * (n - 1)) // 2
        self._log(
            context,
            f"📐 Auto-pares: N={n}, pares={n_pairs:,}".replace(",", "."),
        )
        if n < 2:
            return out

        combiner = instruction.combiner
        if not combiner.is_symmetric:
            self._log(
                context,
                f"⚠ Combinador unilateral '{combiner.value}' em "
                "self-comparação: papéis probe/reference são "
                "convencionais (triangular superior i<j). Considere "
                "REFERENCE_ONLY/PROBE_ONLY apenas em cross-comparação.",
            )

        progress_every = max(1, n_pairs // 20)
        emitted = 0

        for i in range(n):
            probe_item = dataset[i]
            t_i = float(face_scores[i])
            for j in range(i + 1, n):
                reference_item = dataset[j]
                t_j = float(face_scores[j])
                raw_score = self._combine_typicalities(combiner, t_i, t_j)

                # same_person é inferido por __post_init__ a partir das
                # identidades; modo casework é forçado abaixo após o loop.
                out.add(
                    SimilarityScoreSample(
                        probe=probe_item,
                        reference=reference_item,
                        raw_score=raw_score,
                        metadata={
                            "probe_typicality": t_i,
                            "reference_typicality": t_j,
                        },
                    )
                )
                emitted += 1
                if emitted % progress_every == 0:
                    pct = 100.0 * emitted / n_pairs
                    self._log(
                        context,
                        f"  … {emitted:,}/{n_pairs:,} pares ({pct:.0f}%)".replace(",", "."),
                    )

        return out

    def _enumerate_cross(
        self,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
        probe_scores: np.ndarray,
        ref_scores: np.ndarray,
        instruction: TypicalityPairScoringInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> SimilarityScoreDataset:
        name = (
            instruction.output_scores_name
            or f"{probe_dataset.name}_{reference_dataset.name}_TypicalityPairScores"
        )
        out = SimilarityScoreDataset(
            name=name,
            description=(
                f"Tipicidade pareada (cross-comparação) entre "
                f"{probe_dataset.name} e {reference_dataset.name}. "
                f"score_type={instruction.score_type.to_str()}, "
                f"combiner={instruction.combiner.to_str()}."
            ),
            metadata={
                "mode": "cross_comparison",
                "score_kind": "typicality_pair",
                "score_type": instruction.score_type.to_str(),
                "combiner": instruction.combiner.to_str(),
                "probe_dataset_name": probe_dataset.name,
                "reference_dataset_name": reference_dataset.name,
                "uses_dual_typicality_models": bool(
                    instruction.typicality_model_reference_resource_id
                ),
                "is_casework": instruction.is_casework,
                # PR 2: campos contratuais para a nova instrução
                # LR_ARCHITECTURE_VECTORIZATION. Em cross, o modelo da
                # raridade do lado consumido é determinado pelo combiner
                # (PROBE_ONLY → probe; REFERENCE_ONLY → reference).
                "model_resource_id": (
                    instruction.typicality_model_reference_resource_id
                    if instruction.combiner.to_str() == "REFERENCE_ONLY"
                    else instruction.typicality_model_probe_resource_id
                ),
                "model_fingerprint": None,
            },
        )

        n_probe = len(probe_dataset)
        n_ref = len(reference_dataset)
        n_pairs = n_probe * n_ref
        self._log(
            context,
            f"📐 Cross-pares: probe={n_probe}, ref={n_ref}, "
            f"pares={n_pairs:,}".replace(",", "."),
        )
        if n_probe == 0 or n_ref == 0:
            return out

        combiner = instruction.combiner
        progress_every = max(1, n_pairs // 20)
        emitted = 0

        for i, probe_item in enumerate(probe_dataset):
            t_i = float(probe_scores[i])
            for j, reference_item in enumerate(reference_dataset):
                t_j = float(ref_scores[j])
                raw_score = self._combine_typicalities(combiner, t_i, t_j)

                # same_person é inferido por __post_init__ a partir das
                # identidades; modo casework é forçado abaixo após o loop.
                out.add(
                    SimilarityScoreSample(
                        probe=probe_item,
                        reference=reference_item,
                        raw_score=raw_score,
                        metadata={
                            "probe_typicality": t_i,
                            "reference_typicality": t_j,
                        },
                    )
                )
                emitted += 1
                if emitted % progress_every == 0:
                    pct = 100.0 * emitted / n_pairs
                    self._log(
                        context,
                        f"  … {emitted:,}/{n_pairs:,} pares ({pct:.0f}%)".replace(",", "."),
                    )

        return out
