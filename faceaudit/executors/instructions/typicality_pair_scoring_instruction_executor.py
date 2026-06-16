# faceaudit/executors/instructions/typicality_pair_scoring_instruction_executor.py

from __future__ import annotations

from typing import Optional

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.typicality_model_training_instruction_executor import (
    TypicalityModelTrainingInstructionExecutor,
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
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.protocols.typicality_scoring_model_protocol import (
    TypicalityScoringModelProtocol,
    UnsupportedTypicalityScoreTypeError,
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
            model_reference: Optional[TypicalityScoringModelProtocol] = None
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
            probe_batch = self._compute_per_face_batch(
                model_probe, probe_dataset, embedding_column,
                instruction.score_type,
            )
            probe_scores = self._scores_for_pairing(probe_batch)
            self._log(
                context,
                f"📦 Tipicidades probe computadas: N={probe_batch.n} "
                f"(semantics={probe_batch.score_semantics_id})",
            )

            ref_batch: Optional[TypicalityScoreBatch] = None
            if reference_dataset is not None:
                effective_ref_model = model_reference or model_probe
                ref_batch = self._compute_per_face_batch(
                    effective_ref_model, reference_dataset, embedding_column,
                    instruction.score_type,
                )
                ref_scores = self._scores_for_pairing(ref_batch)
                self._log(
                    context,
                    f"📦 Tipicidades reference computadas: N={ref_batch.n}",
                )
                result = self._enumerate_cross(
                    probe_dataset, reference_dataset,
                    probe_scores, ref_scores,
                    instruction, context,
                    probe_batch=probe_batch, ref_batch=ref_batch,
                )
            else:
                result = self._enumerate_self(
                    probe_dataset, probe_scores, instruction, context,
                    probe_batch=probe_batch,
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
    ) -> TypicalityScoringModelProtocol:
        """Resolve um modelo de tipicidade do contexto via protocolo
        polimórfico. `role` é só para mensagens.

        Validações runtime explícitas (PR 3): isinstance via
        `runtime_checkable` é sanity check; pegamos também o
        `is_fitted` quando expostas pelo backend concreto. O
        `compute_scores` em si confirma compatibilidade do score_type.
        """
        res = context.get_resource(resource_id)
        if res is None or res.payload is None:
            raise ValueError(
                f"Recurso de modelo de tipicidade '{resource_id}' "
                f"({role}) ausente ou sem payload."
            )
        payload = res.payload
        if not isinstance(payload, TypicalityScoringModelProtocol):
            raise TypeError(
                f"Recurso '{resource_id}' ({role}) não satisfaz "
                f"TypicalityScoringModelProtocol (tem "
                f"{type(payload).__name__}). Verifique que o recurso "
                f"foi produzido por TYPICALITY_MODEL_TRAINING."
            )
        # is_fitted() é da API legada dos backends concretos; usa quando
        # disponível.
        is_fitted_method = getattr(payload, "is_fitted", None)
        if callable(is_fitted_method) and not is_fitted_method():
            raise RuntimeError(
                f"Modelo de tipicidade '{resource_id}' ({role}) não "
                "está treinado."
            )
        # enabled_score_types deve estar populado pós-fit.
        if not payload.enabled_score_types:
            raise RuntimeError(
                f"Modelo '{resource_id}' ({role}) tem "
                "enabled_score_types vazio — não foi treinado ou ficou "
                "em estado inconsistente."
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
    def _compute_per_face_batch(
        model: TypicalityScoringModelProtocol,
        dataset: EmbeddingDataset,
        embedding_column: str,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch:
        """Vetoriza tipicidade individual via `compute_scores(...)` →
        `TypicalityScoreBatch` tipado (PR 3).

        Validações runtime explícitas (do plano):
        - `score_type ∈ enabled_score_types` → senão
          `UnsupportedTypicalityScoreTypeError`
        - `batch.score_type == score_type`
        - `batch.n == n_faces`
        - `np.all(isfinite(batch.scores))`
        """
        matrix, _ = (
            TypicalityModelTrainingInstructionExecutor
            ._extract_embeddings_and_identities(
                dataset, embedding_column, need_identities=False,
            )
        )

        if score_type not in model.enabled_score_types:
            raise UnsupportedTypicalityScoreTypeError(
                requested=score_type,
                enabled=model.enabled_score_types,
                model_type=model.model_type,
            )

        batch = model.compute_scores(matrix, score_type)

        # Validações runtime no consumidor (sanity beyond runtime_checkable)
        if batch.score_type != score_type:
            raise RuntimeError(
                f"Modelo retornou batch com score_type={batch.score_type} "
                f"diferente do solicitado {score_type}."
            )
        if batch.n != int(matrix.shape[0]):
            raise RuntimeError(
                f"Modelo retornou batch n={batch.n} divergente do "
                f"esperado N={matrix.shape[0]}."
            )
        if not np.all(np.isfinite(batch.scores)):
            raise RuntimeError(
                "Batch contém scores não-finitos — verifique o modelo."
            )
        return batch

    @staticmethod
    def _scores_for_pairing(batch: TypicalityScoreBatch) -> np.ndarray:
        """Aplica a transformação de escala do pair scoring.

        `TYPICALITY_PERCENTILE` vem em [0, 100] no batch (escala canônica
        do contrato); o pair scoring historicamente usa [0, 1] para ficar
        comparável com cosseno e amigável ao LogReg sem padronização.
        Os demais score_types passam direto.
        """
        if batch.score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
            return batch.scores / 100.0
        return batch.scores

    @staticmethod
    def _build_provenance_metadata(
        *,
        instruction: TypicalityPairScoringInstruction,
        probe_batch: TypicalityScoreBatch,
        ref_batch: Optional[TypicalityScoreBatch],
    ) -> dict:
        """Empacota a proveniência completa para o metadata do dataset.

        Plano (PR 3): "Persistir no metadata do dataset:
        - source_typicality_model_resource_id
        - source_typicality_model_contract_fingerprint
        - source_typicality_model_state_fingerprint
        - typicality_score_type
        - typicality_score_semantics_id
        - typicality_percentile_basis_score_type
        - typicality_reference_score_distribution_fingerprint"

        Quando há dois batches (cross + dual models), persiste os dois
        sufixados por _probe / _reference.
        """
        prov: dict = {
            # Identidade do score (vem do batch — modelo decidiu)
            "typicality_score_type": probe_batch.score_type.to_str(),
            "typicality_score_semantics_id": probe_batch.score_semantics_id,
            "typicality_percentile_basis_score_type": (
                probe_batch.percentile_basis_score_type.to_str()
                if probe_batch.percentile_basis_score_type else None
            ),
            "typicality_reference_score_distribution_fingerprint": (
                probe_batch.reference_score_distribution_fingerprint
            ),
            # Proveniência do modelo (probe)
            "source_typicality_model_resource_id":
                instruction.typicality_model_probe_resource_id,
            "source_typicality_model_type":
                probe_batch.model_type.to_str(),
            "source_typicality_model_contract_fingerprint":
                probe_batch.model_contract_fingerprint,
            "source_typicality_model_state_fingerprint":
                probe_batch.model_state_fingerprint,
        }
        if ref_batch is not None:
            prov.update({
                "source_typicality_model_reference_resource_id":
                    instruction.typicality_model_reference_resource_id,
                "source_typicality_model_reference_type":
                    ref_batch.model_type.to_str(),
                "source_typicality_model_reference_contract_fingerprint":
                    ref_batch.model_contract_fingerprint,
                "source_typicality_model_reference_state_fingerprint":
                    ref_batch.model_state_fingerprint,
            })
        return prov

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
        *,
        probe_batch: TypicalityScoreBatch,
    ) -> SimilarityScoreDataset:
        name = (
            instruction.output_scores_name
            or f"{dataset.name}_TypicalityPairScores"
        )
        # PR 3: proveniência completa no metadata do dataset (não em
        # cada item) — vide plano "metadata no dataset, não em cada item".
        provenance = self._build_provenance_metadata(
            instruction=instruction,
            probe_batch=probe_batch,
            ref_batch=None,
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
                # Campo contratual consumido pelo validate_state_compatibility
                # da instrução LR_ARCHITECTURE_VECTORIZATION.
                "model_resource_id": instruction.typicality_model_probe_resource_id,
                # PR 3: agora preenchido a partir do batch (contract fp).
                "model_fingerprint": (
                    probe_batch.model_contract_fingerprint or None
                ),
                **provenance,
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
        *,
        probe_batch: TypicalityScoreBatch,
        ref_batch: TypicalityScoreBatch,
    ) -> SimilarityScoreDataset:
        name = (
            instruction.output_scores_name
            or f"{probe_dataset.name}_{reference_dataset.name}_TypicalityPairScores"
        )
        provenance = self._build_provenance_metadata(
            instruction=instruction,
            probe_batch=probe_batch,
            ref_batch=ref_batch,
        )
        # Em cross, o modelo da raridade do lado consumido é determinado
        # pelo combiner (PROBE_ONLY → probe; REFERENCE_ONLY → reference).
        consumed_side_is_reference = (
            instruction.combiner.to_str() == "REFERENCE_ONLY"
        )
        consumed_model_resource_id = (
            instruction.typicality_model_reference_resource_id
            if consumed_side_is_reference
            else instruction.typicality_model_probe_resource_id
        )
        consumed_batch = ref_batch if consumed_side_is_reference else probe_batch

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
                "model_resource_id": consumed_model_resource_id,
                "model_fingerprint": (
                    consumed_batch.model_contract_fingerprint or None
                ),
                **provenance,
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
