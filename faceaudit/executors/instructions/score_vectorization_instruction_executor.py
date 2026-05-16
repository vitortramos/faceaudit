# faceaudit/executors/instructions/score_vectorization_instruction_executor.py

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.score_vectorization_instruction import (
    ScoreVectorizationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class ScoreVectorizationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução SCORE_VECTORIZATION.

    Recebe N (≥2) SimilarityScoreDataset com os mesmos pares e produz um
    novo dataset com o vetor de scores persistido como **uma única matriz
    `(N_pairs, D)` contígua** em `dataset.metadata["score_vector_matrix"]`
    — uma alocação pra todo o dataset em vez de N alocações de
    `List[float]` por sample (relevante pra datasets com milhões de pares:
    GC pressure baixa, zero-copy pro sklearn). O `raw_score` do output é
    o do primeiro input (preserva utilidade em mining e casework_viz que
    esperam score escalar 1D).

    O CalibrationModelTrainingInstructionExecutor detecta
    `is_score_vector=True` no dataset.metadata e extrai features
    multivariadas pra treinar o calibrador.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, ScoreVectorizationInstruction):
            return self._failure_result(
                "Instrução incompatível com ScoreVectorizationInstructionExecutor."
            )

        self._log(
            context,
            f"▶ Iniciando SCORE_VECTORIZATION: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            input_ids = instruction.all_input_resource_ids()
            datasets = [context.get_resource(rid).payload for rid in input_ids]

            vectorized_dataset, n_pairs, n_features = self._build_vectorized_dataset(
                datasets=datasets,
                source_ids=input_ids,
                require_pair_match=instruction.require_pair_match,
                output_name=(
                    instruction.output_dataset_name
                    or instruction.output_resource_id
                ),
                context=context,
            )

            output_id = instruction.output_resource_id
            output_resource = StrategyResource(
                resource_id=output_id,
                name=output_id,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
                payload=vectorized_dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ Score vectorization produzido: '{output_id}' "
                f"(n_pairs={n_pairs}, dim_features={n_features})"
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_resource_ids": input_ids,
                    "output_resource_id": output_id,
                    "n_pairs": int(n_pairs),
                    "n_features": int(n_features),
                    "is_score_vector": True,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro em SCORE_VECTORIZATION ({instruction.instruction_id}): {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, ScoreVectorizationInstruction):
            result.add_error(
                "A instrução informada não é do tipo ScoreVectorizationInstruction."
            )
            return result

        # A e B continuam canônicos (obrigatórios). Extras são opcionais.
        for field_name, rid in (
            ("input_a_resource_id", instruction.input_a_resource_id),
            ("input_b_resource_id", instruction.input_b_resource_id),
        ):
            if not rid:
                result.add_error(f"A instrução deve definir '{field_name}'.")
            elif not context.has_resource(rid):
                result.add_error(
                    f"O recurso de entrada não foi encontrado: '{rid}'."
                )

        for rid in instruction.extra_input_resource_ids:
            if not rid:
                continue  # entrada vazia ignorada (whitespace na CSV da UI)
            if not context.has_resource(rid):
                result.add_error(
                    f"Recurso extra não encontrado no contexto: '{rid}'."
                )

        # Duplicatas e vetorização "consigo mesmo" não fazem sentido.
        all_inputs = instruction.all_input_resource_ids()
        if len(all_inputs) >= 2:
            seen: set = set()
            for rid in all_inputs:
                if rid in seen:
                    result.add_error(
                        f"Input '{rid}' aparece mais de uma vez — score "
                        "vectorization exige sistemas distintos."
                    )
                seen.add(rid)
        if instruction.input_a_resource_id and instruction.input_a_resource_id == instruction.input_b_resource_id:
            result.add_error(
                "input_a_resource_id e input_b_resource_id não podem ser o "
                "mesmo recurso (score vectorization exige dois sistemas distintos)."
            )

        if not instruction.output_resource_id:
            result.add_error(
                "A instrução deve definir 'output_resource_id'."
            )

        # Verifica que TODOS os payloads são SimilarityScoreDataset.
        for rid in all_inputs:
            if rid and context.has_resource(rid):
                payload = context.get_resource(rid).payload
                if payload is not None and not isinstance(payload, SimilarityScoreDataset):
                    result.add_error(
                        f"Recurso '{rid}' não é um SimilarityScoreDataset "
                        f"(got {type(payload).__name__})."
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ─── Núcleo da vetorização ──────────────────────────────────────────────

    def _build_vectorized_dataset(
        self,
        datasets: List[SimilarityScoreDataset],
        source_ids: List[str],
        require_pair_match: bool,
        output_name: str,
        context: StrategyExecutionContextProtocol,
    ) -> Tuple[SimilarityScoreDataset, int, int]:
        """
        Constrói o SimilarityScoreDataset vetorizado a partir de
        **N (≥2) inputs**.

        Algoritmo:
        1. O primeiro dataset (`datasets[0]`) é o **anchor** — define a ordem
           dos pares e doa `probe/reference/same_person/raw_score`.
        2. Cada dataset subsequente é alinhado contra o anchor:
           - `require_pair_match=True`: exige mesma quantidade e mesma
             sequência de `(probe.face_hash, reference.face_hash)`. Falha
             com mensagem mostrando primeiras divergências.
           - `require_pair_match=False`: reordena via hash → sample dict.
        3. Concatena as colunas de scores num matrix `(N_pairs, D)` em
           `metadata["score_vector_matrix"]`. D = len(datasets).

        Cada sample do output é a instância original de `datasets[0]` (sem
        mutação), preservando metadata por sample. O vetor vive 100% no
        `dataset.metadata` — nada em `sample.metadata`.
        """
        if len(datasets) < 2 or len(source_ids) < 2:
            raise ValueError(
                f"Score vectorization requer pelo menos 2 inputs "
                f"(got {len(datasets)})."
            )
        if len(datasets) != len(source_ids):
            raise ValueError(
                f"datasets ({len(datasets)}) e source_ids ({len(source_ids)}) "
                "devem ter o mesmo tamanho."
            )

        sizes = [len(ds.items) for ds in datasets]
        if any(s == 0 for s in sizes):
            raise ValueError(
                f"Score vectorization requer todos os datasets não vazios "
                f"(sizes={sizes})."
            )

        anchor = datasets[0]
        anchor_id = source_ids[0]
        n = sizes[0]

        # Alinha cada input != anchor contra ele.
        aligned_items_per_input: List[List[SimilarityScoreSample]] = [anchor.items]
        for other, other_id in zip(datasets[1:], source_ids[1:]):
            if require_pair_match:
                if len(other.items) != n:
                    raise ValueError(
                        f"Dataset '{other_id}' tem {len(other.items)} pares, "
                        f"anchor '{anchor_id}' tem {n} — defina "
                        "require_pair_match=False para reordenar por hash."
                    )
                self._validate_strict_pair_alignment(
                    anchor=anchor, other=other,
                    anchor_id=anchor_id, other_id=other_id,
                )
                aligned_items_per_input.append(other.items)
            else:
                aligned_items_per_input.append(
                    self._reorder_to_match_anchor(
                        anchor=anchor, other=other,
                        anchor_id=anchor_id, other_id=other_id,
                    )
                )

        # Extrai os vetores de score em UMA passada por dataset, sem mutar
        # nenhum sample. Cada coluna é uma alocação numpy de tamanho N — D
        # alocações totais em vez de N×D floats Python em metadata.
        score_columns = [
            np.fromiter(
                (float(s.raw_score) for s in items),
                dtype=np.float64, count=n,
            )
            for items in aligned_items_per_input
        ]
        score_vector_matrix = np.column_stack(score_columns)

        # Reusa as instâncias originais do anchor — preserva
        # same_person, identidades, raw_score e metadata individuais.
        vectorized_items: List[SimilarityScoreSample] = list(anchor.items)

        dataset_metadata = dict(anchor.metadata) if anchor.metadata else {}
        dataset_metadata["is_score_vector"] = True
        dataset_metadata["score_vector_source_resources"] = list(source_ids)
        dataset_metadata["score_vector_n_features"] = int(score_vector_matrix.shape[1])
        dataset_metadata[
            SimilarityScoreDataset.SCORE_VECTOR_MATRIX_KEY
        ] = score_vector_matrix

        sources_str = " + ".join(f"'{sid}'" for sid in source_ids)
        vectorized = SimilarityScoreDataset(
            name=output_name,
            description=(
                f"Score vectorization de {sources_str} ({len(source_ids)} sistemas). "
                f"Vetor de scores em metadata['score_vector_matrix'] "
                f"(numpy {score_vector_matrix.shape}); raw_score = score "
                f"do anchor ('{anchor_id}')."
            ),
            items=vectorized_items,
            metadata=dataset_metadata,
        )
        return vectorized, n, int(score_vector_matrix.shape[1])

    def _validate_strict_pair_alignment(
        self,
        anchor: SimilarityScoreDataset,
        other: SimilarityScoreDataset,
        anchor_id: str,
        other_id: str,
    ) -> None:
        """Verifica que `other.items` tem a mesma sequência de pares
        `(probe.face_hash, reference.face_hash)` que `anchor.items`.

        Quando difere, levanta ValueError mostrando as primeiras 5
        posições divergentes com hashes truncados — diagnóstico útil
        em vez da mensagem genérica anterior.
        """
        mismatches: List[Tuple[int, Tuple[str, str], Tuple[str, str]]] = []
        MAX_MISMATCH_EXAMPLES = 5
        for i, (sa, so) in enumerate(zip(anchor.items, other.items)):
            key_a = self._pair_key(sa)
            key_o = self._pair_key(so)
            if key_a != key_o:
                mismatches.append((i, key_a, key_o))
                if len(mismatches) >= MAX_MISMATCH_EXAMPLES:
                    break
        if mismatches:
            rendered = "; ".join(
                f"pos {i}: anchor={a[0][:8]}..→{a[1][:8]}.. vs "
                f"other={o[0][:8]}..→{o[1][:8]}.."
                for i, a, o in mismatches
            )
            raise ValueError(
                f"Pares não batem entre '{anchor_id}' (anchor) e "
                f"'{other_id}' em posições alinhadas (primeiros "
                f"{len(mismatches)} divergentes — {rendered}). "
                "Defina require_pair_match=False para reordenar por hash, "
                "ou verifique que ambos os sistemas processaram o mesmo "
                "conjunto de pares na mesma ordem."
            )

    @staticmethod
    def _pair_key(sample: SimilarityScoreSample) -> Tuple[str, str]:
        """Identifica unicamente um par (probe, reference) por face_hash."""
        probe_hash = getattr(sample.probe, "face_hash", None) or ""
        ref_hash = getattr(sample.reference, "face_hash", None) or ""
        return (str(probe_hash), str(ref_hash))

    def _reorder_to_match_anchor(
        self,
        anchor: SimilarityScoreDataset,
        other: SimilarityScoreDataset,
        anchor_id: str,
        other_id: str,
    ) -> List[SimilarityScoreSample]:
        """Reordena `other` para casar com `anchor` por
        `(probe.face_hash, reference.face_hash)`.

        Falha se algum par do anchor não estiver em `other`. `other`
        pode ter pares extras — são silenciosamente descartados.
        """
        index_other = {self._pair_key(s): s for s in other.items}
        aligned: List[SimilarityScoreSample] = []
        missing: List[Tuple[str, str]] = []
        for sa in anchor.items:
            key = self._pair_key(sa)
            sb = index_other.get(key)
            if sb is None:
                missing.append(key)
                if len(missing) > 5:
                    break
            else:
                aligned.append(sb)
        if missing:
            sample_str = ", ".join(
                f"{m[0][:8]}..→{m[1][:8]}.." for m in missing[:5]
            )
            raise ValueError(
                f"{len(missing)}+ pares do anchor '{anchor_id}' não estão "
                f"em '{other_id}' (ex: {sample_str}). Verifique que ambos "
                "os sistemas processaram o mesmo conjunto de pares."
            )
        return aligned
