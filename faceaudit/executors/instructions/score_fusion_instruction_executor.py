# faceaudit/executors/instructions/score_fusion_instruction_executor.py

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
from faceaudit.models.instructions.score_fusion_instruction import (
    ScoreFusionInstruction,
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


class ScoreFusionInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor da instrução SCORE_FUSION.

    Recebe dois SimilarityScoreDataset com os mesmos pares e produz um terceiro
    onde cada amostra carrega um vetor de scores em `metadata["fused_features"]`.
    O `raw_score` do output é o do primeiro input (preservar utilidade em mining
    e casework_viz que esperam score escalar 1D).

    O CalibrationModelTrainingInstructionExecutor detecta `is_fused=True` no
    dataset.metadata e extrai features multivariadas pra treinar o calibrador.
    """

    _FUSED_FEATURES_KEY = "fused_features"

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, ScoreFusionInstruction):
            return self._failure_result(
                "Instrução incompatível com ScoreFusionInstructionExecutor."
            )

        self._log(
            context,
            f"▶ Iniciando SCORE_FUSION: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            res_a = context.get_resource(instruction.input_a_resource_id)
            res_b = context.get_resource(instruction.input_b_resource_id)
            ds_a = res_a.payload
            ds_b = res_b.payload

            fused_dataset, n_pairs, n_features = self._build_fused_dataset(
                ds_a=ds_a,
                ds_b=ds_b,
                source_a_id=instruction.input_a_resource_id,
                source_b_id=instruction.input_b_resource_id,
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
                payload=fused_dataset,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            self._log(
                context,
                f"✅ Score fusion produzido: '{output_id}' "
                f"(n_pairs={n_pairs}, dim_features={n_features})"
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_a_resource_id": instruction.input_a_resource_id,
                    "input_b_resource_id": instruction.input_b_resource_id,
                    "output_resource_id": output_id,
                    "n_pairs": int(n_pairs),
                    "n_features": int(n_features),
                    "is_fused": True,
                },
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro em SCORE_FUSION ({instruction.instruction_id}): {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, ScoreFusionInstruction):
            result.add_error(
                "A instrução informada não é do tipo ScoreFusionInstruction."
            )
            return result

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

        if instruction.input_a_resource_id == instruction.input_b_resource_id:
            result.add_error(
                "input_a_resource_id e input_b_resource_id não podem ser o mesmo "
                "recurso (fusão exige dois sistemas distintos)."
            )

        if not instruction.output_resource_id:
            result.add_error(
                "A instrução deve definir 'output_resource_id'."
            )

        # Verifica que os payloads são SimilarityScoreDataset
        for field_name, rid in (
            ("A", instruction.input_a_resource_id),
            ("B", instruction.input_b_resource_id),
        ):
            if rid and context.has_resource(rid):
                payload = context.get_resource(rid).payload
                if payload is not None and not isinstance(payload, SimilarityScoreDataset):
                    result.add_error(
                        f"Recurso '{rid}' (sistema {field_name}) não é um "
                        f"SimilarityScoreDataset (got {type(payload).__name__})."
                    )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ─── Núcleo da fusão ────────────────────────────────────────────────────

    def _build_fused_dataset(
        self,
        ds_a: SimilarityScoreDataset,
        ds_b: SimilarityScoreDataset,
        source_a_id: str,
        source_b_id: str,
        require_pair_match: bool,
        output_name: str,
        context: StrategyExecutionContextProtocol,
    ) -> Tuple[SimilarityScoreDataset, int, int]:
        """
        Constrói o SimilarityScoreDataset fundido.

        Cada sample herda do A (probe, reference, same_person, metadata)
        e ganha `metadata["fused_features"] = [score_A, score_B]`.

        Se require_pair_match=True, exige ordem idêntica e mesma quantidade
        em A e B. Senão, reordena B por (probe.face_hash, reference.face_hash)
        para casar com A.
        """
        n_a = len(ds_a.items)
        n_b = len(ds_b.items)

        if n_a == 0 or n_b == 0:
            raise ValueError(
                f"Score fusion requer ambos os datasets não vazios "
                f"(n_a={n_a}, n_b={n_b})."
            )

        if require_pair_match:
            if n_a != n_b:
                raise ValueError(
                    f"Datasets têm tamanhos diferentes "
                    f"({n_a} vs {n_b}) — defina require_pair_match=False "
                    f"para reordenar B por hash."
                )
            # Valida ordem
            for i, (sa, sb) in enumerate(zip(ds_a.items, ds_b.items)):
                if self._pair_key(sa) != self._pair_key(sb):
                    raise ValueError(
                        f"Pares em posição {i} não batem entre A e B "
                        f"(probe.face_hash ou reference.face_hash diferentes). "
                        f"Defina require_pair_match=False para reordenar."
                    )
            items_b_aligned = ds_b.items
        else:
            items_b_aligned = self._reorder_b_to_match_a(ds_a, ds_b)

        fused_items: List[SimilarityScoreSample] = []
        for sa, sb in zip(ds_a.items, items_b_aligned):
            sample_meta = dict(sa.metadata) if sa.metadata else {}
            sample_meta[self._FUSED_FEATURES_KEY] = [
                float(sa.raw_score),
                float(sb.raw_score),
            ]
            fused_items.append(SimilarityScoreSample(
                probe=sa.probe,
                reference=sa.reference,
                raw_score=float(sa.raw_score),  # mantém escalar do A
                same_person=sa.same_person,
                metadata=sample_meta,
            ))

        dataset_metadata = dict(ds_a.metadata) if ds_a.metadata else {}
        dataset_metadata["is_fused"] = True
        dataset_metadata["fused_source_resources"] = [source_a_id, source_b_id]
        dataset_metadata["fused_n_features"] = 2

        fused = SimilarityScoreDataset(
            name=output_name,
            description=(
                f"Score fusion de '{source_a_id}' (A) + '{source_b_id}' (B). "
                f"Cada sample carrega vetor [score_A, score_B] em "
                f"metadata['fused_features']. raw_score = score_A."
            ),
            items=fused_items,
            metadata=dataset_metadata,
        )
        return fused, len(fused_items), 2

    @staticmethod
    def _pair_key(sample: SimilarityScoreSample) -> Tuple[str, str]:
        """Identifica unicamente um par (probe, reference) por face_hash."""
        probe_hash = getattr(sample.probe, "face_hash", None) or ""
        ref_hash = getattr(sample.reference, "face_hash", None) or ""
        return (str(probe_hash), str(ref_hash))

    def _reorder_b_to_match_a(
        self,
        ds_a: SimilarityScoreDataset,
        ds_b: SimilarityScoreDataset,
    ) -> List[SimilarityScoreSample]:
        """Reordena B para casar com A por (probe.face_hash, reference.face_hash).

        Falha se algum par de A não estiver em B. B pode ter pares extras —
        eles são ignorados.
        """
        index_b = {
            self._pair_key(s): s for s in ds_b.items
        }
        aligned: List[SimilarityScoreSample] = []
        missing = []
        for sa in ds_a.items:
            key = self._pair_key(sa)
            sb = index_b.get(key)
            if sb is None:
                missing.append(key)
                if len(missing) > 5:
                    break
            else:
                aligned.append(sb)
        if missing:
            sample_str = ", ".join(f"{m[0][:8]}..→{m[1][:8]}.." for m in missing[:5])
            raise ValueError(
                f"{len(missing)}+ pares de A não estão em B (ex: {sample_str}). "
                "Verifique que os dois sistemas processaram o mesmo conjunto de pares."
            )
        return aligned
