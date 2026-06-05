"""
Testes de integração do `LRArchitectureVectorizationInstructionExecutor`.

Cobre:
- Fail-closed sem `context.frame` (MissingExecutionFrameError).
- Execução PREPARATION + FIT_FULL_CALIBRATION_LEGACY produz matriz
  correta + state com fingerprints estáveis.
- VALIDATION com parameter_mode != INHERIT falha.
- FIT_PER_FOLD na v1 falha.
- INHERIT em VALIDATION: state reaplicado, sem recálculo de s₀/r₀.
- MANUAL: validação per-arquitetura.
- Equivalência matemática: mesmo dataset → mesma matriz que builder direto.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.lr_architecture_vectorization_instruction_executor import (
    LRArchitectureVectorizationInstructionExecutor,
    MissingExecutionFrameError,
)
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
    LRArchitectureVectorizationInstruction,
)
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.lr.lr_architecture_state import LRArchitectureStateFactory
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.instruction_execution_frame import InstructionExecutionFrame
from faceaudit.models.strategy.strategy_resource import StrategyResource


def _mk_emb(idx: int, ident: str) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=np.zeros(8), face_hash=f"h_{idx:06d}",
        identity=ident, image_path=f"i{idx}", file_name=f"i{idx}",
    )


def _build_dataset_with_pairs(
    n_gen: int, n_imp: int, seed: int = 42,
) -> Tuple[
    SimilarityScoreDataset, SimilarityScoreDataset, SimilarityScoreDataset,
    np.ndarray,
]:
    """
    Constrói base (cosseno) + dois rarity datasets (TYPICALITY_PERCENTILE).
    Pair_keys consistentes nos três.
    """
    rng = np.random.RandomState(seed)
    cos_imp = rng.normal(-0.05, 0.08, n_imp)
    cos_gen = rng.normal(0.4, 0.05, n_gen)
    cos_all = np.concatenate([cos_imp, cos_gen])
    same = [False] * n_imp + [True] * n_gen
    n = len(cos_all)

    items: List[SimilarityScoreSample] = []
    for i in range(n):
        pid = f"id_{i//2}" if same[i] else f"id_p_{i}"
        rid = f"id_{i//2}" if same[i] else f"id_r_{i}"
        items.append(SimilarityScoreSample(
            probe=_mk_emb(2 * i, pid), reference=_mk_emb(2 * i + 1, rid),
            raw_score=float(cos_all[i]), same_person=same[i],
        ))

    base = SimilarityScoreDataset(
        name="base", items=items,
        metadata={
            "score_kind": "cosine",
            "pair_role_orientation": "PROBE_REFERENCE",
            "pair_key_mode": "ORDERED",
        },
    )

    rng2 = np.random.RandomState(seed + 1)
    typ_F = rng2.uniform(0.0, 1.0, n)
    typ_P = rng2.uniform(0.0, 1.0, n)

    def _mk_typ(scores: np.ndarray, combiner: str) -> SimilarityScoreDataset:
        ti = [SimilarityScoreSample(
            probe=items[i].probe, reference=items[i].reference,
            raw_score=float(s), same_person=same[i],
        ) for i, s in enumerate(scores)]
        return SimilarityScoreDataset(
            name=f"t_{combiner}", items=ti,
            metadata={
                "score_type": TypicalityScoreType.TYPICALITY_PERCENTILE.value,
                "combiner": combiner,
                "pair_role_orientation": "PROBE_REFERENCE",
            },
        )

    return base, _mk_typ(typ_F, "PROBE_ONLY"), _mk_typ(typ_P, "REFERENCE_ONLY"), np.array(same)


def _mk_ctx(
    base: SimilarityScoreDataset,
    rp: SimilarityScoreDataset,
    rr: SimilarityScoreDataset,
) -> StrategyExecutionContext:
    ctx = StrategyExecutionContext()
    ctx.set_resource(StrategyResource(
        resource_id="base", name="B",
        resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=base))
    ctx.set_resource(StrategyResource(
        resource_id="rp", name="RP",
        resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=rp))
    ctx.set_resource(StrategyResource(
        resource_id="rr", name="RR",
        resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=rr))
    return ctx


def _mk_instr(
    architecture: LRCalibrationArchitecture,
    parameter_mode: ArchitectureParameterMode = ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
    **kwargs,
) -> LRArchitectureVectorizationInstruction:
    base_kwargs = dict(
        instruction_id="ix",
        name="Test",
        instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
        architecture=architecture,
        parameter_mode=parameter_mode,
        base_score_resource_id="base",
        rarity_probe_resource_id="rp",
        rarity_reference_resource_id="rr",
        output_resource_id="out_vec",
        output_raw_input_resource_id="out_raw",
    )
    base_kwargs.update(kwargs)
    return LRArchitectureVectorizationInstruction(**base_kwargs)


# ──────────────────────────────────────────────────────────────────────


class TestFrameRequirement:
    def test_missing_frame_raises(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr)
        instr = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        ex = LRArchitectureVectorizationInstructionExecutor()
        # Sem with_frame, ctx.frame é None
        with pytest.raises(MissingExecutionFrameError):
            ex.execute(instr, ctx)


class TestPreparationFitFullLegacy:
    def test_bilateral_produces_correct_matrix(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success, result.errors
        ds = ctx.get_resource("out_vec").payload
        assert ds.metadata["lr_architecture"] == "BILATERAL_FULL_CENTERED"
        assert ds.metadata["score_vector_dim"] == 7
        assert ds.metadata["score_vector_matrix"].shape[1] == 7
        sd = ds.metadata["architecture_state"]
        assert sd["s0"] is not None
        assert sd["r0_probe"] is not None
        assert sd["r0_reference"] is not None
        assert sd["s0_fit_summary"]["is_stable"]

    def test_cosine_only_skips_typicality(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.COSINE_ONLY,
            rarity_probe_resource_id=None, rarity_reference_resource_id=None,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success, result.errors
        ds = ctx.get_resource("out_vec").payload
        assert ds.metadata["score_vector_dim"] == 1


class TestPhaseGuards:
    def test_validation_without_inherit_fails(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.VALIDATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any("parameter_mode=INHERIT" in e for e in result.errors)

    def test_fit_per_fold_in_v1_raises(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.FIT_PER_FOLD,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any("reservado para v2" in e or "FIT_PER_FOLD" in e for e in result.errors)


class TestInheritReappliesState:
    def test_validation_inherit_uses_state_fingerprint_match(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ex = LRArchitectureVectorizationInstructionExecutor()

        # PREPARATION
        ctx_prep = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr_prep = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        result_prep = ex.execute(instr_prep, ctx_prep)
        assert result_prep.success, result_prep.errors
        prep_ds = ctx_prep.get_resource("out_vec").payload
        prep_fp = prep_ds.metadata["vectorization_contract_fingerprint"]

        # VALIDATION com INHERIT — reusa base/rp/rr (na prática viria de
        # outro dataset, mas para fp match basta o state idêntico)
        ctx_val = _mk_ctx(base, rp, rr)
        ctx_val.set_resource(StrategyResource(
            resource_id="prev", name="Prev",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=prep_ds))
        ctx_val = ctx_val.with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.VALIDATION, instruction_index=0, strategy_id="s"))
        instr_val = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.INHERIT,
            inherit_architecture_state_from_resource_id="prev",
            output_resource_id="out_val", output_raw_input_resource_id="out_val_raw",
        )
        result_val = ex.execute(instr_val, ctx_val)
        assert result_val.success, result_val.errors
        val_ds = ctx_val.get_resource("out_val").payload
        # Mesmo state semântico → mesmo contract_fp
        assert val_ds.metadata["vectorization_contract_fingerprint"] == prep_fp

    def test_inherit_with_corrupted_state_fails(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ex = LRArchitectureVectorizationInstructionExecutor()

        ctx_prep = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr_prep = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        ex.execute(instr_prep, ctx_prep)
        prep_ds = ctx_prep.get_resource("out_vec").payload
        # Corrompe s0 no state sem recalcular fingerprint
        prep_ds.metadata["architecture_state"]["s0"] = 999.0

        ctx_val = _mk_ctx(base, rp, rr)
        ctx_val.set_resource(StrategyResource(
            resource_id="prev", name="P",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=prep_ds))
        ctx_val = ctx_val.with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.VALIDATION, instruction_index=0, strategy_id="s"))
        instr_val = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.INHERIT,
            inherit_architecture_state_from_resource_id="prev",
            output_resource_id="out_val", output_raw_input_resource_id="out_val_raw",
        )
        result = ex.execute(instr_val, ctx_val)
        assert not result.success
        # Erro pode vir do StateIntegrityError (fingerprint mismatch)
        assert any("fingerprint" in e.lower() or "integrity" in e.lower() for e in result.errors)


class TestManualPerArchitecture:
    def test_manual_missing_s0_raises(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.MANUAL,
            # manual_s0=None
            manual_r0_probe=0.5, manual_r0_reference=0.4,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any("manual_s0" in e for e in result.errors)

    def test_manual_complete_succeeds(self):
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.MANUAL,
            manual_s0=0.2, manual_r0_probe=0.5, manual_r0_reference=0.5,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success, result.errors
        sd = ctx.get_resource("out_vec").payload.metadata["architecture_state"]
        assert sd["s0"] == 0.2
        assert sd["r0_probe"] == 0.5
        assert sd["fit_scope"] == "MANUAL"


class TestOrientationPolicy:
    def test_ordered_architecture_without_orientation_fails(self):
        """Base dataset SEM pair_role_orientation explícita + arquitetura ordenada → erro."""
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        base.metadata = {"score_kind": "cosine"}  # sem orientação
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any("orientação" in e.lower() or "PROBE_REFERENCE" in e for e in result.errors)

    def test_symmetric_architecture_allows_no_orientation(self):
        """COSINE_ONLY aceita base sem pair_role_orientation."""
        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        base.metadata = {"score_kind": "cosine"}  # sem orientação
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(
            LRCalibrationArchitecture.COSINE_ONLY,
            rarity_probe_resource_id=None, rarity_reference_resource_id=None,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success, result.errors


class TestEquivalenceAgainstDirectBuilder:
    """
    O executor produz a mesma matriz que `ArchitectureFeatureBuilder.build`
    chamado diretamente com os mesmos inputs (state + arrays). Verifica
    que a camada de instrução não introduz transformações espúrias.
    """

    def test_bilateral_matches_direct_build(self):
        from faceaudit.services import architecture_feature_builder as builder
        from faceaudit.services.rarity_transform_service import to_canonical_rarity

        base, rp, rr, _ = _build_dataset_with_pairs(150, 600)
        ctx = _mk_ctx(base, rp, rr).with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="s"))
        instr = _mk_instr(LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success

        out_ds = ctx.get_resource("out_vec").payload
        matrix_via_executor = out_ds.metadata["score_vector_matrix"]
        state = LRArchitectureStateFactory.from_dict(
            out_ds.metadata["architecture_state"]
        )

        # Reproduz arrays para comparar
        base_scores = np.array([it.raw_score for it in base.items])
        rp_raw = np.array([it.raw_score for it in rp.items])
        rr_raw = np.array([it.raw_score for it in rr.items])
        # Pair-keys são consistentes nos 3, então índices coincidem
        rarity_F, _ = to_canonical_rarity(rp_raw, TypicalityScoreType.TYPICALITY_PERCENTILE)
        rarity_P, _ = to_canonical_rarity(rr_raw, TypicalityScoreType.TYPICALITY_PERCENTILE)

        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.BILATERAL_FULL_CENTERED]
        matrix_direct = builder.build(spec, state, base_scores, rarity_F, rarity_P)

        np.testing.assert_allclose(matrix_via_executor, matrix_direct, atol=1e-12)
