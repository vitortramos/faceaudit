"""
Testes da família ONE_FACE (V2.14.1).

Cobre:
- Registry de arquiteturas ONE_FACE (6 novas).
- Schema `schema_for_base_and_single_rarity` distinto do two-face por
  fingerprint.
- Termos de feature ONE_FACE constroem matrizes coerentes com derivadas.
- Smoke: Pipeline `[ArchitectureVectorizerTransformer + StandardScaler +
  LogReg]` fita para cada arquitetura ONE_FACE.
- Validação: arquitetura ONE_FACE com rarity_probe/reference preenchido → erro;
  arquitetura two-face com rarity_resource_id preenchido → erro.
- State integrity: r0_single populado, fingerprint estável, INHERIT funciona.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.lr_architecture_vectorization_instruction_executor import (
    LRArchitectureVectorizationInstructionExecutor,
)
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
    LRArchitectureVectorizationInstruction,
)
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    schema_for_base_and_rarities,
    schema_for_base_and_single_rarity,
    schema_for_base_only,
)
from faceaudit.models.lr.lr_architecture_spec import (
    LR_ARCHITECTURE_SPECS,
    get_architecture_spec,
)
from faceaudit.models.lr.lr_architecture_state import LRArchitectureStateFactory
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.instruction_execution_frame import InstructionExecutionFrame
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.services import architecture_feature_builder as builder
from faceaudit.services.architecture_vectorizer_transformer import (
    ArchitectureVectorizerTransformer,
)


# As 6 arquiteturas ONE_FACE
ONE_FACE_ARCHITECTURES = [
    LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE,
    LRCalibrationArchitecture.ONE_SIDED_CENTERED_ONE_FACE,
    LRCalibrationArchitecture.LINEAR_NON_CENTERED_ONE_FACE,
    LRCalibrationArchitecture.LINEAR_CENTERED_ONE_FACE,
    LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE,
    LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE,
]


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_one_face_count(self):
        one_face = [
            arch for arch in LRCalibrationArchitecture
            if get_architecture_spec(arch).is_one_face
        ]
        assert len(one_face) == 6

    @pytest.mark.parametrize("arch", ONE_FACE_ARCHITECTURES, ids=lambda a: a.value)
    def test_one_face_flags(self, arch):
        spec = get_architecture_spec(arch)
        assert spec.requires_typicality_single is True
        assert spec.is_one_face is True
        # ONE_FACE não consome probe/reference
        assert spec.requires_typicality_probe is False
        assert spec.requires_typicality_reference is False
        # ONE_FACE é simétrica (sem ordem de papéis)
        assert spec.requires_ordered_pair_roles is False
        # Todas exigem s0
        assert spec.requires_s0 is True

    def test_one_face_output_dims(self):
        """Dimensões esperadas por arquitetura ONE_FACE."""
        expected = {
            LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE: 2,
            LRCalibrationArchitecture.ONE_SIDED_CENTERED_ONE_FACE: 2,
            LRCalibrationArchitecture.LINEAR_NON_CENTERED_ONE_FACE: 2,
            LRCalibrationArchitecture.LINEAR_CENTERED_ONE_FACE: 2,
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE: 3,
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE: 5,
        }
        for arch, d in expected.items():
            assert get_architecture_spec(arch).output_dim == d

    def test_fingerprints_distinct_from_two_face_equivalents(self):
        """ONE_FACE deve ter fingerprint diferente do análogo two-face."""
        pairs = [
            (LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED,
             LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE),
            (LRCalibrationArchitecture.ONE_SIDED_CENTERED,
             LRCalibrationArchitecture.ONE_SIDED_CENTERED_ONE_FACE),
            (LRCalibrationArchitecture.HIERARCHICAL_CENTERED,
             LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE),
            (LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
             LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE),
        ]
        for two, one in pairs:
            fp_two = get_architecture_spec(two).fingerprint()
            fp_one = get_architecture_spec(one).fingerprint()
            assert fp_two != fp_one, f"{two.value} colide com {one.value}"


# ──────────────────────────────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────────────────────────────


class TestSchema:
    def test_one_face_schema_has_two_columns(self):
        schema = schema_for_base_and_single_rarity()
        assert schema.n_columns == 2
        assert schema.columns == ("base_score", "rarity_single")
        assert schema.base_score_column == 0
        assert schema.rarity_single_column == 1
        assert schema.rarity_probe_column is None
        assert schema.rarity_reference_column is None
        assert schema.is_one_face is True

    def test_one_face_schema_fingerprint_distinct(self):
        fp_single = schema_for_base_and_single_rarity().fingerprint()
        fp_rarities = schema_for_base_and_rarities().fingerprint()
        fp_only = schema_for_base_only().fingerprint()
        assert fp_single != fp_rarities
        assert fp_single != fp_only

    def test_one_face_schema_mutually_exclusive(self):
        """Schema com rarity_single + rarity_probe/reference deve falhar."""
        from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
            LRArchitectureRawFeatureSchema,
        )
        with pytest.raises(ValueError, match="exclusivo"):
            LRArchitectureRawFeatureSchema(
                schema_version="1.0",
                columns=("base", "rs", "rp"),
                base_score_column=0,
                rarity_probe_column=2,
                rarity_reference_column=None,
                rarity_single_column=1,
            )


# ──────────────────────────────────────────────────────────────────────
# Builder + derivatives
# ──────────────────────────────────────────────────────────────────────


class TestBuilder:
    @pytest.mark.parametrize("arch", ONE_FACE_ARCHITECTURES, ids=lambda a: a.value)
    def test_build_returns_correct_shape(self, arch):
        spec = get_architecture_spec(arch)
        n = 30
        rng = np.random.RandomState(42)
        base = rng.uniform(-0.2, 0.8, n)
        r_S = rng.uniform(0.0, 1.0, n)
        state = LRArchitectureStateFactory.create(
            spec=spec,
            s0=0.25 if spec.requires_s0 else None,
            r0_single=0.5 if spec.centering_mode.value == "MEAN" else None,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
            fit_scope=ArchitectureFitScope.MANUAL,
            s0_resolution_method=(
                S0ResolutionMethod.MANUAL if spec.requires_s0
                else S0ResolutionMethod.NOT_APPLICABLE
            ),
            r0_single_estimator="MEAN" if spec.centering_mode.value == "MEAN" else None,
            rarity_single_score_type="TYPICALITY_PERCENTILE",
        )
        X = builder.build(spec, state, base, None, None, r_S)
        assert X.shape == (n, spec.output_dim)
        assert X.dtype == np.float64
        # primeira coluna é sempre base
        np.testing.assert_allclose(X[:, 0], base)

    @pytest.mark.parametrize("arch", ONE_FACE_ARCHITECTURES, ids=lambda a: a.value)
    def test_derivative_matches_finite_differences(self, arch):
        spec = get_architecture_spec(arch)
        n = 20
        rng = np.random.RandomState(7)
        # bases longe de s0=0.25
        base = np.concatenate([
            rng.uniform(-0.2, 0.1, n // 2),
            rng.uniform(0.4, 0.8, n // 2),
        ])
        r_S = rng.uniform(0.1, 0.9, n)

        state = LRArchitectureStateFactory.create(
            spec=spec,
            s0=0.25,
            r0_single=0.5 if spec.centering_mode.value == "MEAN" else None,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
            fit_scope=ArchitectureFitScope.MANUAL,
            s0_resolution_method=S0ResolutionMethod.MANUAL,
            r0_single_estimator="MEAN" if spec.centering_mode.value == "MEAN" else None,
        )

        eps = 1e-6
        X_plus = builder.build(spec, state, base + eps, None, None, r_S)
        X_minus = builder.build(spec, state, base - eps, None, None, r_S)
        D_num = (X_plus - X_minus) / (2 * eps)
        D_ana = builder.derivative_matrix_wrt_base(
            spec, state, base, None, None, r_S,
        )
        np.testing.assert_allclose(D_num, D_ana, atol=1e-5)

    def test_missing_rarity_single_raises(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED_ONE_FACE
        )
        state = LRArchitectureStateFactory.create(
            spec=spec, s0=0.25,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
            fit_scope=ArchitectureFitScope.MANUAL,
            s0_resolution_method=S0ResolutionMethod.MANUAL,
        )
        with pytest.raises(ValueError, match="rarity_single"):
            builder.build(spec, state, np.array([0.1, 0.3, 0.5]))


# ──────────────────────────────────────────────────────────────────────
# State factory
# ──────────────────────────────────────────────────────────────────────


class TestStateFactory:
    def test_one_face_state_r0_single_populated(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE
        )
        state = LRArchitectureStateFactory.create(
            spec=spec, s0=0.25, r0_single=0.5,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
            fit_scope=ArchitectureFitScope.MANUAL,
            s0_resolution_method=S0ResolutionMethod.MANUAL,
            r0_single_estimator="MEAN",
        )
        assert state.r0_single == 0.5
        assert state.r0_probe is None
        assert state.r0_reference is None

    def test_one_face_rejects_r0_probe(self):
        # Usa arquitetura com centering=MEAN para chegar à validação ONE_FACE.
        spec = get_architecture_spec(
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE
        )
        with pytest.raises(ValueError, match="ONE_FACE.*não aceita"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25, r0_probe=0.5, r0_single=0.5,
                base_score_kind="cosine",
                pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
                fit_scope=ArchitectureFitScope.MANUAL,
                s0_resolution_method=S0ResolutionMethod.MANUAL,
                r0_single_estimator="MEAN",
            )

    def test_two_face_rejects_r0_single(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        )
        with pytest.raises(ValueError, match="não é ONE_FACE.*r0_single"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25, r0_probe=0.5, r0_reference=0.5,
                r0_single=0.5,  # ilegal
                base_score_kind="cosine",
                pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
                fit_scope=ArchitectureFitScope.MANUAL,
                s0_resolution_method=S0ResolutionMethod.MANUAL,
            )

    def test_one_face_state_roundtrip(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE
        )
        state = LRArchitectureStateFactory.create(
            spec=spec, s0=0.25, r0_single=0.5,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
            fit_scope=ArchitectureFitScope.MANUAL,
            s0_resolution_method=S0ResolutionMethod.MANUAL,
            r0_single_estimator="MEAN",
            rarity_single_score_type="TYPICALITY_PERCENTILE",
        )
        d = state.to_dict()
        assert d["r0_single"] == 0.5
        assert d["rarity_single_score_type"] == "TYPICALITY_PERCENTILE"
        # from_dict round-trip
        state2 = LRArchitectureStateFactory.from_dict(d)
        assert state2.r0_single == 0.5
        assert state2.vectorization_contract_fingerprint == state.vectorization_contract_fingerprint


# ──────────────────────────────────────────────────────────────────────
# Pipeline (sklearn) smoke
# ──────────────────────────────────────────────────────────────────────


class TestPipelineSmoke:
    @pytest.mark.parametrize("arch", ONE_FACE_ARCHITECTURES, ids=lambda a: a.value)
    def test_pipeline_fits_for_each_one_face_arch(self, arch):
        """
        Smoke test: Pipeline [ArchitectureVectorizerTransformer →
        StandardScaler → LogReg] fita sem erro e produz state com
        fingerprints corretos.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.RandomState(42)
        n = 200
        # Sintetiza base + rarity para o transformer
        base = np.concatenate([
            rng.normal(-0.05, 0.05, n // 2),  # impostor
            rng.normal(0.4, 0.05, n // 2),    # genuine
        ])
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        r_single = rng.uniform(0.0, 1.0, n)
        X = np.column_stack([base, r_single])

        schema = schema_for_base_and_single_rarity()
        transformer = ArchitectureVectorizerTransformer(
            architecture=arch,
            schema=schema,
            rarity_single_score_type="TYPICALITY_PERCENTILE",
            pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
        )

        pipeline = Pipeline([
            ("arch_vec", transformer),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(C=1.0, max_iter=1000)),
        ])

        pipeline.fit(X, y)
        # state populado
        state = pipeline.named_steps["arch_vec"].state_
        assert state.s0 is not None
        if LR_ARCHITECTURE_SPECS[arch].centering_mode.value == "MEAN":
            assert state.r0_single is not None
            assert state.r0_probe is None
        else:
            assert state.r0_single is None
        # transform e predict_proba funcionam
        probs = pipeline.predict_proba(X)
        assert probs.shape == (n, 2)


# ──────────────────────────────────────────────────────────────────────
# Executor integration
# ──────────────────────────────────────────────────────────────────────


def _mk_emb(idx: int, ident: str):
    return EmbeddingInfo(
        embedding=np.zeros(8), face_hash=f"h_{idx:06d}",
        identity=ident, image_path=f"i{idx}", file_name=f"i{idx}",
    )


def _build_one_face_datasets(
    n_gen: int, n_imp: int, seed: int = 42,
):
    """Constrói base (cosseno) + 1 rarity dataset (single, ONE_FACE)."""
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
            # ONE_FACE não requer pair_role_orientation, mas declaramos
            # mesmo assim para garantir compat com o executor.
            "pair_role_orientation": "NOT_APPLICABLE",
            "pair_key_mode": "ORDERED",
        },
    )

    rng2 = np.random.RandomState(seed + 1)
    typ = rng2.uniform(0.0, 1.0, n)

    typ_items = [SimilarityScoreSample(
        probe=items[i].probe, reference=items[i].reference,
        raw_score=float(t), same_person=same[i],
    ) for i, t in enumerate(typ)]
    typ_ds = SimilarityScoreDataset(
        name="t_single", items=typ_items,
        metadata={
            "score_type": TypicalityScoreType.TYPICALITY_PERCENTILE.value,
            "combiner": "PROBE_ONLY",
        },
    )
    return base, typ_ds


def _ctx_with_one_face(base, rarity):
    ctx = StrategyExecutionContext()
    ctx.set_resource(StrategyResource(
        resource_id="base", name="B",
        resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=base))
    ctx.set_resource(StrategyResource(
        resource_id="r_single", name="RS",
        resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=rarity))
    return ctx


def _mk_one_face_instr(
    architecture: LRCalibrationArchitecture,
    parameter_mode: ArchitectureParameterMode = ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
    **kwargs,
):
    base_kwargs = dict(
        instruction_id="ix",
        name="Test",
        instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
        architecture=architecture,
        parameter_mode=parameter_mode,
        base_score_resource_id="base",
        rarity_resource_id="r_single",
        output_resource_id="out_vec",
        output_raw_input_resource_id="out_raw",
    )
    base_kwargs.update(kwargs)
    return LRArchitectureVectorizationInstruction(**base_kwargs)


class TestExecutorOneFace:
    def test_executor_succeeds_one_face_bilateral(self):
        base, r_single = _build_one_face_datasets(150, 600)
        ctx = _ctx_with_one_face(base, r_single).with_frame(
            InstructionExecutionFrame(
                phase=StrategyPhase.PREPARATION,
                instruction_index=0,
                strategy_id="s",
            )
        )
        instr = _mk_one_face_instr(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE,
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert result.success, result.errors
        ds = ctx.get_resource("out_vec").payload
        assert ds.metadata["lr_architecture"] == "BILATERAL_FULL_CENTERED_ONE_FACE"
        assert ds.metadata["score_vector_dim"] == 5
        sd = ds.metadata["architecture_state"]
        assert sd["s0"] is not None
        assert sd["r0_single"] is not None
        assert sd["r0_probe"] is None
        assert sd["r0_reference"] is None

    def test_executor_rejects_one_face_with_rarity_probe(self):
        """ONE_FACE com rarity_probe_resource_id preenchido → erro."""
        base, r_single = _build_one_face_datasets(150, 600)
        ctx = _ctx_with_one_face(base, r_single).with_frame(
            InstructionExecutionFrame(
                phase=StrategyPhase.PREPARATION,
                instruction_index=0,
                strategy_id="s",
            )
        )
        instr = _mk_one_face_instr(
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE,
            rarity_probe_resource_id="r_single",  # ilegal em ONE_FACE
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any(
            "ONE_FACE" in e and "rarity_probe" in e
            for e in result.errors
        )

    def test_executor_rejects_two_face_with_rarity_single(self):
        """Two-face com rarity_resource_id preenchido → erro."""
        base, r_single = _build_one_face_datasets(150, 600)
        ctx = StrategyExecutionContext()
        ctx.set_resource(StrategyResource(
            resource_id="base", name="B",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=base))
        ctx.set_resource(StrategyResource(
            resource_id="rp", name="RP",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=r_single))
        ctx.set_resource(StrategyResource(
            resource_id="rr", name="RR",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=r_single))
        # injetar pair_role_orientation correto na base
        base.metadata["pair_role_orientation"] = "PROBE_REFERENCE"
        ctx = ctx.with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.PREPARATION,
            instruction_index=0, strategy_id="s",
        ))
        instr = LRArchitectureVectorizationInstruction(
            instruction_id="ix",
            name="Test",
            instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
            base_score_resource_id="base",
            rarity_probe_resource_id="rp",
            rarity_reference_resource_id="rr",
            rarity_resource_id="r_single",  # ilegal em two-face
            output_resource_id="out_vec",
        )
        ex = LRArchitectureVectorizationInstructionExecutor()
        result = ex.execute(instr, ctx)
        assert not result.success
        assert any(
            "não é ONE_FACE" in e or "ONE_FACE" in e
            for e in result.errors
        )

    def test_inherit_one_face_succeeds(self):
        """INHERIT em VALIDATION com state ONE_FACE reaplicado."""
        base, r_single = _build_one_face_datasets(150, 600)
        ex = LRArchitectureVectorizationInstructionExecutor()

        # PREPARATION
        ctx_prep = _ctx_with_one_face(base, r_single).with_frame(
            InstructionExecutionFrame(
                phase=StrategyPhase.PREPARATION,
                instruction_index=0, strategy_id="s",
            )
        )
        instr_prep = _mk_one_face_instr(
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE,
        )
        result_prep = ex.execute(instr_prep, ctx_prep)
        assert result_prep.success, result_prep.errors
        prep_ds = ctx_prep.get_resource("out_vec").payload
        prep_fp = prep_ds.metadata["vectorization_contract_fingerprint"]

        # VALIDATION com INHERIT
        ctx_val = _ctx_with_one_face(base, r_single)
        ctx_val.set_resource(StrategyResource(
            resource_id="prev", name="P",
            resource_type=StrategyResourceType.SIMILARITY_SCORES, payload=prep_ds,
        ))
        ctx_val = ctx_val.with_frame(InstructionExecutionFrame(
            phase=StrategyPhase.VALIDATION,
            instruction_index=0, strategy_id="s",
        ))
        instr_val = _mk_one_face_instr(
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED_ONE_FACE,
            parameter_mode=ArchitectureParameterMode.INHERIT,
            inherit_architecture_state_from_resource_id="prev",
            output_resource_id="out_val",
            output_raw_input_resource_id="out_val_raw",
        )
        result_val = ex.execute(instr_val, ctx_val)
        assert result_val.success, result_val.errors
        val_ds = ctx_val.get_resource("out_val").payload
        assert val_ds.metadata["vectorization_contract_fingerprint"] == prep_fp


# ──────────────────────────────────────────────────────────────────────
# DTO get_hidden_fields
# ──────────────────────────────────────────────────────────────────────


class TestDTOHiddenFields:
    def test_one_face_hides_probe_reference(self):
        instr = LRArchitectureVectorizationInstruction(
            instruction_id="i",
            name="Test",
            instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED_ONE_FACE,
        )
        hidden = instr.get_hidden_fields()
        assert "rarity_probe_resource_id" in hidden
        assert "rarity_reference_resource_id" in hidden
        # rarity_resource_id deve estar visível
        assert "rarity_resource_id" not in hidden

    def test_two_face_hides_single(self):
        instr = LRArchitectureVectorizationInstruction(
            instruction_id="i",
            name="Test",
            instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        )
        hidden = instr.get_hidden_fields()
        assert "rarity_resource_id" in hidden
        # probe/reference visíveis
        assert "rarity_probe_resource_id" not in hidden
        assert "rarity_reference_resource_id" not in hidden

    def test_cosine_only_hides_all_rarities(self):
        instr = LRArchitectureVectorizationInstruction(
            instruction_id="i",
            name="Test",
            instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
            architecture=LRCalibrationArchitecture.COSINE_ONLY,
        )
        hidden = instr.get_hidden_fields()
        assert "rarity_probe_resource_id" in hidden
        assert "rarity_reference_resource_id" in hidden
        assert "rarity_resource_id" in hidden
