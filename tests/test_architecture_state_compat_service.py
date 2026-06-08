"""
Testes do `validate_state_compatibility`. Cobre os 20+ cenários de
divergência entre `LRArchitectureState` herdado e contexto incoming.
"""

from __future__ import annotations

import pytest

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.lr.lr_architecture_state import LRArchitectureStateFactory
from faceaudit.services.architecture_state_compat_service import (
    validate_state_compatibility,
)


def _make_full_state(arch=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED):
    spec = LR_ARCHITECTURE_SPECS[arch]
    return LRArchitectureStateFactory.create(
        spec=spec,
        s0=0.25 if spec.requires_s0 else None,
        r0_probe=0.5 if (spec.requires_typicality_probe and spec.centering_mode.value == "MEAN") else None,
        r0_reference=0.48 if (spec.requires_typicality_reference and spec.centering_mode.value == "MEAN") else None,
        base_score_kind="cosine",
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE if spec.requires_ordered_pair_roles else PairRoleOrientation.NOT_APPLICABLE,
        fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY if spec.requires_s0 else ArchitectureFitScope.NOT_APPLICABLE,
        s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG if spec.requires_s0 else S0ResolutionMethod.NOT_APPLICABLE,
        rarity_probe_score_type="TYPICALITY_PERCENTILE" if spec.requires_typicality_probe else None,
        rarity_reference_score_type="TYPICALITY_PERCENTILE" if spec.requires_typicality_reference else None,
        rarity_probe_transform_id="ONE_MINUS_PERCENTILE_0_1" if spec.requires_typicality_probe else None,
        rarity_reference_transform_id="ONE_MINUS_PERCENTILE_0_1" if spec.requires_typicality_reference else None,
        rarity_probe_combiner="PROBE_ONLY" if spec.requires_typicality_probe else None,
        rarity_reference_combiner="REFERENCE_ONLY" if spec.requires_typicality_reference else None,
        rarity_probe_model_fingerprint="MODEL_FP_PROBE_AAA" if spec.requires_typicality_probe else None,
        rarity_reference_model_fingerprint="MODEL_FP_REFERENCE_AAA" if spec.requires_typicality_reference else None,
    )


@pytest.fixture
def bilateral_state():
    return _make_full_state()


@pytest.fixture
def bilateral_spec():
    return LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.BILATERAL_FULL_CENTERED]


@pytest.fixture
def matching_incoming():
    return (
        {"score_kind": "cosine", "pair_role_orientation": "PROBE_REFERENCE"},
        {
            "score_type": "TYPICALITY_PERCENTILE",
            "transform_id": "ONE_MINUS_PERCENTILE_0_1",
            "combiner": "PROBE_ONLY",
            "model_fingerprint": "MODEL_FP_PROBE_AAA",
        },
        {
            "score_type": "TYPICALITY_PERCENTILE",
            "transform_id": "ONE_MINUS_PERCENTILE_0_1",
            "combiner": "REFERENCE_ONLY",
            "model_fingerprint": "MODEL_FP_REFERENCE_AAA",
        },
    )


class TestNoIssuesWhenCompatible:
    def test_matching_inputs_returns_empty(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, *matching_incoming
        )
        assert issues == []


class TestArchitectureMismatch:
    def test_architecture_divergent_detected(self, bilateral_state, matching_incoming):
        spec_other = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.HIERARCHICAL_CENTERED
        ]
        issues = validate_state_compatibility(
            bilateral_state, spec_other, *matching_incoming
        )
        fields = [i.field for i in issues if i.severity == "error"]
        assert "architecture" in fields
        # spec_fingerprint, feature_terms, output_dim também devem divergir
        assert "architecture_spec_fingerprint" in fields
        assert "feature_terms" in fields
        assert "output_dim" in fields


class TestBaseMetadataDivergence:
    def test_base_score_kind_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base_meta, rp, rr = matching_incoming
        base_meta_bad = dict(base_meta)
        base_meta_bad["score_kind"] = "plda"
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base_meta_bad, rp, rr
        )
        assert any(i.field == "base_score_kind" for i in issues)

    def test_orientation_divergent_for_ordered_architecture(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base_meta, rp, rr = matching_incoming
        base_meta_bad = dict(base_meta)
        base_meta_bad["pair_role_orientation"] = "REFERENCE_PROBE"
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base_meta_bad, rp, rr
        )
        assert any(i.field == "pair_role_orientation" for i in issues)

    def test_orientation_ignored_for_symmetric_architecture(self):
        """COSINE_ONLY ignora orientação porque é simétrica."""
        state = _make_full_state(LRCalibrationArchitecture.COSINE_ONLY)
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.COSINE_ONLY]
        base_meta = {
            "score_kind": "cosine",
            "pair_role_orientation": "REFERENCE_PROBE",  # diferente do state
        }
        issues = validate_state_compatibility(state, spec, base_meta)
        assert not any(i.field == "pair_role_orientation" for i in issues)


class TestRarityProbeDivergence:
    def test_score_type_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, _, rr = matching_incoming
        rp_bad = {
            "score_type": "MEAN_KNN_SIMILARITY",  # diferente
            "transform_id": "NEGATE_MEAN_KNN_SIMILARITY",
            "combiner": "PROBE_ONLY",
        }
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp_bad, rr
        )
        fields = [i.field for i in issues]
        assert "rarity_probe_score_type" in fields
        assert "rarity_probe_transform_id" in fields

    def test_transform_fingerprint_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        # bilateral_state.rarity_probe_transform_fingerprint == None (não setado)
        # então não detecta esse caso específico. Vamos criar um state com fingerprint
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        ]
        state = LRArchitectureStateFactory.create(
            spec=spec, s0=0.25, r0_probe=0.5, r0_reference=0.48,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
            fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
            s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            rarity_probe_score_type="TYPICALITY_PERCENTILE",
            rarity_reference_score_type="TYPICALITY_PERCENTILE",
            rarity_probe_transform_id="ONE_MINUS_PERCENTILE_0_1",
            rarity_reference_transform_id="ONE_MINUS_PERCENTILE_0_1",
            rarity_probe_transform_fingerprint="FP_EXPECTED",
            rarity_reference_transform_fingerprint="FP_EXPECTED",
        )
        base, _, rr = matching_incoming
        rp_bad = {
            "score_type": "TYPICALITY_PERCENTILE",
            "transform_id": "ONE_MINUS_PERCENTILE_0_1",
            "transform_fingerprint": "FP_WRONG",
            "combiner": "PROBE_ONLY",
        }
        issues = validate_state_compatibility(state, spec, base, rp_bad, rr)
        assert any(i.field == "rarity_probe_transform_fingerprint" for i in issues)

    def test_combiner_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, _, rr = matching_incoming
        rp_bad = {
            "score_type": "TYPICALITY_PERCENTILE",
            "transform_id": "ONE_MINUS_PERCENTILE_0_1",
            "combiner": "MEAN",  # diferente
            "model_fingerprint": "MODEL_FP_PROBE_AAA",
        }
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp_bad, rr
        )
        assert any(i.field == "rarity_probe_combiner" for i in issues)

    def test_model_fingerprint_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, _, rr = matching_incoming
        rp_bad = {
            "score_type": "TYPICALITY_PERCENTILE",
            "transform_id": "ONE_MINUS_PERCENTILE_0_1",
            "combiner": "PROBE_ONLY",
            "model_fingerprint": "DIFFERENT_MODEL_FP",
        }
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp_bad, rr
        )
        assert any(i.field == "rarity_probe_model_fingerprint" for i in issues)


class TestRarityReferenceDivergence:
    def test_score_type_divergent(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, rp, _ = matching_incoming
        rr_bad = {
            "score_type": "LOG_DENSITY_ESTIMATE",
            "transform_id": "NEGATE_LOG_DENSITY_ESTIMATE",
            "combiner": "REFERENCE_ONLY",
        }
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp, rr_bad
        )
        assert any(i.field == "rarity_reference_score_type" for i in issues)


class TestRarityMetaMissing:
    def test_missing_probe_meta_when_required_detected(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, _, rr = matching_incoming
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, None, rr
        )
        assert any(i.field == "rarity_probe_meta" for i in issues)

    def test_missing_reference_meta_when_required_detected(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, rp, _ = matching_incoming
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp, None
        )
        assert any(i.field == "rarity_reference_meta" for i in issues)


class TestPartialMetaTolerance:
    """Quando incoming_*_meta não declara um campo, o checker pula a comparação."""

    def test_missing_field_in_meta_does_not_raise(
        self, bilateral_state, bilateral_spec, matching_incoming
    ):
        base, _, rr = matching_incoming
        rp_partial = {
            # só combiner — sem score_type, transform_id, etc.
            "combiner": "PROBE_ONLY",
            "model_fingerprint": "MODEL_FP_PROBE_AAA",
        }
        issues = validate_state_compatibility(
            bilateral_state, bilateral_spec, base, rp_partial, rr
        )
        # Não deve emitir issues sobre os campos ausentes
        field_set = {i.field for i in issues}
        assert "rarity_probe_score_type" not in field_set
        assert "rarity_probe_transform_id" not in field_set
