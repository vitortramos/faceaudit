"""
Testes da `LRArchitectureStateFactory` + integridade do
`LRArchitectureState`.

Cobre:
- Construção via factory normaliza enums + valida coerência com spec
  (centering, s0, papéis ordenados).
- `from_dict` recalcula fingerprints e levanta `StateIntegrityError`
  em adulteração.
- `has_internal_cv_parameter_leakage_risk` reflete fit_scope corretamente.
- Roundtrip `to_dict` / `from_dict` preserva todos os campos.
- Dois fingerprints distintos: `vectorization_contract` vs
  `provenance_record`.
"""

from __future__ import annotations

import pytest

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.lr.lr_architecture_state import (
    LRArchitectureStateFactory,
    StateIntegrityError,
)


@pytest.fixture
def bilateral_state():
    spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.BILATERAL_FULL_CENTERED]
    return LRArchitectureStateFactory.create(
        spec=spec,
        s0=0.253,
        r0_probe=0.4566,
        r0_reference=0.4962,
        base_score_kind="cosine",
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
        pair_key_mode=PairKeyMode.ORDERED,
        rarity_probe_score_type="TYPICALITY_PERCENTILE",
        rarity_reference_score_type="TYPICALITY_PERCENTILE",
        rarity_probe_transform_id="ONE_MINUS_PERCENTILE_0_1",
        rarity_reference_transform_id="ONE_MINUS_PERCENTILE_0_1",
        fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
        s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
        r0_probe_estimator="MEAN",
        r0_reference_estimator="MEAN",
        n_fitted_pairs=186624,
        n_fitted_genuine_pairs=1728,
        n_fitted_impostor_pairs=184896,
    )


@pytest.fixture
def cosine_state():
    spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.COSINE_ONLY]
    return LRArchitectureStateFactory.create(
        spec=spec,
        base_score_kind="cosine",
        pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
        fit_scope=ArchitectureFitScope.NOT_APPLICABLE,
        s0_resolution_method=S0ResolutionMethod.NOT_APPLICABLE,
    )


class TestCenteringConsistency:
    """Spec.centering_mode == NONE não admite r0_*; MEAN exige."""

    def test_none_with_r0_probe_raises(self):
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED
        ]
        with pytest.raises(ValueError, match="centering=NONE"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25, r0_probe=0.5,  # inválido
                pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
                fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
                s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            )

    def test_mean_without_r0_probe_raises_when_fit_applicable(self):
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        ]
        with pytest.raises(ValueError, match="r0_probe é obrigatório"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25,
                # r0_probe ausente
                r0_reference=0.5,
                pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
                fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
                s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            )


class TestS0Consistency:
    def test_cosine_only_with_s0_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.COSINE_ONLY]
        with pytest.raises(ValueError, match="não usa s₀"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25,
                pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
                fit_scope=ArchitectureFitScope.NOT_APPLICABLE,
                s0_resolution_method=S0ResolutionMethod.NOT_APPLICABLE,
            )

    def test_hinge_only_without_s0_when_fit_applicable_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.HINGE_ONLY]
        with pytest.raises(ValueError, match="s0 é obrigatório"):
            LRArchitectureStateFactory.create(
                spec=spec,  # s0 ausente
                pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
                fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
                s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            )


class TestPairRoleOrientation:
    def test_ordered_architecture_rejects_not_applicable(self):
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        ]
        with pytest.raises(ValueError, match="distingue probe/reference"):
            LRArchitectureStateFactory.create(
                spec=spec, s0=0.25, r0_probe=0.5, r0_reference=0.5,
                pair_role_orientation=PairRoleOrientation.NOT_APPLICABLE,
                fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
                s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            )

    def test_symmetric_architecture_accepts_not_applicable(self, cosine_state):
        # cosine_state já foi criado com NOT_APPLICABLE
        assert (
            cosine_state.pair_role_orientation == PairRoleOrientation.NOT_APPLICABLE
        )


class TestLeakageRiskFlag:
    def test_cosine_has_no_risk(self, cosine_state):
        assert not cosine_state.has_internal_cv_parameter_leakage_risk

    def test_bilateral_legacy_has_risk(self, bilateral_state):
        assert bilateral_state.has_internal_cv_parameter_leakage_risk


class TestRoundtrip:
    def test_to_dict_from_dict_preserves(self, bilateral_state):
        d = bilateral_state.to_dict()
        back = LRArchitectureStateFactory.from_dict(d)
        assert back.s0 == bilateral_state.s0
        assert back.r0_probe == bilateral_state.r0_probe
        assert back.r0_reference == bilateral_state.r0_reference
        assert (
            back.vectorization_contract_fingerprint
            == bilateral_state.vectorization_contract_fingerprint
        )
        assert (
            back.provenance_record_fingerprint
            == bilateral_state.provenance_record_fingerprint
        )
        assert back.feature_terms == bilateral_state.feature_terms

    def test_cosine_roundtrip(self, cosine_state):
        d = cosine_state.to_dict()
        back = LRArchitectureStateFactory.from_dict(d)
        assert back.s0 is None
        assert (
            back.vectorization_contract_fingerprint
            == cosine_state.vectorization_contract_fingerprint
        )


class TestIntegrityDetection:
    def test_adulterated_s0_detected(self, bilateral_state):
        d = bilateral_state.to_dict()
        d["s0"] = 0.999  # adulteração
        with pytest.raises(StateIntegrityError, match="contract_fingerprint"):
            LRArchitectureStateFactory.from_dict(d)

    def test_adulterated_r0_probe_detected(self, bilateral_state):
        d = bilateral_state.to_dict()
        d["r0_probe"] = 0.0  # adulteração
        with pytest.raises(StateIntegrityError):
            LRArchitectureStateFactory.from_dict(d)

    def test_adulterated_fitted_resource_id_detected(self, bilateral_state):
        d = bilateral_state.to_dict()
        d["fitted_on_resource_id"] = "tampered_id"
        with pytest.raises(StateIntegrityError, match="provenance_record"):
            LRArchitectureStateFactory.from_dict(d)

    def test_legacy_state_without_fingerprints_warns_not_raises(self, bilateral_state):
        """States sem fingerprints (legados) passam com warning."""
        import warnings as _w
        d = bilateral_state.to_dict()
        d["vectorization_contract_fingerprint"] = ""
        d["provenance_record_fingerprint"] = ""
        with _w.catch_warnings(record=True) as w_recorded:
            _w.simplefilter("always")
            back = LRArchitectureStateFactory.from_dict(d)
            assert any("modo legado" in str(wi.message) for wi in w_recorded)
        assert back.s0 == bilateral_state.s0


class TestFingerprintSeparation:
    """contract != provenance; volátil só muda provenance."""

    def test_two_fingerprints_distinct(self, bilateral_state):
        assert (
            bilateral_state.vectorization_contract_fingerprint
            != bilateral_state.provenance_record_fingerprint
        )

    def test_volatile_changes_only_provenance(self):
        """
        Dois states com mesma matemática mas IDs diferentes têm o mesmo
        contract_fingerprint e provenance_fingerprint distintos.
        """
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.BILATERAL_FULL_CENTERED]
        common = dict(
            spec=spec, s0=0.25, r0_probe=0.5, r0_reference=0.5,
            base_score_kind="cosine",
            pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
            fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
            s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
            r0_probe_estimator="MEAN", r0_reference_estimator="MEAN",
            rarity_probe_score_type="TYPICALITY_PERCENTILE",
            rarity_reference_score_type="TYPICALITY_PERCENTILE",
        )
        a = LRArchitectureStateFactory.create(**common, fitted_on_resource_id="A")
        b = LRArchitectureStateFactory.create(**common, fitted_on_resource_id="B")
        assert a.vectorization_contract_fingerprint == b.vectorization_contract_fingerprint
        assert a.provenance_record_fingerprint != b.provenance_record_fingerprint
