"""
Testes dos 9 enums novos do subsistema LR.

Cobre roundtrip from_str/to_str, backward-compat de `BILATERAL_SIMPLE`,
e propriedades semânticas adicionais.
"""

from __future__ import annotations

import pytest

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.rarity_centering_mode import RarityCenteringMode
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.enums.strategy_phase import StrategyPhase


class TestRoundtrip:
    """Cada enum: from_str(member.value) deve retornar o membro."""

    @pytest.mark.parametrize("enum_cls", [
        LRCalibrationArchitecture,
        LRFeatureTerm,
        RarityCenteringMode,
        ArchitectureParameterMode,
        ArchitectureFitScope,
        S0ResolutionMethod,
        PairRoleOrientation,
        PairKeyMode,
        StrategyPhase,
    ])
    def test_roundtrip_all_members(self, enum_cls):
        for member in enum_cls:
            result = enum_cls.from_str(member.value)
            assert result == member, f"{enum_cls.__name__}.{member.name} roundtrip falhou"

    @pytest.mark.parametrize("enum_cls", [
        LRCalibrationArchitecture,
        RarityCenteringMode,
        ArchitectureParameterMode,
        ArchitectureFitScope,
        S0ResolutionMethod,
        PairRoleOrientation,
        PairKeyMode,
        StrategyPhase,
    ])
    def test_from_str_idempotent(self, enum_cls):
        """from_str(member) == member."""
        for member in enum_cls:
            assert enum_cls.from_str(member) == member


class TestBackwardCompat:
    def test_bilateral_simple_maps_to_two_sided_hinge_only(self):
        assert (
            LRCalibrationArchitecture.from_str("BILATERAL_SIMPLE")
            == LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY
        )


class TestSemanticFlags:
    def test_lr_feature_term_flags(self):
        assert LRFeatureTerm.BASE_SCORE.requires_s0 is False
        assert LRFeatureTerm.HINGE_POSITIVE.requires_s0
        assert LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW.consumes_rarity_probe
        assert LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW.consumes_hinge_positive
        assert (
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN.is_centered
        )
        assert (
            LRFeatureTerm.HINGE_NEGATIVE.consumes_hinge_negative
            and not LRFeatureTerm.HINGE_NEGATIVE.consumes_hinge_positive
        )

    def test_strategy_phase_allows_fitting(self):
        assert StrategyPhase.PREPARATION.allows_parameter_fitting
        assert not StrategyPhase.VALIDATION.allows_parameter_fitting
        assert not StrategyPhase.CASEWORK.allows_parameter_fitting

    def test_architecture_parameter_mode_inherit(self):
        assert ArchitectureParameterMode.INHERIT.requires_inherit_resource
        assert ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY.is_fit_mode
        assert ArchitectureParameterMode.FIT_PER_FOLD.is_fit_mode
        assert not ArchitectureParameterMode.MANUAL.is_fit_mode

    def test_architecture_fit_scope_leakage_risk(self):
        assert ArchitectureFitScope.FULL_CALIBRATION_LEGACY.has_leakage_risk
        assert not ArchitectureFitScope.PER_FOLD_REFIT_FULL_CALIBRATION.has_leakage_risk
        assert not ArchitectureFitScope.NOT_APPLICABLE.has_leakage_risk
        assert not ArchitectureFitScope.MANUAL.has_leakage_risk

    def test_pair_role_and_key_are_independent(self):
        """PairRoleOrientation e PairKeyMode são enums distintos."""
        assert PairRoleOrientation.PROBE_REFERENCE.value != PairKeyMode.ORDERED.value
        # Diferentes APIs — não há sobreposição semântica
        assert isinstance(PairRoleOrientation.NOT_APPLICABLE, PairRoleOrientation)
        assert isinstance(PairKeyMode.UNORDERED, PairKeyMode)
