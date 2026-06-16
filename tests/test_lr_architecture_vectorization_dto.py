"""
Testes do DTO `LRArchitectureVectorizationInstruction`.

Cobre: visibility per-arquitetura, REACTIVE_FIELDS, roundtrip
to_dict/from_dict, declared_io sync.
"""

from __future__ import annotations

import pytest

from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
    LRArchitectureVectorizationInstruction,
)


def _mk(**overrides):
    """Constrói DTO mínimo com defaults razoáveis."""
    base = dict(
        instruction_id="ix",
        name="LR Vec Test",
        instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
        base_score_resource_id="base",
        output_resource_id="out",
    )
    base.update(overrides)
    return LRArchitectureVectorizationInstruction(**base)


class TestHiddenFieldsPerArchitecture:
    def test_cosine_only_hides_typicality_and_s0(self):
        instr = _mk(architecture=LRCalibrationArchitecture.COSINE_ONLY)
        h = instr.get_hidden_fields()
        assert "rarity_probe_resource_id" in h
        assert "rarity_reference_resource_id" in h
        assert "s0_logreg_config_override" in h

    def test_hinge_only_hides_typicality_only(self):
        instr = _mk(architecture=LRCalibrationArchitecture.HINGE_ONLY)
        h = instr.get_hidden_fields()
        assert "rarity_probe_resource_id" in h
        assert "rarity_reference_resource_id" in h
        # Mas s0_logreg_config_override visível (HINGE_ONLY usa s₀)
        assert "s0_logreg_config_override" not in h

    def test_bilateral_shows_all_inputs_and_s0_config(self):
        instr = _mk(architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED)
        h = instr.get_hidden_fields()
        assert "rarity_probe_resource_id" not in h
        assert "rarity_reference_resource_id" not in h
        assert "s0_logreg_config_override" not in h

    def test_manual_mode_shows_manual_fields(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.MANUAL,
        )
        h = instr.get_hidden_fields()
        assert "manual_s0" not in h
        assert "manual_r0_probe" not in h
        assert "manual_r0_reference" not in h
        assert "s0_logreg_config_override" in h  # MANUAL não usa AUTO_LOGREG

    def test_inherit_mode_shows_only_inherit_field(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.INHERIT,
        )
        h = instr.get_hidden_fields()
        assert "inherit_architecture_state_from_resource_id" not in h
        assert "manual_s0" in h
        assert "s0_logreg_config_override" in h

    def test_manual_per_architecture_hides_r0_for_non_centered(self):
        """ONE_SIDED_NON_CENTERED (centering=NONE) — manual_r0_* devem ficar escondidos."""
        instr = _mk(
            architecture=LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED,
            parameter_mode=ArchitectureParameterMode.MANUAL,
        )
        h = instr.get_hidden_fields()
        assert "manual_s0" not in h  # NON_CENTERED ainda usa s₀
        assert "manual_r0_probe" in h
        assert "manual_r0_reference" in h


class TestRoundtrip:
    def test_to_dict_serializes_enums_to_strings(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.INHERIT,
            inherit_architecture_state_from_resource_id="prev",
            rarity_probe_resource_id="rp",
            rarity_reference_resource_id="rr",
        )
        d = instr.to_dict()
        assert d["architecture"] == "BILATERAL_FULL_CENTERED"
        assert d["parameter_mode"] == "INHERIT"

    def test_from_dict_rehydrates_enums(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.HIERARCHICAL_CENTERED,
            parameter_mode=ArchitectureParameterMode.MANUAL,
            manual_s0=0.25,
            manual_r0_probe=0.5,
            manual_r0_reference=0.4,
            rarity_probe_resource_id="rp",
            rarity_reference_resource_id="rr",
        )
        d = instr.to_dict()
        back = LRArchitectureVectorizationInstruction.from_dict(d)
        assert back.architecture == LRCalibrationArchitecture.HIERARCHICAL_CENTERED
        assert back.parameter_mode == ArchitectureParameterMode.MANUAL
        assert back.manual_s0 == 0.25
        assert back.base_score_resource_id == "base"

    def test_from_dict_handles_legacy_bilateral_simple(self):
        instr = _mk(architecture=LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY)
        d = instr.to_dict()
        d["architecture"] = "BILATERAL_SIMPLE"  # nome legado
        back = LRArchitectureVectorizationInstruction.from_dict(d)
        assert back.architecture == LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY


class TestDeclaredIO:
    def test_inputs_include_all_used_resources(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            rarity_probe_resource_id="rp",
            rarity_reference_resource_id="rr",
            output_raw_input_resource_id="out_raw",
        )
        assert "base" in instr.input_resource_ids
        assert "rp" in instr.input_resource_ids
        assert "rr" in instr.input_resource_ids
        assert "out" in instr.output_resource_ids
        assert "out_raw" in instr.output_resource_ids

    def test_inherit_resource_counts_as_input(self):
        instr = _mk(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            parameter_mode=ArchitectureParameterMode.INHERIT,
            inherit_architecture_state_from_resource_id="state_origin",
            rarity_probe_resource_id="rp",
            rarity_reference_resource_id="rr",
        )
        assert "state_origin" in instr.input_resource_ids
