"""
Testes do `CalibrationModelMetadataService`.

Cobre:
- Tabela assimétrica STRICT × LEGACY_PERMISSIVE para 5 combinações de
  fingerprint (ambos sem, iguais, diferentes, só dataset, só model).
- `bind_architecture_state` copia campos contratuais + provenance.
- Bind é no-op em dataset legado sem fingerprint.
"""

from __future__ import annotations

import warnings

import pytest

from faceaudit.enums.calibration_compatibility_mode import (
    CalibrationCompatibilityMode,
)
from faceaudit.services.calibration_model_metadata_service import (
    CalibrationModelMetadataService,
)


class TestValidateCompatibilityAsymmetricTable:
    def test_both_legacy_warns(self):
        issues = CalibrationModelMetadataService.validate_compatibility({}, {})
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_both_matching_fingerprints_ok(self):
        ds = {"vectorization_contract_fingerprint": "ABC"}
        mdl = {"vectorization_contract_fingerprint": "ABC"}
        assert CalibrationModelMetadataService.validate_compatibility(mdl, ds) == []

    def test_both_distinct_fingerprints_error(self):
        ds = {"vectorization_contract_fingerprint": "ABC"}
        mdl = {"vectorization_contract_fingerprint": "DEF"}
        issues = CalibrationModelMetadataService.validate_compatibility(mdl, ds)
        assert any(i.severity == "error" for i in issues)

    def test_ds_with_fp_model_without_fp_strict_error(self):
        ds = {"vectorization_contract_fingerprint": "ABC"}
        mdl = {}
        issues = CalibrationModelMetadataService.validate_compatibility(
            mdl, ds, compatibility_mode=CalibrationCompatibilityMode.STRICT,
        )
        assert any(i.severity == "error" for i in issues)

    def test_ds_with_fp_model_without_fp_legacy_permissive_warn(self):
        ds = {"vectorization_contract_fingerprint": "ABC"}
        mdl = {}
        with warnings.catch_warnings(record=True) as w_recorded:
            warnings.simplefilter("always")
            issues = CalibrationModelMetadataService.validate_compatibility(
                mdl, ds,
                compatibility_mode=CalibrationCompatibilityMode.LEGACY_PERMISSIVE,
            )
        # Warning issued, no errors
        assert any(i.severity == "warning" for i in issues)
        assert not any(i.severity == "error" for i in issues)
        # Sistema de warnings também invocado
        assert any("LEGACY_PERMISSIVE" in str(wi.message) for wi in w_recorded)

    def test_ds_without_fp_model_with_fp_always_error(self):
        """Combinação SEMPRE proibida, mesmo em LEGACY_PERMISSIVE."""
        ds = {}
        mdl = {"vectorization_contract_fingerprint": "ABC"}
        for mode in CalibrationCompatibilityMode:
            issues = CalibrationModelMetadataService.validate_compatibility(
                mdl, ds, compatibility_mode=mode,
            )
            assert any(i.severity == "error" for i in issues), f"Modo {mode} deveria gerar erro"


class TestBindArchitectureState:
    def test_no_op_when_dataset_legacy(self):
        mdl = {}
        CalibrationModelMetadataService.bind_architecture_state(mdl, {})
        assert mdl == {}

    def test_copies_contractual_fields(self):
        mdl = {}
        ds_meta = {
            "vectorization_contract_fingerprint": "ABC",
            "lr_architecture": "BILATERAL_FULL_CENTERED",
            "lr_architecture_formula_version": "1.0",
            "architecture_spec_fingerprint": "SPEC_FP",
        }
        CalibrationModelMetadataService.bind_architecture_state(mdl, ds_meta)
        assert mdl["vectorization_contract_fingerprint"] == "ABC"
        assert mdl["lr_architecture"] == "BILATERAL_FULL_CENTERED"
        assert mdl["architecture_spec_fingerprint"] == "SPEC_FP"

    def test_copies_training_provenance(self):
        mdl = {}
        ds_meta = {
            "vectorization_contract_fingerprint": "ABC",
            "score_vector_contract_fingerprint": "SVCFP",
            "score_vector_numeric_fingerprint": "SVNFP",
            "raw_input_matrix_fingerprint": "RAWFP",
            "pair_key_order_hash": "PAIRHASH",
        }
        CalibrationModelMetadataService.bind_architecture_state(mdl, ds_meta)
        assert mdl["training_score_vector_contract_fingerprint"] == "SVCFP"
        assert mdl["training_score_vector_numeric_fingerprint"] == "SVNFP"
        assert mdl["training_raw_input_matrix_fingerprint"] == "RAWFP"
        assert mdl["training_pair_key_order_hash"] == "PAIRHASH"

    def test_copies_state_dict(self):
        mdl = {}
        state_dict = {"s0": 0.25, "r0_probe": 0.5}
        ds_meta = {
            "vectorization_contract_fingerprint": "ABC",
            "architecture_state": state_dict,
        }
        CalibrationModelMetadataService.bind_architecture_state(mdl, ds_meta)
        assert mdl["architecture_state"] == state_dict
