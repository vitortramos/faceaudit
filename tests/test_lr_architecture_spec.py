"""
Testes do registry `LR_ARCHITECTURE_SPECS` + fingerprint estável.

Cobre:
- Todas as 9 arquiteturas têm spec.
- Dimensões e propriedades estruturais (centering_mode, requires_*)
  conferem com a tabela canônica.
- `feature_names` reflete a centragem nos sufixos (`_RAW` vs
  `_CENTERED_MEAN`).
- `fingerprint()` é determinístico, distinto entre arquiteturas, e
  cobre `formula_version` de cada termo (drift de implementação).
"""

from __future__ import annotations

import pytest

from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.enums.rarity_centering_mode import RarityCenteringMode
from faceaudit.models.lr.lr_architecture_spec import (
    LR_ARCHITECTURE_SPECS,
    get_architecture_spec,
)


# Tabela canônica vinda do plano. Cada linha: (arquitetura, D, centering,
# requires_s0, requires_typ_probe, requires_typ_ref, requires_ordered,
# requires_typ_single).
# Default requires_typ_single = False (somente ONE_FACE preenche True).
CANONICAL_TABLE = [
    ("COSINE_ONLY", 1, "NONE", False, False, False, False, False),
    ("HINGE_ONLY", 2, "NONE", True, False, False, False, False),
    ("TWO_SIDED_HINGE_ONLY", 3, "NONE", True, False, False, False, False),
    ("ONE_SIDED_NON_CENTERED", 3, "NONE", True, True, True, True, False),
    ("ONE_SIDED_CENTERED", 3, "MEAN", True, True, True, True, False),
    ("LINEAR_NON_CENTERED", 3, "NONE", True, True, True, True, False),
    ("LINEAR_CENTERED", 3, "MEAN", True, True, True, True, False),
    ("HIERARCHICAL_CENTERED", 4, "MEAN", True, True, True, True, False),
    ("BILATERAL_FULL_CENTERED", 7, "MEAN", True, True, True, True, False),
    # ── Família ONE_FACE (V2.14.1) ───────────────────────────────────────
    ("ONE_SIDED_NON_CENTERED_ONE_FACE", 2, "NONE", True, False, False, False, True),
    ("ONE_SIDED_CENTERED_ONE_FACE", 2, "MEAN", True, False, False, False, True),
    ("LINEAR_NON_CENTERED_ONE_FACE", 2, "NONE", True, False, False, False, True),
    ("LINEAR_CENTERED_ONE_FACE", 2, "MEAN", True, False, False, False, True),
    ("HIERARCHICAL_CENTERED_ONE_FACE", 3, "MEAN", True, False, False, False, True),
    ("BILATERAL_FULL_CENTERED_ONE_FACE", 5, "MEAN", True, False, False, False, True),
]


class TestRegistryCoverage:
    def test_all_9_architectures_registered(self):
        # V2.14.1: 15 arquiteturas total (9 originais + 6 ONE_FACE).
        assert set(LR_ARCHITECTURE_SPECS.keys()) == set(LRCalibrationArchitecture)
        assert len(LR_ARCHITECTURE_SPECS) == 15

    @pytest.mark.parametrize("row", CANONICAL_TABLE, ids=lambda r: r[0])
    def test_canonical_properties(self, row):
        name, d, centering, req_s0, req_tp, req_tr, req_ord, req_ts = row
        arch = LRCalibrationArchitecture[name]
        spec = get_architecture_spec(arch)
        assert spec.output_dim == d
        assert spec.centering_mode.value == centering
        assert spec.requires_s0 == req_s0
        assert spec.requires_typicality_probe == req_tp
        assert spec.requires_typicality_reference == req_tr
        assert spec.requires_ordered_pair_roles == req_ord
        assert spec.requires_typicality_single == req_ts


class TestFeatureNames:
    def test_cosine_only_has_one_column(self):
        spec = get_architecture_spec(LRCalibrationArchitecture.COSINE_ONLY)
        assert spec.feature_names == ("BASE_SCORE",)

    def test_one_sided_non_centered_uses_raw_terms(self):
        spec = get_architecture_spec(LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED)
        assert all("_RAW" in n for n in spec.feature_names[1:])
        assert all("_CENTERED_MEAN" not in n for n in spec.feature_names)

    def test_one_sided_centered_uses_centered_terms(self):
        spec = get_architecture_spec(LRCalibrationArchitecture.ONE_SIDED_CENTERED)
        assert all(
            "_CENTERED_MEAN" in n for n in spec.feature_names[1:]
        ), spec.feature_names

    def test_bilateral_full_centered_has_7_columns_with_correct_terms(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        )
        expected = (
            "BASE_SCORE",
            "HINGE_POSITIVE",
            "HINGE_NEGATIVE",
            "HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN",
            "HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN",
            "HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN",
            "HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN",
        )
        assert spec.feature_names == expected


class TestFingerprintStability:
    def test_deterministic(self):
        spec = get_architecture_spec(
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        )
        assert spec.fingerprint() == spec.fingerprint()

    def test_distinct_between_architectures(self):
        fps = {
            arch.value: get_architecture_spec(arch).fingerprint()
            for arch in LRCalibrationArchitecture
        }
        # Todos distintos
        assert len(set(fps.values())) == len(fps)

    def test_distinct_between_non_centered_and_centered(self):
        spec_n = get_architecture_spec(LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED)
        spec_c = get_architecture_spec(LRCalibrationArchitecture.ONE_SIDED_CENTERED)
        # Mesma D=3 mas centering diferente — fingerprints distintos
        assert spec_n.output_dim == spec_c.output_dim
        assert spec_n.fingerprint() != spec_c.fingerprint()


class TestArchitectureBackwardCompat:
    def test_bilateral_simple_legacy_maps_to_two_sided_hinge_only(self):
        # JSONs antigos usavam BILATERAL_SIMPLE; from_str deve mapear.
        result = LRCalibrationArchitecture.from_str("BILATERAL_SIMPLE")
        assert result == LRCalibrationArchitecture.TWO_SIDED_HINGE_ONLY

    def test_unknown_value_falls_back_to_cosine_only(self):
        # Comportamento estabelecido em outros enums do projeto.
        result = LRCalibrationArchitecture.from_str("NOT_A_REAL_ARCH")
        assert result == LRCalibrationArchitecture.COSINE_ONLY
