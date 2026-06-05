"""
Testes do `RarityTransformRegistry` + `RarityTransformSpec`.

Cobre:
- 3 transformações canônicas registradas e funcionais.
- Custom transform via `register()` consumido por `to_canonical_rarity`.
- Duplicado sem override levanta ValueError.
- Override=True substitui com warning explícito.
- Range/NaN/inf rejeitados pela API canônica.
- `legacy_transform_typicality` preserva comportamento histórico (sem
  validar ranges).
- Testes sentinela: input fixo → output esperado por transformação
  canônica.
- `RarityTransformSpec.fingerprint()` é determinístico e muda quando
  formula_version muda.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.services.rarity_transform_service import (
    RarityTransformRegistry,
    RarityTransformSpec,
    legacy_transform_typicality,
    to_canonical_rarity,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reseta o registry antes de cada teste para isolamento."""
    RarityTransformRegistry._reset_for_tests()
    yield
    RarityTransformRegistry._reset_for_tests()


class TestCanonicalRegistrations:
    def test_three_canonical_registered(self):
        RarityTransformRegistry._ensure_default_registrations()
        registered = RarityTransformRegistry.list_registered()
        assert len(registered) == 3
        ids = {s.transform_id for s in registered}
        assert ids == {
            "ONE_MINUS_PERCENTILE_0_1",
            "NEGATE_MEAN_KNN_SIMILARITY",
            "NEGATE_LOG_DENSITY_KNN",
        }

    def test_one_minus_percentile(self):
        v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r, tid = to_canonical_rarity(v, TypicalityScoreType.TYPICALITY_PERCENTILE)
        np.testing.assert_array_equal(r, np.array([1.0, 0.75, 0.5, 0.25, 0.0]))
        assert tid == "ONE_MINUS_PERCENTILE_0_1"

    def test_negate_mean_knn(self):
        v = np.array([-0.5, 0.0, 0.5])
        r, tid = to_canonical_rarity(v, TypicalityScoreType.MEAN_KNN_SIMILARITY)
        np.testing.assert_array_equal(r, np.array([0.5, 0.0, -0.5]))
        assert tid == "NEGATE_MEAN_KNN_SIMILARITY"

    def test_negate_log_density(self):
        v = np.array([-100.0, -50.0, 0.0])
        r, tid = to_canonical_rarity(v, TypicalityScoreType.LOG_DENSITY_KNN)
        np.testing.assert_array_equal(r, np.array([100.0, 50.0, 0.0]))
        assert tid == "NEGATE_LOG_DENSITY_KNN"


class TestValidations:
    def test_percentile_above_max_rejected(self):
        with pytest.raises(ValueError, match="acima de max_input"):
            to_canonical_rarity(
                np.array([0.5, 1.5, 0.7]),
                TypicalityScoreType.TYPICALITY_PERCENTILE,
            )

    def test_percentile_below_min_rejected(self):
        with pytest.raises(ValueError, match="abaixo de min_input"):
            to_canonical_rarity(
                np.array([-0.1, 0.5, 0.7]),
                TypicalityScoreType.TYPICALITY_PERCENTILE,
            )

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="NaN"):
            to_canonical_rarity(
                np.array([0.5, np.nan, 0.7]),
                TypicalityScoreType.TYPICALITY_PERCENTILE,
            )

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="NaN"):  # mesma mensagem "NaN/inf"
            to_canonical_rarity(
                np.array([0.5, np.inf, 0.7]),
                TypicalityScoreType.TYPICALITY_PERCENTILE,
            )

    def test_not_1d_rejected(self):
        with pytest.raises(ValueError, match="1D"):
            to_canonical_rarity(
                np.array([[0.5], [0.6]]),
                TypicalityScoreType.TYPICALITY_PERCENTILE,
            )


class TestCustomRegistration:
    def test_duplicate_without_override_raises(self):
        RarityTransformRegistry._ensure_default_registrations()
        spec = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="CUSTOM",
            formula_version="1.0",
            transform_label="custom",
            fn=lambda v: v,
            min_input=0.0, max_input=1.0,
            higher_means_rarer=True,
        )
        with pytest.raises(ValueError, match="já registrado"):
            RarityTransformRegistry.register(spec)

    def test_override_warns_and_replaces(self):
        RarityTransformRegistry._ensure_default_registrations()
        new_spec = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="CUSTOM_REPLACE",
            formula_version="2.0",
            transform_label="custom v2",
            fn=lambda v: 2.0 * v,
            min_input=0.0, max_input=1.0,
            higher_means_rarer=True,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RarityTransformRegistry.register(new_spec, override=True)
            assert any("SUBSTITUINDO" in str(wi.message) for wi in w)
        # Agora a percentile_0_1 retorna 2*v
        r, tid = to_canonical_rarity(
            np.array([0.0, 0.5, 1.0]),
            TypicalityScoreType.TYPICALITY_PERCENTILE,
        )
        np.testing.assert_array_equal(r, np.array([0.0, 1.0, 2.0]))
        assert tid == "CUSTOM_REPLACE"


class TestSentinels:
    """
    Testes sentinela: input fixo → output esperado por transformação.
    Falha indica que `fn` mudou sem bumpar `formula_version` (drift).
    """

    SENTINEL_INPUT = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_percentile_sentinel(self):
        RarityTransformRegistry._ensure_default_registrations()
        r, _ = to_canonical_rarity(
            self.SENTINEL_INPUT, TypicalityScoreType.TYPICALITY_PERCENTILE
        )
        np.testing.assert_array_equal(r, np.array([1.0, 0.75, 0.5, 0.25, 0.0]))

    def test_mean_knn_sentinel(self):
        RarityTransformRegistry._ensure_default_registrations()
        v = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        r, _ = to_canonical_rarity(v, TypicalityScoreType.MEAN_KNN_SIMILARITY)
        np.testing.assert_array_equal(r, np.array([1.0, 0.5, 0.0, -0.5, -1.0]))


class TestRarityTransformSpecFingerprint:
    def test_deterministic(self):
        spec = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="ONE_MINUS_PERCENTILE_0_1",
            formula_version="1.0",
            transform_label="x",
            fn=lambda v: 1.0 - v,
            min_input=0.0, max_input=1.0,
            higher_means_rarer=True,
        )
        assert spec.fingerprint() == spec.fingerprint()

    def test_changes_on_formula_version_bump(self):
        spec_v1 = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="T", formula_version="1.0",
            transform_label="x", fn=lambda v: v,
            min_input=0.0, max_input=1.0, higher_means_rarer=True,
        )
        spec_v2 = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="T", formula_version="2.0",  # bump
            transform_label="x", fn=lambda v: v,
            min_input=0.0, max_input=1.0, higher_means_rarer=True,
        )
        assert spec_v1.fingerprint() != spec_v2.fingerprint()

    def test_label_change_does_not_affect_fingerprint(self):
        """transform_label é cosmético; não entra no fingerprint."""
        spec_a = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="T", formula_version="1.0",
            transform_label="label A", fn=lambda v: v,
            min_input=0.0, max_input=1.0, higher_means_rarer=True,
        )
        spec_b = RarityTransformSpec(
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            transform_id="T", formula_version="1.0",
            transform_label="label B (diferente)", fn=lambda v: v,
            min_input=0.0, max_input=1.0, higher_means_rarer=True,
        )
        assert spec_a.fingerprint() == spec_b.fingerprint()


class TestLegacyTransform:
    """
    `legacy_transform_typicality` preserva semântica antiga:
    - Aceita valores fora de range sem erro (tolerância histórica).
    - Usa IDs distintos dos canônicos.
    - Suporta flag `invert`.
    """

    def test_accepts_out_of_range(self):
        r, tid = legacy_transform_typicality(
            np.array([0.5, 1.5, -0.1]),
            TypicalityScoreType.TYPICALITY_PERCENTILE,
            invert=True,
        )
        np.testing.assert_array_equal(r, np.array([0.5, -0.5, 1.1]))
        assert tid == "1 - percentile_0_1"  # ID legado distinto do canônico

    def test_invert_false_passthrough(self):
        r, tid = legacy_transform_typicality(
            np.array([0.0, 0.5, 1.0]),
            TypicalityScoreType.TYPICALITY_PERCENTILE,
            invert=False,
        )
        np.testing.assert_array_equal(r, np.array([0.0, 0.5, 1.0]))
        assert tid == "percentile_0_1"

    def test_mean_knn_invert(self):
        r, tid = legacy_transform_typicality(
            np.array([-0.5, 0.0, 0.5]),
            TypicalityScoreType.MEAN_KNN_SIMILARITY,
            invert=True,
        )
        np.testing.assert_array_equal(r, np.array([0.5, 0.0, -0.5]))
        assert tid == "-mean_knn_similarity"
