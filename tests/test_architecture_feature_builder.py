"""
Testes do `ArchitectureFeatureBuilder`.

Cobre:
- `build()` consistente com `derivative_matrix_wrt_base()` em todas as
  9 arquiteturas (numérica vs analítica).
- Validações: arquitetura exige tipicidade mas array ausente → ValueError;
  arquitetura exige s0 mas state.s0=None → ValueError; shape mismatch
  detectado.
- Matriz tem forma `(n, spec.output_dim)` exata.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.services import architecture_feature_builder as builder


def _make_state(arch: LRCalibrationArchitecture):
    """Constrói um state-like com s0 e r0_* baseados no spec."""
    spec = LR_ARCHITECTURE_SPECS[arch]
    return SimpleNamespace(
        s0=0.25 if spec.requires_s0 else None,
        r0_probe=0.5 if (spec.requires_typicality_probe and spec.centering_mode.value == "MEAN") else None,
        r0_reference=0.48 if (spec.requires_typicality_reference and spec.centering_mode.value == "MEAN") else None,
    )


def _inputs(arch: LRCalibrationArchitecture, n: int = 10):
    """Gera (base, r_F, r_P) consistentes com o que a arquitetura consome."""
    spec = LR_ARCHITECTURE_SPECS[arch]
    rng = np.random.RandomState(42)
    base = rng.uniform(-0.2, 0.8, n).astype(np.float64)
    r_F = rng.uniform(0.0, 1.0, n).astype(np.float64) if spec.requires_typicality_probe else None
    r_P = rng.uniform(0.0, 1.0, n).astype(np.float64) if spec.requires_typicality_reference else None
    return base, r_F, r_P


class TestOutputShape:
    @pytest.mark.parametrize("arch", list(LRCalibrationArchitecture))
    def test_build_returns_correct_shape(self, arch):
        spec = LR_ARCHITECTURE_SPECS[arch]
        state = _make_state(arch)
        base, r_F, r_P = _inputs(arch, n=15)
        X = builder.build(spec, state, base, r_F, r_P)
        assert X.shape == (15, spec.output_dim)
        assert X.dtype == np.float64

    @pytest.mark.parametrize("arch", list(LRCalibrationArchitecture))
    def test_derivative_returns_correct_shape(self, arch):
        spec = LR_ARCHITECTURE_SPECS[arch]
        state = _make_state(arch)
        base, r_F, r_P = _inputs(arch, n=15)
        D = builder.derivative_matrix_wrt_base(spec, state, base, r_F, r_P)
        assert D.shape == (15, spec.output_dim)
        assert D.dtype == np.float64


class TestNumericalVsAnalytical:
    """
    Verifica que a derivada analítica do builder coincide com a derivada
    numérica obtida por finite differences. Avalia em pares longe de
    s₀ para evitar a descontinuidade da hinge.
    """

    @pytest.mark.parametrize("arch", list(LRCalibrationArchitecture))
    def test_derivative_matches_finite_differences(self, arch):
        spec = LR_ARCHITECTURE_SPECS[arch]
        state = _make_state(arch)

        # Bases longe de s0=0.25 para todos os pares
        rng = np.random.RandomState(7)
        base = np.concatenate([
            rng.uniform(-0.2, 0.1, 5),  # bem abaixo de s0
            rng.uniform(0.4, 0.8, 5),   # bem acima
        ])
        if spec.requires_typicality_probe:
            r_F = rng.uniform(0.1, 0.9, 10)
        else:
            r_F = None
        if spec.requires_typicality_reference:
            r_P = rng.uniform(0.1, 0.9, 10)
        else:
            r_P = None

        eps = 1e-6
        X_plus = builder.build(spec, state, base + eps, r_F, r_P)
        X_minus = builder.build(spec, state, base - eps, r_F, r_P)
        D_num = (X_plus - X_minus) / (2 * eps)

        D_analytic = builder.derivative_matrix_wrt_base(spec, state, base, r_F, r_P)

        np.testing.assert_allclose(
            D_num, D_analytic, atol=1e-5,
            err_msg=f"Derivada divergiu para {arch.value}",
        )


class TestMissingInputs:
    def test_missing_s0_when_required_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.HINGE_ONLY]
        state = SimpleNamespace(s0=None, r0_probe=None, r0_reference=None)
        base = np.array([0.1, 0.3])
        with pytest.raises(ValueError, match="exige s₀"):
            builder.build(spec, state, base)

    def test_missing_r0_probe_when_centering_raises(self):
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        ]
        state = SimpleNamespace(s0=0.25, r0_probe=None, r0_reference=0.5)
        base = np.array([0.1, 0.3])
        r_F = np.array([0.5, 0.5])
        r_P = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="r0_probe"):
            builder.build(spec, state, base, r_F, r_P)

    def test_missing_rarity_probe_array_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED]
        state = SimpleNamespace(s0=0.25, r0_probe=None, r0_reference=None)
        base = np.array([0.1, 0.3])
        with pytest.raises(ValueError, match="rarity_probe"):
            builder.build(spec, state, base)


class TestShapeErrors:
    def test_2d_base_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.COSINE_ONLY]
        state = SimpleNamespace(s0=None, r0_probe=None, r0_reference=None)
        with pytest.raises(ValueError, match="base deve ser"):
            builder.build(spec, state, np.array([[1.0]]))

    def test_rarity_shape_mismatch_raises(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.ONE_SIDED_NON_CENTERED]
        state = SimpleNamespace(s0=0.25, r0_probe=None, r0_reference=None)
        base = np.array([0.1, 0.3, 0.5])
        r_F = np.array([0.4, 0.6])  # shape diferente
        r_P = np.array([0.4, 0.6, 0.8])
        with pytest.raises(ValueError, match="shape"):
            builder.build(spec, state, base, r_F, r_P)


class TestCorrectnessAgainstExpectedFormulas:
    """Verifica algumas fórmulas explicitamente."""

    def test_cosine_only(self):
        spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.COSINE_ONLY]
        state = SimpleNamespace(s0=None, r0_probe=None, r0_reference=None)
        base = np.array([0.1, 0.5, 0.9])
        X = builder.build(spec, state, base)
        np.testing.assert_array_equal(X[:, 0], base)

    def test_bilateral_full_centered_formulas(self):
        spec = LR_ARCHITECTURE_SPECS[
            LRCalibrationArchitecture.BILATERAL_FULL_CENTERED
        ]
        state = SimpleNamespace(s0=0.25, r0_probe=0.5, r0_reference=0.48)
        base = np.array([-0.1, 0.0, 0.25, 0.5, 0.8])
        r_F = np.array([0.1, 0.5, 0.7, 0.4, 0.9])
        r_P = np.array([0.2, 0.3, 0.6, 0.8, 0.5])
        X = builder.build(spec, state, base, r_F, r_P)

        # col 0 = s
        np.testing.assert_allclose(X[:, 0], base)
        # col 1 = h+ = max(s-0.25, 0)
        np.testing.assert_allclose(X[:, 1], np.array([0.0, 0.0, 0.0, 0.25, 0.55]))
        # col 2 = h- = max(0.25-s, 0)
        np.testing.assert_allclose(X[:, 2], np.array([0.35, 0.25, 0.0, 0.0, 0.0]))
        # col 3 = h+ * (r_F - 0.5)
        np.testing.assert_allclose(X[:, 3], X[:, 1] * (r_F - 0.5))
        # col 4 = h+ * (r_P - 0.48)
        np.testing.assert_allclose(X[:, 4], X[:, 1] * (r_P - 0.48))
        # col 5 = h- * (r_F - 0.5)
        np.testing.assert_allclose(X[:, 5], X[:, 2] * (r_F - 0.5))
        # col 6 = h- * (r_P - 0.48)
        np.testing.assert_allclose(X[:, 6], X[:, 2] * (r_P - 0.48))
