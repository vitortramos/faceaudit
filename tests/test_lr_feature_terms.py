"""
Testes unitários do registry de termos `LR_FEATURE_TERM_SPECS` + do
contexto `LRFeatureBuildContext`. Cobre:

- Todos os 13 termos têm spec registrada.
- BASE_SCORE: φ=s, ∂φ/∂s=1.
- HINGE_POSITIVE / HINGE_NEGATIVE: fórmula + derivada à direita + slopes
  separados (`left_slope_at_s0`, `right_slope_at_s0`).
- Interações com raridade (raw + centradas): fórmula + derivada.
- Lineares: fórmula + derivada constante = raridade.
- Derivada analítica vs numérica (finite-differences) em pontos s ≠ s₀.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.lr_feature_term import LRFeatureTerm
from faceaudit.models.lr.lr_feature_term_spec import (
    LR_FEATURE_TERM_SPECS,
    LRFeatureBuildContext,
    get_term_spec,
)


@pytest.fixture
def ctx() -> LRFeatureBuildContext:
    """Contexto com s0=0.25 e raridades pré-calculadas."""
    base = np.array([-0.1, 0.0, 0.25, 0.5, 0.8], dtype=np.float64)
    s0 = 0.25
    r_F = np.array([0.1, 0.5, 0.7, 0.4, 0.9], dtype=np.float64)
    r_P = np.array([0.2, 0.3, 0.6, 0.8, 0.5], dtype=np.float64)
    r0_F, r0_P = 0.5, 0.48
    return LRFeatureBuildContext(
        base=base, s0=s0, delta=base - s0,
        hinge_positive=np.maximum(base - s0, 0.0),
        hinge_negative=np.maximum(s0 - base, 0.0),
        rarity_probe_raw=r_F, rarity_reference_raw=r_P,
        rarity_probe_centered=r_F - r0_F,
        rarity_reference_centered=r_P - r0_P,
    )


class TestRegistryCoverage:
    def test_all_terms_registered(self):
        # V2.14.1: 18 termos (13 originais + 5 ONE_FACE).
        assert set(LR_FEATURE_TERM_SPECS.keys()) == set(LRFeatureTerm)
        assert len(LR_FEATURE_TERM_SPECS) == 18

    def test_get_term_spec_raises_on_unknown(self):
        # Sem KeyError customizado — KeyError nativo do dict
        with pytest.raises(KeyError):
            LR_FEATURE_TERM_SPECS["foo"]  # type: ignore[index]


class TestBaseScore:
    def test_value_equals_base(self, ctx):
        spec = get_term_spec(LRFeatureTerm.BASE_SCORE)
        np.testing.assert_array_equal(spec.build_fn(ctx), ctx.base)

    def test_derivative_is_one(self, ctx):
        spec = get_term_spec(LRFeatureTerm.BASE_SCORE)
        np.testing.assert_array_equal(spec.derivative_wrt_base_fn(ctx), np.ones(5))

    def test_no_kink(self, ctx):
        spec = get_term_spec(LRFeatureTerm.BASE_SCORE)
        assert not spec.handles_kink_at_s0
        assert spec.left_slope_at_s0_fn is None


class TestHingePositive:
    def test_formula(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_POSITIVE)
        # base = [-0.1, 0.0, 0.25, 0.5, 0.8], s0=0.25
        expected = np.array([0.0, 0.0, 0.0, 0.25, 0.55])
        np.testing.assert_allclose(spec.build_fn(ctx), expected, atol=1e-12)

    def test_derivative_right(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_POSITIVE)
        # 1 se s > 0.25, 0 senão
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(spec.derivative_wrt_base_fn(ctx), expected)

    def test_slopes_at_s0(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_POSITIVE)
        assert spec.handles_kink_at_s0
        np.testing.assert_array_equal(spec.left_slope_at_s0_fn(ctx), np.zeros(5))
        np.testing.assert_array_equal(spec.right_slope_at_s0_fn(ctx), np.ones(5))


class TestHingeNegative:
    def test_formula(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_NEGATIVE)
        # h⁻ = max(0.25 - s, 0)
        expected = np.array([0.35, 0.25, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(spec.build_fn(ctx), expected, atol=1e-12)

    def test_derivative_right(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_NEGATIVE)
        # -1 se s < 0.25, 0 senão
        expected = np.array([-1.0, -1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(spec.derivative_wrt_base_fn(ctx), expected)

    def test_slopes_at_s0(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_NEGATIVE)
        np.testing.assert_array_equal(spec.left_slope_at_s0_fn(ctx), -np.ones(5))
        np.testing.assert_array_equal(spec.right_slope_at_s0_fn(ctx), np.zeros(5))


class TestRarityInteractionTerms:
    """Testa h⁺·r_F, h⁻·r_P (centradas e brutas), lineares."""

    def test_hp_x_rf_raw(self, ctx):
        spec = get_term_spec(LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW)
        expected = ctx.hinge_positive * ctx.rarity_probe_raw
        np.testing.assert_allclose(spec.build_fn(ctx), expected)
        # derivada = 1{s>s0} * r_F
        deriv = (ctx.base > ctx.s0).astype(float) * ctx.rarity_probe_raw
        np.testing.assert_allclose(spec.derivative_wrt_base_fn(ctx), deriv)

    def test_hp_x_rf_centered(self, ctx):
        spec = get_term_spec(
            LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN
        )
        expected = ctx.hinge_positive * ctx.rarity_probe_centered
        np.testing.assert_allclose(spec.build_fn(ctx), expected)

    def test_hn_x_rf_centered(self, ctx):
        spec = get_term_spec(
            LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN
        )
        expected = ctx.hinge_negative * ctx.rarity_probe_centered
        np.testing.assert_allclose(spec.build_fn(ctx), expected)
        # derivada = -1{s<s0} * r̃_F
        deriv = -(ctx.base < ctx.s0).astype(float) * ctx.rarity_probe_centered
        np.testing.assert_allclose(spec.derivative_wrt_base_fn(ctx), deriv)

    def test_linear_x_rf_raw_constant_deriv(self, ctx):
        spec = get_term_spec(LRFeatureTerm.LINEAR_X_RARITY_PROBE_RAW)
        # φ = (s - s0) * r_F
        expected = ctx.delta * ctx.rarity_probe_raw
        np.testing.assert_allclose(spec.build_fn(ctx), expected)
        # derivada = r_F (constante em s)
        np.testing.assert_allclose(spec.derivative_wrt_base_fn(ctx), ctx.rarity_probe_raw)
        assert not spec.handles_kink_at_s0

    def test_linear_x_rp_centered_constant_deriv(self, ctx):
        spec = get_term_spec(LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN)
        np.testing.assert_allclose(
            spec.derivative_wrt_base_fn(ctx), ctx.rarity_reference_centered
        )


class TestAnalyticalVsNumerical:
    """
    Compara ∂φ/∂s analítica (do registry) com derivada numérica via
    finite-differences. Em pontos s ≠ s₀ (onde a função é suave), as
    duas devem coincidir até precisão de máquina.
    """

    @pytest.mark.parametrize("term", list(LRFeatureTerm))
    def test_derivative_matches_finite_differences(self, term):
        spec = get_term_spec(term)
        # Bases longe de s0 para evitar dobras
        base = np.array([-0.5, 0.5, 0.9], dtype=np.float64)
        s0 = 0.0  # s0 = 0 — todos os pares estão em s ≠ s0
        r_F = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        r_P = np.array([0.6, 0.4, 0.2], dtype=np.float64)

        # ONE_FACE (V2.14.1): reusa r_F como r_single para o teste
        r_S = r_F

        def make_ctx(b):
            if spec.term.requires_s0:
                delta = b - s0
                hp = np.maximum(delta, 0)
                hn = np.maximum(-delta, 0)
            else:
                delta, hp, hn = None, None, None
            return LRFeatureBuildContext(
                base=b, s0=s0 if spec.term.requires_s0 else None,
                delta=delta, hinge_positive=hp, hinge_negative=hn,
                rarity_probe_raw=r_F if spec.term.consumes_rarity_probe else None,
                rarity_reference_raw=r_P if spec.term.consumes_rarity_reference else None,
                rarity_probe_centered=(r_F - 0.5) if (spec.term.consumes_rarity_probe and spec.term.is_centered) else None,
                rarity_reference_centered=(r_P - 0.4) if (spec.term.consumes_rarity_reference and spec.term.is_centered) else None,
                rarity_single_raw=r_S if spec.term.consumes_rarity_single else None,
                rarity_single_centered=(r_S - 0.5) if (spec.term.consumes_rarity_single and spec.term.is_centered) else None,
            )

        eps = 1e-6
        ctx_plus = make_ctx(base + eps)
        ctx_minus = make_ctx(base - eps)
        ctx_at = make_ctx(base)

        y_plus = spec.build_fn(ctx_plus)
        y_minus = spec.build_fn(ctx_minus)
        deriv_num = (y_plus - y_minus) / (2 * eps)

        deriv_analytic = spec.derivative_wrt_base_fn(ctx_at)

        np.testing.assert_allclose(
            deriv_num, deriv_analytic, atol=1e-5,
            err_msg=f"Derivada numérica vs analítica diverge para {term.value}",
        )


class TestFeatureTermFlags:
    """Testa as flags semânticas auxiliares (`requires_s0`, `is_centered`, etc.)."""

    def test_requires_s0(self):
        assert not LRFeatureTerm.BASE_SCORE.requires_s0
        assert LRFeatureTerm.HINGE_POSITIVE.requires_s0
        assert LRFeatureTerm.HINGE_NEGATIVE.requires_s0
        assert LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW.requires_s0
        assert LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN.requires_s0

    def test_is_centered(self):
        assert not LRFeatureTerm.BASE_SCORE.is_centered
        assert not LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW.is_centered
        assert LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN.is_centered
        assert LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN.is_centered

    def test_has_kink_at_s0(self):
        assert not LRFeatureTerm.BASE_SCORE.has_kink_at_s0
        assert LRFeatureTerm.HINGE_POSITIVE.has_kink_at_s0
        assert LRFeatureTerm.HINGE_NEGATIVE.has_kink_at_s0
        assert LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW.has_kink_at_s0
        # Lineares são suaves
        assert not LRFeatureTerm.LINEAR_X_RARITY_PROBE_RAW.has_kink_at_s0
