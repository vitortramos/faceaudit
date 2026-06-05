"""
Testes do `s0_resolution_service` — gates de qualidade do AUTO_LOGREG +
passthrough do MANUAL.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.services.s0_resolution_service import (
    S0FitSummary,
    S0ResolutionError,
    resolve_s0_auto_logreg,
    resolve_s0_manual,
)


class TestAutoLogReg:
    def test_recovers_s0_close_to_class_boundary(self):
        """
        Com distribuições bem separadas (impostores ~ -0.05, genuínos ~ 0.35),
        s₀ recuperado deve estar entre as médias.
        """
        rng = np.random.RandomState(42)
        n_imp = 500
        n_gen = 50
        imp = rng.normal(-0.05, 0.06, n_imp)
        gen = rng.normal(0.35, 0.10, n_gen)
        base = np.concatenate([imp, gen])
        y = np.concatenate([np.zeros(n_imp), np.ones(n_gen)]).astype(int)
        summary = resolve_s0_auto_logreg(base, y)
        assert -0.1 < summary.s0_value < 0.4
        assert summary.converged
        assert summary.is_stable
        assert summary.n_genuine == n_gen
        assert summary.n_impostor == n_imp

    def test_only_genuines_raises(self):
        with pytest.raises(S0ResolutionError, match="ambas as classes"):
            resolve_s0_auto_logreg(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([1, 1, 1, 1, 1]),
            )

    def test_only_impostors_raises(self):
        with pytest.raises(S0ResolutionError, match="ambas as classes"):
            resolve_s0_auto_logreg(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([0, 0, 0, 0, 0]),
            )

    def test_too_few_per_class_raises(self):
        # Só 1 genuíno
        with pytest.raises(S0ResolutionError, match="ambas as classes"):
            resolve_s0_auto_logreg(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([1, 0, 0, 0, 0]),
            )

    def test_nan_in_base_raises(self):
        with pytest.raises(S0ResolutionError, match="NaN"):
            resolve_s0_auto_logreg(
                np.array([0.1, np.nan, 0.3, 0.4, 0.5]),
                np.array([1, 1, 0, 0, 0]),
            )

    def test_inf_in_base_raises(self):
        with pytest.raises(S0ResolutionError, match="NaN"):
            resolve_s0_auto_logreg(
                np.array([0.1, np.inf, 0.3, 0.4, 0.5]),
                np.array([1, 1, 0, 0, 0]),
            )

    def test_zero_variance_raises(self):
        with pytest.raises(S0ResolutionError, match="constante"):
            resolve_s0_auto_logreg(
                np.array([0.5] * 10),
                np.array([1, 0] * 5),
            )

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            resolve_s0_auto_logreg(
                np.array([0.1, 0.2, 0.3]),
                np.array([1, 0]),
            )

    def test_summary_to_dict_hex_uses_hex_floats(self):
        rng = np.random.RandomState(0)
        imp = rng.normal(-0.1, 0.05, 100)
        gen = rng.normal(0.3, 0.1, 30)
        base = np.concatenate([imp, gen])
        y = np.concatenate([np.zeros(100), np.ones(30)]).astype(int)
        summary = resolve_s0_auto_logreg(base, y)
        d = summary.to_dict_hex()
        # Floats em hex
        assert d["coef_hex"].startswith(("0x", "-0x"))
        assert d["intercept_hex"].startswith(("0x", "-0x"))
        assert d["s0_value_hex"].startswith(("0x", "-0x"))
        assert isinstance(d["converged"], bool)
        assert isinstance(d["n_genuine"], int)


class TestManual:
    def test_passthrough(self):
        summary = resolve_s0_manual(0.42)
        assert summary.s0_value == 0.42
        assert summary.converged
        assert summary.is_stable

    def test_to_dict_hex(self):
        d = resolve_s0_manual(0.5).to_dict_hex()
        # 0.5 em hex: 0x1.0000000000000p-1
        assert d["s0_value_hex"] == "0x1.0000000000000p-1"
