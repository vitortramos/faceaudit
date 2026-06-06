# faceaudit/models/lr/lr_feature_term_spec.py
"""
Especificação declarativa de cada termo atômico da vetorização LR.

Cada `LRFeatureTerm` tem aqui registrada:
  - `build_fn(ctx)`: produz a coluna da feature a partir de
    `LRFeatureBuildContext` (que já carrega base, hinges pré-calculadas
    e raridades centradas/brutas).
  - `derivative_wrt_base_fn(ctx)`: produz ∂φ/∂s por par para uso pela
    futura `CONDITIONAL_MONOTONICITY_AUDIT`. Em descontinuidades (hinges
    em s=s₀), retorna a derivada à direita (convenção).
  - `left_slope_at_s0_fn(ctx)`, `right_slope_at_s0_fn(ctx)`: para hinges,
    expõem as duas inclinações no ponto da dobra explicitamente.

`formula_version` é bumpada quando build_fn ou derivative_wrt_base_fn
mudam de implementação — entra no fingerprint da arquitetura para
detectar drift de código sem alteração de enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from faceaudit.enums.lr_feature_term import LRFeatureTerm


@dataclass(frozen=True)
class LRFeatureBuildContext:
    """
    Contexto compartilhado pelos termos durante uma execução do
    `ArchitectureFeatureBuilder.build()`.

    Pré-calcula uma vez os arrays comuns para evitar recomputo dentro
    de cada termo. Quando um termo não precisa de determinado campo
    (por exemplo, `delta` em arquitetura COSINE_ONLY), o campo
    correspondente fica `None`.

    Convenções:
      - `base`: array (N,) de scores cosseno brutos.
      - `s0`: float; `None` se a arquitetura não exige s₀.
      - `delta`: `base - s0` (None se s0 é None).
      - `hinge_positive`: `np.maximum(base - s0, 0)` (None se sem s0).
      - `hinge_negative`: `np.maximum(s0 - base, 0)` (None se sem s0).
      - `rarity_probe_raw`, `rarity_reference_raw`: arrays (N,) das
        raridades canônicas (1 - percentil, ou -mean_knn). None se a
        arquitetura não consome raridade do respectivo lado.
      - `rarity_probe_centered`, `rarity_reference_centered`: r - r̄.
        None se centering_mode == NONE ou se o lado não é consumido.
      - `rarity_single_raw`, `rarity_single_centered`: family ONE_FACE
        (V2.14.1) — análogos a `rarity_probe_*` mas para a face única.
        None quando a arquitetura não pertence a ONE_FACE.
    """

    base: np.ndarray
    s0: Optional[float]
    delta: Optional[np.ndarray]
    hinge_positive: Optional[np.ndarray]
    hinge_negative: Optional[np.ndarray]
    rarity_probe_raw: Optional[np.ndarray]
    rarity_reference_raw: Optional[np.ndarray]
    rarity_probe_centered: Optional[np.ndarray]
    rarity_reference_centered: Optional[np.ndarray]
    rarity_single_raw: Optional[np.ndarray] = None
    rarity_single_centered: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LRFeatureTermSpec:
    """
    Registro declarativo de um termo atômico da vetorização LR.

    `formula_version` deve ser bumpada manualmente sempre que `build_fn`
    ou `derivative_wrt_base_fn` mudarem implementação — entra no
    fingerprint da arquitetura. A disciplina do bump é apoiada pelos
    testes sentinela (input fixo → output esperado).
    """

    term: LRFeatureTerm
    formula_version: str
    build_fn: Callable[[LRFeatureBuildContext], np.ndarray]
    derivative_wrt_base_fn: Callable[[LRFeatureBuildContext], np.ndarray]
    handles_kink_at_s0: bool
    left_slope_at_s0_fn: Optional[Callable[[LRFeatureBuildContext], np.ndarray]] = None
    right_slope_at_s0_fn: Optional[Callable[[LRFeatureBuildContext], np.ndarray]] = None


# ─────────────────────────────────────────────────────────────────────
# Implementações dos 13 termos
# ─────────────────────────────────────────────────────────────────────


# BASE_SCORE: φ = s; ∂φ/∂s = 1
def _build_base(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.base.astype(np.float64, copy=True)


def _deriv_base(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.ones_like(ctx.base, dtype=np.float64)


# HINGE_POSITIVE: φ = h⁺ = max(s − s₀, 0); ∂φ/∂s = 1{s > s₀} (à direita)
def _build_hinge_positive(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None, "hinge_positive não pré-calculada"
    return ctx.hinge_positive.astype(np.float64, copy=True)


def _deriv_hinge_positive(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64)


def _left_slope_hinge_positive(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hinge_positive(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.ones_like(ctx.base, dtype=np.float64)


# HINGE_NEGATIVE: φ = h⁻ = max(s₀ − s, 0); ∂φ/∂s = −1{s < s₀} (à esquerda)
def _build_hinge_negative(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_negative is not None
    return ctx.hinge_negative.astype(np.float64, copy=True)


def _deriv_hinge_negative(ctx: LRFeatureBuildContext) -> np.ndarray:
    # Derivada à direita: para s > s₀, h⁻ = 0 e ∂/∂s = 0
    # para s < s₀, h⁻ = s₀ - s e ∂/∂s = -1
    return -(ctx.base < ctx.s0).astype(np.float64)


def _left_slope_hinge_negative(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -np.ones_like(ctx.base, dtype=np.float64)


def _right_slope_hinge_negative(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


# HINGE_POSITIVE_X_RARITY_PROBE_RAW: φ = h⁺·r_F; ∂φ/∂s = 1{s>s₀}·r_F
def _build_hp_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_probe_raw is not None
    return (ctx.hinge_positive * ctx.rarity_probe_raw).astype(np.float64, copy=True)


def _deriv_hp_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_probe_raw


def _left_slope_hp_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_probe_raw.astype(np.float64, copy=True)


# HINGE_POSITIVE_X_RARITY_REFERENCE_RAW: φ = h⁺·r_P
def _build_hp_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_reference_raw is not None
    return (ctx.hinge_positive * ctx.rarity_reference_raw).astype(np.float64, copy=True)


def _deriv_hp_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_reference_raw


def _left_slope_hp_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_reference_raw.astype(np.float64, copy=True)


# HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN: φ = h⁺·r̃_F
def _build_hp_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_probe_centered is not None
    return (ctx.hinge_positive * ctx.rarity_probe_centered).astype(np.float64, copy=True)


def _deriv_hp_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_probe_centered


def _left_slope_hp_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_probe_centered.astype(np.float64, copy=True)


# HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN: φ = h⁺·r̃_P
def _build_hp_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_reference_centered is not None
    return (ctx.hinge_positive * ctx.rarity_reference_centered).astype(np.float64, copy=True)


def _deriv_hp_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_reference_centered


def _left_slope_hp_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_reference_centered.astype(np.float64, copy=True)


# HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN: φ = h⁻·r̃_F
def _build_hn_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_negative is not None and ctx.rarity_probe_centered is not None
    return (ctx.hinge_negative * ctx.rarity_probe_centered).astype(np.float64, copy=True)


def _deriv_hn_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -(ctx.base < ctx.s0).astype(np.float64) * ctx.rarity_probe_centered


def _left_slope_hn_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -ctx.rarity_probe_centered.astype(np.float64, copy=True)


def _right_slope_hn_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


# HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN: φ = h⁻·r̃_P
def _build_hn_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_negative is not None and ctx.rarity_reference_centered is not None
    return (ctx.hinge_negative * ctx.rarity_reference_centered).astype(np.float64, copy=True)


def _deriv_hn_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -(ctx.base < ctx.s0).astype(np.float64) * ctx.rarity_reference_centered


def _left_slope_hn_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -ctx.rarity_reference_centered.astype(np.float64, copy=True)


def _right_slope_hn_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


# LINEAR_X_RARITY_PROBE_RAW: φ = (s − s₀)·r_F; ∂φ/∂s = r_F
def _build_lin_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_probe_raw is not None
    return (ctx.delta * ctx.rarity_probe_raw).astype(np.float64, copy=True)


def _deriv_lin_x_rf_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_probe_raw.astype(np.float64, copy=True)


# LINEAR_X_RARITY_REFERENCE_RAW: φ = (s − s₀)·r_P
def _build_lin_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_reference_raw is not None
    return (ctx.delta * ctx.rarity_reference_raw).astype(np.float64, copy=True)


def _deriv_lin_x_rp_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_reference_raw.astype(np.float64, copy=True)


# LINEAR_X_RARITY_PROBE_CENTERED_MEAN: φ = (s − s₀)·r̃_F
def _build_lin_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_probe_centered is not None
    return (ctx.delta * ctx.rarity_probe_centered).astype(np.float64, copy=True)


def _deriv_lin_x_rf_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_probe_centered.astype(np.float64, copy=True)


# LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN: φ = (s − s₀)·r̃_P
def _build_lin_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_reference_centered is not None
    return (ctx.delta * ctx.rarity_reference_centered).astype(np.float64, copy=True)


def _deriv_lin_x_rp_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_reference_centered.astype(np.float64, copy=True)


# ─────────────────────────────────────────────────────────────────────
# Termos ONE_FACE (V2.14.1) — usam r̃/r de UMA face única
# ─────────────────────────────────────────────────────────────────────


# H_POS_TIMES_SINGLE_RARITY: φ = h⁺·r; ∂φ/∂s = 1{s>s₀}·r
def _build_hp_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_single_raw is not None
    return (ctx.hinge_positive * ctx.rarity_single_raw).astype(np.float64, copy=True)


def _deriv_hp_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_single_raw


def _left_slope_hp_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_single_raw.astype(np.float64, copy=True)


# H_POS_TIMES_SINGLE_RARITY_CENTERED: φ = h⁺·r̃
def _build_hp_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_positive is not None and ctx.rarity_single_centered is not None
    return (ctx.hinge_positive * ctx.rarity_single_centered).astype(np.float64, copy=True)


def _deriv_hp_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return (ctx.base > ctx.s0).astype(np.float64) * ctx.rarity_single_centered


def _left_slope_hp_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


def _right_slope_hp_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_single_centered.astype(np.float64, copy=True)


# H_NEG_TIMES_SINGLE_RARITY_CENTERED: φ = h⁻·r̃
def _build_hn_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.hinge_negative is not None and ctx.rarity_single_centered is not None
    return (ctx.hinge_negative * ctx.rarity_single_centered).astype(np.float64, copy=True)


def _deriv_hn_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -(ctx.base < ctx.s0).astype(np.float64) * ctx.rarity_single_centered


def _left_slope_hn_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return -ctx.rarity_single_centered.astype(np.float64, copy=True)


def _right_slope_hn_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return np.zeros_like(ctx.base, dtype=np.float64)


# LINEAR_TIMES_SINGLE_RARITY: φ = (s − s₀)·r; ∂φ/∂s = r
def _build_lin_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_single_raw is not None
    return (ctx.delta * ctx.rarity_single_raw).astype(np.float64, copy=True)


def _deriv_lin_x_rs_raw(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_single_raw.astype(np.float64, copy=True)


# LINEAR_TIMES_SINGLE_RARITY_CENTERED: φ = (s − s₀)·r̃
def _build_lin_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    assert ctx.delta is not None and ctx.rarity_single_centered is not None
    return (ctx.delta * ctx.rarity_single_centered).astype(np.float64, copy=True)


def _deriv_lin_x_rs_ctd(ctx: LRFeatureBuildContext) -> np.ndarray:
    return ctx.rarity_single_centered.astype(np.float64, copy=True)


# ─────────────────────────────────────────────────────────────────────
# Registry — fonte única de verdade dos 18 termos (13 originais + 5 ONE_FACE)
# ─────────────────────────────────────────────────────────────────────

LR_FEATURE_TERM_SPECS: Mapping[LRFeatureTerm, LRFeatureTermSpec] = {
    LRFeatureTerm.BASE_SCORE: LRFeatureTermSpec(
        term=LRFeatureTerm.BASE_SCORE,
        formula_version="1.0",
        build_fn=_build_base,
        derivative_wrt_base_fn=_deriv_base,
        handles_kink_at_s0=False,
    ),
    LRFeatureTerm.HINGE_POSITIVE: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_POSITIVE,
        formula_version="1.0",
        build_fn=_build_hinge_positive,
        derivative_wrt_base_fn=_deriv_hinge_positive,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hinge_positive,
        right_slope_at_s0_fn=_right_slope_hinge_positive,
    ),
    LRFeatureTerm.HINGE_NEGATIVE: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_NEGATIVE,
        formula_version="1.0",
        build_fn=_build_hinge_negative,
        derivative_wrt_base_fn=_deriv_hinge_negative,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hinge_negative,
        right_slope_at_s0_fn=_right_slope_hinge_negative,
    ),
    LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_RAW,
        formula_version="1.0",
        build_fn=_build_hp_x_rf_raw,
        derivative_wrt_base_fn=_deriv_hp_x_rf_raw,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rf_raw,
        right_slope_at_s0_fn=_right_slope_hp_x_rf_raw,
    ),
    LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_RAW: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_RAW,
        formula_version="1.0",
        build_fn=_build_hp_x_rp_raw,
        derivative_wrt_base_fn=_deriv_hp_x_rp_raw,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rp_raw,
        right_slope_at_s0_fn=_right_slope_hp_x_rp_raw,
    ),
    LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_hp_x_rf_ctd,
        derivative_wrt_base_fn=_deriv_hp_x_rf_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rf_ctd,
        right_slope_at_s0_fn=_right_slope_hp_x_rf_ctd,
    ),
    LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_hp_x_rp_ctd,
        derivative_wrt_base_fn=_deriv_hp_x_rp_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rp_ctd,
        right_slope_at_s0_fn=_right_slope_hp_x_rp_ctd,
    ),
    LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_hn_x_rf_ctd,
        derivative_wrt_base_fn=_deriv_hn_x_rf_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hn_x_rf_ctd,
        right_slope_at_s0_fn=_right_slope_hn_x_rf_ctd,
    ),
    LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_hn_x_rp_ctd,
        derivative_wrt_base_fn=_deriv_hn_x_rp_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hn_x_rp_ctd,
        right_slope_at_s0_fn=_right_slope_hn_x_rp_ctd,
    ),
    LRFeatureTerm.LINEAR_X_RARITY_PROBE_RAW: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_X_RARITY_PROBE_RAW,
        formula_version="1.0",
        build_fn=_build_lin_x_rf_raw,
        derivative_wrt_base_fn=_deriv_lin_x_rf_raw,
        handles_kink_at_s0=False,
    ),
    LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_RAW: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_RAW,
        formula_version="1.0",
        build_fn=_build_lin_x_rp_raw,
        derivative_wrt_base_fn=_deriv_lin_x_rp_raw,
        handles_kink_at_s0=False,
    ),
    LRFeatureTerm.LINEAR_X_RARITY_PROBE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_X_RARITY_PROBE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_lin_x_rf_ctd,
        derivative_wrt_base_fn=_deriv_lin_x_rf_ctd,
        handles_kink_at_s0=False,
    ),
    LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN,
        formula_version="1.0",
        build_fn=_build_lin_x_rp_ctd,
        derivative_wrt_base_fn=_deriv_lin_x_rp_ctd,
        handles_kink_at_s0=False,
    ),
    # ─── Família ONE_FACE (V2.14.1) ──────────────────────────────────────
    LRFeatureTerm.H_POS_TIMES_SINGLE_RARITY: LRFeatureTermSpec(
        term=LRFeatureTerm.H_POS_TIMES_SINGLE_RARITY,
        formula_version="1.0",
        build_fn=_build_hp_x_rs_raw,
        derivative_wrt_base_fn=_deriv_hp_x_rs_raw,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rs_raw,
        right_slope_at_s0_fn=_right_slope_hp_x_rs_raw,
    ),
    LRFeatureTerm.H_POS_TIMES_SINGLE_RARITY_CENTERED: LRFeatureTermSpec(
        term=LRFeatureTerm.H_POS_TIMES_SINGLE_RARITY_CENTERED,
        formula_version="1.0",
        build_fn=_build_hp_x_rs_ctd,
        derivative_wrt_base_fn=_deriv_hp_x_rs_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hp_x_rs_ctd,
        right_slope_at_s0_fn=_right_slope_hp_x_rs_ctd,
    ),
    LRFeatureTerm.H_NEG_TIMES_SINGLE_RARITY_CENTERED: LRFeatureTermSpec(
        term=LRFeatureTerm.H_NEG_TIMES_SINGLE_RARITY_CENTERED,
        formula_version="1.0",
        build_fn=_build_hn_x_rs_ctd,
        derivative_wrt_base_fn=_deriv_hn_x_rs_ctd,
        handles_kink_at_s0=True,
        left_slope_at_s0_fn=_left_slope_hn_x_rs_ctd,
        right_slope_at_s0_fn=_right_slope_hn_x_rs_ctd,
    ),
    LRFeatureTerm.LINEAR_TIMES_SINGLE_RARITY: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_TIMES_SINGLE_RARITY,
        formula_version="1.0",
        build_fn=_build_lin_x_rs_raw,
        derivative_wrt_base_fn=_deriv_lin_x_rs_raw,
        handles_kink_at_s0=False,
    ),
    LRFeatureTerm.LINEAR_TIMES_SINGLE_RARITY_CENTERED: LRFeatureTermSpec(
        term=LRFeatureTerm.LINEAR_TIMES_SINGLE_RARITY_CENTERED,
        formula_version="1.0",
        build_fn=_build_lin_x_rs_ctd,
        derivative_wrt_base_fn=_deriv_lin_x_rs_ctd,
        handles_kink_at_s0=False,
    ),
}


def get_term_spec(term: LRFeatureTerm) -> LRFeatureTermSpec:
    """Lookup do spec de um termo. Levanta KeyError se ausente."""
    return LR_FEATURE_TERM_SPECS[term]
