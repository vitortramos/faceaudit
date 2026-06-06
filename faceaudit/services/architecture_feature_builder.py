# faceaudit/services/architecture_feature_builder.py
"""
Serviço puro de construção da matriz de features de calibração LR.

Recebe um `LRArchitectureSpec` (declaração) + um state-like com
(`s0`, `r0_probe`, `r0_reference`) + arrays brutos (base, rarity_probe,
rarity_reference) e produz:

- `build(...)`: matriz (N, D) com colunas na ordem exata de
  `spec.feature_terms`.
- `derivative_matrix_wrt_base(...)`: matriz (N, D) com ∂φ_j/∂s por par
  para uso pela futura `CONDITIONAL_MONOTONICITY_AUDIT`.

Função pura. Sem I/O de contexto FaceAudit. Despacha cada termo via
`LR_FEATURE_TERM_SPECS` — zero ramos por arquitetura.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from faceaudit.models.lr.lr_architecture_spec import LRArchitectureSpec
from faceaudit.models.lr.lr_feature_term_spec import (
    LR_FEATURE_TERM_SPECS,
    LRFeatureBuildContext,
)


class _StateLike(Protocol):
    """
    Subconjunto do `LRArchitectureState` usado pelo builder.

    O state real (`LRArchitectureState`) implementa este protocolo;
    testes podem usar mocks/stubs simples (e.g., SimpleNamespace) sem
    construir o state completo.

    Para a família ONE_FACE (V2.14.1), `r0_single` é lido do mesmo
    objeto. Quando o state real não expõe o campo (legado), o builder
    cai em `r0_probe` por compat semântica.
    """

    s0: Optional[float]
    r0_probe: Optional[float]
    r0_reference: Optional[float]


def _build_context(
    spec: LRArchitectureSpec,
    state: _StateLike,
    base: np.ndarray,
    rarity_probe: Optional[np.ndarray],
    rarity_reference: Optional[np.ndarray],
    rarity_single: Optional[np.ndarray] = None,
) -> LRFeatureBuildContext:
    """Pré-calcula arrays comuns (delta, hinges, raridades centradas).

    Aceita `rarity_single` para a família ONE_FACE (V2.14.1). Mutuamente
    exclusivo com `rarity_probe`/`rarity_reference` por construção do
    spec — quando `spec.requires_typicality_single=True`, as outras flags
    são False (verificado nos testes do registry).
    """
    base_f64 = base.astype(np.float64, copy=False)

    if spec.requires_s0:
        if state.s0 is None:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} exige s₀, mas state.s0 é None"
            )
        s0_val = float(state.s0)
        delta = base_f64 - s0_val
        hinge_positive = np.maximum(delta, 0.0)
        hinge_negative = np.maximum(-delta, 0.0)
    else:
        s0_val = None
        delta = None
        hinge_positive = None
        hinge_negative = None

    if spec.requires_typicality_probe:
        if rarity_probe is None:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} exige rarity_probe, "
                "mas o array não foi fornecido"
            )
        rarity_probe_raw = rarity_probe.astype(np.float64, copy=False)
        if spec.centering_mode.value == "MEAN":
            if state.r0_probe is None:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} centra raridade, "
                    "mas state.r0_probe é None"
                )
            rarity_probe_centered = rarity_probe_raw - float(state.r0_probe)
        else:
            rarity_probe_centered = None
    else:
        rarity_probe_raw = None
        rarity_probe_centered = None

    if spec.requires_typicality_reference:
        if rarity_reference is None:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} exige rarity_reference, "
                "mas o array não foi fornecido"
            )
        rarity_reference_raw = rarity_reference.astype(np.float64, copy=False)
        if spec.centering_mode.value == "MEAN":
            if state.r0_reference is None:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} centra raridade, "
                    "mas state.r0_reference é None"
                )
            rarity_reference_centered = rarity_reference_raw - float(state.r0_reference)
        else:
            rarity_reference_centered = None
    else:
        rarity_reference_raw = None
        rarity_reference_centered = None

    # ── ONE_FACE (V2.14.1) ──
    rarity_single_raw = None
    rarity_single_centered = None
    if getattr(spec, "requires_typicality_single", False):
        if rarity_single is None:
            raise ValueError(
                f"Arquitetura {spec.architecture.value} exige rarity_single, "
                "mas o array não foi fornecido"
            )
        rarity_single_raw = rarity_single.astype(np.float64, copy=False)
        if spec.centering_mode.value == "MEAN":
            # State pode ter `r0_single` (preferencial) ou reusar `r0_probe`
            # como armazenamento (compat com state factory legado).
            r0_single = getattr(state, "r0_single", None)
            if r0_single is None:
                r0_single = state.r0_probe
            if r0_single is None:
                raise ValueError(
                    f"Arquitetura {spec.architecture.value} centra raridade, "
                    "mas state.r0_single (ou r0_probe legado) é None"
                )
            rarity_single_centered = rarity_single_raw - float(r0_single)

    return LRFeatureBuildContext(
        base=base_f64,
        s0=s0_val,
        delta=delta,
        hinge_positive=hinge_positive,
        hinge_negative=hinge_negative,
        rarity_probe_raw=rarity_probe_raw,
        rarity_reference_raw=rarity_reference_raw,
        rarity_probe_centered=rarity_probe_centered,
        rarity_reference_centered=rarity_reference_centered,
        rarity_single_raw=rarity_single_raw,
        rarity_single_centered=rarity_single_centered,
    )


def build(
    spec: LRArchitectureSpec,
    state: _StateLike,
    base: np.ndarray,
    rarity_probe: Optional[np.ndarray] = None,
    rarity_reference: Optional[np.ndarray] = None,
    rarity_single: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Constrói a matriz (N, D) com `spec.output_dim` colunas na ordem dos
    `feature_terms`.

    `base` é o array (N,) de scores cosseno brutos. `rarity_probe` e
    `rarity_reference` são as raridades canônicas (saída do
    `RarityTransformRegistry.to_rarity`), passadas apenas quando a
    arquitetura as exige. `rarity_single` é usado para a família
    ONE_FACE (V2.14.1).

    Levanta `ValueError` quando falta um input requerido pela arquitetura,
    ou quando o state não carrega o parâmetro ajustado correspondente.
    """
    if base.ndim != 1:
        raise ValueError(f"base deve ser (N,); recebido shape={base.shape}")
    n = base.shape[0]
    if rarity_probe is not None and rarity_probe.shape != (n,):
        raise ValueError(
            f"rarity_probe shape {rarity_probe.shape} != base shape ({n},)"
        )
    if rarity_reference is not None and rarity_reference.shape != (n,):
        raise ValueError(
            f"rarity_reference shape {rarity_reference.shape} != base shape ({n},)"
        )
    if rarity_single is not None and rarity_single.shape != (n,):
        raise ValueError(
            f"rarity_single shape {rarity_single.shape} != base shape ({n},)"
        )

    ctx = _build_context(
        spec, state, base, rarity_probe, rarity_reference, rarity_single,
    )

    columns = [
        LR_FEATURE_TERM_SPECS[term].build_fn(ctx)
        for term in spec.feature_terms
    ]
    matrix = np.column_stack(columns).astype(np.float64, copy=False)

    if matrix.shape != (n, spec.output_dim):
        raise AssertionError(
            f"Builder produziu shape {matrix.shape}, esperado {(n, spec.output_dim)}"
        )
    return matrix


def derivative_matrix_wrt_base(
    spec: LRArchitectureSpec,
    state: _StateLike,
    base: np.ndarray,
    rarity_probe: Optional[np.ndarray] = None,
    rarity_reference: Optional[np.ndarray] = None,
    rarity_single: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Matriz (N, D) com ∂φ_j/∂s por par. Cada coluna j corresponde a
    `spec.feature_terms[j]`.

    Em pontos s = s₀, a derivada usa a convenção à direita (right slope).
    Para inspecionar separadamente left vs right slopes em descontinuidades,
    consulte `LRFeatureTermSpec.left_slope_at_s0_fn` /
    `LRFeatureTermSpec.right_slope_at_s0_fn` diretamente.
    """
    ctx = _build_context(
        spec, state, base, rarity_probe, rarity_reference, rarity_single,
    )
    columns = [
        LR_FEATURE_TERM_SPECS[term].derivative_wrt_base_fn(ctx)
        for term in spec.feature_terms
    ]
    return np.column_stack(columns).astype(np.float64, copy=False)
