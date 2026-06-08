# faceaudit/services/unit_norm_validator.py

"""Validador comum de norma L2 para modelos esféricos.

Reusado por `KNNTypicalityModel` e `MoVMFTypicalityModel`. Centraliza a
política de unit norm declarada pelo config (`UnitNormPolicy`).
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from faceaudit.enums.unit_norm_policy import UnitNormPolicy


class UnitNormViolationError(ValueError):
    """Embeddings violaram a tolerância de norma L2 com política ERROR.

    Atributos:
        max_deviation: máxima `|‖x‖ − 1|` observada.
        tolerance: tolerância configurada.
        n_violations: número de embeddings que excederam a tolerância.
        n_total: total de embeddings examinados.
    """

    def __init__(
        self,
        *,
        max_deviation: float,
        tolerance: float,
        n_violations: int,
        n_total: int,
    ) -> None:
        msg = (
            f"UnitNormViolation: {n_violations}/{n_total} embeddings "
            f"violaram a tolerância de norma L2 (max |‖x‖ − 1| = "
            f"{max_deviation:.3e} > tolerance = {tolerance:.3e}). "
            "Política UnitNormPolicy=ERROR (default forense) rejeita. "
            "Para normalizar implicitamente, configure "
            "UnitNormPolicy=NORMALIZE_WITH_WARNING ou NORMALIZE_SILENTLY."
        )
        super().__init__(msg)
        self.max_deviation = max_deviation
        self.tolerance = tolerance
        self.n_violations = n_violations
        self.n_total = n_total


def apply_unit_norm_policy(
    x: np.ndarray,
    *,
    policy: UnitNormPolicy,
    tolerance: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Avalia a política e devolve a matriz `(N, D)` apropriada.

    Comportamento por modo:

    - **ERROR**: levanta `UnitNormViolationError` se algum embedding
      violar `|‖x‖ − 1| > tolerance`. Retorna `x` inalterado em caso de
      sucesso.
    - **NORMALIZE_WITH_WARNING**: emite `UserWarning` com a magnitude
      máxima observada quando houver violações; depois normaliza
      `x / max(‖x‖, eps)`.
    - **NORMALIZE_SILENTLY**: normaliza sem emitir warning. Default
      migratório do k-NN V2.14.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(
            f"apply_unit_norm_policy: esperado (N, D), recebeu shape={x.shape}."
        )

    norms = np.linalg.norm(x, axis=1)
    deviation = np.abs(norms - 1.0)
    violations_mask = deviation > tolerance
    n_violations = int(violations_mask.sum())
    max_dev = float(deviation.max()) if deviation.size else 0.0

    if policy is UnitNormPolicy.ERROR:
        if n_violations > 0:
            raise UnitNormViolationError(
                max_deviation=max_dev, tolerance=tolerance,
                n_violations=n_violations, n_total=int(x.shape[0]),
            )
        return x

    if policy is UnitNormPolicy.NORMALIZE_WITH_WARNING:
        if n_violations > 0:
            warnings.warn(
                f"UnitNormPolicy=NORMALIZE_WITH_WARNING: "
                f"{n_violations}/{x.shape[0]} embeddings com "
                f"|‖x‖ − 1| > {tolerance:.3e} (max={max_dev:.3e}). "
                "Normalizando para esfera unitária.",
                UserWarning,
                stacklevel=2,
            )
        return x / np.maximum(norms[:, None], eps)

    if policy is UnitNormPolicy.NORMALIZE_SILENTLY:
        return x / np.maximum(norms[:, None], eps)

    raise ValueError(f"UnitNormPolicy desconhecida: {policy}")


def policy_audit_metadata(
    *,
    policy: UnitNormPolicy,
    tolerance: float,
    n_violations: int,
    n_total: int,
    max_deviation: float,
) -> dict:
    """Empacota metadados de auditoria para persistir no `summary` do
    modelo.
    """
    return {
        "unit_norm_policy": policy.to_str(),
        "unit_norm_tolerance": tolerance,
        "unit_norm_violations": n_violations,
        "unit_norm_total": n_total,
        "unit_norm_max_deviation": max_deviation,
    }
