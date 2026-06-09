"""Validador comum da distribuição de referência de scores.

P1-R2-7 (revisão externa 2026-06-09 round 2): garante que estado
restaurado de pickle não passa silenciosamente quando corrompido.
Usado por `MoVMFTypicalityModel.from_dict` e
`KNNTypicalityModel.from_dict`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class InvalidReferenceDistributionError(ValueError):
    """Distribuição de referência inconsistente (corrompida, adulterada
    ou produzida por código com bug)."""

    def __init__(self, field: str, detail: str) -> None:
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.detail = detail


def validate_reference_score_distribution(
    *,
    scores: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cdf: Optional[np.ndarray] = None,
    ood_threshold: Optional[float] = None,
    require_sorted: bool = True,
) -> None:
    """Valida invariantes da distribuição de referência.

    Levanta `InvalidReferenceDistributionError` no primeiro problema.

    Invariantes verificados:
    - scores 1D, finitos, tamanho > 0.
    - scores ordenados crescente (quando require_sorted).
    - weights (se presentes): shape compatível, finitos, não-negativos,
      soma > 0.
    - cdf (se presente): shape compatível, finita, monotônica
      não-decrescente, em [0, 1], último valor ≈ 1.
    - ood_threshold (se presente): finito.

    Não recompõe nada: só valida. O caller decide se aborta ou ajusta.
    """
    # ─── scores ────────────────────────────────────────────────────
    if scores.ndim != 1:
        raise InvalidReferenceDistributionError(
            "scores", f"esperado 1D, recebido shape={scores.shape}",
        )
    if scores.size == 0:
        raise InvalidReferenceDistributionError(
            "scores", "vetor vazio",
        )
    if not np.all(np.isfinite(scores)):
        n_bad = int((~np.isfinite(scores)).sum())
        raise InvalidReferenceDistributionError(
            "scores", f"{n_bad}/{scores.size} valores não finitos",
        )
    if require_sorted and np.any(np.diff(scores) < 0):
        raise InvalidReferenceDistributionError(
            "scores", "vetor deve estar ordenado crescente",
        )

    # ─── weights ───────────────────────────────────────────────────
    if weights is not None:
        if weights.ndim != 1:
            raise InvalidReferenceDistributionError(
                "weights",
                f"esperado 1D, recebido shape={weights.shape}",
            )
        if weights.shape != scores.shape:
            raise InvalidReferenceDistributionError(
                "weights",
                f"shape {weights.shape} ≠ scores {scores.shape}",
            )
        if not np.all(np.isfinite(weights)):
            raise InvalidReferenceDistributionError(
                "weights", "contém valores não finitos",
            )
        if np.any(weights < 0):
            raise InvalidReferenceDistributionError(
                "weights",
                "deve ser não-negativo (encontrou ao menos um valor < 0)",
            )
        total_w = float(weights.sum())
        if total_w <= 0:
            raise InvalidReferenceDistributionError(
                "weights", f"soma = {total_w} (deve ser > 0)",
            )

    # ─── cdf ───────────────────────────────────────────────────────
    if cdf is not None:
        if cdf.ndim != 1:
            raise InvalidReferenceDistributionError(
                "cdf", f"esperado 1D, recebido shape={cdf.shape}",
            )
        if cdf.shape != scores.shape:
            raise InvalidReferenceDistributionError(
                "cdf",
                f"shape {cdf.shape} ≠ scores {scores.shape}",
            )
        if not np.all(np.isfinite(cdf)):
            raise InvalidReferenceDistributionError(
                "cdf", "contém valores não finitos",
            )
        if np.any(cdf < -1e-9) or np.any(cdf > 1.0 + 1e-9):
            raise InvalidReferenceDistributionError(
                "cdf",
                f"valores fora de [0, 1]: min={cdf.min():.4f}, "
                f"max={cdf.max():.4f}",
            )
        if np.any(np.diff(cdf) < -1e-9):
            raise InvalidReferenceDistributionError(
                "cdf", "deve ser monotônica não-decrescente",
            )
        if abs(float(cdf[-1]) - 1.0) > 1e-6:
            raise InvalidReferenceDistributionError(
                "cdf",
                f"último valor = {float(cdf[-1])} (esperado ≈ 1.0)",
            )

    # ─── ood_threshold ─────────────────────────────────────────────
    if ood_threshold is not None:
        if not np.isfinite(ood_threshold):
            raise InvalidReferenceDistributionError(
                "ood_threshold",
                f"valor não finito: {ood_threshold}",
            )
