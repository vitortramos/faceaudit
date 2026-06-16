# faceaudit/services/s0_resolution_service.py
"""
Resolução de s₀ (ponto de evidência neutra do score base) com gates de
validação numérica e gating explícito.

Modos:
  - AUTO_LOGREG: ajusta LogReg univariado `base ~ same_person` e calcula
    `s₀ = −intercept/coef`. Gates: ambas as classes presentes,
    convergência, `|coef| > tolerance`.
  - MANUAL: retorna o valor passado pelo usuário.

Retorna sempre (s0_value, S0FitSummary) — o summary é persistido no
`LRArchitectureState.s0_fit_summary` para auditoria.

NUNCA usar em casework: AUTO_LOGREG exige ground truth. O caller é
responsável por checar fase antes de chamar — esta camada só executa.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class S0FitSummary:
    """
    Resumo auditável do ajuste de s₀ via AUTO_LOGREG.

    Persistido como dict no `LRArchitectureState.s0_fit_summary`. Os
    floats são convertidos para hex via `to_dict_hex()` para garantir
    estabilidade bit-exact em fingerprints de provenance.
    """

    converged: bool
    coef: float
    intercept: float
    n_genuine: int
    n_impostor: int
    score_variance: float
    is_stable: bool
    s0_value: float

    def to_dict_hex(self) -> Dict[str, Any]:
        """Floats em hex para uso em fingerprints canônicos."""
        return {
            "converged": bool(self.converged),
            "coef_hex": float(self.coef).hex(),
            "intercept_hex": float(self.intercept).hex(),
            "n_genuine": int(self.n_genuine),
            "n_impostor": int(self.n_impostor),
            "score_variance_hex": float(self.score_variance).hex(),
            "is_stable": bool(self.is_stable),
            "s0_value_hex": float(self.s0_value).hex(),
        }


class S0ResolutionError(Exception):
    """Erro irrecuperável na resolução de s₀."""


def resolve_s0_auto_logreg(
    base_scores: np.ndarray,
    same_person: np.ndarray,
    *,
    C: float = 1.0,
    solver: str = "liblinear",
    class_weight: Optional[str] = "balanced",
    max_iter: int = 1000,
    coef_tolerance: float = 1e-8,
) -> S0FitSummary:
    """
    Ajusta LogReg univariado base ~ same_person e calcula s₀.

    Gates de qualidade:
      - `same_person` deve conter as duas classes (genuíno e impostor).
      - O optimizer deve convergir.
      - `|coef|` deve ser maior que `coef_tolerance` — abaixo disso, o
        cálculo `s₀ = −intercept/coef` é numericamente instável.

    Levanta `S0ResolutionError` em qualquer gate violado.
    """
    base = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    y = np.asarray(same_person, dtype=int).reshape(-1)
    if base.shape != y.shape:
        raise ValueError(
            f"base_scores shape {base.shape} != same_person shape {y.shape}"
        )
    if not np.isfinite(base).all():
        raise S0ResolutionError("base_scores contém NaN/inf — rejeitado")

    n_genuine = int((y == 1).sum())
    n_impostor = int((y == 0).sum())
    if n_genuine < 2 or n_impostor < 2:
        raise S0ResolutionError(
            f"AUTO_LOGREG exige ambas as classes presentes (n_genuine>=2 E n_impostor>=2); "
            f"recebido n_genuine={n_genuine}, n_impostor={n_impostor}"
        )

    score_variance = float(base.var(ddof=0))
    if score_variance < 1e-20:
        raise S0ResolutionError(
            f"base_scores praticamente constante (var={score_variance}); LogReg "
            "não pode aprender discriminação"
        )

    lr = LogisticRegression(
        C=C, solver=solver, class_weight=class_weight, max_iter=max_iter,
    )
    lr.fit(base.reshape(-1, 1), y)
    coef = float(lr.coef_.ravel()[0])
    intercept = float(lr.intercept_.ravel()[0])

    # sklearn não expõe diretamente convergência. Para `liblinear`, n_iter_ é
    # array; consideramos não convergido se atingiu max_iter ou houve erro
    # numérico em coef.
    n_iter_arr = getattr(lr, "n_iter_", None)
    if n_iter_arr is not None:
        max_observed = int(np.asarray(n_iter_arr).max())
        converged = max_observed < max_iter
    else:
        converged = True  # solvers que não reportam — assumir OK

    is_coef_stable = abs(coef) > coef_tolerance

    if not converged:
        raise S0ResolutionError(
            f"AUTO_LOGREG não convergiu em {max_iter} iterações"
        )
    if not is_coef_stable:
        raise S0ResolutionError(
            f"AUTO_LOGREG coef={coef:.3e} abaixo de tolerance={coef_tolerance:.3e}; "
            "s₀ = −intercept/coef seria numericamente instável"
        )

    s0_value = -intercept / coef

    return S0FitSummary(
        converged=converged,
        coef=coef,
        intercept=intercept,
        n_genuine=n_genuine,
        n_impostor=n_impostor,
        score_variance=score_variance,
        is_stable=is_coef_stable,
        s0_value=s0_value,
    )


def resolve_s0_manual(s0_value: float) -> S0FitSummary:
    """
    Retorna S0FitSummary para s₀ MANUAL. Sem ajuste — apenas wrapper.
    """
    return S0FitSummary(
        converged=True,
        coef=0.0,
        intercept=0.0,
        n_genuine=0,
        n_impostor=0,
        score_variance=0.0,
        is_stable=True,
        s0_value=float(s0_value),
    )
