# faceaudit/services/cllr_math.py

"""
Math puro para cálculo de Cllr (Cost of Log-Likelihood Ratio) — métrica
forense central de calibração de scores em bits.

Extraído de `BiGaussianizedCalibrationModel._compute_cllr` pra evitar
import circular quando outros services (validation_math_service,
calibrator_cross_validation_service) precisam do mesmo cálculo. Função
pura, sem estado, sem dependências de modelos.

Definição:

    Cllr = 0.5 · (mean over gen log₂(1 + e^(-llr))
                  + mean over imp log₂(1 + e^(+llr)))

Estabilidade numérica: usa `np.logaddexp(0, x)` (= ln(1+e^x)) em vez de
exp direto, evitando overflow com log-LRs extremos.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


_LN2 = float(np.log(2.0))


def compute_cllr(
    llr_gen: np.ndarray,
    llr_imp: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Calcula Cllr empírico em bits + decomposição per-hypothesis.

    Args:
        llr_gen: log-LRs dos pares genuínos (Hp).
        llr_imp: log-LRs dos pares impostores (Hd).

    Returns:
        (cllr, cllr_gen, cllr_imp) em bits, onde:

            cllr_gen = mean over gen de log₂(1 + e^(-llr))   # SEM fator 0.5
            cllr_imp = mean over imp de log₂(1 + e^(+llr))   # SEM fator 0.5
            cllr     = 0.5 · (cllr_gen + cllr_imp)

        ATENÇÃO à convenção: cllr_gen e cllr_imp são as MÉDIAS por hipótese
        sem o fator 0.5 — portanto NÃO somam ao cllr total. Isso é
        deliberado pra que cada termo possa ser interpretado isoladamente
        como "Cllr_se_tivessemos_só_genuinos" / "_só_impostores".

        Cada termo é uma média de softplus, sempre ≥ 0.
        Arrays vazios contribuem 0.0 (evita NaN); chamador é responsável
        por validar tamanhos antes se quiser flagrar isso.
    """
    g = np.asarray(llr_gen, dtype=np.float64)
    i = np.asarray(llr_imp, dtype=np.float64)
    term_gen = float(np.mean(np.logaddexp(0, -g)) / _LN2) if g.size else 0.0
    term_imp = float(np.mean(np.logaddexp(0, i)) / _LN2) if i.size else 0.0
    cllr = 0.5 * (term_gen + term_imp)
    return cllr, term_gen, term_imp
