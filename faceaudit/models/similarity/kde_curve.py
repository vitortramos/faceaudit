"""Curva KDE para gráficos de distribuição, com proteção contra degeneração.

Uma amostra praticamente constante (ex.: pares genuínos que comparam a
MESMA imagem consigo mesma — escore 1,0 com desvio ~1e-16) faz o
`gaussian_kde` produzir um pico de altura ~1/bandwidth (≈1e13). No gráfico
compartilhado, esse pico define o eixo Y e achata todas as outras séries
contra o zero — os impostores "somem".

Regra: a curva só é desenhada quando a banda do kernel é resolvível na
grade do gráfico (bandwidth ≥ passo da grade). Abaixo disso a curva não
representa densidade nenhuma visível — o histograma já mostra a massa
concentrada num único bin.

Funções puras (numpy/scipy, sem matplotlib/Qt).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Tolerância para a integral numérica de uma curva gravada: uma densidade
# legítima integra ≤ 1 no intervalo plotado (menos quando há massa fora
# dele); o excedente cobre só o erro da regra do trapézio.
_MAX_PLAUSIBLE_INTEGRAL = 1.5


def kde_curve(values: Sequence[float], xs: np.ndarray) -> Optional[np.ndarray]:
    """Avalia a KDE gaussiana (regra de Scott) de `values` na grade `xs`.

    Retorna None quando a curva não é desenhável: menos de 2 valores
    finitos, grade com menos de 2 pontos, variância nula (o scipy levanta
    erro de matriz singular) ou bandwidth menor que o passo da grade.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    xs = np.asarray(xs, dtype=np.float64)
    if arr.size < 2 or xs.size < 2:
        return None
    grid_step = float(np.min(np.abs(np.diff(xs))))
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(arr)
    except Exception:
        return None
    bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    if not np.isfinite(bandwidth) or bandwidth < grid_step:
        return None
    return kde(xs)


def is_renderable_density_curve(
    xs: Optional[Sequence[float]], ys: Optional[Sequence[float]],
) -> bool:
    """Diz se uma curva de densidade JÁ CALCULADA pode ser desenhada.

    Usada sobre curvas lidas de sidecars `_dist.json` gravados antes da
    proteção de `kde_curve`: um pico degenerado aparece como integral
    numérica muito acima de 1 (ou valores não finitos).
    """
    if xs is None or ys is None:
        return False
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not np.all(np.isfinite(y)):
        return False
    return float(np.trapezoid(y, x)) <= _MAX_PLAUSIBLE_INTEGRAL
