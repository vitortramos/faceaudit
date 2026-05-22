"""Renderização da distribuição empírica de T_k (LOO) de um KNNTypicalityModel.

Funções puras de matplotlib — sem dependência de Qt. Mesmo padrão das demais
plot helpers (`distribution_plot.py`, `casework_visualization_plot.py`,
`validation_result_plot.py`):

  - `draw_typicality_distribution_from_model(figure, model)` — desenho
    matplotlib puro, reusado pelo `TypicalityDistributionWidget` (UI Qt)
    e pelo renderizador PNG (headless Agg).
  - `render_typicality_distribution_png(model, target_path)` — wrapper
    headless usando `FigureCanvasAgg`, gravado pelo
    `StrategyExecutor.persist_produced_resource` após cada execução de
    `KNN_TYPICALITY_MODEL_TRAINING`.
  - `typicality_distribution_png_path(...)` — convenção de path estável,
    usada tanto na escrita (executor) quanto na leitura (painel inline).

Subfolder `typicality_models/` em `<project>/graphics/<strategy_id>/` evita
poluir a pasta com PNGs de outros tipos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


def draw_typicality_distribution_from_model(
    figure: Figure,
    model: Any,
    *,
    show_title: bool = True,
) -> None:
    """Histograma + KDE da distribuição LOO de T_k + linha OOD threshold.

    O modelo precisa estar treinado (`is_fitted()`). Se não estiver, ou
    se a referência leave-one-out estiver vazia, desenha placeholder.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    if model is None or not getattr(model, "is_fitted", lambda: False)():
        ax.text(
            0.5, 0.5, "(modelo não treinado)", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#888",
        )
        ax.axis("off")
        return

    loo = getattr(model, "loo_distribution", None)
    if loo is None or np.asarray(loo).size == 0:
        ax.text(
            0.5, 0.5, "(sem distribuição LOO)", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#888",
        )
        ax.axis("off")
        return

    loo = np.asarray(loo, dtype=np.float64)
    n = int(loo.size)

    x_min = float(np.min(loo))
    x_max = float(np.max(loo))
    if x_min == x_max:
        x_min -= 0.01
        x_max += 0.01
    span = x_max - x_min
    x_min -= 0.05 * span
    x_max += 0.05 * span

    n_bins = min(60, max(20, int(np.sqrt(n))))
    bins = np.linspace(max(-1.0, x_min), min(1.0, x_max), n_bins + 1)

    ax.hist(
        loo, bins=bins, density=True, alpha=0.45,
        color="#1565c0", edgecolor="#0d3a6c", linewidth=0.5,
        label=f"T_k LOO  (n={n:,})".replace(",", "."),
    )

    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(x_min, x_max, 400)
        kde = gaussian_kde(loo)
        ax.plot(xs, kde(xs), color="#0d3a6c", linewidth=1.6, alpha=0.95)
    except Exception:
        pass

    ood = getattr(model, "ood_threshold", None)
    if ood is not None and np.isfinite(ood):
        ax.axvline(
            float(ood), color="#c62828", linestyle="--", linewidth=1.6,
            alpha=0.85,
            label=f"OOD threshold = {ood:.4f}",
        )

    if show_title:
        pop = getattr(model, "population_label", None) or "(sem rótulo)"
        ax.set_title(
            f"Distribuição empírica de T_k (LOO) — {pop}", fontsize=11,
        )
    ax.set_xlabel(r"$T_k$ — similaridade média aos $k$ vizinhos")
    ax.set_ylabel("Densidade")
    ax.grid(True, linestyle="--", alpha=0.4)

    k = getattr(model, "k", "?")
    d = getattr(model, "embedding_dim", "?")
    med = float(np.median(loo))
    mu = float(np.mean(loo))
    sd = float(np.std(loo))
    info = (
        f"k = {k}\n"
        f"d = {d}\n"
        f"mediana = {med:.4f}\n"
        f"média = {mu:.4f}\n"
        f"std = {sd:.4f}"
    )
    ax.text(
        0.985, 0.985, info, transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#bbb", lw=0.7),
    )

    ax.legend(loc="upper left", fontsize=9)


def render_typicality_distribution_png(
    model: Any,
    target_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> None:
    """Renderiza headless via FigureCanvasAgg (sem tocar em pyplot)."""
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    draw_typicality_distribution_from_model(fig, model, show_title=True)
    fig.savefig(target, format="png")


def typicality_distribution_png_path(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
) -> Path:
    """Path absoluto e determinístico do PNG do modelo de tipicidade.

    Convenção:
    `<project>/graphics/<strategy_id>/typicality_models/<resource_id>_dist.png`.
    Subfolder dedicado evita mistura com PNGs de outros tipos no diretório.
    """
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / _sanitize_name(strategy_id)
        / "typicality_models"
        / f"{_sanitize_name(resource_id)}_dist.png"
    )


def _sanitize_name(raw: str) -> str:
    """Normaliza identificador pra uso em nome de arquivo. Mesmo helper que
    `distribution_plot._sanitize_name` — duplicado pra não acoplar módulos."""
    bad = '\\/:*?"<>| '
    out = raw or "unknown"
    for ch in bad:
        out = out.replace(ch, "_")
    return out
