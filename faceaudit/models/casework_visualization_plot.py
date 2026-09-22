"""Renderização da visualização forense de caso (CASEWORK_VISUALIZATION).

Funções puras de matplotlib — sem dependência de Qt. Espelham o padrão
do `distribution_plot.py` (snapshot PNG persistido + plot vivo no widget
delegam ao mesmo render, garantindo idéntico visual).

Usado em três contextos:
  - `CaseworkVisualizationWidget` (UI ao vivo, FigureCanvas Qt).
  - Snapshot PNG gravado em `<project>/graphics/` quando o executor
    persiste um `CaseworkVisualizationResult`.
  - Eventuais exports forenses futuros (laudo PDF, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


_KDE_MIN_CASE_SAMPLES = 30


def draw_casework_visualization(
    figure: Figure,
    result: Any,
    *,
    show_title: bool = True,
) -> None:
    """Desenha distribuição forense (caso sobreposto à referência).

    `result` é um `CaseworkVisualizationResult` (mesmo schema usado pelo
    widget vivo). Lógica idêntica à do `CaseworkVisualizationWidget._draw`
    pre-refator — só extraída pra uma função pura aqui pra que o snapshot
    PNG persistido produza exatamente o mesmo gráfico.
    """
    from faceaudit.models.similarity.kde_curve import kde_curve

    figure.clear()
    figure.set_layout_engine("constrained")
    ax = figure.add_subplot(111)

    gen = np.asarray(result.genuine_scores, dtype=np.float64)
    imp = np.asarray(result.impostor_scores, dtype=np.float64)
    case = np.asarray(result.case_scores, dtype=np.float64)

    all_vals = []
    if gen.size > 0:
        all_vals.append(gen)
    if imp.size > 0:
        all_vals.append(imp)
    if case.size > 0:
        all_vals.append(case)
    if not all_vals:
        ax.text(
            0.5, 0.5, "Sem dados para plotar.",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    combined = np.concatenate(all_vals)
    x_min = float(np.min(combined))
    x_max = float(np.max(combined))
    pad = max(0.05 * (x_max - x_min), 0.5)
    x_min -= pad
    x_max += pad
    xs = np.linspace(x_min, x_max, 500)

    if imp.size > 0:
        ax.hist(
            imp, bins=50, density=True, color="#d62728", alpha=0.35,
            label=f"Impostores (n={imp.size:,})".replace(",", "."),
        )
        kde_imp = kde_curve(imp, xs)
        if kde_imp is not None:
            ax.plot(xs, kde_imp, color="#8b0000", linewidth=1.4, alpha=0.9)

    if gen.size > 0:
        ax.hist(
            gen, bins=50, density=True, color="#2ca02c", alpha=0.35,
            label=f"Genuínos (n={gen.size:,})".replace(",", "."),
        )
        kde_gen = kde_curve(gen, xs)
        if kde_gen is not None:
            ax.plot(xs, kde_gen, color="#1d4d1d", linewidth=1.4, alpha=0.9)

    if case.size >= _KDE_MIN_CASE_SAMPLES:
        ax.hist(
            case, bins=30, density=True, color="#1f77b4", alpha=0.35,
            label=f"Caso (n={case.size:,}, KDE)".replace(",", "."),
        )
        kde_case = kde_curve(case, xs)
        if kde_case is not None:
            ax.plot(xs, kde_case, color="#0d3b66", linewidth=1.6, alpha=0.95)
    else:
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        for i, val in enumerate(case):
            label = (
                f"Caso (n={case.size}, marcas)"
                if i == 0 else None
            )
            ax.axvline(
                val, color="#1f77b4", linestyle="--", linewidth=1.4,
                alpha=0.9, label=label,
            )
            ax.annotate(
                f"{val:+.2f}",
                xy=(val, y_top * 0.95),
                xytext=(0, -5),
                textcoords="offset points",
                rotation=90,
                color="#0d3b66",
                fontsize=8,
                ha="center", va="top",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white", ec="#1f77b4", lw=0.8, alpha=0.85,
                ),
            )

    if result.score_type == "CALIBRATED":
        ax.axvline(
            0.0, color="#444", linestyle=":", linewidth=1.0,
            alpha=0.7, label="log-LR = 0 (limiar neutro)",
        )
    elif result.score_type == "RAW" and result.eer_threshold is not None:
        ax.axvline(
            result.eer_threshold, color="#444", linestyle=":", linewidth=1.0,
            alpha=0.7,
            label=f"EER threshold = {result.eer_threshold:.4f}",
        )

    x_label = (
        "log-LR calibrado"
        if result.score_type == "CALIBRATED" else "raw_score"
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Densidade")
    ax.set_xlim(x_min, x_max)
    if show_title:
        ax.set_title("Distribuição forense — Caso sobreposto à referência")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def render_casework_png(
    result: Any,
    target_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> None:
    """Renderiza a visualização forense como PNG headless (Agg backend).

    Lazy import do Agg pra não impactar quem só usa
    `draw_casework_visualization` via FigureCanvas Qt. Cria diretório
    destino se ausente.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    draw_casework_visualization(fig, result, show_title=True)
    fig.savefig(target, format="png")


def casework_visualization_png_path(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
    *,
    strategy_name: str = "",
) -> Path:
    """Path absoluto do PNG snapshot pra um output de CASEWORK_VISUALIZATION.

    Convenção:
    `<project>/graphics/<strategy_folder>/casework_visualization/<resource_id>_casework.png`.
    `strategy_folder` = `<sanitize(strategy_name)>_<strategy_id>`.
    """
    from faceaudit.models.strategy.strategy_folder import (
        sanitize_name, strategy_folder_name,
    )
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / strategy_folder_name(strategy_id, strategy_name)
        / "casework_visualization"
        / f"{sanitize_name(resource_id)}_casework.png"
    )
