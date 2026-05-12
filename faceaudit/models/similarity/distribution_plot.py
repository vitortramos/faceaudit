"""Renderização da distribuição de scores (genuínos × impostores × caso).

Funções puras de matplotlib — sem dependência de Qt. Usadas em três
contextos:
  - `ScoreDistributionWidget` (UI ao vivo, FigureCanvas Qt).
  - Snapshot PNG gravado em `<project>/graphics/` ao persistir um
    `SimilarityScoreDataset` (executor → painel de resultados mostra
    o PNG instantâneo via QPixmap, sem reconstruir a figura matplotlib).
  - Eventuais exports forenses futuros (laudo PDF, etc.).

Mantém o mesmo estilo visual em todos os pontos (mesmas cores, mesmos
bins, mesmo título) — caller passa o `figure` (`Figure` matplotlib) e
o dict `summary` pré-computado pelo
`SimilarityScoreDataset._compute_distribution_summary`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


def draw_distribution_from_summary(
    figure: Figure,
    summary: dict,
    *,
    show_title: bool = True,
) -> None:
    """Desenha histograma + KDE no `figure` a partir de `summary` pré-computado.

    `summary` é o dict produzido por
    `SimilarityScoreDataset._compute_distribution_summary` (schema_version=2):
      score_type ('raw' | 'log_lr'), name, n_gen, n_imp, n_case,
      x_min, x_max, bins, gen_hist_density, imp_hist_density,
      case_hist_density | case_values, kde_x, kde_gen_y, kde_imp_y,
      kde_case_y.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    score_type = summary.get("score_type", "raw")
    use_log_lr = score_type == "log_lr"
    n_gen = int(summary.get("n_gen", 0))
    n_imp = int(summary.get("n_imp", 0))
    n_case = int(summary.get("n_case", 0))
    case_values = summary.get("case_values") or []

    if n_gen == 0 and n_imp == 0 and n_case == 0:
        ax.text(
            0.5, 0.5, "(sem scores)", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#888",
        )
        return

    bins = np.asarray(summary["bins"], dtype=np.float64)
    widths = np.diff(bins)
    centers = bins[:-1] + widths / 2.0

    if n_imp > 0:
        ax.bar(
            centers, summary["imp_hist_density"],
            width=widths, alpha=0.45,
            color="#c62828", edgecolor="#7a0000", linewidth=0.5,
            label=f"Impostores  (n={n_imp:,})".replace(",", "."),
        )
    if n_gen > 0:
        ax.bar(
            centers, summary["gen_hist_density"],
            width=widths, alpha=0.45,
            color="#2e7d32", edgecolor="#0d3a10", linewidth=0.5,
            label=f"Genuínos  (n={n_gen:,})".replace(",", "."),
        )
    if n_case > 0:
        case_hist = summary.get("case_hist_density")
        if case_hist is not None and len(case_hist) == centers.size:
            ax.bar(
                centers, case_hist, width=widths, alpha=0.45,
                color="#1565c0", edgecolor="#0d3a6c", linewidth=0.5,
                label=f"Caso  (n={n_case:,})".replace(",", "."),
            )
        elif case_values:
            for i, val in enumerate(case_values):
                ax.axvline(
                    float(val), color="#1565c0", linestyle="--",
                    linewidth=1.4, alpha=0.85,
                    label=(f"Caso  (n={n_case})" if i == 0 else None),
                )

    kde_x = summary.get("kde_x")
    if kde_x is not None:
        kde_x_arr = np.asarray(kde_x, dtype=np.float64)
        kde_imp_y = summary.get("kde_imp_y")
        kde_gen_y = summary.get("kde_gen_y")
        kde_case_y = summary.get("kde_case_y")
        if kde_imp_y is not None:
            ax.plot(
                kde_x_arr, np.asarray(kde_imp_y, dtype=np.float64),
                color="#7a0000", linewidth=1.4, alpha=0.9,
            )
        if kde_gen_y is not None:
            ax.plot(
                kde_x_arr, np.asarray(kde_gen_y, dtype=np.float64),
                color="#0d3a10", linewidth=1.4, alpha=0.9,
            )
        if kde_case_y is not None:
            ax.plot(
                kde_x_arr, np.asarray(kde_case_y, dtype=np.float64),
                color="#0d3a6c", linewidth=1.4, alpha=0.9,
            )

    if show_title:
        # Título enxuto — antes era "Distribuição de Scores — <dataset>"
        # numa linha + "Total: N pares" em outra. Em cells estreitos do
        # painel de resultados (~280px) o título quebrava feio e nem
        # cabia. Nome do dataset + contagens já aparecem no header do
        # card (3ª coluna).
        ax.set_title(
            "Distribuição de log-LR" if use_log_lr else "Distribuição de Escores",
            fontsize=11,
        )
    ax.set_xlabel("log-LR calibrado" if use_log_lr else "Escore Bruto")
    ax.set_ylabel("Densidade")
    if use_log_lr:
        ax.axvline(
            0.0, color="#444", linestyle=":", linewidth=1.0,
            alpha=0.6, label="log-LR = 0",
        )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def render_distribution_png(
    summary: dict,
    target_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> None:
    """Renderiza a distribuição como PNG headless (Agg backend).

    Lazy import do Agg pra não impactar quem só usa `draw_distribution_from_summary`
    via FigureCanvas Qt. Cria diretório destino se ausente.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Use Figure direto + FigureCanvasAgg — não toca em `pyplot` (que
    # poderia conflitar com o backend Qt usado pela UI).
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    draw_distribution_from_summary(fig, summary, show_title=True)
    fig.savefig(target, format="png")


def score_distribution_png_path(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
    *,
    calibrated: bool = False,
) -> Path:
    """Path absoluto do PNG snapshot de distribuição de scores.

    Convenção:
    `<project>/graphics/<strategy_id>/<subfolder>/<resource_id>_dist.png`,
    com `subfolder` = `similarity_scores` (default) ou `calibrated_scores`
    (quando `calibrated=True`). Subfolders por tipo de saída evitam que
    a pasta `graphics/` vire um lixão flat de centenas de PNGs.

    Estável e determinístico — mesmo path é usado pra gravar (executor
    pós-persist), ler (painel de resultados) e apagar (Apply Changes).
    """
    subfolder = "calibrated_scores" if calibrated else "similarity_scores"
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / _sanitize_name(strategy_id)
        / subfolder
        / f"{_sanitize_name(resource_id)}_dist.png"
    )


def _sanitize_name(raw: str) -> str:
    """Normaliza um identificador pra uso em nome de arquivo."""
    bad = '\\/:*?"<>| '
    out = raw or "unknown"
    for ch in bad:
        out = out.replace(ch, "_")
    return out
