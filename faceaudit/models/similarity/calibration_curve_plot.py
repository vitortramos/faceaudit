"""Renderização da curva de calibração (raw_score × log-LR).

Plot complementar à distribuição: revela a função de calibração em si —
onde fica o crossover LR=1, se há saturação nos extremos, se a curva
tem regiões não-monotônicas (problema clássico de KDE com bandwidth
ruim).

Funções puras de matplotlib — sem dependência de Qt. Usadas em dois
contextos hoje:
  - Snapshot PNG gravado em `<project>/graphics/<strategy>/calibrated_scores/`
    ao persistir um `SimilarityScoreDataset` calibrado. O painel de
    resultados mostra o PNG via QPixmap, sem reconstruir matplotlib.
  - Eventual widget Qt inline / dialog vivo no futuro (reusa o mesmo
    helper `draw_calibration_curve_from_summary`).

O scatter é amostrado pra ≤20k pontos por classe (estratificado),
suficiente pra visualização e leve em memória (~500 KB JSON sidecar).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


_MAX_POINTS_PER_CLASS = 20_000


def compute_calibration_curve_summary(
    items: Iterable,
    *,
    max_points_per_class: int = _MAX_POINTS_PER_CLASS,
    seed: int = 0,
) -> Optional[dict]:
    """Extrai (raw_score, log_lr) por classe a partir dos samples.

    Retorna None se o dataset estiver vazio ou nenhum sample tiver
    `metadata["log_lr"]` (caso típico antes da SCORE_CALIBRATION rodar).

    Subamostragem **estratificada por classe**: cada classe (gen/imp/case)
    é capada em `max_points_per_class` independentemente. Conta total
    preservada em `n_*_total` pra que a legenda mostre o tamanho real
    do dataset com sufixo "(amostragem N)" quando houver corte.
    """
    items = list(items) if items is not None else []
    if not items:
        return None

    gen_pairs: list[tuple[float, float]] = []
    imp_pairs: list[tuple[float, float]] = []
    case_pairs: list[tuple[float, float]] = []

    for sample in items:
        md = getattr(sample, "metadata", None)
        log_lr = md.get("log_lr") if isinstance(md, dict) else None
        if log_lr is None:
            continue
        raw = float(sample.raw_score)
        lr = float(log_lr)
        sp = sample.same_person
        if sp is True:
            gen_pairs.append((raw, lr))
        elif sp is False:
            imp_pairs.append((raw, lr))
        else:
            case_pairs.append((raw, lr))

    if not (gen_pairs or imp_pairs or case_pairs):
        return None

    rng = np.random.default_rng(seed)

    def _maybe_sample(
        pairs: list[tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        total = len(pairs)
        if total == 0:
            return np.zeros(0), np.zeros(0), 0
        if total <= max_points_per_class:
            arr = np.asarray(pairs, dtype=np.float64)
        else:
            idx = rng.choice(total, size=max_points_per_class, replace=False)
            arr = np.asarray([pairs[i] for i in idx], dtype=np.float64)
        return arr[:, 0], arr[:, 1], total

    gen_x, gen_y, n_gen_total = _maybe_sample(gen_pairs)
    imp_x, imp_y, n_imp_total = _maybe_sample(imp_pairs)
    case_x, case_y, n_case_total = _maybe_sample(case_pairs)

    return {
        "schema_version": 1,
        "n_gen": int(gen_x.size),
        "n_imp": int(imp_x.size),
        "n_case": int(case_x.size),
        "n_gen_total": int(n_gen_total),
        "n_imp_total": int(n_imp_total),
        "n_case_total": int(n_case_total),
        "max_points_per_class": int(max_points_per_class),
        "gen_x": gen_x.tolist(),
        "gen_y": gen_y.tolist(),
        "imp_x": imp_x.tolist(),
        "imp_y": imp_y.tolist(),
        "case_x": case_x.tolist(),
        "case_y": case_y.tolist(),
    }


def draw_calibration_curve_from_summary(
    figure: Figure,
    summary: dict,
    *,
    show_title: bool = True,
) -> None:
    """Desenha scatter raw_score × log-LR a partir do summary."""
    figure.clear()
    ax = figure.add_subplot(111)

    n_gen = int(summary.get("n_gen", 0))
    n_imp = int(summary.get("n_imp", 0))
    n_case = int(summary.get("n_case", 0))
    n_gen_total = int(summary.get("n_gen_total", n_gen))
    n_imp_total = int(summary.get("n_imp_total", n_imp))
    n_case_total = int(summary.get("n_case_total", n_case))

    if n_gen == 0 and n_imp == 0 and n_case == 0:
        ax.text(
            0.5, 0.5, "(sem pares calibrados)", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="#888",
        )
        return

    def _legend_label(class_label: str, n: int, n_total: int) -> str:
        if n < n_total:
            return (
                f"{class_label}  (n={n_total:,}, "
                f"amostragem {n:,})".replace(",", ".")
            )
        return f"{class_label}  (n={n_total:,})".replace(",", ".")

    if n_imp > 0:
        ax.scatter(
            summary["imp_x"], summary["imp_y"],
            s=4, alpha=0.30, color="#c62828",
            edgecolors="none",
            label=_legend_label("Impostores", n_imp, n_imp_total),
        )
    if n_gen > 0:
        ax.scatter(
            summary["gen_x"], summary["gen_y"],
            s=4, alpha=0.30, color="#2e7d32",
            edgecolors="none",
            label=_legend_label("Genuínos", n_gen, n_gen_total),
        )
    if n_case > 0:
        # Casework: poucos pontos é o caso típico — marcador maior, sem
        # transparência pra que cada par seja distinguível.
        ax.scatter(
            summary["case_x"], summary["case_y"],
            s=30, alpha=0.9, color="#1565c0",
            edgecolors="#0d3a6c", linewidths=0.8,
            label=_legend_label("Caso", n_case, n_case_total),
        )

    # Linha de referência LR=1 (log-LR=0): qualquer par acima favorece
    # Hp; abaixo, favorece Hd. Crossover visual da função calibrada.
    ax.axhline(
        0.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.6,
        label="log-LR = 0  (LR=1)",
    )

    if show_title:
        ax.set_title("Curva de Calibração", fontsize=11)
    ax.set_xlabel("Escore Bruto")
    ax.set_ylabel("log-LR calibrado")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def render_calibration_curve_png(
    summary: dict,
    target_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> None:
    """Renderiza a curva como PNG headless (Agg backend).

    Lazy import do Agg pra não impactar quem só usa o helper de desenho
    via FigureCanvas Qt. Cria diretório destino se ausente.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    draw_calibration_curve_from_summary(fig, summary, show_title=True)
    fig.savefig(target, format="png")


def calibration_curve_png_path(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
    *,
    strategy_name: str = "",
) -> Path:
    """Path absoluto do PNG snapshot da curva de calibração.

    Convenção:
    `<project>/graphics/<strategy_folder>/calibrated_scores/<resource_id>_curve.png`,
    com `strategy_folder` = `<sanitize(strategy_name)>_<strategy_id>`.

    Estável e determinístico — mesmo path é usado pra gravar (executor
    pós-persist), ler (painel de resultados) e apagar (Apply Changes).
    Subfolder `calibrated_scores` reusa a convenção da distribuição
    pra que tudo associado a um output CALIBRATED_SCORES viva junto.
    """
    from faceaudit.models.strategy.strategy_folder import (
        sanitize_name, strategy_folder_name,
    )
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / strategy_folder_name(strategy_id, strategy_name)
        / "calibrated_scores"
        / f"{sanitize_name(resource_id)}_curve.png"
    )


def _sanitize_name(raw: str) -> str:
    """Normaliza um identificador pra uso em nome de arquivo.

    Espelha `distribution_plot._sanitize_name` — mesmos caracteres
    proibidos.
    """
    bad = '\\/:*?"<>| '
    out = raw or "unknown"
    for ch in bad:
        out = out.replace(ch, "_")
    return out
