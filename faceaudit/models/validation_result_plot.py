"""Renderização dos 4 plots de STRATEGY_EVALUATION (distribuição de LRs,
Tippett, DET, ECE).

Funções puras de matplotlib — sem dependência de Qt. Espelham o padrão
do `distribution_plot.py` e `casework_visualization_plot.py`: o snapshot
PNG persistido em `<project>/graphics/` e o plot vivo no widget delegam
ao mesmo render, garantindo idéntico visual.

Diferente dos outros (1 plot), STRATEGY_EVALUATION produz **4 PNGs
separados** — um por aba — porque a `ValidationResultWidget` é um
`QTabWidget`. Path helper devolve dict com os 4 paths; render helper
gera os 4 PNGs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


# Chaves canônicas usadas no dict de paths e na ordem das abas (UI).
PLOT_KEYS = ("lr_distribution", "tippett", "det", "ece")

# Sufixo de cada plot no nome do arquivo PNG.
_PLOT_SUFFIX = {
    "lr_distribution": "_lr_distribution",
    "tippett":         "_tippett",
    "det":             "_det",
    "ece":             "_ece",
}

# Título legível pra cada aba (UI inline + dialog).
PLOT_TAB_TITLE = {
    "lr_distribution": "Distribuição de LRs",
    "tippett":         "Tippett",
    "det":             "Curva DET",
    "ece":             "Calibração (ECE)",
}


# =========================================================================
# Draw functions — puro matplotlib, recebem (figure, result)
# =========================================================================


def draw_lr_distribution(figure: Figure, result: Any) -> None:
    """Histograma + KDE dos log-LRs por classe."""
    figure.clear()
    ax = figure.add_subplot(111)

    pd = result.plot_data
    gen = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
    imp = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

    if gen.size == 0 and imp.size == 0:
        ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return

    all_lrs = np.concatenate([gen, imp])
    x_min, x_max = float(np.min(all_lrs)), float(np.max(all_lrs))
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    bins = np.linspace(x_min, x_max, 60)

    if imp.size > 0:
        ax.hist(imp, bins=bins, density=True, alpha=0.45,
                color="#c62828", edgecolor="#7a0000", linewidth=0.5,
                label=f"Impostores  (n={imp.size:,})".replace(",", "."))
    if gen.size > 0:
        ax.hist(gen, bins=bins, density=True, alpha=0.45,
                color="#2e7d32", edgecolor="#0d3a10", linewidth=0.5,
                label=f"Genuínos  (n={gen.size:,})".replace(",", "."))

    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(x_min, x_max, 400)
        if imp.size > 1:
            ax.plot(xs, gaussian_kde(imp)(xs), color="#7a0000",
                    linewidth=1.4, alpha=0.9)
        if gen.size > 1:
            ax.plot(xs, gaussian_kde(gen)(xs), color="#0d3a10",
                    linewidth=1.4, alpha=0.9)
    except Exception:
        pass

    ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
               alpha=0.6, label="log-LR = 0")
    ax.set_title("Distribuição de log-LRs", fontsize=11)
    ax.set_xlabel("log-LR calibrado")
    ax.set_ylabel("Densidade")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    figure.tight_layout()


def draw_tippett(figure: Figure, result: Any) -> None:
    """Curvas cumulativas (Tippett) por classe."""
    figure.clear()
    ax = figure.add_subplot(111)

    pd = result.plot_data
    gen = np.sort(np.asarray(pd.genuine_log_lrs, dtype=np.float64))
    imp = np.sort(np.asarray(pd.impostor_log_lrs, dtype=np.float64))

    if gen.size == 0 or imp.size == 0:
        ax.text(0.5, 0.5, "(requer ambas as classes)", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return

    gen_cum = np.arange(1, gen.size + 1) / gen.size
    imp_cum_above = 1.0 - np.arange(1, imp.size + 1) / imp.size

    ax.plot(gen, gen_cum, color="#2e7d32", linewidth=1.6,
            label="Genuínos: P(log-LR ≤ x | Hp)")
    ax.plot(imp, imp_cum_above, color="#c62828", linewidth=1.6,
            label="Impostores: P(log-LR > x | Hd)")

    ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
               alpha=0.6, label="log-LR = 0")

    ax.set_title("Tippett Plot", fontsize=11)
    ax.set_xlabel("log-LR (limiar de decisão)")
    ax.set_ylabel("Proporção cumulativa")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)
    figure.tight_layout()


def draw_det_curve(figure: Figure, result: Any) -> None:
    """Curva DET (FPR × FNR em escala normit) + ponto EER."""
    figure.clear()
    ax = figure.add_subplot(111)

    pd = result.plot_data
    fpr = np.asarray(pd.fpr_array, dtype=np.float64)
    fnr = np.asarray(pd.fnr_array, dtype=np.float64)
    m = result.metrics

    if fpr.size == 0 or fnr.size == 0:
        ax.text(0.5, 0.5, "(sem dados de DET)", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return

    eps = 1e-6
    fpr_c = np.clip(fpr, eps, 1.0 - eps)
    fnr_c = np.clip(fnr, eps, 1.0 - eps)

    try:
        from scipy.stats import norm
        x = norm.ppf(fpr_c)
        y = norm.ppf(fnr_c)
    except Exception:
        x, y = fpr_c, fnr_c

    ax.plot(x, y, color="#1a4f8a", linewidth=1.8, label="DET")

    try:
        from scipy.stats import norm
        eer_n = norm.ppf(np.clip(m.eer, eps, 1.0 - eps))
        ax.scatter([eer_n], [eer_n], color="#E8A020", s=80, zorder=5,
                   edgecolor="#7a4f00", linewidth=1.4,
                   label=f"EER = {m.eer*100:.2f}%")
        ax.axhline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
    except Exception:
        pass

    try:
        from scipy.stats import norm
        tick_pcts = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
        tick_locs = norm.ppf(tick_pcts)
        tick_labels = [f"{p*100:g}%" for p in tick_pcts]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels)
    except Exception:
        pass

    ax.set_title("Curva DET (Detection Error Tradeoff)", fontsize=11)
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Falsos Negativos (FNR)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
    figure.tight_layout()


def draw_ece(figure: Figure, result: Any) -> None:
    """Empirical Cross-Entropy: curva real + curva mínima PAVA + cal_loss."""
    figure.clear()
    ax = figure.add_subplot(111)

    pd = result.plot_data
    x = np.asarray(pd.ece_prior_log_odds, dtype=np.float64)
    y_real = np.asarray(pd.ece_values, dtype=np.float64)
    y_min = np.asarray(pd.ece_min_values, dtype=np.float64)

    if x.size == 0 or y_real.size == 0:
        ax.text(0.5, 0.5, "(sem dados de ECE)", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        return

    if y_min.size == y_real.size:
        ax.fill_between(x, y_min, y_real, where=(y_real >= y_min),
                        color="#E8A020", alpha=0.18,
                        label="Perda de calibração")

    ax.plot(x, y_real, color="#1a4f8a", linewidth=1.8,
            label="ECE (modelo)")
    if y_min.size:
        ax.plot(x, y_min, color="#2e7d32", linewidth=1.6, linestyle="--",
                label="ECE mínimo (PAVA)")

    ax.axvline(0.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_title("Empirical Cross-Entropy (ECE)", fontsize=11)
    ax.set_xlabel("Prior log₁₀-odds (P(Hp)/P(Hd))")
    ax.set_ylabel("Cross-entropy (bits)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", fontsize=9)
    figure.tight_layout()


_DRAW_FN = {
    "lr_distribution": draw_lr_distribution,
    "tippett":         draw_tippett,
    "det":             draw_det_curve,
    "ece":             draw_ece,
}


# =========================================================================
# Render headless (Agg) + path resolver
# =========================================================================


def render_validation_pngs(
    result: Any,
    paths: Dict[str, Union[str, Path]],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> Dict[str, bool]:
    """Renderiza os 4 PNGs de STRATEGY_EVALUATION (Agg headless).

    `paths` é um dict `{plot_key: target_path}` com as chaves canônicas
    de `PLOT_KEYS`. Renderiza apenas os plots cujas chaves estão em
    `paths` — chamador pode passar subset se quiser regenerar só um.

    Retorna dict `{plot_key: bool}` indicando sucesso de cada renderização.
    Falha em um plot NÃO interrompe os demais — best-effort por plot.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    out: Dict[str, bool] = {}
    for key, target in paths.items():
        draw_fn = _DRAW_FN.get(key)
        if draw_fn is None:
            out[key] = False
            continue
        try:
            target_path = Path(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig = Figure(figsize=figsize, dpi=dpi)
            FigureCanvasAgg(fig)
            draw_fn(fig, result)
            fig.savefig(target_path, format="png")
            out[key] = True
        except Exception:
            out[key] = False
    return out


# =========================================================================
# Sidecar de métricas escalares — sobrevive ao drop do payload
# =========================================================================


def validation_metrics_sidecar_path(pickle_path: Union[str, Path]) -> Path:
    """Path do sidecar `<stem>_metrics.json` ao lado do pickle do
    StrategyValidationResult. Carrega instantâneo (~200 bytes JSON) —
    usado pelo painel de resultados pra mostrar a tabela de métricas
    sem precisar hidratar o pickle completo após o usuário trocar de
    estratégia (payload é dropped da memória)."""
    p = Path(pickle_path)
    return p.with_name(p.stem + "_metrics.json")


def write_validation_metrics_sidecar(
    pickle_path: Union[str, Path],
    result: Any,
) -> bool:
    """Grava o sidecar JSON com métricas escalares + contagens.

    Schema (key=valor, todos float/int):
        n_genuines, n_impostors,
        eer, eer_threshold, auc, cllr, cllr_min, cal_loss

    `cal_loss` é derivado (cllr − cllr_min, com floor 0) pra evitar que
    o consumidor tenha que recalcular. Falha silenciosa retorna False.
    """
    import json
    try:
        metrics = getattr(result, "metrics", None)
        if metrics is None:
            return False
        cllr = getattr(metrics, "cllr", None)
        cllr_min = getattr(metrics, "cllr_min", None)
        cal_loss = None
        if cllr is not None and cllr_min is not None:
            cal_loss = max(0.0, float(cllr) - float(cllr_min))

        payload = {
            "schema_version": 1,
            "n_genuines": int(getattr(result, "n_genuines", 0) or 0),
            "n_impostors": int(getattr(result, "n_impostors", 0) or 0),
            "eer": float(getattr(metrics, "eer", 0.0) or 0.0),
            "eer_threshold": float(getattr(metrics, "eer_threshold", 0.0) or 0.0),
            "auc": float(getattr(metrics, "auc", 0.0) or 0.0),
            "cllr": float(cllr) if cllr is not None else None,
            "cllr_min": float(cllr_min) if cllr_min is not None else None,
            "cal_loss": cal_loss,
        }
        sidecar = validation_metrics_sidecar_path(pickle_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8",
        )
        return True
    except Exception:
        return False


def try_load_validation_metrics_sidecar(
    pickle_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Tenta ler `<stem>_metrics.json` do disco. Retorna dict ou None.

    Stale-check via mtime: se o pickle foi modificado depois do sidecar
    (+1s de folga pra clock skew), invalida — força quem chamou a
    hidratar o payload e reescrever o sidecar.
    """
    import json
    try:
        p = Path(pickle_path)
        sidecar = validation_metrics_sidecar_path(p)
        if not sidecar.exists():
            return None
        if p.exists() and p.stat().st_mtime > sidecar.stat().st_mtime + 1.0:
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def validation_result_png_paths(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
) -> Dict[str, Path]:
    """Paths absolutos dos 4 PNGs snapshot pra um output de STRATEGY_EVALUATION.

    Convenção:
    `<project>/graphics/<strategy_id>/validation_result/<resource_id>_<plot>.png`
    com `<plot>` ∈ {`lr_distribution`, `tippett`, `det`, `ece`}.
    """
    # Reusa o `_sanitize_name` de distribution_plot pra evitar drift.
    from faceaudit.models.similarity.distribution_plot import _sanitize_name

    folder = (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / _sanitize_name(strategy_id)
        / "validation_result"
    )
    safe_resource = _sanitize_name(resource_id)
    return {
        key: folder / f"{safe_resource}{_PLOT_SUFFIX[key]}.png"
        for key in PLOT_KEYS
    }
