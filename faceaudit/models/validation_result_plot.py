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
PLOT_KEYS = (
    "lr_distribution", "tippett", "det", "ece", "curve",
    "affine_diag", "quantile_log_lr",
)

# Sufixo de cada plot no nome do arquivo PNG.
_PLOT_SUFFIX = {
    "lr_distribution": "_lr_distribution",
    "tippett":         "_tippett",
    "det":             "_det",
    "ece":             "_ece",
    "curve":           "_calibration_curve",
    "affine_diag":     "_affine_diagnostic",
    "quantile_log_lr": "_quantile_log_lr",
}

# Título legível pra cada aba (UI inline + dialog).
PLOT_TAB_TITLE = {
    "lr_distribution": "Distribuição de LRs",
    "tippett":         "Tippett",
    "det":             "Curva DET",
    "ece":             "Calibração (ECE)",
    "curve":           "Curva de Calibração",
    "affine_diag":     "Diagnóstico Afim",
    "quantile_log_lr": "Quantis por Classe",
}


# =========================================================================
# Draw functions — puro matplotlib, recebem (figure, result)
# =========================================================================


_ANNOTATION_BBOX = dict(
    boxstyle="round,pad=0.5",
    facecolor="white",
    edgecolor="#bbb",
    alpha=0.85,
    linewidth=0.6,
)


def _add_annotation(ax, text: str, loc: str = "upper right") -> None:
    """Anotação textual discreta num canto do plot.

    `loc` aceita 'upper right', 'upper left', 'lower right', 'lower left'.
    Posicionada em coordenadas dos axes (0-1), com bbox semi-transparente
    pra não competir com os dados.
    """
    pos = {
        "upper right": (0.98, 0.97, "right", "top"),
        "upper left":  (0.02, 0.97, "left",  "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left":  (0.02, 0.03, "left",  "bottom"),
    }.get(loc, (0.98, 0.97, "right", "top"))
    ax.text(
        pos[0], pos[1], text,
        transform=ax.transAxes,
        ha=pos[2], va=pos[3],
        fontsize=8.5, color="#333",
        bbox=_ANNOTATION_BBOX,
        family="monospace",
    )


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

    # Anotação: taxas de "sinal errado" (Tier 1 forense)
    m = getattr(result, "metrics", None)
    wsg = getattr(m, "wrong_sign_gen_rate", None) if m else None
    wsi = getattr(m, "wrong_sign_imp_rate", None) if m else None
    if wsg is not None and wsi is not None:
        text = (
            f"Falso+ (LR>0|Hd): {wsi*100:.2f}%\n"
            f"Falso− (LR<0|Hp): {wsg*100:.2f}%"
        )
        _add_annotation(ax, text, loc="upper right")

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
    """Empirical Cross-Entropy: curva real + curva mínima PAVA + curva de
    referência (sistema neutro LR=1) + cal_loss.

    A curva de referência é a **entropia do prior** — ECE de um sistema que
    sempre reporta LR=1 (não decide). Convenção forense estabelecida
    (Brümmer & van Leeuwen 2006): só dá pra dizer que o modelo é
    informativo quando ECE_model < ECE_null. Se ECE_model ultrapassa
    ECE_null em algum prior, o sistema é literalmente pior que não decidir
    ali.

    Computada on-the-fly de `ece_prior_log_odds` (não precisa regravar
    pickles antigos).
    """
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

    # Curva de referência: ECE_null = H(P) onde P = sigmoid(x · ln10).
    # Em log₁₀-odds=0, P=0.5 e H=1 bit (pico). Nas caudas, P→0 ou 1 e H→0.
    p = 1.0 / (1.0 + np.exp(-x * np.log(10.0)))
    # Clip pra evitar log(0). Bits → / ln2.
    p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
    y_null = (-p_safe * np.log(p_safe) - (1.0 - p_safe) * np.log(1.0 - p_safe)) / np.log(2.0)

    if y_min.size == y_real.size:
        ax.fill_between(x, y_min, y_real, where=(y_real >= y_min),
                        color="#E8A020", alpha=0.18,
                        label="Perda de calibração")

    # Plot ordering: null (atrás, cinza) → min (verde tracejado) → real (azul).
    ax.plot(x, y_null, color="#888", linewidth=1.2, linestyle=":",
            label="Sistema neutro (LR=1) — referência")
    if y_min.size:
        ax.plot(x, y_min, color="#2e7d32", linewidth=1.6, linestyle="--",
                label="ECE mínimo (PAVA)")
    ax.plot(x, y_real, color="#1a4f8a", linewidth=1.8,
            label="ECE (modelo)")

    ax.axvline(0.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_title("Empirical Cross-Entropy (ECE)", fontsize=11)
    ax.set_xlabel("Prior log₁₀-odds (P(Hp)/P(Hd))")
    ax.set_ylabel("Cross-entropy (bits)")
    # Y vai até 1.05 (pouco além da entropia máxima do prior = 1.0).
    # Se modelo exceder 1.0 em algum prior (sistema pior que neutro nessa
    # região), permite ver até onde vai sem clipar agressivamente.
    y_max_plot = max(1.05, float(np.nanmax(y_real)) * 1.05)
    ax.set_ylim(0, y_max_plot)
    ax.grid(True, linestyle="--", alpha=0.4)
    # `upper center` sobreposta no pico da curva. Mover pra esquerda — em
    # forense ECE tipicamente tem pico no centro/direita, esquerda é onde
    # priors de Hd dominam (área visualmente mais limpa).
    ax.legend(loc="upper left", fontsize=8.5)

    # Anotação: métricas numéricas da ECE (Tier 4)
    m = getattr(result, "metrics", None)
    loss_area = getattr(m, "ece_calibration_loss_area", None) if m else None
    max_gap = getattr(m, "ece_max_gap", None) if m else None
    argmax_prior = getattr(m, "ece_argmax_prior_log_odds", None) if m else None
    if loss_area is not None and max_gap is not None:
        lines = [f"Perda total: {loss_area:.4f} bits·dec"]
        if argmax_prior is not None:
            lines.append(f"Gap máx: {max_gap:.4f} @ prior={argmax_prior:+.2f}")
        else:
            lines.append(f"Gap máx: {max_gap:.4f} bits")
        _add_annotation(ax, "\n".join(lines), loc="upper right")

    figure.tight_layout()


def draw_calibration_curve(figure: Figure, result: Any) -> None:
    """Curva de Calibração v2: scatter raw × log-LR + curva pós-afim
    Brümmer + Δ residual.

    Layout 2-panel quando o pickle traz o grid pré-computado em
    `plot_data.cal_curve_*` (versão nova do executor). Fallback pra
    layout single-panel (visual legado) quando o grid está vazio
    (pickles antigos).

    **Plot top:**
      - scatter (gen=verde, imp=vermelho) — pontos observados;
      - linha sólida azul = calibrador atual f(x) interpolado no grid;
      - linha tracejada laranja = pós-afim Brümmer-Doddington:
        `slope·f(x) + intercept`. Só vira reta no plano raw×logLR
        quando f é linear (LogReg). Pra KDE/BiGauss é curva;
      - axhline log-LR=0 (LR=1, neutralidade forense);
      - axvline em raw_score onde a pós-afim cruza log-LR=0;
      - rug ticks de gen/imp na borda inferior — contextualiza onde
        há massa de dados.

    **Plot bottom:**
      - Δ log-LR após recalibração afim = `(slope−1)·f(x) + intercept`;
      - axhline em 0 (ponto onde a afim não muda o log-LR).
    """
    pd = result.plot_data
    gen_raw = np.asarray(getattr(pd, "genuine_raw_scores", []), dtype=np.float64)
    imp_raw = np.asarray(getattr(pd, "impostor_raw_scores", []), dtype=np.float64)
    gen_lr = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
    imp_lr = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

    if gen_raw.size == 0 and imp_raw.size == 0:
        figure.clear()
        ax = figure.add_subplot(111)
        ax.text(
            0.5, 0.5,
            "(curva indisponível — pickle pré-curva,\n"
            "re-execute STRATEGY_EVALUATION)",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=10,
        )
        return

    grid_x = np.asarray(getattr(pd, "cal_curve_grid_x", []), dtype=np.float64)
    has_grid = grid_x.size >= 2

    # Subsample do scatter (estratificado por classe).
    rng = np.random.default_rng(0)
    MAX_PER_CLASS = 20_000

    def _maybe_sample(raw: np.ndarray, lr: np.ndarray):
        n = min(raw.size, lr.size)
        if n <= MAX_PER_CLASS:
            return raw[:n].tolist(), lr[:n].tolist(), n
        idx = rng.choice(n, size=MAX_PER_CLASS, replace=False)
        return raw[idx].tolist(), lr[idx].tolist(), n

    gen_x_s, gen_y_s, n_gen_total = _maybe_sample(gen_raw, gen_lr)
    imp_x_s, imp_y_s, n_imp_total = _maybe_sample(imp_raw, imp_lr)

    summary = {
        "schema_version": 1,
        "n_gen": len(gen_x_s),
        "n_imp": len(imp_x_s),
        "n_case": 0,
        "n_gen_total": n_gen_total,
        "n_imp_total": n_imp_total,
        "n_case_total": 0,
        "max_points_per_class": MAX_PER_CLASS,
        "gen_x": gen_x_s, "gen_y": gen_y_s,
        "imp_x": imp_x_s, "imp_y": imp_y_s,
        "case_x": [], "case_y": [],
    }

    from faceaudit.models.similarity.calibration_curve_plot import (
        draw_calibration_curve_from_summary,
    )

    if has_grid:
        figure.clear()
        # Constrained layout reserva espaço pra ylabel/xlabel/legenda em
        # ambos os subplots — compatível com gridspec (tight_layout não é).
        figure.set_layout_engine("constrained")
        gs = figure.add_gridspec(2, 1, height_ratios=[3, 1])
        ax_top = figure.add_subplot(gs[0])
        ax_bot = figure.add_subplot(gs[1], sharex=ax_top)
        # Scatter no ax_top. Contagens N_imp/N_gen vão pro título (ajuste
        # após buscar as métricas, mais abaixo), então `show_counts=False`
        # deixa as labels da legenda mais enxutas.
        draw_calibration_curve_from_summary(
            figure, summary, show_title=False, ax=ax_top, show_counts=False,
        )
    else:
        # Single-panel legado: helper limpa figure e cria seu próprio subplot.
        draw_calibration_curve_from_summary(figure, summary, show_title=True)
        ax_top = figure.axes[0] if figure.axes else None
        ax_bot = None

    m = getattr(result, "metrics", None)
    slope = getattr(m, "calibration_slope", None) if m else None
    intercept = getattr(m, "calibration_intercept", None) if m else None
    cllr_aff = getattr(m, "cllr_affine", None) if m else None
    cllr_val = getattr(m, "cllr", None) if m else None
    cllr_min_val = getattr(m, "cllr_min", None) if m else None

    # Diagnóstico do slope — usado tanto no subtítulo do gráfico quanto
    # (até a versão anterior) na label da pós-afim.
    if slope is None:
        slope_diag = None
    elif slope < 0.95:
        slope_diag = "over-confident"
    elif slope > 1.05:
        slope_diag = "under-confident"
    else:
        slope_diag = "bem calibrado"

    # Título multilinha:
    # (1) identidade do gráfico + contagens;
    # (2) decomposição de Cllr (Cllr / Cllr_affine / Cllr_min);
    # (3) parâmetros do afim (slope+diag, intercept).
    # Quebrado pra evitar overflow horizontal no painel inline (figure
    # redimensionada pra ~400-500 px). Move informação numérica do
    # quadro da legenda pro topo — legenda fica só com handles visuais.
    if has_grid and ax_top is not None:
        n_imp_total = int(summary.get("n_imp_total", 0))
        n_gen_total = int(summary.get("n_gen_total", 0))
        title_line1 = (
            f"Curva de Calibração  "
            f"({n_imp_total:,} impostores · {n_gen_total:,} genuínos)"
        ).replace(",", ".")
        line2_parts: list[str] = []
        if cllr_val is not None:
            line2_parts.append(f"Cllr={cllr_val:.4f}")
        if cllr_aff is not None:
            line2_parts.append(f"Cllr_affine={cllr_aff:.4f}")
        if cllr_min_val is not None:
            line2_parts.append(f"Cllr_min={cllr_min_val:.4f}")
        line3_parts: list[str] = []
        if slope is not None and slope_diag is not None:
            line3_parts.append(f"slope={slope:.3f} ({slope_diag})")
        if intercept is not None:
            line3_parts.append(f"intercept={intercept:+.3f}")
        title_lines = [title_line1]
        if line2_parts:
            title_lines.append("  ·  ".join(line2_parts))
        if line3_parts:
            title_lines.append("  ·  ".join(line3_parts))
        ax_top.set_title("\n".join(title_lines), fontsize=10)

    # ── Camadas adicionais no top axes (só com grid) ─────────────────────
    if has_grid and ax_top is not None:
        llr_y = np.asarray(pd.cal_curve_llr_y, dtype=np.float64)
        affine_y = np.asarray(pd.cal_curve_affine_y, dtype=np.float64)
        zero_raw = getattr(pd, "cal_curve_zero_raw_score", None)
        rug_gen = np.asarray(getattr(pd, "cal_curve_rug_gen_x", []), dtype=np.float64)
        rug_imp = np.asarray(getattr(pd, "cal_curve_rug_imp_x", []), dtype=np.float64)

        # Curva f(x) — calibrador atual.
        ax_top.plot(
            grid_x, llr_y,
            color="#1a4f8a", linewidth=2.0, alpha=0.95,
            label="Calibrador atual f(x)",
        )
        # Curva pós-afim Brümmer-Doddington. Apenas a fórmula vai na
        # label — diagnóstico (over-/under-confident) e Cllr_affine
        # estão no subtítulo do gráfico, fora da legenda.
        if slope is not None and intercept is not None:
            sign = "+" if intercept >= 0 else "−"
            ax_top.plot(
                grid_x, affine_y,
                color="#E8A020", linewidth=2.0, linestyle="--", alpha=0.95,
                label=(
                    f"Pós-afim: {slope:.3f}·logLR {sign} {abs(intercept):.3f}"
                ),
            )
        # Zero-crossing da pós-afim no eixo raw_score. Label entra na
        # legenda como entrada com o tracinho amarelo característico.
        if zero_raw is not None:
            ax_top.axvline(
                float(zero_raw),
                color="#E8A020", linestyle=":", linewidth=1.2, alpha=0.7,
                label=(
                    f"x₀={float(zero_raw):+.3f} "
                    f"(pós-afim → logLR=0)"
                ),
            )

        # Rug ticks (gen=verde, imp=vermelho) na borda inferior — pega
        # ylim DEPOIS das linhas (que podem ter expandido a faixa).
        ymin, ymax = ax_top.get_ylim()
        rug_h = (ymax - ymin) * 0.025
        if rug_imp.size > 0:
            ax_top.vlines(
                rug_imp, ymin, ymin + rug_h,
                color="#c62828", alpha=0.5, linewidth=0.8,
            )
        if rug_gen.size > 0:
            ax_top.vlines(
                rug_gen, ymin, ymin + rug_h,
                color="#2e7d32", alpha=0.5, linewidth=0.8,
            )
        ax_top.set_ylim(ymin - rug_h * 0.4, ymax)

        # Legenda única no upper-left consolidando tudo: scatters (sem
        # contagens — vão no título), referência logLR=0, curva f(x),
        # pós-afim (com slope/intercept/diag/Cllr_affine na label) e
        # x₀ via axvline. Substitui a antiga caixa de anotação textual.
        ax_top.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

        # No 2-panel, top compartilha eixo x com bottom — esconde xticks/label.
        ax_top.set_xlabel("")
        ax_top.tick_params(labelbottom=False)

    # ── Bottom axes: Δ log-LR (resíduo) ─────────────────────────────────
    if has_grid and ax_bot is not None:
        delta_y = np.asarray(pd.cal_curve_affine_delta_y, dtype=np.float64)
        ax_bot.plot(
            grid_x, delta_y,
            color="#E8A020", linewidth=1.5, alpha=0.95,
        )
        ax_bot.axhline(
            0.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.6,
        )
        ax_bot.set_xlabel("Escore Bruto")
        ax_bot.set_ylabel("Δ log-LR", fontsize=9)
        ax_bot.grid(True, linestyle="--", alpha=0.4)
        ax_bot.tick_params(labelsize=8)
    else:
        # Single-panel: constrained não está ligado, tight_layout dá ajuste.
        figure.tight_layout()


def _compute_pava_mapping(
    gen_lr: np.ndarray, imp_lr: np.ndarray, x_grid: np.ndarray,
) -> Optional[np.ndarray]:
    """Calcula o mapeamento PAVA (isotônico) avaliado num grid de x's.

    PAVA = Pool Adjacent Violators — regressão isotônica que produz o
    mapeamento monotônico ótimo. Equivale ao calibrador que atinge
    `Cllr_min` (limite teórico de calibração monotônica).

    Sequência:
      1. Fit isotônico em (gen+imp scores, labels);
      2. Avalia no `x_grid` via interpolação (out_of_bounds='clip');
      3. Converte probabilidade → log-LR (com correção de prior_log_odds
         pra remover o efeito do prior empírico do dataset).

    Retorna None quando não há ambas as classes (PAVA precisa de gen E
    imp). Mantém a convenção do `ValidationMathService._get_calibrated_llrs`.
    """
    if gen_lr.size == 0 or imp_lr.size == 0:
        return None
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return None
    scores = np.concatenate([gen_lr, imp_lr])
    labels = np.concatenate([
        np.ones(gen_lr.size, dtype=np.float64),
        np.zeros(imp_lr.size, dtype=np.float64),
    ])
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    p = ir.transform(x_grid)
    eps = 1e-14
    p = np.clip(p, eps, 1.0 - eps)
    prior_log_odds = float(np.log(gen_lr.size / imp_lr.size))
    return np.log(p / (1.0 - p)) - prior_log_odds


def draw_affine_diagnostic(figure: Figure, result: Any) -> None:
    """Diagnóstico Afim em escala log-LR — decomposição completa de Cllr.

    Reescala a calibração para o eixo log-LR (unidade comum entre
    estratégias) e mostra a decomposição:
      ``Cllr  ≥  Cllr_affine  ≥  Cllr_min``
       │              │              │
       │              │              └ PAVA (verde-azulado): limite
       │              │                teórico monotônico.
       │              └ Pós-afim (laranja): melhor correção linear.
       └ Atual: status quo, equivalente a y=x neste plano.

    Camadas:
      - Identidade y=x (cinza tracejada) — sem recalibração afim;
      - Pós-afim y=slope·x + intercept (laranja sólida) — Brümmer ótimo;
      - PAVA monotônico (verde-azulado, step) — Cllr_min reachable;
      - axhline y=0 + axvline x=0 (cinza pontilhadas — neutralidade);
      - Rug ticks dos log-LRs observados na borda inferior.

    Leituras:
      - **Gap y=x → afim** (vertical) = `Cllr − Cllr_affine` (recuperável
        por slope+intercept).
      - **Gap afim → PAVA** (vertical) = `Cllr_affine − Cllr_min`
        (recuperável APENAS por mapeamento não-linear monotônico —
        afim não consegue).
      - **PAVA colada na afim** → calibrador é essencialmente linear.
      - **PAVA curva nas caudas** afastando-se da afim → não-linearidade
        irrecuperável pela afim (típico de over-confidence nas caudas).
    """
    figure.clear()
    # Constrained layout reserva espaço corretamente pra título multilinha;
    # tight_layout não acomoda bem subtítulos com 5+ métricas.
    figure.set_layout_engine("constrained")
    ax = figure.add_subplot(111)

    pd = result.plot_data
    gen_lr = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
    imp_lr = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

    m = getattr(result, "metrics", None)
    slope = getattr(m, "calibration_slope", None) if m else None
    intercept = getattr(m, "calibration_intercept", None) if m else None
    cllr_aff = getattr(m, "cllr_affine", None) if m else None
    cllr_min_val = getattr(m, "cllr_min", None) if m else None

    if (gen_lr.size == 0 and imp_lr.size == 0) or slope is None or intercept is None:
        ax.text(
            0.5, 0.5,
            "(diagnóstico afim indisponível — slope/intercept ou\n"
            "log-LRs não presentes no pickle)",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=10,
        )
        return

    # Range do eixo X = log-LR atual observado.
    all_lr = np.concatenate([gen_lr, imp_lr])
    x_min = float(all_lr.min())
    x_max = float(all_lr.max())
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    x_grid = np.linspace(x_min - pad, x_max + pad, 200)

    # Identidade y=x: referência de "nenhuma correção afim necessária".
    # NÃO é o calibrador afim ótimo — quem é ótimo é a reta laranja
    # (slope, intercept dados pelo Brümmer-Doddington).
    ax.plot(
        x_grid, x_grid,
        color="#888", linewidth=1.2, linestyle="--", alpha=0.85,
        label="y = x  (sem recalibração afim)",
    )
    # Reta pós-afim.
    sign = "+" if intercept >= 0 else "−"
    ax.plot(
        x_grid, slope * x_grid + intercept,
        color="#E8A020", linewidth=2.0, alpha=0.95,
        label=f"Pós-afim: {slope:.3f}·logLR {sign} {abs(intercept):.3f}",
    )
    # PAVA monotônico — limite teórico (Cllr_min reachable).
    pava_grid = _compute_pava_mapping(gen_lr, imp_lr, x_grid)
    if pava_grid is not None:
        ax.plot(
            x_grid, pava_grid,
            color="#1f7a8c", linewidth=1.6, alpha=0.9,
            drawstyle="steps-mid",
            label="PAVA monotônico (Cllr_min)",
        )
    # Eixos de neutralidade.
    ax.axhline(0.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.5)

    # Determine y range pós-plot (precisamos do ylim final antes do rug)
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    rug_h = (ymax - ymin) * 0.025
    if imp_lr.size > 0:
        # Rug subamostrado (cap 2000 por classe) — mesmas convenções do
        # rug da Curva de Calibração.
        rng = np.random.default_rng(0)
        imp_rug = imp_lr if imp_lr.size <= 2000 else imp_lr[
            rng.choice(imp_lr.size, size=2000, replace=False)
        ]
        ax.vlines(
            imp_rug, ymin, ymin + rug_h,
            color="#c62828", alpha=0.5, linewidth=0.8,
        )
    if gen_lr.size > 0:
        rng = np.random.default_rng(1)
        gen_rug = gen_lr if gen_lr.size <= 2000 else gen_lr[
            rng.choice(gen_lr.size, size=2000, replace=False)
        ]
        ax.vlines(
            gen_rug, ymin, ymin + rug_h,
            color="#2e7d32", alpha=0.5, linewidth=0.8,
        )
    ax.set_ylim(ymin - rug_h * 0.4, ymax)

    # Diagnóstico textual no subtítulo + decomposição numérica de Cllr.
    # Quebrado em 2 linhas: linha 2 = parâmetros do afim, linha 3 =
    # decomposição numérica de Cllr. Painel inline da UI redimensiona
    # a figure pra ~400-500 px de largura — single-line subtitle estoura.
    if slope < 0.95:
        diag = "over-confident (afim comprime)"
    elif slope > 1.05:
        diag = "under-confident (afim expande)"
    else:
        diag = "bem calibrado (afim ≈ identidade)"
    line2_parts = [f"slope={slope:.3f} ({diag})", f"intercept={intercept:+.3f}"]
    line3_parts = []
    if cllr_aff is not None:
        line3_parts.append(f"Cllr_affine={cllr_aff:.4f}")
    if cllr_min_val is not None:
        line3_parts.append(f"Cllr_min={cllr_min_val:.4f}")
    # Perda não-linear (gap afim → PAVA): a parte do nonlinear_cal_loss
    # que o afim não recupera.
    if cllr_aff is not None and cllr_min_val is not None:
        nl_loss = max(0.0, float(cllr_aff) - float(cllr_min_val))
        line3_parts.append(f"nonlinear_loss={nl_loss:.4f}")
    title_lines = ["Diagnóstico Afim em Escala log-LR",
                   "  ·  ".join(line2_parts)]
    if line3_parts:
        title_lines.append("  ·  ".join(line3_parts))
    ax.set_title("\n".join(title_lines), fontsize=10)

    ax.set_xlabel("log-LR atual")
    ax.set_ylabel("log-LR pós-correção")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)


def draw_quantile_log_lr(figure: Figure, result: Any) -> None:
    """Quantis por Classe — percentis do log-LR per-classe.

    Eixo X: percentis fixos P1, P5, P10, P25, P50, P75, P90, P95, P99
    em escala linear (1..99). Eixo Y: log-LR calibrado nos respectivos
    percentis, separado por classe.

    Como o calibrador é monotônico (raw_score → log-LR), os percentis
    do raw_score por classe coincidem com os percentis do log-LR por
    classe. Trabalha-se direto com `genuine_log_lrs` / `impostor_log_lrs`.

    Leituras forenses:
      - **P95/P99 dos impostores** = impostores mais perigosos, parecidos
        com genuínos. log-LR aqui próximo de 0 ou positivo é sinal de
        falsos+ potenciais.
      - **P1/P5 dos genuínos** = genuínos mais difíceis. log-LR aqui
        próximo de 0 ou negativo é sinal de falsos− potenciais.
      - Distância vertical entre as duas séries em cada percentil mostra
        o gap discriminativo da estratégia naquele ponto de operação.

    Escala de X (percentil) é comum entre estratégias — permite overlay
    direto na Comparação independente da unidade de raw_score.
    """
    figure.clear()
    figure.set_layout_engine("constrained")
    ax = figure.add_subplot(111)

    pd = result.plot_data
    gen_lr = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
    imp_lr = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

    if gen_lr.size == 0 and imp_lr.size == 0:
        ax.text(
            0.5, 0.5,
            "(quantis indisponíveis — log-LRs não presentes no pickle)",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=10,
        )
        return

    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=np.float64)
    # Percentis críticos da região de overlap forense:
    # - P1/P5 dos genuínos = genuínos mais difíceis (potenciais falsos−).
    # - P95/P99 dos impostores = impostores mais perigosos (potenciais falsos+).
    gen_critical_mask = np.isin(percentiles, [1.0, 5.0])
    imp_critical_mask = np.isin(percentiles, [95.0, 99.0])

    gen_q = imp_q = None
    if gen_lr.size > 0:
        gen_q = np.percentile(gen_lr, percentiles)
        ax.plot(
            percentiles, gen_q,
            color="#2e7d32", marker="o", markersize=6,
            linewidth=1.8, alpha=0.95,
            label=f"Genuínos (n={gen_lr.size:,})".replace(",", "."),
        )
        # Halo nos pontos críticos: anel verde-escuro vazado.
        if gen_critical_mask.any():
            ax.scatter(
                percentiles[gen_critical_mask], gen_q[gen_critical_mask],
                s=160, facecolors="none", edgecolors="#1b5e20",
                linewidths=2.2, zorder=5, label="_nolegend_",
            )
    if imp_lr.size > 0:
        imp_q = np.percentile(imp_lr, percentiles)
        ax.plot(
            percentiles, imp_q,
            color="#c62828", marker="s", markersize=6,
            linewidth=1.8, alpha=0.95,
            label=f"Impostores (n={imp_lr.size:,})".replace(",", "."),
        )
        # Halo nos pontos críticos: anel vermelho-escuro vazado.
        if imp_critical_mask.any():
            ax.scatter(
                percentiles[imp_critical_mask], imp_q[imp_critical_mask],
                s=160, facecolors="none", edgecolors="#8b0000",
                linewidths=2.2, zorder=5, label="_nolegend_",
            )

    ax.axhline(
        0.0, color="#444", linestyle=":", linewidth=1.0, alpha=0.6,
        label="log-LR = 0  (LR=1)",
    )

    ax.set_xticks(percentiles)
    ax.set_xticklabels([f"P{int(p)}" for p in percentiles])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentil por classe")
    ax.set_ylabel("log-LR calibrado")

    m = getattr(result, "metrics", None)
    cllr_val = getattr(m, "cllr", None) if m else None
    cllr_min_val = getattr(m, "cllr_min", None) if m else None
    line2_parts = []
    if cllr_val is not None:
        line2_parts.append(f"Cllr={cllr_val:.4f}")
    if cllr_min_val is not None:
        line2_parts.append(f"Cllr_min={cllr_min_val:.4f}")
    # Tail gaps assimétricos — pontos mais informativos da região de overlap:
    # - hard-tail gap = Gen P5 − Imp P95
    # - extreme-tail gap = Gen P1 − Imp P99
    # Positivo = boa separação; ≈0 ou negativo = overlap relevante naquela
    # região (genuíno cai abaixo do impostor da mesma "extremidade").
    line3_parts = []
    if gen_q is not None and imp_q is not None:
        idx_p1, idx_p5 = 0, 1  # percentiles[0]=1, [1]=5
        idx_p95, idx_p99 = 7, 8  # percentiles[7]=95, [8]=99
        hard_tail_gap = float(gen_q[idx_p5] - imp_q[idx_p95])
        extreme_tail_gap = float(gen_q[idx_p1] - imp_q[idx_p99])
        line3_parts.append(f"hard-tail (P5g − P95i)={hard_tail_gap:+.2f}")
        line3_parts.append(f"extreme (P1g − P99i)={extreme_tail_gap:+.2f}")
    title_lines = ["Quantis por Classe"]
    if line2_parts:
        title_lines.append("  ·  ".join(line2_parts))
    if line3_parts:
        title_lines.append("  ·  ".join(line3_parts))
    ax.set_title("\n".join(title_lines), fontsize=10)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.85)


_DRAW_FN = {
    "lr_distribution": draw_lr_distribution,
    "tippett":         draw_tippett,
    "det":             draw_det_curve,
    "ece":             draw_ece,
    "curve":           draw_calibration_curve,
    "affine_diag":     draw_affine_diagnostic,
    "quantile_log_lr": draw_quantile_log_lr,
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


def build_metrics_table_rows(
    payload: Dict[str, Any],
) -> "list[tuple[str, str, str, bool]]":
    """Constrói rows da tabela de métricas a partir do schema do sidecar.

    Aceita um dict (schema v1 ou v2 do `_metrics.json` sidecar) OU um
    `dataclasses.asdict()` de `ValidationMetrics`. Campos None viram "—".

    Retorna lista de `(label, value, description, is_section_header)`.
    `description` é uma explicação curta da métrica (1 linha) usada na
    3ª coluna da tabela. Section headers têm `is_section_header=True`
    e `description=""`.

    Útil pra **dois caminhos**: o `ValidationResultWidget` (dialog 🔍 Expandir)
    e o painel inline `_TabbedPngSnapshotWidget` — ambos consomem a mesma
    tabela; o helper garante consistência visual.
    """
    def _get(*keys):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                return v
        return None

    def fmt_f(x, decimals=4):
        return "—" if x is None else f"{float(x):.{decimals}f}"

    def fmt_pct(x, decimals=2):
        return "—" if x is None else f"{float(x)*100:.{decimals}f}%"

    def fmt_thr(x):
        return "—" if x is None else f"{float(x):+.4f}"

    rows: list[tuple[str, str, str, bool]] = []

    def add(label: str, value: str, desc: str = "") -> None:
        rows.append((label, value, desc, False))

    def section(label: str) -> None:
        rows.append((label, "", "", True))

    # ── Tier 1: Discriminação ───────────────────────────────────────────
    section("DISCRIMINAÇÃO")
    add("EER", fmt_pct(_get("eer")),
        "Equal Error Rate — taxa onde FPR=FNR. Mede discriminação bruta independente de calibração.")
    add("EER threshold (log-LR)", fmt_thr(_get("eer_threshold")),
        "Valor de log-LR no ponto de igual erro. Útil para diagnóstico de discriminação, mas NÃO é o limiar decisório ótimo em uso forense — esse depende de custos e priors.")
    add("AUC (ROC)", fmt_f(_get("auc"), 6),
        "Área sob a curva ROC. 1.0 = discriminação perfeita; 0.5 = aleatório.")
    add("Falso+ (LR>0 | Hd)", fmt_pct(_get("wrong_sign_imp_rate")),
        "Fração de impostores com log-LR positivo — evidência favorece Hp apesar de Hd ser verdadeiro.")
    add("Falso− (LR<0 | Hp)", fmt_pct(_get("wrong_sign_gen_rate")),
        "Fração de genuínos com log-LR negativo — evidência favorece Hd apesar de Hp ser verdadeiro.")

    # ── Calibração: custo informacional (principal) ─────────────────────
    cllr = _get("cllr")
    cllr_min = _get("cllr_min")
    cal_loss = _get("cal_loss")
    if cal_loss is None and cllr is not None and cllr_min is not None:
        cal_loss = max(0.0, float(cllr) - float(cllr_min))
    section("CALIBRAÇÃO — CUSTO INFORMACIONAL")
    add("Cllr", fmt_f(cllr),
        "Cost of Log-LR em bits. Mede o custo informacional médio das decisões. Menor é melhor.")
    add("Cllr_min (PAVA)", fmt_f(cllr_min),
        "Cllr ótimo após calibração isotônica (PAVA). Teto teórico — depende só da discriminação.")
    add("cal_loss = Cllr − Cllr_min", fmt_f(cal_loss),
        "Perda de calibração: custo extra por LRs mal escalados. Alto = recalibrar.")
    if cal_loss is not None:
        add("Bem calibrado? (cal_loss < 0.1)",
            "Sim ✅" if cal_loss < 0.1 else "Não ⚠",
            "Heurística forense: cal_loss < 0.1 bits = calibração aceitável pra laudo.")

    # ── Decomposição por hipótese (Tier 2) ──────────────────────────────
    # Os campos `cllr_gen`/`cllr_imp` no DTO são médias por hipótese SEM o
    # fator 0.5 — convenção do `compute_cllr`. Na tabela mostramos como
    # CONTRIBUIÇÕES REAIS (dividindo por 2), que somam ao Cllr total.
    cllr_gen_raw = _get("cllr_gen")
    cllr_imp_raw = _get("cllr_imp")
    cllr_min_gen_raw = _get("cllr_min_gen")
    cllr_min_imp_raw = _get("cllr_min_imp")
    if cllr_gen_raw is not None and cllr_imp_raw is not None:
        # Contribuições reais (somam ao Cllr).
        cllr_gen_c = float(cllr_gen_raw) / 2.0
        cllr_imp_c = float(cllr_imp_raw) / 2.0
        cllr_min_gen_c = float(cllr_min_gen_raw) / 2.0 if cllr_min_gen_raw is not None else None
        cllr_min_imp_c = float(cllr_min_imp_raw) / 2.0 if cllr_min_imp_raw is not None else None
        cal_loss_gen = (
            max(0.0, cllr_gen_c - cllr_min_gen_c)
            if cllr_min_gen_c is not None else None
        )
        cal_loss_imp = (
            max(0.0, cllr_imp_c - cllr_min_imp_c)
            if cllr_min_imp_c is not None else None
        )
        section("CALIBRAÇÃO — DECOMPOSIÇÃO POR HIPÓTESE")
        add("Cllr_gen (contribuição)", fmt_f(cllr_gen_c),
            "Contribuição dos genuínos ao Cllr total. Mostrada com fator 0.5 aplicado — soma com Cllr_imp = Cllr. Alto = LRs de genuínos puxados pro lado negativo (erro custoso).")
        add("Cllr_imp (contribuição)", fmt_f(cllr_imp_c),
            "Contribuição dos impostores ao Cllr total. Mostrada com fator 0.5 aplicado — soma com Cllr_gen = Cllr. Alto = LRs de impostores puxados pro lado positivo (incriminação injusta).")
        add("Cllr_min_gen (contribuição)", fmt_f(cllr_min_gen_c),
            "Contribuição dos genuínos no Cllr_min (PAVA). Teto teórico do termo.")
        add("Cllr_min_imp (contribuição)", fmt_f(cllr_min_imp_c),
            "Contribuição dos impostores no Cllr_min (PAVA). Teto teórico do termo.")
        add("cal_loss_gen", fmt_f(cal_loss_gen),
            "Perda de calibração nos genuínos (contribuição). Alto = sistema subestima evidência a favor da acusação.")
        add("cal_loss_imp", fmt_f(cal_loss_imp),
            "Perda de calibração nos impostores (contribuição). Alto = sistema produz LRs altos pra pares diferentes (risco de incriminação).")

    # ── Recalibração afim (Brümmer + Van Calster/Steyerberg) ────────────
    cllr_affine = _get("cllr_affine")
    slope = _get("calibration_slope")
    intercept = _get("calibration_intercept")
    if cllr_affine is not None:
        affine_cal_loss = (
            max(0.0, float(cllr) - float(cllr_affine))
            if cllr is not None else None
        )
        nonlinear_cal_loss = (
            max(0.0, float(cllr_affine) - float(cllr_min))
            if cllr_min is not None else None
        )
        section("CALIBRAÇÃO — RECALIBRAÇÃO AFIM")
        add("Cllr_affine", fmt_f(cllr_affine),
            "Cllr após recalibração afim ótima dos log-LRs: logLR' = a·logLR + b. Diagnóstico inspirado na calibração logística de log-LRs (Brümmer) e na análise de calibration slope/intercept (Van Calster/Steyerberg).")
        if slope is not None:
            if slope < 0.95:
                diag = " — over-confident: log-LRs precisam encolher pra calibrar."
            elif slope > 1.05:
                diag = " — under-confident: log-LRs precisam crescer pra calibrar."
            else:
                diag = " — afim quase ideal (alvo: a=1)."
            add("Calibration slope (a em a·logLR + b)", f"{slope:.4f}",
                f"Inclinação da reta correta. Alvo a=1. < 1 = log-LRs muito extremos; > 1 = log-LRs muito tímidos.{diag}")
        else:
            add("Calibration slope (a em a·logLR + b)", "—",
                "Inclinação da reta correta. Alvo a=1. < 1 = log-LRs muito extremos; > 1 = log-LRs muito tímidos.")
        add("Calibration intercept (b)",
            f"{intercept:+.4f}" if intercept is not None else "—",
            "Offset da correção afim. Alvo b=0. ≠ 0 = viés sistemático na escala dos log-LRs.")
        add("affine_cal_loss (recuperável)", fmt_f(affine_cal_loss),
            "Perda que sumiria SÓ com recalibração afim (slope+intercept). Alto = recalibrar é fácil.")
        add("nonlinear_cal_loss (residual)", fmt_f(nonlinear_cal_loss),
            "Perda que sobra após afim. Alto = distorção estrutural, só PAVA/isotônica resolve.")

    # ── Curva de Cross-Entropy Empírica por prior (antiga "ECE numérico") ─
    ece_area = _get("ece_calibration_loss_area")
    if ece_area is not None:
        section("CURVA DE CROSS-ENTROPY EMPÍRICA")
        add("Perda total (área)", f"{ece_area:.4f} bits·dec",
            "Área sombreada do plot: integral da diferença entre cross-entropy real e mínima (PAVA) ao longo dos priors log10-odds.")
        add("Gap máximo", fmt_f(_get("ece_max_gap")),
            "Pior diferença pontual entre cross-entropy real e mínima. Em bits — quantos bits de informação são perdidos no pior cenário.")
        argmax_prior = _get("ece_argmax_prior_log_odds")
        add("Prior log₁₀-odds do gap máx",
            "—" if argmax_prior is None else f"{float(argmax_prior):+.3f}",
            "Em qual cenário forense (log10 da prior odds) o sistema é menos calibrado. >0 = priors a favor da acusação.")

    # ── Calibrador — Validação cruzada no domínio de calibração ─────────
    # Mostrado apenas quando CALIBRATION_MODEL_TRAINING rodou com k_folds>0.
    # Cllr CV mede generalização DENTRO do domínio de calibração, NÃO mede
    # performance em validação externa — gaps decompõem efeitos diferentes.
    cllr_cv = _get("calibrator_cllr_cv")
    if cllr_cv is not None:
        cllr_train = _get("calibrator_cllr_train")
        section("CALIBRADOR — VALIDAÇÃO CRUZADA NO DOMÍNIO DE CALIBRAÇÃO")
        add("Cllr in-sample (treino calib)", fmt_f(cllr_train),
            "Cllr do calibrador medido nos mesmos dados usados pra treiná-lo. Otimista — calibrador pode overfittar.")
        add("Cllr CV out-of-fold (treino calib)", fmt_f(cllr_cv),
            "Cllr estimado out-of-fold no domínio de calibração via k-fold. Não mede performance em validação externa.")
        if cllr_train is not None:
            gap_in = float(cllr_cv) - float(cllr_train)
            sign = "+" if gap_in >= 0 else ""
            add("Gap CV − in-sample (overfit in-domain)", f"{sign}{gap_in:.4f}",
                "Quanto o calibrador overfita no próprio treino. > 0 = overfit; ≈ 0 = generaliza bem dentro do domínio.")
        if cllr is not None:
            gap_out = float(cllr) - float(cllr_cv)
            sign = "+" if gap_out >= 0 else ""
            add("Gap validação − CV (perda de transferência)", f"{sign}{gap_out:.4f}",
                "Perda de transferência entre treino do calibrador e validação externa. Valores positivos sugerem diferença de domínio/protocolo/dificuldade entre os conjuntos. NÃO prova isoladamente a causa.")
        cv_gen_raw = _get("calibrator_cllr_cv_gen")
        cv_imp_raw = _get("calibrator_cllr_cv_imp")
        if cv_gen_raw is not None and cv_imp_raw is not None:
            # Mesma convenção da decomposição principal: contribuições somam ao Cllr CV.
            add("Cllr CV gen (contribuição)", fmt_f(float(cv_gen_raw) / 2.0),
                "Contribuição dos genuínos no Cllr CV (fator 0.5 aplicado).")
            add("Cllr CV imp (contribuição)", fmt_f(float(cv_imp_raw) / 2.0),
                "Contribuição dos impostores no Cllr CV (fator 0.5 aplicado).")

    # ── Cauda / outliers ────────────────────────────────────────────────
    if _get("loss_gen_q95") is not None:
        section("CAUDA / OUTLIERS")
        add("Loss q95 genuínos", fmt_f(_get("loss_gen_q95")),
            "Percentil 95 do custo por amostra genuína. Indica magnitude do pior 5% dos genuínos.")
        add("Loss q95 impostores", fmt_f(_get("loss_imp_q95")),
            "Percentil 95 do custo por amostra impostora. Indica magnitude do pior 5% dos impostores.")
        max_gen = _get("max_loss_gen")
        max_imp = _get("max_loss_imp")
        if max_gen is not None:
            add("Loss máx genuínos", fmt_f(max_gen),
                "Pior caso individual no lado genuíno. Em bits. Útil pra identificar samples patológicos.")
        if max_imp is not None:
            add("Loss máx impostores", fmt_f(max_imp),
                "Pior caso individual no lado impostor. Em bits.")
        # Alerta visual quando outliers dominam o Cllr.
        top_1pct = _get("top_1pct_loss_share")
        top_5pct = _get("top_5pct_loss_share")
        n_top_1 = _get("n_top_1pct")
        if top_1pct is not None:
            warn = " ⚠️ Cllr dominado por cauda" if float(top_1pct) > 0.5 else ""
            n_str = f" (N={int(n_top_1)})" if n_top_1 is not None else ""
            add(f"Top 1% loss share{warn}",
                f"{float(top_1pct)*100:.2f}%{n_str}",
                "Fração do Cllr total vinda dos 1% piores casos. > 50% = outliers dominam — robustez nas caudas é o ponto crítico, não a média.")
        if top_5pct is not None:
            add("Top 5% loss share", fmt_pct(top_5pct),
                "Fração do Cllr total vinda dos 5% piores casos. Útil pra contextualizar o top 1%.")

    # ── Contagens ───────────────────────────────────────────────────────
    n_gen = payload.get("n_genuines")
    n_imp = payload.get("n_impostors")
    if n_gen is not None or n_imp is not None:
        section("CONTAGENS")
        add("N genuínos",
            f"{int(n_gen):,}".replace(",", ".") if n_gen is not None else "—",
            "Número de pares mesma-pessoa (Hp) no dataset de validação.")
        add("N impostores",
            f"{int(n_imp):,}".replace(",", ".") if n_imp is not None else "—",
            "Número de pares pessoas-diferentes (Hd) no dataset de validação.")

    return rows


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
    """Grava o sidecar JSON com TODAS as métricas escalares + contagens.

    Schema v2: inclui Tier 1 (discriminação) + Tier 2 (per-hypothesis) +
    Tier 3 (affine Brümmer) + Tier 4 (ECE numérico) + Tier 5 (tail).
    Campos não calculados (None nas métricas) viram None no JSON. Schema
    v1 leitores ignoram campos novos sem quebrar.

    Falha silenciosa retorna False.
    """
    import json
    try:
        metrics = getattr(result, "metrics", None)
        if metrics is None:
            return False

        def _get(attr):
            v = getattr(metrics, attr, None)
            return float(v) if v is not None else None

        cllr = _get("cllr")
        cllr_min = _get("cllr_min")
        cal_loss = (
            max(0.0, cllr - cllr_min)
            if cllr is not None and cllr_min is not None else None
        )

        payload = {
            "schema_version": 2,
            "n_genuines": int(getattr(result, "n_genuines", 0) or 0),
            "n_impostors": int(getattr(result, "n_impostors", 0) or 0),
            # Tier 1: principais
            "eer": _get("eer") or 0.0,
            "eer_threshold": _get("eer_threshold") or 0.0,
            "auc": _get("auc") or 0.0,
            "cllr": cllr,
            "cllr_min": cllr_min,
            "cal_loss": cal_loss,
            "wrong_sign_gen_rate": _get("wrong_sign_gen_rate"),
            "wrong_sign_imp_rate": _get("wrong_sign_imp_rate"),
            # Tier 2: per-hypothesis
            "cllr_gen": _get("cllr_gen"),
            "cllr_imp": _get("cllr_imp"),
            "cllr_min_gen": _get("cllr_min_gen"),
            "cllr_min_imp": _get("cllr_min_imp"),
            # Tier 3: affine (Brümmer)
            "cllr_affine": _get("cllr_affine"),
            "calibration_slope": _get("calibration_slope"),
            "calibration_intercept": _get("calibration_intercept"),
            # Tier 4: ECE numérico
            "ece_calibration_loss_area": _get("ece_calibration_loss_area"),
            "ece_max_gap": _get("ece_max_gap"),
            "ece_argmax_prior_log_odds": _get("ece_argmax_prior_log_odds"),
            # Tier 5: tail / outliers
            "loss_gen_q95": _get("loss_gen_q95"),
            "loss_imp_q95": _get("loss_imp_q95"),
            "max_loss_gen": _get("max_loss_gen"),
            "max_loss_imp": _get("max_loss_imp"),
            "top_1pct_loss_share": _get("top_1pct_loss_share"),
            "top_5pct_loss_share": _get("top_5pct_loss_share"),
            "n_top_1pct": _get("n_top_1pct"),
            # Calibrador — CV diagnóstico (in-domain)
            "calibrator_cllr_train": _get("calibrator_cllr_train"),
            "calibrator_cllr_train_gen": _get("calibrator_cllr_train_gen"),
            "calibrator_cllr_train_imp": _get("calibrator_cllr_train_imp"),
            "calibrator_cllr_cv": _get("calibrator_cllr_cv"),
            "calibrator_cllr_cv_gen": _get("calibrator_cllr_cv_gen"),
            "calibrator_cllr_cv_imp": _get("calibrator_cllr_cv_imp"),
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
    *,
    strategy_name: str = "",
) -> Dict[str, Path]:
    """Paths absolutos dos 4 PNGs snapshot pra um output de STRATEGY_EVALUATION.

    Convenção:
    `<project>/graphics/<strategy_folder>/validation_result/<resource_id>_<plot>.png`
    com `<plot>` ∈ {`lr_distribution`, `tippett`, `det`, `ece`} e
    `strategy_folder` = `<sanitize(strategy_name)>_<strategy_id>`.
    """
    from faceaudit.models.strategy.strategy_folder import (
        sanitize_name, strategy_folder_name,
    )
    folder = (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / strategy_folder_name(strategy_id, strategy_name)
        / "validation_result"
    )
    safe_resource = sanitize_name(resource_id)
    return {
        key: folder / f"{safe_resource}{_PLOT_SUFFIX[key]}.png"
        for key in PLOT_KEYS
    }
