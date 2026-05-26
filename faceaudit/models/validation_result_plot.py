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
PLOT_KEYS = ("lr_distribution", "tippett", "det", "ece", "curve")

# Sufixo de cada plot no nome do arquivo PNG.
_PLOT_SUFFIX = {
    "lr_distribution": "_lr_distribution",
    "tippett":         "_tippett",
    "det":             "_det",
    "ece":             "_ece",
    "curve":           "_calibration_curve",
}

# Título legível pra cada aba (UI inline + dialog).
PLOT_TAB_TITLE = {
    "lr_distribution": "Distribuição de LRs",
    "tippett":         "Tippett",
    "det":             "Curva DET",
    "ece":             "Calibração (ECE)",
    "curve":           "Curva de Calibração",
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
    """Curva de Calibração: scatter raw_score × log-LR colorido por classe.

    Reusa a infraestrutura do `calibration_curve_plot` (mesmo helper já
    usado pra renderizar a aba do SCORE_CALIBRATION). Requer que o
    `result.plot_data` tenha `genuine_raw_scores` e `impostor_raw_scores`
    populados — campos opcionais adicionados em conjunto com a 5ª aba.
    Pickles antigos (sem esses campos) caem em placeholder gracioso.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    pd = result.plot_data
    gen_raw = np.asarray(getattr(pd, "genuine_raw_scores", []), dtype=np.float64)
    imp_raw = np.asarray(getattr(pd, "impostor_raw_scores", []), dtype=np.float64)
    gen_lr = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
    imp_lr = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

    if gen_raw.size == 0 and imp_raw.size == 0:
        ax.text(
            0.5, 0.5,
            "(curva indisponível — pickle pré-curva,\n"
            "re-execute STRATEGY_EVALUATION)",
            ha="center", va="center", transform=ax.transAxes,
            color="#888", fontsize=10,
        )
        return

    # Reusa o renderer canônico de calibration_curve_plot pra manter
    # visual idéntico ao da aba do SCORE_CALIBRATION. Constrói o summary
    # no formato esperado (idêntico ao output de
    # `compute_calibration_curve_summary` para uma lista de samples).
    from faceaudit.models.similarity.calibration_curve_plot import (
        draw_calibration_curve_from_summary,
    )

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
    draw_calibration_curve_from_summary(figure, summary, show_title=True)

    # Anotação: parâmetros do affine fit (slope/intercept = correção ótima)
    m = getattr(result, "metrics", None)
    slope = getattr(m, "calibration_slope", None) if m else None
    intercept = getattr(m, "calibration_intercept", None) if m else None
    cllr_aff = getattr(m, "cllr_affine", None) if m else None
    if slope is not None and intercept is not None:
        ax = figure.axes[0] if figure.axes else None
        if ax is not None:
            if slope < 0.95:
                slope_diag = " (over-confident)"
            elif slope > 1.05:
                slope_diag = " (under-confident)"
            else:
                slope_diag = " (~bem calibrado)"
            lines = [
                f"slope={slope:.3f}{slope_diag}",
                f"intercept={intercept:+.3f}",
            ]
            if cllr_aff is not None:
                lines.append(f"Cllr_affine={cllr_aff:.4f}")
            _add_annotation(ax, "\n".join(lines), loc="upper left")


_DRAW_FN = {
    "lr_distribution": draw_lr_distribution,
    "tippett":         draw_tippett,
    "det":             draw_det_curve,
    "ece":             draw_ece,
    "curve":           draw_calibration_curve,
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
        "Valor de log-LR no ponto de igual erro. Threshold ótimo se aceitar igual taxa de falso+/falso−.")
    add("AUC (ROC)", fmt_f(_get("auc"), 6),
        "Área sob a curva ROC. 1.0 = discriminação perfeita; 0.5 = aleatório.")
    add("Falso+ (LR>0 | Hd)", fmt_pct(_get("wrong_sign_imp_rate")),
        "Fração de impostores com log-LR positivo — sistema diz 'genuíno' pra par de pessoas diferentes.")
    add("Falso− (LR<0 | Hp)", fmt_pct(_get("wrong_sign_gen_rate")),
        "Fração de genuínos com log-LR negativo — sistema diz 'impostor' pra par da mesma pessoa.")

    # ── Calibração: principal ───────────────────────────────────────────
    cllr = _get("cllr")
    cllr_min = _get("cllr_min")
    cal_loss = _get("cal_loss")
    if cal_loss is None and cllr is not None and cllr_min is not None:
        cal_loss = max(0.0, float(cllr) - float(cllr_min))
    section("CALIBRAÇÃO — PRINCIPAL")
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

    # ── Per-hypothesis (Tier 2) ─────────────────────────────────────────
    cllr_gen = _get("cllr_gen")
    cllr_imp = _get("cllr_imp")
    cllr_min_gen = _get("cllr_min_gen")
    cllr_min_imp = _get("cllr_min_imp")
    if cllr_gen is not None and cllr_imp is not None:
        cal_loss_gen = (
            max(0.0, float(cllr_gen) - float(cllr_min_gen))
            if cllr_min_gen is not None else None
        )
        cal_loss_imp = (
            max(0.0, float(cllr_imp) - float(cllr_min_imp))
            if cllr_min_imp is not None else None
        )
        section("CALIBRAÇÃO — POR HIPÓTESE")
        add("Cllr_gen (contrib. genuínos)", fmt_f(cllr_gen),
            "Penaliza falsos negativos. Alto = LRs de genuínos puxados pro lado negativo (erro custoso).")
        add("Cllr_imp (contrib. impostores)", fmt_f(cllr_imp),
            "Penaliza falsos positivos. Alto = LRs de impostores puxados pro lado positivo (incriminação injusta).")
        add("Cllr_min_gen", fmt_f(cllr_min_gen),
            "Cllr_gen após calibração PAVA. Teto teórico do termo dos genuínos.")
        add("Cllr_min_imp", fmt_f(cllr_min_imp),
            "Cllr_imp após calibração PAVA. Teto teórico do termo dos impostores.")
        add("cal_loss_gen", fmt_f(cal_loss_gen),
            "Perda de calibração nos genuínos. Alto = sistema subestima evidência a favor da acusação.")
        add("cal_loss_imp", fmt_f(cal_loss_imp),
            "Perda nos impostores. Alto = sistema produz LRs altos pra pares diferentes (risco de incriminação).")

    # ── Análise affine Brümmer (Tier 3) ─────────────────────────────────
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
        section("CALIBRAÇÃO — ANÁLISE AFFINE (Brümmer)")
        add("Cllr_affine", fmt_f(cllr_affine),
            "Cllr após recalibração linear ótima (slope · LR + intercept). Brümmer-Doddington.")
        if slope is not None:
            if slope < 0.95:
                diag = " — over-confident: LRs precisam encolher pra calibrar."
            elif slope > 1.05:
                diag = " — under-confident: LRs precisam crescer pra calibrar."
            else:
                diag = " — afim quase ideal (~bem calibrado affinely)."
            add("Calibration slope (a em a·x+b)", f"{slope:.4f}",
                f"Inclinação da reta correta. < 1 = LRs muito extremos; > 1 = LRs muito tímidos.{diag}")
        else:
            add("Calibration slope (a em a·x+b)", "—",
                "Inclinação da reta correta. < 1 = LRs muito extremos; > 1 = LRs muito tímidos.")
        add("Calibration intercept (b)",
            f"{intercept:+.4f}" if intercept is not None else "—",
            "Offset da correção linear. ≠ 0 = viés sistemático na escala dos LRs.")
        add("affine_cal_loss (recuperável)", fmt_f(affine_cal_loss),
            "Perda que sumiria SÓ com rescaling linear (slope+intercept). Alto = recalibrar é fácil.")
        add("nonlinear_cal_loss (residual)", fmt_f(nonlinear_cal_loss),
            "Perda que sobra após affine. Alto = distorção estrutural, só PAVA/isotônica resolve.")

    # ── ECE numérico (Tier 4) ───────────────────────────────────────────
    ece_area = _get("ece_calibration_loss_area")
    if ece_area is not None:
        section("CALIBRAÇÃO — ECE NUMÉRICO")
        add("Perda total ECE (área)", f"{ece_area:.4f} bits·dec",
            "Área sombreada do plot ECE: integral da diferença entre ECE_real e ECE_min ao longo dos priors.")
        add("Gap máx ECE", fmt_f(_get("ece_max_gap")),
            "Pior diferença pontual entre ECE_real e ECE_min. Em bits — quantos bits de informação são perdidos no pior cenário.")
        argmax_prior = _get("ece_argmax_prior_log_odds")
        add("Prior log₁₀-odds do gap máx",
            "—" if argmax_prior is None else f"{float(argmax_prior):+.3f}",
            "Em qual cenário forense (log10 da prior odds) o sistema é menos calibrado. >0 = priors a favor da acusação.")

    # ── Calibrador — CV diagnóstico (in-domain) ─────────────────────────
    # Mostrado apenas quando CALIBRATION_MODEL_TRAINING rodou com k_folds>0.
    # Cllr CV mede generalização DENTRO do domínio de calibração, NÃO mede
    # performance em validação externa — gaps decompõem efeitos diferentes.
    cllr_cv = _get("calibrator_cllr_cv")
    if cllr_cv is not None:
        cllr_train = _get("calibrator_cllr_train")
        section("CALIBRADOR — CV DIAGNÓSTICO (treino do calibrador)")
        add("Cllr in-sample (treino calib)", fmt_f(cllr_train),
            "Cllr do calibrador medido nos mesmos dados usados pra treiná-lo. Otimista — calibrador pode overfittar.")
        add("Cllr CV out-of-fold (treino calib)", fmt_f(cllr_cv),
            "Cllr não-viesado no DOMÍNIO de calibração via k-fold. NÃO mede performance em validação externa.")
        if cllr_train is not None:
            gap_in = float(cllr_cv) - float(cllr_train)
            sign = "+" if gap_in >= 0 else ""
            add("Gap CV − in-sample (in-domain overfit)", f"{sign}{gap_in:.4f}",
                "Quanto o calibrador overfita no próprio treino. > 0 = overfit; ≈ 0 = generaliza bem dentro do domínio.")
        if cllr is not None:
            gap_out = float(cllr) - float(cllr_cv)
            sign = "+" if gap_out >= 0 else ""
            add("Gap validação − CV (transfer loss)", f"{sign}{gap_out:.4f}",
                "Domain shift entre treino do calibrador e validação externa. > 0 = validação é pior (típico CFP → BFW).")
        cv_gen = _get("calibrator_cllr_cv_gen")
        cv_imp = _get("calibrator_cllr_cv_imp")
        if cv_gen is not None and cv_imp is not None:
            add("Cllr CV gen", fmt_f(cv_gen),
                "Contribuição dos genuínos no CV. Compara com cllr_train_gen pra ver onde o overfit aparece.")
            add("Cllr CV imp", fmt_f(cv_imp),
                "Contribuição dos impostores no CV.")

    # ── Tail / outliers (Tier 5) ────────────────────────────────────────
    if _get("loss_gen_q95") is not None:
        section("CAUDA / OUTLIERS (diagnóstico)")
        add("Loss q95 genuínos", fmt_f(_get("loss_gen_q95")),
            "Percentil 95 do custo por amostra genuína. Indica magnitude do pior 5% dos genuínos.")
        add("Loss q95 impostores", fmt_f(_get("loss_imp_q95")),
            "Percentil 95 do custo por amostra impostora. Indica magnitude do pior 5% dos impostores.")
        add("Top 1% loss share", fmt_pct(_get("top_1pct_loss_share")),
            "Fração do Cllr total vinda dos 1% piores casos. > 50% = outliers dominam, considere robustez.")

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
            "top_1pct_loss_share": _get("top_1pct_loss_share"),
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
