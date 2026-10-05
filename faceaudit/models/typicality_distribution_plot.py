"""Renderização da distribuição empírica de referência de um modelo de
tipicidade treinado.

Funções puras de matplotlib — sem dependência de Qt. Polimórfico via
`TypicalityModelType`:

- **KNN** (`KNNTypicalityModel`): histograma + KDE da distribuição
  leave-one-out de `T_k` (similaridade média aos k vizinhos), com linha
  do OOD threshold (em escala de `T_k`).
- **MOVMF** (`MoVMFTypicalityModel`): histograma + KDE das
  log-verossimilhanças de treino, com linha do OOD threshold (em escala
  de log-likelihood).

Ambos compartilham o mesmo padrão visual e info-box. Mesma assinatura
de função de saída:

  - `draw_typicality_distribution_from_model(figure, model)` — desenho
    matplotlib puro, reusado pelo `TypicalityDistributionWidget` (UI Qt)
    e pelo renderizador PNG (headless Agg).
  - `render_typicality_distribution_png(model, target_path)` — wrapper
    headless usando `FigureCanvasAgg`, gravado pelo
    `StrategyExecutor.persist_produced_resource` após cada execução de
    `TYPICALITY_MODEL_TRAINING`.
  - `typicality_distribution_png_path(...)` — convenção de path estável,
    usada tanto na escrita (executor) quanto na leitura (painel inline).

Subfolder `typicality_models/` em `<project>/graphics/<strategy_id>/` evita
poluir a pasta com PNGs de outros tipos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder


@dataclass(frozen=True)
class _DistributionSpec:
    """Dados normalizados de distribuição de referência prontos para plot."""

    values: np.ndarray            # vetor 1D float64 ordenado ou não
    ood_threshold: Optional[float]
    score_label: str              # rótulo do eixo X em LaTeX
    series_label: str             # rótulo da série no legend
    info_lines: List[str]         # linhas do info-box (canto superior direito)
    title: str
    # Limites do eixo X. None = autoescalar. KNN usa [-1, +1] como teto
    # natural; MoVMF deixa autoescalar pra log-likelihood.
    x_clip: Optional[Tuple[float, float]] = None


def _build_spec_knn(model: Any) -> Optional[_DistributionSpec]:
    """KNN: distribuição LOO de T_k (similaridade média aos k vizinhos)."""
    loo = getattr(model, "loo_distribution", None)
    if loo is None:
        return None
    loo = np.asarray(loo, dtype=np.float64)
    if loo.size == 0:
        return None

    ood = getattr(model, "ood_threshold", None)
    ood = float(ood) if (ood is not None and np.isfinite(ood)) else None

    k = getattr(model, "k", "?")
    d = getattr(model, "embedding_dim", "?")
    n = int(loo.size)
    med = float(np.median(loo))
    mu = float(np.mean(loo))
    sd = float(np.std(loo))
    info = [
        f"k = {k}",
        f"d = {d}",
        f"n = {n:,}".replace(",", "."),
        f"mediana = {med:.4f}",
        f"média = {mu:.4f}",
        f"std = {sd:.4f}",
    ]
    return _DistributionSpec(
        values=loo,
        ood_threshold=ood,
        score_label=r"$T_k$ — similaridade média aos $k$ vizinhos",
        series_label=f"T_k LOO  (n={n:,})".replace(",", "."),
        info_lines=info,
        title="Distribuição empírica de T_k (LOO)",
        x_clip=(-1.0, 1.0),
    )


def _build_spec_movmf(model: Any) -> Optional[_DistributionSpec]:
    """MoVMF: distribuição de log-verossimilhança das amostras de treino."""
    ll = getattr(model, "reference_log_likelihoods", None)
    if ll is None:
        return None
    ll = np.asarray(ll, dtype=np.float64)
    if ll.size == 0:
        return None
    # filtra não-finitos (não devem aparecer mas se aparecerem matam o KDE)
    finite_mask = np.isfinite(ll)
    if not finite_mask.all():
        ll = ll[finite_mask]
        if ll.size == 0:
            return None

    ood = getattr(model, "ood_threshold", None)
    ood = float(ood) if (ood is not None and np.isfinite(ood)) else None

    # n_components fitted (após eventual drop de componentes mortos)
    n_components = "?"
    params = getattr(model, "_params", None)
    if params is not None:
        n_components = int(getattr(params, "n_components", "?"))

    d = getattr(model, "embedding_dim", "?")
    n = int(ll.size)
    med = float(np.median(ll))
    mu = float(np.mean(ll))
    sd = float(np.std(ll))

    info = [
        f"K = {n_components}",
        f"d = {d}",
        f"n = {n:,}".replace(",", "."),
        f"mediana = {med:.4f}",
        f"média = {mu:.4f}",
        f"std = {sd:.4f}",
    ]
    # AIC/BIC quando disponíveis — diagnóstico paramétrico.
    aic = getattr(model, "aic_", None)
    bic = getattr(model, "bic_", None)
    if aic is not None and np.isfinite(aic):
        info.append(f"AIC = {aic:.2f}")
    if bic is not None and np.isfinite(bic):
        info.append(f"BIC = {bic:.2f}")

    return _DistributionSpec(
        values=ll,
        ood_threshold=ood,
        score_label=r"$\log p(x \mid \hat{\theta})$ — log-verossimilhança",
        series_label=f"log-likelihood treino  (n={n:,})".replace(",", "."),
        info_lines=info,
        title="Distribuição de log-verossimilhança (treino)",
        x_clip=None,
    )


def _build_distribution_spec(model: Any) -> Optional[_DistributionSpec]:
    """Dispatch via `model.model_type`. Fallback duck-typed pra modelos
    legados (k-NN sem `model_type` ainda usa `loo_distribution`)."""
    model_type = getattr(model, "model_type", None)
    type_str = ""
    if model_type is not None:
        type_str = model_type.to_str() if hasattr(model_type, "to_str") else str(model_type)

    if type_str == "MOVMF":
        return _build_spec_movmf(model)
    if type_str == "KNN":
        return _build_spec_knn(model)

    # Fallback duck-typed: priorize loo_distribution (k-NN legado), depois
    # reference_log_likelihoods (MoVMF puro carregado de pickle antigo).
    if hasattr(model, "loo_distribution"):
        return _build_spec_knn(model)
    if hasattr(model, "reference_log_likelihoods"):
        return _build_spec_movmf(model)
    return None


def _draw_placeholder(ax: Axes, msg: str) -> None:
    ax.text(
        0.5, 0.5, msg, transform=ax.transAxes,
        ha="center", va="center", fontsize=12, color="#888",
    )
    ax.axis("off")


def _draw_spec(ax: Axes, spec: _DistributionSpec, *, show_title: bool) -> None:
    values = spec.values
    n = int(values.size)

    x_min = float(np.min(values))
    x_max = float(np.max(values))
    if x_min == x_max:
        x_min -= 0.01
        x_max += 0.01
    span = x_max - x_min
    x_min -= 0.05 * span
    x_max += 0.05 * span

    if spec.x_clip is not None:
        clip_lo, clip_hi = spec.x_clip
        x_min = max(clip_lo, x_min)
        x_max = min(clip_hi, x_max)
        if x_min >= x_max:
            x_min, x_max = float(np.min(values)), float(np.max(values))
            if x_min == x_max:
                x_min -= 0.01
                x_max += 0.01

    n_bins = min(60, max(20, int(np.sqrt(n))))
    bins = np.linspace(x_min, x_max, n_bins + 1)

    ax.hist(
        values, bins=bins, density=True, alpha=0.45,
        color="#1565c0", edgecolor="#0d3a6c", linewidth=0.5,
        label=spec.series_label,
    )

    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(x_min, x_max, 400)
        kde = gaussian_kde(values)
        ax.plot(xs, kde(xs), color="#0d3a6c", linewidth=1.6, alpha=0.95)
    except Exception:
        pass

    if spec.ood_threshold is not None:
        ax.axvline(
            spec.ood_threshold, color="#c62828", linestyle="--", linewidth=1.6,
            alpha=0.85,
            label=f"OOD threshold = {spec.ood_threshold:.4f}",
        )

    if show_title:
        ax.set_title(spec.title, fontsize=11)
    ax.set_xlabel(spec.score_label)
    ax.set_ylabel("Densidade de frequência")
    ax.grid(True, linestyle="--", alpha=0.4)

    info_text = "\n".join(spec.info_lines)
    ax.text(
        0.985, 0.985, info_text, transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#bbb", lw=0.7),
    )

    ax.legend(loc="upper left", fontsize=9)


def draw_typicality_distribution_from_model(
    figure: Figure,
    model: Any,
    *,
    show_title: bool = True,
) -> None:
    """Histograma + KDE da distribuição de referência + linha OOD threshold.

    Polimórfico via `model.model_type`:
    - KNN  → distribuição leave-one-out de `T_k`.
    - MOVMF → distribuição de log-verossimilhança das amostras de treino.

    O modelo precisa estar treinado (`is_fitted()`). Se não estiver, ou
    se a distribuição de referência estiver vazia, desenha placeholder.
    """
    figure.clear()
    ax = figure.add_subplot(111)

    if model is None or not getattr(model, "is_fitted", lambda: False)():
        _draw_placeholder(ax, "(modelo não treinado)")
        return

    spec = _build_distribution_spec(model)
    if spec is None:
        _draw_placeholder(ax, "(sem distribuição de referência disponível)")
        return

    _draw_spec(ax, spec, show_title=show_title)


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
    *,
    strategy_name: str = "",
) -> Path:
    """Path absoluto e determinístico do PNG do modelo de tipicidade.

    Convenção:
    `<project>/graphics/<strategy_folder>/typicality_models/<resource_id>_dist.png`,
    com `strategy_folder` = `<sanitize(strategy_name)>_<strategy_id>`.
    Subfolder dedicado evita mistura com PNGs de outros tipos.
    """
    from faceaudit.models.strategy.strategy_folder import (
        sanitize_name, strategy_folder_name,
    )
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / strategy_folder_name(strategy_id, strategy_name)
        / "typicality_models"
        / f"{sanitize_name(resource_id)}_dist.png"
    )


def _sanitize_name(raw: str) -> str:
    """Normaliza identificador pra uso em nome de arquivo. Mesmo helper que
    `distribution_plot._sanitize_name` — duplicado pra não acoplar módulos."""
    bad = '\\/:*?"<>| '
    out = raw or "unknown"
    for ch in bad:
        out = out.replace(ch, "_")
    return out
