"""Widget de visualização do resultado UMAP.

Renderiza scatter 2D dos embeddings projetados, com combobox no topo
pra trocar o atributo de coloração sem refazer o fit. Categóricos
usam paleta + legenda; contínuos usam colormap + colorbar; identidade
(alta cardinalidade) usa top-N coloridos + "outros" cinza claro.

Reusa o pattern dos outros plot widgets (hubness/scatter/boxplot) —
FigureCanvas + NavigationToolbar.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


# Quantos valores mais populosos colorimos quando o atributo é
# alta-cardinalidade (identity). Acima disso vai pra "outros" cinza.
_TOP_N_CATEGORIES = 20

# Atributos que devem ser tratados como contínuos (colormap + colorbar)
_CONTINUOUS_ATTRS = frozenset({"yaw", "pitch", "fiqa", "norm"})

# Bins de yaw (mesma estratificação do hubness pra coerência)
_YAW_BINS = [-180, -60, -45, -30, -15, 15, 30, 45, 60, 180]
_YAW_BIN_LABELS = [
    "yaw < -60°", "yaw -60 a -45°", "yaw -45 a -30°", "yaw -30 a -15°",
    "Frontal (|yaw| < 15°)", "yaw 15 a 30°", "yaw 30 a 45°",
    "yaw 45 a 60°", "yaw > 60°",
]


class UmapScatterWidget(QtWidgets.QWidget):
    """Painel pra visualização interativa do UMAP."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self._result = None  # UmapEmbeddingResult; populado em render_result
        self._coords: Optional[np.ndarray] = None  # cache N×2 ou N×3
        self._n_components: int = 2

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        # ── Topo: combobox de coloração ──
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(4, 4, 4, 0)
        top_row.setSpacing(8)
        lbl = QtWidgets.QLabel("<b>Colorir por:</b>")
        lbl.setStyleSheet("color: #1a3d6e; font-size: 12px;")
        top_row.addWidget(lbl)
        self._combo_color = QtWidgets.QComboBox()
        self._combo_color.setMinimumWidth(180)
        self._combo_color.currentIndexChanged.connect(
            self._on_color_attribute_changed
        )
        top_row.addWidget(self._combo_color)
        top_row.addStretch()
        outer.addLayout(top_row)

        # ── Canvas matplotlib ──
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        outer.addWidget(self._toolbar)
        outer.addWidget(self._canvas, 1)

    # ── API pública ──────────────────────────────────────────────────

    def render_result(self, result) -> None:
        """Recebe `UmapEmbeddingResult`, popula combobox + plota
        coloração default."""
        self._result = result
        if result is None or not result.samples:
            self.clear()
            return

        # Cache coords (N×2 ou N×3 conforme n_components) — reuso em
        # cada recolor sem refit do UMAP.
        n = len(result.samples)
        n_comp = int(getattr(result, "n_components", 2)) or 2
        if n_comp not in (2, 3):
            n_comp = 2
        self._n_components = n_comp
        coords = np.empty((n, n_comp), dtype=np.float32)
        for i, s in enumerate(result.samples):
            coords[i, 0] = s.x
            coords[i, 1] = s.y
            if n_comp == 3:
                # `s.z` é Optional[float] no DTO; em modo 3D o service
                # garante que está populado.
                coords[i, 2] = float(s.z if s.z is not None else 0.0)
        self._coords = coords

        # Popula combobox sem disparar render (vamos chamar manualmente)
        self._combo_color.blockSignals(True)
        self._combo_color.clear()
        for attr in result.available_attributes:
            self._combo_color.addItem(self._humanize_attr(attr), attr)
        # Default: identity se disponível, senão primeiro
        default_idx = self._combo_color.findData("identity")
        if default_idx < 0:
            default_idx = 0
        self._combo_color.setCurrentIndex(max(0, default_idx))
        self._combo_color.blockSignals(False)

        # Render inicial
        self._render_plot()

    def clear(self) -> None:
        self._result = None
        self._coords = None
        self._n_components = 2
        self._combo_color.blockSignals(True)
        self._combo_color.clear()
        self._combo_color.blockSignals(False)
        self._figure.clear()
        self._canvas.draw_idle()

    # ── Slots / dispatch ─────────────────────────────────────────────

    def _on_color_attribute_changed(self) -> None:
        """Recolora o scatter sem refit."""
        self._render_plot()

    def _render_plot(self) -> None:
        if self._result is None or self._coords is None:
            self._figure.clear()
            self._canvas.draw_idle()
            return

        attr = self._combo_color.currentData()
        if not attr:
            return

        self._figure.clear()
        # Axes 3D quando o resultado tem 3 componentes — mpl_toolkits
        # registra o projection='3d' globalmente ao importar.
        if self._n_components == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registro)
            ax = self._figure.add_subplot(111, projection="3d")
        else:
            ax = self._figure.add_subplot(111)

        if attr in _CONTINUOUS_ATTRS:
            self._render_continuous(ax, attr)
        elif attr == "yaw_bin":
            self._render_yaw_bins(ax)
        else:
            # Categórico — identity (alta cardinalidade), gender,
            # ethnicity, origin, source.
            self._render_categorical(ax, attr)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        if self._n_components == 3:
            ax.set_zlabel("UMAP 3")
        ax.set_title(
            f"UMAP {self._n_components}D — {self._result.n_samples_used:,} "
            f"pontos · colorido por {self._humanize_attr(attr)}"
        )
        # Grid 3D fica meio carregado; mantemos só em 2D
        if self._n_components == 2:
            ax.grid(alpha=0.2, linestyle="--")
        self._canvas.draw_idle()

    # ── Renderers por tipo de atributo ───────────────────────────────

    def _render_continuous(self, ax, attr: str) -> None:
        """Colormap viridis + colorbar. Pontos sem valor → cinza.
        Funciona em 2D ou 3D — desempacota coords via `*coords.T`."""
        coords = self._coords
        values: List[Optional[float]] = [
            getattr(s, attr) for s in self._result.samples
        ]
        # Separa em dois grupos: com valor e sem valor (NaN)
        has_value_mask = np.array(
            [v is not None for v in values], dtype=bool
        )
        # Sem-valor primeiro (fundo cinza, alpha baixo)
        if not np.all(has_value_mask):
            ax.scatter(
                *coords[~has_value_mask].T,
                c="#cccccc", s=4, alpha=0.3, edgecolors="none",
                label=f"sem {self._humanize_attr(attr)}",
            )
        # Com-valor por cima
        if np.any(has_value_mask):
            arr = np.asarray(
                [v for v, h in zip(values, has_value_mask) if h],
                dtype=np.float32,
            )
            sc = ax.scatter(
                *coords[has_value_mask].T,
                c=arr, s=6, alpha=0.7, cmap="viridis",
                edgecolors="none",
            )
            cb = self._figure.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label(self._humanize_attr(attr), fontsize=10)

    def _render_categorical(self, ax, attr: str) -> None:
        """Categórico: paleta tab20 + legenda. Identity (alta
        cardinalidade) usa top-N coloridos + 'outros' cinza claro."""
        coords = self._coords
        values: List[Optional[str]] = [
            getattr(s, attr) for s in self._result.samples
        ]

        # Conta cardinalidade
        from collections import Counter
        counter = Counter(v for v in values if v)
        unique_count = len(counter)

        # Decide se vai usar top-N ou todos
        use_top_n = (
            attr == "identity" or unique_count > _TOP_N_CATEGORIES
        )
        if use_top_n:
            top_keys = [k for k, _ in counter.most_common(_TOP_N_CATEGORIES)]
            top_set = set(top_keys)
        else:
            top_keys = sorted(counter.keys())
            top_set = set(top_keys)

        # Cores: tab20 cobre 20 categorias com tons distinguíveis
        import matplotlib.cm as cm
        cmap = cm.get_cmap("tab20", max(len(top_keys), 1))
        color_by_value: Dict[str, tuple] = {}
        for i, k in enumerate(top_keys):
            color_by_value[k] = cmap(i)

        # Plota "outros" primeiro (atrás), depois cada top key separadamente
        # (cada um vira uma entrada da legenda).
        others_mask = np.array(
            [(v is None) or (v not in top_set) for v in values],
            dtype=bool,
        )
        if np.any(others_mask):
            label = "(outros / sem valor)" if use_top_n else "(sem valor)"
            ax.scatter(
                *coords[others_mask].T,
                c="#cccccc", s=4, alpha=0.4, edgecolors="none",
                label=label,
            )

        for key in top_keys:
            mask = np.array([v == key for v in values], dtype=bool)
            if not np.any(mask):
                continue
            count = counter.get(key, 0)
            label = f"{key}  ({count:,})" if count else str(key)
            ax.scatter(
                *coords[mask].T,
                c=[color_by_value[key]], s=8, alpha=0.75,
                edgecolors="none",
                label=label,
            )

        # Legenda — fora do plot pra não ocupar área visual quando
        # há muitas categorias (top-20 fica grande).
        if use_top_n:
            ax.legend(
                loc="center left", bbox_to_anchor=(1.01, 0.5),
                fontsize=8, framealpha=0.9,
                title=(
                    f"Top {_TOP_N_CATEGORIES} {self._humanize_attr(attr)}"
                    if attr == "identity" else self._humanize_attr(attr)
                ),
                title_fontsize=9,
            )
        else:
            ax.legend(
                loc="best", fontsize=9, framealpha=0.9,
                title=self._humanize_attr(attr),
            )

    def _render_yaw_bins(self, ax) -> None:
        """Yaw discretizado em faixas (placeholder)."""
        coords = self._coords
        yaws = np.array(
            [s.yaw if s.yaw is not None else np.nan
             for s in self._result.samples],
            dtype=np.float32,
        )
        bin_idx = np.digitize(yaws, _YAW_BINS) - 1
        # NaN cai em bin -1 → mascarado
        for b, label in enumerate(_YAW_BIN_LABELS):
            mask = (bin_idx == b)
            if not np.any(mask):
                continue
            count = int(mask.sum())
            ax.scatter(
                *coords[mask].T,
                s=6, alpha=0.7, edgecolors="none",
                label=f"{label}  ({count:,})",
            )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _humanize_attr(attr: str) -> str:
        """Rótulos amigáveis pro combobox + título do plot."""
        return {
            "identity": "Identidade",
            "gender": "Sexo",
            "ethnicity": "Etnia",
            "origin": "Origem",
            "source": "Source",
            "yaw": "Yaw (°)",
            "pitch": "Pitch (°)",
            "fiqa": "FIQA",
            "norm": "Norma do embedding",
        }.get(attr, attr)
