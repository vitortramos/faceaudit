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
from PyQt6 import QtCore, QtGui, QtWidgets

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
        # Modo runtime — em "centroids", click no scatter aciona a
        # busca de lookalikes via cosseno em 512D.
        self._is_centroids_mode: bool = False
        # Cache da matriz K×D dos centroides L2-norm (só em centroids
        # mode) pra computar cosseno on-click sem recalcular.
        self._embeddings_matrix: Optional[np.ndarray] = None
        # Highlight do ponto selecionado pelo click — guarda o
        # collection do scatter pra remover/recriar a cada click.
        self._highlight_artist = None
        # Marker do lookalike selecionado na tabela — círculo azul,
        # distinto da estrela amarela do clicked.
        self._lookalike_marker_artist = None

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
        # Hint sobre click — só visível em modo centroides
        self._lbl_click_hint = QtWidgets.QLabel(
            "<i>Clique numa identidade pra ver lookalikes →</i>"
        )
        self._lbl_click_hint.setStyleSheet(
            "color: #c62828; font-size: 11px;"
        )
        self._lbl_click_hint.hide()
        top_row.addWidget(self._lbl_click_hint)
        top_row.addStretch()
        outer.addLayout(top_row)

        # ── Splitter horizontal: scatter à esquerda, lookalikes à direita ──
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # Painel esquerdo — toolbar + canvas matplotlib
        left_pane = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(2)
        self._figure = Figure(figsize=(8, 6), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        left_layout.addWidget(self._toolbar)
        left_layout.addWidget(self._canvas, 1)
        splitter.addWidget(left_pane)

        # Painel direito — lookalikes (escondido por default)
        self._lookalikes_pane = self._build_lookalikes_pane()
        self._lookalikes_pane.hide()
        splitter.addWidget(self._lookalikes_pane)
        # Tamanho default — scatter ocupa ~70%, lookalikes ~30%
        splitter.setSizes([700, 300])

        # Wire matplotlib click → handler. Sempre conectado, mas o
        # handler só age quando _is_centroids_mode == True.
        self._canvas.mpl_connect("button_press_event", self._on_canvas_click)

    def _build_lookalikes_pane(self) -> QtWidgets.QWidget:
        """Painel lateral com tabela de identidades mais próximas."""
        pane = QtWidgets.QFrame()
        pane.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(pane)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._lookalikes_header = QtWidgets.QLabel(
            "<b>Lookalikes</b>"
        )
        self._lookalikes_header.setStyleSheet(
            "font-size: 13px; color: #1a3d6e; padding: 4px 0;"
        )
        self._lookalikes_header.setWordWrap(True)
        layout.addWidget(self._lookalikes_header)

        self._lookalikes_table = QtWidgets.QTableWidget()
        self._lookalikes_table.setColumnCount(3)
        self._lookalikes_table.setHorizontalHeaderLabels(
            ["Identidade", "Cosseno", "Nº fotos"]
        )
        self._lookalikes_table.verticalHeader().setVisible(False)
        self._lookalikes_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._lookalikes_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._lookalikes_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        h = self._lookalikes_table.horizontalHeader()
        h.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        h.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        h.setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        # Click numa linha → highlight do ponto correspondente no scatter
        self._lookalikes_table.itemSelectionChanged.connect(
            self._on_lookalike_row_selected
        )
        layout.addWidget(self._lookalikes_table, 1)

        # Thumb da identidade selecionada na tabela
        self._lookalike_thumb = QtWidgets.QLabel()
        self._lookalike_thumb.setFixedSize(160, 160)
        self._lookalike_thumb.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        self._lookalike_thumb.setStyleSheet(
            "background: #fafafa; border: 1px solid #ddd;"
        )
        layout.addWidget(self._lookalike_thumb, 0,
                         QtCore.Qt.AlignmentFlag.AlignCenter)

        return pane

    # ── API pública ──────────────────────────────────────────────────

    def render_result(self, result) -> None:
        """Recebe `UmapEmbeddingResult`, popula combobox + plota
        coloração default. Se for modo centroides, mostra o painel
        lateral de lookalikes e cacheia a matriz 512D pra cosseno
        on-click."""
        self._result = result
        if result is None or not result.samples:
            self.clear()
            return

        # Detecta modo + cacheia matriz 512D dos centroides (só em
        # centroids mode; em faces mode fica None).
        cfg_mode = getattr(result.config, "mode", "faces")
        self._is_centroids_mode = (cfg_mode == "centroids")
        self._embeddings_matrix = (
            getattr(result, "embeddings_matrix", None)
            if self._is_centroids_mode else None
        )

        # Mostra/esconde painel lateral + hint conforme o modo
        if self._is_centroids_mode and self._embeddings_matrix is not None:
            self._lookalikes_pane.show()
            self._lbl_click_hint.show()
            self._reset_lookalikes_panel()
        else:
            self._lookalikes_pane.hide()
            self._lbl_click_hint.hide()

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

        # Popula combobox sem disparar render (vamos chamar manualmente).
        # Em modo centroides, exclui "identity" — cada ponto JÁ é uma
        # identidade única, então colorir por isso seria 49k cores
        # (ou top-20 aleatório com legenda inútil ocupando espaço).
        attrs = [
            a for a in result.available_attributes
            if not (self._is_centroids_mode and a == "identity")
        ]
        self._combo_color.blockSignals(True)
        self._combo_color.clear()
        for attr in attrs:
            self._combo_color.addItem(self._humanize_attr(attr), attr)
        # Default: identity (se modo faces) ou primeiro disponível
        default_idx = self._combo_color.findData("identity")
        if default_idx < 0:
            default_idx = 0
        self._combo_color.setCurrentIndex(max(0, default_idx))
        self._combo_color.blockSignals(False)

        # Render inicial
        self._highlight_artist = None
        self._render_plot()

    def clear(self) -> None:
        self._result = None
        self._coords = None
        self._n_components = 2
        self._is_centroids_mode = False
        self._embeddings_matrix = None
        self._highlight_artist = None
        self._lookalike_marker_artist = None
        self._combo_color.blockSignals(True)
        self._combo_color.clear()
        self._combo_color.blockSignals(False)
        self._figure.clear()
        self._canvas.draw_idle()
        self._lookalikes_pane.hide()
        self._lbl_click_hint.hide()
        self._reset_lookalikes_panel()

    # ── Slots / dispatch ─────────────────────────────────────────────

    def _on_color_attribute_changed(self) -> None:
        """Recolora o scatter sem refit. Limpa markers (estrela
        amarela do clicked + círculo azul do lookalike selecionado)
        — ficariam órfãos num scatter recolorido."""
        self._highlight_artist = None
        self._lookalike_marker_artist = None
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

    # ── Click handler + lookalikes (modo centroids) ──────────────────

    # Quantos lookalikes mostrar por click. Top-K é leve mesmo com K
    # = 49k identidades (1 multiplicação K×D + sort).
    _LOOKALIKES_TOP_K = 15

    def _on_canvas_click(self, event) -> None:
        """Handler do click no canvas matplotlib. Em modo centroides,
        encontra a identidade mais próxima do click no espaço UMAP e
        popula a tabela de lookalikes (cosseno em 512D)."""
        if not self._is_centroids_mode:
            return
        if self._coords is None or self._embeddings_matrix is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        # Em 3D, mpl emite coords 2D (projeção da câmera) — o nearest
        # search funciona razoavelmente bem mesmo assim, mas pode ser
        # impreciso por causa do parallax. Aceito por enquanto.
        target = np.array(
            [event.xdata, event.ydata],
            dtype=np.float32,
        )
        # Busca o ponto mais próximo do clique nas coords 2D primárias
        dx = self._coords[:, 0] - target[0]
        dy = self._coords[:, 1] - target[1]
        d2 = dx * dx + dy * dy
        clicked_idx = int(np.argmin(d2))

        # Tolerância: descarta clique muito longe de qualquer ponto
        # (~5% do range médio dos eixos). NumPy 2.x removeu o método
        # `ndarray.ptp()`; usa a função `np.ptp(arr)` que segue válida.
        x_range = float(np.ptp(self._coords[:, 0]))
        y_range = float(np.ptp(self._coords[:, 1]))
        max_d = max(x_range, y_range) * 0.05
        if max_d > 0 and d2[clicked_idx] > max_d * max_d:
            return

        self._show_lookalikes(clicked_idx)

    def _show_lookalikes(self, clicked_idx: int) -> None:
        """Computa cosseno em 512D entre o centroide clicado e todos
        os outros, popula a tabela de lookalikes ordenada por cosseno
        desc, e desenha um marker no scatter."""
        if self._embeddings_matrix is None or self._result is None:
            return

        # Limpa marker do lookalike anterior — vai ser substituído
        # quando a nova primeira linha for auto-selecionada.
        self._clear_lookalike_marker()

        clicked = self._embeddings_matrix[clicked_idx]
        # Cosseno = produto interno entre vetores L2-norm. Como o
        # service já normalizou todos os centroides, é só matmul.
        sims = self._embeddings_matrix @ clicked
        # Excluir o próprio (cosseno=1.0); usa -inf pra mantê-lo fora
        # do top-K via argpartition.
        sims = sims.copy()
        sims[clicked_idx] = -np.inf
        K = min(self._LOOKALIKES_TOP_K, sims.size - 1)
        top_idx = np.argpartition(-sims, K)[:K]
        # Ordena por cosseno desc dentro do top-K
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        # Header com a identidade clicada
        clicked_sample = self._result.samples[clicked_idx]
        clicked_name = clicked_sample.identity or "?"
        n_faces = clicked_sample.n_faces_in_identity or "?"
        self._lookalikes_header.setText(
            f"<b>Lookalikes de:</b><br>"
            f"<span style='color: #c62828;'>{clicked_name}</span> "
            f"<i>({n_faces} fotos)</i>"
        )

        # Popula a tabela
        self._populate_lookalikes_table(top_idx, sims)

        # Highlight do clicked no scatter (marker amarelo grande)
        self._draw_highlight(clicked_idx)

    def _populate_lookalikes_table(
        self, top_idx: np.ndarray, sims: np.ndarray,
    ) -> None:
        """Preenche a tabela com top-K lookalikes."""
        self._lookalikes_table.setUpdatesEnabled(False)
        self._lookalikes_table.blockSignals(True)
        try:
            self._lookalikes_table.setRowCount(len(top_idx))
            for row, idx in enumerate(top_idx):
                idx = int(idx)
                sample = self._result.samples[idx]
                ident = sample.identity or "?"
                cos = float(sims[idx])
                n_faces = sample.n_faces_in_identity or 0

                cell0 = QtWidgets.QTableWidgetItem(ident)
                cell0.setData(
                    QtCore.Qt.ItemDataRole.UserRole, idx,
                )
                cell0.setData(
                    QtCore.Qt.ItemDataRole.UserRole + 1,
                    sample.image_path or "",
                )
                self._lookalikes_table.setItem(row, 0, cell0)

                cell1 = QtWidgets.QTableWidgetItem(f"{cos:.3f}")
                cell1.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                self._lookalikes_table.setItem(row, 1, cell1)

                cell2 = QtWidgets.QTableWidgetItem(f"{n_faces:,}")
                cell2.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                self._lookalikes_table.setItem(row, 2, cell2)

            # Pré-seleciona a primeira linha pra mostrar o thumb logo
            if len(top_idx) > 0:
                self._lookalikes_table.selectRow(0)
        finally:
            self._lookalikes_table.setUpdatesEnabled(True)
            self._lookalikes_table.blockSignals(False)
            # Trigger manual do _on_lookalike_row_selected pra carregar
            # o thumb da primeira linha (blockSignals impediu o auto)
            if len(top_idx) > 0:
                self._on_lookalike_row_selected()

    def _on_lookalike_row_selected(self) -> None:
        """Quando o usuário clica numa linha da tabela de lookalikes,
        carrega o thumb da identidade correspondente E desenha um
        círculo azul no ponto correspondente do scatter."""
        rows = sorted({i.row() for i in self._lookalikes_table.selectedIndexes()})
        if not rows:
            self._lookalike_thumb.clear()
            self._clear_lookalike_marker()
            return
        cell = self._lookalikes_table.item(rows[0], 0)
        if cell is None:
            return

        # Marker no scatter — círculo azul no ponto da identidade
        sample_idx = cell.data(QtCore.Qt.ItemDataRole.UserRole)
        if sample_idx is not None:
            self._draw_lookalike_marker(int(sample_idx))

        # Thumb da face representante
        path = cell.data(QtCore.Qt.ItemDataRole.UserRole + 1) or ""
        if not path:
            self._lookalike_thumb.clear()
            self._lookalike_thumb.setText("(sem path)")
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self._lookalike_thumb.clear()
            self._lookalike_thumb.setText("(não encontrada)")
            return
        scaled = pix.scaled(
            self._lookalike_thumb.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._lookalike_thumb.setPixmap(scaled)

    def _draw_highlight(self, idx: int) -> None:
        """Marca o ponto clicado com um marker grande amarelo. Remove
        o highlight anterior se houver."""
        ax = (
            self._figure.axes[0] if self._figure.axes else None
        )
        if ax is None:
            return
        # Remove highlight anterior
        if self._highlight_artist is not None:
            try:
                self._highlight_artist.remove()
            except Exception:
                pass
            self._highlight_artist = None

        coord = self._coords[idx]
        if self._n_components == 3:
            self._highlight_artist = ax.scatter(
                [coord[0]], [coord[1]], [coord[2]],
                s=180, c="#ffeb3b", marker="*",
                edgecolors="#000", linewidths=1.5, zorder=10,
            )
        else:
            self._highlight_artist = ax.scatter(
                [coord[0]], [coord[1]],
                s=180, c="#ffeb3b", marker="*",
                edgecolors="#000", linewidths=1.5, zorder=10,
            )
        self._canvas.draw_idle()

    def _draw_lookalike_marker(self, idx: int) -> None:
        """Desenha um círculo AZUL no ponto da identidade selecionada
        na tabela de lookalikes. Distinto da estrela amarela do
        clicked (origem). Substitui o anterior se houver."""
        ax = (
            self._figure.axes[0] if self._figure.axes else None
        )
        if ax is None:
            return
        self._clear_lookalike_marker()

        coord = self._coords[idx]
        if self._n_components == 3:
            self._lookalike_marker_artist = ax.scatter(
                [coord[0]], [coord[1]], [coord[2]],
                s=160, facecolors="none", edgecolors="#1976d2",
                linewidths=2.5, marker="o", zorder=9,
            )
        else:
            self._lookalike_marker_artist = ax.scatter(
                [coord[0]], [coord[1]],
                s=160, facecolors="none", edgecolors="#1976d2",
                linewidths=2.5, marker="o", zorder=9,
            )
        self._canvas.draw_idle()

    def _clear_lookalike_marker(self) -> None:
        """Remove o círculo do lookalike selecionado (sem redraw —
        caller decide se precisa)."""
        if self._lookalike_marker_artist is not None:
            try:
                self._lookalike_marker_artist.remove()
            except Exception:
                pass
            self._lookalike_marker_artist = None

    def _reset_lookalikes_panel(self) -> None:
        """Limpa header + tabela + thumb. Chamado em clear() e em
        cada novo render_result."""
        self._lookalikes_header.setText("<b>Lookalikes</b>")
        self._lookalikes_table.setRowCount(0)
        self._lookalike_thumb.clear()

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
