"""Widget matplotlib pra plotar histograma de uma métrica de qualidade.

Consumido pela `DatasetAnalysisView` quando o usuário seleciona uma
auditoria do tipo `histogram_*` na árvore. Aceita um vetor de valores
+ rótulo da métrica e renderiza histograma com KDE leve por cima
(usando numpy puro pra evitar dependência de scipy só pra isso).

Quando `image_paths` é fornecido (paralelo a `values`), exibe duas
faixas de thumbnails abaixo do gráfico — top-10 cauda esquerda e
top-10 cauda direita — pra inspeção visual rápida das faces extremas.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from interface.oriented_image import ler_pixmap_orientado


_TAIL_SIZE = 10
_THUMB_PX = 80


class _TailThumb(QtWidgets.QFrame):
    """Thumb fixo com imagem em cima e legenda embaixo. Sem seleção
    — é só inspeção visual da cauda do histograma. `caption` opcional
    sobrescreve o formato default (`{value:.3f}`); usado pelo audit
    de balanceamento que mostra `<identidade>\\nn=<count>`."""

    def __init__(
        self,
        image_path: str,
        value: float,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        caption: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #fafafa; border: 1px solid #ddd; }"
        )
        self.setFixedWidth(_THUMB_PX + 8)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(2)

        self._img_lbl = QtWidgets.QLabel()
        self._img_lbl.setFixedSize(_THUMB_PX, _THUMB_PX)
        self._img_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background: #eee; color: #999;")
        self._load_thumb(image_path)
        v.addWidget(self._img_lbl)

        cap_text = caption if caption is not None else f"{value:.3f}"
        cap = QtWidgets.QLabel(cap_text)
        cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("font-size: 10px; color: #333; border: none;")
        cap.setWordWrap(True)
        v.addWidget(cap)

        # Tooltip mostra o path completo + valor (sempre numérico, mesmo
        # com caption customizada — é a fonte da verdade pra debug).
        self.setToolTip(f"{image_path}\nvalor: {value:.4f}")

    def _load_thumb(self, image_path: str) -> None:
        """Carrega e escala a imagem. Em falha (path inexistente,
        formato inválido), mostra placeholder textual."""
        if not image_path:
            self._img_lbl.setText("(sem path)")
            return
        pix = ler_pixmap_orientado(image_path)
        if pix.isNull():
            self._img_lbl.setText("(não\nencontrada)")
            return
        scaled = pix.scaled(
            _THUMB_PX, _THUMB_PX,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._img_lbl.setPixmap(scaled)


class HistogramWidget(QtWidgets.QWidget):
    """Histograma simples com toolbar de navegação. Reusa a Figure entre
    renders pra evitar piscadas e leak de canvases ao trocar finding."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self._figure = Figure(figsize=(6, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        # Dataset corrente — usado pra resolver paths em modo RELATIVE
        # antes de carregar thumbnails das caudas.
        self._dataset: Optional[Any] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

        # Faixas de thumbs das caudas — aparecem só quando `image_paths`
        # é passado pra `render_distribution`.
        self._tails_widget = self._build_tails_widget()
        self._tails_widget.hide()
        layout.addWidget(self._tails_widget)

    def set_dataset(self, dataset: Optional[Any]) -> None:
        """Recebe dataset corrente pra resolver paths em modo RELATIVE."""
        self._dataset = dataset

    def _build_tails_widget(self) -> QtWidgets.QWidget:
        """Constrói o container das duas faixas (esquerda/direita) com
        labels e scroll horizontal — tela menor que 900px ainda permite
        ver os 10 thumbs via scroll."""
        outer = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(outer)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(4)

        self._lbl_left = QtWidgets.QLabel(
            "<b>Cauda esquerda</b> — 10 menores valores"
        )
        self._lbl_left.setStyleSheet("color: #1a3d6e; font-size: 11px;")
        v.addWidget(self._lbl_left)

        self._left_strip = self._make_strip()
        v.addWidget(self._left_strip["scroll"])

        self._lbl_right = QtWidgets.QLabel(
            "<b>Cauda direita</b> — 10 maiores valores"
        )
        self._lbl_right.setStyleSheet("color: #1a3d6e; font-size: 11px;")
        v.addWidget(self._lbl_right)

        self._right_strip = self._make_strip()
        v.addWidget(self._right_strip["scroll"])

        return outer

    @staticmethod
    def _make_strip() -> dict:
        """Cria um QScrollArea horizontal contendo um QWidget+HBox.
        Retorna dict com 'scroll' (widget pra adicionar ao layout) e
        'layout' (HBox pra adicionar/remover thumbs)."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addStretch()

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        scroll.setFixedHeight(_THUMB_PX + 36)
        return {"scroll": scroll, "layout": layout, "container": container}

    @staticmethod
    def _populate_strip(
        strip: dict,
        paths: List[str],
        values: List[float],
        captions: Optional[List[str]] = None,
    ) -> None:
        """Limpa a faixa e popula com novos thumbs. `paths`/`values`/
        `captions` têm a mesma cardinalidade. `captions=None` usa
        o default (formato numérico do valor)."""
        layout = strip["layout"]
        # Remove tudo (incluindo o stretch trailing). takeAt(0) preserva
        # ordem e evita iterar count() que muda durante a remoção.
        while layout.count() > 0:
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # Re-popula
        for i, (path, value) in enumerate(zip(paths, values)):
            cap = captions[i] if captions is not None else None
            layout.addWidget(_TailThumb(path, value, caption=cap))
        layout.addStretch()

    def render_distribution(
        self,
        values: Sequence[float],
        metric_label: str,
        *,
        bins: int = 50,
        image_paths: Optional[Sequence[str]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> None:
        """Plota o histograma de `values`. Limpa o axes anterior.

        Quando `image_paths` é fornecido (mesmo length de `values`),
        mostra duas faixas de thumbs embaixo: as 10 imagens com menor
        valor e as 10 com maior. Senão, esconde a seção.

        `captions` opcional, mesma cardinalidade de `values` —
        sobrescreve a legenda numérica padrão dos thumbs (usado pelo
        audit de balanceamento, onde a legenda traz `identidade + n=X`).
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not values:
            ax.text(
                0.5, 0.5, "Sem valores para plotar.",
                ha="center", va="center", transform=ax.transAxes,
                color="#666",
            )
            self._tails_widget.hide()
            self._canvas.draw_idle()
            return

        arr = np.asarray(list(values), dtype=float)
        # Bins automáticos quando o usuário não pede explícito — heurística
        # leve pra ter resolução sem virar ruído.
        n = arr.size
        n_bins = min(bins, max(10, int(np.sqrt(n))))

        ax.hist(
            arr, bins=n_bins, color="#1976d2", edgecolor="#0d47a1",
            alpha=0.78,
        )
        ax.set_xlabel(metric_label)
        ax.set_ylabel("Contagem (faces)")
        ax.set_title(f"Distribuição — {metric_label}  (n={n:,})")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Linhas verticais de média e mediana — referência rápida
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        ax.axvline(
            mean, color="#c62828", linestyle="-", linewidth=1.4,
            label=f"média = {mean:.3f}",
        )
        ax.axvline(
            median, color="#2e7d32", linestyle="--", linewidth=1.4,
            label=f"mediana = {median:.3f}",
        )
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        self._canvas.draw_idle()

        # Caudas — só quando paths foram fornecidos
        if image_paths is None or len(image_paths) != len(values):
            self._tails_widget.hide()
            return
        captions_list = (
            list(captions) if captions is not None
            and len(captions) == len(values) else None
        )
        self._populate_tails(arr, list(image_paths), captions_list)
        self._tails_widget.show()

    def _populate_tails(
        self,
        values: np.ndarray,
        paths: List[str],
        captions: Optional[List[str]],
    ) -> None:
        """Ordena por valor e popula as duas faixas com top-N de cada
        cauda. Usa np.argsort estável pra manter ordem em empates."""
        n = values.size
        if n == 0:
            self._tails_widget.hide()
            return
        k = min(_TAIL_SIZE, n)
        # argsort retorna índices em ordem crescente
        order = np.argsort(values, kind="stable")
        left_idx = order[:k]
        right_idx = order[-k:][::-1]  # maiores primeiro

        # Resolve paths via dataset em modo RELATIVE (Image Pipeline).
        # _TailThumb consome path direto pra QPixmap, então o resolve
        # acontece aqui antes de criar os widgets.
        def _resolve(p: str) -> str:
            if not p or self._dataset is None:
                return p
            return str(self._dataset.resolve_image_path(p))

        left_paths = [_resolve(paths[i]) for i in left_idx]
        left_vals = [float(values[i]) for i in left_idx]
        right_paths = [_resolve(paths[i]) for i in right_idx]
        right_vals = [float(values[i]) for i in right_idx]

        left_captions = (
            [captions[i] for i in left_idx] if captions is not None else None
        )
        right_captions = (
            [captions[i] for i in right_idx] if captions is not None else None
        )

        # Atualiza labels com count efetivo (caso n < 10)
        self._lbl_left.setText(
            f"<b>Cauda esquerda</b> — {k} menores valor(es)"
        )
        self._lbl_right.setText(
            f"<b>Cauda direita</b> — {k} maiores valor(es)"
        )

        self._populate_strip(
            self._left_strip, left_paths, left_vals, left_captions,
        )
        self._populate_strip(
            self._right_strip, right_paths, right_vals, right_captions,
        )

    def clear(self) -> None:
        """Limpa o canvas — chamado quando nenhuma auditoria está
        selecionada ou ao trocar pra um finding não-histograma."""
        self._figure.clear()
        self._canvas.draw_idle()
        self._tails_widget.hide()
