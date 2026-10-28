"""ImageCompareWidget — preview lado-a-lado original × processada
(× questionada, quando o pipeline tem uma) (PR I2).

Coluna direita do editor. Mostra duas QLabels com pixmaps:
  - Esquerda: imagem original (ou face alinhada, conforme
    `input_source` do pipeline)
  - Direita: imagem processada (após aplicar o pipeline)

Controles:
  - ◀ ▶ — navegar entre faces do dataset
  - 🎲 — escolher aleatória
  - ▶ Aplicar — re-roda o pipeline na imagem corrente
  - Label embaixo: nome do file_name corrente

Sinais pro controller:
  - `prev_clicked`, `next_clicked`, `random_clicked`
  - `apply_clicked`

Setters do controller:
  - `show_original(pixmap)` / `show_processed(pixmap)` — atualiza
    pixmap de cada lado
  - `show_status(text)` — label de filename ou mensagem de erro
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet, current_token_fn


_PREVIEW_MIN = 220  # lado mínimo de cada painel; cresce com a janela


class _ImagePane(QtWidgets.QWidget):
    """Container de uma das duas imagens com título + label."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.lbl_title = QtWidgets.QLabel(title, parent=self)
        self.lbl_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self.lbl_title,
            lambda t: (
                "font-weight: bold; padding: 4px; "
                "background: " + t("SURFACE_ALT") + "; border-radius: 3px;"
            ),
        )
        layout.addWidget(self.lbl_title)

        self.lbl_image = QtWidgets.QLabel("(sem imagem)", parent=self)
        self.lbl_image.setMinimumSize(_PREVIEW_MIN, _PREVIEW_MIN)
        # Ignored: o tamanho do pixmap não puxa o layout — sem isso, cada
        # reescala aumentaria a dica de tamanho e a janela cresceria sozinha.
        self.lbl_image.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored,
        )
        self.lbl_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self.lbl_image,
            lambda t: (
                "background: " + t("SURFACE_RAISED") + "; "
                "border: 1px solid " + t("BORDER") + "; "
                "border-radius: 4px; color: " + t("TEXT_MUTED") + ";"
            ),
        )
        # A imagem leva todo o espaço que sobrar; o título fica no tamanho dele.
        layout.addWidget(self.lbl_image, 1)

    def set_pixmap(self, pixmap: Optional[QtGui.QPixmap], placeholder: str = "(sem imagem)"):
        self._pixmap = pixmap
        if pixmap is None:
            self.lbl_image.clear()
            self.lbl_image.setText(placeholder)
            return
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        lado = self.lbl_image.size()
        # Imagem pequena (ex.: questionada de 168 px) é ampliada para caber,
        # sem suavizar: o grão é justamente o que se quer comparar.
        modo = (
            QtCore.Qt.TransformationMode.FastTransformation
            if self._pixmap.width() < lado.width() and self._pixmap.height() < lado.height()
            else QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_image.setPixmap(self._pixmap.scaled(
            lado, QtCore.Qt.AspectRatioMode.KeepAspectRatio, modo))

    def set_dimensions_text(self, dims: str):
        """Adiciona dimensões ao título (ex: '480×640')."""
        base = self.lbl_title.text().split("(")[0].strip()
        self.lbl_title.setText(f"{base}  ({dims})")


class ImageCompareWidget(QtWidgets.QWidget):
    """Preview lado-a-lado + controles de navegação/aplicação."""

    prev_clicked = QtCore.pyqtSignal()
    next_clicked = QtCore.pyqtSignal()
    random_clicked = QtCore.pyqtSignal()
    apply_clicked = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Linha de navegação
        nav_row = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QToolButton()
        self.btn_prev.setText("◀")
        self.btn_prev.setToolTip("Imagem anterior")
        self.btn_prev.clicked.connect(self.prev_clicked.emit)

        self.btn_next = QtWidgets.QToolButton()
        self.btn_next.setText("▶")
        self.btn_next.setToolTip("Próxima imagem")
        self.btn_next.clicked.connect(self.next_clicked.emit)

        self.btn_random = QtWidgets.QToolButton()
        self.btn_random.setText("🎲")
        self.btn_random.setToolTip("Imagem aleatória")
        self.btn_random.clicked.connect(self.random_clicked.emit)

        for b in (self.btn_prev, self.btn_next, self.btn_random):
            apply_themed_stylesheet(
                b,
                lambda t: (
                    "QToolButton { padding: 4px 10px; border: 1px solid "
                    + t("BORDER") + "; "
                    "border-radius: 3px; background: " + t("SURFACE") + "; } "
                    "QToolButton:hover { background: " + t("PRIMARY_BG") + "; }"
                ),
            )

        self.lbl_index = QtWidgets.QLabel("(sem dataset)")
        apply_themed_stylesheet(
            self.lbl_index,
            lambda t: "color: " + t("TEXT_MUTED") + "; padding: 0 8px;",
        )

        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.btn_next)
        nav_row.addWidget(self.btn_random)
        nav_row.addWidget(self.lbl_index)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        # Painéis lado-a-lado
        panes_row = QtWidgets.QHBoxLayout()
        panes_row.setSpacing(12)

        self.pane_original = _ImagePane("Original", parent=self)
        self.pane_processed = _ImagePane("Processada", parent=self)
        self.pane_questioned = _ImagePane("Questionada", parent=self)
        self.pane_questioned.setVisible(False)

        panes_row.addWidget(self.pane_original)
        panes_row.addWidget(self.pane_processed)
        panes_row.addWidget(self.pane_questioned)
        layout.addLayout(panes_row, 1)

        # Status (filename + erro)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_status,
            lambda t: "color: " + t("TEXT") + "; padding: 4px; font-size: 11px;",
        )
        layout.addWidget(self.lbl_status)

        # Botão Aplicar
        apply_row = QtWidgets.QHBoxLayout()
        self.btn_apply = QtWidgets.QPushButton("▶ Aplicar")
        self.btn_apply.setToolTip(
            "Re-aplica o pipeline atual na imagem mostrada"
        )
        apply_themed_stylesheet(
            self.btn_apply,
            lambda t: (
                "QPushButton { padding: 6px 16px; background: "
                + t("PRIMARY") + "; "
                "color: " + t("HIGHLIGHT_TEXT") + "; font-weight: bold; "
                "border-radius: 4px; } "
                "QPushButton:hover { background: " + t("PRIMARY_STRONG") + "; } "
                "QPushButton:disabled { background: " + t("BORDER") + "; }"
            ),
        )
        self.btn_apply.clicked.connect(self.apply_clicked.emit)
        apply_row.addStretch()
        apply_row.addWidget(self.btn_apply)
        layout.addLayout(apply_row)

    # ── API pra controller ─────────────────────────────────────────────────

    def show_original(
        self,
        pixmap: Optional[QtGui.QPixmap],
        dims: Optional[str] = None,
    ):
        self.pane_original.set_pixmap(pixmap)
        if dims:
            self.pane_original.set_dimensions_text(dims)

    def show_processed(
        self,
        pixmap: Optional[QtGui.QPixmap],
        dims: Optional[str] = None,
    ):
        self.pane_processed.set_pixmap(pixmap)
        if dims:
            self.pane_processed.set_dimensions_text(dims)

    def show_questioned(
        self,
        pixmap: Optional[QtGui.QPixmap],
        *,
        visible: bool,
        dims: Optional[str] = None,
        placeholder: str = "(sem imagem)",
    ):
        """Painel da questionada — só aparece quando o pipeline tem uma."""
        self.pane_questioned.setVisible(visible)
        self.pane_questioned.set_pixmap(pixmap, placeholder=placeholder)
        if dims:
            self.pane_questioned.set_dimensions_text(dims)

    def show_status(self, text: str, is_error: bool = False):
        self.lbl_status.setText(text)
        # QSS montado por estado (erro × normal): lê o token no momento.
        t = current_token_fn()
        color = t("DANGER") if is_error else t("TEXT")
        self.lbl_status.setStyleSheet(
            "color: " + color + "; padding: 4px; font-size: 11px;"
        )

    def show_index(self, idx: int, total: int, file_name: Optional[str]):
        if total == 0:
            self.lbl_index.setText("(sem dataset)")
            return
        suffix = f"  · {file_name}" if file_name else ""
        self.lbl_index.setText(f"{idx + 1} / {total}{suffix}")

    def set_navigation_enabled(self, enabled: bool):
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_random.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
