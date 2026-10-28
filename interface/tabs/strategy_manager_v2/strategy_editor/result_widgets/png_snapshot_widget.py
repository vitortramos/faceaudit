# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/png_snapshot_widget.py

"""
Widget compartilhado de snapshot PNG escalado mantendo aspect ratio.

Usado em dois contextos:
- 3ª coluna do editor de estratégias (`InstructionResultsPanel`),
  pra exibir o PNG gravado pelo executor durante a persistência do recurso.
- Janela de Comparação de Estratégias, pelas views per-estratégia
  (calibration distribution, casework results, validation lr_distribution)
  — substitui matplotlib widgets pesados que carregam pickles inteiros.

Genérico — não conhece nada sobre o tipo do plot que está exibindo.
Recebe um path e carrega/escala/pinta o PNG.

Quando o arquivo PNG está ausente (recurso ainda não executado ou
invalidado), mostra placeholder discreto.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PyQt6 import QtCore, QtGui, QtWidgets
from interface.oriented_image import ler_pixmap_orientado


class PngSnapshotWidget(QtWidgets.QWidget):
    """Exibe um PNG escalado mantendo aspect ratio.

    Pintamos via `paintEvent` em vez de usar `QLabel.setPixmap` pra
    evitar feedback loop: `setPixmap` altera o `sizeHint` do label, o
    layout recalcula, o widget cresce, `resizeEvent` dispara novo
    scale, sizeHint cresce de novo, etc. Sem QLabel a renderização
    desacopla do sizing.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        # Largura: expande pra preencher a coluna. Altura: derivada do
        # aspect ratio do pixmap via heightForWidth.
        policy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setMinimumSize(1, 1)

        # Pixmap original em resolução cheia. paintEvent escala on-demand.
        # Sem manter o original, resizes sucessivos acumulariam perda
        # de qualidade.
        self._full_pixmap: Optional[QtGui.QPixmap] = None
        self._placeholder_text: str = "Execute para gerar visualização"

    def hasHeightForWidth(self) -> bool:  # noqa: N802 — Qt API
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 — Qt API
        """Altura pedida pra uma dada largura — derivada do aspect ratio
        do PNG carregado. Permite que o layout dê espaço vertical
        suficiente pra exibir a imagem cheia sem padding lateral."""
        if self._full_pixmap is None or self._full_pixmap.width() <= 0:
            return 200  # placeholder default
        aspect_h_w = self._full_pixmap.height() / self._full_pixmap.width()
        return max(120, int(width * aspect_h_w))

    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 — Qt API
        # Largura intencionalmente pequena pra deixar o layout decidir
        # (Expanding); altura derivada do aspect quando há imagem.
        w = 320
        return QtCore.QSize(w, self.heightForWidth(w))

    def show_placeholder(self, text: str) -> None:
        self._full_pixmap = None
        self._placeholder_text = text or ""
        self.updateGeometry()
        self.update()

    def show_image(self, png_path: Union[str, Path]) -> bool:
        """Carrega o PNG do disco. Retorna True se sucesso, False senão."""
        p = Path(png_path)
        if not p.exists():
            return False
        pixmap = ler_pixmap_orientado(p)
        if pixmap.isNull():
            return False
        self._full_pixmap = pixmap
        self._placeholder_text = ""
        # updateGeometry sinaliza ao layout que sizeHint/heightForWidth
        # mudaram — sem isso a coluna não recalcula a altura ao trocar
        # de imagem.
        self.updateGeometry()
        self.update()
        return True

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        painter = QtGui.QPainter(self)
        rect = self.rect()

        if self._full_pixmap is not None:
            # Escala pela LARGURA — preenche toda a largura disponível.
            # Altura segue o aspect ratio do PNG. heightForWidth garante
            # que o layout dê altura compatível, então não cropamos.
            scaled = self._full_pixmap.scaledToWidth(
                rect.width(),
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            # Centraliza verticalmente caso o widget receba altura > pedida.
            # Largura sempre encosta nas duas bordas.
            y = max(0, (rect.height() - scaled.height()) // 2)
            painter.drawPixmap(0, y, scaled)
            return

        if self._placeholder_text:
            painter.setPen(QtGui.QColor("#888"))
            font = painter.font()
            font.setItalic(True)
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(
                rect,
                QtCore.Qt.AlignmentFlag.AlignCenter
                | QtCore.Qt.TextFlag.TextWordWrap,
                self._placeholder_text,
            )
