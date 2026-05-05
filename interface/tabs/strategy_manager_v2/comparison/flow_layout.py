# interface/tabs/strategy_manager_v2/comparison/flow_layout.py

"""
FlowLayout — porta PyQt6 da implementação canônica do Qt.

Posiciona widgets-filho em linhas, quebrando para a linha seguinte quando
a largura disponível não permite mais um item. Não faz cálculo manual de
colunas — confia no mecanismo `hasHeightForWidth/heightForWidth` do Qt.

Cada widget contribui com `sizeHint()` para o cálculo. Como a janela
recalcula `heightForWidth` automaticamente em cada resize, o reflow é
"responsivo de verdade" sem código adicional.

Tradução fiel de:
  https://code.qt.io/cgit/qt/qtbase.git/tree/examples/widgets/layouts/flowlayout?h=6.11

Para uniformizar a largura visual entre cells, o caller deve setar
`setFixedSize(W, H)` (ou `setMinimumSize/setMaximumSize` iguais) em cada
widget filho — todos terão exatamente a mesma área.

CELL_WIDTH e CELL_HEIGHT são as dimensões padronizadas usadas pelas 4
views da janela de comparação (calibração, casework, validação overlay,
validação LR distribution).
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore, QtWidgets


# Tamanho fixo dos cells — todos os grids da ComparisonWindow usam esse
# tamanho. Mantém visual consistente entre tipos de comparação.
CELL_WIDTH = 540
CELL_HEIGHT = 420


class FlowLayout(QtWidgets.QLayout):
    """
    Layout que posiciona itens em linhas, quebrando quando não cabem.

    Diferentemente de QGridLayout, não tem conceito de coluna fixa: cada
    item ocupa seu sizeHint, e a próxima linha começa quando a soma das
    larguras excede a largura disponível.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        margin: int = 0,
        h_spacing: int = -1,
        v_spacing: int = -1,
    ):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._items: List[QtWidgets.QLayoutItem] = []

    # ── QLayout interface ────────────────────────────────────────────────────

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QtWidgets.QLayoutItem]:
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QtCore.QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    # ── Spacing helpers ──────────────────────────────────────────────────────

    def horizontal_spacing(self) -> int:
        if self._h_spacing >= 0:
            return self._h_spacing
        return self._smart_spacing(
            QtWidgets.QStyle.PixelMetric.PM_LayoutHorizontalSpacing
        )

    def vertical_spacing(self) -> int:
        if self._v_spacing >= 0:
            return self._v_spacing
        return self._smart_spacing(
            QtWidgets.QStyle.PixelMetric.PM_LayoutVerticalSpacing
        )

    def _smart_spacing(self, pm: QtWidgets.QStyle.PixelMetric) -> int:
        parent = self.parent()
        if parent is None:
            return -1
        if parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        return parent.spacing()

    # ── Core layout algorithm ────────────────────────────────────────────────

    def _do_layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(), margins.top(),
            -margins.right(), -margins.bottom(),
        )
        x = effective.x()
        y = effective.y()
        line_height = 0

        for item in self._items:
            wid = item.widget()
            space_x = self.horizontal_spacing()
            if space_x == -1:
                space_x = wid.style().layoutSpacing(
                    QtWidgets.QSizePolicy.ControlType.PushButton,
                    QtWidgets.QSizePolicy.ControlType.PushButton,
                    QtCore.Qt.Orientation.Horizontal,
                )
            space_y = self.vertical_spacing()
            if space_y == -1:
                space_y = wid.style().layoutSpacing(
                    QtWidgets.QSizePolicy.ControlType.PushButton,
                    QtWidgets.QSizePolicy.ControlType.PushButton,
                    QtCore.Qt.Orientation.Vertical,
                )

            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > effective.right() and line_height > 0:
                x = effective.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0

            if not test_only:
                item.setGeometry(
                    QtCore.QRect(QtCore.QPoint(x, y), item.sizeHint())
                )

            x = next_x
            line_height = max(line_height, item.sizeHint().height())

        return y + line_height - rect.y() + margins.bottom()
