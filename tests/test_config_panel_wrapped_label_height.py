"""Rótulos com word-wrap no painel de configuração não podem se sobrepor.

Regressão da tela de configuração da instrução EMBEDDING_AGGREGATION: no
bloco aninhado "Regras de Agregação", a descrição contextual do enum
"Método de fusão" aparecia comprimida por cima do combo acima e do
checkbox abaixo.

Causa: `QLabel.sizeHint()` com word-wrap é heurística — estima a altura
numa largura arbitrária. `QFormLayout` não propaga `heightForWidth` para
linhas que abrangem as duas colunas, então essa estimativa virava a
altura reservada. Numa coluna mais estreita que a estimada, o texto
quebrava em mais linhas do que cabia e as linhas vizinhas eram
comprimidas — daí a sobreposição.

Correção: `WrappedLabel` fixa `minimumHeight = heightForWidth(largura)`,
piso duro que sobe pela cadeia de layouts até o QScrollArea, que passa a
rolar em vez de comprimir.

O teste exercita a largura ESTREITA (400 px): é lá que a heurística do
sizeHint erra. Em largura folgada o bug não aparece nem sem a correção.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from faceaudit.enums.strategy_instruction_type import (  # noqa: E402
    StrategyInstructionType,
)
from faceaudit.models.aggregation_config import AggregationConfig  # noqa: E402
from faceaudit.models.instructions.embedding_aggregation_instruction import (  # noqa: E402
    EmbeddingAggregationInstruction,
)
from interface.tabs.strategy_manager_v2.strategy_editor.instruction_config_panel import (  # noqa: E402
    InstructionConfigPanel,
)
from interface.tabs.strategy_manager_v2.strategy_editor.nested_dto_widget import (  # noqa: E402
    NestedDTOInlineWidget,
)
from interface.tabs.strategy_manager_v2.strategy_editor.widget_helpers import (  # noqa: E402
    WrappedLabel,
)

# A 2ª coluna do editor de estratégias nasce com ~400 px. É abaixo da
# largura que a heurística do sizeHint assume, e é onde o texto quebrava
# em mais linhas do que o layout reservava.
NARROW_WIDTH_PX = 400


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _spanning_labels(form: QtWidgets.QFormLayout):
    """Rótulos que ocupam a linha inteira — ajuda e descrição de enum."""
    out = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.SpanningRole)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QtWidgets.QLabel) and widget.text().strip():
            out.append((row, widget))
    return out


def _build_nested_block(app, width: int):
    """Bloco "Regras de Agregação" isolado, numa coluna estreita."""
    host = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    # AlignTop reproduz o empilhamento do QScrollArea do painel: cada
    # bloco recebe a própria altura preferida, não uma fatia esticada.
    layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
    block = NestedDTOInlineWidget(
        title="Regras de Agregação",
        dto=AggregationConfig(),
        parent=host,
    )
    layout.addWidget(block)
    host.resize(width, 700)
    host.show()
    for _ in range(4):
        app.processEvents()
    return host, block


def test_nested_description_fits_at_narrow_width(app):
    """A descrição do enum tem altura para o texto na largura real."""
    host, block = _build_nested_block(app, NARROW_WIDTH_PX)
    try:
        labels = _spanning_labels(block._fields_layout)
        assert labels, "o bloco deveria ter descrição contextual de enum"
        for _row, lbl in labels:
            needed = lbl.heightForWidth(lbl.width())
            assert lbl.height() >= needed, (
                f"rótulo comprimido: {lbl.text()[:40]!r} recebeu "
                f"{lbl.height()}px e precisa de {needed}px "
                f"em {lbl.width()}px de largura"
            )
    finally:
        host.close()
        host.deleteLater()


def test_nested_rows_do_not_overlap_at_narrow_width(app):
    """Nenhuma linha do bloco invade a faixa vertical da linha seguinte."""
    host, block = _build_nested_block(app, NARROW_WIDTH_PX)
    try:
        form = block._fields_layout
        rects = []
        for row in range(form.rowCount()):
            for role in (
                QtWidgets.QFormLayout.ItemRole.LabelRole,
                QtWidgets.QFormLayout.ItemRole.FieldRole,
                QtWidgets.QFormLayout.ItemRole.SpanningRole,
            ):
                item = form.itemAt(row, role)
                if item is None:
                    continue
                widget = item.widget()
                if widget is None:
                    continue
                rects.append((row, widget, widget.geometry()))

        for i, (row_i, w_i, r_i) in enumerate(rects):
            for row_j, w_j, r_j in rects[i + 1:]:
                if row_i == row_j:
                    continue  # rótulo e campo da mesma linha dividem a faixa
                if w_i is w_j:
                    continue  # mesmo widget exposto em dois papéis
                overlap = min(r_i.bottom(), r_j.bottom()) - max(r_i.top(), r_j.top())
                assert overlap <= 0, (
                    f"sobreposição de {overlap + 1}px entre a linha {row_i} "
                    f"({type(w_i).__name__}) e a linha {row_j} "
                    f"({type(w_j).__name__})"
                )
    finally:
        host.close()
        host.deleteLater()


def test_config_panel_labels_fit(app):
    """O painel inteiro monta sem rótulo cortado (fumaça de integração)."""
    panel = InstructionConfigPanel()
    panel.load_instruction(
        EmbeddingAggregationInstruction(
            instruction_id="i1",
            name="",
            instruction_type=StrategyInstructionType.EMBEDDING_AGGREGATION,
        )
    )
    panel.show()
    for _ in range(4):
        app.processEvents()
    try:
        labels = [
            lbl
            for lbl in panel.findChildren(WrappedLabel)
            if lbl.isVisible() and lbl.text().strip()
        ]
        assert labels, "o painel deveria usar WrappedLabel nas caixas de ajuda"
        for lbl in labels:
            needed = lbl.heightForWidth(lbl.width())
            assert lbl.height() >= needed, (
                f"rótulo cortado: {lbl.text()[:40]!r} recebeu {lbl.height()}px "
                f"e precisa de {needed}px em {lbl.width()}px"
            )
    finally:
        panel.close()
        panel.deleteLater()
