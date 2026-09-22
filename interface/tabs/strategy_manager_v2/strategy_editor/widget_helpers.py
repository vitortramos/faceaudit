# interface/tabs/strategy_manager_v2/strategy_editor/widget_helpers.py

"""
Helpers compartilhados pelos widgets do editor de estratégias.

Funções utilitárias pequenas que precisam ser reutilizadas entre o
InstructionConfigPanel, o NestedDTOInlineWidget e (possivelmente) outros.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtGui, QtWidgets


# Mínimo em pixels que combos/lineedits aceitam quando a coluna é
# estreitada. Acima disso aparece scrollbar horizontal no scroll do
# form. ~80 px = ~10 caracteres num font default.
MIN_FIELD_WIDTH_PX = 80


def make_combo_shrinkable(combo: QtWidgets.QComboBox) -> None:
    """
    Permite que o QComboBox encolha bem abaixo da largura do item mais
    longo do dropdown. Por default, o sizeHint do combo é dimensionado
    pelo item mais largo (ex: "Centering_CoralLW_WhiPCA256_L2" empurra
    o sizeHint pra ~250 px). Combinado com QFormLayout, isso cria um
    minimum width que dispara scrollbar horizontal antes do necessário.

    AdjustToMinimumContentsLengthWithIcon + minimumContentsLength curto
    diminui o minimumSizeHint do combo; setMinimumWidth complementa pra
    garantir piso final visualmente usável (~10 caracteres).
    """
    combo.setSizeAdjustPolicy(
        QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(8)
    combo.setMinimumWidth(MIN_FIELD_WIDTH_PX)


def make_lineedit_shrinkable(line: QtWidgets.QLineEdit) -> None:
    """Mesmo princípio: garante que o QLineEdit não se recuse a encolher."""
    line.setMinimumWidth(MIN_FIELD_WIDTH_PX)


class WrappedLabel(QtWidgets.QLabel):
    """
    QLabel com word-wrap que informa ao layout a altura REAL do texto
    quebrado na largura corrente.

    Motivação: `QLabel.sizeHint()` com word-wrap é heurística — o Qt
    arbitra uma largura "razoável" para estimar a altura, sem saber a
    largura que a coluna vai realmente ter. Dentro de um `QFormLayout`
    (que não propaga `heightForWidth` para linhas que abrangem as duas
    colunas) essa estimativa vira a altura efetiva da linha. Quando a
    coluna real é mais estreita que a estimada, o texto quebra em mais
    linhas do que o layout reservou: as linhas vizinhas são comprimidas
    e o texto aparece sobreposto.

    A correção é fixar `minimumHeight = heightForWidth(largura atual)`.
    Altura mínima é piso duro: sobe por toda a cadeia de layouts até o
    `QScrollArea`, que passa a rolar em vez de comprimir o conteúdo.

    Usar em QUALQUER rótulo com word-wrap que viva dentro de um
    `QFormLayout`/`QScrollArea` — caixas de ajuda, descrições
    contextuais de enum, resumos de regra.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        policy = self.sizePolicy()
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt API
        super().setText(text)
        # O texto novo pode ser mais curto que o anterior; zerar o piso
        # antes de recalcular evita travar a altura antiga.
        self.setMinimumHeight(0)
        self._sync_minimum_height()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        self._sync_minimum_height()

    def _sync_minimum_height(self) -> None:
        width = self.width()
        if width <= 0:
            return
        needed = self.heightForWidth(width)
        # Converge em uma passada extra: a largura é ditada pelo viewport
        # (barra de rolagem sempre reservada), então recalcular com a
        # mesma largura devolve o mesmo valor e o ciclo para.
        if needed > 0 and needed != self.minimumHeight():
            self.setMinimumHeight(needed)


def enable_height_for_width(widget: QtWidgets.QWidget) -> None:
    """
    Faz o widget declarar ao layout do pai que sua altura depende da
    largura — e não de um `sizeHint` fixo.

    `QWidgetItem.hasHeightForWidth()` consulta a *size policy*, não o
    layout interno. Um container comum (`QWidget` + `QFormLayout`) cujos
    filhos quebram texto tem `heightForWidth` correto, mas a política diz
    "não tenho", então o layout do pai reserva a altura do `sizeHint` —
    estimado numa largura que pode não ser a real. Esta chamada religa a
    cadeia: o pai passa a perguntar a altura PARA a largura que de fato
    vai dar ao widget.

    Aplicar em containers que embrulham rótulos com word-wrap
    (`WrappedLabel`), não em widgets de altura fixa.
    """
    policy = widget.sizePolicy()
    policy.setHeightForWidth(True)
    widget.setSizePolicy(policy)
