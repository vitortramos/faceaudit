# interface/tabs/strategy_manager_v2/strategy_editor/widget_helpers.py

"""
Helpers compartilhados pelos widgets do editor de estratégias.

Funções utilitárias pequenas que precisam ser reutilizadas entre o
InstructionConfigPanel, o NestedDTOInlineWidget e (possivelmente) outros.
"""

from __future__ import annotations

from PyQt6 import QtWidgets


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
