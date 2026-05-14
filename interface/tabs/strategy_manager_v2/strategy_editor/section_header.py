# interface/tabs/strategy_manager_v2/strategy_editor/section_header.py

from __future__ import annotations

from PyQt6 import QtWidgets, QtCore


class SectionHeader(QtWidgets.QFrame):
    """Faixa colorida com título da seção do InstructionConfigPanel."""

    _VARIANTS = {
        "identification": ("#F4F6FA", "#555E6D", "#C8D0DC"),
        "configuration":  ("#D6E4F7", "#1A2942", "#2E5090"),
        "outputs":        ("#E8F5E9", "#2E7D32", "#4CAF50"),
        "validation":     ("#FFF8E8", "#8B5E00", "#E8A020"),
    }

    def __init__(self, title: str, variant: str = "identification", parent=None):
        super().__init__(parent)
        if variant not in self._VARIANTS:
            variant = "identification"

        bg, fg, accent = self._VARIANTS[variant]

        self.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {bg};"
            f"  border-left: 4px solid {accent};"
            f"  border-top: 1px solid {accent};"
            f"  border-bottom: 1px solid {accent};"
            f"}}"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(0)

        self._label = QtWidgets.QLabel(title.upper())
        self._label.setStyleSheet(
            f"background: transparent; border: none;"
            f"color: {fg}; font-size: 11px; font-weight: bold;"
            f"letter-spacing: 0.5px;"
        )
        layout.addWidget(self._label)
        layout.addStretch(1)
        self._layout = layout

    def setTitle(self, title: str) -> None:
        self._label.setText(title.upper())

    def addActionWidget(self, widget: QtWidgets.QWidget) -> None:
        """
        Adiciona um widget à direita do título (após o stretch). Útil para
        botões de ação contextuais à seção (ex: "Restaurar padrões" na
        seção de regras de validação).
        """
        self._layout.addWidget(widget)


class ColumnHeader(QtWidgets.QFrame):
    """
    Cabeçalho principal de uma das três colunas do editor de estratégias
    (Instruções / Configurações / Saídas).

    Visualmente mais proeminente que `SectionHeader` (que rotula sub-seções
    internas a uma coluna): fundo cinza-azulado mais marcado, fonte maior
    e altura fixa um pouco maior pra criar hierarquia clara entre o título
    da coluna e os títulos das subseções.

    Padrão único reutilizado nas 3 colunas para garantir consistência visual.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        bg = "#2C3E50"
        fg = "#FFFFFF"

        self.setFixedHeight(30)
        self.setStyleSheet(
            f"QFrame {{"
            f"  background-color: {bg};"
            f"  border-radius: 3px;"
            f"}}"
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(0)

        self._label = QtWidgets.QLabel(title.upper())
        self._label.setStyleSheet(
            f"background: transparent; border: none;"
            f" color: {fg}; font-size: 13px; font-weight: bold;"
            f" letter-spacing: 1px;"
        )
        layout.addWidget(self._label)
        layout.addStretch(1)

    def setTitle(self, title: str) -> None:
        self._label.setText(title.upper())
