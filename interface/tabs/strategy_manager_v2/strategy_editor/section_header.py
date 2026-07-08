# interface/tabs/strategy_manager_v2/strategy_editor/section_header.py

from __future__ import annotations

from PyQt6 import QtWidgets, QtCore

from interface.theme import apply_themed_stylesheet


class SectionHeader(QtWidgets.QFrame):
    """Faixa colorida com título da seção do InstructionConfigPanel."""

    # (bg_token, fg_token, accent_token) por variante.
    # fg = cor do título: LEGÍVEL nos dois modos (nada de TEXT_MUTED apagado).
    # accent = barra/borda: um token VISÍVEL (não BORDER/HEADER_BG, quase
    # invisíveis no escuro).
    _VARIANTS = {
        "identification": ("SURFACE_RAISED", "TEXT", "TEXT_MUTED"),
        "configuration":  ("PRIMARY_BG", "PRIMARY_STRONG", "PRIMARY"),
        "outputs":        ("SUCCESS_BG", "SUCCESS", "SUCCESS"),
        "validation":     ("ACCENT_BG", "ACCENT_TEXT", "ACCENT"),
    }

    def __init__(self, title: str, variant: str = "identification", parent=None):
        super().__init__(parent)
        if variant not in self._VARIANTS:
            variant = "identification"

        bg_tok, fg_tok, accent_tok = self._VARIANTS[variant]

        # apply_themed_stylesheet → re-aplica ao trocar de tema ao vivo.
        apply_themed_stylesheet(
            self,
            lambda t: (
                "QFrame {"
                "  background-color: " + t(bg_tok) + ";"
                "  border-left: 4px solid " + t(accent_tok) + ";"
                "  border-top: 1px solid " + t(accent_tok) + ";"
                "  border-bottom: 1px solid " + t(accent_tok) + ";"
                "}"
            ),
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(0)

        self._label = QtWidgets.QLabel(title.upper())
        apply_themed_stylesheet(
            self._label,
            lambda t: (
                "background: transparent; border: none;"
                " color: " + t(fg_tok) + "; font-size: 11px; font-weight: bold;"
                " letter-spacing: 0.5px;"
            ),
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

        self.setFixedHeight(30)
        apply_themed_stylesheet(
            self,
            lambda t: (
                "QFrame {"
                "  background-color: " + t("HEADER_BG") + ";"
                "  border-radius: 3px;"
                "}"
            ),
        )

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(0)

        self._label = QtWidgets.QLabel(title.upper())
        apply_themed_stylesheet(
            self._label,
            lambda t: (
                "background: transparent; border: none;"
                " color: " + t("HEADER_TEXT") + "; font-size: 13px; font-weight: bold;"
                " letter-spacing: 1px;"
            ),
        )
        layout.addWidget(self._label)
        layout.addStretch(1)

    def setTitle(self, title: str) -> None:
        self._label.setText(title.upper())
