# interface/tabs/strategy_manager_v2/strategy_editor/section_header.py

from __future__ import annotations

from PyQt6 import QtWidgets, QtCore


class SectionHeader(QtWidgets.QFrame):
    """Faixa colorida com título da seção do InstructionConfigPanel."""

    _VARIANTS = {
        "identification": ("#F4F6FA", "#555E6D", "#C8D0DC"),
        "configuration":  ("#D6E4F7", "#1A2942", "#2E5090"),
        "outputs":        ("#E8F5E9", "#2E7D32", "#4CAF50"),
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

    def setTitle(self, title: str) -> None:
        self._label.setText(title.upper())
