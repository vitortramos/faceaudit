"""Ribbon de processamento de imagens (PR Ribbon refactor).

Placeholder pro tema Imagens (futuro). Os 2 botões estão estruturados
mas desabilitados — serão habilitados quando o tema Imagens for
implementado (pipeline interativo de blur/JPEG/WebP/crop/oclusão +
batch processing).

Decisão #2 do usuário: opção C (botões já presentes, desabilitados).
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..toolbar_widgets import ToolbarButton, ToolbarSeparator


class ImagensRibbon(QWidget):
    """Ribbon "Imagens" — placeholder pro pipeline de processamento."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # ── 2 botões estruturados, desabilitados (decisão #2 — opção C) ──
        self.btn_image_pipeline_editor = ToolbarButton(
            "Editor", "🎨",
            tooltip="(em desenvolvimento) Janela de edição interativa "
                    "do pipeline de processamento de imagens — preview "
                    "lado-a-lado da imagem original e processada, com "
                    "scores de qualidade ao vivo.",
        )
        self.btn_image_pipeline_editor.setEnabled(False)

        self.btn_image_pipeline_apply = ToolbarButton(
            "Processar", "🌀",
            tooltip="(em desenvolvimento) Aplicar pipeline de imagens "
                    "(blur, JPEG, WebP, crop, oclusão) a todo o dataset.",
        )
        self.btn_image_pipeline_apply.setEnabled(False)

        layout.addWidget(self.btn_image_pipeline_editor)
        layout.addWidget(self.btn_image_pipeline_apply)
        layout.addWidget(ToolbarSeparator())

        # Label placeholder — comunica claramente que vem aí
        self._lbl_placeholder = QLabel("⚙ Funcionalidade em desenvolvimento")
        self._lbl_placeholder.setStyleSheet(
            "color: #888; font-style: italic; padding-left: 12px;"
        )
        self._lbl_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(self._lbl_placeholder)

        layout.addStretch()
