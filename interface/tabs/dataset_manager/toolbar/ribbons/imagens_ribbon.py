"""Ribbon de processamento de imagens.

PR Ribbon refactor — botões já estruturados.
PR I2 — botão Editor habilitado (preview-only).
PR I4 — Editor ganha persistência (Salvar/Carregar).
PR I5 — botão Processar (batch) habilitado.
"""

from PyQt6.QtWidgets import QHBoxLayout, QWidget

from ..toolbar_widgets import ToolbarButton


class ImagensRibbon(QWidget):
    """Ribbon "Imagens" — placeholder pro pipeline de processamento."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # ── PR I2 — Editor habilitado (preview-only; salvar em I4) ──
        self.btn_image_pipeline_editor = ToolbarButton(
            "Editor", "🎨",
            tooltip="Janela de edição interativa do pipeline de "
                    "processamento de imagens — preview lado-a-lado "
                    "da imagem original e processada. "
                    "Persistência (Salvar) virá em PR I4; scores de "
                    "qualidade ao vivo virão em PR I3.",
        )
        # Habilitado dinamicamente quando dataset estiver carregado
        # (set_dataset_actions_enabled controla o estado).

        # ── PR I5 — Processar (batch) habilitado ──
        self.btn_image_pipeline_apply = ToolbarButton(
            "Processar", "🌀",
            tooltip="Aplicar pipeline de imagens da biblioteca global a "
                    "todo o dataset. Produz dataset enxuto com PNGs "
                    "processados + manifest forense.",
        )
        # Habilitado dinamicamente pelo controlador pai quando dataset
        # estiver carregado (set_dataset_actions_enabled).

        layout.addWidget(self.btn_image_pipeline_editor)
        layout.addWidget(self.btn_image_pipeline_apply)
        layout.addStretch()
