"""Ribbon de operações de arquivo (PR Ribbon refactor).

Botões: Novo, Abrir, Salvar, Salvar Como, Salvar na Biblioteca,
Fechar.
"""

from PyQt6.QtWidgets import QHBoxLayout

from ..toolbar_widgets import ToolbarButton, ToolbarSeparator
from .responsive_ribbon import ResponsiveRibbon


class ArquivoRibbon(ResponsiveRibbon):
    """Ribbon "Arquivo" — operações de I/O sobre o arquivo de dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Criação / leitura
        self.btn_new = ToolbarButton(
            "Novo Dataset", "➕", style="success",
            tooltip="Criar novo dataset",
        )
        self.btn_load = ToolbarButton("Abrir", "📂", style="primary")

        layout.addWidget(self.btn_new)
        layout.addWidget(self.btn_load)
        layout.addWidget(ToolbarSeparator())

        # Persistência
        self.btn_save = ToolbarButton("Salvar", "💾")
        self.btn_save_as = ToolbarButton("Salvar Como...", "💾")

        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_save_as)
        layout.addWidget(ToolbarSeparator())

        # Exportação
        self.btn_save_to_library = ToolbarButton(
            "Biblioteca", "📚",
            tooltip="Adiciona o dataset atual à biblioteca global "
                    "(reusável entre projetos do Strategy Manager V2)",
        )
        layout.addWidget(self.btn_save_to_library)
        layout.addWidget(ToolbarSeparator())

        # Encerramento
        self.btn_close = ToolbarButton("Fechar", "❌", style="danger")
        layout.addWidget(self.btn_close)

        layout.addStretch()

        self._install_responsive_behavior()
