"""Ribbon "Biblioteca Global" — recursos reutilizáveis entre projetos.

Botões: Pipeline, Mining, Split, Gerenciar Biblioteca.

Reusa `ToolbarButton` + `ResponsiveRibbon` da toolbar do Dataset
Manager — visual idêntico ao ProjetoRibbon e ao Dataset Manager.
"""

from PyQt6.QtWidgets import QHBoxLayout

from interface.tabs.dataset_manager.toolbar.ribbons.responsive_ribbon import (
    ResponsiveRibbon,
)
from interface.tabs.dataset_manager.toolbar.toolbar_widgets import (
    ToolbarButton,
    ToolbarSeparator,
)


class BibliotecaRibbon(ResponsiveRibbon):
    """Ribbon "Biblioteca Global" — criar/editar configs reutilizáveis
    + abrir gerenciador de recursos."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Configs reutilizáveis
        self.btn_pipeline = ToolbarButton(
            "Pipeline", "🔧",
            tooltip="Criar ou editar configuração de pipeline de "
                    "transformação",
        )
        self.btn_mining = ToolbarButton(
            "Mining", "⛏",
            tooltip="Criar ou editar configuração de mineração de pares",
        )
        self.btn_split = ToolbarButton(
            "Split", "✂️",
            tooltip="Criar ou editar configuração de divisão de dataset",
        )
        layout.addWidget(self.btn_pipeline)
        layout.addWidget(self.btn_mining)
        layout.addWidget(self.btn_split)
        layout.addWidget(ToolbarSeparator())

        # Gerenciamento global
        self.btn_manage = ToolbarButton(
            "Gerenciar Biblioteca", "📚",
            tooltip="Visualizar e remover recursos da biblioteca global",
        )
        layout.addWidget(self.btn_manage)

        layout.addStretch()

        self._install_responsive_behavior()
