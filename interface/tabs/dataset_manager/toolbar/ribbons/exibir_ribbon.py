"""Ribbon "Exibir" — visualizações + relatórios sobre o dataset.

Botões: Galeria (alterna entre tabela e grid de thumbnails),
Relatório (cobertura por coluna, exportável CSV/Excel).
"""

from PyQt6.QtWidgets import QHBoxLayout

from ..toolbar_widgets import ToolbarButton
from .responsive_ribbon import ResponsiveRibbon


class ExibirRibbon(ResponsiveRibbon):
    """Ribbon "Exibir" — controles de visualização (não-mutativos)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.btn_toggle_gallery = ToolbarButton(
            "Galeria", "🖼",
            tooltip="Alternar entre visualização de tabela e grid "
                    "de thumbnails",
        )
        self.btn_toggle_gallery.setCheckable(True)

        self.btn_schema_report = ToolbarButton(
            "Colunas", "📋",
            tooltip="Painel de colunas: marca quais ficam visíveis na "
                    "tabela + relatório de cobertura (% NaN, distinct, "
                    "min/max). Inclui exportação para CSV/Excel.",
        )

        layout.addWidget(self.btn_toggle_gallery)
        layout.addWidget(self.btn_schema_report)
        layout.addStretch()

        self._install_responsive_behavior()
