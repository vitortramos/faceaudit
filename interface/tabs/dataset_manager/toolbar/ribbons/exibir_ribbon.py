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

        # Toggle Tabela↔Galeria — o rótulo mostra o DESTINO: em modo tabela
        # exibe "Galeria" (clique = ir pra galeria); em modo galeria exibe
        # "Tabela" (clique = voltar). Estado interno via isChecked (checked =
        # galeria), preservado.
        self.btn_toggle_gallery = ToolbarButton(
            "Galeria", "🖼",
            tooltip="Alternar para o grid de thumbnails (galeria)",
        )
        self.btn_toggle_gallery.setCheckable(True)
        self.btn_toggle_gallery.toggled.connect(self._on_gallery_toggled)

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

    def _on_gallery_toggled(self, is_gallery: bool) -> None:
        """Rótulo/ícone refletem o DESTINO da troca: tabela → 'Galeria',
        galeria → 'Tabela'. Dirigido por `toggled`, então setChecked
        programático (ex.: ao fechar o dataset) também reverte o rótulo."""
        if is_gallery:
            self.btn_toggle_gallery.set_label("Tabela", "▦")
            self.btn_toggle_gallery.setToolTip("Voltar para a visualização de tabela")
        else:
            self.btn_toggle_gallery.set_label("Galeria", "🖼")
            self.btn_toggle_gallery.setToolTip("Alternar para o grid de thumbnails (galeria)")
