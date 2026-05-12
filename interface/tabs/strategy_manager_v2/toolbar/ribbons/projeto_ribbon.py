"""Ribbon "Projeto" — ações sobre o projeto FaceAudit (.fa) aberto.

Botões: Novo, Abrir, Salvar, Fechar, 🔒 Selar.

Reusa `ToolbarButton` + `ResponsiveRibbon` da toolbar do Dataset
Manager — mesma aparência visual e comportamento responsivo
(alterna full ↔ compact por largura).
"""

from PyQt6.QtWidgets import QHBoxLayout

from interface.tabs.dataset_manager.toolbar.ribbons.responsive_ribbon import (
    ResponsiveRibbon,
)
from interface.tabs.dataset_manager.toolbar.toolbar_widgets import (
    ToolbarButton,
    ToolbarSeparator,
)


class ProjetoRibbon(ResponsiveRibbon):
    """Ribbon "Projeto" — ciclo de vida do .fa: criar, abrir, salvar,
    fechar e selar (empacotar pra entrega forense)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Criação / leitura
        self.btn_new = ToolbarButton(
            "Novo", "➕", style="success",
            tooltip="Criar novo projeto FaceAudit",
        )
        self.btn_open = ToolbarButton(
            "Abrir", "📂", style="primary",
            tooltip="Abrir projeto FaceAudit existente (.fa)",
        )
        layout.addWidget(self.btn_new)
        layout.addWidget(self.btn_open)
        layout.addWidget(ToolbarSeparator())

        # Persistência
        self.btn_save = ToolbarButton(
            "Salvar", "💾",
            tooltip="Salvar projeto atual",
        )
        layout.addWidget(self.btn_save)
        layout.addWidget(ToolbarSeparator())

        # Empacotamento forense
        self.btn_seal = ToolbarButton(
            "Selar", "🔒",
            tooltip="Selar projeto: copia todas as imagens referenciadas "
                    "pra dentro do .fa e reescreve paths como relativos. "
                    "Empacota o projeto pra entrega forense (USB/zip/laudo).",
        )
        layout.addWidget(self.btn_seal)
        layout.addWidget(ToolbarSeparator())

        # Encerramento
        self.btn_close = ToolbarButton(
            "Fechar", "❌", style="danger",
            tooltip="Fechar projeto atual",
        )
        layout.addWidget(self.btn_close)

        layout.addStretch()

        self._install_responsive_behavior()
