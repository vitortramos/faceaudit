"""Ribbon de operações de arquivo (PR Ribbon refactor).

Botões: Novo, Abrir, Salvar, Salvar Como, Salvar na Biblioteca,
→ Estratégia, Fechar.
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
            tooltip="Criar novo dataset a partir de uma pasta de imagens",
        )
        self.btn_new_video = ToolbarButton(
            "Dataset de Vídeo", "🎬", style="success",
            tooltip="Criar dataset extraindo faces de um arquivo de vídeo",
        )
        self.btn_import_video_run = ToolbarButton(
            "Importar extração de vídeo", "📥",
            tooltip="Importar uma extração de vídeo já produzida (revisar + gerar dataset)",
        )
        self.btn_load = ToolbarButton("Abrir", "📂", style="primary")

        layout.addWidget(self.btn_new)
        layout.addWidget(self.btn_new_video)
        layout.addWidget(self.btn_import_video_run)
        layout.addWidget(self.btn_load)
        layout.addWidget(ToolbarSeparator())

        # Persistência
        self.btn_save = ToolbarButton("Salvar", "💾")
        self.btn_save_as = ToolbarButton("Salvar Como...", "💾")
        self.btn_save_selected = ToolbarButton(
            "Salvar Selecionados", "🧩",
            tooltip="Grava os registros selecionados como um dataset novo, "
                    "sem trocar o que está aberto. Permite copiar as "
                    "imagens junto (autocontido).",
        )

        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_save_as)
        layout.addWidget(self.btn_save_selected)
        layout.addWidget(ToolbarSeparator())

        # Exportação
        self.btn_save_to_library = ToolbarButton(
            "Biblioteca", "📚",
            tooltip="Adiciona o dataset atual à biblioteca global "
                    "(reusável entre projetos do Strategy Manager V2)",
        )
        layout.addWidget(self.btn_save_to_library)

        # Envio pra outra aba (movido de Dataset → Arquivo)
        self.btn_send_to_strategy = ToolbarButton(
            "→ Estratégia", "🎯",
            tooltip="Envia o dataset atual para o Strategy Manager V2 — "
                    "switch de aba + import automático no projeto aberto "
                    "(ou prompt pra abrir/criar projeto)",
        )
        layout.addWidget(self.btn_send_to_strategy)
        layout.addWidget(ToolbarSeparator())

        # Encerramento
        self.btn_close = ToolbarButton("Fechar", "❌", style="danger")
        layout.addWidget(self.btn_close)

        layout.addStretch()

        self._install_responsive_behavior()
