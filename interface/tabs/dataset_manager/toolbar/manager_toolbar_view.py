from PyQt6.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout
from .toolbar_widgets import ToolbarButton, ToolbarSeparator, StatusLabel

class ManagerToolbarView(QFrame):
    """
    View da Barra de Ferramentas.
    Responsável pelo layout dos botões e do status.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_style()
        self._init_ui()

    def _init_style(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMaximumHeight(90)
        self.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 5px; border: 1px solid #e0e0e0; }")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        # Layout Horizontal dos Botões
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        # --- GRUPO 1: ARQUIVO ---
        self.btn_new = ToolbarButton("Novo Dataset", "➕", style="success", tooltip="Criar novo dataset")
        self.btn_load = ToolbarButton("Abrir", "📂", style="primary")
        self.btn_close = ToolbarButton("Fechar", "❌", style="danger")
        self.btn_save = ToolbarButton("Salvar", "💾")
        self.btn_save_as = ToolbarButton("Salvar Como...", "💾")

        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_close)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_save_as)

        btn_layout.addWidget(ToolbarSeparator())

        # --- GRUPO 2: FERRAMENTAS ---
        self.btn_enrich = ToolbarButton("Enriquecer", "💎", tooltip="Adicionar métricas (OFIQ, SER-FIQ) a um dataset existente")
        self.btn_merge = ToolbarButton("Juntar", "🔗")
        self.btn_split = ToolbarButton("Dividir", "✂️")
        self.btn_compare = ToolbarButton("Comparar", "⚖️", tooltip="Comparar este dataset com outro")

        btn_layout.addWidget(self.btn_enrich)
        btn_layout.addWidget(self.btn_merge)
        btn_layout.addWidget(self.btn_split)
        btn_layout.addWidget(self.btn_compare)

        btn_layout.addWidget(ToolbarSeparator())

        # --- GRUPO 3: ANÁLISE ---
        self.btn_stats = ToolbarButton("Estatísticas", "📊")

        # <--- [NOVO] Botão de Análise de Integridade e Mineração --->
        self.btn_analyze = ToolbarButton("Analisar", "🔎", tooltip="Auditoria interna de integridade e hard-mining biométrico")

        # <--- [PR 24, Tema 5.3+5.5] Relatório de schema + export CSV/Excel --->
        self.btn_schema_report = ToolbarButton(
            "Relatório", "📋",
            tooltip="Relatório de cobertura por coluna (% NaN, distinct, "
                    "min/max). Inclui exportação para CSV/Excel.",
        )

        self.btn_filter = ToolbarButton("Filtrar", "🔍")
        self.btn_clear_filter = ToolbarButton("Limpar", "🗑️", tooltip="Remover filtros")
        self.btn_clear_filter.setEnabled(False)

        # <--- [PR 8] Spreadsheet UX: adicionar coluna custom ao dataset --->
        self.btn_add_column = ToolbarButton(
            "Coluna", "➕📋",
            tooltip="Adicionar coluna custom ao dataset (anotação livre)",
        )

        # <--- [PR 15] Modo galeria: toggle entre tabela e grid de thumbnails --->
        self.btn_toggle_gallery = ToolbarButton(
            "Galeria", "🖼",
            tooltip="Alternar entre visualização de tabela e grid de thumbnails",
        )
        self.btn_toggle_gallery.setCheckable(True)

        # <--- [PR 16] Adicionar à Biblioteca Global --->
        self.btn_save_to_library = ToolbarButton(
            "Biblioteca", "📚",
            tooltip="Adiciona o dataset atual à biblioteca global "
                    "(reusável entre projetos do Strategy Manager V2)",
        )

        # <--- [PR 17] Enviar pra Estratégia (V2) --->
        self.btn_send_to_strategy = ToolbarButton(
            "→ Estratégia", "🎯",
            tooltip="Envia o dataset atual para o Strategy Manager V2 — "
                    "switch de aba + import automático no projeto aberto "
                    "(ou prompt pra abrir/criar projeto)",
        )

        btn_layout.addWidget(self.btn_stats)
        btn_layout.addWidget(self.btn_analyze) # <--- [NOVO] Adicionado ao layout
        btn_layout.addWidget(self.btn_schema_report)  # <--- [PR 24]
        btn_layout.addWidget(self.btn_filter)
        btn_layout.addWidget(self.btn_clear_filter)
        btn_layout.addWidget(self.btn_add_column)
        btn_layout.addWidget(self.btn_toggle_gallery)
        btn_layout.addWidget(self.btn_save_to_library)
        btn_layout.addWidget(self.btn_send_to_strategy)
        
        btn_layout.addStretch() # Empurra tudo para a esquerda

        # --- LABEL DE STATUS ---
        self.lbl_status = StatusLabel()

        # Adiciona ao layout principal
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.lbl_status)

    # --- API de Manipulação Visual ---

    def set_dataset_actions_enabled(self, enabled: bool):
        """Habilita/Desabilita botões que dependem de um dataset carregado."""
        buttons = [
            self.btn_close, self.btn_save, self.btn_save_as,
            self.btn_enrich,
            self.btn_merge, self.btn_split, self.btn_compare,
            self.btn_stats,
            self.btn_analyze, # <--- [NOVO] Controlado dinamicamente
            self.btn_schema_report,  # <--- [PR 24]
            self.btn_filter,
            self.btn_add_column,  # <--- [PR 8]
            self.btn_toggle_gallery,  # <--- [PR 15]
            self.btn_save_to_library,  # <--- [PR 16]
            self.btn_send_to_strategy,  # <--- [PR 17]
        ]
        for btn in buttons:
            btn.setEnabled(enabled)
            
        # O btn_clear_filter tem lógica própria, então forçamos False se desabilitar tudo
        if not enabled:
            self.btn_clear_filter.setEnabled(False)

    def set_status_text(self, text: str, is_filtered: bool = False):
        """Atualiza o texto de status e o estado do botão de filtro."""
        self.lbl_status.setText(text)
        
        if is_filtered:
            self.btn_clear_filter.setEnabled(True)
            self.btn_filter.set_style("warning") # Muda cor para amarelo
        else:
            if self.btn_filter.isEnabled(): 
                self.btn_clear_filter.setEnabled(False)
                self.btn_filter.set_style("default") # Volta ao normal