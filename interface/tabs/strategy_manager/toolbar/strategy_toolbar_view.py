from PyQt6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox
from PyQt6.QtCore import Qt

class StrategyToolbarView(QFrame):
    """
    Toolbar do StrategyManager (similar ao DatasetManager).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_style()
        self._init_ui()

    def _init_style(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(60)
        self.setMaximumHeight(90)
        self.setStyleSheet(
            "QFrame { background-color: #f5f5f5; border-radius: 5px; border: 1px solid #e0e0e0; }"
        )

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(8)

        # Ações sempre disponíveis
        self.btn_create = QPushButton("📄 Criar")
        self.btn_open = QPushButton("📂 Abrir")

        # Ações que dependem de workspace aberto
        self.btn_save = QPushButton("💾 Salvar")
        self.btn_close = QPushButton("✕ Fechar")

        # Adicionar estratégia (depende de workspace aberto)
        self.combo_add_type = QComboBox()
        self.combo_add_type.addItems(["PLDA", "KDE"])
        self.combo_add_type.setFixedWidth(120)

        self.btn_add = QPushButton("➕ Adicionar Estratégia")
        
        # --- Botões de Criação de Regras Globais/Locais ---
        self.btn_mining_config = QPushButton("⛏️ Nova Mineração")
        self.btn_mining_config.setToolTip("Cria uma nova regra de curadoria/mineração de pares para este projeto.")
        
        self.btn_new_pipeline = QPushButton("🛠️ Novo Pipeline")
        self.btn_new_pipeline.setToolTip("Cria uma nova regra de transformação geométrica (PCA/LDA).")

        self.btn_new_split = QPushButton("✂️ Novo Split") # <--- NOVO BOTÃO DE SPLIT
        self.btn_new_split.setToolTip("Cria uma nova regra de corte (Arca de Noé) para PLDA vs Calibração.")

        # Montagem do Layout Horizontal
        row.addWidget(self.btn_create)
        row.addWidget(self.btn_open)

        row.addSpacing(12)

        row.addWidget(self.btn_save)
        row.addWidget(self.btn_close)

        row.addSpacing(12)

        row.addWidget(self.combo_add_type)
        row.addWidget(self.btn_add)
        
        row.addSpacing(12)
        
        row.addWidget(self.btn_mining_config)
        row.addWidget(self.btn_new_pipeline)
        row.addWidget(self.btn_new_split) # <--- INSERIDO NA BARRA

        row.addStretch()

        self.lbl_status = QLabel("Nenhuma configuração aberta.")
        self.lbl_status.setStyleSheet("background-color: white; padding: 2px 10px; border-radius: 3px; border: 1px solid #ddd;")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        main_layout.addLayout(row)
        main_layout.addWidget(self.lbl_status)

        # Estado inicial: workspace fechado
        self.set_workspace_open(False)

    def set_workspace_open(self, is_open: bool):
        """Habilita apenas os botões que exigem um projeto ativo."""
        self.btn_save.setEnabled(is_open)
        self.btn_close.setEnabled(is_open)
        self.combo_add_type.setEnabled(is_open)
        self.btn_add.setEnabled(is_open)
        
        # REMOVIDO: self.btn_mining_config.setEnabled(is_open)
        # REMOVIDO: self.btn_new_pipeline.setEnabled(is_open)
        # REMOVIDO: self.btn_new_split.setEnabled(is_open)
        # Como as linhas foram removidas, eles ficam True o tempo todo (estado padrão da criação)

    def set_status_text(self, text: str):
        self.lbl_status.setText(text)