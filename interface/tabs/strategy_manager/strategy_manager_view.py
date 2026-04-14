# interface/tabs/strategy_manager/strategy_manager_view.py
from PyQt6 import QtWidgets, QtCore, QtGui
from .panels.preparation.preparation_view import PreparationView
from .panels.validation.validation_view import ValidationView
from .panels.analysis.analysis_view import AnalysisView

class StrategyManagerView(QtWidgets.QWidget):
    """
    View Container que organiza a aba de estratégias.
    Utiliza um QStackedWidget para alternar entre uma tela de boas-vindas
    e o ambiente de trabalho de 3 colunas.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_layout()

    def _init_layout(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(5, 5, 5, 5)

        # 1. TOOLBAR (Sempre visível no topo)
        self.toolbar_container = QtWidgets.QWidget()
        self.toolbar_container.setMinimumHeight(65)
        self.toolbar_layout = QtWidgets.QVBoxLayout(self.toolbar_container)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 5)
        root.addWidget(self.toolbar_container)

        # 2. STACKED WIDGET (O conteúdo que alterna)
        self.content_stack = QtWidgets.QStackedWidget()
        
        # PÁGINA 0: Tela de Boas-Vindas (Estado Inicial)
        self.welcome_page = self._create_welcome_page()
        self.content_stack.addWidget(self.welcome_page)

        # PÁGINA 1: O Workspace de 3 Colunas (Estado Ativo)
        self.workspace_page = QtWidgets.QWidget()
        self._setup_workspace_layout()
        self.content_stack.addWidget(self.workspace_page)

        root.addWidget(self.content_stack)
        
        # Inicia na tela de boas-vindas
        self.show_welcome_screen()

    def _create_welcome_page(self):
        """Cria uma tela centralizada com a nova comunicação pericial."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        # Ícone representativo (Escudo/Justiça/Análise)
        icon = QtWidgets.QLabel("⚖️") 
        icon.setStyleSheet("font-size: 60px; margin-bottom: 15px;")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        title = QtWidgets.QLabel("Gerenciador de Estratégias")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        
        # Texto final acordado
        subtitle = QtWidgets.QLabel(
            "Crie um novo projeto ou abra um existente para:\n\n"
            "• **Configurar** estratégias de cálculo de Razão de Verosimilhança (LR)\n"
            "• **Validar** a performance das estratégias definidas\n"
            "• **Aplicar** em análises de casos específicos"
        )
        subtitle.setStyleSheet("""
            font-size: 15px; 
            color: #5f6368; 
            line-height: 180%;
            padding: 20px;
        """)
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(title, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        
        return page

    def _setup_workspace_layout(self):
        """Monta a estrutura de 3 colunas (Preparação, Validação, Análise)."""
        layout = QtWidgets.QHBoxLayout(self.workspace_page)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view_prep = PreparationView()
        self.view_val = ValidationView()
        self.view_ana = AnalysisView()

        self._add_column(layout, self.view_prep, "1. PREPARAÇÃO DE ESTRATÉGIAS", 1)
        self._add_separator(layout)
        self._add_column(layout, self.view_val, "2. VALIDAÇÃO DE ESTRATÉGIAS", 1)
        self._add_separator(layout)
        self._add_column(layout, self.view_ana, "3. APLICAÇÃO DE ESTRATÉGIAS", 1)

    def _add_column(self, layout, widget, title, stretch):
        group = QtWidgets.QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox { 
                font-weight: bold; font-size: 11px; color: #555; text-transform: uppercase;
                border: 1px solid #dcdde1; border-radius: 4px; margin-top: 20px;
                background-color: #f8f9fa;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        group_layout = QtWidgets.QVBoxLayout(group)
        group_layout.setContentsMargins(5, 15, 5, 5)
        group_layout.addWidget(widget)
        layout.addWidget(group, stretch)

    def _add_separator(self, layout):
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setStyleSheet("color: #dcdde1; margin: 20px 0 5px 0;")
        layout.addWidget(line)

    # --- CONTROLE DE NAVEGAÇÃO ---
    def show_welcome_screen(self):
        self.content_stack.setCurrentIndex(0)

    def show_workspace_screen(self):
        self.content_stack.setCurrentIndex(1)