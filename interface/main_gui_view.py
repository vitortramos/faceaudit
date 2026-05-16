import os
from PyQt6 import QtWidgets, uic, QtCore, QtGui
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton

class MainGuiView(QtWidgets.QMainWindow):
    """
    View da Janela Principal.
    Responsável apenas pela renderização da casca da aplicação e containers de abas.
    """
    config_requested = QtCore.pyqtSignal()

    def __init__(self, ui_path: str):
        super().__init__()
        try:
            uic.loadUi(ui_path, self)
        except Exception as e:
            raise RuntimeError(f"Falha crítica ao carregar arquivo de interface (.ui): {e}")

        self.main_tabs = self.findChild(QtWidgets.QTabWidget)
        self._apply_main_tabs_style()
        self._setup_corner_config_button()

    def _apply_main_tabs_style(self):
        """Estilo Office accent bar nas tabs principais — mesma família
        visual das tabs do ribbon (manager_toolbar_view), mas com
        accent navy escuro (`#1a3d6e`) + barra mais grossa (4px) +
        padding maior pra reforçar a hierarquia: tabs principais (nível 1
        — navegação no app) > tabs do ribbon (nível 2 — grupos de
        ferramentas)."""
        self.main_tabs.setStyleSheet(
            # Pane (área do conteúdo da aba)
            "QTabWidget::pane { "
            "    border: 1px solid #c8c8c8; "
            "    background: white; "
            "    top: -1px; "
            "} "
            # Aba inativa
            "QTabBar::tab { "
            "    background: #e0e0e0; "
            "    color: #555; "
            "    padding: 10px 24px; "
            "    font-size: 12px; "
            "    border: 1px solid #c8c8c8; "
            "    border-top: 4px solid transparent; "
            "    border-right: none; "
            "    border-top-left-radius: 5px; "
            "    border-top-right-radius: 5px; "
            "    margin-right: 1px; "
            "} "
            # Hover
            "QTabBar::tab:hover:!selected { "
            "    background: #d2d2d2; "
            "    color: #1a3d6e; "
            "} "
            # Aba selecionada — accent bar navy + texto navy + bold
            "QTabBar::tab:selected { "
            "    background: white; "
            "    color: #1a3d6e; "
            "    font-weight: bold; "
            "    border-top: 4px solid #1a3d6e; "
            "    border-bottom: 1px solid white; "
            "} "
            "QTabBar::tab:last { margin-right: 0; }"
        )

    def add_tab(self, index: int, widget: QWidget, label: str):
        """Insere um widget MVC em uma posição específica do QTabWidget."""
        if self.main_tabs.count() > index:
            self.main_tabs.removeTab(index)
        self.main_tabs.insertTab(index, widget, label)

    def _setup_corner_config_button(self):
        """Posiciona o botão de engrenagem no canto superior direito das abas."""
        self.btn_config = QPushButton("⚙️")
        self.btn_config.setToolTip("Configurações do Sistema")
        self.btn_config.setFixedSize(35, 30)
        self.btn_config.setStyleSheet("""
            QPushButton { border: none; background: transparent; font-size: 18px; color: #333; }
            QPushButton:hover { background-color: #e0e0e0; border-radius: 5px; }
        """)
        self.btn_config.clicked.connect(self.config_requested.emit)

        # Container para auxiliar o alinhamento vertical na barra de abas
        container = QWidget()
        l = QVBoxLayout(container)
        l.setContentsMargins(0, 0, 15, 0)
        l.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        l.addWidget(self.btn_config)

        self.main_tabs.setCornerWidget(container, QtCore.Qt.Corner.TopRightCorner)