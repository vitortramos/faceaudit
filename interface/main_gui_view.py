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
        self._setup_corner_config_button()

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