from PyQt6 import QtCore
from .strategy_toolbar_view import StrategyToolbarView

class StrategyToolbarController(QtCore.QObject):
    action_create = QtCore.pyqtSignal()
    action_open = QtCore.pyqtSignal()
    action_save = QtCore.pyqtSignal()
    action_close = QtCore.pyqtSignal()
    action_add_strategy = QtCore.pyqtSignal(str)
    
    # Sinais para criação de regras
    action_new_mining_config = QtCore.pyqtSignal()
    action_new_pipeline = QtCore.pyqtSignal() 
    action_new_split = QtCore.pyqtSignal() # <--- NOVO SINAL DO SPLIT

    def __init__(self, parent_layout=None):
        super().__init__()
        self.view = StrategyToolbarView()
        self._connect_signals()

        if parent_layout is not None:
            parent_layout.addWidget(self.view)

    def _connect_signals(self):
        self.view.btn_create.clicked.connect(self.action_create.emit)
        self.view.btn_open.clicked.connect(self.action_open.emit)
        self.view.btn_save.clicked.connect(self.action_save.emit)
        self.view.btn_close.clicked.connect(self.action_close.emit)

        self.view.btn_add.clicked.connect(
            lambda: self.action_add_strategy.emit(self.view.combo_add_type.currentText())
        )
        
        self.view.btn_mining_config.clicked.connect(self.action_new_mining_config.emit)
        self.view.btn_new_pipeline.clicked.connect(self.action_new_pipeline.emit) 
        self.view.btn_new_split.clicked.connect(self.action_new_split.emit) # <--- CONEXÃO DO NOVO BOTÃO

    def set_workspace_open(self, is_open: bool):
        self.view.set_workspace_open(is_open)

    def set_status_text(self, text: str):
        self.view.set_status_text(text)

    def get_widget(self):
        return self.view