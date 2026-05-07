from PyQt6 import QtCore
from .manager_toolbar_view import ManagerToolbarView

class ManagerToolbarController(QtCore.QObject):
    """
    Controlador da Barra de Ferramentas.
    Emite sinais para o Controlador Principal quando ações ocorrem.
    """
    # Sinais para comunicar com o Pai
    action_new = QtCore.pyqtSignal()
    action_load = QtCore.pyqtSignal()
    action_close = QtCore.pyqtSignal()
    action_save = QtCore.pyqtSignal()
    action_save_as = QtCore.pyqtSignal()
    
    # Sinais de Ferramentas
    action_enrich = QtCore.pyqtSignal()
    action_merge = QtCore.pyqtSignal()
    action_split = QtCore.pyqtSignal()
    action_compare = QtCore.pyqtSignal()
    
    # Sinais de Análise
    action_stats = QtCore.pyqtSignal()
    action_analyze = QtCore.pyqtSignal() # <--- [NOVO] Sinal de Análise Interna
    action_filter = QtCore.pyqtSignal()
    action_clear_filter = QtCore.pyqtSignal()

    # PR 8 — Spreadsheet UX
    action_add_column = QtCore.pyqtSignal()

    # PR 15 — Modo galeria
    action_toggle_gallery = QtCore.pyqtSignal(bool)  # bool = is_gallery_mode

    def __init__(self, parent_layout=None):
        super().__init__()
        self.view = ManagerToolbarView()
        self._connect_signals()

        if parent_layout:
            parent_layout.addWidget(self.view)

    def _connect_signals(self):
        """Conecta os cliques dos botões aos sinais deste controller."""
        self.view.btn_new.clicked.connect(self.action_new.emit)
        self.view.btn_load.clicked.connect(self.action_load.emit)
        self.view.btn_close.clicked.connect(self.action_close.emit)
        self.view.btn_save.clicked.connect(self.action_save.emit)
        self.view.btn_save_as.clicked.connect(self.action_save_as.emit)
        
        self.view.btn_enrich.clicked.connect(self.action_enrich.emit)
        self.view.btn_merge.clicked.connect(self.action_merge.emit)
        self.view.btn_split.clicked.connect(self.action_split.emit)
        self.view.btn_compare.clicked.connect(self.action_compare.emit)
        
        self.view.btn_stats.clicked.connect(self.action_stats.emit)
        self.view.btn_analyze.clicked.connect(self.action_analyze.emit) # <--- [NOVO] Conexão
        self.view.btn_filter.clicked.connect(self.action_filter.emit)
        self.view.btn_clear_filter.clicked.connect(self.action_clear_filter.emit)
        self.view.btn_add_column.clicked.connect(self.action_add_column.emit)  # PR 8
        # PR 15: toggled emite com checked state — controller pode tratar
        # como flag (gallery on/off) sem precisar consultar o botão.
        self.view.btn_toggle_gallery.toggled.connect(self.action_toggle_gallery.emit)

    def get_widget(self):
        return self.view

    # --- Métodos Públicos chamados pelo Pai ---

    def set_dataset_loaded_state(self, loaded: bool, status_text: str):
        """Atualiza a toolbar inteira baseado se há dataset ou não."""
        self.view.set_dataset_actions_enabled(loaded)
        self.view.set_status_text(status_text, is_filtered=False)

    def update_filter_status(self, status_text: str, is_filtered: bool):
        """Atualiza especificamente o estado de filtro."""
        self.view.set_status_text(status_text, is_filtered=is_filtered)