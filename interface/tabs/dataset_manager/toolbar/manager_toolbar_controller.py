from PyQt6 import QtCore
from .manager_toolbar_view import ManagerToolbarView

class ManagerToolbarController(QtCore.QObject):
    """
    Controlador da Barra de Ferramentas.
    Emite sinais para o Controlador Principal quando ações ocorrem.
    """
    # Sinais para comunicar com o Pai
    action_new = QtCore.pyqtSignal()
    action_new_from_video = QtCore.pyqtSignal()    # 🎬 criar dataset de vídeo
    action_import_video_run = QtCore.pyqtSignal()  # 📥 importar run de vídeo
    action_load = QtCore.pyqtSignal()
    action_close = QtCore.pyqtSignal()
    action_save = QtCore.pyqtSignal()
    action_save_as = QtCore.pyqtSignal()
    
    # Sinais de Ferramentas
    action_enrich = QtCore.pyqtSignal()
    action_balance = QtCore.pyqtSignal()  # Balanceamento por identidade
    action_merge = QtCore.pyqtSignal()
    action_split = QtCore.pyqtSignal()
    action_compare = QtCore.pyqtSignal()
    
    # Sinais de Análise
    action_stats = QtCore.pyqtSignal()
    action_analyze = QtCore.pyqtSignal() # <--- [NOVO] Sinal de Análise Interna
    action_schema_report = QtCore.pyqtSignal()  # PR 24 — relatório de schema + export
    action_audit_log = QtCore.pyqtSignal()      # PR 25 — audit log forense
    action_modify_paths = QtCore.pyqtSignal()   # PR L1d — modificação de caminhos
    action_filter = QtCore.pyqtSignal()
    action_clear_filter = QtCore.pyqtSignal()

    # PR 8 — Spreadsheet UX
    action_add_column = QtCore.pyqtSignal()
    # Edição centralizada de metadados (nome, descrição, is_aligned, specs).
    action_metadata = QtCore.pyqtSignal()

    # PR 15 — Modo galeria
    action_toggle_gallery = QtCore.pyqtSignal(bool)  # bool = is_gallery_mode

    # PR 16 — Adicionar à Biblioteca Global
    action_save_to_library = QtCore.pyqtSignal()

    # PR 17 — Enviar pra Estratégia (V2)
    action_send_to_strategy = QtCore.pyqtSignal()

    # PR I2 — Editor de pipeline de imagens
    action_image_pipeline_editor = QtCore.pyqtSignal()
    # PR I5 — Processar (batch) pipeline de imagens
    action_image_pipeline_apply = QtCore.pyqtSignal()

    def __init__(self, parent_layout=None):
        super().__init__()
        self.view = ManagerToolbarView()
        self._connect_signals()

        if parent_layout:
            parent_layout.addWidget(self.view)

    def _connect_signals(self):
        """Conecta os cliques dos botões aos sinais deste controller."""
        self.view.btn_new.clicked.connect(self.action_new.emit)
        self.view.btn_new_video.clicked.connect(self.action_new_from_video.emit)
        self.view.btn_import_video_run.clicked.connect(self.action_import_video_run.emit)
        self.view.btn_load.clicked.connect(self.action_load.emit)
        self.view.btn_close.clicked.connect(self.action_close.emit)
        self.view.btn_save.clicked.connect(self.action_save.emit)
        self.view.btn_save_as.clicked.connect(self.action_save_as.emit)
        
        self.view.btn_enrich.clicked.connect(self.action_enrich.emit)
        self.view.btn_balance.clicked.connect(self.action_balance.emit)
        self.view.btn_merge.clicked.connect(self.action_merge.emit)
        self.view.btn_split.clicked.connect(self.action_split.emit)
        self.view.btn_compare.clicked.connect(self.action_compare.emit)
        
        self.view.btn_stats.clicked.connect(self.action_stats.emit)
        self.view.btn_analyze.clicked.connect(self.action_analyze.emit) # <--- [NOVO] Conexão
        self.view.btn_schema_report.clicked.connect(self.action_schema_report.emit)  # PR 24
        self.view.btn_audit_log.clicked.connect(self.action_audit_log.emit)  # PR 25
        self.view.btn_modify_paths.clicked.connect(self.action_modify_paths.emit)  # PR L1d
        self.view.btn_filter.clicked.connect(self.action_filter.emit)
        self.view.btn_clear_filter.clicked.connect(self.action_clear_filter.emit)
        self.view.btn_add_column.clicked.connect(self.action_add_column.emit)  # PR 8
        self.view.btn_metadata.clicked.connect(self.action_metadata.emit)
        # PR 15: toggled emite com checked state — controller pode tratar
        # como flag (gallery on/off) sem precisar consultar o botão.
        self.view.btn_toggle_gallery.toggled.connect(self.action_toggle_gallery.emit)
        # PR 16: adicionar à biblioteca global
        self.view.btn_save_to_library.clicked.connect(self.action_save_to_library.emit)
        # PR 17: enviar pra Estratégia
        self.view.btn_send_to_strategy.clicked.connect(self.action_send_to_strategy.emit)
        # PR I2: editor de pipeline de imagens
        self.view.btn_image_pipeline_editor.clicked.connect(
            self.action_image_pipeline_editor.emit
        )
        # PR I5: aplicar pipeline em batch
        self.view.btn_image_pipeline_apply.clicked.connect(
            self.action_image_pipeline_apply.emit
        )

    def get_widget(self):
        return self.view

    # --- Métodos Públicos chamados pelo Pai ---

    def set_dataset_loaded_state(
        self, loaded: bool, status_text: str, tooltip: str = "",
        is_filtered: bool = False,
    ):
        """Atualiza a toolbar inteira baseado se há dataset ou não.

        `is_filtered` precisa ser repassado: ele governa o botão "Limpar
        filtro" e a cor do botão de filtro. Fixá-lo em False aqui apagava
        o indicador a cada recarga de tela (salvar, editar célula), com o
        subconjunto ainda na tela e o "Limpar filtro" desabilitado.
        """
        self.view.set_dataset_actions_enabled(loaded)
        self.view.set_status_text(
            status_text, is_filtered=is_filtered, tooltip=tooltip,
        )

    def update_filter_status(self, status_text: str, is_filtered: bool, tooltip: str = ""):
        """Atualiza especificamente o estado de filtro."""
        self.view.set_status_text(status_text, is_filtered=is_filtered, tooltip=tooltip)