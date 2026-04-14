from PyQt6 import QtWidgets, QtCore, QtGui

class ValidationView(QtWidgets.QWidget):
    """
    Coluna 2 (Validação): A Arena de Combate.
    Dividida em Configuração (Top) e Resultados (Bottom).
    """
    
    # Sinais da Interface
    config_changed = QtCore.pyqtSignal()
    execute_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(15)

        # Para permitir redimensionamento entre Configuração e Resultados
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.main_layout.addWidget(self.splitter)

        # =========================================================
        # PARTE SUPERIOR: CONFIGURAÇÃO DA ARENA
        # =========================================================
        self.config_widget = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(self.config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(10)
        config_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop) # Mantém tudo compactado no topo

        # 1. Gabarito Universal (Datasets)
        group_data = QtWidgets.QGroupBox("1. SELEÇÃO DE DATASETS DE VALIDAÇÃO")
        form_data = QtWidgets.QFormLayout(group_data)
        form_data.setSpacing(8)
        
        self.combo_probe = QtWidgets.QComboBox()
        self.lbl_stats_probe = QtWidgets.QLabel("-")
        self.lbl_stats_probe.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")
        v_probe = QtWidgets.QVBoxLayout()
        v_probe.addWidget(self.combo_probe)
        v_probe.addWidget(self.lbl_stats_probe)
        
        self.combo_ref = QtWidgets.QComboBox()
        self.lbl_stats_ref = QtWidgets.QLabel("-")
        self.lbl_stats_ref.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")
        v_ref = QtWidgets.QVBoxLayout()
        v_ref.addWidget(self.combo_ref)
        v_ref.addWidget(self.lbl_stats_ref)
        
        form_data.addRow("Probe (Questionada):", v_probe)
        form_data.addRow("Reference (Padrão):", v_ref)
        config_layout.addWidget(group_data)

        # 2. Competidores (Checkboxes Dinâmicos)
        group_strats = QtWidgets.QGroupBox("2. SELEÇÃO DE ESTRATÉGIAS PARA VALIDAÇÃO")
        l_strats = QtWidgets.QVBoxLayout(group_strats)
        
        self.scroll_strats = QtWidgets.QScrollArea()
        self.scroll_strats.setWidgetResizable(True)
        self.scroll_strats.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll_strats.setMinimumHeight(60)
        self.scroll_strats.setMaximumHeight(120) # Extremamente compacto
        
        self.strats_container = QtWidgets.QWidget()
        self.strats_layout = QtWidgets.QVBoxLayout(self.strats_container)
        self.strats_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.strats_layout.setContentsMargins(5, 5, 5, 5)
        self.scroll_strats.setWidget(self.strats_container)
        
        l_strats.addWidget(self.scroll_strats)
        config_layout.addWidget(group_strats)

        # 3. O Gatekeeper (Auditoria Forense)
        group_audit = QtWidgets.QGroupBox("3. Auditoria Forense (Data Leakage)")
        l_audit = QtWidgets.QVBoxLayout(group_audit)
        
        self.lbl_audit_status = QtWidgets.QLabel("Aguardando configuração...")
        self.lbl_audit_status.setWordWrap(True)
        self.lbl_audit_status.setStyleSheet("background-color: #e9ecef; border-radius: 4px; padding: 8px; font-size: 11px;")
        l_audit.addWidget(self.lbl_audit_status)
        
        config_layout.addWidget(group_audit)

        # 4. Ação Principal
        self.btn_execute = QtWidgets.QPushButton("⚔️ EXECUTAR VALIDAÇÃO COMPARATIVA")
        self.btn_execute.setStyleSheet("""
            QPushButton { 
                background-color: #0d6efd; color: white; font-weight: bold; 
                padding: 12px; border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.btn_execute.setEnabled(False)
        config_layout.addWidget(self.btn_execute)
        
        # --- A CORREÇÃO DE LAYOUT ESTÁ AQUI ---
        config_layout.addStretch() # Mola que impede a área superior de esticar e invadir a tela toda

        self.splitter.addWidget(self.config_widget)

        # =========================================================
        # PARTE INFERIOR: RESULTADOS
        # =========================================================
        self.results_widget = QtWidgets.QGroupBox("4. RESULTADO DA VALIDAÇÃO")
        self.results_widget.setStyleSheet("""
            QGroupBox { font-weight: bold; color: #495057; border: 1px solid #dee2e6; border-radius: 4px; margin-top: 8px;}
        """)
        results_layout = QtWidgets.QVBoxLayout(self.results_widget)
        
        # --- TABELA DE LEADERBOARD ---
        self.table_results = QtWidgets.QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["Estratégia", "Cllr", "Min Cllr", "EER"])
        self.table_results.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_results.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_results.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_results.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_results.setAlternatingRowColors(True)
        self.table_results.verticalHeader().setVisible(False)
        self.table_results.setMaximumHeight(150) # Tabela fininha no topo dos resultados
        
        results_layout.addWidget(self.table_results)

        # --- ABAS DE GRÁFICOS ---
        self.tabs_graphs = QtWidgets.QTabWidget()
        
        # Aba 1: Distribuição de Scores (PDF) - AGORA É A PRIMEIRA
        self.tab_dist = QtWidgets.QWidget()
        self.layout_dist = QtWidgets.QVBoxLayout(self.tab_dist)
        self.tabs_graphs.addTab(self.tab_dist, "Distribuição de LRs")
        
        # Aba 2: DET Curve
        self.tab_det = QtWidgets.QWidget()
        self.layout_det = QtWidgets.QVBoxLayout(self.tab_det)
        self.tabs_graphs.addTab(self.tab_det, "DET Curve")
        
        # Aba 3: Tippett Plot
        self.tab_tippett = QtWidgets.QWidget()
        self.layout_tippett = QtWidgets.QVBoxLayout(self.tab_tippett)
        self.tabs_graphs.addTab(self.tab_tippett, "Tippett Plot")
        
        # Aba 4: Calibração (ECE)
        self.tab_ece = QtWidgets.QWidget()
        self.layout_ece = QtWidgets.QVBoxLayout(self.tab_ece)
        self.tabs_graphs.addTab(self.tab_ece, "Calibração (ECE)")
        
        results_layout.addWidget(self.tabs_graphs)
        
        # --- SEGUNDA CORREÇÃO DE LAYOUT ---
        # Removi o "self.results_widget.setVisible(False)". 
        # Manter ele visível evita que a interface pisque ou estique repentinamente
        
        self.splitter.addWidget(self.results_widget)
        
        # Proporção do Splitter
        self.splitter.setStretchFactor(0, 0) 
        self.splitter.setStretchFactor(1, 1) 

        self._connect_internal_signals()

    def _connect_internal_signals(self):
        self.combo_probe.currentIndexChanged.connect(self.config_changed.emit)
        self.combo_ref.currentIndexChanged.connect(self.config_changed.emit)
        self.btn_execute.clicked.connect(self.execute_requested.emit)

    def set_gatekeeper_feedback(self, is_pure: bool, text: str):
        self.lbl_audit_status.setText(text)
        if is_pure:
            self.lbl_audit_status.setStyleSheet("background-color: #d1e7dd; color: #0f5132; border-radius: 4px; padding: 8px; font-weight: bold;")
        elif "CRÍTICO" in text or "❌" in text:
            self.lbl_audit_status.setStyleSheet("background-color: #f8d7da; color: #842029; border-radius: 4px; padding: 8px; font-weight: bold;")
        else:
            self.lbl_audit_status.setStyleSheet("background-color: #fff3cd; color: #856404; border-radius: 4px; padding: 8px;")

    def clear_strategies_list(self):
        while self.strats_layout.count():
            item = self.strats_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
    def set_stats_text(self, label_widget: QtWidgets.QLabel, n_imgs: int, n_ids: int):
        """Atualiza a UI com as quantidades do dataset selecionado."""
        txt = f"📸 {n_imgs} imgs | 👤 {n_ids} ids" if n_imgs > 0 else "-"
        label_widget.setText(txt)