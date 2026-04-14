from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QFrame, QSpinBox, QCheckBox, QToolButton, 
    QFormLayout, QLineEdit, QGroupBox, QSizePolicy, QDoubleSpinBox
)
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.models.plda_validation_result import PLDAValidationResult, ValidationSeverity

class PLDACardView(QFrame):
    
    # Sinais para o Controller
    remove_requested = QtCore.pyqtSignal(object) 
    train_requested = QtCore.pyqtSignal()
    run_prep_requested = QtCore.pyqtSignal()
    visualize_requested = QtCore.pyqtSignal() 
    simulate_split_requested = QtCore.pyqtSignal() 
    
    # --- Sinais de Edição ---
    edit_source_pipeline_requested = QtCore.pyqtSignal()
    edit_target_pipeline_requested = QtCore.pyqtSignal()
    edit_split_requested = QtCore.pyqtSignal()    # <--- NOVO
    edit_mining_requested = QtCore.pyqtSignal()   # <--- NOVO
    
    def __init__(self, card_id: int, parent=None):
        super().__init__(parent)
        self.card_id = card_id
        
        self.setObjectName("pldaCard")
        self.setStyleSheet("""
            #pldaCard {
                background-color: white; border: 1px solid #ced4da;
                border-radius: 8px; margin-bottom: 10px;
            }
            QGroupBox {
                font-weight: bold; color: #495057; border: 1px solid #dee2e6;
                border-radius: 4px; margin-top: 8px; padding-top: 10px; padding-bottom: 5px;
            }
        """)
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # =========================================================
        # HEADER
        # =========================================================
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"ESTRATÉGIA PLDA_??", self)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #0d6efd; letter-spacing: 1px;")

        self.btn_close = QToolButton(self)
        self.btn_close.setText("✕")
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.setStyleSheet("QToolButton { border: none; font-weight: bold; color: #dc3545; font-size: 14px; } QToolButton:hover { background-color: #f8d7da; border-radius: 3px; }")
        self.btn_close.clicked.connect(lambda: self.remove_requested.emit(self))

        header_layout.addWidget(self.lbl_title) 
        header_layout.addStretch()
        header_layout.addWidget(self.btn_close)
        main_layout.addLayout(header_layout)

        # =========================================================
        # 1. ESTRATÉGIA
        # =========================================================
        group_strat = QGroupBox("1. Estratégia e Vetores", self)
        l_strat = QFormLayout(group_strat)
        
        self.combo_strategy = QComboBox(group_strat)
        self.combo_embedding = QComboBox(group_strat)
        
        l_strat.addRow("Método PLDA:", self.combo_strategy)
        l_strat.addRow("Fonte do Vetor:", self.combo_embedding)
        main_layout.addWidget(group_strat)

        # =========================================================
        # 2. DADOS
        # =========================================================
        group_data = QGroupBox("2. Seleção de Dados", self)
        l_data = QFormLayout(group_data)

        self.lbl_train_ds = QLabel("Dataset Treino:", group_data)
        self.combo_train_data = QComboBox(group_data)
        self.lbl_stats_train = QLabel("-", group_data)
        self.lbl_stats_train.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")

        self.lbl_calib_ds = QLabel("Dataset Calibração:", group_data)
        self.combo_calib_data = QComboBox(group_data)
        self.lbl_stats_calib = QLabel("-", group_data)
        self.lbl_stats_calib.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")

        v_train = QVBoxLayout()
        v_train.addWidget(self.combo_train_data)
        v_train.addWidget(self.lbl_stats_train)
        
        v_calib = QVBoxLayout()
        v_calib.addWidget(self.combo_calib_data)
        v_calib.addWidget(self.lbl_stats_calib)

        l_data.addRow(self.lbl_train_ds, v_train)
        l_data.addRow(self.lbl_calib_ds, v_calib)
        
        self.lbl_val_data = QLabel("", group_data)
        self.lbl_val_data.setWordWrap(True)
        self.lbl_val_data.setVisible(False)
        l_data.addRow(self.lbl_val_data)
        main_layout.addWidget(group_data)

        # =========================================================
        # 3. PIPELINES 
        # =========================================================
        group_pipe = QGroupBox("3. Pipelines de Transformação", self) 
        l_pipe = QVBoxLayout(group_pipe)
        form_pipe = QFormLayout()
        
        self.combo_pipe_source = QComboBox(group_pipe) 
        self.btn_edit_source = QToolButton(group_pipe) 
        self.btn_edit_source.setText("✏️") 
        self.btn_edit_source.setEnabled(False) 
        self.btn_edit_source.clicked.connect(self.edit_source_pipeline_requested.emit)

        h_combo_source = QHBoxLayout()
        h_combo_source.setContentsMargins(0, 0, 0, 0)
        h_combo_source.addWidget(self.combo_pipe_source, stretch=1)
        h_combo_source.addWidget(self.btn_edit_source)

        self.lbl_pipe_source_details = QLabel("-", group_pipe) 
        self.lbl_pipe_source_details.setStyleSheet("color: #6c757d; font-size: 10px;")
        v_pipe_source = QVBoxLayout()
        v_pipe_source.addLayout(h_combo_source)
        v_pipe_source.addWidget(self.lbl_pipe_source_details)
        
        self.combo_pipe_target = QComboBox(group_pipe) 
        self.btn_edit_target = QToolButton(group_pipe) 
        self.btn_edit_target.setText("✏️") 
        self.btn_edit_target.setEnabled(False)
        self.btn_edit_target.clicked.connect(self.edit_target_pipeline_requested.emit)

        h_combo_target = QHBoxLayout()
        h_combo_target.setContentsMargins(0, 0, 0, 0)
        h_combo_target.addWidget(self.combo_pipe_target, stretch=1)
        h_combo_target.addWidget(self.btn_edit_target)

        self.lbl_pipe_target_details = QLabel("-", group_pipe) 
        self.lbl_pipe_target_details.setStyleSheet("color: #6c757d; font-size: 10px;")
        v_pipe_target = QVBoxLayout()
        v_pipe_target.addLayout(h_combo_target)
        v_pipe_target.addWidget(self.lbl_pipe_target_details)

        form_pipe.addRow("Source:", v_pipe_source)
        form_pipe.addRow("Target:", v_pipe_target)
        l_pipe.addLayout(form_pipe)

        self.lbl_val_pipe = QLabel("", group_pipe)
        self.lbl_val_pipe.setWordWrap(True)
        self.lbl_val_pipe.setVisible(False)
        l_pipe.addWidget(self.lbl_val_pipe)

        self.btn_run_prep = QPushButton("Executar Preparação", group_pipe)
        self.btn_run_prep.clicked.connect(self.run_prep_requested.emit)
        l_pipe.addWidget(self.btn_run_prep)

        self.btn_visualize = QPushButton("👁️ Visualizar Trajetória 3D", group_pipe)
        self.btn_visualize.clicked.connect(self.visualize_requested.emit)
        self.btn_visualize.setVisible(False)
        l_pipe.addWidget(self.btn_visualize)
        main_layout.addWidget(group_pipe)

        # =========================================================
        # 4. SPLIT E MINERAÇÃO 
        # =========================================================
        group_split = QGroupBox("4. Split e Mineração", self)
        l_split = QVBoxLayout(group_split)
        form_split = QFormLayout()
        
        # --- Combo Split e Botão ---
        self.combo_split_config = QComboBox(group_split)
        self.btn_edit_split = QToolButton(group_split)
        self.btn_edit_split.setText("✏️")
        self.btn_edit_split.setEnabled(False)
        self.btn_edit_split.clicked.connect(self.edit_split_requested.emit)
        
        h_split = QHBoxLayout()
        h_split.setContentsMargins(0,0,0,0)
        h_split.addWidget(self.combo_split_config, stretch=1)
        h_split.addWidget(self.btn_edit_split)

        # --- Combo Mining e Botão ---
        self.combo_mining_config = QComboBox(group_split)
        self.btn_edit_mining = QToolButton(group_split)
        self.btn_edit_mining.setText("✏️")
        self.btn_edit_mining.setEnabled(False)
        self.btn_edit_mining.clicked.connect(self.edit_mining_requested.emit)
        
        h_mining = QHBoxLayout()
        h_mining.setContentsMargins(0,0,0,0)
        h_mining.addWidget(self.combo_mining_config, stretch=1)
        h_mining.addWidget(self.btn_edit_mining)
        
        form_split.addRow("Regra de Split:", h_split)
        form_split.addRow("Regra de Mineração:", h_mining)
        l_split.addLayout(form_split)
        
        self.btn_simulate_split = QPushButton("✂️ Simular Split de Dados", group_split)
        self.btn_simulate_split.clicked.connect(self.simulate_split_requested.emit)
        l_split.addWidget(self.btn_simulate_split)
        
        self.lbl_split_result = QLabel("Aguardando simulação...", group_split)
        self.lbl_split_result.setWordWrap(True)
        self.lbl_split_result.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px; color: #6c757d;")
        l_split.addWidget(self.lbl_split_result)

        main_layout.addWidget(group_split)

        # =========================================================
        # 5. CONFIGURAÇÃO PLDA 
        # =========================================================
        group_plda = QGroupBox("5. Parâmetros PLDA (SpeechBrain)", self)
        l_plda = QFormLayout(group_plda)
        
        self.combo_plda_mode = QComboBox(group_plda)
        self.spin_latent_dim = QSpinBox(group_plda)
        self.spin_latent_dim.setRange(2, 2048)
        self.spin_latent_dim.setValue(128)
        self.combo_plda_mode.currentIndexChanged.connect(self._on_plda_mode_changed)

        self.spin_iter = QSpinBox(group_plda)
        self.spin_iter.setRange(1, 1000)
        self.spin_iter.setValue(20)
        
        self.spin_logreg_c = QDoubleSpinBox(group_plda)
        self.spin_logreg_c.setRange(0.0001, 100.0)
        self.spin_logreg_c.setSingleStep(0.1)
        self.spin_logreg_c.setValue(1.0)

        l_plda.addRow("Modo de Rank:", self.combo_plda_mode)
        l_plda.addRow("Dimensão Latente:", self.spin_latent_dim)
        l_plda.addRow("Iterações EM:", self.spin_iter)
        l_plda.addRow("Calibração C:", self.spin_logreg_c)

        self.lbl_val_config = QLabel("Aguardando preparação...", group_plda)
        self.lbl_val_config.setWordWrap(True)
        self.lbl_val_config.setStyleSheet("background-color: #e9ecef; border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px;")
        l_plda.addRow(self.lbl_val_config)

        self.lbl_val_postprep = QLabel("Aguardando preparação...", group_plda)
        self.lbl_val_postprep.setWordWrap(True)
        self.lbl_val_postprep.setStyleSheet("background-color: #e9ecef; border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px;")
        l_plda.addRow(self.lbl_val_postprep)

        main_layout.addWidget(group_plda)

        # Botão Treinar
        self.btn_train = QPushButton("TREINAR MODELO PLDA", self)
        self.btn_train.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.btn_train.setStyleSheet("""
            QPushButton { 
                background-color: #198754; color: white; font-weight: bold; 
                padding: 10px; border-radius: 4px; font-size: 12px;
            }
            QPushButton:hover { background-color: #157347; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        self.btn_train.setEnabled(False)
        self.btn_train.clicked.connect(self.train_requested.emit)
        main_layout.addWidget(self.btn_train)
        main_layout.addStretch()

    # --- LÓGICA DE UI INTERNA ---
    def _on_plda_mode_changed(self):
        mode = self.combo_plda_mode.currentData()
        if mode:
            is_low_rank = (mode == PLDAMode.LOW_RANK)
            self.spin_latent_dim.setEnabled(is_low_rank)
            if not is_low_rank:
                self.spin_latent_dim.setToolTip("Ignorado no modo Full-Rank.")
            else:
                self.spin_latent_dim.setToolTip("Define o tamanho do vetor de identidade.")
     
    def update_dataset_labels(self, strategy_type: PLDATrainingStrategy):
        if strategy_type is PLDATrainingStrategy.SUPERVISED_ADAPTATION:
                self.lbl_train_ds.setText("Source Domain (Origem):")
                self.lbl_calib_ds.setText("Target Domain (Alvo):")
        else:
            self.lbl_train_ds.setText("Dataset Treino:")
            self.lbl_calib_ds.setText("Dataset Calibração:")

    def set_stats_text(self, label: QLabel, n_imgs: int, n_ids: int):
        label.setText(f"📸 {n_imgs} imgs | 👤 {n_ids} ids" if n_imgs > 0 else "-")
            
    def set_prep_loading(self, loading: bool):
        if loading:
            self.btn_run_prep.setText("Processando...")
            self.btn_run_prep.setEnabled(False)
            self.btn_visualize.setEnabled(False)
            self.btn_simulate_split.setEnabled(False)
        else:
            self.btn_run_prep.setText("Executar Preparação")
            self.btn_run_prep.setEnabled(True)
            self.btn_visualize.setEnabled(True)
            self.btn_simulate_split.setEnabled(True)