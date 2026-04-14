from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
    QPushButton, QFrame, QSpinBox, QCheckBox, QToolButton, 
    QFormLayout, QGroupBox, QDoubleSpinBox
)

from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.kde_kernel_type import KDEKernelType
from faceaudit.parameters.kde_parameters import KDEParameters

# --- NOVO IMPORT PARA OS DEFAULTS DE AGREGAÇÃO ---
from faceaudit.models.aggregation_config import AggregationConfig

class KDECardView(QFrame):
    
    remove_requested = QtCore.pyqtSignal(object)
    run_prep_requested = QtCore.pyqtSignal()
    train_requested = QtCore.pyqtSignal()
    view_plot_requested = QtCore.pyqtSignal()
    
    edit_mining_requested = QtCore.pyqtSignal()

    def __init__(self, card_index: int, parent=None):
        super().__init__(parent)
        self.card_index = card_index
        
        self.setObjectName("kdeCard")
        self.setStyleSheet("""
            #kdeCard { background-color: white; border: 1px solid #ced4da; border-radius: 8px; margin-bottom: 10px; }
            QGroupBox { font-weight: bold; color: #495057; border: 1px solid #dee2e6; border-radius: 4px; margin-top: 8px; padding-top: 10px; padding-bottom: 5px; }
        """)
        
        self._init_ui()
        self._update_agg_visibility()
        self._update_bw_visibility()

    def _init_ui(self):
        # --- CARREGAMENTO DE DEFAULTS SEGURO ---
        # KDEParameters agora só tem a parte matemática
        kde_defaults = KDEParameters()
        # AggregationConfig tem os padrões de agregação
        agg_defaults = AggregationConfig()
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # =========================================================
        # HEADER E 1. SELEÇÃO DE DATASETS
        # =========================================================
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"ESTRATÉGIA KDE_??", self)
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

        group_data = QGroupBox("1. Seleção de Datasets", self)
        l_data = QFormLayout(group_data)
        l_data.setSpacing(8)
        self.lbl_probe = QLabel("Probe (Questionado):", group_data)
        self.combo_probe_data = QComboBox(group_data)
        self.lbl_stats_probe = QLabel("-", group_data)
        self.lbl_stats_probe.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")
        v_probe = QVBoxLayout()
        v_probe.addWidget(self.combo_probe_data)
        v_probe.addWidget(self.lbl_stats_probe)
        l_data.addRow(self.lbl_probe, v_probe)
        self.lbl_ref = QLabel("Reference (Padrão):", group_data)
        self.combo_ref_data = QComboBox(group_data)
        self.lbl_stats_ref = QLabel("-", group_data)
        self.lbl_stats_ref.setStyleSheet("color: #6c757d; font-size: 10px; margin-left: 2px;")
        v_ref = QVBoxLayout()
        v_ref.addWidget(self.combo_ref_data)
        v_ref.addWidget(self.lbl_stats_ref)
        l_data.addRow(self.lbl_ref, v_ref)
        self.lbl_val_data = QLabel("", group_data)
        self.lbl_val_data.setWordWrap(True)
        self.lbl_val_data.setVisible(False)
        self.lbl_val_data.setStyleSheet("border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px;")
        l_data.addRow(self.lbl_val_data)
        main_layout.addWidget(group_data)

        # =========================================================
        # 2. AGREGAÇÃO DE EMBEDDINGS
        # =========================================================
        group_prep = QGroupBox("2. Agregação de Embeddings", self)
        l_prep = QVBoxLayout(group_prep)
        self.chk_aggregation = QCheckBox("Habilitar Agregação de Embeddings", group_prep)
        
        # Como é um card novo, o DTO aggregation estaria como None no config.
        # Portanto, o comportamento "como antes" (default) é começar desmarcado.
        self.chk_aggregation.setChecked(False) 
        
        l_prep.addWidget(self.chk_aggregation)
        self.w_agg_opts = QWidget(group_prep)
        self.l_agg_opts = QFormLayout(self.w_agg_opts) 
        self.l_agg_opts.setContentsMargins(10, 0, 0, 0)
        self.combo_agg_method = QComboBox(self.w_agg_opts)
        self.l_agg_opts.addRow("Método:", self.combo_agg_method)
        self.spin_group_size = QSpinBox(self.w_agg_opts)
        self.spin_group_size.setRange(2, 50)
        
        # Uso seguro do AggregationConfig
        self.spin_group_size.setValue(agg_defaults.combination_group_size)
        
        self.l_agg_opts.addRow("Tam. Grupo (K):", self.spin_group_size)
        self.chk_weighted = QCheckBox("Usar Ponderação por Qualidade", self.w_agg_opts)
        
        # Uso seguro do AggregationConfig
        self.chk_weighted.setChecked(agg_defaults.use_quality_weighting)
        
        self.l_agg_opts.addRow(self.chk_weighted)
        self.combo_quality_type = QComboBox(self.w_agg_opts)
        self.l_agg_opts.addRow("Tipo de Score:", self.combo_quality_type)
        l_prep.addWidget(self.w_agg_opts)
        self.w_agg_action = QWidget(group_prep)
        l_agg_action = QVBoxLayout(self.w_agg_action)
        l_agg_action.setContentsMargins(0, 5, 0, 0)
        self.lbl_val_prep = QLabel("Aguardando Preparação...", self.w_agg_action)
        self.lbl_val_prep.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_val_prep.setStyleSheet("background-color: #e9ecef; border-radius: 4px; padding: 6px; color: #6c757d;")
        self.lbl_val_prep.setVisible(True)
        l_agg_action.addWidget(self.lbl_val_prep)
        self.btn_run_prep = QPushButton("Executar Agregação", self.w_agg_action)
        self.btn_run_prep.clicked.connect(self.run_prep_requested.emit)
        l_agg_action.addWidget(self.btn_run_prep)
        l_prep.addWidget(self.w_agg_action)
        self.chk_aggregation.toggled.connect(self._update_agg_visibility)
        self.combo_agg_method.currentIndexChanged.connect(self._update_agg_visibility)
        self.chk_weighted.toggled.connect(self._update_agg_visibility)
        main_layout.addWidget(group_prep)

        # =========================================================
        # 3. CONFIGURAÇÃO KDE
        # =========================================================
        group_kde = QGroupBox("3. Parâmetros KDE", self)
        self.l_kde = QFormLayout(group_kde) 
        
        self.combo_mining_config = QComboBox(group_kde)
        
        self.btn_edit_mining = QToolButton(group_kde)
        self.btn_edit_mining.setText("✏️")
        self.btn_edit_mining.setToolTip("Editar Regra Selecionada")
        self.btn_edit_mining.setEnabled(False) 
        self.btn_edit_mining.clicked.connect(self.edit_mining_requested.emit)

        h_mining = QHBoxLayout()
        h_mining.setContentsMargins(0, 0, 0, 0)
        h_mining.setSpacing(2)
        h_mining.addWidget(self.combo_mining_config, 1)
        h_mining.addWidget(self.btn_edit_mining)
        
        self.l_kde.addRow("Regra de Mineração:", h_mining)

        self.combo_kernel = QComboBox(group_kde)
        self.l_kde.addRow("Função Kernel:", self.combo_kernel)
        self.chk_gridsearch = QCheckBox("Otimizar Bandwidth (GridSearch)", group_kde)
        self.chk_gridsearch.setChecked(kde_defaults.use_gridsearch)
        self.l_kde.addRow(self.chk_gridsearch)
        
        self.w_optim_samples = QWidget(group_kde)
        l_optim = QHBoxLayout(self.w_optim_samples)
        l_optim.setContentsMargins(0, 0, 0, 0)
        self.lbl_max_grid = QLabel("Amostra Grid:", self.w_optim_samples)
        self.spin_max_grid = QSpinBox(self.w_optim_samples)
        self.spin_max_grid.setRange(1000, 5_000_000)
        self.spin_max_grid.setValue(kde_defaults.max_samples_for_gridsearch)
        self.spin_max_grid.setSingleStep(1000)
        self.spin_max_grid.setSuffix(" pts")
        self.lbl_max_store = QLabel("Amostra Modelo:", self.w_optim_samples)
        self.spin_max_store = QSpinBox(self.w_optim_samples)
        self.spin_max_store.setRange(1000, 1_000_000)
        self.spin_max_store.setValue(kde_defaults.max_stored_samples)
        self.spin_max_store.setSingleStep(1000)
        self.spin_max_store.setSuffix(" pts")
        l_optim.addWidget(self.lbl_max_grid)
        l_optim.addWidget(self.spin_max_grid)
        l_optim.addSpacing(15)
        l_optim.addWidget(self.lbl_max_store)
        l_optim.addWidget(self.spin_max_store)
        l_optim.addStretch() 
        self.l_kde.addRow("Otimização:", self.w_optim_samples)

        self.w_manual_bw = QWidget(group_kde)
        l_manual = QHBoxLayout(self.w_manual_bw)
        l_manual.setContentsMargins(0, 0, 0, 0)
        self.spin_bw_gen = QDoubleSpinBox(self.w_manual_bw)
        self.spin_bw_gen.setRange(0.001, 5.0)
        self.spin_bw_gen.setSingleStep(0.01)
        self.spin_bw_gen.setValue(kde_defaults.bandwidth_genuines)
        self.spin_bw_gen.setPrefix("Gen: ")
        self.spin_bw_imp = QDoubleSpinBox(self.w_manual_bw)
        self.spin_bw_imp.setRange(0.001, 5.0)
        self.spin_bw_imp.setSingleStep(0.01)
        self.spin_bw_imp.setValue(kde_defaults.bandwidth_impostors)
        self.spin_bw_imp.setPrefix("Imp: ")
        l_manual.addWidget(self.spin_bw_gen)
        l_manual.addWidget(self.spin_bw_imp)
        self.l_kde.addRow("Bandwidth Manual:", self.w_manual_bw)

        self.chk_gridsearch.toggled.connect(self._update_bw_visibility)

        self.lbl_val_config = QLabel("Parâmetros OK", group_kde)
        self.lbl_val_config.setStyleSheet("color: green; font-size: 11px;")
        self.l_kde.addRow(self.lbl_val_config)
        main_layout.addWidget(group_kde)

        # =========================================================
        # 4. AÇÃO DE TREINO E VISUALIZAÇÃO
        # =========================================================
        self.btn_train = QPushButton("CALIBRAR MODELO KDE", self)
        self.btn_train.setIcon(QtGui.QIcon.fromTheme("document-save"))
        self.btn_train.setStyleSheet("QPushButton { background-color: #198754; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px; } QPushButton:hover { background-color: #157347; } QPushButton:disabled { background-color: #6c757d; }")
        self.btn_train.clicked.connect(self.train_requested.emit)
        main_layout.addWidget(self.btn_train)

        self.btn_view_plot = QPushButton("📈 Ver Gráfico da Distribuição", self)
        self.btn_view_plot.setStyleSheet("QPushButton { background-color: #0d6efd; color: white; font-weight: bold; padding: 10px; border-radius: 4px; margin-top: 5px; } QPushButton:hover { background-color: #0b5ed7; }")
        self.btn_view_plot.setVisible(False) 
        self.btn_view_plot.clicked.connect(self.view_plot_requested.emit)
        main_layout.addWidget(self.btn_view_plot)
        main_layout.addStretch()

    # --- Lógica Visual Dinâmica ---

    def _update_agg_visibility(self):
        is_agg = self.chk_aggregation.isChecked()
        self.w_agg_opts.setVisible(is_agg)
        self.w_agg_action.setVisible(is_agg)
        if is_agg:
            current_method = self.combo_agg_method.currentData()
            # Tratamento para quando o combo ainda não foi populado pelo Controller
            is_comb = (current_method == AggregationMethod.COMBINATORIAL) if current_method else False
            
            self.spin_group_size.setVisible(is_comb)
            lbl_group = self.l_agg_opts.labelForField(self.spin_group_size)
            if lbl_group: lbl_group.setVisible(is_comb)
            
            is_weighted = self.chk_weighted.isChecked()
            self.combo_quality_type.setVisible(is_weighted)
            lbl_quality = self.l_agg_opts.labelForField(self.combo_quality_type)
            if lbl_quality: lbl_quality.setVisible(is_weighted)

    def _update_bw_visibility(self):
        is_grid = self.chk_gridsearch.isChecked()
        self.lbl_max_grid.setVisible(is_grid)
        self.spin_max_grid.setVisible(is_grid)
        self.lbl_max_store.setVisible(True)
        self.spin_max_store.setVisible(True)
        self.w_manual_bw.setVisible(not is_grid)
        lbl_bw = self.l_kde.labelForField(self.w_manual_bw)
        if lbl_bw: lbl_bw.setVisible(not is_grid)

    def set_stats_text(self, label_widget, n_imgs, n_ids):
        txt = f"📸 {n_imgs} imgs | 👤 {n_ids} ids" if n_imgs > 0 else "-"
        label_widget.setText(txt)
        
    def set_label_feedback(self, label_widget, severity, text):
        label_widget.setText(text)
        label_widget.setVisible(True)
        bg, color = ("#e9ecef", "#495057")
        if severity.name == "OK": bg, color = "#d1e7dd", "#0f5132"
        elif severity.name == "WARNING": bg, color = "#fff3cd", "#856404"
        elif severity.name == "CRITICAL": bg, color = "#f8d7da", "#842029"
        label_widget.setStyleSheet(f"background-color: {bg}; color: {color}; border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px;")

    def set_prep_loading(self, is_loading: bool):
        if is_loading:
            self.btn_run_prep.setEnabled(False)
            self.btn_run_prep.setText("Processando...")
            self.lbl_val_prep.setText("⏳ Executando agregação...")
            self.lbl_val_prep.setStyleSheet("background-color: #fff3cd; color: #856404; border-radius: 4px; padding: 6px;")
        else:
            self.btn_run_prep.setEnabled(True)
            self.btn_run_prep.setText("Executar Agregação")