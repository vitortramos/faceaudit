from PyQt6 import QtWidgets, QtCore, QtGui

class StrategyManagerUI(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_layout()

    def _init_layout(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # ==========================================================
        # COLUNA DA ESQUERDA (Configurações e Seleção)
        # ==========================================================
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        
        left_content = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_content)
        left_layout.setSpacing(15)
        
        # --- 1. SELEÇÃO DE ESTRATÉGIAS ---
        group_strat = QtWidgets.QGroupBox("1. Estratégias de LR")
        group_strat.setStyleSheet("QGroupBox { font-weight: bold; }")
        hbox_strat = QtWidgets.QHBoxLayout(group_strat)
        hbox_strat.setSpacing(20)
        
        self.chk_plda = QtWidgets.QCheckBox("PLDA")
        self.chk_plda.setToolTip("Probabilistic Linear Discriminant Analysis")
        self.chk_plda.setChecked(True)
        
        self.chk_kde = QtWidgets.QCheckBox("KDE")
        self.chk_kde.setToolTip("Kernel Density Estimation (Cosseno)")
        
        hbox_strat.addWidget(self.chk_plda)
        hbox_strat.addWidget(self.chk_kde)
        hbox_strat.addStretch()
        
        left_layout.addWidget(group_strat)

        # --- 2. SELEÇÃO DE DADOS (6 DATASETS) ---
        group_data = QtWidgets.QGroupBox("2. Seleção de Dados")
        group_data.setStyleSheet("QGroupBox { font-weight: bold; }")
        form_data = QtWidgets.QFormLayout(group_data)
        form_data.setVerticalSpacing(8)
        
        # A. FASE DE APRENDIZADO (MODELAGEM)
        lbl_learning = QtWidgets.QLabel("--- Fase 1: Aprendizado (Construção do Modelo) ---")
        lbl_learning.setStyleSheet("color: #555; font-weight: bold; margin-top: 5px; margin-bottom: 5px;")
        form_data.addRow(lbl_learning)

        self.combo_train = QtWidgets.QComboBox()
        self.combo_train.setToolTip("Dataset de fundo para aprendizado de características (Projeção Latente).")
        form_data.addRow("Treino (Background):", self.combo_train)
        
        self.combo_calib = QtWidgets.QComboBox()
        self.combo_calib.setToolTip("Dataset para calibrar os Scores em Likelihood Ratios.")
        form_data.addRow("Calibração:", self.combo_calib)

        # B. FASE DE VALIDAÇÃO (TESTE DE PERFORMANCE)
        lbl_testing = QtWidgets.QLabel("--- Fase 2: Validação (Avaliação da Estratégia) ---")
        lbl_testing.setStyleSheet("color: #28a745; font-weight: bold; margin-top: 15px; margin-bottom: 5px;")
        form_data.addRow(lbl_testing)

        self.combo_test_probe = QtWidgets.QComboBox()
        self.combo_test_probe.setToolTip("Dataset de teste com faces desconhecidas (Questionadas/Probe).")
        form_data.addRow("Teste - Questionadas (Probe):", self.combo_test_probe)

        self.combo_test_ref = QtWidgets.QComboBox()
        self.combo_test_ref.setToolTip("Dataset de teste com faces desconhecidas (Padrão/Reference).")
        form_data.addRow("Teste - Padrão (Ref):", self.combo_test_ref)

        # C. FASE DE APLICAÇÃO (CASO REAL)
        lbl_case = QtWidgets.QLabel("--- Fase 3: Aplicação Forense (Caso em Análise) ---")
        lbl_case.setStyleSheet("color: #d9534f; font-weight: bold; margin-top: 15px; margin-bottom: 5px;")
        form_data.addRow(lbl_case)

        self.combo_case_probe = QtWidgets.QComboBox()
        self.combo_case_probe.setToolTip("Evidência questionada do caso real.")
        form_data.addRow("Caso - Questionada (Q):", self.combo_case_probe)

        self.combo_case_ref = QtWidgets.QComboBox()
        self.combo_case_ref.setToolTip("Material padrão do caso real.")
        form_data.addRow("Caso - Padrão (K):", self.combo_case_ref)
        
        left_layout.addWidget(group_data)

        # --- 3. PIPELINES DE TRANSFORMAÇÃO ---
        group_trans = QtWidgets.QGroupBox("3. Pré-Processamento (Pipelines)")
        group_trans.setStyleSheet("QGroupBox { font-weight: bold; }")
        form_trans = QtWidgets.QFormLayout(group_trans)
        
        self.combo_pipe_source = QtWidgets.QComboBox()
        self.combo_pipe_source.setToolTip("Aplicado ao Dataset de Treino (Source). Geralmente inclui CORAL.")
        
        self.combo_pipe_target = QtWidgets.QComboBox()
        self.combo_pipe_target.setToolTip("Aplicado a TODOS os outros datasets (Target).")

        self.btn_manage_pipelines = QtWidgets.QPushButton("⚙️ Configurar Pipelines")
        self.btn_manage_pipelines.setFlat(True)
        self.btn_manage_pipelines.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_manage_pipelines.setStyleSheet("text-align: right; color: #0078d7;")

        form_trans.addRow("Pipeline Source (Treino):", self.combo_pipe_source)
        form_trans.addRow("Pipeline Target (Outros):", self.combo_pipe_target)
        form_trans.addRow("", self.btn_manage_pipelines)
        
        left_layout.addWidget(group_trans)

        # --- BOTÃO DE AÇÃO ---
        self.btn_run_all = QtWidgets.QPushButton("🚀 EXECUTAR FLUXO COMPLETO")
        self.btn_run_all.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_run_all.setStyleSheet("""
            QPushButton {
                background-color: #0078d7; 
                color: white; 
                font-weight: bold; 
                font-size: 14px; 
                height: 60px; 
                border-radius: 6px;
                margin-top: 10px;
            }
            QPushButton:hover { background-color: #0063b1; }
            QPushButton:pressed { background-color: #004f8e; }
        """)
        left_layout.addWidget(self.btn_run_all)
        
        left_layout.addStretch()
        
        # Adiciona o conteúdo ao scroll area
        left_scroll.setWidget(left_content)

        # ==========================================================
        # COLUNA DA DIREITA (Logs e Resultados)
        # ==========================================================
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_log = QtWidgets.QLabel("Log de Execução e Resultados:")
        lbl_log.setStyleSheet("font-weight: bold;")
        
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: Consolas, Monospace; font-size: 12px; background-color: #f9f9f9;")
        self.txt_log.setPlaceholderText("Aguardando execução...\n\nO sistema irá:\n1. Treinar Pipelines\n2. Treinar Modelos (PLDA/KDE)\n3. Avaliar Performance (AUC/EER) nos dados de Teste\n4. Calcular LR para o Caso em Análise")
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar { height: 20px; border-radius: 4px; text-align: center; } QProgressBar::chunk { background-color: #0078d7; }")

        right_layout.addWidget(lbl_log)
        right_layout.addWidget(self.txt_log)
        right_layout.addWidget(self.progress_bar)

        # ==========================================================
        # MONTAGEM FINAL
        # ==========================================================
        main_layout.addWidget(left_scroll, 2) # Aumentei um pouco a esquerda (2/5)
        main_layout.addWidget(right_panel, 3) # (3/5)