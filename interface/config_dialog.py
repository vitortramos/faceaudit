# interface/tabs/config_dialog.py
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QTabWidget, QWidget, QSpinBox, QDoubleSpinBox, QFileDialog
)
from faceaudit.models.system_config import AppConfig

class ConfigDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Configurações do Sistema")
        self.resize(600, 450) # Aumentei um pouco
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Widget de Abas
        self.tabs = QTabWidget()
        
        # Abas existentes
        self.tab_ff = QWidget()
        self._setup_ff_tab()
        self.tabs.addTab(self.tab_ff, "Forensic Face")
        
        self.tab_ofiq = QWidget()
        self._setup_ofiq_tab()
        self.tabs.addTab(self.tab_ofiq, "OFIQ")
        
        self.tab_serfiq = QWidget()
        self._setup_serfiq_tab()
        self.tabs.addTab(self.tab_serfiq, "SER-FIQ")
        
        # Aba Geral (Atualizada)
        self.tab_geral = QWidget()
        self._setup_general_tab()
        self.tabs.addTab(self.tab_geral, "Diretórios (Paths)")

        layout.addWidget(self.tabs)

        # Botões Salvar/Cancelar
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._save_and_close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ... (MÉTODOS _setup_ff_tab, _setup_ofiq_tab, _setup_serfiq_tab IGUAIS AO ANTERIOR) ...
    # Vou omitir para economizar espaço, mantenha os que você já tem.

    def _setup_ff_tab(self):
        layout = QFormLayout(self.tab_ff)
        self.ff_model = QLineEdit(self.config.ff.model)
        self.ff_det_size = QSpinBox()
        self.ff_det_size.setRange(160, 1024)
        self.ff_det_size.setValue(self.config.ff.det_size)
        self.ff_gpu = QCheckBox("Usar GPU (CUDA)")
        self.ff_gpu.setChecked(self.config.ff.use_gpu)
        layout.addRow("Modelo:", self.ff_model)
        layout.addRow("Det Size:", self.ff_det_size)
        layout.addRow("GPU:", self.ff_gpu)

    def _setup_ofiq_tab(self):
        layout = QFormLayout(self.tab_ofiq)
        self.ofiq_exe = QLineEdit(self.config.ofiq.ofiq_exe_path or "")
        self.ofiq_data = QLineEdit(self.config.ofiq.ofiq_data_path or "")
        layout.addRow("Executável:", self.ofiq_exe)
        layout.addRow("Data/Config:", self.ofiq_data)

    def _setup_serfiq_tab(self):
        layout = QFormLayout(self.tab_serfiq)
        self.serfiq_t = QSpinBox()
        self.serfiq_t.setValue(self.config.serfiq.t)
        self.serfiq_alpha = QDoubleSpinBox()
        self.serfiq_alpha.setValue(self.config.serfiq.alpha)
        layout.addRow("T:", self.serfiq_t)
        layout.addRow("Alpha:", self.serfiq_alpha)

    def _setup_general_tab(self):
        """Configura a aba de diretórios com seletores de pasta."""
        layout = QFormLayout(self.tab_geral)
        
        # Helper para criar linha com botão "Browse"
        def create_path_selector(initial_text):
            widget = QWidget()
            hbox = QHBoxLayout(widget)
            hbox.setContentsMargins(0, 0, 0, 0)
            
            line_edit = QLineEdit(initial_text)
            btn = QPushButton("...")
            btn.setFixedSize(30, 25)
            # Conecta o botão para abrir dialogo e setar texto no line_edit
            btn.clicked.connect(lambda: self._browse_folder(line_edit))
            
            hbox.addWidget(line_edit)
            hbox.addWidget(btn)
            return widget, line_edit

        # 1. Datasets
        w_data, self.input_datasets = create_path_selector(self.config.paths.datasets)
        layout.addRow("Pasta de Datasets:", w_data)
        
        # 2. Pipelines
        w_pipe, self.input_pipelines = create_path_selector(self.config.paths.pipelines)
        layout.addRow("Pasta de Pipelines:", w_pipe)
        
        # 3. Estratégias
        w_strat, self.input_strategies = create_path_selector(self.config.paths.strategies)
        layout.addRow("Pasta de Estratégias:", w_strat)
        
        # Nota explicativa
        lbl_note = QLabel("<i>Nota: As pastas serão criadas automaticamente se não existirem ao salvar.</i>")
        lbl_note.setStyleSheet("color: gray;")
        layout.addRow(lbl_note)

    def _browse_folder(self, line_edit_widget):
        """Abre o seletor de diretório e preenche o campo."""
        folder = QFileDialog.getExistingDirectory(self, "Selecionar Pasta", line_edit_widget.text())
        if folder:
            line_edit_widget.setText(folder)

    def _save_and_close(self):
        """Coleta os dados e salva."""
        # FF
        self.config.ff.model = self.ff_model.text()
        self.config.ff.det_size = self.ff_det_size.value()
        self.config.ff.use_gpu = self.ff_gpu.isChecked()
        
        # OFIQ
        self.config.ofiq.ofiq_exe_path = self.ofiq_exe.text() or None
        self.config.ofiq.ofiq_data_path = self.ofiq_data.text() or None
        
        # SER-FIQ
        self.config.serfiq.t = self.serfiq_t.value()
        self.config.serfiq.alpha = self.serfiq_alpha.value()
        
        # PATHS (Novo)
        self.config.paths.datasets = self.input_datasets.text()
        self.config.paths.pipelines = self.input_pipelines.text()
        self.config.paths.strategies = self.input_strategies.text()
        
        # Persistência
        self.config.save_to_file()
        self.accept()