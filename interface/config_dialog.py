# interface/config_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QWidget, QFileDialog, QGroupBox,
)

from faceaudit.models.system_config import SystemConfig


class ConfigDialog(QDialog):
    """Diálogo de configurações do sistema.

    Escopo enxuto após a Fase 1 da limpeza:
    - **Pastas globais**: `library` (raiz da biblioteca compartilhada entre
      projetos) e `dir_templates` (templates de ingestão do diálogo Novo
      Dataset).
    - **GPU**: capability de hardware do ForensicFace.
    - **OFIQ**: caminho do executável e da pasta de dados.

    Parâmetros operacionais (modelo de embedding, det_size, T/alpha do
    SER-FIQ) NÃO estão aqui — vivem no `DatasetIngestionConfig` por
    execução, configurados nos diálogos Novo Dataset / Enriquecer.
    """

    def __init__(self, config: SystemConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Configurações do Sistema")
        self.setMinimumWidth(560)
        self._init_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ─── Pastas globais ─────────────────────────────────────────────
        gb_paths = QGroupBox("📁 Pastas globais")
        form_paths = QFormLayout(gb_paths)

        w_lib, self.input_library = self._make_path_selector(self.config.paths.library)
        form_paths.addRow("Biblioteca:", w_lib)

        w_tmpl, self.input_dir_templates = self._make_path_selector(
            self.config.paths.dir_templates
        )
        form_paths.addRow("Templates de ingestão:", w_tmpl)

        note = QLabel(
            "<i>A biblioteca é uma única pasta-raiz que contém datasets, "
            "pipeline_configs, mining_configs, split_configs e "
            "strategy_templates. As subpastas são criadas automaticamente.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 11px;")
        form_paths.addRow(note)

        layout.addWidget(gb_paths)

        # ─── Hardware (GPU) ─────────────────────────────────────────────
        gb_hw = QGroupBox("🖥️ Hardware")
        form_hw = QFormLayout(gb_hw)
        self.chk_use_gpu = QCheckBox("Usar GPU (CUDA) no ForensicFace")
        self.chk_use_gpu.setChecked(self.config.ff.use_gpu)
        form_hw.addRow(self.chk_use_gpu)
        layout.addWidget(gb_hw)

        # ─── OFIQ (paths externos) ──────────────────────────────────────
        gb_ofiq = QGroupBox("⚙️ OFIQ (executável externo)")
        form_ofiq = QFormLayout(gb_ofiq)
        self.input_ofiq_exe = QLineEdit(self.config.ofiq.ofiq_exe_path or "")
        self.input_ofiq_data = QLineEdit(self.config.ofiq.ofiq_data_path or "")
        form_ofiq.addRow("Executável:", self.input_ofiq_exe)
        form_ofiq.addRow("Pasta de dados:", self.input_ofiq_data)
        layout.addWidget(gb_ofiq)

        # ─── Botões ─────────────────────────────────────────────────────
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self._save_and_close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _make_path_selector(self, initial_text: str):
        widget = QWidget()
        hbox = QHBoxLayout(widget)
        hbox.setContentsMargins(0, 0, 0, 0)
        line_edit = QLineEdit(initial_text or "")
        btn = QPushButton("...")
        btn.setFixedSize(30, 25)
        btn.clicked.connect(lambda: self._browse_folder(line_edit))
        hbox.addWidget(line_edit)
        hbox.addWidget(btn)
        return widget, line_edit

    def _browse_folder(self, line_edit_widget):
        folder = QFileDialog.getExistingDirectory(
            self, "Selecionar Pasta", line_edit_widget.text()
        )
        if folder:
            line_edit_widget.setText(folder)

    # ──────────────────────────────────────────────────────────────────
    # Persistência
    # ──────────────────────────────────────────────────────────────────

    def _save_and_close(self):
        self.config.paths.library = self.input_library.text()
        self.config.paths.dir_templates = self.input_dir_templates.text()

        self.config.ff.use_gpu = self.chk_use_gpu.isChecked()

        self.config.ofiq.ofiq_exe_path = self.input_ofiq_exe.text() or ""
        self.config.ofiq.ofiq_data_path = self.input_ofiq_data.text() or ""

        self.config.save_to_file()
        self.accept()
