# interface/config_dialog.py
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QWidget, QFileDialog, QGroupBox, QSpinBox, QComboBox,
)

from faceaudit.enums.sepael_model import SepaelModel
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

        # Batch size — controla quantas faces o ForensicFace processa
        # em paralelo numa única chamada ONNX. Maior = mais throughput
        # (especialmente em GPU), porém mais memória. 1 desliga batching.
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 1024)
        self.spin_batch_size.setSingleStep(8)
        self.spin_batch_size.setValue(int(self.config.ff.batch_size or 32))
        self.spin_batch_size.setToolTip(
            "Quantas faces processar em paralelo numa única chamada ONNX "
            "(extração de embedding / FIQA). Maior = mais throughput, "
            "porém mais memória. Típico: 32 em GPU, 8-16 em CPU. Use 1 "
            "para desligar batching (comportamento legado)."
        )
        form_hw.addRow("Batch size (ForensicFace):", self.spin_batch_size)

        # Modelo padrão do ForensicFace — populado dinamicamente com os
        # modelos detectados em `~/.forensicface/models/`. Vazio quando
        # nenhum modelo está instalado (caso primeira execução).
        self.combo_default_model = QComboBox()
        self._available_models = SepaelModel.available_on_disk()
        if not self._available_models:
            # Fallback: mostra todos os membros pra o usuário ver o que
            # precisa instalar. Item desabilitado pra deixar claro que
            # nenhum está disponível.
            self.combo_default_model.addItem(
                "(nenhum modelo instalado em ~/.forensicface/models/)",
                userData=None,
            )
            self.combo_default_model.setEnabled(False)
        else:
            for member in self._available_models:
                self.combo_default_model.addItem(member.label_pt, userData=member.ff_name)
            # Seleciona o `default_model` salvo, ou o primeiro disponível.
            current = self.config.ff.default_model
            target_idx = 0
            if current:
                for i in range(self.combo_default_model.count()):
                    if self.combo_default_model.itemData(i) == current:
                        target_idx = i
                        break
            self.combo_default_model.setCurrentIndex(target_idx)
        self.combo_default_model.setToolTip(
            "Modelo padrão do ForensicFace. Diálogos de Novo/Enriquecer "
            "Dataset herdam essa escolha quando o usuário não trocar "
            "explicitamente. Lista populada com os modelos instalados em "
            "~/.forensicface/models/."
        )
        form_hw.addRow("Modelo padrão:", self.combo_default_model)
        layout.addWidget(gb_hw)

        # ─── OFIQ (paths externos) ──────────────────────────────────────
        gb_ofiq = QGroupBox("⚙️ OFIQ (executável externo)")
        form_ofiq = QFormLayout(gb_ofiq)
        self.input_ofiq_exe = QLineEdit(self.config.ofiq.ofiq_exe_path or "")
        self.input_ofiq_data = QLineEdit(self.config.ofiq.ofiq_data_path or "")
        self.input_ofiq_exe.setPlaceholderText(
            "vazio = procurar automaticamente (PATH e locais padrão)"
        )
        self.input_ofiq_data.setPlaceholderText(
            "vazio = procurar automaticamente (OFIQ_DATA e locais padrão)"
        )
        form_ofiq.addRow("Executável:", self.input_ofiq_exe)
        form_ofiq.addRow("Pasta de dados:", self.input_ofiq_data)
        self.lbl_ofiq_status = QLabel()
        self.lbl_ofiq_status.setWordWrap(True)
        form_ofiq.addRow("", self.lbl_ofiq_status)
        self.input_ofiq_exe.textChanged.connect(self._atualizar_status_ofiq)
        self.input_ofiq_data.textChanged.connect(self._atualizar_status_ofiq)
        self._atualizar_status_ofiq()
        layout.addWidget(gb_ofiq)

        # ─── ExifTool (metadados embutidos) ─────────────────────────────
        gb_exif = QGroupBox("🏷️ ExifTool (executável externo)")
        form_exif = QFormLayout(gb_exif)
        self.input_exiftool = QLineEdit(
            getattr(getattr(self.config, "exiftool", None), "exiftool_exe_path", "")
            or ""
        )
        self.input_exiftool.setPlaceholderText("exiftool (resolve pelo PATH)")
        self.input_exiftool.setToolTip(
            "Caminho do executável do ExifTool. Deixe 'exiftool' para "
            "resolver pelo PATH, ou informe o caminho completo quando o "
            "binário vier de um zip solto (ex.: C:/Exiftool/exiftool.exe). "
            "Vazio desliga a leitura pelo ExifTool — a extração cai no "
            "leitor do Pillow, que lê bem menos."
        )
        form_exif.addRow("Executável:", self.input_exiftool)
        self.lbl_exiftool_status = QLabel()
        self.lbl_exiftool_status.setWordWrap(True)
        form_exif.addRow("", self.lbl_exiftool_status)
        self.input_exiftool.textChanged.connect(self._atualizar_status_exiftool)
        self._atualizar_status_exiftool()
        layout.addWidget(gb_exif)

        # ─── Aparência (tema claro/escuro) ──────────────────────────────
        gb_theme = QGroupBox("🎨 Aparência")
        form_theme = QFormLayout(gb_theme)
        self.combo_theme = QComboBox()
        self.combo_theme.addItem("Claro", userData="light")
        self.combo_theme.addItem("Escuro", userData="dark")
        current_mode = getattr(self.config.theme, "mode", "light")
        idx = self.combo_theme.findData(current_mode)
        self.combo_theme.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_theme.setToolTip(
            "Tema da interface. A troca é aplicada imediatamente ao salvar."
        )
        form_theme.addRow("Tema:", self.combo_theme)
        layout.addWidget(gb_theme)

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

    def _atualizar_status_ofiq(self, *_args) -> None:
        """Mostra ONDE o OFIQ será procurado, e se foi achado.

        Campo vazio não é defeito: significa "resolver automaticamente"
        (PATH, variável de ambiente e locais padrão). Sem dizer isso, dois
        campos em branco pareciam configuração faltando, quando o OFIQ
        podia estar funcionando o tempo todo.

        A resolução é PERGUNTADA ao serviço, não reimplementada aqui — do
        contrário a tela e o comportamento real divergiriam.
        """
        import os

        from faceaudit.services.ofiq_service import OFIQService

        exe_digitado = self.input_ofiq_exe.text().strip()
        data_digitado = self.input_ofiq_data.text().strip()
        exe = OFIQService.resolve_exe_path(exe_digitado)
        data = OFIQService.resolve_data_path(data_digitado)

        partes = []
        if os.path.exists(exe):
            origem = "" if exe_digitado else " (encontrado automaticamente)"
            partes.append(f"✅ Executável: {exe}{origem}")
        else:
            partes.append(f"⚠️ Executável não encontrado em: {exe}")
        if os.path.isdir(data):
            origem = "" if data_digitado else " (encontrada automaticamente)"
            partes.append(f"✅ Pasta de dados: {data}{origem}")
        else:
            partes.append(f"⚠️ Pasta de dados não encontrada em: {data}")

        if any(p.startswith("⚠️") for p in partes):
            partes.append(
                "Sem os dois, a qualidade ISO (OFIQ) fica indisponível na "
                "criação e no enriquecimento."
            )
        self.lbl_ofiq_status.setText("<br>".join(partes))

    def _atualizar_status_exiftool(self, *_args) -> None:
        """Diz se o caminho informado resolve, sem esperar o próximo uso.

        Descobrir que o ExifTool não está no lugar só na hora de criar o
        dataset custa uma ingestão inteira.
        """
        from faceaudit.services.exif_extraction_service import ExifExtractionService

        servico = ExifExtractionService(self.input_exiftool.text().strip())
        if servico.exiftool_available:
            versao = servico.exiftool_version()
            self.lbl_exiftool_status.setText(
                f"✅ ExifTool encontrado (versão {versao})" if versao
                else "✅ ExifTool encontrado"
            )
        else:
            self.lbl_exiftool_status.setText(
                "⚠️ Não encontrado — a extração usará o Pillow, que lê só o "
                "EXIF básico (sem XMP, IPTC ou maker notes)."
            )

    def _save_and_close(self):
        self.config.paths.library = self.input_library.text()
        self.config.paths.dir_templates = self.input_dir_templates.text()

        self.config.ff.use_gpu = self.chk_use_gpu.isChecked()
        self.config.ff.batch_size = int(self.spin_batch_size.value())
        # `default_model` é o ff_name (ex: "sepaelv2") guardado como
        # userData de cada item; None quando nenhum modelo disponível.
        self.config.ff.default_model = self.combo_default_model.currentData()

        self.config.ofiq.ofiq_exe_path = self.input_ofiq_exe.text() or ""
        self.config.ofiq.ofiq_data_path = self.input_ofiq_data.text() or ""

        self.config.exiftool.exiftool_exe_path = self.input_exiftool.text() or ""

        self.config.theme.mode = self.combo_theme.currentData() or "light"

        self.config.save_to_file()
        self.accept()
