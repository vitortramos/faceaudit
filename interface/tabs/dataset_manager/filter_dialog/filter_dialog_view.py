from PyQt6 import QtWidgets
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QScrollArea, QWidget, QGroupBox, QFormLayout, QDialogButtonBox,
    QCheckBox
)

class FilterDialogView(QDialog):
    """
    View do Diálogo de Filtros.
    Suporta ComboBoxes para campos categóricos (Ground Truth).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configurar Filtros de Auditoria")
        self.resize(500, 800)

        # Armazena referências aos widgets
        self.widgets_map = {}

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. ÁREA DE ROLAGEM
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()

        self.form_container = QVBoxLayout(scroll_content)

        # --- SEÇÃO 1: GROUND TRUTH (TOPO - CRÍTICO) ---
        group_meta = QGroupBox("Filtros Categóricos (Ground Truth)")
        layout_meta = QFormLayout(group_meta)

        # Chaves devem bater com o FilterConfig.
        self._add_combo_row(layout_meta, "identity", "Identidade")
        self._add_combo_row(layout_meta, "gender", "Gênero")
        self._add_combo_row(layout_meta, "ethnicity", "Etnia")
        # Mantemos "source" para compatibilidade com seu FilterConfig atual,
        # mesmo o rótulo sendo "Origem/Fonte".
        self._add_combo_row(layout_meta, "source", "Origem/Fonte")

        self.form_container.addWidget(group_meta)

        # --- SEÇÃO 2: QUALIDADE ---
        self._create_group("Métricas de Qualidade", [
            ("fiqa_score", "FIQA Score"),
            ("ofiq_score", "OFIQ Score"),
            ("serfiq_score", "SER-FIQ Score"),
        ])

        # --- SEÇÃO 3: POSE OFIQ ---
        self._create_group("Pose da Cabeça (OFIQ)", [
            ("ofiq_yaw", "OFIQ Yaw"),
            ("ofiq_pitch", "OFIQ Pitch"),
            ("ofiq_roll", "OFIQ Roll"),
        ])

        # --- SEÇÃO 4: POSE FF ---
        self._create_group("Pose da Cabeça (ForensicFace)", [
            ("ff_yaw", "FF Yaw"),
            ("ff_pitch", "FF Pitch"),
            ("ff_roll", "FF Roll"),
        ])

        # --- SEÇÃO 5: OUTROS ---
        self._create_group("Outras Métricas", [
            ("age", "Idade"),
            ("ipd", "IPD (Dist. Interpupilar)"),
        ])

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 2. BARRA DE PERSISTÊNCIA
        persist_layout = QHBoxLayout()
        self.btn_load_cfg = QtWidgets.QPushButton("📂 Carregar...")
        self.btn_save_cfg = QtWidgets.QPushButton("💾 Salvar...")

        persist_layout.addWidget(self.btn_load_cfg)
        persist_layout.addWidget(self.btn_save_cfg)
        persist_layout.addStretch()
        main_layout.addLayout(persist_layout)

        # 3. BOTÕES APLICAR / CANCELAR
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Aplicar Filtro")

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)

    # --- Helpers de Construção ---

    def _add_combo_row(self, layout, key, label_text):
        """Cria uma linha com Label e ComboBox."""
        combo = QComboBox()
        combo.addItem("Todos")
        self.widgets_map[key] = combo
        layout.addRow(f"<b>{label_text}:</b>", combo)

    def _create_group(self, title, fields):
        group = QGroupBox(title)
        layout = QFormLayout(group)
        for key, label in fields:
            self._add_numeric_row(layout, key, label)
        self.form_container.addWidget(group)

    def _add_numeric_row(self, layout, key_base, label_text):
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        edit_val = QLineEdit(); edit_val.setPlaceholderText("Valor")
        edit_min = QLineEdit(); edit_min.setPlaceholderText("Mín")
        edit_max = QLineEdit(); edit_max.setPlaceholderText("Máx")
        chk_abs = QCheckBox("Abs")
        chk_abs.setToolTip("Considera o valor absoluto (ignora sinal +/-)")

        style = "max-width: 55px;"
        for w in [edit_val, edit_min, edit_max]:
            w.setStyleSheet(style)

        self.widgets_map[key_base] = edit_val
        self.widgets_map[f"min_{key_base}"] = edit_min
        self.widgets_map[f"max_{key_base}"] = edit_max
        self.widgets_map[f"abs_{key_base}"] = chk_abs

        row.addWidget(edit_val)
        row.addWidget(QLabel("ou"))
        row.addWidget(edit_min)
        row.addWidget(QLabel("-"))
        row.addWidget(edit_max)
        row.addSpacing(5)
        row.addWidget(chk_abs)
        row.addStretch()

        layout.addRow(f"{label_text}:", container)

    # --- API de Dados ---

    def populate_combo_options(self, key: str, options: list):
        """Método chamado pelo Controller para encher o dropdown."""
        widget = self.widgets_map.get(key)
        if widget is None or not isinstance(widget, QComboBox):
            return

        widget.blockSignals(True)
        try:
            widget.clear()
            widget.addItem("Todos")
            cleaned = [str(x) for x in options if x is not None and str(x).strip() != ""]
            widget.addItems(sorted(set(cleaned)))
        finally:
            widget.blockSignals(False)

    def set_form_data(self, data_dict: dict):
        """Preenche os campos visuais."""
        for key, value in data_dict.items():
            if key not in self.widgets_map:
                continue

            widget = self.widgets_map[key]

            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))

            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "")

            elif isinstance(widget, QComboBox):
                if not value:
                    widget.setCurrentIndex(0)
                else:
                    idx = widget.findText(str(value))
                    widget.setCurrentIndex(idx if idx >= 0 else 0)

    def get_raw_form_data(self) -> dict:
        """Retorna dicionário com os valores da UI."""
        data = {}
        for key, widget in self.widgets_map.items():
            if isinstance(widget, QCheckBox):
                data[key] = widget.isChecked()

            elif isinstance(widget, QLineEdit):
                data[key] = widget.text()

            elif isinstance(widget, QComboBox):
                val = widget.currentText()
                data[key] = None if val == "Todos" else val

        return data
