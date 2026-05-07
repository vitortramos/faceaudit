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

        # PR 21 (Tema 4.2) — widgets dinâmicos por coluna custom.
        # Cada entrada: col_name → dict {kind, widgets...}
        # kind ∈ {"combo", "numeric"}; widgets dependem do kind.
        self.custom_column_widgets: dict = {}
        self._custom_group = None  # QGroupBox criado on-demand

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

    # =========================================================================
    # COLUNAS CUSTOM (PR 21 — Tema 4.2)
    # =========================================================================

    def populate_custom_columns(self, custom_specs: list):
        """Constrói (ou reconstrói) a seção 'Colunas Custom' na UI a partir
        de uma lista de specs descobertos do `column_specs` do dataset.

        Cada item em `custom_specs` é um dict com:
          - name (str): nome da coluna
          - label (str): label amigável
          - kind (str): "combo" | "numeric"
          - options (list[str], opcional): para "combo" (vocab + valores
            únicos do df)

        Reentrante: chamadas subsequentes destroem o group anterior antes
        de reconstruir. Quando `custom_specs` está vazio, remove a seção.
        """
        # Limpa estado anterior
        if self._custom_group is not None:
            self._custom_group.setParent(None)
            self._custom_group.deleteLater()
            self._custom_group = None
        self.custom_column_widgets.clear()

        if not custom_specs:
            return

        group = QGroupBox("Colunas Custom")
        layout = QFormLayout(group)

        for spec in custom_specs:
            name = spec["name"]
            label = spec.get("label") or name
            kind = spec.get("kind", "combo")

            if kind == "combo":
                combo = QComboBox()
                combo.addItem("Todos")
                opts = spec.get("options", []) or []
                cleaned = [str(x) for x in opts if x is not None and str(x).strip() != ""]
                combo.addItems(sorted(set(cleaned)))
                self.custom_column_widgets[name] = {"kind": "combo", "combo": combo}
                layout.addRow(f"<b>{label}:</b>", combo)

            elif kind == "numeric":
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                edit_min = QLineEdit()
                edit_min.setPlaceholderText("Mín")
                edit_max = QLineEdit()
                edit_max.setPlaceholderText("Máx")
                chk_abs = QCheckBox("Abs")
                chk_abs.setToolTip("Considera o valor absoluto (ignora sinal +/-)")
                style = "max-width: 70px;"
                edit_min.setStyleSheet(style)
                edit_max.setStyleSheet(style)
                row.addWidget(edit_min)
                row.addWidget(QLabel("-"))
                row.addWidget(edit_max)
                row.addSpacing(5)
                row.addWidget(chk_abs)
                row.addStretch()
                self.custom_column_widgets[name] = {
                    "kind": "numeric", "min": edit_min, "max": edit_max, "abs": chk_abs,
                }
                layout.addRow(f"{label}:", container)

        # Insere antes da barra de persistência. form_container só tem
        # QGroupBoxes, então append no final é seguro.
        self.form_container.addWidget(group)
        self._custom_group = group

    def set_custom_data(self, column_filters: dict):
        """Carrega os widgets custom a partir de um dict {col: spec}."""
        if not column_filters:
            return
        for name, spec in column_filters.items():
            entry = self.custom_column_widgets.get(name)
            if entry is None or not spec:
                continue
            if entry["kind"] == "combo":
                value = spec.get("value")
                if value:
                    idx = entry["combo"].findText(str(value))
                    entry["combo"].setCurrentIndex(idx if idx >= 0 else 0)
                else:
                    entry["combo"].setCurrentIndex(0)
            elif entry["kind"] == "numeric":
                c_min = spec.get("min")
                c_max = spec.get("max")
                use_abs = bool(spec.get("abs"))
                entry["min"].setText("" if c_min is None else str(c_min))
                entry["max"].setText("" if c_max is None else str(c_max))
                entry["abs"].setChecked(use_abs)

    def get_custom_data(self) -> dict:
        """Lê os widgets custom e devolve o dict {col: spec} pra
        `FilterConfig.column_filters`. Entradas vazias são omitidas (não
        aparecem no dict — sem filtro pra aquela coluna)."""
        out: dict = {}
        for name, entry in self.custom_column_widgets.items():
            if entry["kind"] == "combo":
                val = entry["combo"].currentText()
                if val and val != "Todos":
                    out[name] = {"value": val}
            elif entry["kind"] == "numeric":
                txt_min = entry["min"].text().strip().replace(",", ".")
                txt_max = entry["max"].text().strip().replace(",", ".")
                use_abs = entry["abs"].isChecked()
                spec: dict = {}
                if txt_min:
                    try:
                        spec["min"] = float(txt_min)
                    except ValueError:
                        pass
                if txt_max:
                    try:
                        spec["max"] = float(txt_max)
                    except ValueError:
                        pass
                if use_abs:
                    spec["abs"] = True
                if spec:
                    out[name] = spec
        return out
