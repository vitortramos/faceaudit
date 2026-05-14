# interface/tabs/dataset_manager/column_dialogs/dataset_metadata_editor_dialog.py

"""
Diálogo global de edição de metadados do dataset.

Centraliza num único lugar:
    1. Campos do dataset: name, description, is_aligned.
    2. Tabela bulk de todas as colunas (`column_specs`) com edit de label,
       role, dtype, descrição, visibilidade default, etc.
    3. Sub-painel reativo por role da coluna selecionada:
        - EMBEDDING:    dim, model_name
        - QUALITY:      direction (HIGHER/LOWER), scale (min,max),
                        quality_subtype, metric_name
        - POSE:         axis, unit
        - DEMOGRAPHIC:  is_estimate, vocabulary
        - CUSTOM:       default_value, vocabulary (qualquer STR)
        - STR (qualquer role): vocabulary editável
    4. Botão "Verificar coerência" — relatório de valores fora do
       vocabulário por coluna.

Política de locks:
    - `name` SEMPRE read-only — é o identificador físico no parquet.
      Mudar exige reescrita; fluxo separado fora deste diálogo.
    - `source` SEMPRE read-only — sinaliza a origem (FF/OFIQ/SERFIQ/
      MANUAL/CUSTOM/DERIVED) e não é editável pelo usuário.
    - `role`, `dtype`: editáveis apenas em colunas com `source=CUSTOM`.
      Singletons (IDENTITY, IMAGE_PATH, FILE_HASH) com role travado mesmo
      em CUSTOM — evita criar uma segunda IDENTITY.
    - `label`, `description`, `visible_default`, `editable`,
      `aliases`, `vocabulary`: sempre editáveis (vocab respeita dtype=STR).

Retorna `(dataset_changes, column_changes)` via `get_result()`:
    - dataset_changes: dict com chaves entre {"name","description","is_aligned"}
    - column_changes: dict[col_name → novo ColumnSpec via dataclasses.replace]
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


# Roles disponíveis — uma fonte só. Carregado lazy do enum.
def _column_role_choices() -> List[Tuple[str, str]]:
    """Retorna [(value, label)] de cada role. Label = `.value`
    capitalizado pra UI."""
    from faceaudit.models.column_catalog import ColumnRole
    return [(r.value, r.value.replace("_", " ").title()) for r in ColumnRole]


def _dtype_choices() -> List[Tuple[str, str]]:
    from faceaudit.models.column_catalog import ColumnDType
    return [(d.value, d.value.upper()) for d in ColumnDType]


def _direction_choices() -> List[Tuple[str, str]]:
    from faceaudit.models.column_catalog import QualityDirection
    return [
        (QualityDirection.HIGHER_IS_BETTER.value, "Maior é melhor"),
        (QualityDirection.LOWER_IS_BETTER.value, "Menor é melhor"),
    ]


def _subtype_choices() -> List[Tuple[str, str]]:
    from faceaudit.models.column_catalog import QualitySubtype
    return [
        (QualitySubtype.PRIMARY.value, "Primário (score agregado)"),
        (QualitySubtype.AUX.value, "Auxiliar (não no dropdown de balance)"),
        (QualitySubtype.DIMENSION.value, "Dimensão fina"),
    ]


def _axis_choices() -> List[Tuple[str, str]]:
    from faceaudit.models.column_catalog import PoseAxis
    return [
        (PoseAxis.YAW.value, "Yaw (horizontal)"),
        (PoseAxis.PITCH.value, "Pitch (vertical)"),
        (PoseAxis.ROLL.value, "Roll (rotação)"),
    ]


# Singletons cujos roles não podem ser duplicados nem alterados.
_SINGLETON_ROLES = frozenset({"identity", "image_path", "file_hash"})


class DatasetMetadataEditorDialog(QtWidgets.QDialog):
    """Modal de edição completa de metadados do dataset.

    Construtor aceita o `FaceDataset` (somente lido) — o diálogo NÃO
    aplica mudanças no dataset diretamente; quem chamou consulta
    `get_result()` em sucesso e decide como persistir (ex: chamando
    `model.update_column_spec`, escrevendo no campo do dataset, etc.).
    """

    def __init__(self, dataset: Any, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._dataset = dataset
        # Snapshot inicial dos specs — mutado in-place enquanto o usuário edita.
        # Em "OK" calculamos o diff vs. snapshot pra emitir só o que mudou.
        from copy import deepcopy
        original_specs = dict(getattr(dataset, "column_specs", {}) or {})
        # Cada entrada do _specs é um spec mutável (replace cria nova versão).
        self._specs: Dict[str, Any] = deepcopy(original_specs)
        self._original_specs: Dict[str, Any] = original_specs
        self._current_col_name: Optional[str] = None

        self.setWindowTitle(f"Metadados — {getattr(dataset, 'name', 'Dataset')}")
        self.setModal(True)
        self.resize(1100, 720)
        self._build_ui()
        self._populate_dataset_section()
        self._populate_columns_table()
        if self._specs:
            # Seleciona primeira coluna pra usuário não ver painel vazio.
            self._table_columns.selectRow(0)

    # ── API pública ───────────────────────────────────────────────────────

    def get_result(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Retorna `(dataset_changes, column_changes)` aplicáveis pelo caller.

        - `dataset_changes`: subset de {"name", "description", "is_aligned"}
          contendo apenas os campos que mudaram. Vazio se nada mudou.
        - `column_changes`: dict {col_name → novo ColumnSpec} contendo
          apenas colunas cujo spec mudou. Vazio se nada mudou.
        """
        ds_changes: Dict[str, Any] = {}

        new_name = self.txt_name.text().strip()
        if new_name and new_name != getattr(self._dataset, "name", ""):
            ds_changes["name"] = new_name

        new_desc = self.txt_description.toPlainText().strip() or None
        if new_desc != getattr(self._dataset, "description", None):
            ds_changes["description"] = new_desc

        new_is_aligned = bool(self.chk_is_aligned.isChecked())
        if new_is_aligned != bool(getattr(self._dataset, "is_aligned", False)):
            ds_changes["is_aligned"] = new_is_aligned

        col_changes: Dict[str, Any] = {}
        for name, spec in self._specs.items():
            original = self._original_specs.get(name)
            if original is None or spec is original:
                # Sem original ou mesmo objeto (não replaceado) → não mudou.
                continue
            if not _specs_equal(spec, original):
                col_changes[name] = spec

        return ds_changes, col_changes

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        self._tabs = QtWidgets.QTabWidget(parent=self)
        root.addWidget(self._tabs, 1)

        # ── Tab 1: Dataset ──────────────────────────────────────────────
        ds_tab = QtWidgets.QWidget(parent=self._tabs)
        ds_layout = QtWidgets.QFormLayout(ds_tab)
        ds_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ds_layout.setVerticalSpacing(8)
        ds_layout.setHorizontalSpacing(10)

        self.txt_name = QtWidgets.QLineEdit(parent=ds_tab)
        self.txt_name.setToolTip(
            "Nome do dataset, gravado em `faceaudit:dataset_name` no parquet."
        )
        ds_layout.addRow("Nome:", self.txt_name)

        self.txt_description = QtWidgets.QPlainTextEdit(parent=ds_tab)
        self.txt_description.setMaximumHeight(120)
        self.txt_description.setPlaceholderText("Descrição livre do dataset...")
        ds_layout.addRow("Descrição:", self.txt_description)

        self.chk_is_aligned = QtWidgets.QCheckBox(
            "Imagens já alinhadas (112×112 — pula detect+align no FFService)",
            parent=ds_tab,
        )
        self.chk_is_aligned.setToolTip(
            "Marque quando as imagens em `image_path` já são crops 112×112 "
            "pós-alinhamento (saída típica do Image Pipeline com "
            "input_source=ALIGNED_FACE). Re-enriquecimentos vão direto pra "
            "extração de embedding sem detect+align."
        )
        ds_layout.addRow("", self.chk_is_aligned)

        info_lbl = QtWidgets.QLabel(
            "<b>Path mode:</b> {} {}".format(
                getattr(getattr(self._dataset, "path_mode", None), "value", "n/a"),
                "(`{}`)".format(getattr(self._dataset, "path_root", "") or "")
                if getattr(getattr(self._dataset, "path_mode", None), "value", None) == "relative"
                else "",
            ),
            parent=ds_tab,
        )
        info_lbl.setStyleSheet("color: #666; font-size: 11px;")
        info_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        info_lbl.setWordWrap(True)
        ds_layout.addRow("", info_lbl)
        hint_lbl = QtWidgets.QLabel(
            "Para alterar `path_mode`/`path_root` use o botão "
            "<b>Localizar</b> no ribbon (operação separada que migra os "
            "image_paths).",
            parent=ds_tab,
        )
        hint_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        hint_lbl.setWordWrap(True)
        ds_layout.addRow("", hint_lbl)

        self._tabs.addTab(ds_tab, "Dataset")

        # ── Tab 2: Colunas ──────────────────────────────────────────────
        cols_tab = QtWidgets.QWidget(parent=self._tabs)
        cols_layout = QtWidgets.QVBoxLayout(cols_tab)
        cols_layout.setContentsMargins(0, 0, 0, 0)
        cols_layout.setSpacing(4)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, parent=cols_tab)
        splitter.setChildrenCollapsible(False)

        # Tabela à esquerda
        self._table_columns = QtWidgets.QTableWidget(parent=splitter)
        self._table_columns.setColumnCount(5)
        self._table_columns.setHorizontalHeaderLabels(
            ["Nome", "Rótulo", "Role", "Source", "Tipo"]
        )
        self._table_columns.verticalHeader().setVisible(False)
        self._table_columns.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table_columns.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table_columns.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table_columns.setAlternatingRowColors(True)
        self._table_columns.setStyleSheet(
            "QTableWidget { font-size: 11px; gridline-color: #e0e0e0;"
            " alternate-background-color: #f8f9fa; }"
            "QHeaderView::section { background: #eef2f7; padding: 4px 6px;"
            " border: none; border-bottom: 1px solid #c0c8d0; font-weight: bold; }"
        )
        self._table_columns.itemSelectionChanged.connect(
            self._on_column_selected
        )

        # Painel direito — propriedades editáveis da coluna selecionada
        self._props_panel = self._build_props_panel(parent=splitter)

        splitter.addWidget(self._table_columns)
        splitter.addWidget(self._props_panel)
        splitter.setSizes([520, 540])

        cols_layout.addWidget(splitter, 1)
        self._tabs.addTab(cols_tab, "Colunas")

        # ── Bottom buttons ──────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_coherence = QtWidgets.QPushButton(
            "🔍 Verificar coerência", parent=self,
        )
        self.btn_coherence.setToolTip(
            "Lista quantos valores estão fora do vocabulário em cada coluna "
            "STR com vocab definido. Política tolerante — valores fora-de-vocab "
            "permanecem aceitos, só ficam flagados visualmente na tabela."
        )
        self.btn_coherence.clicked.connect(self._on_check_coherence)
        btn_row.addWidget(self.btn_coherence)
        btn_row.addStretch(1)

        self.btn_cancel = QtWidgets.QPushButton("Cancelar", parent=self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_ok = QtWidgets.QPushButton("Salvar", parent=self)
        self.btn_ok.setDefault(True)
        self.btn_ok.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white;"
            " padding: 5px 18px; border-radius: 3px; font-weight: bold; }"
        )
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_ok)
        root.addLayout(btn_row)

    def _build_props_panel(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Constrói o painel direito com os campos editáveis da coluna ativa."""
        wrap = QtWidgets.QScrollArea(parent=parent)
        wrap.setWidgetResizable(True)
        wrap.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        host = QtWidgets.QWidget(parent=wrap)
        form = QtWidgets.QFormLayout(host)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setVerticalSpacing(6)
        form.setHorizontalSpacing(8)
        host.setLayout(form)

        # Universais
        self.props_lbl_name = self._readonly_label("—")
        form.addRow("Nome:", self.props_lbl_name)

        self.props_txt_label = QtWidgets.QLineEdit(parent=host)
        self.props_txt_label.editingFinished.connect(
            lambda: self._apply_field("label", self.props_txt_label.text().strip())
        )
        form.addRow("Rótulo:", self.props_txt_label)

        self.props_combo_role = QtWidgets.QComboBox(parent=host)
        for value, label in _column_role_choices():
            self.props_combo_role.addItem(label, userData=value)
        self.props_combo_role.currentIndexChanged.connect(self._on_role_changed)
        form.addRow("Role:", self.props_combo_role)

        self.props_lbl_source = self._readonly_label("—")
        form.addRow("Source:", self.props_lbl_source)

        self.props_combo_dtype = QtWidgets.QComboBox(parent=host)
        for value, label in _dtype_choices():
            self.props_combo_dtype.addItem(label, userData=value)
        self.props_combo_dtype.currentIndexChanged.connect(self._on_dtype_changed)
        form.addRow("Tipo:", self.props_combo_dtype)

        self.props_txt_category = QtWidgets.QLineEdit(parent=host)
        self.props_txt_category.editingFinished.connect(
            lambda: self._apply_field(
                "category", self.props_txt_category.text().strip() or None,
            )
        )
        form.addRow("Categoria:", self.props_txt_category)

        self.props_txt_description = QtWidgets.QPlainTextEdit(parent=host)
        self.props_txt_description.setMaximumHeight(60)
        self.props_txt_description.textChanged.connect(
            lambda: self._apply_field(
                "description",
                self.props_txt_description.toPlainText().strip() or None,
            )
        )
        form.addRow("Descrição:", self.props_txt_description)

        self.props_chk_visible_default = QtWidgets.QCheckBox(
            "Visível ao abrir", parent=host,
        )
        self.props_chk_visible_default.toggled.connect(
            lambda checked: self._apply_field("visible_default", bool(checked))
        )
        form.addRow("", self.props_chk_visible_default)

        self.props_chk_editable = QtWidgets.QCheckBox(
            "Editável célula a célula", parent=host,
        )
        self.props_chk_editable.toggled.connect(
            lambda checked: self._apply_field("editable", bool(checked))
        )
        form.addRow("", self.props_chk_editable)

        # Aliases
        form.addRow(self._section_header("Aliases (nomes legados)"))
        self.props_txt_aliases = QtWidgets.QPlainTextEdit(parent=host)
        self.props_txt_aliases.setMaximumHeight(60)
        self.props_txt_aliases.setPlaceholderText(
            "Um por linha. Aceitos como sinônimos do nome canônico ao "
            "ler parquets antigos."
        )
        self.props_txt_aliases.textChanged.connect(self._on_aliases_changed)
        form.addRow("", self.props_txt_aliases)

        # Sub-painel reativo — visível conforme role/dtype.
        self._role_box = QtWidgets.QGroupBox(
            "Propriedades específicas do role", parent=host,
        )
        self._role_box.setStyleSheet(
            "QGroupBox { font-size: 11px; font-weight: bold;"
            " border: 1px solid #ccc; border-radius: 4px; margin-top: 8px;"
            " padding-top: 12px; }"
            "QGroupBox::title { left: 8px; subcontrol-position: top left; }"
        )
        self._role_form = QtWidgets.QFormLayout(self._role_box)
        self._role_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._role_form.setVerticalSpacing(6)
        form.addRow(self._role_box)

        self._build_role_specific_widgets(self._role_box)

        # Vocabulário (qualquer STR — pode coexistir com fields role-specific).
        form.addRow(self._section_header("Vocabulário (valores aceitos — STR)"))

        self._vocab_hint = QtWidgets.QLabel(
            "Lista um valor por linha. Vazia = qualquer valor é aceito. "
            "Valores fora desta lista permanecem aceitos mas são flagados "
            "visualmente na tabela.",
            parent=host,
        )
        self._vocab_hint.setStyleSheet("color: #666; font-size: 11px;")
        self._vocab_hint.setWordWrap(True)
        form.addRow("", self._vocab_hint)

        self.props_txt_vocab = QtWidgets.QPlainTextEdit(parent=host)
        self.props_txt_vocab.setMaximumHeight(120)
        self.props_txt_vocab.setPlaceholderText("Masculino\nFeminino\nNão Informado")
        self.props_txt_vocab.textChanged.connect(self._on_vocab_changed)
        form.addRow("", self.props_txt_vocab)

        # CUSTOM: default_value
        self.props_lbl_default = QtWidgets.QLabel("Valor padrão:", parent=host)
        self.props_txt_default = QtWidgets.QLineEdit(parent=host)
        self.props_txt_default.setPlaceholderText("(vazio = nulo)")
        self.props_txt_default.editingFinished.connect(
            lambda: self._apply_field(
                "default_value", self.props_txt_default.text() or None,
            )
        )
        form.addRow(self.props_lbl_default, self.props_txt_default)

        wrap.setWidget(host)
        return wrap

    def _build_role_specific_widgets(self, parent: QtWidgets.QWidget) -> None:
        """Cria todos os widgets role-specific. Ficam ocultos quando o
        role corrente não corresponde — toggles em `_refresh_role_visibility`."""
        # QUALITY
        self._q_direction = QtWidgets.QComboBox(parent=parent)
        for value, label in _direction_choices():
            self._q_direction.addItem(label, userData=value)
        self._q_direction.currentIndexChanged.connect(
            lambda _: self._apply_field("direction",
                                         self._coerce_direction(self._q_direction.currentData()))
        )
        self._role_form.addRow("Direção (Quality):", self._q_direction)

        self._q_subtype = QtWidgets.QComboBox(parent=parent)
        for value, label in _subtype_choices():
            self._q_subtype.addItem(label, userData=value)
        self._q_subtype.currentIndexChanged.connect(
            lambda _: self._apply_field("quality_subtype",
                                         self._coerce_subtype(self._q_subtype.currentData()))
        )
        self._role_form.addRow("Subtipo (Quality):", self._q_subtype)

        self._q_scale_row = QtWidgets.QWidget(parent=parent)
        scale_layout = QtWidgets.QHBoxLayout(self._q_scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setSpacing(4)
        self._q_scale_min = QtWidgets.QDoubleSpinBox(parent=self._q_scale_row)
        self._q_scale_min.setRange(-1e9, 1e9)
        self._q_scale_min.setDecimals(4)
        self._q_scale_max = QtWidgets.QDoubleSpinBox(parent=self._q_scale_row)
        self._q_scale_max.setRange(-1e9, 1e9)
        self._q_scale_max.setDecimals(4)
        for sb in (self._q_scale_min, self._q_scale_max):
            sb.valueChanged.connect(self._on_scale_changed)
        scale_layout.addWidget(QtWidgets.QLabel("min:"))
        scale_layout.addWidget(self._q_scale_min)
        scale_layout.addWidget(QtWidgets.QLabel("max:"))
        scale_layout.addWidget(self._q_scale_max)
        self._role_form.addRow("Escala (Quality):", self._q_scale_row)

        self._q_metric_name = QtWidgets.QLineEdit(parent=parent)
        self._q_metric_name.editingFinished.connect(
            lambda: self._apply_field(
                "metric_name", self._q_metric_name.text().strip() or None,
            )
        )
        self._role_form.addRow("Métrica (Quality):", self._q_metric_name)

        # POSE
        self._p_axis = QtWidgets.QComboBox(parent=parent)
        for value, label in _axis_choices():
            self._p_axis.addItem(label, userData=value)
        self._p_axis.currentIndexChanged.connect(
            lambda _: self._apply_field(
                "axis", self._coerce_axis(self._p_axis.currentData())
            )
        )
        self._role_form.addRow("Eixo (Pose):", self._p_axis)

        self._p_unit = QtWidgets.QLineEdit(parent=parent)
        self._p_unit.setPlaceholderText("deg")
        self._p_unit.editingFinished.connect(
            lambda: self._apply_field("unit", self._p_unit.text().strip() or None)
        )
        self._role_form.addRow("Unidade (Pose):", self._p_unit)

        # EMBEDDING
        self._e_dim = QtWidgets.QSpinBox(parent=parent)
        self._e_dim.setRange(0, 100_000)
        self._e_dim.setSpecialValueText("(sem)")
        self._e_dim.valueChanged.connect(
            lambda v: self._apply_field("dim", int(v) if v > 0 else None)
        )
        self._role_form.addRow("Dim. embedding:", self._e_dim)

        self._e_model_name = QtWidgets.QLineEdit(parent=parent)
        self._e_model_name.editingFinished.connect(
            lambda: self._apply_field(
                "model_name", self._e_model_name.text().strip() or None,
            )
        )
        self._role_form.addRow("Modelo (Embedding):", self._e_model_name)

        # DEMOGRAPHIC
        self._d_is_estimate = QtWidgets.QCheckBox(
            "Valor é estimativa (não anotação humana)", parent=parent,
        )
        self._d_is_estimate.toggled.connect(
            lambda checked: self._apply_field("is_estimate", bool(checked))
        )
        self._role_form.addRow("", self._d_is_estimate)

    @staticmethod
    def _coerce_direction(value: Any) -> Any:
        if value is None:
            return None
        from faceaudit.models.column_catalog import QualityDirection
        return QualityDirection(value) if isinstance(value, str) else value

    @staticmethod
    def _coerce_subtype(value: Any) -> Any:
        if value is None:
            return None
        from faceaudit.models.column_catalog import QualitySubtype
        return QualitySubtype(value) if isinstance(value, str) else value

    @staticmethod
    def _coerce_axis(value: Any) -> Any:
        if value is None:
            return None
        from faceaudit.models.column_catalog import PoseAxis
        return PoseAxis(value) if isinstance(value, str) else value

    # ── População inicial ────────────────────────────────────────────────

    def _populate_dataset_section(self) -> None:
        self.txt_name.setText(str(getattr(self._dataset, "name", "") or ""))
        self.txt_description.setPlainText(
            str(getattr(self._dataset, "description", "") or "")
        )
        self.chk_is_aligned.setChecked(
            bool(getattr(self._dataset, "is_aligned", False))
        )

    def _populate_columns_table(self) -> None:
        from faceaudit.models.column_catalog import ColumnRole
        self._table_columns.setRowCount(0)
        if not self._specs:
            return
        names = list(self._specs.keys())
        self._table_columns.setRowCount(len(names))
        for row, name in enumerate(names):
            spec = self._specs[name]
            role = getattr(getattr(spec, "role", None), "value", "")
            source = getattr(getattr(spec, "source", None), "value", "")
            dtype = getattr(getattr(spec, "dtype", None), "value", "")
            label = str(getattr(spec, "label", "") or name)

            for col, text in enumerate([name, label, role, source, dtype]):
                item = QtWidgets.QTableWidgetItem(str(text))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                # Singleton: pequena trava visual no role pra deixar claro.
                if col == 2 and role in _SINGLETON_ROLES:
                    item.setToolTip("Role singleton — não pode mudar.")
                self._table_columns.setItem(row, col, item)
        self._table_columns.resizeColumnsToContents()

    def _on_column_selected(self) -> None:
        items = self._table_columns.selectedItems()
        if not items:
            self._current_col_name = None
            self._props_panel.setEnabled(False)
            return
        name = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        self._current_col_name = name
        self._props_panel.setEnabled(True)
        self._load_props_for(name)

    def _load_props_for(self, name: str) -> None:
        """Popula painel direito com o spec atual da coluna `name`."""
        spec = self._specs.get(name)
        if spec is None:
            return

        # Bloqueia signals em massa pra evitar disparar _apply_field a cada
        # setText/setCurrentIndex durante a (re)população.
        with _block_signals(
            self.props_txt_label,
            self.props_combo_role,
            self.props_combo_dtype,
            self.props_txt_category,
            self.props_txt_description,
            self.props_chk_visible_default,
            self.props_chk_editable,
            self.props_txt_aliases,
            self.props_txt_vocab,
            self.props_txt_default,
            self._q_direction,
            self._q_subtype,
            self._q_scale_min,
            self._q_scale_max,
            self._q_metric_name,
            self._p_axis,
            self._p_unit,
            self._e_dim,
            self._e_model_name,
            self._d_is_estimate,
        ):
            self.props_lbl_name.setText(name)
            self.props_txt_label.setText(str(getattr(spec, "label", "") or ""))

            role_value = getattr(getattr(spec, "role", None), "value", "")
            idx = self.props_combo_role.findData(role_value)
            if idx >= 0:
                self.props_combo_role.setCurrentIndex(idx)

            self.props_lbl_source.setText(
                getattr(getattr(spec, "source", None), "value", "—")
            )

            dtype_value = getattr(getattr(spec, "dtype", None), "value", "")
            idx_dt = self.props_combo_dtype.findData(dtype_value)
            if idx_dt >= 0:
                self.props_combo_dtype.setCurrentIndex(idx_dt)

            self.props_txt_category.setText(str(getattr(spec, "category", "") or ""))
            self.props_txt_description.setPlainText(
                str(getattr(spec, "description", "") or "")
            )
            self.props_chk_visible_default.setChecked(
                bool(getattr(spec, "visible_default", True))
            )
            self.props_chk_editable.setChecked(
                bool(getattr(spec, "editable", False))
            )

            aliases = getattr(spec, "aliases", None) or ()
            self.props_txt_aliases.setPlainText("\n".join(aliases))

            vocab = getattr(spec, "vocabulary", None) or ()
            self.props_txt_vocab.setPlainText("\n".join(vocab))

            default = getattr(spec, "default_value", None)
            self.props_txt_default.setText("" if default is None else str(default))

            # Role-specific fields
            self._q_direction.setCurrentIndex(
                max(0, self._q_direction.findData(
                    getattr(getattr(spec, "direction", None), "value", ""),
                ))
            )
            self._q_subtype.setCurrentIndex(
                max(0, self._q_subtype.findData(
                    getattr(getattr(spec, "quality_subtype", None), "value", ""),
                ))
            )
            scale = getattr(spec, "scale", None)
            if scale and len(scale) == 2:
                self._q_scale_min.setValue(float(scale[0]))
                self._q_scale_max.setValue(float(scale[1]))
            else:
                self._q_scale_min.setValue(0.0)
                self._q_scale_max.setValue(0.0)
            self._q_metric_name.setText(str(getattr(spec, "metric_name", "") or ""))

            self._p_axis.setCurrentIndex(
                max(0, self._p_axis.findData(
                    getattr(getattr(spec, "axis", None), "value", ""),
                ))
            )
            self._p_unit.setText(str(getattr(spec, "unit", "") or ""))

            dim = getattr(spec, "dim", None)
            self._e_dim.setValue(int(dim) if isinstance(dim, int) else 0)
            self._e_model_name.setText(str(getattr(spec, "model_name", "") or ""))

            self._d_is_estimate.setChecked(bool(getattr(spec, "is_estimate", False)))

        self._refresh_locks(spec)
        self._refresh_role_visibility(spec)

    # ── Locks ──────────────────────────────────────────────────────────────

    def _refresh_locks(self, spec: Any) -> None:
        """Aplica política de read-only conforme source/role do spec."""
        source = getattr(getattr(spec, "source", None), "value", "")
        role_value = getattr(getattr(spec, "role", None), "value", "")
        is_custom = (source == "custom")

        # `name` e `source` SEMPRE read-only — já são QLabel, sem ação.

        # role e dtype: editáveis só em CUSTOM, e nunca pra singletons.
        is_singleton = role_value in _SINGLETON_ROLES
        self.props_combo_role.setEnabled(is_custom and not is_singleton)
        self.props_combo_dtype.setEnabled(is_custom)

        # `default_value` só faz sentido em CUSTOM.
        self.props_lbl_default.setVisible(is_custom)
        self.props_txt_default.setVisible(is_custom)

    def _refresh_role_visibility(self, spec: Any) -> None:
        """Mostra apenas o sub-painel do role corrente.
        Vocabulary aparece pra qualquer dtype=STR independente do role."""
        role_value = getattr(getattr(spec, "role", None), "value", "")
        dtype_value = getattr(getattr(spec, "dtype", None), "value", "")

        # Cada widget role-specific é envolvido pelo row do form. Achar o
        # row pelo widget e setVisible no row inteiro.
        def _set_row_visible(widget: QtWidgets.QWidget, visible: bool) -> None:
            self._role_form.setRowVisible(widget, visible)

        is_quality = role_value == "quality"
        is_pose = role_value == "pose"
        is_embedding = role_value == "embedding"
        is_demographic = role_value == "demographic"
        is_str = dtype_value == "str"

        _set_row_visible(self._q_direction, is_quality)
        _set_row_visible(self._q_subtype, is_quality)
        _set_row_visible(self._q_scale_row, is_quality)
        _set_row_visible(self._q_metric_name, is_quality)
        _set_row_visible(self._p_axis, is_pose)
        _set_row_visible(self._p_unit, is_pose)
        _set_row_visible(self._e_dim, is_embedding)
        _set_row_visible(self._e_model_name, is_embedding)
        _set_row_visible(self._d_is_estimate, is_demographic)

        any_role_field = is_quality or is_pose or is_embedding or is_demographic
        self._role_box.setVisible(any_role_field)

        # Vocab: visível pra qualquer STR (inclui DEMOGRAPHIC e CUSTOM).
        self.props_txt_vocab.setVisible(is_str)
        self._vocab_hint.setVisible(is_str)

    # ── Edit handlers ──────────────────────────────────────────────────────

    def _apply_field(self, field_name: str, value: Any) -> None:
        """Atualiza o spec ativo via `dataclasses.replace(spec, field=value)`.

        Mantém `_specs` sempre apontando pro spec replace-ado mais recente.
        """
        if not self._current_col_name:
            return
        spec = self._specs.get(self._current_col_name)
        if spec is None:
            return
        try:
            new_spec = dataclasses.replace(spec, **{field_name: value})
        except Exception:
            return  # campo inválido pro dataclass; silenciosa pra robustez
        self._specs[self._current_col_name] = new_spec
        # Atualiza coluna da tabela se for label/role/dtype
        if field_name == "label":
            self._update_table_cell(self._current_col_name, 1, str(value or ""))
        elif field_name == "role":
            v = getattr(value, "value", str(value or ""))
            self._update_table_cell(self._current_col_name, 2, v)
        elif field_name == "dtype":
            v = getattr(value, "value", str(value or ""))
            self._update_table_cell(self._current_col_name, 4, v)

    def _update_table_cell(self, name: str, col: int, text: str) -> None:
        for row in range(self._table_columns.rowCount()):
            item = self._table_columns.item(row, 0)
            if item and item.data(QtCore.Qt.ItemDataRole.UserRole) == name:
                self._table_columns.item(row, col).setText(text)
                return

    def _on_role_changed(self) -> None:
        from faceaudit.models.column_catalog import ColumnRole
        value = self.props_combo_role.currentData()
        if not value:
            return
        self._apply_field("role", ColumnRole(value))
        # Re-atualiza visibilidade dos sub-painéis.
        spec = self._specs.get(self._current_col_name) if self._current_col_name else None
        if spec is not None:
            self._refresh_role_visibility(spec)

    def _on_dtype_changed(self) -> None:
        from faceaudit.models.column_catalog import ColumnDType
        value = self.props_combo_dtype.currentData()
        if not value:
            return
        self._apply_field("dtype", ColumnDType(value))
        spec = self._specs.get(self._current_col_name) if self._current_col_name else None
        if spec is not None:
            self._refresh_role_visibility(spec)

    def _on_aliases_changed(self) -> None:
        raw = self.props_txt_aliases.toPlainText().strip()
        if not raw:
            self._apply_field("aliases", None)
            return
        items = tuple(line.strip() for line in raw.splitlines() if line.strip())
        self._apply_field("aliases", items or None)

    def _on_vocab_changed(self) -> None:
        raw = self.props_txt_vocab.toPlainText().strip()
        if not raw:
            self._apply_field("vocabulary", None)
            return
        items = tuple(line.strip() for line in raw.splitlines() if line.strip())
        # Dedup preservando ordem.
        seen, dedup = set(), []
        for v in items:
            if v not in seen:
                seen.add(v)
                dedup.append(v)
        self._apply_field("vocabulary", tuple(dedup) or None)

    def _on_scale_changed(self) -> None:
        v_min = float(self._q_scale_min.value())
        v_max = float(self._q_scale_max.value())
        if v_min == 0.0 and v_max == 0.0:
            self._apply_field("scale", None)
        else:
            self._apply_field("scale", (v_min, v_max))

    # ── "Verificar coerência" ──────────────────────────────────────────────

    def _on_check_coherence(self) -> None:
        """Lista, por coluna com vocab definido, quantos valores reais não
        estão no vocab. Política tolerante — só relatório, sem bloqueio.

        Lê a fonte mais fresca disponível: prefere o DataFrame em memória
        (faces atuais) via `to_pandas()`. Pode ser custoso em datasets
        grandes; usa wait cursor.
        """
        from PyQt6.QtWidgets import QApplication

        QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            df = self._dataset.to_pandas()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(
                self, "Verificar coerência",
                f"Falha ao montar DataFrame do dataset:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        reports: List[str] = []
        for name, spec in self._specs.items():
            vocab = getattr(spec, "vocabulary", None)
            dtype_value = getattr(getattr(spec, "dtype", None), "value", "")
            if not vocab or dtype_value != "str":
                continue
            if name not in df.columns:
                continue
            vocab_set = set(vocab)
            series = df[name].dropna().astype(str)
            mask = ~series.isin(vocab_set)
            out_of_vocab = series[mask]
            if out_of_vocab.empty:
                reports.append(f"✅ <b>{name}</b>: 0 fora do vocabulário.")
                continue
            counts = out_of_vocab.value_counts()
            top = counts.head(5)
            top_str = ", ".join(
                f"'{v}' ({c})" for v, c in zip(top.index, top.values)
            )
            extra = f" + {len(counts) - len(top)} outros" if len(counts) > len(top) else ""
            total = int(counts.sum())
            reports.append(
                f"⚠️ <b>{name}</b>: {total} valor(es) fora do vocabulário — "
                f"top: {top_str}{extra}"
            )

        if not reports:
            QtWidgets.QMessageBox.information(
                self, "Verificar coerência",
                "Nenhuma coluna STR com vocabulário definido — nada a auditar.",
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Verificar coerência")
        dlg.resize(720, 420)
        layout = QtWidgets.QVBoxLayout(dlg)
        intro = QtWidgets.QLabel(
            "<b>Relatório de coerência de vocabulário</b><br/>"
            "Valores fora do vocabulário permanecem aceitos no dataset "
            "(política tolerante) e são flagados visualmente na tabela.",
            parent=dlg,
        )
        intro.setTextFormat(QtCore.Qt.TextFormat.RichText)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        text = QtWidgets.QTextBrowser(parent=dlg)
        text.setHtml("<br/><br/>".join(reports))
        layout.addWidget(text, 1)

        btn = QtWidgets.QPushButton("Fechar", parent=dlg)
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    # ── Helpers visuais ────────────────────────────────────────────────────

    @staticmethod
    def _readonly_label(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(str(text))
        lbl.setStyleSheet("color: #555;")
        lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        return lbl

    @staticmethod
    def _section_header(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(f"<b>{text}</b>")
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setStyleSheet("color: #2c3e50; margin-top: 6px;")
        return lbl


# ── Util ──────────────────────────────────────────────────────────────────

class _block_signals:
    """Context manager — bloqueia signals de uma coleção de QWidgets."""

    def __init__(self, *widgets: QtWidgets.QWidget):
        self._widgets = widgets
        self._prev: List[bool] = []

    def __enter__(self):
        self._prev = [w.blockSignals(True) for w in self._widgets]
        return self

    def __exit__(self, exc_type, exc, tb):
        for w, prev in zip(self._widgets, self._prev):
            w.blockSignals(prev)


def _specs_equal(a: Any, b: Any) -> bool:
    """Comparação tolerante entre dois ColumnSpec dataclasses. Usa to_dict
    quando disponível pra evitar diff em fields auxiliares; cai em
    dataclasses.asdict como fallback."""
    if a is b:
        return True
    try:
        if hasattr(a, "to_dict") and hasattr(b, "to_dict"):
            return a.to_dict() == b.to_dict()
        return dataclasses.asdict(a) == dataclasses.asdict(b)
    except Exception:
        return False
