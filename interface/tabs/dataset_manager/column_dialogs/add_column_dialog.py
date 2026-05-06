"""Diálogo para adicionar coluna custom ao FaceDataset.

Coleta nome, label, tipo e valor default. Usado na nova UX "tipo planilha"
do Dataset Manager (PR 8). A coluna criada vira um ColumnSpec com role=custom
+ source=custom; persistido na metadata do parquet quando o usuário salvar.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from PyQt6 import QtCore, QtWidgets


_DTYPE_OPTIONS = [
    ("str", "Texto (string)"),
    ("float", "Número decimal"),
    ("int", "Número inteiro"),
    ("bool", "Verdadeiro/Falso"),
]


class AddColumnDialog(QtWidgets.QDialog):
    """Coleta name, label, dtype e default_value para uma nova coluna custom."""

    def __init__(self, existing_column_names: set, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Coluna")
        self.setModal(True)
        self.resize(420, 260)
        self._existing = set(existing_column_names or set())
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.txt_name = QtWidgets.QLineEdit()
        self.txt_name.setPlaceholderText("ex: minha_anotacao")
        self.txt_name.setToolTip(
            "Nome físico da coluna no parquet. Use snake_case, sem espaços."
        )
        form.addRow("Nome:", self.txt_name)

        self.txt_label = QtWidgets.QLineEdit()
        self.txt_label.setPlaceholderText("ex: Minha Anotação")
        self.txt_label.setToolTip("Rótulo amigável exibido no header da tabela.")
        form.addRow("Rótulo:", self.txt_label)

        self.cmb_dtype = QtWidgets.QComboBox()
        for value, label in _DTYPE_OPTIONS:
            self.cmb_dtype.addItem(label, userData=value)
        self.cmb_dtype.currentIndexChanged.connect(self._on_dtype_changed)
        form.addRow("Tipo:", self.cmb_dtype)

        # Default value: stack que troca entre QLineEdit (str/float/int) e
        # QCheckBox (bool). Mais simples do que tentar normalizar via texto.
        self._default_stack = QtWidgets.QStackedWidget()

        self.txt_default_str = QtWidgets.QLineEdit()
        self.txt_default_str.setPlaceholderText("(vazio = NaN)")
        self._default_stack.addWidget(self.txt_default_str)

        self.spin_default_float = QtWidgets.QDoubleSpinBox()
        self.spin_default_float.setRange(-1e9, 1e9)
        self.spin_default_float.setDecimals(4)
        self.spin_default_float.setValue(0.0)
        self._default_stack.addWidget(self.spin_default_float)

        self.spin_default_int = QtWidgets.QSpinBox()
        self.spin_default_int.setRange(-2_000_000_000, 2_000_000_000)
        self.spin_default_int.setValue(0)
        self._default_stack.addWidget(self.spin_default_int)

        self.chk_default_bool = QtWidgets.QCheckBox("Marcado por padrão")
        self.chk_default_bool.setChecked(False)
        self._default_stack.addWidget(self.chk_default_bool)

        form.addRow("Valor padrão:", self._default_stack)

        layout.addLayout(form)
        layout.addStretch()

        # Botões
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.txt_name.setFocus()

    def _on_dtype_changed(self, idx: int):
        # Reordena pra bater com a ordem dos add → 0=str, 1=float, 2=int, 3=bool
        self._default_stack.setCurrentIndex(idx)

    def _on_accept(self):
        name = self.txt_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Erro", "Nome é obrigatório.")
            return

        # Caracteres válidos: letras/números/underscore. Sem espaços/acentos
        # — vai virar coluna do parquet, vale o conservadorismo.
        if not name.replace("_", "").isalnum():
            QtWidgets.QMessageBox.warning(
                self, "Nome inválido",
                "Use apenas letras, números e underscore. Sem espaços ou acentos.",
            )
            return

        if name in self._existing:
            QtWidgets.QMessageBox.warning(
                self, "Coluna existe",
                f"Já existe uma coluna chamada '{name}'.",
            )
            return

        self.accept()

    # ── API pública ─────────────────────────────────────────────────────────

    def get_result(self) -> Tuple[str, str, str, Any]:
        """Retorna (name, label, dtype_str, default_value)."""
        name = self.txt_name.text().strip()
        label = self.txt_label.text().strip() or name
        dtype = self.cmb_dtype.currentData()

        if dtype == "str":
            txt = self.txt_default_str.text()
            default = txt if txt else None
        elif dtype == "float":
            default = float(self.spin_default_float.value())
        elif dtype == "int":
            default = int(self.spin_default_int.value())
        elif dtype == "bool":
            default = bool(self.chk_default_bool.isChecked())
        else:
            default = None

        return name, label, dtype, default
