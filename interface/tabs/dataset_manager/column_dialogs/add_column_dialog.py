"""Diálogo para adicionar coluna custom ao FaceDataset.

Coleta nome, label, tipo e valor default. Usado na nova UX "tipo planilha"
do Dataset Manager (PR 8). A coluna criada vira um ColumnSpec com role=custom
+ source=custom; persistido na metadata do parquet quando o usuário salvar.

PR 23 (Tema 4.3) — adiciona modo "coluna computada": checkbox + textarea
de fórmula. Quando ativo, esconde o campo de valor padrão (computadas são
read-only, valor calculado on-the-fly por linha). Validação de sintaxe
acontece no Aceitar via `expression_evaluator.validate_expression`.
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
        self.resize(460, 360)
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

        # ── PR 23 (Tema 4.3) — coluna computada ─────────────────────────
        self.chk_computed = QtWidgets.QCheckBox(
            "Coluna computada (fórmula simbólica, read-only)"
        )
        self.chk_computed.setToolTip(
            "Quando marcado, o valor é calculado por linha a partir da "
            "fórmula. Não é editável e não consome espaço no parquet — só "
            "a fórmula é persistida."
        )
        self.chk_computed.toggled.connect(self._on_computed_toggled)
        layout.addWidget(self.chk_computed)

        self._computed_box = QtWidgets.QGroupBox("Fórmula")
        comp_layout = QtWidgets.QVBoxLayout(self._computed_box)
        hint = QtWidgets.QLabel(
            "Use nomes de colunas existentes como variáveis. "
            "Funções permitidas: <code>abs, min, max, round, len, str, int, "
            "float, bool, sqrt, log, exp, floor, ceil</code>.<br>"
            "Operadores: <code>+ - * / // % ** == != &lt; &gt; and or not</code>. "
            "Suporta ternário <code>a if b else c</code>."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666; font-size: 11px;")
        comp_layout.addWidget(hint)

        self.txt_expression = QtWidgets.QPlainTextEdit()
        self.txt_expression.setPlaceholderText(
            "Exemplos:\n"
            "  abs(yaw)\n"
            "  fiqa_score * 100\n"
            "  1 if gender == 'F' else 0\n"
            "  round(sqrt(yaw**2 + pitch**2), 2)"
        )
        self.txt_expression.setMaximumHeight(110)
        comp_layout.addWidget(self.txt_expression)

        self._computed_box.setVisible(False)
        layout.addWidget(self._computed_box)

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

    def _on_computed_toggled(self, checked: bool):
        """Mostra/esconde o box de fórmula. Quando computada, esconde o
        valor padrão (irrelevante — valor é calculado)."""
        self._computed_box.setVisible(checked)
        self._default_stack.setEnabled(not checked)

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

        # PR 23 — valida sintaxe da fórmula antes de aceitar. Erro mostra
        # mensagem amigável; usuário corrige antes de criar a coluna.
        if self.chk_computed.isChecked():
            expr = self.txt_expression.toPlainText().strip()
            if not expr:
                QtWidgets.QMessageBox.warning(
                    self, "Fórmula vazia",
                    "Marque 'Coluna computada' apenas se quiser definir uma fórmula.",
                )
                return
            from faceaudit.services.expression_evaluator import (
                validate_expression, ExpressionError,
            )
            try:
                validate_expression(expr)
            except ExpressionError as e:
                QtWidgets.QMessageBox.warning(
                    self, "Fórmula inválida", str(e),
                )
                return

        self.accept()

    # ── API pública ─────────────────────────────────────────────────────────

    def get_result(self) -> Tuple[str, str, str, Any, Optional[str]]:
        """Retorna (name, label, dtype_str, default_value, computed_expr).

        PR 23 — `computed_expr` é a fórmula simbólica quando a checkbox
        'Coluna computada' está marcada; None caso contrário. Quando
        computed_expr é não-nulo, `default_value` é ignorado pelo model
        (computadas não têm valor armazenado).
        """
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

        computed_expr: Optional[str] = None
        if self.chk_computed.isChecked():
            expr = self.txt_expression.toPlainText().strip()
            computed_expr = expr if expr else None

        return name, label, dtype, default, computed_expr
