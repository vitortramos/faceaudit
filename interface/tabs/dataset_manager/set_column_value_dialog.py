"""Diálogo "Definir valor de coluna" — usado pelo menu de contexto da galeria.

Coleta (coluna editável, valor) para aplicar à seleção de faces. A galeria não
tem colunas sob o cursor (ao contrário da tabela), então o usuário escolhe a
coluna num dropdown. O campo de valor se adapta ao TIPO da coluna:

  - vocabulário fechado (gênero, etnia)  → combo com as opções válidas;
  - bool                                 → combo Verdadeiro/Falso (grava bool);
  - int / float                          → campo numérico com dica de formato;
  - texto                                → campo livre.

Devolver o valor já no tipo certo é essencial: o modelo da tabela grava o valor
direto no DataFrame, e atribuir a string "True" a uma coluna booleana (ou texto
a uma coluna numérica) falha silenciosamente.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet


class SetColumnValueDialog(QtWidgets.QDialog):
    """Escolhe uma coluna editável + valor a aplicar a `n_selected` faces.

    `columns`: lista de `(label, key, vocabulary, dtype)` — `dtype` é opcional
    (3-tuplas são aceitas) e, quando presente, é a string do `ColumnDType`
    (`"str"`, `"int"`, `"float"`, `"bool"`, ...). `get_result()` devolve
    `(key, value)` com `value` JÁ no tipo certo, ou `None` se cancelado.
    """

    def __init__(
        self,
        columns: Sequence[Tuple],
        n_selected: int,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Definir valor de coluna")
        # Normaliza para 4-tuplas (label, key, vocabulary, dtype).
        self._columns: List[Tuple] = [
            (c[0], c[1], c[2], (c[3] if len(c) > 3 else None)) for c in columns
        ]
        self._result: Optional[Tuple[str, object]] = None

        form = QtWidgets.QFormLayout(self)

        info = QtWidgets.QLabel(
            f"Aplicar a <b>{n_selected}</b> face(s) selecionada(s):", parent=self)
        info.setWordWrap(True)
        form.addRow(info)

        self.cmb_col = QtWidgets.QComboBox(self)
        for label, key, vocab, dtype in self._columns:
            self.cmb_col.addItem(
                label, (key, tuple(vocab) if vocab else None, dtype))
        form.addRow("Coluna:", self.cmb_col)

        # Widgets de valor (mostro um por vez conforme o tipo da coluna).
        self.cmb_value = QtWidgets.QComboBox(self)        # vocabulário OU bool
        self.txt_value = QtWidgets.QLineEdit(self)        # texto OU numérico
        form.addRow("Valor:", self.cmb_value)
        form.addRow("", self.txt_value)

        self.lbl_hint = QtWidgets.QLabel("", parent=self)
        apply_themed_stylesheet(
            self.lbl_hint,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;",
        )
        form.addRow("", self.lbl_hint)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.cmb_col.currentIndexChanged.connect(self._on_column_changed)
        self._on_column_changed()  # estado inicial coerente

    def _current(self):
        """(key, vocab, dtype) da coluna selecionada."""
        data = self.cmb_col.currentData()
        if not data:
            return (None, None, None)
        return data

    def _on_column_changed(self, *_):
        _key, vocab, dtype = self._current()
        if vocab:
            self.cmb_value.clear()
            self.cmb_value.addItems(list(vocab))
            self._show_combo(hint="")
        elif dtype == "bool":
            self.cmb_value.clear()
            self.cmb_value.addItems(["Verdadeiro", "Falso"])
            self._show_combo(hint="Coluna booleana — escolha verdadeiro ou falso.")
        elif dtype == "int":
            self._show_text(QtGui.QIntValidator(self),
                            "Número inteiro (ex.: 42).")
        elif dtype == "float":
            v = QtGui.QDoubleValidator(self)
            # Notação com ponto independente do locale do SO (não bloqueia "3.14").
            v.setNotation(QtGui.QDoubleValidator.Notation.StandardNotation)
            v.setLocale(QtCore.QLocale(QtCore.QLocale.Language.C))
            self._show_text(v, "Número decimal — use ponto (ex.: 3.14).")
        else:
            self._show_text(None, "Texto livre.")

    def _show_combo(self, *, hint: str):
        self.cmb_value.setVisible(True)
        self.txt_value.setVisible(False)
        self.lbl_hint.setText(hint)

    def _show_text(self, validator, hint: str):
        self.txt_value.setValidator(validator)   # None limpa o validador anterior
        self.txt_value.clear()                   # não vaza valor de um tipo p/ outro
        self.cmb_value.setVisible(False)
        self.txt_value.setVisible(True)
        self.lbl_hint.setText(hint)

    def _on_accept(self):
        key, vocab, dtype = self._current()
        if not key:
            self.reject()
            return
        if vocab:
            value = self.cmb_value.currentText()
        elif dtype == "bool":
            value = (self.cmb_value.currentText() == "Verdadeiro")
        elif dtype in ("int", "float"):
            raw = self.txt_value.text().strip().replace(",", ".")
            if raw == "":
                self._warn("Digite um número.")
                return
            try:
                value = int(raw) if dtype == "int" else float(raw)
            except ValueError:
                self._warn(f"'{self.txt_value.text()}' não é um número válido.")
                return
        else:
            value = self.txt_value.text()
        self._result = (key, value)
        self.accept()

    def _warn(self, msg: str):
        QtWidgets.QMessageBox.warning(self, "Valor inválido", msg)

    def get_result(self) -> Optional[Tuple[str, object]]:
        return self._result
