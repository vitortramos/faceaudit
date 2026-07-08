# interface/tabs/strategy_manager_v2/strategy_editor/save_validation_rule_template_dialog.py

from __future__ import annotations

import re
from typing import Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet


_TEMPLATE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")


class SaveValidationRuleTemplateDialog(QtWidgets.QDialog):
    """
    Modal disparado pelo 📤 numa linha de regra. Pede ao usuário:
    - `template_id`: nome do arquivo (sem extensão). Validado pra ser
      compatível com filesystem em todos OS (alfanum + underscore + hífen).
    - `name`: rótulo legível exibido no `AddValidationRuleDialog`.
    - `description`: descrição livre (opcional).
    - destino: Global library / Projeto / Ambos. Quando não há projeto
      aberto, força "Global".

    O salvamento físico é feito pelo controller (que tem acesso aos
    storages). Aqui só coletamos os metadados.
    """

    DEST_GLOBAL = "global"
    DEST_PROJECT = "project"
    DEST_BOTH = "both"

    def __init__(
        self,
        has_project: bool,
        default_name: str = "",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Salvar como Template")
        self.setModal(True)
        self.resize(440, 280)
        self._has_project = bool(has_project)
        self._result: Optional[Tuple[str, str, Optional[str], str]] = None
        self._build_ui(default_name)
        self._recompute_ok()

    # ── API ────────────────────────────────────────────────────────────────

    def get_result(self) -> Optional[Tuple[str, str, Optional[str], str]]:
        """
        Retorna `(template_id, name, description, destination)` em aceito.
        `destination` ∈ {"global", "project", "both"}.
        None se cancelado.
        """
        return self._result

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self, default_name: str) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)

        self.input_id = QtWidgets.QLineEdit(parent=self)
        self.input_id.setPlaceholderText("ex: fisher_strict_bfw")
        self.input_id.setText(self._sanitize(default_name))
        self.input_id.setToolTip(
            "Identificador do arquivo no disco — alfanuméricos, _ e - apenas."
        )
        self.input_id.textChanged.connect(self._recompute_ok)
        form.addRow(QtWidgets.QLabel("Identificador *", parent=self), self.input_id)

        self.input_name = QtWidgets.QLineEdit(parent=self)
        self.input_name.setPlaceholderText("ex: Fisher rigoroso BFW")
        self.input_name.setText(default_name)
        self.input_name.setToolTip(
            "Nome legível exibido no diálogo Adicionar Regra."
        )
        self.input_name.textChanged.connect(self._recompute_ok)
        form.addRow(QtWidgets.QLabel("Nome amigável *", parent=self), self.input_name)

        self.input_desc = QtWidgets.QPlainTextEdit(parent=self)
        self.input_desc.setPlaceholderText("Quando usar este template, por quê...")
        self.input_desc.setMaximumHeight(80)
        form.addRow(QtWidgets.QLabel("Descrição", parent=self), self.input_desc)

        root.addLayout(form)

        # Destino
        dest_group = QtWidgets.QGroupBox("Destino", parent=self)
        dest_group.setStyleSheet("QGroupBox { font-size: 11px; }")
        dest_layout = QtWidgets.QVBoxLayout(dest_group)
        dest_layout.setContentsMargins(10, 10, 10, 10)
        dest_layout.setSpacing(4)

        self.rb_global = QtWidgets.QRadioButton(
            "Biblioteca global (disponível em todos os projetos)",
            parent=self,
        )
        self.rb_project = QtWidgets.QRadioButton(
            "Apenas neste projeto", parent=self,
        )
        self.rb_both = QtWidgets.QRadioButton(
            "Ambos", parent=self,
        )
        dest_layout.addWidget(self.rb_global)
        dest_layout.addWidget(self.rb_project)
        dest_layout.addWidget(self.rb_both)

        # Quando não há projeto, força global (e desabilita as outras).
        if self._has_project:
            self.rb_both.setChecked(True)
        else:
            self.rb_global.setChecked(True)
            self.rb_project.setEnabled(False)
            self.rb_both.setEnabled(False)
            self.rb_global.setToolTip(
                "Sem projeto aberto — só é possível salvar na biblioteca global."
            )
        root.addWidget(dest_group)

        # Botões
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QtWidgets.QPushButton("Cancelar", parent=self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_ok = QtWidgets.QPushButton("Salvar", parent=self)
        apply_themed_stylesheet(
            self.btn_ok,
            lambda t: (
                "QPushButton { background: " + t("SUCCESS") + ";"
                " color: " + t("HIGHLIGHT_TEXT") + ";"
                " padding: 5px 18px; border-radius: 3px; font-weight: bold; }"
                "QPushButton:disabled { background: " + t("BORDER") + ";"
                " color: " + t("TEXT_MUTED") + "; }"
            ),
        )
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self.btn_ok)
        root.addLayout(btn_row)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize(text: str) -> str:
        """Converte um nome humano em template_id válido: lowercase +
        substitui não-alfanums por underscore."""
        s = (text or "").strip().lower()
        s = re.sub(r"[^a-z0-9_\-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    def _recompute_ok(self) -> None:
        tid = self.input_id.text().strip()
        name = self.input_name.text().strip()
        valid = bool(tid) and bool(name) and bool(_TEMPLATE_ID_RE.match(tid))
        self.btn_ok.setEnabled(valid)
        # Tooltip explicativo no caso inválido.
        if tid and not _TEMPLATE_ID_RE.match(tid):
            self.input_id.setToolTip(
                "Caracteres inválidos — use apenas letras, números, "
                "underscore (_) e hífen (-)."
            )
        else:
            self.input_id.setToolTip(
                "Identificador do arquivo no disco — alfanuméricos, _ e - apenas."
            )

    def _on_ok(self) -> None:
        if self.rb_global.isChecked():
            dest = self.DEST_GLOBAL
        elif self.rb_project.isChecked():
            dest = self.DEST_PROJECT
        else:
            dest = self.DEST_BOTH

        self._result = (
            self.input_id.text().strip(),
            self.input_name.text().strip(),
            self.input_desc.toPlainText().strip() or None,
            dest,
        )
        self.accept()
