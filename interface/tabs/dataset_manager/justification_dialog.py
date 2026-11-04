"""Confirmação de uma operação sobre o dataset, com justificativa obrigatória.

Base dos diálogos que colhem o PORQUÊ de uma decisão do perito para a
cadeia de custódia (excluir registros, renomear ou mesclar identidades).
O botão de confirmar fica desabilitado até haver texto não-branco; a regra
em si mora no backend (`audit_log_service.require_justification`), que
recusa gravar a entrada sem ela — o diálogo só não deixa o perito chegar lá
sem escrever.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets


class JustificationDialog(QtWidgets.QDialog):
    """Pergunta se confirma a operação e colhe a justificativa."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        titulo_janela: str,
        titulo_html: str,
        aviso_html: str = "",
        texto_confirmar: str = "Confirmar",
        exemplo: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(titulo_janela)
        self.setMinimumWidth(480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        titulo = QtWidgets.QLabel(titulo_html, parent=self)
        titulo.setWordWrap(True)
        layout.addWidget(titulo)

        if aviso_html:
            aviso = QtWidgets.QLabel(aviso_html, parent=self)
            aviso.setWordWrap(True)
            layout.addWidget(aviso)

        layout.addWidget(QtWidgets.QLabel(
            "Justificativa (obrigatória — vai para a cadeia de custódia):",
            parent=self,
        ))
        self._txt = QtWidgets.QPlainTextEdit(parent=self)
        if exemplo:
            self._txt.setPlaceholderText(exemplo)
        self._txt.setFixedHeight(80)
        layout.addWidget(self._txt)

        self._botoes = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._ok = self._botoes.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self._ok.setText(texto_confirmar)
        self._ok.setToolTip("Escreva a justificativa para confirmar.")
        self._botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        ).setDefault(True)
        self._botoes.accepted.connect(self.accept)
        self._botoes.rejected.connect(self.reject)
        layout.addWidget(self._botoes)

        self._txt.textChanged.connect(self._atualizar_confirmar)
        self._atualizar_confirmar()

        QtCore.QTimer.singleShot(0, self._txt.setFocus)

    def _atualizar_confirmar(self) -> None:
        self._ok.setEnabled(bool(self.justificativa()))

    def pode_confirmar(self) -> bool:
        return self._ok.isEnabled()

    def accept(self) -> None:  # noqa: D401 — Qt override
        # Enter no campo ou atalho não passam por cima do botão desabilitado.
        if not self.justificativa():
            return
        super().accept()

    def set_justificativa(self, texto: str) -> None:
        self._txt.setPlainText(texto)

    def justificativa(self) -> str:
        return self._txt.toPlainText().strip()
