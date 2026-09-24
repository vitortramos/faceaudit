"""Confirmação de exclusão de registros, com justificativa para a custódia.

Excluir faces de um dataset pericial é decisão do perito, e a trilha de
auditoria precisa registrar não só quantas saíram, mas POR QUÊ. O campo é
livre e pode ficar vazio — exigir texto levaria a "teste", "limpeza", "ok",
que não informam nada —, mas ele nasce em foco justamente para convidar a
escrever.

A exclusão é do DATASET, não do disco: nenhum arquivo de imagem é apagado,
e o efeito só é gravado no arquivo no save seguinte.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets


class RemoveFacesDialog(QtWidgets.QDialog):
    """Pergunta se confirma a exclusão e colhe a justificativa."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        n_faces: int = 0,
        nome_dataset: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Excluir registros do dataset")
        self.setMinimumWidth(480)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        titulo = QtWidgets.QLabel(
            f"<b>Excluir {n_faces} registro(s)</b>"
            + (f" de <i>{nome_dataset}</i>" if nome_dataset else ""),
            parent=self,
        )
        titulo.setWordWrap(True)
        layout.addWidget(titulo)

        aviso = QtWidgets.QLabel(
            "Os registros saem do dataset. <b>Nenhum arquivo de imagem é "
            "apagado</b> do disco, e a exclusão só é gravada no arquivo "
            "quando você salvar.",
            parent=self,
        )
        aviso.setWordWrap(True)
        layout.addWidget(aviso)

        layout.addWidget(
            QtWidgets.QLabel("Justificativa (vai para a cadeia de custódia):", parent=self)
        )
        self._txt = QtWidgets.QPlainTextEdit(parent=self)
        self._txt.setPlaceholderText(
            "Ex.: faces com detecção incorreta; imagens duplicadas; "
            "registros fora do escopo do exame."
        )
        self._txt.setFixedHeight(80)
        layout.addWidget(self._txt)

        botoes = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("Excluir")
        botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        ).setDefault(True)
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        layout.addWidget(botoes)

        QtCore.QTimer.singleShot(0, self._txt.setFocus)

    def justificativa(self) -> str:
        return self._txt.toPlainText().strip()
