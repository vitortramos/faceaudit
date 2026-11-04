"""Confirmação de exclusão de registros, com justificativa para a custódia.

Excluir faces de um dataset pericial é decisão do perito, e a trilha de
auditoria precisa registrar não só quantas saíram, mas POR QUÊ. A
justificativa é obrigatória: o botão Excluir fica desabilitado até haver
texto, e o backend recusa gravar REMOVE_FACES sem ela.

A exclusão é do DATASET, não do disco: nenhum arquivo de imagem é apagado,
e o efeito só é gravado no arquivo no save seguinte.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtWidgets

from .justification_dialog import JustificationDialog


class RemoveFacesDialog(JustificationDialog):
    """Pergunta se confirma a exclusão e colhe a justificativa."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        n_faces: int = 0,
        nome_dataset: str = "",
    ) -> None:
        super().__init__(
            parent,
            titulo_janela="Excluir registros do dataset",
            titulo_html=(
                f"<b>Excluir {n_faces} registro(s)</b>"
                + (f" de <i>{nome_dataset}</i>" if nome_dataset else "")
            ),
            aviso_html=(
                "Os registros saem do dataset. <b>Nenhum arquivo de imagem é "
                "apagado</b> do disco, e a exclusão só é gravada no arquivo "
                "quando você salvar."
            ),
            texto_confirmar="Excluir",
            exemplo=(
                "Ex.: faces com detecção incorreta; imagens duplicadas; "
                "registros fora do escopo do exame."
            ),
        )
