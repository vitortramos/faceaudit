"""Opções do "Salvar Como" do Dataset Manager.

Duas decisões que o app antes tomava sozinho — e nas duas escolhia mal:

1. **Como levar as imagens** quando o destino é outra pasta. `path_root` é
   relativo à pasta DO PARQUET, então gravar noutra pasta sem tratar os
   caminhos reancora as imagens onde elas não estão. A pergunta é feita em
   termos de consequência (portátil × referenciado), não de "relativo ×
   absoluto", que é a representação interna.

2. **Qual dataset continua aberto** quando o que está na tela é resultado de
   um filtro. O padrão passa a ser continuar no dataset de origem: o fluxo
   típico é derivar VÁRIOS recortes do mesmo dataset grande, e trocar para o
   recorte recém-salvo obrigava a fechar e reabrir o grande a cada um.

Cada grupo só aparece quando a decisão existe: salvando na mesma pasta não
há o que decidir sobre imagens; sem filtro ativo não há dois datasets.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets


class SaveAsOptionsDialog(QtWidgets.QDialog):
    """Coleta as opções do Salvar Como numa única tela."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        ask_images: bool = False,
        ask_which_stays: bool = False,
        nome_destino: str = "",
        nome_origem: str = "",
        n_salvar: int = 0,
        n_origem: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Salvar Como — opções")
        self.setMinimumWidth(520)

        self._rb_portatil: Optional[QtWidgets.QRadioButton] = None
        self._rb_manter_origem: Optional[QtWidgets.QRadioButton] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        if ask_images:
            layout.addWidget(self._grupo_imagens(nome_destino))
        if ask_which_stays:
            layout.addWidget(
                self._grupo_dataset_aberto(
                    nome_destino, nome_origem, n_salvar, n_origem,
                )
            )

        botoes = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("Salvar")
        botoes.accepted.connect(self.accept)
        botoes.rejected.connect(self.reject)
        layout.addWidget(botoes)

    # ── Grupos ───────────────────────────────────────────────────────────

    def _grupo_imagens(self, nome_destino: str) -> QtWidgets.QGroupBox:
        grupo = QtWidgets.QGroupBox("Imagens", parent=self)
        v = QtWidgets.QVBoxLayout(grupo)

        aviso = QtWidgets.QLabel(
            "Você está salvando numa pasta diferente da atual.", parent=grupo,
        )
        aviso.setWordWrap(True)
        v.addWidget(aviso)

        self._rb_portatil = QtWidgets.QRadioButton(
            "Copiar as imagens (autocontido)", parent=grupo,
        )
        self._rb_portatil.setChecked(True)
        detalhe_portatil = QtWidgets.QLabel(
            f"Cria a pasta <i>{nome_destino}/</i> com o arquivo e as imagens "
            "dentro. A pasta inteira pode ser movida, compactada ou levada "
            "para outra máquina sem quebrar nada.",
            parent=grupo,
        )
        detalhe_portatil.setWordWrap(True)
        detalhe_portatil.setIndent(22)

        rb_ref = QtWidgets.QRadioButton(
            "Referenciar as imagens onde estão", parent=grupo,
        )
        detalhe_ref = QtWidgets.QLabel(
            "Grava os caminhos absolutos das imagens originais. Rápido e sem "
            "duplicar disco, mas o dataset só abre as imagens em máquinas "
            "que enxerguem esses caminhos.",
            parent=grupo,
        )
        detalhe_ref.setWordWrap(True)
        detalhe_ref.setIndent(22)

        for w in (self._rb_portatil, detalhe_portatil, rb_ref, detalhe_ref):
            v.addWidget(w)
        return grupo

    def _grupo_dataset_aberto(
        self, nome_destino: str, nome_origem: str, n_salvar: int, n_origem: int,
    ) -> QtWidgets.QGroupBox:
        grupo = QtWidgets.QGroupBox(
            "Qual dataset continua aberto", parent=self,
        )
        v = QtWidgets.QVBoxLayout(grupo)

        self._rb_manter_origem = QtWidgets.QRadioButton(
            f"{nome_origem}  —  {n_origem:,} faces".replace(",", "."),
            parent=grupo,
        )
        self._rb_manter_origem.setChecked(True)
        detalhe_origem = QtWidgets.QLabel(
            "O recorte é gravado no disco e você continua no dataset de "
            "origem, pronto para aplicar outro filtro e salvar de novo.",
            parent=grupo,
        )
        detalhe_origem.setWordWrap(True)
        detalhe_origem.setIndent(22)

        rb_salvo = QtWidgets.QRadioButton(
            f"{nome_destino}  —  {n_salvar:,} faces (o que você acabou de "
            "filtrar)".replace(",", "."),
            parent=grupo,
        )
        detalhe_salvo = QtWidgets.QLabel(
            "Abre o recorte recém-salvo. O dataset de origem sai da tela.",
            parent=grupo,
        )
        detalhe_salvo.setWordWrap(True)
        detalhe_salvo.setIndent(22)

        for w in (self._rb_manter_origem, detalhe_origem, rb_salvo, detalhe_salvo):
            v.addWidget(w)
        return grupo

    # ── Resultado ────────────────────────────────────────────────────────

    def portable(self) -> Optional[bool]:
        """True = copiar imagens; False = referenciar; None = não perguntado."""
        if self._rb_portatil is None:
            return None
        return self._rb_portatil.isChecked()

    def keep_original(self) -> bool:
        """True quando o dataset de origem deve continuar aberto."""
        if self._rb_manter_origem is None:
            return False
        return self._rb_manter_origem.isChecked()
