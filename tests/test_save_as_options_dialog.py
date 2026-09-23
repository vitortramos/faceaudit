"""Opções do Salvar Como: padrões e grupos condicionais.

Os dois padrões são decisões de projeto, não detalhe:

- **imagens → copiar (autocontido)**: o resultado é uma pasta movível; a
  alternativa (referenciar) depende da máquina enxergar os caminhos;
- **dataset aberto → o de origem**: o fluxo típico é derivar vários
  recortes do mesmo dataset grande, e trocar para o recorte recém-salvo
  obrigava a fechar e reabrir o grande a cada um.

Cada grupo só aparece quando a decisão existe.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from interface.tabs.dataset_manager.save_as_options_dialog import (  # noqa: E402
    SaveAsOptionsDialog,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_padroes(app):
    dlg = SaveAsOptionsDialog(
        ask_images=True, ask_which_stays=True,
        nome_destino="C1 - QUEST", nome_origem="RGMA completo",
        n_salvar=308, n_origem=16321,
    )
    try:
        assert dlg.portable() is True, "o padrão é o resultado portátil"
        assert dlg.keep_original() is True, "o padrão é continuar na origem"
    finally:
        dlg.deleteLater()


def test_sem_filtro_nao_pergunta_qual_fica(app):
    dlg = SaveAsOptionsDialog(ask_images=True, ask_which_stays=False)
    try:
        assert dlg.portable() is True
        assert dlg.keep_original() is False, (
            "sem dataset de origem, o salvo é o que fica"
        )
    finally:
        dlg.deleteLater()


def test_mesma_pasta_nao_pergunta_imagens(app):
    dlg = SaveAsOptionsDialog(ask_images=False, ask_which_stays=True)
    try:
        assert dlg.portable() is None, (
            "sem mudança de pasta não há decisão sobre imagens"
        )
        assert dlg.keep_original() is True
    finally:
        dlg.deleteLater()


def test_contagens_aparecem_para_o_perito(app):
    """Os números são o que deixa a escolha óbvia na tela."""
    dlg = SaveAsOptionsDialog(
        ask_which_stays=True, nome_destino="recorte",
        nome_origem="base", n_salvar=308, n_origem=16321,
    )
    try:
        textos = " | ".join(
            rb.text() for rb in dlg.findChildren(QtWidgets.QRadioButton)
        )
        assert "16.321" in textos and "308" in textos
        assert "base" in textos and "recorte" in textos
    finally:
        dlg.deleteLater()
