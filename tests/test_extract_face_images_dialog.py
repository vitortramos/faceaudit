"""Diálogo de geração de imagens de face: o que ele coleta e o que barra.

A view não tem regra — quem produz as imagens é o backend. O que ela
precisa garantir é que a intenção chegue completa e que combinações
destrutivas não passem: gravar por cima da pasta de origem destruiria o
material original do exame.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.enums.dataset_image_kind import DatasetImageKind  # noqa: E402
from interface.tabs.dataset_manager.extract_face_images_dialog import (  # noqa: E402
    ExtractFaceImagesDialog,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_padroes(app):
    dlg = ExtractFaceImagesDialog()
    try:
        assert dlg.image_kind() is DatasetImageKind.CROP
        assert dlg.margin_factor() == 2.0, "mesma margem padrão da extração de vídeo"
        assert dlg.recursive() is True
    finally:
        dlg.deleteLater()


def test_margem_so_vale_para_recorte(app):
    dlg = ExtractFaceImagesDialog()
    try:
        assert dlg._spn_margem.isEnabled()
        dlg._rb_aligned.setChecked(True)
        assert not dlg._spn_margem.isEnabled(), (
            "alinhamento não tem margem — o enquadramento é canônico"
        )
    finally:
        dlg.deleteLater()


def test_destino_igual_a_origem_e_recusado(app, tmp_path, monkeypatch):
    avisos = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning",
        lambda *a, **k: avisos.append(a),
    )
    dlg = ExtractFaceImagesDialog()
    try:
        dlg._ed_origem.setText(str(tmp_path))
        dlg._ed_destino.setText(str(tmp_path))
        dlg._on_aceitar()
        assert dlg.result() != QtWidgets.QDialog.DialogCode.Accepted
        assert avisos, "gravar por cima da origem precisa ser barrado"
    finally:
        dlg.deleteLater()


def test_origem_inexistente_e_recusada(app, tmp_path, monkeypatch):
    avisos = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "warning", lambda *a, **k: avisos.append(a),
    )
    dlg = ExtractFaceImagesDialog()
    try:
        dlg._ed_origem.setText(str(tmp_path / "nao_existe"))
        dlg._ed_destino.setText(str(tmp_path / "saida"))
        dlg._on_aceitar()
        assert avisos
    finally:
        dlg.deleteLater()


def test_coleta_completa(app, tmp_path):
    origem = tmp_path / "fotos"
    origem.mkdir()
    dlg = ExtractFaceImagesDialog()
    try:
        dlg._ed_origem.setText(str(origem))
        dlg._ed_destino.setText(str(tmp_path / "saida"))
        dlg._rb_aligned.setChecked(True)
        dlg._chk_recursivo.setChecked(False)
        dlg._on_aceitar()

        assert dlg.result() == QtWidgets.QDialog.DialogCode.Accepted
        assert dlg.source_dir() == origem
        assert dlg.target_dir() == tmp_path / "saida"
        assert dlg.image_kind() is DatasetImageKind.ALIGNED
        assert dlg.recursive() is False
    finally:
        dlg.deleteLater()
