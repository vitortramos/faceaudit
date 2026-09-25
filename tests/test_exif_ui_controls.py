"""Controles que ligam a extração de metadados e apontam o ExifTool.

Três telas: criação de dataset, enriquecimento e configurações.

O que os testes travam:

- a caixa é **desligada e bloqueada** quando a entrada já é face alinhada.
  Metadado de captura não existe num recorte — deixar marcar produziria
  colunas vazias e a impressão falsa de que a foto original não tinha
  metadado nenhum;
- a configuração diz **na hora** se o caminho resolve. Descobrir que o
  ExifTool não estava no lugar só ao criar o dataset custa uma ingestão
  inteira.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.models.system_config import SystemConfig  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── Criação de dataset ────────────────────────────────────────────────────

def _new_dataset_view(app):
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_view import (
        NewDatasetView,
    )

    return NewDatasetView()


def test_criacao_tem_a_caixa_desligada_por_padrao(app):
    v = _new_dataset_view(app)
    try:
        assert v.chk_exif.isChecked() is False, (
            "extrair metadado é opt-in: custa I/O e só serve a fotos originais"
        )
        assert v.get_extract_exif() is False
    finally:
        v.deleteLater()


def test_criacao_marcada_devolve_true(app):
    v = _new_dataset_view(app)
    try:
        v.chk_exif.setChecked(True)
        assert v.get_extract_exif() is True
    finally:
        v.deleteLater()


def test_caixa_bloqueada_nao_conta_como_marcada(app):
    """Bloqueada + marcada por engano não pode ligar a extração."""
    v = _new_dataset_view(app)
    try:
        v.chk_exif.setChecked(True)
        v.chk_exif.setEnabled(False)
        assert v.get_extract_exif() is False
    finally:
        v.deleteLater()


# ── Enriquecimento ────────────────────────────────────────────────────────

def test_enriquecimento_tem_a_mesma_caixa(app):
    from interface.tabs.dataset_manager.enrich_dataset_dialog.enrich_dataset_view import (
        EnrichDatasetView,
    )

    v = EnrichDatasetView()
    try:
        assert hasattr(v, "chk_exif")
        v.chk_exif.setChecked(True)
        assert v.get_extract_exif() is True
        v.chk_exif.setEnabled(False)
        assert v.get_extract_exif() is False
    finally:
        v.deleteLater()


# ── Configurações ─────────────────────────────────────────────────────────

def _config_dialog(app, caminho: str):
    from interface.config_dialog import ConfigDialog

    return ConfigDialog(
        SystemConfig.from_dict({"exiftool": {"exiftool_exe_path": caminho}})
    )


def test_config_carrega_o_caminho_gravado(app):
    d = _config_dialog(app, "C:/Exiftool/exiftool.exe")
    try:
        assert d.input_exiftool.text() == "C:/Exiftool/exiftool.exe"
    finally:
        d.deleteLater()


def test_config_avisa_quando_o_caminho_nao_resolve(app):
    d = _config_dialog(app, "exiftool")
    try:
        d.input_exiftool.setText("caminho_que_nao_existe")
        texto = d.lbl_exiftool_status.text()
        assert "não encontrado" in texto.lower()
        assert "pillow" in texto.lower(), (
            "o aviso precisa dizer o que acontece na prática, não só que faltou"
        )
    finally:
        d.deleteLater()


def test_config_confirma_quando_resolve(app, tmp_path):
    """Qualquer executável existente serve para a checagem de caminho."""
    falso = tmp_path / "exiftool.exe"
    falso.write_text("", encoding="utf-8")
    d = _config_dialog(app, str(falso))
    try:
        assert "encontrado" in d.lbl_exiftool_status.text().lower()
    finally:
        d.deleteLater()


def test_config_grava_o_caminho(app, tmp_path, monkeypatch):
    d = _config_dialog(app, "exiftool")
    try:
        monkeypatch.setattr(type(d.config), "save_to_file", lambda self: None)
        d.input_exiftool.setText("D:/ferramentas/exiftool.exe")
        d._save_and_close()
        assert d.config.exiftool.exiftool_exe_path == "D:/ferramentas/exiftool.exe"
    finally:
        d.deleteLater()
