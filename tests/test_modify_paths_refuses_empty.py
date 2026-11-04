"""Modificar Caminhos não aplica um resultado sem faces.

O defeito: falha de cópia REMOVE a face do resultado (não há imagem a que
apontar) e `_on_modificar_caminhos_concluido` seguia mesmo com zero faces:
trocava o dataset aberto por um vazio, marcava sujo e propunha "Salvar
Como". O parquet de zero linhas resultante é aceito pela interface, listado
como recurso e some dos seletores sem explicação — foi assim que dois
datasets periciais viraram arquivos de 798 bytes.

A decisão mora no resultado (`LocalizationResult.refusal_message`); a
interface só a consulta. Falha PARCIAL também não passa calada: o resultado
só é aplicado com confirmação.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.localization_result import LocalizationResult


def _resultado(n_faces: int, falhas: int) -> LocalizationResult:
    faces = [FaceInfo(image_path=f"f{i}.png") for i in range(n_faces)]
    return LocalizationResult(
        dataset=FaceDataset(name="loc", faces=faces),
        target_dir=Path("C:/destino"),
        mode=LocalizationMode.COPY_ORIGINALS,
        naming=NamingStrategy.PRESERVE,
        n_processed=n_faces + falhas,
        n_copied=n_faces,
        n_skipped=0,
        n_failed=falhas,
        failures=tuple((f"x{i}.png", "arquivo não encontrado") for i in range(falhas)),
    )


# ── Regra (backend) ──────────────────────────────────────────────────────

def test_sem_faces_e_recusado_e_mostra_as_falhas():
    r = _resultado(0, 7)
    assert r.is_empty
    msg = r.refusal_message()
    assert msg is not None
    assert "7 falha(s)" in msg
    assert "x0.png: arquivo não encontrado" in msg
    assert "(+2)" in msg  # prévia limitada a 5


def test_com_faces_pode_seguir():
    assert _resultado(3, 2).refusal_message() is None
    assert _resultado(3, 0).refusal_message() is None
    assert _resultado(3, 2).n_faces_remaining == 3


# ── Interface: só consulta a regra ───────────────────────────────────────

def _controller_falso():
    chamadas = []
    fake = SimpleNamespace(
        view=None,
        dataset="ORIGINAL",
        current_path="C:/caso/caso.parquet",
        dataset_original=None,
        _audit=lambda *a, **k: chamadas.append("audit"),
        set_dirty=lambda v: chamadas.append(("dirty", v)),
        carregar_dataset_na_tela=lambda *a, **k: chamadas.append("carregar"),
    )
    return fake, chamadas


@pytest.fixture
def caixas(monkeypatch):
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager import dataset_manager_controller as mod

    registro = {"critical": [], "question": [], "information": [], "resp": None}

    class _MB:
        StandardButton = mod.QMessageBox.StandardButton

        @staticmethod
        def critical(parent, titulo, texto):
            registro["critical"].append(texto)

        @staticmethod
        def question(parent, titulo, texto, *a, **k):
            registro["question"].append(texto)
            return registro["resp"]

        @staticmethod
        def information(parent, titulo, texto):
            registro["information"].append(texto)

        @staticmethod
        def warning(parent, titulo, texto):
            registro["critical"].append(texto)

    monkeypatch.setattr(mod, "QMessageBox", _MB)
    return mod, registro


def test_interface_recusa_resultado_vazio(caixas):
    mod, registro = caixas
    fake, chamadas = _controller_falso()
    mod.DatasetManagerController._on_modificar_caminhos_concluido(
        fake, _resultado(0, 4),
    )
    assert fake.dataset == "ORIGINAL", "o dataset aberto foi trocado por um vazio"
    assert chamadas == [], "nada pode ser aplicado, marcado ou auditado"
    assert registro["critical"] and "x0.png" in registro["critical"][0]


def test_falha_parcial_exige_confirmacao(caixas):
    mod, registro = caixas
    registro["resp"] = mod.QMessageBox.StandardButton.Cancel
    fake, chamadas = _controller_falso()
    mod.DatasetManagerController._on_modificar_caminhos_concluido(
        fake, _resultado(3, 2),
    )
    assert registro["question"], "falha parcial passou sem pergunta"
    assert fake.dataset == "ORIGINAL"
    assert "carregar" not in chamadas


def test_falha_parcial_confirmada_aplica(caixas):
    mod, registro = caixas
    registro["resp"] = mod.QMessageBox.StandardButton.Yes
    fake, chamadas = _controller_falso()
    r = _resultado(3, 2)
    mod.DatasetManagerController._on_modificar_caminhos_concluido(fake, r)
    assert fake.dataset is r.dataset
    assert "carregar" in chamadas
