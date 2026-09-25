"""Costura da revisão no fluxo de criação de dataset.

Testa o encadeamento que o usuário percebe: a ingestão termina, a revisão
aparece, a decisão é aplicada, o parquet é gravado — e a cadeia de custódia
registra o que foi incluído e o que foi descartado.

O que os testes travam:

- a gravação NÃO acontece antes da revisão (senão o arquivo nasceria com
  linhas que a decisão logo invalida);
- descartar tudo não grava nada e avisa — parquet de zero linhas é aceito
  pela interface e desaparece dos seletores sem explicação;
- a custódia registra quantas foram incluídas, quantas descartadas, os
  nomes dos arquivos e quantas faces só apareceram depois de girar. Sem
  isso ninguém reconstrói por que o dataset tem menos linhas que a pasta
  de origem tinha arquivos;
- a criação pela interface liga `review_undetected`; chamadores sem tela
  não, senão o dataset esperaria por uma decisão que nunca vem.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.enums.sepael_model import SepaelModel  # noqa: E402
from faceaudit.models.dataset_ingestion_config import (  # noqa: E402
    DatasetIngestionConfig,
)
from faceaudit.models.face_dataset import FaceDataset  # noqa: E402
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.models.ff_info import FFInfo  # noqa: E402

SLOT = SepaelModel.default().ffinfo_slot


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _face(nome: str, *, embedding: bool = True, rotacao=None) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}")
    f.identity = "id1"
    if embedding:
        f.ff_info = FFInfo(
            detection_rotation=rotacao, **{SLOT: np.ones(4, dtype=np.float32)},
        )
    return f


_METODOS = (
    "_on_criacao_concluida", "_revisar_nao_detectadas_e_gravar",
    "_finalizar_e_gravar", "_concluir_criacao", "_descricao_da_criacao",
    "_metadados_da_deteccao", "_contar_giradas", "_nomes_de_arquivo",
)


def _controller(app, tmp_path):
    """Duplo que empresta os métodos REAIS do controlador.

    Construir o `DatasetManagerController` de verdade arrastaria a aba
    inteira, e ele é um `QObject` (não aceita `object.__new__`). O duplo
    mantém as funções sob teste sendo as de produção — o que muda é só o
    `self` que elas recebem.
    """
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        DatasetManagerController,
    )

    # `__dict__` e não `getattr`: preserva o descritor `staticmethod`.
    # Via getattr, uma staticmethod chega como função simples e volta a
    # receber `self` ao ser posta na classe do duplo.
    corpo = {
        nome: DatasetManagerController.__dict__[nome] for nome in _METODOS
    }
    ctrl = type("_ControllerDuplo", (), corpo)()
    ctrl.view = None
    ctrl._ingestion_config = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="ds", dataset_dir=tmp_path,
        review_undetected=True,
    )
    ctrl._revisao_incluidas = []
    ctrl._revisao_descartadas = []
    return ctrl


# ── Proveniência na custódia ─────────────────────────────────────────────

def test_descricao_cita_as_faces_giradas(app, tmp_path):
    ctrl = _controller(app, tmp_path)
    ds = FaceDataset(name="ds", faces=[
        _face("a.jpg", rotacao=90), _face("b.jpg", rotacao=270), _face("c.jpg"),
    ])

    texto = ctrl._descricao_da_criacao(ds)

    assert "2 face(s) só foram detectadas com a imagem girada" in texto


def test_descricao_omite_rotacao_quando_nao_houve(app, tmp_path):
    ctrl = _controller(app, tmp_path)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])

    assert "girada" not in ctrl._descricao_da_criacao(ds)


def test_descricao_cita_a_decisao_da_revisao(app, tmp_path):
    ctrl = _controller(app, tmp_path)
    ctrl._revisao_incluidas = [_face("x.jpg", embedding=False)]
    ctrl._revisao_descartadas = [
        _face("y.jpg", embedding=False), _face("z.jpg", embedding=False),
    ]
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])

    texto = ctrl._descricao_da_criacao(ds)

    assert "1 incluída(s) por decisão do operador" in texto
    assert "2 descartada(s)" in texto


def test_metadados_trazem_os_nomes_dos_arquivos(app, tmp_path):
    """A contagem diz quanto; o nome diz qual — é o nome que permite
    conferir depois."""
    ctrl = _controller(app, tmp_path)
    ctrl._revisao_incluidas = [_face("incluida.jpg", embedding=False)]
    ctrl._revisao_descartadas = [_face("fora.jpg", embedding=False)]
    ds = FaceDataset(name="ds", faces=[_face("a.jpg", rotacao=180)])

    meta = ctrl._metadados_da_deteccao(ds)

    assert meta["undetected_total"] == 2
    assert meta["undetected_included"] == 1
    assert meta["undetected_discarded"] == 1
    assert meta["undetected_included_files"] == ["incluida.jpg"]
    assert meta["undetected_discarded_files"] == ["fora.jpg"]
    assert meta["faces_detected_after_rotation"] == 1


def test_metadados_ficam_vazios_quando_nada_aconteceu(app, tmp_path):
    """Dataset sem incidente não polui a custódia com zeros."""
    ctrl = _controller(app, tmp_path)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])

    assert ctrl._metadados_da_deteccao(ds) == {}


def test_lista_de_nomes_e_limitada(app, tmp_path):
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        _MAX_AUDIT_IDENTIFIERS,
    )

    ctrl = _controller(app, tmp_path)
    muitas = [_face(f"f{i}.jpg", embedding=False) for i in range(_MAX_AUDIT_IDENTIFIERS + 20)]

    assert len(ctrl._nomes_de_arquivo(muitas)) == _MAX_AUDIT_IDENTIFIERS


# ── Encadeamento ─────────────────────────────────────────────────────────

def test_dataset_sem_pendentes_vai_direto_para_a_conclusao(app, tmp_path, monkeypatch):
    ctrl = _controller(app, tmp_path)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])
    chamadas = []
    monkeypatch.setattr(
        type(ctrl), "_concluir_criacao", lambda self, d: chamadas.append(d),
    )
    monkeypatch.setattr(
        type(ctrl), "_revisar_nao_detectadas_e_gravar",
        lambda self, d: pytest.fail("não havia o que revisar"),
    )

    ctrl._on_criacao_concluida(ds)

    assert chamadas == [ds]


def test_dataset_com_gravacao_adiada_abre_a_revisao(app, tmp_path, monkeypatch):
    ctrl = _controller(app, tmp_path)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])
    ds._save_deferred = True
    ds._undetected_faces = [_face("ruim.jpg", embedding=False)]
    chamadas = []
    monkeypatch.setattr(
        type(ctrl), "_revisar_nao_detectadas_e_gravar",
        lambda self, d: chamadas.append(d),
    )
    monkeypatch.setattr(
        type(ctrl), "_concluir_criacao",
        lambda self, d: pytest.fail("não pode concluir antes da revisão"),
    )

    ctrl._on_criacao_concluida(ds)

    assert chamadas == [ds], "a revisão vem ANTES de gravar"


def test_decisao_de_incluir_devolve_a_face_ao_dataset(app, tmp_path, monkeypatch):
    ctrl = _controller(app, tmp_path)
    pendente = _face("ruim.jpg", embedding=False)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])
    ds._save_deferred = True
    ds._undetected_faces = [pendente]

    _fingir_revisao(monkeypatch, incluir=[pendente], descartar=[])
    gravados = []
    monkeypatch.setattr(
        type(ctrl), "_finalizar_e_gravar",
        lambda self, d, inc, desc: gravados.append((d, inc, desc)),
    )

    ctrl._revisar_nao_detectadas_e_gravar(ds)

    assert len(ds) == 2
    assert any("ruim.jpg" in f.image_path for f in ds.faces)
    assert gravados and gravados[0][1] == [pendente]


def test_decisao_de_descartar_nao_devolve_a_face(app, tmp_path, monkeypatch):
    ctrl = _controller(app, tmp_path)
    pendente = _face("ruim.jpg", embedding=False)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])
    ds._save_deferred = True
    ds._undetected_faces = [pendente]

    _fingir_revisao(monkeypatch, incluir=[], descartar=[pendente])
    gravados = []
    monkeypatch.setattr(
        type(ctrl), "_finalizar_e_gravar",
        lambda self, d, inc, desc: gravados.append((d, inc, desc)),
    )

    ctrl._revisar_nao_detectadas_e_gravar(ds)

    assert len(ds) == 1
    assert gravados[0][2] == [pendente]


def test_descartar_tudo_nao_grava_e_avisa(app, tmp_path, monkeypatch):
    """Sem nenhuma face, gravar produziria um parquet de zero linhas."""
    ctrl = _controller(app, tmp_path)
    pendente = _face("ruim.jpg", embedding=False)
    ds = FaceDataset(name="ds", faces=[])
    ds._save_deferred = True
    ds._undetected_faces = [pendente]

    _fingir_revisao(monkeypatch, incluir=[], descartar=[pendente])
    monkeypatch.setattr(
        type(ctrl), "_finalizar_e_gravar",
        lambda self, d, inc, desc: pytest.fail("não pode gravar dataset vazio"),
    )
    avisos = []
    import interface.tabs.dataset_manager.dataset_manager_controller as mod

    monkeypatch.setattr(
        mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: avisos.append(a)),
    )

    ctrl._revisar_nao_detectadas_e_gravar(ds)

    assert avisos, "o usuário precisa saber que nada foi criado"
    texto = " ".join(str(p) for p in avisos[0])
    assert "vazio" in texto


def _fingir_revisao(monkeypatch, *, incluir, descartar):
    """Substitui a janela por uma decisão pronta."""
    import interface.tabs.dataset_manager.undetected_review_dialog as pkg

    class _Falso:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return True

        def decision(self):
            return list(incluir), list(descartar)

    monkeypatch.setattr(pkg, "UndetectedReviewController", _Falso)


# ── Quem liga o adiamento ────────────────────────────────────────────────

def test_criacao_pela_interface_liga_a_revisao(app):
    """A tela tem como perguntar, então liga."""
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (  # noqa: E501
        NewDatasetController,
    )
    import inspect

    fonte = inspect.getsource(NewDatasetController.get_config)
    assert "review_undetected=True" in fonte


def test_default_do_dto_e_desligado():
    """Chamador sem tela não pode ligar: o dataset ficaria esperando por uma
    decisão que nunca vem, e nada seria gravado."""
    cfg = DatasetIngestionConfig(input_path="x", dataset_name="y")

    assert cfg.review_undetected is False
    assert cfg.try_rotated_detection is True, (
        "girar é ganho puro e não depende de ninguém decidir nada"
    )
