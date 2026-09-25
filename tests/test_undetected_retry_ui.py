"""A retentativa dentro da janela de revisão.

O ponto forense desta tela: em dataset de faces recortadas ou alinhadas, a
inclusão nasce bloqueada porque sem detecção não há caixa, logo não há
recorte a gerar. Recuperada a face, existe caixa — e o bloqueio LEVANTA
para ela. O bloqueio é "enquanto não detectada", não "para sempre".

O que os testes travam:

- a face recuperada entra marcada (o motivo de estar na janela deixou de
  existir) e passa a ser marcável mesmo sob política bloqueada;
- as já recuperadas não são tentadas de novo;
- fechar a janela pelo X nunca descarta uma recuperada — ela tem biometria,
  e descartá-la jogaria fora o trabalho da retentativa;
- `recovered_faces()` só devolve as que de fato vão entrar, porque é essa
  lista que dispara os estágios de qualidade/recorte pendentes;
- sem fábrica de FFService (uso programático), a janela não oferece
  retentativa em vez de estourar.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from faceaudit.enums.dataset_image_kind import DatasetImageKind  # noqa: E402
from faceaudit.enums.sepael_model import SepaelModel  # noqa: E402
from faceaudit.models.dataset_ingestion_config import (  # noqa: E402
    DatasetIngestionConfig,
)
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.models.ff_info import FFInfo  # noqa: E402

SLOT = SepaelModel.default().ffinfo_slot


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _face(nome: str) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}")
    f.error = True
    return f


class _ServicoFalso:
    def __init__(self, det_size: int, minimo: int = 640):
        self.det_size = det_size
        self._minimo = minimo

    def process_face(self, face, **kwargs):
        if self.det_size >= self._minimo:
            face.ff_info = FFInfo(**{SLOT: np.ones(4, dtype=np.float32)})
        return face


def _controlador(app, *, n=2, minimo=640, com_fabrica=True, **cfg):
    from interface.tabs.dataset_manager.undetected_review_dialog import (
        UndetectedReviewController,
    )

    base = dict(input_path="C:/fotos", dataset_name="t", ff_det_size=320)
    base.update(cfg)
    fabrica = (lambda ds: _ServicoFalso(ds, minimo=minimo)) if com_fabrica else None
    return UndetectedReviewController(
        None,
        faces=[_face(f"{i}.jpg") for i in range(n)],
        config=DatasetIngestionConfig(**base),
        n_total=n + 5,
        ff_service_factory=fabrica,
    )


def _fechar(ctrl):
    ctrl._encerrar_worker()
    ctrl.view.deleteLater()


def _rodar_sincrono(ctrl, det_size, so_selecionada=False):
    """Executa a retentativa sem thread, para o teste ser determinístico."""
    indices = ctrl._indices_para_retentativa(so_selecionada)
    resultado = ctrl._retry.retry(
        [ctrl._faces[i] for i in indices],
        det_size=det_size, baseline_det_size=ctrl._baseline_det_size,
    )
    ctrl._ao_terminar_retentativa(resultado, indices, det_size)
    return resultado


# ── Oferta da retentativa ────────────────────────────────────────────────

def test_oferece_apenas_resolucoes_acima_da_usada(app):
    ctrl = _controlador(app)
    try:
        assert all(r > 320 for r in ctrl._resolucoes)
        assert ctrl.view.cmb_resolucao.count() == len(ctrl._resolucoes)
    finally:
        _fechar(ctrl)


def test_ingestao_ja_em_alta_resolucao_oferece_menos_opcoes(app):
    """Quem já ingeriu a 960 não deve ver 640 como "tentar de novo"."""
    ctrl = _controlador(app, ff_det_size=960)
    try:
        assert all(r > 960 for r in ctrl._resolucoes)
    finally:
        _fechar(ctrl)


def test_sem_fabrica_nao_oferece_retentativa(app):
    """Uso programático não tem como construir FFService — a janela abre
    sem os controles, em vez de estourar."""
    ctrl = _controlador(app, com_fabrica=False)
    try:
        assert ctrl._resolucoes == ()
        assert not hasattr(ctrl.view, "cmb_resolucao")
    finally:
        _fechar(ctrl)


# ── Recuperação na janela ────────────────────────────────────────────────

def test_recuperada_entra_marcada(app):
    """O motivo de ela estar nesta janela deixou de existir."""
    ctrl = _controlador(app)
    try:
        _rodar_sincrono(ctrl, 640)

        incluir, descartar = ctrl.decision()
        assert len(incluir) == 2
        assert descartar == []
    finally:
        _fechar(ctrl)


def test_recuperada_aparece_marcada_na_grade(app):
    ctrl = _controlador(app)
    try:
        _rodar_sincrono(ctrl, 640)

        assert "640 px" in ctrl.view.lista.item(0).text()
    finally:
        _fechar(ctrl)


def test_nada_recuperado_avisa_e_sugere_subir(app):
    ctrl = _controlador(app, minimo=1280)
    try:
        _rodar_sincrono(ctrl, 640)

        aviso = ctrl.view.lbl_retentativa.text()
        assert "Nenhuma face encontrada" in aviso
        assert "maior" in aviso, "o próximo passo tem de estar dito"
        assert ctrl.decision()[0] == []
    finally:
        _fechar(ctrl)


def test_ja_recuperada_nao_e_tentada_de_novo(app):
    ctrl = _controlador(app)
    try:
        _rodar_sincrono(ctrl, 640)

        assert ctrl._indices_para_retentativa(so_selecionada=False) == []
    finally:
        _fechar(ctrl)


def test_botoes_voltam_a_funcionar_depois_da_rodada(app):
    ctrl = _controlador(app, minimo=1280)
    try:
        ctrl.view.definir_retentativa_habilitada(False)
        _rodar_sincrono(ctrl, 640)

        assert ctrl.view.btn_tentar_uma.isEnabled() is True
    finally:
        _fechar(ctrl)


# ── O bloqueio que levanta ───────────────────────────────────────────────

def test_dataset_de_recortes_bloqueia_antes_de_recuperar(app):
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.CROP,
    )
    try:
        assert ctrl._pode_incluir(0) is False
        ctrl._ao_marcar(0, True)
        assert ctrl.decision()[0] == []
    finally:
        _fechar(ctrl)


def test_recuperada_libera_a_inclusao_mesmo_em_dataset_de_recortes(app):
    """Agora existe caixa da face, logo existe recorte a gerar — o motivo do
    bloqueio deixou de valer para ELA."""
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.CROP,
    )
    try:
        _rodar_sincrono(ctrl, 640)

        assert ctrl._pode_incluir(0) is True
        assert len(ctrl.decision()[0]) == 2
        item = ctrl.view.lista.item(0)
        assert item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable
    finally:
        _fechar(ctrl)


def test_bloqueio_segue_valendo_para_quem_nao_recuperou(app):
    """Recuperar uma não abre a porta para as outras."""
    ctrl = _controlador(
        app, minimo=640, self_contained=True, image_kind=DatasetImageKind.ALIGNED,
    )
    try:
        # Recupera só a primeira, tentando apenas a selecionada.
        ctrl.view.lista.setCurrentRow(0)
        indices = ctrl._indices_para_retentativa(so_selecionada=True)
        resultado = ctrl._retry.retry(
            [ctrl._faces[i] for i in indices], det_size=640, baseline_det_size=320,
        )
        ctrl._ao_terminar_retentativa(resultado, indices, 640)

        assert ctrl._pode_incluir(0) is True
        assert ctrl._pode_incluir(1) is False
    finally:
        _fechar(ctrl)


def test_aviso_explica_que_a_inclusao_liberou(app):
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.CROP,
    )
    try:
        _rodar_sincrono(ctrl, 640)

        assert "podem ser incluídas" in ctrl.view.lbl_retentativa.text()
    finally:
        _fechar(ctrl)


# ── Fechar sem concluir ──────────────────────────────────────────────────

def test_fechar_sem_concluir_preserva_a_recuperada(app, monkeypatch):
    """Descartá-la jogaria fora justamente o trabalho da retentativa."""
    ctrl = _controlador(
        app, minimo=640, self_contained=True, image_kind=DatasetImageKind.CROP,
    )
    try:
        _rodar_sincrono(ctrl, 640)
        # Simula fechar pelo X (exec devolve Rejected).
        monkeypatch.setattr(
            ctrl.view, "exec",
            lambda: QtWidgets.QDialog.DialogCode.Rejected,
        )
        ctrl.exec()

        assert len(ctrl.decision()[0]) == 2
    finally:
        _fechar(ctrl)


# ── Lista para os estágios pendentes ─────────────────────────────────────

def test_recovered_faces_alimenta_os_estagios_pendentes(app):
    ctrl = _controlador(app)
    try:
        _rodar_sincrono(ctrl, 640)

        recuperadas = ctrl.recovered_faces()
        assert len(recuperadas) == 2
        assert all(f.ff_info is not None for f in recuperadas)
    finally:
        _fechar(ctrl)


def test_recovered_faces_ignora_a_recuperada_que_o_usuario_desmarcou(app):
    """Ela não vai entrar no dataset, então não há estágio a completar."""
    ctrl = _controlador(app)
    try:
        _rodar_sincrono(ctrl, 640)
        ctrl._ao_marcar(0, False)

        nomes = [f.image_path for f in ctrl.recovered_faces()]
        assert "C:/fotos/0.jpg" not in nomes
        assert len(nomes) == 1
    finally:
        _fechar(ctrl)


def test_recovered_faces_vazio_sem_retentativa(app):
    ctrl = _controlador(app)
    try:
        assert ctrl.recovered_faces() == []
    finally:
        _fechar(ctrl)
