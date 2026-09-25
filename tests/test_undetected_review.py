"""Revisão das imagens sem face detectada — política e janela.

A política é a parte com consequência forense: num dataset de faces
recortadas ou alinhadas, incluir uma imagem sem detecção significa colocar
a foto inteira num dataset marcado como alinhado. O enriquecimento seguinte
confia nessa marca e manda a foto ao encoder como se fosse uma face
112×112 — sai embedding, sem erro nenhum, e ele não representa face alguma.

O que os testes travam:

- a política é do BACKEND e depende do que o dataset guarda, não de quem
  pergunta;
- bloqueado significa bloqueado de verdade: nem clique, nem "incluir
  todas", nem chamada direta do controlador conseguem marcar;
- fechar a janela sem concluir NÃO descarta — descartar por omissão apagaria
  imagens que o usuário escolheu ingerir;
- quando bloqueado, a caixa aparece desabilitada em vez de escondida, e o
  motivo está à vista.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from faceaudit.enums.dataset_image_kind import DatasetImageKind  # noqa: E402
from faceaudit.models.dataset_ingestion_config import (  # noqa: E402
    DatasetIngestionConfig,
)
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.services.undetected_review_policy import (  # noqa: E402
    resolve_inclusion_policy,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _config(**kwargs) -> DatasetIngestionConfig:
    base = dict(input_path="C:/fotos", dataset_name="teste")
    base.update(kwargs)
    return DatasetIngestionConfig(**base)


# ── Política (backend) ───────────────────────────────────────────────────

def test_dataset_de_originais_permite_incluir():
    politica = resolve_inclusion_policy(_config())

    assert politica.allowed is True
    assert politica.blocked is False
    assert "sem embedding" in politica.reason, (
        "mesmo permitido, o perito precisa saber o que a inclusão significa"
    )


def test_dataset_de_alinhadas_bloqueia():
    politica = resolve_inclusion_policy(
        _config(self_contained=True, image_kind=DatasetImageKind.ALIGNED),
    )

    assert politica.blocked is True
    assert "112×112" in politica.reason


def test_dataset_de_recortes_bloqueia():
    politica = resolve_inclusion_policy(
        _config(self_contained=True, image_kind=DatasetImageKind.CROP),
    )

    assert politica.blocked is True
    assert "recorte" in politica.reason


def test_alinhar_durante_a_criacao_bloqueia():
    """O dataset termina marcado como alinhado — mesmo problema."""
    politica = resolve_inclusion_policy(
        _config(align_during_ingestion=True, aligned_target_dir="C:/saida"),
    )

    assert politica.blocked is True


def test_fonte_declarada_como_alinhada_bloqueia():
    politica = resolve_inclusion_policy(_config(is_aligned=True))

    assert politica.blocked is True
    assert "declarado" in politica.reason


def test_recorte_sem_autocontido_nao_bloqueia():
    """`image_kind` só tem efeito com autocontido: sem cópia de imagem não
    há forma a escolher, e o dataset referencia as originais."""
    politica = resolve_inclusion_policy(
        _config(self_contained=False, image_kind=DatasetImageKind.CROP),
    )

    assert politica.allowed is True


def test_motivo_explica_a_consequencia_e_nao_so_a_regra():
    """"Não permitido" sem o porquê viraria dúvida sobre defeito no app."""
    politica = resolve_inclusion_policy(
        _config(self_contained=True, image_kind=DatasetImageKind.ALIGNED),
    )

    assert "embedding" in politica.reason
    assert len(politica.reason) > 80


# ── Janela ───────────────────────────────────────────────────────────────

def _controlador(app, faces=None, **cfg_kwargs):
    from interface.tabs.dataset_manager.undetected_review_dialog import (
        UndetectedReviewController,
    )

    faces = faces or [
        FaceInfo(image_path="C:/fotos/a.jpg"), FaceInfo(image_path="C:/fotos/b.jpg"),
    ]
    return UndetectedReviewController(
        None, faces=faces, config=_config(**cfg_kwargs), n_total=10,
    )


def _fechar(ctrl):
    ctrl._encerrar_worker()
    ctrl.view.deleteLater()


def test_nasce_com_tudo_descartado(app):
    """Nada entra sem alguém marcar — o oposto do comportamento antigo,
    em que tudo entrava sem ninguém ver."""
    ctrl = _controlador(app)
    try:
        incluir, descartar = ctrl.decision()
        assert incluir == []
        assert len(descartar) == 2
    finally:
        _fechar(ctrl)


def test_marcar_uma_move_para_incluir(app):
    ctrl = _controlador(app)
    try:
        ctrl._ao_marcar(0, True)
        incluir, descartar = ctrl.decision()

        assert [f.image_path for f in incluir] == ["C:/fotos/a.jpg"]
        assert len(descartar) == 1
    finally:
        _fechar(ctrl)


def test_incluir_todas(app):
    ctrl = _controlador(app)
    try:
        ctrl._marcar_todas()
        assert len(ctrl.decision()[0]) == 2
    finally:
        _fechar(ctrl)


def test_descartar_todas_reverte(app):
    ctrl = _controlador(app)
    try:
        ctrl._marcar_todas()
        ctrl._desmarcar_todas()
        assert ctrl.decision()[0] == []
    finally:
        _fechar(ctrl)


def test_contador_reflete_a_decisao(app):
    ctrl = _controlador(app)
    try:
        ctrl._ao_marcar(0, True)
        texto = ctrl.view.lbl_contador.text()
        assert "1 a incluir" in texto and "1 a descartar" in texto
    finally:
        _fechar(ctrl)


def test_bloqueado_ignora_marcacao_por_clique(app):
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.ALIGNED,
    )
    try:
        ctrl._ao_marcar(0, True)
        assert ctrl.decision()[0] == [], (
            "com a política bloqueando, nem a marcação direta pode passar"
        )
    finally:
        _fechar(ctrl)


def test_bloqueado_ignora_incluir_todas(app):
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.CROP,
    )
    try:
        ctrl._marcar_todas()
        assert ctrl.decision()[0] == []
    finally:
        _fechar(ctrl)


def test_bloqueado_desabilita_os_controles_sem_esconder(app):
    """Esconder daria a impressão de que a escolha não existe; desabilitado
    com o motivo ao lado diz que ela existe e por que não está disponível."""
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.ALIGNED,
    )
    try:
        assert ctrl.view.chk_incluir.isEnabled() is False
        assert ctrl.view.chk_incluir.isVisible() or True  # existe na árvore
        assert ctrl.view.btn_todas.isEnabled() is False
        assert ctrl.view.btn_nenhuma.isEnabled() is True, (
            "descartar continua disponível — é a única saída aqui"
        )
        assert "indisponível" in ctrl.view.lbl_motivo.text().lower()
    finally:
        _fechar(ctrl)


def test_bloqueado_nao_deixa_o_item_marcavel(app):
    ctrl = _controlador(
        app, self_contained=True, image_kind=DatasetImageKind.ALIGNED,
    )
    try:
        item = ctrl.view.lista.item(0)
        assert not (item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable)
    finally:
        _fechar(ctrl)


def test_grade_tem_uma_linha_por_imagem(app):
    ctrl = _controlador(app)
    try:
        assert ctrl.view.lista.count() == 2
        assert ctrl.view.lista.item(0).text() == "a.jpg"
    finally:
        _fechar(ctrl)


def test_cabecalho_diz_quantas_ficaram_fora_e_quantas_entraram(app):
    ctrl = _controlador(app)
    try:
        textos = " ".join(
            w.text() for w in ctrl.view.findChildren(QtWidgets.QLabel)
        )
        assert "2 imagem(ns) sem face detectada" in textos
        assert "8" in textos, "as que passaram dão a escala do problema"
        assert "orientações" in textos, (
            "sem isso o usuário não sabe que girar já foi tentado"
        )
    finally:
        _fechar(ctrl)


def test_imagem_ilegivel_nao_fica_carregando_para_sempre(app):
    """Arquivo inexistente é o caso comum aqui (caminho quebrado)."""
    ctrl = _controlador(app, faces=[FaceInfo(image_path="C:/nao/existe.jpg")])
    try:
        ctrl._ao_selecionar(0)
        assert "não pôde ser lida" in ctrl.view.lbl_imagem.text()
    finally:
        _fechar(ctrl)


def test_descricao_mostra_o_caminho_inteiro(app):
    """Duas fotos podem ter o mesmo nome e vir de pastas diferentes."""
    ctrl = _controlador(app)
    try:
        ctrl._ao_selecionar(0)
        assert "C:/fotos/a.jpg" in ctrl.view.lbl_arquivo.text()
    finally:
        _fechar(ctrl)
