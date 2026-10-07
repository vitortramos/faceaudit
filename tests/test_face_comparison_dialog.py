"""A janela que mostra as duas faces de uma comparação lado a lado.

A tabela diz de onde veio cada escore; esta janela é onde se confere a
comparação, e num confronto é a figura do exame: as duas faces no mesmo
tamanho, os cinco pontos marcados em cada uma e o número entre elas.

O que os testes travam:

- os pontos vêm do DATASET, não de uma detecção feita na hora. São os que
  definiram o alinhamento de onde saiu o embedding, e portanto descrevem a
  face que o sistema de fato comparou; redetectar poderia marcar outra face
  do quadro sem ninguém notar;
- sem geometria a janela abre assim mesmo, sem marcação, e diz por quê;
- a marcação usa a MESMA transformação da imagem, que é o que mantém os
  pontos sobre a face em qualquer tamanho de janela;
- o duplo clique encontra a linha pelo item e não pelo índice visual, senão
  ordenar a tabela passaria a abrir a comparação errada.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from faceaudit.services.score_comparison_table_service import (
    CLASSE_GENUINO,
    ComparisonRow,
)


@pytest.fixture(scope="module")
def app():
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_CAIXA = (10.0, 20.0, 110.0, 150.0)
_PONTOS = ((30, 50), (80, 50), (55, 80), (35, 110), (75, 110))


def _png(caminho, largura=200, altura=300):
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (largura, altura), (180, 150, 130)).save(caminho)
    return str(caminho)


def _linha(tmp_path, *, com_geometria=True, com_imagem=True, log_lr=None):
    a = _png(tmp_path / "a.png") if com_imagem else None
    b = _png(tmp_path / "b.png") if com_imagem else None
    return ComparisonRow(
        probe_identity="P1", probe_face="a.png",
        reference_identity="P1", reference_face="b.png",
        score=0.8123, same_person=True, log_lr=log_lr, classe=CLASSE_GENUINO,
        probe_hash="h1", reference_hash="h2",
        probe_image_path=a, reference_image_path=b,
        probe_bbox=_CAIXA if com_geometria else None,
        probe_keypoints=_PONTOS if com_geometria else None,
        reference_bbox=_CAIXA if com_geometria else None,
        reference_keypoints=_PONTOS if com_geometria else None,
    )


def _dialogo(linha, parent=None):
    from interface.tabs.strategy_manager_v2.strategy_editor.face_comparison_dialog import (  # noqa: E501
        FaceComparisonDialog,
    )

    return FaceComparisonDialog(linha, parent=parent)


# ── O conteúdo ───────────────────────────────────────────────────────────

def test_mostra_as_duas_faces(app, tmp_path):
    d = _dialogo(_linha(tmp_path))
    try:
        assert d._probe._tela._pixmap is not None
        assert d._referencia._tela._pixmap is not None
    finally:
        d.deleteLater()


def test_a_marcacao_vem_do_dataset(app, tmp_path):
    """Não de uma detecção feita agora: estes são os pontos que definiram
    o alinhamento de onde saiu o embedding comparado."""
    d = _dialogo(_linha(tmp_path))
    try:
        assert d._probe._tela._keypoints == _PONTOS
        assert d._probe._tela._bbox == _CAIXA
    finally:
        d.deleteLater()


def test_o_escore_aparece_em_convencao_brasileira(app, tmp_path):
    from PyQt6 import QtWidgets

    d = _dialogo(_linha(tmp_path))
    try:
        textos = [w.text() for w in d.findChildren(QtWidgets.QLabel)]
        assert any("0,8123" in t for t in textos), textos
    finally:
        d.deleteLater()


def test_log_lr_so_aparece_quando_existe(app, tmp_path):
    from PyQt6 import QtWidgets

    sem = _dialogo(_linha(tmp_path))
    com = _dialogo(_linha(tmp_path, log_lr=2.5))
    try:
        def _rotulos(d):
            return " ".join(w.text() for w in d.findChildren(QtWidgets.QLabel))

        assert "log-LR" not in _rotulos(sem)
        assert "log-LR" in _rotulos(com)
    finally:
        sem.deleteLater()
        com.deleteLater()


def test_a_legenda_traz_identidade_e_arquivo(app, tmp_path):
    d = _dialogo(_linha(tmp_path))
    try:
        assert "P1" in d._probe._legenda.text()
        assert "a.png" in d._probe._legenda.text()
    finally:
        d.deleteLater()


# ── Sem geometria e sem imagem ───────────────────────────────────────────

def test_sem_geometria_abre_e_explica(app, tmp_path):
    """Melhor sem marcação do que marcada por uma detecção feita agora, que
    poderia apontar outra face do quadro."""
    from PyQt6 import QtWidgets

    d = _dialogo(_linha(tmp_path, com_geometria=False))
    try:
        assert d._probe._tela._pixmap is not None
        assert d._probe._tela._keypoints is None
        assert not d._check_marcar.isEnabled()
        rodape = " ".join(w.text() for w in d.findChildren(QtWidgets.QLabel))
        assert "não guarda os pontos" in rodape
    finally:
        d.deleteLater()


def test_sem_imagem_abre_e_explica(app, tmp_path):
    d = _dialogo(_linha(tmp_path, com_imagem=False))
    try:
        assert d._probe._tela._pixmap is None
        assert d._probe._tela._mensagem
    finally:
        d.deleteLater()


def test_imagem_fora_do_caminho_registrado(app, tmp_path):
    linha = _linha(tmp_path)
    from dataclasses import replace

    sumiu = replace(linha, probe_image_path=str(tmp_path / "nao_existe.png"))

    d = _dialogo(sumiu)
    try:
        assert "caminho" in d._probe._tela._mensagem.lower()
    finally:
        d.deleteLater()


# ── A marcação acompanha a imagem ────────────────────────────────────────

def test_sem_enquadramento_a_imagem_inteira_cabe(app, tmp_path):
    """A marcação usa a MESMA transformação da imagem, e é isso que mantém
    os pontos sobre a face em qualquer tamanho de janela. Escalas
    independentes afastariam os pontos ao redimensionar."""
    d = _dialogo(_linha(tmp_path))
    try:
        d._check_enquadrar.setChecked(False)
        tela = d._probe._tela
        tela.resize(400, 400)   # imagem é 200×300
        escala, dx, dy = tela._transformacao()

        # Cabe pela altura: 400/300, e centraliza na horizontal.
        assert escala == pytest.approx(400 / 300)
        assert dx == pytest.approx((400 - 200 * escala) / 2)
        assert dy == pytest.approx(0)
    finally:
        d.deleteLater()


def test_o_enquadramento_na_face_vem_ligado(app, tmp_path):
    """É o que a janela existe para fazer: as duas faces comparáveis a
    olho."""
    d = _dialogo(_linha(tmp_path))
    try:
        assert d._check_enquadrar.isChecked()
        assert d._probe._tela._enquadrar_na_face
    finally:
        d.deleteLater()


def test_o_enquadramento_amplia_a_face(app, tmp_path):
    d = _dialogo(_linha(tmp_path))
    try:
        tela = d._probe._tela
        tela.resize(400, 400)
        com, _, _ = tela._transformacao()
        tela.set_enquadrar_na_face(False)
        sem, _, _ = tela._transformacao()

        assert com > sem
    finally:
        d.deleteLater()


def test_as_duas_faces_saem_do_mesmo_tamanho(app, tmp_path):
    """O ponto do enquadramento: uma foto de documento e um quadro de
    vídeo têm resoluções e distâncias completamente diferentes, e sem isso
    os rostos aparecem lado a lado em tamanhos incomparáveis."""
    from dataclasses import replace

    pequena = _png(tmp_path / "peq.png", 200, 200)
    grande = _png(tmp_path / "gra.png", 1200, 1200)
    # A mesma face, proporcionalmente, em duas resoluções bem diferentes.
    caixa_peq = (60.0, 60.0, 100.0, 110.0)
    caixa_gra = (360.0, 360.0, 600.0, 660.0)

    linha = replace(
        _linha(tmp_path),
        probe_image_path=pequena, probe_bbox=caixa_peq,
        reference_image_path=grande, reference_bbox=caixa_gra,
    )
    d = _dialogo(linha)
    try:
        for tela in (d._probe._tela, d._referencia._tela):
            tela.resize(400, 400)

        def _altura_da_face(tela, caixa):
            escala, _, _ = tela._transformacao()
            return (caixa[3] - caixa[1]) * escala

        alt_p = _altura_da_face(d._probe._tela, caixa_peq)
        alt_r = _altura_da_face(d._referencia._tela, caixa_gra)

        assert alt_p == pytest.approx(alt_r, rel=0.02), (alt_p, alt_r)
    finally:
        d.deleteLater()


def test_sem_caixa_nos_dois_lados_nao_oferece_enquadramento(app, tmp_path):
    d = _dialogo(_linha(tmp_path, com_geometria=False))
    try:
        assert not d._check_enquadrar.isEnabled()
        assert not d._check_enquadrar.isChecked()
    finally:
        d.deleteLater()


def test_o_desenho_nao_quebra_sem_imagem(app, tmp_path):
    """O `paintEvent` roda com placeholder; um erro ali derrubaria a
    janela inteira."""
    from PyQt6 import QtGui

    d = _dialogo(_linha(tmp_path, com_imagem=False))
    try:
        pixmap = QtGui.QPixmap(300, 300)
        d._probe._tela.render(pixmap)   # não deve levantar
    finally:
        d.deleteLater()


def test_o_desenho_roda_com_marcacao(app, tmp_path):
    from PyQt6 import QtGui

    d = _dialogo(_linha(tmp_path))
    try:
        pixmap = QtGui.QPixmap(300, 300)
        d._probe._tela.render(pixmap)
        d._on_marcar(False)
        d._probe._tela.render(pixmap)
    finally:
        d.deleteLater()


# ── O caminho até a janela ───────────────────────────────────────────────

def test_a_tabela_guarda_o_indice_do_backend(app, tmp_path):
    """A tabela é ordenável: o índice visual deixa de corresponder ao da
    lista assim que o usuário clica num cabeçalho, e abrir a comparação
    errada seria pior que não abrir."""
    from PyQt6 import QtCore

    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets.score_comparison_table_widget import (  # noqa: E501
        _PAPEL_INDICE_DA_LINHA,
    )
    from faceaudit.services.score_comparison_table_service import ComparisonTable

    linhas = []
    from dataclasses import replace

    base = _linha(tmp_path)
    for i, escore in enumerate((0.1, 0.9, 0.5)):
        linhas.append(replace(base, score=escore, probe_face=f"f{i}.png"))
    tabela = ComparisonTable(
        rows=tuple(linhas), mode="extremos", total_pairs=3,
        n_genuines=3, n_impostors=0, truncated=False, has_log_lr=False,
        has_file_names=True, note="nota", has_images=True,
    )

    w = ScoreComparisonTableWidget()
    try:
        w.set_table(tabela)
        coluna_escore = next(
            i for i in range(w._grid.columnCount())
            if w._grid.horizontalHeaderItem(i).text() == "Escore"
        )
        w._grid.sortItems(coluna_escore, QtCore.Qt.SortOrder.AscendingOrder)

        # Depois de ordenar, a primeira linha visual é a de menor escore,
        # que no backend é o índice 0.
        indice = w._grid.item(0, 0).data(_PAPEL_INDICE_DA_LINHA)
        assert tabela.rows[int(indice)].score == pytest.approx(0.1)

        # E a última é a de maior escore.
        ultimo = w._grid.item(w._grid.rowCount() - 1, 0).data(
            _PAPEL_INDICE_DA_LINHA
        )
        assert tabela.rows[int(ultimo)].score == pytest.approx(0.9)
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_duplo_clique_sem_tabela_nao_quebra(app):
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    w = ScoreComparisonTableWidget()
    try:
        w._on_linha_ativada(0, 0)   # não deve levantar
    finally:
        w.deleteLater()


# ── A geometria chega do dataset até a linha ─────────────────────────────

def test_a_geometria_do_parquet_chega_a_linha(tmp_path):
    """Prova do caminho inteiro: o dataset guarda a geometria, o localizador
    a lê pelo hash e a tabela a entrega pronta para desenhar."""
    import numpy as np

    from faceaudit.enums.path_mode import PathMode
    from faceaudit.models.embedding_info import EmbeddingInfo
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo
    from faceaudit.models.ff_info import FFInfo
    from faceaudit.models.similarity.similarity_score_dataset import (
        SimilarityScoreDataset,
    )
    from faceaudit.models.similarity.similarity_score_sample import (
        SimilarityScoreSample,
    )
    from faceaudit.services.score_comparison_table_service import (
        attach_face_images, build_from_parquet,
    )

    imagem = _png(tmp_path / "imgs" / "P1.png")
    face = FaceInfo(image_path=imagem)
    face.identity = "P1"
    face.file_name = "P1.png"
    face.hash = "h1"
    face.ff_info = FFInfo(
        bbox=np.asarray(_CAIXA, dtype=np.float32),
        keypoints=np.asarray(_PONTOS, dtype=np.float32),
        embedding_sepaelv2=np.ones(4, dtype=np.float32),
    )
    ds_faces = tmp_path / "ds.parquet"
    FaceDataset(
        name="ds", faces=[face], path_mode=PathMode.ABSOLUTE,
    ).to_parquet(ds_faces)

    def _emb(h):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=h,
            image_path="", identity="P1", file_name="",
        )

    escores = tmp_path / "escores.parquet"
    SimilarityScoreDataset(
        name="s",
        items=[SimilarityScoreSample(
            probe=_emb("h1"), reference=_emb("h1"),
            raw_score=0.9, same_person=True,
        )],
    ).to_parquet(escores)

    tabela = attach_face_images(build_from_parquet(escores), [ds_faces])

    linha = tabela.rows[0]
    assert linha.tem_geometria
    assert linha.probe_keypoints == _PONTOS
    assert tuple(linha.probe_bbox) == _CAIXA


def test_dataset_sem_geometria_nao_promete_marcacao(tmp_path):
    """Todos os datasets anteriores a esta mudança estão nessa situação."""
    import numpy as np

    from faceaudit.enums.path_mode import PathMode
    from faceaudit.models.embedding_info import EmbeddingInfo
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo
    from faceaudit.models.ff_info import FFInfo
    from faceaudit.models.similarity.similarity_score_dataset import (
        SimilarityScoreDataset,
    )
    from faceaudit.models.similarity.similarity_score_sample import (
        SimilarityScoreSample,
    )
    from faceaudit.services.score_comparison_table_service import (
        attach_face_images, build_from_parquet,
    )

    face = FaceInfo(image_path=_png(tmp_path / "imgs" / "P1.png"))
    face.identity = "P1"
    face.file_name = "P1.png"
    face.hash = "h1"
    face.ff_info = FFInfo(embedding_sepaelv2=np.ones(4, dtype=np.float32))
    ds_faces = tmp_path / "ds.parquet"
    FaceDataset(
        name="ds", faces=[face], path_mode=PathMode.ABSOLUTE,
    ).to_parquet(ds_faces)

    def _emb(h):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=h,
            image_path="", identity="P1", file_name="",
        )

    escores = tmp_path / "escores.parquet"
    SimilarityScoreDataset(
        name="s",
        items=[SimilarityScoreSample(
            probe=_emb("h1"), reference=_emb("h1"),
            raw_score=0.9, same_person=True,
        )],
    ).to_parquet(escores)

    tabela = attach_face_images(build_from_parquet(escores), [ds_faces])

    assert tabela.has_images          # a imagem foi achada
    assert not tabela.rows[0].tem_geometria   # a marcação, não
