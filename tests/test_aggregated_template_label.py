"""Template agregado aparece como template, não como foto que sumiu.

`EMBEDDING_AGGREGATION` funde N faces da mesma identidade num template. O
resultado não é uma foto: `image_path` sai vazio de propósito, e o
`face_hash` vira um digest do conjunto ordenado de faces contribuintes,
com prefixo que o distingue de qualquer imagem real.

Na tabela de comparações isso aparecia mal de duas formas. A célula ficava
vazia, como se a imagem tivesse sumido, ou mostrava o nome que a agregação
inventou, `AGG_P1_k3_<digest>`, que não é nome de arquivo nenhum e mandaria
o perito procurar uma foto que nunca existiu.

O digest é de mão única: a lista de faces não fica gravada, então não há
como mostrá-las. O que dá para recuperar é QUANTAS eram, porque a
agregação escreve o número no nome.

O que os testes travam:

- o prefixo não pode divergir do que a agregação de fato usa;
- o rótulo sai do nome, e quando o nome não revela o número a linha diz
  "Template agregado" em vez de chutar uma contagem;
- a linha continua aparecendo, com escore e tudo — é comparação válida,
  só não tem foto;
- a dica do mouse e o painel da janela dizem o que o objeto é, em vez de
  "imagem não encontrada".
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.services.aggregated_template_identity import (
    SYNTH_PREFIX,
    contributing_faces,
    is_synthetic_template,
    template_label,
    template_tooltip,
)
from faceaudit.services.score_comparison_table_service import build_from_dataset


@pytest.fixture(scope="module")
def app():
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


_HASH_TEMPLATE = SYNTH_PREFIX + "a" * 64
_NOME_TEMPLATE = "AGG_P1_k3_abc123def456"


# ── O prefixo não pode divergir ──────────────────────────────────────────

def test_o_prefixo_e_o_mesmo_da_agregacao():
    """Sentinela: o prefixo está duplicado aqui porque o da agregação é
    privado e o módulo é caro de importar. Se um mudar sem o outro, o
    rótulo some sem erro nenhum e o template volta a parecer foto."""
    from faceaudit.services.embedding_math_service import _SYNTH_PREFIX

    assert SYNTH_PREFIX == _SYNTH_PREFIX


def test_reconhece_pelo_prefixo():
    assert is_synthetic_template(_HASH_TEMPLATE)
    assert not is_synthetic_template("a" * 64)
    assert not is_synthetic_template("")
    assert not is_synthetic_template(None)


# ── O número vem do nome ─────────────────────────────────────────────────

@pytest.mark.parametrize("nome,esperado", [
    ("AGG_P1_k3_abc123", 3),
    ("AGG_identidade com espaço_k12_deadbeef", 12),
    ("AGG_P1_k1_abc", 1),
])
def test_le_quantas_faces(nome, esperado):
    assert contributing_faces(nome) == esperado


@pytest.mark.parametrize("nome", ["", None, "foto.png", "AGG_P1_kx_abc", "AGG_P1_k0_a"])
def test_nome_que_nao_revela_o_numero(nome):
    assert contributing_faces(nome) is None


def test_o_rotulo_diz_quantas():
    assert template_label(_HASH_TEMPLATE, _NOME_TEMPLATE) == "Template de 3 faces"


def test_singular_quando_e_uma_face_so():
    assert template_label(_HASH_TEMPLATE, "AGG_P1_k1_abc") == "Template de 1 face"


def test_sem_o_numero_nao_chuta():
    """Inventar uma contagem seria pior que admitir que não se sabe."""
    assert template_label(_HASH_TEMPLATE, None) == "Template agregado"
    assert template_label(_HASH_TEMPLATE, "nome_estranho") == "Template agregado"


def test_foto_nao_ganha_rotulo_de_template():
    """O `None` é o que faz quem chama seguir com o nome de arquivo."""
    assert template_label("a" * 64, "foto.png") is None


# ── A tabela ─────────────────────────────────────────────────────────────

def _par(hash_probe, nome_probe, hash_ref, nome_ref, escore=0.8):
    def _face(h, n):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=h,
            image_path="", identity="P1", file_name=n,
        )

    amostra = SimilarityScoreSample(
        probe=_face(hash_probe, nome_probe),
        reference=_face(hash_ref, nome_ref),
        raw_score=escore, same_person=True,
    )
    amostra.same_person = True
    return amostra


def test_a_tabela_rotula_o_template():
    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )

    t = build_from_dataset(ds)

    linha = t.rows[0]
    assert linha.probe_face == "Template de 3 faces"
    assert linha.probe_is_template
    assert linha.probe_template_faces == 3


def test_o_outro_lado_continua_sendo_foto():
    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )

    linha = build_from_dataset(ds).rows[0]

    assert linha.reference_face == "foto.png"
    assert not linha.reference_is_template
    assert linha.reference_template_faces is None


def test_a_comparacao_de_template_nao_some_da_tabela():
    """É comparação válida, com escore e tudo; só não tem foto."""
    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, _HASH_TEMPLATE, "AGG_P2_k5_x")],
    )

    t = build_from_dataset(ds)

    assert len(t.rows) == 1
    assert t.rows[0].score == pytest.approx(0.8)
    assert t.rows[0].reference_face == "Template de 5 faces"


def test_template_nao_promete_miniatura():
    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, _HASH_TEMPLATE, "AGG_P2_k5_x")],
    )

    t = build_from_dataset(ds)

    assert not t.has_images
    assert t.rows[0].probe_image_path is None


def test_lido_do_parquet_o_template_e_reconhecido(tmp_path):
    """Do parquet de escores só vem o hash, sem nome: dá para saber QUE é
    template, não de quantas faces."""
    from faceaudit.services.score_comparison_table_service import (
        build_from_parquet,
    )

    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )
    caminho = tmp_path / "escores.parquet"
    ds.to_parquet(caminho)

    linha = build_from_parquet(caminho).rows[0]

    assert linha.probe_is_template
    assert linha.probe_face == "Template agregado"
    assert linha.probe_template_faces == 0


def test_o_dataset_agregado_devolve_o_numero(tmp_path):
    """O nome, e com ele a contagem, está no parquet do dataset agregado,
    que é um EmbeddingDataset — o localizador tem de aceitar a coluna
    `face_hash` dele, e não só a `hash` do FaceDataset."""
    from faceaudit.models.embedding_dataset import EmbeddingDataset
    from faceaudit.services.score_comparison_table_service import (
        attach_face_images, build_from_parquet,
    )

    agregado = tmp_path / "templates.parquet"
    EmbeddingDataset(
        name="agg",
        items=[EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=_HASH_TEMPLATE,
            image_path="", identity="P1", file_name=_NOME_TEMPLATE,
        )],
    ).to_parquet(agregado)

    escores = tmp_path / "escores.parquet"
    SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    ).to_parquet(escores)

    t = attach_face_images(build_from_parquet(escores), [agregado])

    assert t.rows[0].probe_face == "Template de 3 faces"
    assert t.rows[0].probe_template_faces == 3
    # E continua sem imagem: `image_path` vazio não pode virar a pasta do
    # parquet e fazer a linha passar por face com foto.
    assert t.rows[0].probe_image_path is None


# ── A interface explica ──────────────────────────────────────────────────

def test_a_dica_do_mouse_diz_o_que_e(app):
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )
    w = ScoreComparisonTableWidget()
    try:
        w.set_table(build_from_dataset(ds))
        coluna = next(
            i for i in range(w._grid.columnCount())
            if w._grid.horizontalHeaderItem(i).text() in ("Probe", "Face (probe)")
        )
        dica = w._grid.item(0, coluna).toolTip()

        assert "Template" in dica
        assert "não corresponde a imagem nenhuma" in dica.lower()
        assert "3 faces" in dica
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_a_celula_mostra_o_rotulo(app):
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )
    w = ScoreComparisonTableWidget()
    try:
        w.set_table(build_from_dataset(ds))
        textos = [
            w._grid.item(0, c).text() for c in range(w._grid.columnCount())
        ]
        assert any("Template de 3 faces" in t for t in textos), textos
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_a_janela_explica_em_vez_de_dizer_imagem_faltando(app, tmp_path):
    """Dizer "imagem não encontrada" mandaria o perito caçar um arquivo
    que nunca existiu."""
    from interface.tabs.strategy_manager_v2.strategy_editor.face_comparison_dialog import (  # noqa: E501
        FaceComparisonDialog,
    )

    ds = SimilarityScoreDataset(
        name="s",
        items=[_par(_HASH_TEMPLATE, _NOME_TEMPLATE, "b" * 64, "foto.png")],
    )
    linha = build_from_dataset(ds).rows[0]

    d = FaceComparisonDialog(linha)
    try:
        mensagem = d._probe._tela._mensagem
        assert "Template" in mensagem
        assert "média dos embeddings" in mensagem
        assert "não encontrad" not in mensagem.lower()
    finally:
        d.deleteLater()


def test_a_explicacao_sem_numero_nao_inventa():
    dica = template_tooltip(_HASH_TEMPLATE, None)

    assert "Template agregado" in dica
    assert "faces da mesma identidade" not in dica
