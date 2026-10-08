"""Duas distribuições na mesma figura usam a MESMA grade de faixas.

O defeito: a figura de contextualização do caso pedia um número de faixas
por classe, pela raiz do tamanho de cada uma, e deixava cada histograma
usar o intervalo da própria classe. No caso que o expôs, 374 comparações
de mesma pessoa espalhadas por [-0,06; 0,99] contra 106.222 de pessoas
diferentes em [-0,20; 0,45]: as barras de mesma pessoa saíam 4,28 vezes
mais largas, desalinhadas das outras, e das 19 faixas pedidas só umas 8
tinham conteúdo. A mesma referência, desenhada pela instrução de escores,
aparecia como distribuição inteira — duas figuras dos mesmos números, com
cara diferente.

Com `density=True` a altura é contagem dividida por N e pela largura da
faixa, então larguras diferentes produzem alturas que não se comparam.
Numa figura de laudo isso é apresentar como comparável o que não é.

A intenção da regra por classe era suavizar o grupo pequeno para que o
serrilhado amostral não fosse lido como estrutura. O preço era
comparabilidade, que aqui vale mais: grupo escasso parece escasso, e isso
é verdade sobre ele.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.models.plot_histogram_bins import (
    N_ARESTAS,
    arestas_compartilhadas,
)


# ── A grade ──────────────────────────────────────────────────────────────

def test_cobre_todas_as_series():
    a = arestas_compartilhadas([0.0, 0.5], [-0.2, 0.1], [0.9])

    assert a[0] == pytest.approx(-0.2)
    assert a[-1] == pytest.approx(0.9)
    assert a.size == N_ARESTAS


def test_serie_vazia_e_ignorada():
    a = arestas_compartilhadas([0.0, 1.0], [], None)

    assert a[0] == pytest.approx(0.0) and a[-1] == pytest.approx(1.0)


def test_todas_vazias_devolve_grade_vazia():
    assert arestas_compartilhadas([], None).size == 0


def test_valor_unico_abre_o_intervalo():
    """`np.linspace` com extremos iguais devolveria arestas coincidentes e
    o histograma ficaria indefinido."""
    a = arestas_compartilhadas([0.7, 0.7, 0.7])

    assert a[0] < 0.7 < a[-1]


def test_a_largura_e_constante():
    a = arestas_compartilhadas([-0.2, 0.99])
    larguras = np.diff(a)

    assert np.allclose(larguras, larguras[0])


# ── A figura de caso ─────────────────────────────────────────────────────

class _Resultado:
    def __init__(self, gen, imp, caso):
        self.genuine_scores = gen
        self.impostor_scores = imp
        self.case_scores = caso
        self.score_type = "RAW"
        self.eer_threshold = None
        self.case_summary = {}
        self.metadata = {}


def _desenhar(gen, imp, caso):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from faceaudit.models.casework_visualization_plot import (
        draw_casework_visualization,
    )

    fig = Figure(figsize=(8, 4))
    draw_casework_visualization(fig, _Resultado(gen, imp, caso))
    return fig.axes[0]


def _larguras_das_barras(ax):
    return sorted({
        round(p.get_width(), 6)
        for p in ax.patches
        if getattr(p, "get_height", None) and p.get_height() > 0
    })


def test_as_duas_classes_tem_a_mesma_largura_de_barra():
    """A proporção do caso real: grupo pequeno e espalhado contra grupo
    grande e estreito. Era onde as larguras divergiam 4,28 vezes."""
    rng = np.random.default_rng(7)
    gen = np.concatenate([rng.normal(0.88, 0.05, 360), [-0.06, 0.1, 0.3]])
    imp = rng.normal(0.05, 0.08, 20000)

    ax = _desenhar(gen, imp, [0.66])

    assert len(_larguras_das_barras(ax)) == 1, _larguras_das_barras(ax)


def test_o_grupo_pequeno_nao_perde_resolucao():
    """Eram 8 barras onde a grade compartilhada dá dezenas."""
    rng = np.random.default_rng(7)
    gen = np.concatenate([rng.normal(0.88, 0.05, 360), [-0.06, 0.1, 0.3]])
    imp = rng.normal(0.05, 0.08, 20000)

    ax = _desenhar(gen, imp, [0.66])

    # Barras verdes: as de mesma pessoa. Contadas pela cor do rótulo da
    # legenda seria frágil; o que importa é o total de barras visíveis.
    visiveis = [
        p for p in ax.patches
        if getattr(p, "get_height", None) and p.get_height() > 0
    ]
    assert len(visiveis) > 30, len(visiveis)


def test_a_grade_cobre_as_tres_series():
    ax = _desenhar([0.9], [0.0], [0.66])

    # Sem dados suficientes para histograma, mas o eixo tem de alcançar
    # todas as séries.
    inicio, fim = ax.get_xlim()
    assert inicio < 0.0 and fim > 0.9


def test_sem_genuinos_nao_quebra():
    rng = np.random.default_rng(1)
    ax = _desenhar([], rng.normal(0.05, 0.08, 500), [0.6])

    assert _larguras_das_barras(ax)


def test_sem_nada_nao_quebra():
    ax = _desenhar([], [], [])

    assert not _larguras_das_barras(ax)


# ── As duas figuras dos mesmos escores concordam ────────────────────────

def test_o_sidecar_usa_a_mesma_grade():
    """A outra figura dos mesmos escores, a da instrução de escores, sai do
    sidecar de distribuição. Se as grades divergirem, as duas voltam a ter
    cara diferente."""
    from faceaudit.models.embedding_info import EmbeddingInfo
    from faceaudit.models.similarity.similarity_score_dataset import (
        SimilarityScoreDataset,
    )
    from faceaudit.models.similarity.similarity_score_sample import (
        SimilarityScoreSample,
    )

    rng = np.random.default_rng(3)
    gen = rng.normal(0.88, 0.05, 50)
    imp = rng.normal(0.05, 0.08, 500)

    def _face(i):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=f"h{i}",
            image_path="", identity=f"P{i}", file_name="",
        )

    itens = []
    for i, v in enumerate(gen):
        a = SimilarityScoreSample(
            probe=_face(i), reference=_face(i), raw_score=float(v),
            same_person=True,
        )
        a.same_person = True
        itens.append(a)
    for i, v in enumerate(imp):
        a = SimilarityScoreSample(
            probe=_face(f"a{i}"), reference=_face(f"b{i}"),
            raw_score=float(v), same_person=False,
        )
        a.same_person = False
        itens.append(a)

    resumo = SimilarityScoreDataset(
        name="s", items=itens,
    )._compute_distribution_summary()

    esperada = arestas_compartilhadas(gen, imp, [])
    assert np.allclose(np.asarray(resumo["bins"]), esperada)


def test_a_regra_por_classe_saiu_de_cena():
    """Enquanto ela existir, alguém a chama de novo."""
    from faceaudit.models import casework_visualization_plot as mod

    assert not hasattr(mod, "_faixas_para")
