"""Rótulos de eixo dos gráficos que vão para laudo.

Esses rótulos são lidos por quem não conhece o nome interno das colunas, e
foi por aí que escapou um `raw_score` no eixo X da visualização forense de
caso: identificador de código exposto numa figura de laudo, enquanto os
demais gráficos do app já diziam "Escore Bruto".

Travar o rótulo em teste é barato e pega a regressão exata que aconteceu:
alguém traduz um arquivo e deixa outro para trás.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pytest

from matplotlib.figure import Figure

from faceaudit.models.casework_visualization_plot import (
    draw_casework_visualization,
)


@dataclass
class _ResultadoFalso:
    """Mínimo que o renderizador consome."""

    score_type: str = "RAW"
    genuine_scores: Any = field(
        default_factory=lambda: np.random.default_rng(1).normal(0.7, 0.08, 400),
    )
    impostor_scores: Any = field(
        default_factory=lambda: np.random.default_rng(2).normal(0.05, 0.06, 4000),
    )
    case_scores: Any = field(default_factory=lambda: np.array([0.53, 0.56]))
    eer_threshold: Optional[float] = 0.35
    case_summary: str = ""
    n_case: int = 2


def _desenhar(**kwargs) -> Any:
    fig = Figure(figsize=(6, 4))
    draw_casework_visualization(fig, _ResultadoFalso(**kwargs))
    return fig.axes[0]


def test_eixo_x_em_escore_bruto_nao_expoe_nome_de_coluna():
    """`raw_score` é nome de coluna, não rótulo de figura de laudo."""
    rotulo = _desenhar(score_type="RAW").get_xlabel()

    assert rotulo == "Escore de similaridade"
    assert "raw" not in rotulo.lower()
    assert "_" not in rotulo, "nome de identificador vazou para o eixo"


def test_eixo_x_muda_com_o_tipo_de_escore():
    """Escore calibrado não é escore de similaridade: é log-LR, e o eixo
    tem de dizer isso, senão a figura afirma a grandeza errada."""
    assert _desenhar(score_type="CALIBRATED").get_xlabel() == "log-LR calibrado"


def test_nenhum_rotulo_de_eixo_tem_identificador_de_codigo():
    """Varredura: sublinhado ou minúscula-com-sublinhado em rótulo de eixo é
    sinal de nome interno exposto."""
    for tipo in ("RAW", "CALIBRATED"):
        ax = _desenhar(score_type=tipo)
        for rotulo in (ax.get_xlabel(), ax.get_ylabel()):
            assert rotulo, "eixo sem rótulo"
            assert "_" not in rotulo, f"identificador exposto: {rotulo!r}"


# ── Padronização entre todos os gráficos de distribuição ─────────────────

_X_ESPERADO = "Escore de similaridade"
_Y_ESPERADO = "Densidade de frequência"


def test_todos_os_graficos_de_distribuicao_usam_o_mesmo_rotulo():
    """Antes desta padronização o MESMO eixo tinha três grafias no app:
    `raw_score`, "Escore Bruto" e "Score bruto", esta última meio em inglês.
    Como o laudo junta figuras de telas diferentes, elas apareciam lado a
    lado com nomes distintos para a mesma grandeza.
    """
    from pathlib import Path

    arquivos = [
        "faceaudit/models/casework_visualization_plot.py",
        "faceaudit/models/similarity/distribution_plot.py",
        "faceaudit/models/similarity/calibration_curve_plot.py",
        "faceaudit/models/validation_result_plot.py",
        "faceaudit/models/typicality_distribution_plot.py",
        "interface/tabs/strategy_manager_v2/strategy_editor/result_widgets"
        "/score_distribution_widget.py",
        "interface/tabs/strategy_manager_v2/strategy_manager_v2_controller.py",
    ]
    proibidos = ('"Escore Bruto"', '"Score bruto"', '"Densidade"', '"raw_score"')

    for caminho in arquivos:
        fonte = Path(caminho).read_text(encoding="utf-8")
        for termo in proibidos:
            # Ignora menções em comentário, que documentam o histórico.
            linhas = [
                l for l in fonte.splitlines()
                if termo in l and not l.strip().startswith("#")
            ]
            assert not linhas, (
                f"{caminho} ainda usa {termo} fora de comentário: {linhas[:1]}"
            )


def test_eixo_y_nomeia_a_grandeza_e_nao_e_porcentagem():
    """"Densidade de frequência" é frequência relativa sobre amplitude de
    classe, que é o que `density=True` produz. Marcar o eixo como `%` seria
    incorreto: medido no caso real, o pico chega a 5,40, o que leria 540%.
    """
    ax = _desenhar()

    assert ax.get_ylabel() == _Y_ESPERADO
    assert "%" not in ax.get_ylabel()


def test_legenda_informa_o_n_de_cada_grupo():
    """Como cada distribuição é normalizada separadamente, a ALTURA não diz
    quantas comparações cada grupo tem. O n na legenda é o que impede o
    leitor de concluir que os grupos têm peso comparável.
    """
    ax = _desenhar()
    rotulos = [t.get_text() for t in ax.get_legend().get_texts()]

    # Vocabulário do LEITOR DO LAUDO, não do perito: "genuíno" e "impostor"
    # são termos do campo e exigiriam glossário na peça.
    for grupo in ("Pessoas diferentes", "Mesma pessoa", "Caso"):
        casou = [r for r in rotulos if grupo in r]
        assert casou, f"{grupo} ausente da legenda: {rotulos}"
        assert "n=" in casou[0], f"sem n na legenda de {grupo}: {casou[0]}"

    texto = " ".join(rotulos)
    for termo in ("Impostor", "Genuíno", "Genuínos", "threshold", "EER ="):
        assert termo not in texto, (
            f"termo de uso interno ou em inglês na legenda: {termo!r}"
        )


# ── Números em convenção brasileira ──────────────────────────────────────

def test_decimal_com_virgula():
    from faceaudit.models.plot_number_format import formatar_numero

    assert formatar_numero(0.35, 4) == "0,3500"
    assert formatar_numero(0.5292, 2) == "0,53"
    assert formatar_numero(-1.5, 1) == "-1,5"


def test_milhar_com_ponto_e_decimal_com_virgula_ao_mesmo_tempo():
    """A troca ingênua de ponto por vírgula corromperia isto: "1,234.50"
    viraria "1,234,50". Daí os dois passos com marcador intermediário."""
    from faceaudit.models.plot_number_format import formatar_numero

    assert formatar_numero(1234.5, 2) == "1.234,50"
    assert formatar_numero(364742, 0) == "364.742"


def test_tick_sem_zeros_a_direita():
    """Precisão fixa polui marca de eixo: "0,2000" em vez de "0,2"."""
    from faceaudit.models.plot_number_format import formatar_tick

    assert formatar_tick(0.2) == "0,2"
    assert formatar_tick(0.0) == "0"
    assert formatar_tick(-0.5) == "-0,5"
    assert formatar_tick(3.0) == "3"


def test_tick_nao_produz_menos_zero():
    """-0,0 numa marca de eixo é artefato de arredondamento, não valor."""
    from faceaudit.models.plot_number_format import formatar_tick

    assert formatar_tick(-0.00001) == "0"


def test_valor_invalido_nao_estoura():
    from faceaudit.models.plot_number_format import formatar_numero, formatar_tick

    assert formatar_numero(None) == "None"
    assert formatar_tick("x") == "x"


def test_a_figura_toda_usa_virgula():
    """Legenda com vírgula e eixo com ponto seria pior que ponto uniforme:
    a mesma figura exibiria as duas convenções."""
    fig = Figure(figsize=(7, 4.5))
    draw_casework_visualization(fig, _ResultadoFalso())
    fig.canvas.draw()
    ax = fig.axes[0]

    textos = (
        [t.get_text() for t in ax.get_xticklabels()]
        + [t.get_text() for t in ax.get_yticklabels()]
        + [t.get_text() for t in ax.get_legend().get_texts()]
        + [a.get_text() for a in ax.texts]
    )
    com_ponto_decimal = [
        t for t in textos
        # Ponto seguido de dígito é decimal; ponto de milhar vem seguido de
        # três dígitos e precedido de dígito, e esse é legítimo.
        if any(
            t[i] == "." and i + 1 < len(t) and t[i + 1].isdigit()
            and not (i + 4 <= len(t) and t[i + 1:i + 4].isdigit()
                     and (i + 4 == len(t) or not t[i + 4].isdigit()))
            for i in range(len(t))
        )
    ]
    assert not com_ponto_decimal, (
        f"textos com ponto decimal na figura: {com_ponto_decimal}"
    )


# ── Padrões da figura de laudo ───────────────────────────────────────────

def test_padrao_e_sem_curva_suavizada():
    """A curva é um MODELO: a largura de banda não aparece na figura e
    inventa cauda onde não há observação, justamente na faixa por onde
    passam os escores do caso. O padrão mostra o dado."""
    import inspect

    from faceaudit.models.casework_visualization_plot import (
        draw_casework_visualization as f,
    )

    assert inspect.signature(f).parameters["show_kde"].default is False


def test_padrao_e_legenda_embaixo():
    """Embaixo e fora: não cobre os valores do caso nem a distribuição, e o
    desenho fica com a largura inteira."""
    import inspect

    from faceaudit.models.casework_visualization_plot import (
        draw_casework_visualization as f,
    )

    assert inspect.signature(f).parameters["legend_below"].default is True


def test_concordancia_de_numero_na_legenda_do_caso():
    """"1 marcas" numa figura de laudo salta aos olhos."""
    import numpy as np

    for n, esperado in ((1, "marca)"), (2, "marcas)"), (3, "marcas)")):
        fig = Figure()
        draw_casework_visualization(
            fig, _ResultadoFalso(case_scores=np.linspace(0.5, 0.6, n), n_case=n),
        )
        caso = [
            t.get_text() for t in fig.axes[0].get_legend().get_texts()
            if t.get_text().startswith("Caso")
        ]
        assert caso and caso[0].endswith(esperado), f"n={n}: {caso}"


def test_as_duas_classes_dividem_a_grade_de_faixas():
    """A premissa anterior era o contrário: faixas pela raiz do tamanho de
    CADA classe, para suavizar o grupo pequeno. O preço era
    comparabilidade — com `density=True`, larguras diferentes produzem
    alturas que não se comparam, e num laudo isso apresenta como
    comparável o que não é. Grupo escasso parece escasso, e isso é
    verdade sobre ele. Detalhe em `tests/test_shared_histogram_bins.py`.
    """
    ax = _desenhar(score_type="RAW")

    larguras = {
        round(p.get_width(), 6)
        for p in ax.patches
        if getattr(p, "get_height", None) and p.get_height() > 0
    }
    assert len(larguras) == 1, larguras


def test_limiar_de_eer_fica_fora_por_padrao():
    """É critério de DECISÃO derivado da referência. Numa figura de
    contextualização, convida o leitor a raciocinar por corte, que é a
    leitura binária que a contextualização existe para substituir."""
    ax = _desenhar(score_type="RAW")
    rotulos = [t.get_text() for t in ax.get_legend().get_texts()]

    assert not any("EER" in r for r in rotulos), rotulos
    assert len(rotulos) == 3


def test_limiar_de_eer_continua_disponivel():
    """Desligado por padrão, não removido: há quem queira a linha."""
    fig = Figure()
    draw_casework_visualization(
        fig, _ResultadoFalso(score_type="RAW"), show_eer_threshold=True,
    )
    rotulos = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    assert any("igual taxa de erro (EER)" in r for r in rotulos), rotulos


def test_linha_de_evidencia_neutra_permanece_no_modo_calibrado():
    """log-LR igual a zero marca evidência neutra, não critério de decisão —
    é coisa diferente do limiar de EER e continua na figura."""
    ax = _desenhar(score_type="CALIBRATED")
    rotulos = [t.get_text() for t in ax.get_legend().get_texts()]

    assert any("log-LR = 0" in r for r in rotulos), rotulos
