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

    for grupo in ("Impostores", "Genuínos", "Caso"):
        casou = [r for r in rotulos if grupo in r]
        assert casou, f"{grupo} ausente da legenda: {rotulos}"
        assert "n=" in casou[0], f"sem n na legenda de {grupo}: {casou[0]}"
