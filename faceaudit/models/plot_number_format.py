"""Números em convenção brasileira nas figuras: vírgula decimal.

As figuras vão para laudo em português, onde o separador decimal é vírgula.
O matplotlib escreve ponto, e o `f"{v:.4f}"` do Python também.

Tratar isso só na legenda e deixar as marcas do eixo com ponto é pior que
não tratar: a mesma figura passaria a exibir as duas convenções, e a
discrepância chama mais atenção que o ponto uniforme. Daí a função que
ajusta os dois eixos de uma vez, e não um `replace` espalhado por cada
rótulo.
"""

from __future__ import annotations

from typing import Any

_MARCADOR = "\x00"


def formatar_numero(valor: float, decimais: int = 4) -> str:
    """Número em convenção brasileira: milhar com ponto, decimal com vírgula.

    A troca é feita em dois passos com um marcador intermediário. O caminho
    ingênuo, trocar ponto por vírgula direto, corromperia valores que já
    têm separador de milhar: "1,234.50" viraria "1,234,50".
    """
    try:
        texto = f"{float(valor):,.{int(decimais)}f}"
    except (TypeError, ValueError):
        return str(valor)
    return texto.replace(",", _MARCADOR).replace(".", ",").replace(_MARCADOR, ".")


def formatar_tick(valor: float, _pos: Any = None) -> str:
    """Rótulo de marca de eixo, sem zeros à direita.

    Precisão fixa em marca de eixo polui ("0,2000" em vez de "0,2"), e a
    precisão que o matplotlib escolheria depende de interna dele. Quatro
    decimais cobrem a faixa de escores e de log-LR do sistema; o que não
    precisar sai limpo.
    """
    try:
        texto = f"{float(valor):.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(valor)
    if texto in ("", "-", "-0"):
        texto = "0"
    return texto.replace(".", ",")


def aplicar_formato_brasileiro(ax) -> None:
    """Põe vírgula decimal nas marcas dos DOIS eixos do `ax`.

    Chamar depois de definir os limites, porque o matplotlib escolhe as
    marcas a partir deles.
    """
    from matplotlib.ticker import FuncFormatter

    formatador = FuncFormatter(formatar_tick)
    ax.xaxis.set_major_formatter(formatador)
    ax.yaxis.set_major_formatter(formatador)
