"""A grade de faixas do histograma, uma só por figura.

Duas distribuições na mesma figura precisam da MESMA grade. Com
`density=True` a altura da barra é contagem dividida por N e pela largura
da faixa, então faixas de larguras diferentes produzem alturas que não se
comparam — e num laudo isso é apresentar como comparável o que não é.

Era o que acontecia na figura de contextualização do caso: ela pedia um
número de faixas por classe, pela raiz do tamanho de cada uma, e deixava
cada histograma usar o intervalo da própria classe. No caso que expôs o
problema, 374 comparações de mesma pessoa espalhadas por [-0,06; 0,99]
contra 106.222 de pessoas diferentes em [-0,20; 0,45]: as barras de mesma
pessoa saíam 4,28 vezes mais largas, desalinhadas das outras, e das 19
faixas pedidas só umas 8 tinham conteúdo. A mesma referência, desenhada
pela instrução de escores, aparecia como distribuição inteira.

A intenção da regra por classe era boa, suavizar o grupo pequeno para que
o serrilhado amostral não fosse lido como estrutura do dado. Mas o preço
foi comparabilidade, que numa figura de laudo vale mais: grupo escasso
parece escasso, e isso é verdade sobre ele.

A grade cobre o intervalo dos DADOS. A folga do eixo é assunto de quem
desenha — faixa vazia nas pontas desloca as barras sem informar nada.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Número de ARESTAS, logo 59 faixas. Herdado do sidecar de distribuição,
# que já usava 60, e mantido para que as duas figuras dos mesmos escores
# tenham a mesma grade.
N_ARESTAS = 60


def arestas_compartilhadas(
    *series: Sequence[float],
    n_arestas: int = N_ARESTAS,
) -> np.ndarray:
    """Arestas das faixas cobrindo todas as séries da figura.

    Série vazia é ignorada; todas vazias devolve grade vazia, e quem
    desenha já trata o caso de "sem dados". Quando todos os valores são
    iguais o intervalo é aberto à força, senão `np.linspace` devolveria
    arestas coincidentes e o histograma ficaria indefinido.
    """
    nao_vazias = [
        np.asarray(s, dtype=np.float64)
        for s in series
        if s is not None and len(s) > 0
    ]
    if not nao_vazias:
        return np.zeros(0)

    todos = np.concatenate(nao_vazias)
    minimo = float(np.min(todos))
    maximo = float(np.max(todos))
    if minimo == maximo:
        minimo -= 0.5
        maximo += 0.5
    return np.linspace(minimo, maximo, int(n_arestas))
