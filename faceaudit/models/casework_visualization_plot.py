"""Renderização da visualização forense de caso (CASEWORK_VISUALIZATION).

Funções puras de matplotlib — sem dependência de Qt. Espelham o padrão
do `distribution_plot.py` (snapshot PNG persistido + plot vivo no widget
delegam ao mesmo render, garantindo idéntico visual).

Usado em três contextos:
  - `CaseworkVisualizationWidget` (UI ao vivo, FigureCanvas Qt).
  - Snapshot PNG gravado em `<project>/graphics/` quando o executor
    persiste um `CaseworkVisualizationResult`.
  - Eventuais exports forenses futuros (laudo PDF, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np
from matplotlib.figure import Figure

from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.models.plot_number_format import (
    aplicar_formato_brasileiro,
    formatar_numero,
)


_KDE_MIN_CASE_SAMPLES = 30

# Paleta da figura. Amostrada do exemplo aprovado pelo usuário.
#
# Tons suaves e não saturados: a figura vai para laudo impresso, onde
# vermelho e verde fortes brigam entre si e cansam a leitura. O azul escuro
# do caso se destaca contra os dois.
_COR_MESMA_PESSOA = "#86c586"
_COR_PESSOAS_DIFERENTES = "#e08585"
# Azul escuro para o caso: destaca-se contra as barras e não se confunde
# com o azul claro de "mesma pessoa", mesmo impresso em preto e branco,
# porque a diferença de valor sobrevive à conversão.
_COR_CASO = "#1a3a8f"

# Curvas suavizadas em tom mais fechado da mesma cor da barra, para a
# correspondência ser imediata.
_COR_MESMA_PESSOA_LINHA = "#2e7d32"
_COR_PESSOAS_DIFERENTES_LINHA = "#a52a2a"

# Barras com folga entre si, como no exemplo: separadas, a leitura por
# faixa fica explícita e o histograma não vira bloco contínuo.
_LARGURA_RELATIVA_BARRA = 0.9

# Quanto o eixo Y se estende acima do dado, para a legenda não cobrir
# as anotações do caso.
_FOLGA_SUPERIOR = 1.32


def _faixas_para(n: int) -> int:
    """Número de faixas do histograma, a partir do tamanho DO GRUPO.

    Um valor fixo para todos os grupos trata mal o menor deles: 50 faixas
    servem para centenas de milhares de comparações e deixam ~9 observações
    por faixa quando o grupo tem poucas centenas, produzindo um serrilhado
    que é ruído amostral e que, numa figura de laudo, pode ser lido como
    estrutura do dado.

    Raiz de n é a regra simples e usual; o piso evita histograma de três
    barras e o teto mantém a figura legível.
    """
    import math

    if n <= 0:
        return 10
    return max(10, min(50, int(round(math.sqrt(n)))))


def draw_casework_visualization(
    figure: Figure,
    result: Any,
    *,
    show_title: bool = True,
    show_kde: bool = False,
    legend_below: bool = True,
) -> None:
    """Desenha distribuição forense (caso sobreposto à referência).

    Esta figura vai para laudo, lida por quem não é da área. Daí o
    vocabulário: "Mesma pessoa" e "Pessoas diferentes" em vez de "genuíno" e
    "impostor", que são termos do campo e exigiriam glossário, e nenhuma
    palavra em inglês nos rótulos. Ao alterar qualquer texto daqui, vale
    lembrar que o destinatário é o leitor do laudo, não o perito.

    `show_kde=False` (padrão) desenha só os histogramas. A curva suavizada é um
    MODELO da distribuição: a largura de banda não aparece na figura e
    influencia a aparência das caudas, que é onde costuma estar o argumento
    forense. Com poucos pares de mesma pessoa, a cauda suavizada é
    extrapolação sobre poucas observações. Sem a curva, a figura mostra o
    dado observado.

    `result` é um `CaseworkVisualizationResult` (mesmo schema usado pelo
    widget vivo). Lógica idêntica à do `CaseworkVisualizationWidget._draw`
    pre-refator — só extraída pra uma função pura aqui pra que o snapshot
    PNG persistido produza exatamente o mesmo gráfico.
    """
    from faceaudit.models.similarity.kde_curve import kde_curve

    figure.clear()
    figure.set_layout_engine("constrained")
    ax = figure.add_subplot(111)

    gen = np.asarray(result.genuine_scores, dtype=np.float64)
    imp = np.asarray(result.impostor_scores, dtype=np.float64)
    case = np.asarray(result.case_scores, dtype=np.float64)

    all_vals = []
    if gen.size > 0:
        all_vals.append(gen)
    if imp.size > 0:
        all_vals.append(imp)
    if case.size > 0:
        all_vals.append(case)
    if not all_vals:
        ax.text(
            0.5, 0.5, "Sem dados para plotar.",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    combined = np.concatenate(all_vals)
    x_min = float(np.min(combined))
    x_max = float(np.max(combined))
    # Folga proporcional, com piso pequeno. O piso anterior de 0,5 era
    # largo demais para escore de similaridade, que vive em [-1, 1]:
    # deixava um terço do eixo vazio de cada lado.
    pad = max(0.05 * (x_max - x_min), 0.05)
    x_min -= pad
    x_max += pad
    xs = np.linspace(x_min, x_max, 500)

    if imp.size > 0:
        ax.hist(
            imp, bins=_faixas_para(imp.size), density=True,
            color=_COR_PESSOAS_DIFERENTES, alpha=0.75,
            rwidth=_LARGURA_RELATIVA_BARRA,
            label=f"Pessoas diferentes (n={imp.size:,})".replace(",", "."),
        )
        kde_imp = kde_curve(imp, xs) if show_kde else None
        if kde_imp is not None:
            ax.plot(xs, kde_imp, color=_COR_PESSOAS_DIFERENTES_LINHA,
                    linewidth=1.4, alpha=0.9)

    if gen.size > 0:
        ax.hist(
            gen, bins=_faixas_para(gen.size), density=True,
            color=_COR_MESMA_PESSOA, alpha=0.75,
            rwidth=_LARGURA_RELATIVA_BARRA,
            label=f"Mesma pessoa (n={gen.size:,})".replace(",", "."),
        )
        kde_gen = kde_curve(gen, xs) if show_kde else None
        if kde_gen is not None:
            ax.plot(xs, kde_gen, color=_COR_MESMA_PESSOA_LINHA,
                    linewidth=1.4, alpha=0.9)

    if case.size >= _KDE_MIN_CASE_SAMPLES:
        ax.hist(
            case, bins=_faixas_para(case.size), density=True,
            color=_COR_CASO, alpha=0.55,
            rwidth=_LARGURA_RELATIVA_BARRA,
            label=(
                f"Caso (n={case.size:,}, KDE)" if show_kde
                else f"Caso (n={case.size:,})"
            ).replace(",", "."),
        )
        kde_case = kde_curve(case, xs) if show_kde else None
        if kde_case is not None:
            ax.plot(xs, kde_case, color=_COR_CASO, linewidth=1.6, alpha=0.95)
    else:
        # Altura das BARRAS, não do eixo: a folga do topo é da legenda.
        y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        for i, val in enumerate(case):
            # Concordância de número: "1 marca", "2 marcas". Numa figura
            # de laudo o deslize salta aos olhos.
            label = (
                f"Caso (n={case.size}, "
                + ("marca" if case.size == 1 else "marcas")
                + ")"
            ) if i == 0 else None
            ax.axvline(
                val, color=_COR_CASO, linestyle="--", linewidth=1.6,
                alpha=0.9, label=label,
            )
            ax.annotate(
                formatar_numero(val, 2),
                xy=(val, y_top * 0.95),
                xytext=(0, -5),
                textcoords="offset points",
                rotation=90,
                color=_COR_CASO,
                fontsize=8,
                ha="center", va="top",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white", ec=_COR_CASO, lw=0.8, alpha=0.85,
                ),
            )

    if result.score_type == "CALIBRATED":
        ax.axvline(
            0.0, color="#444", linestyle=":", linewidth=1.0,
            alpha=0.7, label="log-LR = 0 (limiar neutro)",
        )
    elif result.score_type == "RAW" and result.eer_threshold is not None:
        ax.axvline(
            result.eer_threshold, color="#444", linestyle=":", linewidth=1.0,
            alpha=0.7,
            # "EER" é sigla inglesa (Equal Error Rate). O leitor do laudo
            # recebe a explicação em português; a sigla fica entre
            # parênteses para quem precisa ligar à bibliografia.
            label=(
                "Limiar de igual taxa de erro (EER) = "
                + formatar_numero(result.eer_threshold, 4)
            ),
        )

    # "Escore de similaridade" e não "raw_score": o rótulo vai para figura
    # de laudo, lida por quem não conhece o nome interno da coluna. O eixo
    # muda de significado com o tipo de escore, daí a alternativa.
    x_label = (
        "log-LR calibrado"
        if result.score_type == "CALIBRATED" else "Escore de similaridade"
    )
    ax.set_xlabel(x_label)
    # "Densidade de frequência" e não "Densidade": as três curvas usam
    # `density=True`, ou seja, frequência relativa dividida pela amplitude
    # da classe, que é a definição do termo. Nomear a grandeza evita que o
    # leitor tente ler a altura como contagem.
    ax.set_ylabel("Densidade de frequência")
    ax.set_xlim(x_min, x_max)
    # Depois do set_xlim: as marcas são escolhidas a partir dos limites.
    aplicar_formato_brasileiro(ax)
    if show_title:
        ax.set_title("Contextualização do(s) escore(s) do caso")
    ax.grid(True, linestyle="--", alpha=0.4)

    if legend_below:
        # Fora da área do gráfico: não cobre nada, e o desenho fica com a
        # largura toda. Em duas colunas porque a entrada do limiar é longa e
        # numa linha só espremeria as demais.
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.13),
            ncol=2, fontsize=9, frameon=False,
        )
    else:
        # Faixa livre no topo. Sem ela, a legenda à direita cobre os valores
        # anotados sobre as linhas do caso, que ficam justamente no alto — e
        # o valor do caso é o número mais importante da figura.
        topo = ax.get_ylim()[1]
        ax.set_ylim(0.0, topo * _FOLGA_SUPERIOR)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)


def render_casework_png(
    result: Any,
    target_path: Union[str, Path],
    *,
    figsize: tuple = (8, 5),
    dpi: int = 100,
) -> None:
    """Renderiza a visualização forense como PNG headless (Agg backend).

    Lazy import do Agg pra não impactar quem só usa
    `draw_casework_visualization` via FigureCanvas Qt. Cria diretório
    destino se ausente.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=figsize, dpi=dpi)
    FigureCanvasAgg(fig)
    draw_casework_visualization(fig, result, show_title=True)
    fig.savefig(target, format="png")


def casework_visualization_png_path(
    project_root: Union[str, Path],
    strategy_id: str,
    resource_id: str,
    *,
    strategy_name: str = "",
) -> Path:
    """Path absoluto do PNG snapshot pra um output de CASEWORK_VISUALIZATION.

    Convenção:
    `<project>/graphics/<strategy_folder>/casework_visualization/<resource_id>_casework.png`.
    `strategy_folder` = `<sanitize(strategy_name)>_<strategy_id>`.
    """
    from faceaudit.models.strategy.strategy_folder import (
        sanitize_name, strategy_folder_name,
    )
    return (
        Path(project_root)
        / ProjectFolder.GRAPHICS.value
        / strategy_folder_name(strategy_id, strategy_name)
        / "casework_visualization"
        / f"{sanitize_name(resource_id)}_casework.png"
    )
