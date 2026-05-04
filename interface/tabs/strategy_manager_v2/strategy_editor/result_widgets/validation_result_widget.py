# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/validation_result_widget.py

"""
Widget de visualização de StrategyValidationResult.

QTabWidget com 4 abas (Distribuição de LRs, Tippett, DET, ECE) + header
âmbar com métricas escalares (EER, AUC, Cllr, cal_loss). Cada aba
contém `FigureCanvas` + opcional `NavigationToolbar` + caixa de ajuda
contextual com explicação dos plots.

A lógica de plot foi extraída de `ValidationResultViewerDialog` sem
alteração — o dialog agora envolve este widget com um botão Fechar.

Suporta lazy rendering: as figures de cada aba só são desenhadas quando
a aba é exibida pela primeira vez. Ideal para uso inline na coluna de
Resultados, onde o usuário pode mudar de instrução antes de olhar todas
as abas. Cache interno por aba evita re-render desnecessário; uma nova
chamada a `set_result(...)` invalida o cache.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"


class ValidationResultWidget(QtWidgets.QWidget):
    """
    Painel de 4 abas para um StrategyValidationResult.

    Args:
        parent: pai opcional.
        result: StrategyValidationResult com `.metrics` e `.plot_data`.
            Pode ser None (placeholder).
        show_header: se True, exibe o header âmbar com métricas escalares.
            Default True; no painel inline geralmente fica True também
            porque as métricas são úteis sem sair pra outro lugar.
        show_toolbar: se True, cada aba ganha sua NavigationToolbar.
            Default True (modo dialog); inline geralmente passa False
            para reduzir clutter — o botão expandir abre o dialog que
            traz a toolbar.
        show_help: se True, exibe a caixa de ajuda azul abaixo de cada
            plot. Default True; inline geralmente passa False.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        result: Any = None,
        show_header: bool = True,
        show_toolbar: bool = True,
        show_help: bool = True,
    ):
        super().__init__(parent)
        self._result = result
        self._show_toolbar = bool(show_toolbar)
        self._show_help = bool(show_help)

        # Cache: {tab_index: (canvas, figure, draw_fn, drawn_for_result_id)}
        self._tab_state: Dict[int, Dict[str, Any]] = {}
        # Identificador opaco do resultado atual — quando muda, invalida
        # todas as figures cacheadas.
        self._result_token: int = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if show_header:
            self._header_frame = self._build_header_frame()
            layout.addWidget(self._header_frame)
        else:
            self._header_frame = None

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._setup_tabs()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Renderiza header + força primeiro draw da aba ativa.
        self._refresh_header()
        self._on_tab_changed(self.tabs.currentIndex())

    # ── Public API ────────────────────────────────────────────────────────────

    def set_result(self, result: Any) -> None:
        """Troca o resultado exibido — invalida cache e redesenha."""
        self._result = result
        self._result_token += 1
        # Limpa figures cacheadas para forçar redraw.
        for state in self._tab_state.values():
            state["drawn_token"] = None
        self._refresh_header()
        self._on_tab_changed(self.tabs.currentIndex())

    @property
    def result(self) -> Any:
        return self._result

    # ── UI building ───────────────────────────────────────────────────────────

    def _build_header_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        self._header_label = QtWidgets.QLabel("(aguardando resultado)")
        self._header_label.setStyleSheet(
            "background: transparent; color: #5a4500; font-size: 11px;"
        )
        self._header_label.setWordWrap(True)
        h.addWidget(self._header_label, 1)
        return frame

    def _refresh_header(self) -> None:
        if self._header_frame is None:
            return
        if self._result is None:
            self._header_label.setText("(aguardando resultado)")
            return

        m = self._result.metrics
        n_gen = self._result.n_genuines
        n_imp = self._result.n_impostors
        cal_loss = max(0.0, m.cllr - m.cllr_min)

        text = (
            f"<b>{self._result.strategy_name}</b>"
            f" &nbsp;·&nbsp; {n_gen:,} genuínos / {n_imp:,} impostores"
            f" &nbsp;·&nbsp; <b>EER</b>={m.eer*100:.2f}%"
            f" &nbsp;·&nbsp; <b>AUC</b>={m.auc:.4f}"
            f" &nbsp;·&nbsp; <b>Cllr</b>={m.cllr:.3f}"
            f" &nbsp;·&nbsp; <b>Cllr_min</b>={m.cllr_min:.3f}"
            f" &nbsp;·&nbsp; <b>cal_loss</b>={cal_loss:.3f}"
        ).replace(",", ".")
        self._header_label.setText(text)

    def _setup_tabs(self) -> None:
        """Cria as 4 abas vazias (figures populadas on-demand)."""
        self._add_lazy_tab("Distribuição de LRs",
                           self._draw_lr_distribution,
                           self._lr_distribution_help() if self._show_help else "")
        self._add_lazy_tab("Tippett",
                           self._draw_tippett,
                           self._tippett_help() if self._show_help else "")
        self._add_lazy_tab("Curva DET",
                           self._draw_det_curve,
                           self._det_help() if self._show_help else "")
        self._add_lazy_tab("Calibração (ECE)",
                           self._draw_ece,
                           self._ece_help() if self._show_help else "")

    def _add_lazy_tab(
        self, title: str, draw_fn: Callable[[Figure], None], help_html: str = ""
    ) -> None:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(4, 4, 4, 4)
        page_layout.setSpacing(4)

        figure = Figure(figsize=(8, 5), dpi=100)
        canvas = FigureCanvas(figure)

        if self._show_toolbar:
            toolbar = NavigationToolbar(canvas, page)
            page_layout.addWidget(toolbar)

        page_layout.addWidget(canvas, 1)

        if help_html:
            help_container = QtWidgets.QScrollArea()
            help_container.setWidgetResizable(True)
            help_container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            help_container.setMaximumHeight(180)
            help_container.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            help_inner = QtWidgets.QFrame()
            help_inner.setStyleSheet(
                "QFrame { background: #f4f6fa; border-left: 3px solid #2E5090;"
                " border-radius: 2px; }"
            )
            inner_layout = QtWidgets.QVBoxLayout(help_inner)
            inner_layout.setContentsMargins(10, 6, 10, 6)
            help_lbl = QtWidgets.QLabel(help_html)
            help_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            help_lbl.setWordWrap(True)
            help_lbl.setStyleSheet(
                "background: transparent; color: #1a2942; font-size: 11px;"
            )
            inner_layout.addWidget(help_lbl)
            help_container.setWidget(help_inner)
            page_layout.addWidget(help_container)

        idx = self.tabs.addTab(page, title)
        self._tab_state[idx] = {
            "canvas": canvas,
            "figure": figure,
            "draw_fn": draw_fn,
            "drawn_token": None,  # invalida render quando difere de _result_token
        }

    def _on_tab_changed(self, index: int) -> None:
        """Lazy: desenha a aba só na primeira ativação para o resultado atual."""
        state = self._tab_state.get(index)
        if state is None:
            return

        if state["drawn_token"] == self._result_token:
            return  # já está desenhado para este resultado

        figure: Figure = state["figure"]
        canvas: FigureCanvas = state["canvas"]
        draw_fn: Callable[[Figure], None] = state["draw_fn"]

        figure.clear()
        if self._result is None:
            ax = figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "(aguardando resultado)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#888",
            )
            ax.axis("off")
        else:
            try:
                draw_fn(figure)
            except Exception as exc:
                figure.clear()
                ax = figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    f"Erro ao gerar plot:\n{exc}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#a33", fontsize=10,
                )
                ax.axis("off")
        canvas.draw_idle()
        state["drawn_token"] = self._result_token

    # ── Plot functions (extraídas do dialog antigo) ───────────────────────────

    def _draw_lr_distribution(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        gen = np.asarray(pd.genuine_log_lrs, dtype=np.float64)
        imp = np.asarray(pd.impostor_log_lrs, dtype=np.float64)

        if gen.size == 0 and imp.size == 0:
            ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        all_lrs = np.concatenate([gen, imp])
        x_min, x_max = float(np.min(all_lrs)), float(np.max(all_lrs))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        bins = np.linspace(x_min, x_max, 60)

        if imp.size > 0:
            ax.hist(imp, bins=bins, density=True, alpha=0.45,
                    color="#c62828", edgecolor="#7a0000", linewidth=0.5,
                    label=f"Impostores  (n={imp.size:,})".replace(",", "."))
        if gen.size > 0:
            ax.hist(gen, bins=bins, density=True, alpha=0.45,
                    color="#2e7d32", edgecolor="#0d3a10", linewidth=0.5,
                    label=f"Genuínos  (n={gen.size:,})".replace(",", "."))

        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x_min, x_max, 400)
            if imp.size > 1:
                ax.plot(xs, gaussian_kde(imp)(xs), color="#7a0000",
                        linewidth=1.4, alpha=0.9)
            if gen.size > 1:
                ax.plot(xs, gaussian_kde(gen)(xs), color="#0d3a10",
                        linewidth=1.4, alpha=0.9)
        except Exception:
            pass

        ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="log-LR = 0")
        ax.set_title("Distribuição de log-LRs", fontsize=11)
        ax.set_xlabel("log-LR calibrado")
        ax.set_ylabel("Densidade")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)
        figure.tight_layout()

    def _draw_tippett(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        gen = np.sort(np.asarray(pd.genuine_log_lrs, dtype=np.float64))
        imp = np.sort(np.asarray(pd.impostor_log_lrs, dtype=np.float64))

        if gen.size == 0 or imp.size == 0:
            ax.text(0.5, 0.5, "(requer ambas as classes)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        gen_cum = np.arange(1, gen.size + 1) / gen.size
        imp_cum_above = 1.0 - np.arange(1, imp.size + 1) / imp.size

        ax.plot(gen, gen_cum, color="#2e7d32", linewidth=1.6,
                label="Genuínos: P(log-LR ≤ x | Hp)")
        ax.plot(imp, imp_cum_above, color="#c62828", linewidth=1.6,
                label="Impostores: P(log-LR > x | Hd)")

        ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                   alpha=0.6, label="log-LR = 0")

        ax.set_title("Tippett Plot", fontsize=11)
        ax.set_xlabel("log-LR (limiar de decisão)")
        ax.set_ylabel("Proporção cumulativa")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)
        figure.tight_layout()

    def _draw_det_curve(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        fpr = np.asarray(pd.fpr_array, dtype=np.float64)
        fnr = np.asarray(pd.fnr_array, dtype=np.float64)
        m = self._result.metrics

        if fpr.size == 0 or fnr.size == 0:
            ax.text(0.5, 0.5, "(sem dados de DET)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        eps = 1e-6
        fpr_c = np.clip(fpr, eps, 1.0 - eps)
        fnr_c = np.clip(fnr, eps, 1.0 - eps)

        try:
            from scipy.stats import norm
            x = norm.ppf(fpr_c)
            y = norm.ppf(fnr_c)
        except Exception:
            x, y = fpr_c, fnr_c

        ax.plot(x, y, color="#1a4f8a", linewidth=1.8, label="DET")

        try:
            from scipy.stats import norm
            eer_n = norm.ppf(np.clip(m.eer, eps, 1.0 - eps))
            ax.scatter([eer_n], [eer_n], color="#E8A020", s=80, zorder=5,
                       edgecolor="#7a4f00", linewidth=1.4,
                       label=f"EER = {m.eer*100:.2f}%")
            ax.axhline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
        except Exception:
            pass

        try:
            from scipy.stats import norm
            tick_pcts = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
            tick_locs = norm.ppf(tick_pcts)
            tick_labels = [f"{p*100:g}%" for p in tick_pcts]
            ax.set_xticks(tick_locs)
            ax.set_xticklabels(tick_labels)
            ax.set_yticks(tick_locs)
            ax.set_yticklabels(tick_labels)
        except Exception:
            pass

        ax.set_title("Curva DET (Detection Error Tradeoff)", fontsize=11)
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
        ax.set_ylabel("Taxa de Falsos Negativos (FNR)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
        figure.tight_layout()

    def _draw_ece(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        x = np.asarray(pd.ece_prior_log_odds, dtype=np.float64)
        y_real = np.asarray(pd.ece_values, dtype=np.float64)
        y_min = np.asarray(pd.ece_min_values, dtype=np.float64)

        if x.size == 0 or y_real.size == 0:
            ax.text(0.5, 0.5, "(sem dados de ECE)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        if y_min.size == y_real.size:
            ax.fill_between(x, y_min, y_real, where=(y_real >= y_min),
                            color="#E8A020", alpha=0.18,
                            label="Perda de calibração")

        ax.plot(x, y_real, color="#1a4f8a", linewidth=1.8,
                label="ECE (modelo)")
        if y_min.size:
            ax.plot(x, y_min, color="#2e7d32", linewidth=1.6, linestyle="--",
                    label="ECE mínimo (PAVA)")

        ax.axvline(0.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title("Empirical Cross-Entropy (ECE)", fontsize=11)
        ax.set_xlabel("Prior log₁₀-odds (P(Hp)/P(Hd))")
        ax.set_ylabel("Cross-entropy (bits)")
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper center", fontsize=9)
        figure.tight_layout()

    # ── Help text ─────────────────────────────────────────────────────────────

    @staticmethod
    def _lr_distribution_help() -> str:
        return (
            "<b>O que mostra:</b> distribuição empírica dos log-LR calibrados "
            "para os pares <span style='color:#2e7d32;'><b>genuínos</b></span> "
            "(mesma pessoa, Hp) e <span style='color:#c62828;'><b>impostores</b></span> "
            "(pessoas diferentes, Hd). A linha pontilhada em log-LR=0 marca "
            "o ponto de neutralidade (LR=1, evidência inconclusiva)."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• <b>Boa separação</b> entre as duas distribuições indica que o "
            "sistema discrimina bem genuínos de impostores.<br>"
            "• Genuínos concentrados em <b>log-LR &gt; 0</b> e impostores em "
            "<b>log-LR &lt; 0</b> indica calibração no sentido correto.<br>"
            "• Cauda dos genuínos &lt; 0 = pares de mesma pessoa para os quais "
            "o sistema gerou evidência <i>contra</i> a hipótese da acusação "
            "(potenciais falsos negativos no laudo).<br>"
            "• Cauda dos impostores &gt; 0 = pares de pessoas diferentes com "
            "evidência <i>a favor</i> da acusação (potenciais falsos positivos)."
        )

    @staticmethod
    def _tippett_help() -> str:
        return (
            "<b>O que mostra:</b> proporção cumulativa de log-LR por classe. A "
            "curva <span style='color:#2e7d32;'><b>verde</b></span> é "
            "P(log-LR &le; x | Hp) — fração de genuínos cujo LR ficou abaixo "
            "do limiar x. A curva <span style='color:#c62828;'><b>vermelha</b></span> "
            "é P(log-LR &gt; x | Hd) — fração de impostores cujo LR ficou "
            "acima do limiar x."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• Em log-LR=0, a curva verde dá a <b>taxa de falsos negativos</b> "
            "e a vermelha dá a <b>taxa de falsos positivos</b> caso o "
            "limiar de decisão fosse LR=1.<br>"
            "• Curvas que sobem rapidamente perto de log-LR=0 indicam "
            "<b>boa discriminação</b> — poucos pares ambíguos.<br>"
            "• <b>Cruzamento</b> próximo a log-LR=0 sugere calibração "
            "razoável (taxas de erro simétricas no ponto neutro). Cruzamento "
            "à esquerda de 0 = sistema otimista (mais falsos positivos); "
            "à direita = sistema conservador (mais falsos negativos).<br>"
            "• Distância vertical entre curvas em qualquer log-LR mostra a "
            "<b>magnitude do erro</b> naquele ponto de operação."
        )

    @staticmethod
    def _det_help() -> str:
        return (
            "<b>O que mostra:</b> Detection Error Tradeoff — taxa de falsos "
            "positivos (FPR) no eixo X versus taxa de falsos negativos (FNR) "
            "no eixo Y, ambos em <i>escala normit</i> (probit). O ponto "
            "<span style='color:#E8A020;'><b>âmbar</b></span> marca o EER "
            "(Equal Error Rate), onde FPR = FNR."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• Sistema aleatório seria uma <b>diagonal</b> (50% de erro em "
            "tudo). Quanto mais a curva se desloca para o <b>canto inferior "
            "esquerdo</b>, melhor o poder discriminativo.<br>"
            "• <b>EER baixo</b> = bom poder discriminativo bruto, "
            "independente de calibração. Valores forenses de referência: "
            "&lt; 1% (excelente), 1–5% (bom), 5–10% (aceitável), &gt; 10% "
            "(precisa investigar dados/calibração).<br>"
            "• Comparando estratégias, a que tiver curva mais afastada da "
            "diagonal é a melhor — pode-se escolher um ponto de operação "
            "com FPR baixo se for crítico evitar falsos positivos.<br>"
            "• A escala normit comprime as caudas, tornando visíveis "
            "diferenças em FPR/FNR &lt; 1% (impossíveis de ver em escala "
            "linear)."
        )

    @staticmethod
    def _ece_help() -> str:
        return (
            "<b>O que mostra:</b> Empirical Cross-Entropy — custo "
            "informacional do sistema (em <b>bits</b>) ao longo de "
            "diferentes cenários forenses, simulados variando a chance a "
            "priori de Hp/Hd no eixo X (log₁₀-odds de -3.5 a +3.5). "
            "<span style='color:#1a4f8a;'><b>Curva azul</b></span>: ECE do "
            "modelo atual. <span style='color:#2e7d32;'><b>Curva verde "
            "tracejada</b></span>: ECE mínimo possível após PAVA "
            "(calibração ótima). A área <span style='color:#E8A020;'><b>âmbar</b></span> "
            "entre as duas é a <b>perda de calibração</b> (cal_loss)."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• Eixo X: log₁₀-odds=0 é o cenário neutro (Hp e Hd igualmente "
            "prováveis). -3 = defesa 1000× mais provável a priori; +3 = "
            "acusação 1000× mais provável. Curvas baixas em todo o intervalo "
            "indicam que o sistema vale a pena em <b>todos</b> os cenários.<br>"
            "• <b>Curvas próximas</b> (área âmbar pequena) = boa calibração "
            "— os LR reportados são honestos no peso da evidência.<br>"
            "• <b>Curva azul muito acima da verde</b> = sistema discrimina "
            "bem mas dá certezas <i>exageradas</i> (LRs muito altos/baixos "
            "que não se sustentam) ou <i>conservadoras</i> demais. Recalibrar "
            "(refit do calibrador) reduz a área âmbar.<br>"
            "• Se a curva azul fica acima de 1.0 em alguma região, o sistema "
            "<b>perde informação</b> nesse cenário — relatar LR=1 (evidência "
            "neutra) seria melhor que usar o modelo lá."
        )
