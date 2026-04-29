# interface/tabs/strategy_manager_v2/strategy_editor/validation_result_viewer_dialog.py

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"


class ValidationResultViewerDialog(QtWidgets.QDialog):
    """
    Dialog com QTabWidget de 4 abas para visualizar um StrategyValidationResult:

    1. Distribuição de log-LRs   — histograma + KDE de genuínos vs impostores.
    2. Tippett Plot              — proporção cumulativa em função do log-LR.
    3. Curva DET (escala normit) — Detection Error Tradeoff com marca no EER.
    4. Calibração (ECE)          — cross-entropy real vs mínimo (PAVA).

    Header com métricas escalares (EER, AUC, Cllr, cal_loss). Toolbar de
    navegação matplotlib (zoom, pan, salvar PNG) por aba.
    """

    def __init__(
        self,
        result,  # StrategyValidationResult
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle(f"Resultado de Validação: {result.strategy_name}")
        self.resize(1100, 780)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ─── Header com métricas ──────────────────────────────────────
        header = self._build_header()
        layout.addWidget(header)

        # ─── Tabs com os 4 plots ──────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._add_plot_tab(
            "Distribuição de LRs",
            self._draw_lr_distribution,
            self._lr_distribution_help(),
        )
        self._add_plot_tab(
            "Tippett",
            self._draw_tippett,
            self._tippett_help(),
        )
        self._add_plot_tab(
            "Curva DET",
            self._draw_det_curve,
            self._det_help(),
        )
        self._add_plot_tab(
            "Calibração (ECE)",
            self._draw_ece,
            self._ece_help(),
        )

        # ─── Botão Fechar ─────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ─── UI helpers ────────────────────────────────────────────────────

    def _build_header(self) -> QtWidgets.QWidget:
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

        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; border-left: 3px solid {_BORDER_HEADER};"
            f" border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("background: transparent; color: #5a4500; font-size: 11px;")
        lbl.setWordWrap(True)
        h.addWidget(lbl, 1)
        return frame

    def _add_plot_tab(self, title: str, draw_fn, help_html: str = "") -> None:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(4, 4, 4, 4)
        page_layout.setSpacing(4)

        figure = Figure(figsize=(10, 7), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, page)

        page_layout.addWidget(toolbar)
        page_layout.addWidget(canvas, 1)

        try:
            draw_fn(figure)
        except Exception as exc:
            ax = figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Erro ao gerar plot:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                color="#a33", fontsize=10,
            )
        canvas.draw()

        # Caixa de explicação abaixo do plot — visível sempre. Texto rico
        # via HTML, com word-wrap e área scrollável caso ultrapasse o limite
        # de altura para não comer espaço do canvas.
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

        self.tabs.addTab(page, title)

    # ─── Plot functions ────────────────────────────────────────────────

    def _draw_lr_distribution(self, figure: Figure) -> None:
        """Histograma + KDE de genuínos (verde) vs impostores (vermelho) em log-LR."""
        figure.clear()
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
        """
        Tippett Plot: proporção cumulativa de log-LRs em cada classe.

        Curva genuínos (verde): P(LR ≤ x | Hp) — fração de genuínos com LR menor
        que x. Cresce da esquerda pra direita; idealmente fica baixa em x grande.
        Curva impostores (vermelho): P(LR > x | Hd) — fração de impostores com LR
        maior que x. Decresce da esquerda pra direita; idealmente fica baixa em
        x positivo.

        Cruzamento em log-LR≈0 indica calibração razoável.
        """
        figure.clear()
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        gen = np.sort(np.asarray(pd.genuine_log_lrs, dtype=np.float64))
        imp = np.sort(np.asarray(pd.impostor_log_lrs, dtype=np.float64))

        if gen.size == 0 or imp.size == 0:
            ax.text(0.5, 0.5, "(requer ambas as classes)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        # Hp: fração cumulativa de genuínos abaixo do limiar (proporção de
        # falsos negativos se decidirmos por limiar = x).
        gen_cum = np.arange(1, gen.size + 1) / gen.size
        # Hd: fração de impostores acima do limiar (proporção de falsos
        # positivos se decidirmos por limiar = x).
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
        """
        Curva DET em escala normit (probit): converte FPR/FNR via norm.ppf
        para que sistemas perto da random-guess apareçam como reta diagonal
        e bons sistemas curvem para o canto inferior esquerdo. Marca o EER.
        """
        figure.clear()
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        fpr = np.asarray(pd.fpr_array, dtype=np.float64)
        fnr = np.asarray(pd.fnr_array, dtype=np.float64)
        m = self._result.metrics

        if fpr.size == 0 or fnr.size == 0:
            ax.text(0.5, 0.5, "(sem dados de DET)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        # Clip para evitar normit infinito nas bordas.
        eps = 1e-6
        fpr_c = np.clip(fpr, eps, 1.0 - eps)
        fnr_c = np.clip(fnr, eps, 1.0 - eps)

        try:
            from scipy.stats import norm
            x = norm.ppf(fpr_c)
            y = norm.ppf(fnr_c)
        except Exception:
            # Fallback: log-scale simples se scipy não disponível.
            x, y = fpr_c, fnr_c

        ax.plot(x, y, color="#1a4f8a", linewidth=1.8, label="DET")

        # Marca do EER (FPR=FNR no ponto do EER).
        try:
            eer_n = norm.ppf(np.clip(m.eer, eps, 1.0 - eps))
            ax.scatter([eer_n], [eer_n], color="#E8A020", s=80, zorder=5,
                       edgecolor="#7a4f00", linewidth=1.4,
                       label=f"EER = {m.eer*100:.2f}%")
            ax.axhline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axvline(eer_n, color="#888", linestyle=":", linewidth=0.8, alpha=0.5)
        except Exception:
            pass

        # Eixos com ticks em valores convencionais (porcentagens) na escala normit.
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
        """
        Empirical Cross-Entropy (ECE) plot: avalia a qualidade da calibração
        em função das chances a priori (prior log10-odds de -3.5 a +3.5).

        - Curva azul (real): ECE do modelo atual.
        - Curva verde (mínima): ECE mínimo após PAVA (calibração ótima possível).
        - Área entre as duas: perda de calibração (cal_loss em diferentes priores).
        """
        figure.clear()
        ax = figure.add_subplot(111)

        pd = self._result.plot_data
        x = np.asarray(pd.ece_prior_log_odds, dtype=np.float64)
        y_real = np.asarray(pd.ece_values, dtype=np.float64)
        y_min = np.asarray(pd.ece_min_values, dtype=np.float64)

        if x.size == 0 or y_real.size == 0:
            ax.text(0.5, 0.5, "(sem dados de ECE)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        # Pinta a área de perda entre as duas curvas (cal_loss visualizado).
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

    # ─── Help text por aba ─────────────────────────────────────────────

    def _lr_distribution_help(self) -> str:
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

    def _tippett_help(self) -> str:
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

    def _det_help(self) -> str:
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

    def _ece_help(self) -> str:
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
