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
from PyQt6 import QtCore, QtGui, QtWidgets

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

        self.tabs = QtWidgets.QTabWidget(parent=self)
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
        frame = QtWidgets.QFrame(parent=self)
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(10, 6, 10, 6)
        v.setSpacing(2)
        self._header_label = QtWidgets.QLabel("(aguardando resultado)", parent=frame)
        self._header_label.setStyleSheet(
            "background: transparent; color: #5a4500; font-size: 11px;"
        )
        self._header_label.setWordWrap(True)
        self._header_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        v.addWidget(self._header_label)
        self._header_label2 = QtWidgets.QLabel("", parent=frame)
        self._header_label2.setStyleSheet(
            "background: transparent; color: #6b5800; font-size: 10.5px;"
        )
        self._header_label2.setWordWrap(True)
        self._header_label2.setTextFormat(QtCore.Qt.TextFormat.RichText)
        v.addWidget(self._header_label2)
        return frame

    def _refresh_header(self) -> None:
        if self._header_frame is None:
            return
        if self._result is None:
            self._header_label.setText("(aguardando resultado)")
            self._header_label2.setText("")
            return

        m = self._result.metrics
        n_gen = self._result.n_genuines
        n_imp = self._result.n_impostors
        cal_loss = max(0.0, m.cllr - m.cllr_min)

        # Linha 1: identidade + métricas principais Tier 1
        text1 = (
            f"<b>{self._result.strategy_name}</b>"
            f" &nbsp;·&nbsp; {n_gen:,} genuínos / {n_imp:,} impostores"
            f" &nbsp;·&nbsp; <b>EER</b>={m.eer*100:.2f}%"
            f" &nbsp;·&nbsp; <b>AUC</b>={m.auc:.4f}"
            f" &nbsp;·&nbsp; <b>Cllr</b>={m.cllr:.4f}"
            f" &nbsp;·&nbsp; <b>Cllr_min</b>={m.cllr_min:.4f}"
            f" &nbsp;·&nbsp; <b>cal_loss</b>={cal_loss:.4f}"
        ).replace(",", ".")
        self._header_label.setText(text1)

        # Linha 2: métricas forenses + diagnóstico de calibração
        parts2 = []
        if m.wrong_sign_imp_rate is not None:
            parts2.append(f"<b>Falso+</b>={m.wrong_sign_imp_rate*100:.2f}%")
        if m.wrong_sign_gen_rate is not None:
            parts2.append(f"<b>Falso−</b>={m.wrong_sign_gen_rate*100:.2f}%")
        if m.calibration_slope is not None:
            slope_emoji = "≈" if 0.95 <= m.calibration_slope <= 1.05 else (
                "↑" if m.calibration_slope > 1 else "↓"
            )
            parts2.append(f"<b>slope</b>={m.calibration_slope:.3f} {slope_emoji}")
        if m.cllr_affine is not None:
            parts2.append(f"<b>Cllr_affine</b>={m.cllr_affine:.4f}")
        text2 = " &nbsp;·&nbsp; ".join(parts2)
        self._header_label2.setText(text2)

    def _setup_tabs(self) -> None:
        """Cria as 5 abas de plot + 1 aba de Métricas (tabela)."""
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
        self._add_lazy_tab("Curva de Calibração",
                           self._draw_calibration_curve,
                           self._calibration_curve_help() if self._show_help else "")
        self._add_lazy_tab("Diagnóstico Afim",
                           self._draw_affine_diagnostic,
                           self._affine_diagnostic_help() if self._show_help else "")
        self._add_lazy_tab("Quantis por Classe",
                           self._draw_quantile_log_lr,
                           self._quantile_log_lr_help() if self._show_help else "")
        self._add_metrics_tab()

    def _add_metrics_tab(self) -> None:
        """Cria a aba 'Métricas' com tabela completa agrupada por tier.

        QTableWidget com 2 colunas (Métrica | Valor). Rows de section
        header em fundo cinza, fonte negrito. Demais rows alternadas
        em branco/cinza-claro. Copiável (Ctrl+C) em qualquer seleção.
        """
        page = QtWidgets.QWidget(parent=self)
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.setSpacing(4)

        table = QtWidgets.QTableWidget(parent=page)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Métrica", "Valor", "Descrição"])
        # Col 0/1 Interactive (user-resizable); Col 2 Stretch (auto-fill).
        # Scrollbar horizontal aparece se viewport for muito estreito.
        hh = table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Interactive)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hh.setMinimumSectionSize(60)
        hh.setStretchLastSection(False)
        table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        table.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        # Word wrap OFF + ellipsis + altura fixa: rows compactas, descrição
        # completa fica no tooltip on-hover.
        table.setWordWrap(False)
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        table.verticalHeader().setMinimumSectionSize(18)
        table.verticalHeader().setDefaultSectionSize(20)
        table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        # Min modesto pra não forçar QTabWidget alto demais quando outras
        # abas são gráficos curtos. Vertical scrollbar do QTableWidget
        # cobre o overflow. No dialog 🔍 Expandir (1100x780), há espaço
        # suficiente pra tabela respirar (~700px disponíveis).
        # AdjustIgnored: tabela aceita altura do parent sem propagar
        # sizeHint do conteúdo (que seria ~700px) — sem isso QTabWidget
        # tenta crescer pro tamanho total e bagunça o layout das abas PNG.
        table.setMinimumHeight(300)
        table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored
        )
        table.setStyleSheet(
            "QTableWidget { font-size: 11px; background: white; gridline-color: #e0e0e0; }"
            "QTableWidget::item { padding: 2px 6px; }"
            "QHeaderView::section { background: #f0f4f8; padding: 4px 6px; "
            "border: 0; border-bottom: 1px solid #c0c8d0; font-weight: bold; }"
        )
        page_layout.addWidget(table, 1)

        # Botão "Copiar tudo" pra exportar tabela completa
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_copy = QtWidgets.QPushButton("📋 Copiar tabela completa", parent=page)
        btn_copy.setStyleSheet(
            "QPushButton { border: 1px solid #ccc; border-radius: 3px; "
            "background: white; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #e8f0fe; }"
        )
        btn_copy.clicked.connect(lambda: self._copy_metrics_table_to_clipboard(table))
        btn_row.addWidget(btn_copy)
        page_layout.addLayout(btn_row)

        idx = self.tabs.addTab(page, "Métricas")
        # Reusa a mesma estrutura de cache do `_add_lazy_tab` pra que
        # `_on_tab_changed` chame `_refresh_metrics_table` quando a aba
        # ficar ativa.
        self._tab_state[idx] = {
            "canvas": None,
            "figure": None,
            "draw_fn": lambda _fig: self._refresh_metrics_table(table),
            "drawn_token": None,
            "is_metrics_table": True,
            "table": table,
        }

    def _copy_metrics_table_to_clipboard(self, table: QtWidgets.QTableWidget) -> None:
        """Copia a tabela inteira como TSV (colável em Excel/Sheets/etc.)."""
        lines = []
        for row in range(table.rowCount()):
            cells = []
            for col in range(table.columnCount()):
                item = table.item(row, col)
                cells.append(item.text() if item is not None else "")
            lines.append("\t".join(cells))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def _refresh_metrics_table(self, table: QtWidgets.QTableWidget) -> None:
        """Repopula a tabela de métricas. Chamada via `_on_tab_changed`."""
        table.setRowCount(0)

        if self._result is None:
            table.setRowCount(1)
            placeholder = QtWidgets.QTableWidgetItem("(aguardando resultado)")
            placeholder.setForeground(QtGui.QBrush(QtGui.QColor("#888")))
            table.setItem(0, 0, placeholder)
            return

        m = self._result.metrics
        rows = self._build_metrics_rows(m)
        table.setRowCount(len(rows))
        section_bg = QtGui.QBrush(QtGui.QColor("#e9eef5"))
        section_font = QtGui.QFont()
        section_font.setBold(True)
        for r, (label, value, desc, is_section) in enumerate(rows):
            label_item = QtWidgets.QTableWidgetItem(label)
            value_item = QtWidgets.QTableWidgetItem(value)
            desc_item = QtWidgets.QTableWidgetItem(desc)
            desc_item.setForeground(QtGui.QBrush(QtGui.QColor("#555")))
            if desc:
                desc_item.setToolTip(desc)
                label_item.setToolTip(f"{label}: {desc}")
            if is_section:
                label_item.setBackground(section_bg)
                value_item.setBackground(section_bg)
                desc_item.setBackground(section_bg)
                label_item.setFont(section_font)
                label_item.setForeground(QtGui.QBrush(QtGui.QColor("#1a3550")))
            table.setItem(r, 0, label_item)
            table.setItem(r, 1, value_item)
            table.setItem(r, 2, desc_item)
        # NÃO chama resizeRowsToContents — defaultSectionSize fixa altura.
        # Cols 0/1 sized inicialmente to contents; user pode arrastar depois.
        table.resizeColumnToContents(0)
        table.resizeColumnToContents(1)

    def _build_metrics_rows(self, m) -> List[Tuple[str, str, bool]]:
        """Constrói a sequência de rows da tabela. Cada entry é
        (label, value, is_section_header).

        Delega ao helper compartilhado `build_metrics_table_rows` (em
        `validation_result_plot.py`) — mesma fonte de verdade do painel
        inline. Converte `ValidationMetrics` → dict-like via `asdict()` +
        merge de contagens.
        """
        from dataclasses import asdict
        from faceaudit.models.validation_result_plot import (
            build_metrics_table_rows,
        )
        payload = asdict(m)
        # cal_loss não é field — é property; injeta manualmente
        payload["cal_loss"] = m.cal_loss
        payload["n_genuines"] = self._result.n_genuines
        payload["n_impostors"] = self._result.n_impostors
        return build_metrics_table_rows(payload)

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
            # Help inline ao invés de QScrollArea: o QLabel com wordWrap
            # tem altura natural que cabe todo o texto na largura disponível,
            # sem barra de rolagem. Stretch=0 (canvas tem stretch=1), então
            # ao maximizar o dialog é o canvas que cresce. Em janelas muito
            # pequenas o canvas comprime; o help mantém o texto inteiro
            # visível — comportamento desejado segundo feedback do usuário.
            help_inner = QtWidgets.QFrame(parent=self)
            help_inner.setStyleSheet(
                "QFrame { background: #f4f6fa; border-left: 3px solid #2E5090;"
                " border-radius: 2px; }"
            )
            inner_layout = QtWidgets.QVBoxLayout(help_inner)
            inner_layout.setContentsMargins(10, 6, 10, 6)
            help_lbl = QtWidgets.QLabel(help_html, parent=help_inner)
            help_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            help_lbl.setWordWrap(True)
            help_lbl.setStyleSheet(
                "background: transparent; color: #1a2942; font-size: 11px;"
            )
            inner_layout.addWidget(help_lbl)
            page_layout.addWidget(help_inner)

        idx = self.tabs.addTab(page, title)
        self._tab_state[idx] = {
            "canvas": canvas,
            "figure": figure,
            "draw_fn": draw_fn,
            "drawn_token": None,  # invalida render quando difere de _result_token
        }

    def _on_tab_changed(self, index: int) -> None:
        """Lazy: desenha a aba só na primeira ativação para o resultado atual.

        Cursor de espera durante a render — abas como Tippett e DET fazem
        sort/cumsum em milhões de pontos e podem travar a UI por segundos
        sem feedback visual nenhum.

        A aba "Métricas" não tem figure — só uma QTableWidget; o draw_fn
        é um wrapper sobre `_refresh_metrics_table` e o caminho de
        rendering pula a parte de matplotlib.
        """
        state = self._tab_state.get(index)
        if state is None:
            return

        if state["drawn_token"] == self._result_token:
            return  # já está desenhado para este resultado

        # Wait cursor só quando há resultado pra renderizar
        cursor_pushed = False
        if self._result is not None:
            QtWidgets.QApplication.setOverrideCursor(
                QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
            )
            QtWidgets.QApplication.processEvents()
            cursor_pushed = True

        try:
            if state.get("is_metrics_table"):
                # Aba "Métricas": só popula a tabela. Sem figure/canvas.
                draw_fn = state["draw_fn"]
                try:
                    draw_fn(None)
                except Exception:
                    pass  # tabela vazia já é placeholder OK
                state["drawn_token"] = self._result_token
                return

            figure: Figure = state["figure"]
            canvas: FigureCanvas = state["canvas"]
            draw_fn = state["draw_fn"]

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
        finally:
            if cursor_pushed:
                QtWidgets.QApplication.restoreOverrideCursor()

    # ── Plot functions: delegam pros helpers puros em
    # `faceaudit.models.validation_result_plot`. Wrappers mantidos por
    # compat (o `_setup_tabs` registra esses callbacks). A lógica de
    # desenho foi extraída pra que o snapshot PNG gerado durante a
    # persistência use exatamente o mesmo render. ────────────────────────

    def _draw_lr_distribution(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_lr_distribution
        draw_lr_distribution(figure, self._result)

    def _draw_tippett(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_tippett
        draw_tippett(figure, self._result)

    def _draw_det_curve(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_det_curve
        draw_det_curve(figure, self._result)

    def _draw_ece(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_ece
        draw_ece(figure, self._result)

    def _draw_calibration_curve(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_calibration_curve
        draw_calibration_curve(figure, self._result)

    def _draw_affine_diagnostic(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_affine_diagnostic
        draw_affine_diagnostic(figure, self._result)

    def _draw_quantile_log_lr(self, figure: Figure) -> None:
        from faceaudit.models.validation_result_plot import draw_quantile_log_lr
        draw_quantile_log_lr(figure, self._result)

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
            "(calibração ótima). <span style='color:#888;'><b>Curva cinza "
            "pontilhada</b></span>: ECE do <b>sistema neutro</b> — referência "
            "de um sistema que sempre reporta LR=1. A área "
            "<span style='color:#E8A020;'><b>âmbar</b></span> entre azul e "
            "verde é a <b>perda de calibração</b> (cal_loss)."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• Eixo X: log₁₀-odds=0 é o cenário neutro (Hp e Hd igualmente "
            "prováveis). -3 = defesa 1000× mais provável a priori; +3 = "
            "acusação 1000× mais provável.<br>"
            "• <b>Referência cinza tem pico em 1 bit</b> (em prior log-odds = 0) "
            "— é a entropia do prior. Só dá pra dizer que o modelo é "
            "<b>informativo</b> quando ECE_modelo (azul) fica abaixo dela. "
            "Se a azul ultrapassa a cinza em algum prior, o sistema é pior "
            "que <i>não decidir</i> ali — reportar LR=1 seria melhor.<br>"
            "• <b>Curvas próximas</b> (área âmbar pequena) = boa calibração "
            "— os LR reportados são honestos no peso da evidência.<br>"
            "• <b>Curva azul muito acima da verde</b> = sistema discrimina "
            "bem mas dá certezas <i>exageradas</i> (LRs muito altos/baixos "
            "que não se sustentam) ou <i>conservadoras</i> demais. Recalibrar "
            "(refit do calibrador) reduz a área âmbar."
        )

    @staticmethod
    def _calibration_curve_help() -> str:
        return (
            "<b>O que mostra:</b> a função de calibração no plano "
            "<i>raw_score</i> (eixo X) × <i>log-LR calibrado</i> (eixo Y). "
            "O painel superior tem três camadas: scatter dos pares observados "
            "(verde = genuínos, vermelho = impostores), <span "
            "style='color:#1a4f8a;'><b>curva azul sólida</b></span> = "
            "calibrador atual <i>f(x)</i> interpolado, e "
            "<span style='color:#E8A020;'><b>curva laranja tracejada</b></span> "
            "= recalibração afim Brümmer-Doddington: "
            "<i>slope·logLR + intercept</i>. Rugs na borda inferior mostram "
            "onde há massa de raw_scores por classe. O painel inferior "
            "mostra Δ log-LR = (pós-afim − atual) — quanto a afim empurraria "
            "cada região da curva."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• <b>Forma da curva azul</b>: linear pura = LogReg "
            "(decision_function); monotônica com inflexão = BiGauss; com "
            "degraus = KDE com bandwidth ruim.<br>"
            "• <b>Curva laranja vs azul</b>: se coincidem, slope≈1 e "
            "intercept≈0 — calibrador já está afinemente ótimo. Se a "
            "laranja é mais inclinada, a afim <i>expandiria</i> os "
            "log-LRs (slope>1, sistema atualmente under-confident); se "
            "menos inclinada, <i>encolheria</i> (slope&lt;1, "
            "over-confident). Deslocamento vertical entre elas = "
            "intercept (viés sistemático).<br>"
            "• <b>Importante</b>: o slope/intercept operam sobre o "
            "<i>log-LR</i> atual, não sobre o raw_score. Por isso a "
            "curva laranja só é uma reta no plano raw×logLR quando f(x) "
            "já é linear (caso LogReg). Pra KDE/BiGauss, é uma curva "
            "transformada.<br>"
            "• <b>Linha vertical tracejada</b>: raw_score onde a curva "
            "pós-afim cruza log-LR=0 — limiar onde a evidência após "
            "recalibração afim seria neutra (LR=1). Marca onde a "
            "fronteira de decisão estaria se aplicasse a afim ótima.<br>"
            "• <b>Painel inferior (Δ log-LR)</b>: zero em toda faixa → "
            "calibrador já é afinemente ótimo (Cllr ≈ Cllr_affine). "
            "Crescimento/decrescimento monotônico → problema principal "
            "é de slope. Constante deslocado de zero → problema "
            "principal é de intercept. <b>Curvatura</b> ou mudanças de "
            "sinal → há não-linearidade residual irrecuperável pela "
            "afim (responsável pelo <i>nonlinear_cal_loss</i> = "
            "Cllr_affine − Cllr_min, visível só no PAVA da aba ECE).<br>"
            "• <b>Rugs na borda</b>: contexto de onde estão os dados. "
            "Uma região com curva muito torta mas sem massa de scores "
            "ali é menos relevante que torção numa região densa.<br>"
            "• <b>⚠️ Escala intra-estratégia</b>: o slope absoluto "
            "depende da escala do raw_score subjacente (Cosine, PLDA, "
            "PLDA+pipeline têm escalas diferentes). Útil como "
            "diagnóstico <i>dentro</i> de uma estratégia; comparações "
            "diretas de slope <i>entre</i> estratégias com escalas de "
            "raw_score diferentes exigem cautela.<br>"
            "• <b>Para mais de 20 mil pares por classe</b>, scatter usa "
            "amostragem aleatória de 20k — a forma agregada é preservada."
        )

    @staticmethod
    def _affine_diagnostic_help() -> str:
        return (
            "<b>O que mostra:</b> a decomposição completa de Cllr "
            "reescalada no plano log-LR × log-LR. Eixo X = log-LR "
            "atual produzido pelo calibrador; eixo Y = log-LR após "
            "diferentes graus de correção:"
            "<br><br>"
            "• <span style='color:#888;'><b>Linha cinza tracejada</b></span> "
            "= identidade y=x. Referência de <i>sem recalibração</i> — o "
            "Cllr atual da estratégia.<br>"
            "• <span style='color:#E8A020;'><b>Reta laranja</b></span> = "
            "pós-afim ótima y=slope·logLR + intercept (Brümmer-Doddington). "
            "Atinge <i>Cllr_affine</i>: o melhor possível com correção "
            "linear.<br>"
            "• <span style='color:#1f7a8c;'><b>Curva verde-azulada (step)</b></span> "
            "= mapeamento <b>PAVA monotônico</b> (regressão isotônica). "
            "Atinge <i>Cllr_min</i>: limite teórico de calibração "
            "monotônica.<br>"
            "• Rugs na borda inferior indicam onde estão os log-LRs "
            "observados por classe (gen=verde, imp=vermelho)."
            "<br><br>"
            "<b>Decomposição numérica</b> (no subtítulo):<br>"
            "<i>Cllr ≥ Cllr_affine ≥ Cllr_min</i><br>"
            "• Gap <i>y=x → afim</i> (vertical) = "
            "<b>Cllr − Cllr_affine</b> = <i>afim_cal_loss</i>, "
            "recuperável por slope+intercept.<br>"
            "• Gap <i>afim → PAVA</i> (vertical) = "
            "<b>Cllr_affine − Cllr_min</b> = <i>nonlinear_loss</i>, "
            "<b>NÃO</b> recuperável por afim — só por correção "
            "não-linear monotônica."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• <b>Reta laranja sobre y=x</b> → slope≈1 e intercept≈0; "
            "calibrador já é afinemente ótimo.<br>"
            "• <b>Slope &lt; 1</b> (reta mais flat que y=x) → "
            "<i>over-confident</i>; log-LRs estão extremos demais, afim "
            "reduz a magnitude (comprime em direção a 0).<br>"
            "• <b>Slope &gt; 1</b> (reta mais íngreme que y=x) → "
            "<i>under-confident</i>; log-LRs estão conservadores, afim "
            "aumenta a magnitude (expande pra longe de 0).<br>"
            "• <b>Intercept ≠ 0</b> (reta deslocada verticalmente) → "
            "viés sistemático: o sistema favorece uma das hipóteses por "
            "default.<br>"
            "• <b>PAVA colada na afim</b> → calibrador é essencialmente "
            "linear; recalibração afim já captura quase tudo.<br>"
            "• <b>PAVA se afasta da afim nas caudas</b> → não-linearidade "
            "irrecuperável pela afim (típico de over-confidence nas "
            "caudas em KDE/BiGauss).<br>"
            "• <b>✅ Comparação inter-estratégia</b>: ao contrário da "
            "Curva de Calibração (cuja escala X depende do raw_score), "
            "este plot tem eixo X em log-LR (unidade comum). Pode "
            "sobrepor várias estratégias na janela de Comparação. "
            "(PAVA fica oculto na Comparação pra reduzir poluição "
            "visual — só afim é sobreposto.)"
        )

    @staticmethod
    def _quantile_log_lr_help() -> str:
        return (
            "<b>O que mostra:</b> os percentis P1, P5, P10, P25, P50, P75, "
            "P90, P95 e P99 do log-LR calibrado, separados por classe. "
            "<span style='color:#2e7d32;'><b>Série verde</b></span> = "
            "percentis dos genuínos. "
            "<span style='color:#c62828;'><b>Série vermelha</b></span> = "
            "percentis dos impostores. Cada ponto é o valor de log-LR no "
            "respectivo percentil <i>dentro da classe</i> (não do "
            "conjunto combinado). Os pontos com <b>halo destacado</b> "
            "(P1/P5 dos genuínos e P95/P99 dos impostores) marcam os "
            "extremos críticos da região de overlap forense."
            "<br><br>"
            "<b>Como interpretar:</b><br>"
            "• <b>P95/P99 dos impostores (vermelho à direita)</b> = "
            "impostores mais perigosos — pares de pessoas diferentes "
            "para os quais o sistema gerou os maiores log-LRs. Valores "
            "próximos de 0 ou positivos = potenciais falsos+ no laudo.<br>"
            "• <b>P1/P5 dos genuínos (verde à esquerda)</b> = genuínos "
            "mais difíceis — pares de mesma pessoa com menor log-LR. "
            "Valores próximos de 0 ou negativos = potenciais falsos− "
            "no laudo.<br>"
            "• <b>Comparação assimétrica nas caudas</b>: para avaliar a "
            "região de overlap, o mais informativo é comparar P5 dos "
            "genuínos contra P95 dos impostores (e P1 contra P99) — não "
            "o mesmo percentil em ambas as classes. Os <b>tail gaps</b> "
            "exibidos no subtítulo são:<br>"
            "&nbsp;&nbsp;<i>hard-tail (P5g − P95i)</i> &gt; 0 → boa "
            "separação na cauda; ≈ 0 = boundary case; &lt; 0 = parte dos "
            "genuínos abaixo de parte dos impostores nessa cauda "
            "(inversão grave).<br>"
            "&nbsp;&nbsp;<i>extreme (P1g − P99i)</i> sondam os 1% mais "
            "extremos de cada classe; tipicamente é o gap mais "
            "apertado.<br>"
            "• <b>Intersecção das séries</b> em qualquer percentil é "
            "sinal grave: parte dos genuínos tem log-LR menor que parte "
            "dos impostores no mesmo P — quebra da ordenação esperada.<br>"
            "• <b>✅ Comparação inter-estratégia</b>: percentil é "
            "invariante a transformações monotônicas do raw_score, e "
            "log-LR é unidade comum. Pode sobrepor várias estratégias "
            "na janela de Comparação e ler diretamente qual delas "
            "lida melhor com as caudas difíceis."
        )
