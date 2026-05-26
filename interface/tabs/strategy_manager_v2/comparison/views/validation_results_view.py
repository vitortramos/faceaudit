# interface/tabs/strategy_manager_v2/comparison/views/validation_results_view.py

"""
View do tipo VALIDATION_RESULTS.

Estrutura: QVBoxLayout pai com 4 linhas
    [0] Tabela de métricas (sizePolicy Maximum vertical)
    [1] FlowLayout com 3 cells overlay (DET, Tippett, ECE)
    [2] FlowLayout com N cells de distribuição log-LR
    [3] FlowLayout com N cells de Curva de Calibração

Cada cell de plot tem tamanho fixo CELL_WIDTH×CELL_HEIGHT — todos visualmente
iguais. Os FlowLayouts cuidam do reflow automático em mudanças de largura.

Plots usam `figure.set_layout_engine("constrained")` — o constrained layout
do matplotlib ajusta automaticamente xlabel, ticks, legendas externas
(incluindo bbox_to_anchor) sem sobreposição e sem deixar espaço branco.
Substitui o frágil `tight_layout(rect=[...])`.

A distribuição de log-LR e a curva de calibração ficam em seções separadas
(uma cell por estratégia) porque cada estratégia tem sua própria escala de
raw_score (Cosine vs PLDA produzem ranges diferentes) — overlay seria
ininteligível. Ambas reaproveitam os PNGs snapshot gerados pelo
STRATEGY_EVALUATION via `validation_result_png_paths`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.validation_result_plot import validation_result_png_paths

from ...strategy_editor.result_widgets.png_snapshot_widget import (
    PngSnapshotWidget,
)
from ..availability_resolver import ComparisonItem
from ..flow_layout import (
    CELL_HEIGHT,
    CELL_WIDTH,
    FlowLayout,
    STRATEGY_COLOR_PALETTE as _PALETTE,
)


class ValidationResultsView(QtWidgets.QWidget):
    def __init__(
        self,
        items: List[ComparisonItem],
        definitions: List[LRStrategyDefinition],
        project_root: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._items = list(items)
        self._definitions = list(definitions)
        self._project_root = project_root

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if not self._items:
            self._render_empty(outer)
            return

        # Carrega todos os results de uma vez (pickles leves; resta cacheado).
        self._loaded: List[Tuple[ComparisonItem, Any]] = []
        for it in self._items:
            res = self._load_result(it)
            if res is not None:
                self._loaded.append((it, res))

        if not self._loaded:
            lbl = QtWidgets.QLabel(
                "Falha ao carregar os resultados de validação selecionados."
            )
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #a33; font-size: 12px; padding: 20px;")
            outer.addWidget(lbl)
            return

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        # Linha 1: tabela
        content_layout.addWidget(self._build_metrics_table())

        # Linha 2: 3 plots overlay via FlowLayout
        content_layout.addWidget(
            self._build_flow_section(
                title="Plots por estratégia (DET, Tippett, ECE)",
                cells=[
                    self._build_plot_cell("Curva DET", self._draw_det_overlay),
                    self._build_plot_cell(
                        "Tippett Plot", self._draw_tippett_overlay
                    ),
                    self._build_plot_cell(
                        "Empirical Cross-Entropy (ECE)", self._draw_ece_overlay
                    ),
                ],
            )
        )

        # Linha 3: distribuição de log-LR (1 plot por estratégia).
        # Passa o índice pra cada cell receber a cor da paleta correspondente
        # à mesma linha da tabela — assim fica óbvio qual cell pertence a
        # qual estratégia, mesmo com vários cells lado a lado.
        content_layout.addWidget(
            self._build_flow_section(
                title="Distribuição de log-LR por estratégia",
                cells=[
                    self._build_png_cell(
                        item,
                        idx,
                        png_key="lr_distribution",
                        placeholder_msg=(
                            "PNG não disponível — execute a instrução"
                            " STRATEGY_EVALUATION."
                        ),
                    )
                    for idx, (item, _result) in enumerate(self._loaded)
                ],
            )
        )

        # Linha 4: curva de calibração (raw_score × log-LR) por estratégia.
        # Mesma lógica das cells de log-LR: PNG snapshot per-strategy + borda
        # esquerda colorida pra cross-reference visual com a tabela.
        content_layout.addWidget(
            self._build_flow_section(
                title="Curva de Calibração por estratégia",
                cells=[
                    self._build_png_cell(
                        item,
                        idx,
                        png_key="curve",
                        placeholder_msg=(
                            "Curva indisponível — pickle pré-curva ou"
                            " STRATEGY_EVALUATION antigo. Re-execute"
                            " a instrução."
                        ),
                    )
                    for idx, (item, _result) in enumerate(self._loaded)
                ],
            )
        )

        # Stretch final empurra tudo pra cima — sem espaço vazio entre linhas.
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Tabela de métricas ────────────────────────────────────────────────────

    def _build_metrics_table(self) -> QtWidgets.QWidget:
        wrapper = QtWidgets.QFrame()
        wrapper.setStyleSheet(
            "QFrame { background: white; border: 1px solid #d0d8e0;"
            " border-radius: 4px; }"
        )
        # Maximum vertical: cresce até a altura natural (header + linhas)
        # e PARA. Sem isso o wrapper inflaria absorvendo stretch do scroll.
        wrapper.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        wrap_layout = QtWidgets.QVBoxLayout(wrapper)
        wrap_layout.setContentsMargins(6, 6, 6, 6)
        wrap_layout.setSpacing(4)

        title = QtWidgets.QLabel("Tabela de Métricas")
        title.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 12px; font-weight: bold;"
        )
        wrap_layout.addWidget(title)

        # Ordem das colunas: agrupada por relevância forense.
        # Tier 1 (discriminação): EER, AUC, Falso+, Falso−
        # Tier 1 (calibração): Cllr, Cllr_min, cal_loss
        # Tier 3 (affine): Cllr_affine, slope
        # Contagens: n_gen, n_imp
        headers = [
            "", "Estratégia", "Resultado", "Dim. sim.",
            "EER", "Threshold EER", "AUC",
            "Falso+", "Falso−",
            "Cllr", "Cllr_min", "cal_loss",
            "Cllr_affine", "slope",
            "n_gen", "n_imp",
        ]
        # Tooltips por coluna pra deixar o significado claro sem inflar o header
        header_tooltips = {
            "Falso+": "P(log-LR > 0 | Hd) — impostores incorretamente com LR positivo",
            "Falso−": "P(log-LR < 0 | Hp) — genuínos incorretamente com LR negativo",
            "Cllr_affine": "Cllr após recalibração linear ótima (slope·LR + intercept)",
            "slope": "Inclinação a do affine fit. < 1 = over-confident; > 1 = under-confident",
            "Threshold EER": "Valor de log-LR no ponto de igual erro",
            "cal_loss": "Cllr − Cllr_min: perda por má calibração",
        }
        table = QtWidgets.QTableWidget(len(self._loaded), len(headers))
        table.setHorizontalHeaderLabels(headers)
        for col, h in enumerate(headers):
            if h in header_tooltips:
                table.horizontalHeaderItem(col).setToolTip(header_tooltips[h])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setAlternatingRowColors(True)
        table.setStyleSheet(
            "QTableWidget { font-size: 11px; gridline-color: #e0e0e0;"
            " alternate-background-color: #f8f9fa; background: white; }"
            "QHeaderView::section { background: #eef2f7; color: #2c3e50;"
            " font-weight: bold; padding: 4px 6px; border: none;"
            " border-bottom: 1px solid #c0c8d0; }"
        )

        def _fmt_pct(v) -> str:
            return f"{v*100:.2f}%" if v is not None else "n/a"

        def _fmt_f(v, decimals=4) -> str:
            return f"{v:.{decimals}f}" if v is not None else "n/a"

        def _fmt_slope(v) -> str:
            if v is None:
                return "n/a"
            arrow = "≈" if 0.95 <= v <= 1.05 else ("↑" if v > 1 else "↓")
            return f"{v:.3f} {arrow}"

        for row, (item, result) in enumerate(self._loaded):
            m = result.metrics
            n_gen = getattr(result, "n_genuines", 0)
            n_imp = getattr(result, "n_impostors", 0)
            cal_loss = max(0.0, m.cllr - m.cllr_min)
            dim = self._resolve_similarity_input_dim(item.strategy_id)
            color_hex = _PALETTE[row % len(_PALETTE)]

            color_cell = QtWidgets.QTableWidgetItem("")
            color_cell.setBackground(QtGui.QBrush(QtGui.QColor(color_hex)))
            color_cell.setToolTip(f"Cor desta linha nos plots: {color_hex}")
            table.setItem(row, 0, color_cell)

            # Tier 1+3 forense — campos podem ser None em pickles antigos
            # (pré-v2 do ValidationMetrics) → "n/a".
            cells = [
                item.strategy_name,
                item.resource_name,
                str(dim) if dim is not None else "n/a",
                _fmt_pct(m.eer),
                _fmt_f(m.eer_threshold),
                _fmt_f(m.auc) if m.auc is not None else "n/a",
                _fmt_pct(getattr(m, "wrong_sign_imp_rate", None)),
                _fmt_pct(getattr(m, "wrong_sign_gen_rate", None)),
                _fmt_f(m.cllr),
                _fmt_f(m.cllr_min),
                _fmt_f(cal_loss),
                _fmt_f(getattr(m, "cllr_affine", None)),
                _fmt_slope(getattr(m, "calibration_slope", None)),
                f"{n_gen:,}".replace(",", "."),
                f"{n_imp:,}".replace(",", "."),
            ]
            for offset, text in enumerate(cells):
                col = offset + 1
                cell = QtWidgets.QTableWidgetItem(text)
                if col >= 4:
                    cell.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, col, cell)

        table.setColumnWidth(0, 32)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 32)
        # Horizontal scrollbar quando colunas excedem largura disponível
        # (15 colunas → tabela larga; em projeto multi-strategy é normal).
        table.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        # Altura natural compacta: header + linhas + folga, cap em 160px.
        row_h = table.verticalHeader().defaultSectionSize()
        header_h = table.horizontalHeader().sizeHint().height()
        natural_h = header_h + row_h * len(self._loaded) + 6
        # Adiciona 16px pra dar espaço pra horizontal scrollbar se aparecer
        natural_h += 16
        table.setMaximumHeight(min(natural_h, 180))
        table.setMinimumHeight(min(natural_h, 180))
        wrap_layout.addWidget(table)

        return wrapper

    # ── Flow section helper ──────────────────────────────────────────────────

    def _build_flow_section(
        self, title: str, cells: List[QtWidgets.QWidget],
    ) -> QtWidgets.QWidget:
        """
        Wrapper com título + FlowLayout dos cells. Maximum vertical pra que
        a section ocupe só o espaço que precisa, sem deixar faixa vazia.
        """
        wrapper = QtWidgets.QFrame()
        wrapper.setStyleSheet(
            "QFrame { background: white; border: 1px solid #d0d8e0;"
            " border-radius: 4px; }"
        )
        wrapper.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        wrap_layout = QtWidgets.QVBoxLayout(wrapper)
        wrap_layout.setContentsMargins(6, 6, 6, 6)
        wrap_layout.setSpacing(4)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 12px; font-weight: bold;"
        )
        wrap_layout.addWidget(title_lbl)

        # Container só pra abrigar o FlowLayout (FlowLayout precisa de
        # parent QWidget pra calcular contentsMargins e style).
        flow_container = QtWidgets.QWidget()
        flow_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        flow = FlowLayout(flow_container, margin=0, h_spacing=8, v_spacing=8)
        for c in cells:
            flow.addWidget(c)

        wrap_layout.addWidget(flow_container)
        return wrapper

    # ── Plot cell builders ───────────────────────────────────────────────────

    def _build_plot_cell(self, title: str, draw_fn) -> QtWidgets.QWidget:
        """
        Cell padrão para plots overlay (DET/Tippett/ECE). Tamanho fixo,
        figure com layout_engine 'constrained' que ajusta automaticamente
        ticks, xlabel e legendas externas sem sobreposição.
        """
        cell = QtWidgets.QFrame()
        cell.setStyleSheet(
            "QFrame { background: white; border: 1px solid #e0e0e0;"
            " border-radius: 3px; }"
        )
        cell.setFixedSize(CELL_WIDTH, CELL_HEIGHT)

        cell_layout = QtWidgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(4, 4, 4, 4)
        cell_layout.setSpacing(2)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 11px; font-weight: bold;"
            " padding: 2px 4px;"
        )
        cell_layout.addWidget(title_lbl)

        figure = Figure(figsize=(5, 3.5), dpi=100)
        # Constrained layout: o engine moderno do matplotlib que evita
        # sobreposição entre xlabel, ticks, legenda externa, e elimina
        # espaço branco residual. Substitui tight_layout(rect=...) frágil.
        figure.set_layout_engine("constrained")

        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, cell)
        cell_layout.addWidget(toolbar)
        cell_layout.addWidget(canvas, 1)

        try:
            draw_fn(figure)
        except Exception as exc:
            figure.clear()
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Erro:\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="#a33", fontsize=10)
            ax.axis("off")
        canvas.draw_idle()
        return cell

    def _build_png_cell(
        self,
        item: ComparisonItem,
        idx: int,
        *,
        png_key: str,
        placeholder_msg: str,
    ) -> QtWidgets.QWidget:
        """
        Cell per-strategy que exibe um PNG snapshot do STRATEGY_EVALUATION.

        Usado tanto pela seção de distribuição log-LR (`png_key`
        ``"lr_distribution"``) quanto pela curva de calibração
        (`png_key` ``"curve"``). Reproduz o PNG já gravado pelo executor
        em vez de reconstruir matplotlib — mantém o visual idêntico ao
        do widget inline.

        Recebe `idx` pra colorir a borda esquerda com a mesma cor da
        paleta usada na linha correspondente da tabela e nos overlays
        (cross-reference visual rápido).

        Os overlays DET/Tippett/ECE (`_build_plot_cell`) permanecem
        matplotlib porque combinam várias estratégias num mesmo gráfico
        — o PNG por-recurso não cobre esse caso.
        """
        cell = QtWidgets.QFrame()
        color_hex = _PALETTE[idx % len(_PALETTE)]
        cell.setStyleSheet(
            "QFrame { background: white; border: 1px solid #e0e0e0;"
            f" border-left: 4px solid {color_hex};"
            " border-radius: 3px; }"
        )
        cell.setFixedSize(CELL_WIDTH, CELL_HEIGHT)

        cell_layout = QtWidgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(4, 4, 4, 4)
        cell_layout.setSpacing(2)

        header = QtWidgets.QLabel(
            f"<b>[{item.strategy_name}]</b> &nbsp; {item.resource_name}"
        )
        header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 11px; padding: 2px 4px;"
        )
        header.setWordWrap(True)
        cell_layout.addWidget(header)

        widget = PngSnapshotWidget(parent=cell)
        cell_layout.addWidget(widget, 1)

        png_path = self._resolve_validation_png(item, png_key)
        if png_path is None or not widget.show_image(png_path):
            widget.show_placeholder(placeholder_msg)
        return cell

    def _resolve_validation_png(
        self, item: ComparisonItem, png_key: str,
    ) -> Optional[Path]:
        """Path absoluto do PNG snapshot do output de STRATEGY_EVALUATION.

        `png_key` é uma das chaves canônicas de
        `validation_result_plot.PLOT_KEYS` (`lr_distribution`, `tippett`,
        `det`, `ece`, `curve`).
        """
        if (
            self._project_root is None
            or not item.strategy_id
            or not item.resource_id
        ):
            return None
        return validation_result_png_paths(
            self._project_root, item.strategy_id, item.resource_id,
            strategy_name=item.strategy_name or "",
        ).get(png_key)

    # ── Plot drawing functions (overlays) ────────────────────────────────────

    def _draw_det_overlay(self, figure: Figure) -> None:
        from scipy.stats import norm
        ax = figure.add_subplot(111)
        eps = 1e-6
        any_drawn = False
        for idx, (item, result) in enumerate(self._loaded):
            pd = result.plot_data
            fpr = np.asarray(pd.fpr_array, dtype=np.float64)
            fnr = np.asarray(pd.fnr_array, dtype=np.float64)
            if fpr.size == 0 or fnr.size == 0:
                continue
            fpr_c = np.clip(fpr, eps, 1.0 - eps)
            fnr_c = np.clip(fnr, eps, 1.0 - eps)
            color = _PALETTE[idx % len(_PALETTE)]
            label = self._legend_label(item, result)
            # Anexa o valor de EER no label da legenda; aparece num único
            # entry combinado com o marker do scatter abaixo.
            eer_val = getattr(getattr(result, "metrics", None), "eer", None)
            if eer_val is not None:
                label = f"{label} (EER={eer_val*100:.2f}%)"
            ax.plot(
                norm.ppf(fpr_c), norm.ppf(fnr_c),
                color=color, linewidth=1.6,
                label=label, alpha=0.9,
            )
            # Marca o EER na curva: scatter na cor da estratégia, sem
            # crosshair (várias estratégias = poluição visual). Hidden
            # da legenda — o ponto compartilha entry com a linha.
            if eer_val is not None:
                try:
                    eer_n = norm.ppf(np.clip(eer_val, eps, 1.0 - eps))
                    ax.scatter(
                        [eer_n], [eer_n], color=color, s=55, zorder=5,
                        edgecolor="#222", linewidth=1.0, label="_nolegend_",
                    )
                except Exception:
                    pass
            any_drawn = True

        if not any_drawn:
            ax.text(0.5, 0.5, "(sem dados de DET)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        # 4 ticks horizontais bem espaçados em escala normit.
        tick_pcts = [0.001, 0.01, 0.1, 0.4]
        tick_locs = norm.ppf(tick_pcts)
        tick_labels = [f"{p*100:g}%" for p in tick_pcts]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_locs)
        ax.set_yticklabels(tick_labels)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=9)
        ax.set_ylabel("Taxa de Falsos Negativos (FNR)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        # Constrained layout cuida do posicionamento — bbox_to_anchor
        # negativo joga a legenda pra fora do axes; o engine reserva
        # automaticamente o espaço necessário.
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=1, fontsize=8, frameon=True,
        )

    def _draw_tippett_overlay(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)
        any_drawn = False
        for idx, (item, result) in enumerate(self._loaded):
            pd = result.plot_data
            gen = np.sort(np.asarray(pd.genuine_log_lrs, dtype=np.float64))
            imp = np.sort(np.asarray(pd.impostor_log_lrs, dtype=np.float64))
            if gen.size == 0 or imp.size == 0:
                continue
            gen_cum = np.arange(1, gen.size + 1) / gen.size
            imp_cum_above = 1.0 - np.arange(1, imp.size + 1) / imp.size
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(
                gen, gen_cum, color=color, linewidth=1.4,
                label=self._legend_label(item, result), alpha=0.9,
            )
            ax.plot(imp, imp_cum_above, color=color, linewidth=1.4,
                    linestyle="--", alpha=0.9)
            any_drawn = True

        if not any_drawn:
            ax.text(0.5, 0.5, "(sem dados)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        ax.axvline(0.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("log-LR (limiar de decisão)", fontsize=9)
        ax.set_ylabel("Proporção cumulativa", fontsize=9)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=1, fontsize=8, frameon=True,
            title="Cheia: P(LR≤x|Hp)  •  Tracejada: P(LR>x|Hd)",
            title_fontsize=8,
        )

    def _draw_ece_overlay(self, figure: Figure) -> None:
        ax = figure.add_subplot(111)
        any_drawn = False
        ref_x = None  # x do primeiro item válido — usado pra plotar a ref.
        max_y = 1.05  # ECE_null pico = 1 bit; default cap em 1.05.
        for idx, (item, result) in enumerate(self._loaded):
            pd = result.plot_data
            x = np.asarray(pd.ece_prior_log_odds, dtype=np.float64)
            y_real = np.asarray(pd.ece_values, dtype=np.float64)
            y_min = np.asarray(pd.ece_min_values, dtype=np.float64)
            if x.size == 0 or y_real.size == 0:
                continue
            if ref_x is None:
                ref_x = x
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(
                x, y_real, color=color, linewidth=1.5,
                label=self._legend_label(item, result), alpha=0.9,
            )
            if y_min.size == y_real.size:
                ax.plot(x, y_min, color=color, linewidth=1.2,
                        linestyle="--", alpha=0.85)
            # Se algum modelo exceder 1.0 (pior que sistema neutro em
            # alguma região), expande ylim adaptativamente.
            ymax_local = float(np.nanmax(y_real))
            if ymax_local * 1.05 > max_y:
                max_y = ymax_local * 1.05
            any_drawn = True

        if not any_drawn:
            ax.text(0.5, 0.5, "(sem dados de ECE)", ha="center", va="center",
                    transform=ax.transAxes, color="#888")
            return

        # Curva de referência (sistema neutro LR=1): entropia do prior =
        # -P·log2(P) - (1-P)·log2(1-P), onde P = sigmoid(x · ln10).
        # Pico em prior log10-odds=0 → P=0.5 → ECE_null=1 bit.
        # Convenção forense (Brümmer & van Leeuwen 2006). Modelos só são
        # informativos quando ECE_modelo fica abaixo dela.
        p = 1.0 / (1.0 + np.exp(-ref_x * np.log(10.0)))
        p_safe = np.clip(p, 1e-12, 1.0 - 1e-12)
        y_null = (-p_safe * np.log(p_safe)
                  - (1.0 - p_safe) * np.log(1.0 - p_safe)) / np.log(2.0)
        ax.plot(
            ref_x, y_null, color="#888", linewidth=1.2, linestyle=":",
            label="Sistema neutro (LR=1)", zorder=1,
        )

        ax.axvline(0.0, color="#444", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Prior log₁₀-odds (P(Hp)/P(Hd))", fontsize=9)
        ax.set_ylabel("Cross-entropy (bits)", fontsize=9)
        # Y vai até 1.05 (entropia máxima do prior = 1.0) ou além, se algum
        # modelo for pior que neutro em algum prior.
        ax.set_ylim(0, max_y)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.18),
            ncol=1, fontsize=8, frameon=True,
            title="Cheia: ECE modelo  •  Tracejada: ECE mínimo (PAVA)",
            title_fontsize=8,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _legend_label(self, item: ComparisonItem, result: Any) -> str:
        return f"[{item.strategy_name}] {item.resource_name}"

    def _load_result(self, item: ComparisonItem) -> Any:
        try:
            with open(item.artifact_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _resolve_similarity_input_dim(self, strategy_id: str) -> Optional[int]:
        if self._project_root is None:
            return None
        defn = next(
            (d for d in self._definitions if d.strategy_id == strategy_id),
            None,
        )
        if defn is None:
            return None

        sim_model_resource_id = None
        for instr in defn.instructions:
            if instr.instruction_type != StrategyInstructionType.SIMILARITY_MODEL_TRAINING:
                continue
            sim_model_resource_id = getattr(
                instr, "trained_similarity_model_resource_id", None
            )
            if sim_model_resource_id:
                break
        if not sim_model_resource_id:
            return None

        res = defn.get_resource_by_id(sim_model_resource_id)
        if res is None or not res.has_artifact:
            return None
        rel = res.artifact_ref.path.replace("\\", "/")
        abs_path = Path(self._project_root) / rel
        if not abs_path.exists():
            return None
        try:
            with open(abs_path, "rb") as f:
                model = pickle.load(f)
            dim = getattr(model, "input_dimension", None)
            return int(dim) if dim is not None else None
        except Exception:
            return None

    def _render_empty(self, layout: QtWidgets.QVBoxLayout) -> None:
        lbl = QtWidgets.QLabel(
            "Marque um ou mais itens à esquerda para comparar."
        )
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        layout.addStretch(1)
        layout.addWidget(lbl)
        layout.addStretch(2)
