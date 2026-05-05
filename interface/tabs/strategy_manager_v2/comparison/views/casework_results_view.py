# interface/tabs/strategy_manager_v2/comparison/views/casework_results_view.py

"""
View do tipo CASEWORK_RESULTS.

Mesma estrutura de validation: tabela compacta no topo
(cor + estratégia + resultado + score_type + n_gen + n_imp + n_case +
case_mean + EER thr) seguida de FlowLayout com 1 cell por result. Cada
cell tem borda esquerda colorida correspondente à linha da tabela.
"""

from __future__ import annotations

import pickle
from typing import Any, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from ...strategy_editor.result_widgets.casework_visualization_widget import (
    CaseworkVisualizationWidget,
)
from ..availability_resolver import ComparisonItem
from ..flow_layout import (
    CELL_HEIGHT,
    CELL_WIDTH,
    FlowLayout,
    STRATEGY_COLOR_PALETTE,
)


class CaseworkResultsView(QtWidgets.QWidget):
    def __init__(
        self,
        items: List[ComparisonItem],
        resource_loader: Any = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._items = list(items)
        self._resource_loader = resource_loader

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if not self._items:
            self._render_empty(outer)
            return

        # Pre-load todos os pickles. Caso usado tanto pra tabela quanto pelos
        # cells — caching evita ler 2× e mantém os widgets sincronizados.
        self._loaded: List[Tuple[ComparisonItem, Any]] = []
        for it in self._items:
            res = self._load_result(it)
            if res is not None:
                self._loaded.append((it, res))

        if not self._loaded:
            lbl = QtWidgets.QLabel(
                "Falha ao carregar os resultados de caso selecionados."
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

        content_layout.addWidget(self._build_table())
        content_layout.addWidget(self._build_flow_section())
        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Tabela ────────────────────────────────────────────────────────────────

    def _build_table(self) -> QtWidgets.QWidget:
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

        title = QtWidgets.QLabel("Resultados do Caso")
        title.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 12px; font-weight: bold;"
        )
        wrap_layout.addWidget(title)

        headers = [
            "", "Estratégia", "Resultado", "Tipo", "n_gen (ref)",
            "n_imp (ref)", "n_caso", "Média caso", "EER thr",
        ]
        table = QtWidgets.QTableWidget(len(self._loaded), len(headers))
        table.setHorizontalHeaderLabels(headers)
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

        for row, (item, result) in enumerate(self._loaded):
            color_hex = STRATEGY_COLOR_PALETTE[row % len(STRATEGY_COLOR_PALETTE)]

            color_cell = QtWidgets.QTableWidgetItem("")
            color_cell.setBackground(QtGui.QBrush(QtGui.QColor(color_hex)))
            color_cell.setToolTip(f"Cor desta linha no plot: {color_hex}")
            table.setItem(row, 0, color_cell)

            score_type = getattr(result, "score_type", None) or "n/a"
            n_gen = getattr(result, "n_genuine", None)
            n_imp = getattr(result, "n_impostor", None)
            n_case = getattr(result, "n_case", None)
            cs = getattr(result, "case_summary", None)
            case_mean = (
                cs.get("mean") if isinstance(cs, dict) else None
            )
            eer_thr = getattr(result, "eer_threshold", None)

            cells = [
                item.strategy_name,
                item.resource_name,
                str(score_type),
                f"{n_gen:,}".replace(",", ".") if n_gen is not None else "n/a",
                f"{n_imp:,}".replace(",", ".") if n_imp is not None else "n/a",
                f"{n_case:,}".replace(",", ".") if n_case is not None else "n/a",
                f"{case_mean:+.3f}" if case_mean is not None else "n/a",
                f"{eer_thr:.4f}" if eer_thr is not None else "—",
            ]
            for offset, text in enumerate(cells):
                col = offset + 1
                cell = QtWidgets.QTableWidgetItem(text)
                if col >= 4:  # numéricos alinhados à direita
                    cell.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, col, cell)

        table.setColumnWidth(0, 32)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 32)

        row_h = table.verticalHeader().defaultSectionSize()
        header_h = table.horizontalHeader().sizeHint().height()
        natural_h = header_h + row_h * len(self._loaded) + 6
        table.setMaximumHeight(min(natural_h, 160))
        table.setMinimumHeight(min(natural_h, 160))
        wrap_layout.addWidget(table)

        return wrapper

    # ── Flow section ─────────────────────────────────────────────────────────

    def _build_flow_section(self) -> QtWidgets.QWidget:
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

        title_lbl = QtWidgets.QLabel("Distribuição forense por estratégia")
        title_lbl.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 12px; font-weight: bold;"
        )
        wrap_layout.addWidget(title_lbl)

        flow_container = QtWidgets.QWidget()
        flow_container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        flow = FlowLayout(flow_container, margin=0, h_spacing=8, v_spacing=8)
        for idx, (item, result) in enumerate(self._loaded):
            flow.addWidget(self._build_cell(item, result, idx))

        wrap_layout.addWidget(flow_container)
        return wrapper

    # ── Cell ─────────────────────────────────────────────────────────────────

    def _build_cell(
        self, item: ComparisonItem, result: Any, idx: int,
    ) -> QtWidgets.QWidget:
        color_hex = STRATEGY_COLOR_PALETTE[idx % len(STRATEGY_COLOR_PALETTE)]

        cell = QtWidgets.QFrame()
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

        # show_title=False — header próprio do cell substitui o do plot.
        widget = CaseworkVisualizationWidget(
            parent=cell,
            result=result,
            show_header=False,
            show_toolbar=False,
            show_help=False,
            show_title=False,
        )
        cell_layout.addWidget(widget, 1)

        return cell

    def _load_result(self, item: ComparisonItem) -> Any:
        try:
            with open(item.artifact_path, "rb") as f:
                return pickle.load(f)
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
