# interface/tabs/strategy_manager_v2/comparison/views/casework_results_view.py

"""
View do tipo CASEWORK_RESULTS.

FlowLayout (canônico Qt) com 1 cell por item. Cells de tamanho fixo —
mesmo CELL_WIDTH×CELL_HEIGHT dos outros tipos de comparação.
"""

from __future__ import annotations

import pickle
from typing import Any, List, Optional

from PyQt6 import QtCore, QtWidgets

from ...strategy_editor.result_widgets.casework_visualization_widget import (
    CaseworkVisualizationWidget,
)
from ..availability_resolver import ComparisonItem
from ..flow_layout import CELL_HEIGHT, CELL_WIDTH, FlowLayout


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

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        container = QtWidgets.QWidget()
        flow = FlowLayout(container, margin=8, h_spacing=8, v_spacing=8)
        for item in self._items:
            flow.addWidget(self._build_cell(item))

        scroll.setWidget(container)
        outer.addWidget(scroll)

    def _build_cell(self, item: ComparisonItem) -> QtWidgets.QWidget:
        cell = QtWidgets.QFrame()
        cell.setStyleSheet(
            "QFrame { background: white; border: 1px solid #d0d8e0;"
            " border-radius: 4px; }"
        )
        cell.setFixedSize(CELL_WIDTH, CELL_HEIGHT)

        cell_layout = QtWidgets.QVBoxLayout(cell)
        cell_layout.setContentsMargins(6, 6, 6, 6)
        cell_layout.setSpacing(4)

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

        result = self._load_result(item)
        widget = CaseworkVisualizationWidget(
            parent=cell,
            result=result,
            show_header=False,
            show_toolbar=False,
            show_help=False,
        )
        cell_layout.addWidget(widget, 1)

        if result is not None:
            summary_lbl = QtWidgets.QLabel(self._summary_text(result))
            summary_lbl.setStyleSheet(
                "color: #555; font-size: 10px; padding: 2px 4px;"
            )
            summary_lbl.setWordWrap(True)
            cell_layout.addWidget(summary_lbl)

        return cell

    def _load_result(self, item: ComparisonItem) -> Any:
        try:
            with open(item.artifact_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    @staticmethod
    def _summary_text(result: Any) -> str:
        try:
            parts = []
            score_type = getattr(result, "score_type", None)
            if score_type:
                parts.append(str(score_type))
            n_case = getattr(result, "n_case", None)
            if n_case is not None:
                parts.append(f"caso: {n_case} pares")
            cs = getattr(result, "case_summary", None)
            if isinstance(cs, dict) and cs.get("mean") is not None:
                parts.append(f"mean={cs['mean']:+.2f}")
            eer = getattr(result, "eer_threshold", None)
            if eer is not None:
                parts.append(f"EER thr={eer:.4f}")
            return " · ".join(parts) if parts else ""
        except Exception:
            return ""

    def _render_empty(self, layout: QtWidgets.QVBoxLayout) -> None:
        lbl = QtWidgets.QLabel(
            "Marque um ou mais itens à esquerda para comparar."
        )
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        layout.addStretch(1)
        layout.addWidget(lbl)
        layout.addStretch(2)
