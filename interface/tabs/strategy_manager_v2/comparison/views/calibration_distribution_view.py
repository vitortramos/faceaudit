# interface/tabs/strategy_manager_v2/comparison/views/calibration_distribution_view.py

"""
View do tipo CALIBRATION_DATASET_DISTRIBUTION.

Plots dispostos via FlowLayout (canônico Qt) — quebra automática quando
não cabe mais um cell na largura disponível. Cells com tamanho fixo
(`CELL_WIDTH × CELL_HEIGHT`) garantem visual consistente entre os
3 tipos de comparação.
"""

from __future__ import annotations

from typing import Any, List, Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)

from ...strategy_editor.result_widgets.score_distribution_widget import (
    ScoreDistributionWidget,
)
from ..availability_resolver import ComparisonItem
from ..flow_layout import CELL_HEIGHT, CELL_WIDTH, FlowLayout


class CalibrationDistributionView(QtWidgets.QWidget):
    """
    FlowLayout com 1 cell por item. Sem tabela, só plots.
    """

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

        # Container com FlowLayout — Qt cuida do reflow + heightForWidth.
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
        # Tamanho fixo padronizado — todos os grids de comparação usam o
        # mesmo CELL_WIDTH×CELL_HEIGHT pra visual consistente.
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

        widget = ScoreDistributionWidget(
            parent=cell, score_dataset=None, show_toolbar=False,
        )
        cell_layout.addWidget(widget, 1)

        # Tenta sidecar primeiro (rápido). Fallback: hidrata parquet.
        summary = SimilarityScoreDataset.try_load_distribution_summary(
            item.artifact_path
        )
        if summary is not None:
            widget.set_distribution_summary(summary)
            return cell

        if self._resource_loader is not None:
            try:
                from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
                from faceaudit.enums.strategy_resource_type import StrategyResourceType
                from faceaudit.models.storage.strategy_artifact_reference import (
                    StrategyArtifactReference,
                )
                from faceaudit.models.strategy.strategy_resource import StrategyResource

                ref = StrategyArtifactReference(
                    artifact_type=StrategyArtifactType.SIMILARITY_SCORES,
                    path=str(item.artifact_path),
                    hash=None,
                )
                placeholder = StrategyResource(
                    resource_id=item.resource_id,
                    name=item.resource_name,
                    resource_type=StrategyResourceType.SIMILARITY_SCORES,
                    artifact_ref=ref,
                )
                self._resource_loader.hydrate(placeholder)
                if placeholder.has_payload:
                    widget.set_dataset(placeholder.payload)
            except Exception:
                pass
        return cell

    def _render_empty(self, layout: QtWidgets.QVBoxLayout) -> None:
        lbl = QtWidgets.QLabel(
            "Marque um ou mais itens à esquerda para comparar."
        )
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        layout.addStretch(1)
        layout.addWidget(lbl)
        layout.addStretch(2)
