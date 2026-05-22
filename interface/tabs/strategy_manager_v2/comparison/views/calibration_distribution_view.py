# interface/tabs/strategy_manager_v2/comparison/views/calibration_distribution_view.py

"""
View do tipo CALIBRATION_DATASET_DISTRIBUTION.

Mesma estrutura visual de validation_results_view: tabela compacta no
topo (cor + estratégia + nome do recurso + n_gen + n_imp) seguida de um
FlowLayout com 1 cell por item. Cada cell tem borda esquerda colorida
correspondendo à linha da tabela — cross-reference visual rápido.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.similarity.distribution_plot import (
    score_distribution_png_path,
)
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)

from ...strategy_editor.result_widgets.png_snapshot_widget import (
    PngSnapshotWidget,
)
from ..availability_resolver import ComparisonItem
from ..flow_layout import (
    CELL_HEIGHT,
    CELL_WIDTH,
    FlowLayout,
    STRATEGY_COLOR_PALETTE,
)


class CalibrationDistributionView(QtWidgets.QWidget):
    """Tabela + grid de plots, todos consistentes visualmente."""

    def __init__(
        self,
        items: List[ComparisonItem],
        resource_loader: Any = None,
        project_root: Optional[Path] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._items = list(items)
        self._resource_loader = resource_loader
        # project_root explícito (preferido) ou derivado do loader.
        if project_root is None and resource_loader is not None:
            storage = getattr(resource_loader, "_storage", None)
            project_root = getattr(storage, "project_root", None) if storage else None
        self._project_root = project_root

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if not self._items:
            self._render_empty(outer)
            return

        # Pre-resolve sidecar de cada item — populamos a tabela com n_gen
        # e n_imp do sidecar (sem hidratar parquet completo). Cada entrada
        # vira (item, summary_or_None) e é reusada na construção de cada cell.
        self._loaded: List[Tuple[ComparisonItem, Optional[dict]]] = [
            (
                item,
                SimilarityScoreDataset.try_load_distribution_summary(
                    item.artifact_path
                ),
            )
            for item in self._items
        ]

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        # Linha 1: tabela com cor / estratégia / resultado / n_gen / n_imp
        content_layout.addWidget(self._build_table())

        # Linha 2: grid de plots, 1 cell por item, com borda esquerda colorida
        flow_section = self._build_flow_section()
        content_layout.addWidget(flow_section)

        content_layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll)

    # ── Tabela de cabeçalho ───────────────────────────────────────────────────

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

        title = QtWidgets.QLabel("Datasets de Calibração")
        title.setStyleSheet(
            "background: transparent; border: none;"
            " color: #2c3e50; font-size: 12px; font-weight: bold;"
        )
        wrap_layout.addWidget(title)

        headers = ["", "Estratégia", "Resultado", "n_gen", "n_imp"]
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

        for row, (item, summary) in enumerate(self._loaded):
            color_hex = STRATEGY_COLOR_PALETTE[row % len(STRATEGY_COLOR_PALETTE)]

            color_cell = QtWidgets.QTableWidgetItem("")
            color_cell.setBackground(QtGui.QBrush(QtGui.QColor(color_hex)))
            color_cell.setToolTip(f"Cor desta linha no plot: {color_hex}")
            table.setItem(row, 0, color_cell)

            n_gen = summary.get("n_gen") if summary else None
            n_imp = summary.get("n_imp") if summary else None

            cells = [
                item.strategy_name,
                item.resource_name,
                f"{n_gen:,}".replace(",", ".") if n_gen is not None else "n/a",
                f"{n_imp:,}".replace(",", ".") if n_imp is not None else "n/a",
            ]
            for offset, text in enumerate(cells):
                col = offset + 1
                cell = QtWidgets.QTableWidgetItem(text)
                if col >= 3:  # numéricos alinhados à direita
                    cell.setTextAlignment(
                        QtCore.Qt.AlignmentFlag.AlignRight
                        | QtCore.Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, col, cell)

        table.setColumnWidth(0, 32)
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 32)

        # Altura compacta — header + linhas + folga, cap em 160 px.
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

        title_lbl = QtWidgets.QLabel("Distribuição por dataset de calibração")
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
        for idx, (item, summary) in enumerate(self._loaded):
            flow.addWidget(self._build_cell(item, summary, idx))

        wrap_layout.addWidget(flow_container)
        return wrapper

    # ── Cell ─────────────────────────────────────────────────────────────────

    def _build_cell(
        self, item: ComparisonItem, summary: Optional[dict], idx: int,
    ) -> QtWidgets.QWidget:
        color_hex = STRATEGY_COLOR_PALETTE[idx % len(STRATEGY_COLOR_PALETTE)]

        cell = QtWidgets.QFrame()
        # Borda esquerda 4 px na cor da estratégia (mesma da linha da tabela).
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

        # PngSnapshotWidget — usa o PNG persistido pelo executor de
        # SIMILARITY_SCORING (output do tipo SIMILARITY_SCORES). Pula
        # totalmente a hidratação do parquet (que pode ter milhões de
        # pares) e o rebuild matplotlib.
        widget = PngSnapshotWidget(parent=cell)
        cell_layout.addWidget(widget, 1)

        png_path = self._resolve_png_path(item)
        if png_path is None or not widget.show_image(png_path):
            widget.show_placeholder(
                "PNG não disponível — execute a instrução SIMILARITY_SCORING."
            )

        return cell

    def _resolve_png_path(self, item: ComparisonItem) -> Optional[Path]:
        """Path absoluto do PNG do output `SIMILARITY_SCORES` (não calibrado
        — o kind CALIBRATION_DATASET_DISTRIBUTION lista apenas inputs de
        calibração, que são scores RAW). None se faltar project_root."""
        if self._project_root is None or not item.strategy_id or not item.resource_id:
            return None
        return score_distribution_png_path(
            self._project_root, item.strategy_id, item.resource_id,
            strategy_name=item.strategy_name or "",
            calibrated=False,
        )

    def _render_empty(self, layout: QtWidgets.QVBoxLayout) -> None:
        lbl = QtWidgets.QLabel(
            "Marque um ou mais itens à esquerda para comparar."
        )
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        layout.addStretch(1)
        layout.addWidget(lbl)
        layout.addStretch(2)
