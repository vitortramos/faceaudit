"""Tabela das comparações que produziram os escores.

A distribuição mostra a forma do conjunto; esta tabela mostra de onde cada
ponto veio. Num confronto de caso é a informação central do exame, e fora
dele são os extremos que revelam onde o sistema erra.

A seleção das linhas é decidida no backend
(`faceaudit.services.score_comparison_table_service`). Este widget recebe
um `ComparisonTable` pronto e apenas o exibe, incluindo a frase que explica
por que são aquelas linhas — sem ela, o regime de extremos passa por tabela
completa e uma amostra enviesada é lida como se fosse o conjunto.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.plot_number_format import formatar_numero
from faceaudit.services.score_comparison_table_service import (
    CLASSE_GENUINO,
    CLASSE_IMPOSTOR,
    ComparisonTable,
)
from interface.theme import apply_themed_stylesheet, current_token_fn

from ..widget_helpers import WrappedLabel


class _NumericItem(QtWidgets.QTableWidgetItem):
    """Célula que ordena pelo valor, não pelo texto.

    Com o texto em convenção brasileira, a ordenação alfabética do
    `QTableWidgetItem` põe "0,9" antes de "0,85" — exatamente o erro que
    uma tabela de escores não pode cometer.
    """

    def __init__(self, texto: str, valor: float):
        super().__init__(texto)
        self._valor = float(valor)
        self.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

    def __lt__(self, outro) -> bool:  # type: ignore[override]
        if isinstance(outro, _NumericItem):
            return self._valor < outro._valor
        return super().__lt__(outro)


class ScoreComparisonTableWidget(QtWidgets.QWidget):
    """Nota explicativa + tabela de comparações."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        table: Optional[ComparisonTable] = None,
    ):
        super().__init__(parent)
        self._table: Optional[ComparisonTable] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._note = WrappedLabel("", parent=self)
        apply_themed_stylesheet(
            self._note,
            lambda t: (
                "font-size: 11px; color: " + t("TEXT_MUTED") + ";"
                " background: " + t("SURFACE_RAISED") + ";"
                " border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                " padding: 6px 8px;"
            ),
        )
        layout.addWidget(self._note)

        self._grid = QtWidgets.QTableWidget(parent=self)
        self._grid.setAlternatingRowColors(True)
        self._grid.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._grid.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._grid.verticalHeader().setVisible(False)
        self._grid.setSortingEnabled(True)
        self._grid.setWordWrap(False)
        apply_themed_stylesheet(
            self._grid,
            lambda t: (
                "QTableWidget {"
                "  background: " + t("SURFACE") + "; color: " + t("TEXT") + ";"
                "  border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                "  gridline-color: " + t("BORDER") + "; font-size: 11px;"
                "}"
                "QTableWidget::item { padding: 2px 6px; }"
                "QTableWidget::item:selected {"
                "  background: " + t("HIGHLIGHT") + ";"
                # `color: inherit` resolveria para preto no tema escuro.
                "  color: " + t("HIGHLIGHT_TEXT") + ";"
                "}"
                "QHeaderView::section {"
                "  background: " + t("SURFACE_ALT") + "; color: " + t("TEXT") + ";"
                "  border: none; border-bottom: 1px solid " + t("BORDER") + ";"
                "  padding: 4px 6px; font-weight: bold; font-size: 11px;"
                "}"
            ),
        )
        layout.addWidget(self._grid, 1)

        if table is not None:
            self.set_table(table)
        else:
            self.show_placeholder()

    # ── API pública ──────────────────────────────────────────────────────

    def show_placeholder(self, mensagem: str = "") -> None:
        self._table = None
        self._grid.setSortingEnabled(False)
        self._grid.clear()
        self._grid.setRowCount(0)
        self._grid.setColumnCount(0)
        self._grid.setSortingEnabled(True)
        self._note.setText(
            mensagem or "Execute a instrução para ver as comparações."
        )

    def set_table(self, table: Optional[ComparisonTable]) -> None:
        if table is None:
            self.show_placeholder()
            return
        self._table = table
        self._note.setText(table.note)

        colunas = [
            "Identidade (probe)", "Face (probe)",
            "Identidade (referência)", "Face (referência)",
            "Escore", "Classe",
        ]
        if table.has_log_lr:
            colunas.append("log-LR")

        # Preencher com a ordenação ligada reordena a cada inserção e
        # embaralha as linhas que o backend já ordenou.
        self._grid.setSortingEnabled(False)
        self._grid.clear()
        self._grid.setColumnCount(len(colunas))
        self._grid.setHorizontalHeaderLabels(colunas)
        self._grid.setRowCount(len(table.rows))

        t = current_token_fn()
        cor_classe = {
            CLASSE_GENUINO: QtGui.QColor(t("SUCCESS")),
            CLASSE_IMPOSTOR: QtGui.QColor(t("DANGER")),
        }

        for linha, row in enumerate(table.rows):
            self._grid.setItem(linha, 0, QtWidgets.QTableWidgetItem(row.probe_identity))
            self._grid.setItem(linha, 1, QtWidgets.QTableWidgetItem(row.probe_face))
            self._grid.setItem(
                linha, 2, QtWidgets.QTableWidgetItem(row.reference_identity)
            )
            self._grid.setItem(
                linha, 3, QtWidgets.QTableWidgetItem(row.reference_face)
            )
            self._grid.setItem(
                linha, 4, _NumericItem(formatar_numero(row.score, 4), row.score)
            )
            classe = QtWidgets.QTableWidgetItem(row.classe_label)
            cor = cor_classe.get(row.classe)
            if cor is not None:
                classe.setForeground(cor)
            self._grid.setItem(linha, 5, classe)
            if table.has_log_lr:
                if row.log_lr is None:
                    self._grid.setItem(linha, 6, QtWidgets.QTableWidgetItem("—"))
                else:
                    self._grid.setItem(
                        linha, 6,
                        _NumericItem(formatar_numero(row.log_lr, 3), row.log_lr),
                    )

        self._grid.setSortingEnabled(True)
        cabecalho = self._grid.horizontalHeader()
        cabecalho.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        cabecalho.setStretchLastSection(True)
