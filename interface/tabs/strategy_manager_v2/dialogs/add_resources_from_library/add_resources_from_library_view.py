# interface/tabs/strategy_manager_v2/dialogs/add_resources_from_library/
# add_resources_from_library_view.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PyQt6 import QtCore, QtWidgets

from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)


class AddResourcesFromLibraryView(QtWidgets.QDialog):
    """
    Diálogo de adição em lote de recursos da biblioteca de configurações
    ao projeto atual.

    Mostra 4 abas — Datasets, Split Configs, Mining Configs, Pipeline
    Configs — cada uma com a lista correspondente da biblioteca em modo
    multi-seleção. Confirmar copia todos os selecionados para as pastas
    do projeto via `LocalProjectArtifactStorage.copy_artifact_to_project`.

    Uso:
        view = AddResourcesFromLibraryView(parent)
        view.populate(datasets=..., splits=..., minings=..., pipelines=...)
        if view.exec() == QDialog.DialogCode.Accepted:
            selections = view.selected_refs()
            # selections: dict com 4 listas de StrategyArtifactReference
    """

    _KEYS = ("datasets", "splits", "minings", "pipelines")
    _LABELS = {
        "datasets": "Datasets",
        "splits": "Split Configs",
        "minings": "Mining Configs",
        "pipelines": "Pipeline Configs",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Recursos ao Projeto")
        self.setMinimumSize(720, 480)
        self.setModal(True)
        self._lists: Dict[str, QtWidgets.QListWidget] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QtWidgets.QLabel(
            "Selecione os recursos da biblioteca para adicionar ao projeto. "
            "Use <b>Ctrl+clique</b> ou <b>Shift+clique</b> para multi-selecionar."
        )
        header.setWordWrap(True)
        header.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(header)

        self.tabs = QtWidgets.QTabWidget()
        for key in self._KEYS:
            list_widget = QtWidgets.QListWidget()
            list_widget.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection,
            )
            list_widget.setAlternatingRowColors(True)
            list_widget.itemSelectionChanged.connect(self._update_counts)
            self._lists[key] = list_widget
            self.tabs.addTab(list_widget, self._LABELS[key])
        layout.addWidget(self.tabs, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        self.btn_ok = QtWidgets.QPushButton("Adicionar")
        self.btn_ok.setStyleSheet(
            "QPushButton { border: 1px solid #1565c0; border-radius: 4px;"
            " background: #1976d2; color: white; padding: 5px 14px; "
            " font-weight: bold; }"
            "QPushButton:hover { background: #1565c0; }"
            "QPushButton:disabled { background: #ddd; color: #888; "
            " border: 1px solid #bbb; }"
        )
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)

        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

    def populate(
        self,
        datasets: List[StrategyArtifactReference],
        splits: List[StrategyArtifactReference],
        minings: List[StrategyArtifactReference],
        pipelines: List[StrategyArtifactReference],
    ) -> None:
        """Popula as 4 listas com refs da biblioteca global."""
        mapping = {
            "datasets": datasets,
            "splits": splits,
            "minings": minings,
            "pipelines": pipelines,
        }
        for key, refs in mapping.items():
            lw = self._lists[key]
            lw.clear()
            for ref in refs:
                stem = Path(ref.path).stem
                item = QtWidgets.QListWidgetItem(stem)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, ref)
                lw.addItem(item)
        self._update_counts()

    def selected_refs(self) -> Dict[str, List[StrategyArtifactReference]]:
        """Retorna dict com a chave do tipo e lista de refs selecionadas."""
        result: Dict[str, List[StrategyArtifactReference]] = {}
        for key, lw in self._lists.items():
            picked = []
            for i in range(lw.count()):
                item = lw.item(i)
                if item.isSelected():
                    picked.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
            result[key] = picked
        return result

    def _update_counts(self) -> None:
        """Atualiza o título de cada aba com `(selecionados/total)` e
        habilita o botão Adicionar quando pelo menos um está selecionado."""
        total_selected = 0
        for idx, key in enumerate(self._KEYS):
            lw = self._lists[key]
            n_sel = len(lw.selectedItems())
            n_total = lw.count()
            total_selected += n_sel
            self.tabs.setTabText(
                idx, f"{self._LABELS[key]} ({n_sel}/{n_total})"
            )
        self.btn_ok.setText(
            f"Adicionar ({total_selected})" if total_selected
            else "Adicionar"
        )
        self.btn_ok.setEnabled(total_selected > 0)
