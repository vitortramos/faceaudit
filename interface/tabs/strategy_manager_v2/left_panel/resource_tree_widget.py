# interface/tabs/strategy_manager_v2/left_panel/resource_tree_widget.py

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.strategy_resource import StrategyResource


_TYPE_LABELS: Dict[StrategyResourceType, str] = {
    StrategyResourceType.DATASET: "Datasets",
    StrategyResourceType.INTERMEDIATE_DATASET: "Datasets Intermediários",
    StrategyResourceType.SIMILARITY_SCORES: "Scores de Similaridade",
    StrategyResourceType.CALIBRATED_SCORES: "Scores Calibrados",
    StrategyResourceType.VALIDATION_RESULT: "Resultados de Validação",
    StrategyResourceType.TRAINED_PIPELINE: "Pipelines Treinados",
    StrategyResourceType.TRAINED_SIMILARITY_MODEL: "Modelos de Similaridade",
    StrategyResourceType.TRAINED_CALIBRATION_MODEL: "Modelos de Calibração",
    StrategyResourceType.TRAJECTORY_DATA: "Trajetórias 3D",
    StrategyResourceType.PIPELINE_CONFIG: "Configs de Pipeline",
    StrategyResourceType.SPLIT_CONFIG: "Configs de Split",
    StrategyResourceType.MERGE_CONFIG: "Configs de Merge",
    StrategyResourceType.SIMILARITY_MODEL_CONFIG: "Configs de Similaridade",
    StrategyResourceType.CALIBRATION_MODEL_CONFIG: "Configs de Calibração",
    StrategyResourceType.DATASET_STATS: "Estatísticas de Dataset",
    StrategyResourceType.ARTIFACT: "Artefatos",
}


# Tipos de recurso "produzidos" por instruções de estratégia — exibidos com
# sub-agrupamento por estratégia na árvore. Recursos de input (DATASET,
# *_CONFIG) ficam como lista plana porque podem ser compartilhados entre
# múltiplas estratégias.
_PRODUCED_TYPES = {
    StrategyResourceType.INTERMEDIATE_DATASET,
    StrategyResourceType.SIMILARITY_SCORES,
    StrategyResourceType.CALIBRATED_SCORES,
    StrategyResourceType.TRAINED_PIPELINE,
    StrategyResourceType.TRAINED_SIMILARITY_MODEL,
    StrategyResourceType.TRAINED_CALIBRATION_MODEL,
    StrategyResourceType.TRAJECTORY_DATA,
    StrategyResourceType.VALIDATION_RESULT,
}


class ResourceTreeWidget(QtWidgets.QWidget):
    """
    Árvore de recursos do projeto, agrupados por StrategyResourceType.

    Sinais:
        resource_selected(StrategyResource)
        add_dataset_requested
        add_split_config_requested
        add_mining_config_requested
        add_pipeline_config_requested
    """

    resource_selected = QtCore.pyqtSignal(object)
    resource_deselected = QtCore.pyqtSignal()
    add_dataset_requested = QtCore.pyqtSignal()
    add_split_config_requested = QtCore.pyqtSignal()
    add_mining_config_requested = QtCore.pyqtSignal()
    add_pipeline_config_requested = QtCore.pyqtSignal()
    remove_resource_requested = QtCore.pyqtSignal(object)  # StrategyResource

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_resource = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Recursos do Projeto"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                font-size: 11px;
                border: none;
                background: white;
            }
            QHeaderView::section {
                font-weight: bold;
                font-size: 12px;
                color: #2c3e50;
                background: #eef2f7;
                border: none;
                border-bottom: 1px solid #c0c8d0;
                padding: 5px 8px;
            }
        """)
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

        # Botão "Adicionar Recurso" com dropdown
        self.btn_add = QtWidgets.QToolButton()
        self.btn_add.setText("Adicionar Recurso")
        self.btn_add.setToolTip("Importar recurso para o projeto")
        self.btn_add.setEnabled(False)
        self.btn_add.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.btn_add.setFixedHeight(24)
        self.btn_add.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_add.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        self.btn_add.setStyleSheet("""
            QToolButton {
                border: 1px solid #bbb; border-radius: 4px;
                background: white; padding: 3px 8px; font-size: 11px;
            }
            QToolButton:hover { background: #e8f0fe; border-color: #4a90d9; }
            QToolButton:disabled { background: #f0f0f0; color: #aaa; border-color: #ddd; }
            QToolButton::menu-button {
                border-left: 1px solid #bbb; width: 16px; border-radius: 0 4px 4px 0;
            }
        """)
        menu = QtWidgets.QMenu(self.btn_add)
        menu.addAction("📦  Dataset  (.parquet)", self.add_dataset_requested)
        menu.addAction("✂   Split Config  (.json)", self.add_split_config_requested)
        menu.addAction("⛏   Mining Config  (.json)", self.add_mining_config_requested)
        menu.addAction("🔗  Pipeline Config  (.json)", self.add_pipeline_config_requested)
        self.btn_add.setMenu(menu)
        self.btn_add.clicked.connect(self.btn_add.showMenu)

        self.btn_remove = QtWidgets.QPushButton("Remover Recurso")
        self.btn_remove.setToolTip("Remove o arquivo do recurso selecionado do projeto")
        self.btn_remove.setFixedHeight(24)
        self.btn_remove.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.btn_remove.setEnabled(False)
        self.btn_remove.setStyleSheet("""
            QPushButton {
                border: 1px solid #bbb; border-radius: 4px;
                background: white; padding: 3px 8px; font-size: 11px;
            }
            QPushButton:hover { background: #fdecea; border-color: #e53935; color: #c62828; }
            QPushButton:disabled { background: #f0f0f0; color: #aaa; border-color: #ddd; }
        """)
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        # Botões NÃO são adicionados ao layout aqui — são posicionados
        # externamente pelo left_panel_widget como rodapé do frame de recursos.

    # ── Public API ────────────────────────────────────────────────────────────

    def populate(
        self,
        resources: List[StrategyResource],
        resource_to_strategy: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Repopula a árvore agrupando recursos por tipo.

        Para tipos em `_PRODUCED_TYPES` e quando `resource_to_strategy` é
        fornecido, cria um nível intermediário de sub-agrupamento por
        estratégia (ex: `Datasets Intermediários (5) / ▼ PLDA / item1`).
        """
        self.tree.clear()
        resource_to_strategy = resource_to_strategy or {}

        grouped: Dict[StrategyResourceType, List[StrategyResource]] = {}
        for res in resources:
            grouped.setdefault(res.resource_type, []).append(res)

        for res_type, label in _TYPE_LABELS.items():
            group_resources = grouped.get(res_type, [])
            if not group_resources:
                continue

            root_item = QtWidgets.QTreeWidgetItem([f"{label} ({len(group_resources)})"])
            root_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            self.tree.addTopLevelItem(root_item)

            if res_type in _PRODUCED_TYPES and resource_to_strategy:
                self._add_grouped_by_strategy(
                    parent_item=root_item,
                    resources=group_resources,
                    resource_to_strategy=resource_to_strategy,
                )
            else:
                for res in group_resources:
                    self._add_resource_child(root_item, res)

            root_item.setExpanded(True)

    def _add_grouped_by_strategy(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        resources: List[StrategyResource],
        resource_to_strategy: Dict[str, str],
    ) -> None:
        """
        Adiciona um nível intermediário 'Estratégia: <nome>' agrupando os
        recursos pela estratégia que os produziu.
        """
        by_strategy: Dict[str, List[StrategyResource]] = {}
        for res in resources:
            strategy_name = resource_to_strategy.get(res.resource_id, "(sem estratégia)")
            by_strategy.setdefault(strategy_name, []).append(res)

        for strategy_name in sorted(by_strategy.keys()):
            strategy_resources = by_strategy[strategy_name]
            strategy_item = QtWidgets.QTreeWidgetItem(
                [f"📋 {strategy_name} ({len(strategy_resources)})"]
            )
            strategy_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            parent_item.addChild(strategy_item)

            for res in strategy_resources:
                self._add_resource_child(strategy_item, res)

            strategy_item.setExpanded(True)

    def _add_resource_child(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        res: StrategyResource,
    ) -> None:
        """Adiciona um item folha representando um recurso à árvore."""
        path_str = res.artifact_ref.path if res.artifact_ref else "(sem arquivo)"
        display_name = res.name if res.name else res.resource_id
        child = QtWidgets.QTreeWidgetItem([display_name])
        child.setData(0, QtCore.Qt.ItemDataRole.UserRole, res)
        child.setToolTip(0, f"ID: {res.resource_id}\nPath: {path_str}")
        parent_item.addChild(child)

    def clear(self) -> None:
        self.tree.clear()
        self._current_resource = None
        self.btn_remove.setEnabled(False)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        res = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if res is not None:
            self._current_resource = res
            self.btn_remove.setEnabled(True)
            self.resource_selected.emit(res)
        else:
            # Clique no grupo — limpa seleção
            self._current_resource = None
            self.btn_remove.setEnabled(False)
            self.resource_deselected.emit()

    def _on_remove_clicked(self) -> None:
        if self._current_resource is not None:
            self.remove_resource_requested.emit(self._current_resource)
