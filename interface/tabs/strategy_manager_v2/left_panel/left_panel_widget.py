# interface/tabs/strategy_manager_v2/left_panel/left_panel_widget.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

from .strategy_list_widget import StrategyListWidget
from .resource_tree_widget import ResourceTreeWidget
from .resource_info_widget import ResourceInfoWidget


class LeftPanelWidget(QtWidgets.QWidget):
    """
    Painel esquerdo da aba V2: QSplitter vertical com 2 seções.

    Seção 1 — StrategyListWidget: lista de estratégias do projeto
    Seção 2 — Recursos: árvore (ResourceTreeWidget) + detalhes tabulados
              (ResourceInfoWidget) num splitter interno

    Delega sinais para os widgets internos.
    """

    # Delegação de sinais das seções internas
    strategy_selected = QtCore.pyqtSignal(object)       # LRStrategyDefinition
    strategy_opened = QtCore.pyqtSignal(object)         # LRStrategyDefinition
    new_strategy_requested = QtCore.pyqtSignal()
    duplicate_requested = QtCore.pyqtSignal(object)
    rename_requested = QtCore.pyqtSignal(object)
    remove_requested = QtCore.pyqtSignal(object)
    run_strategy_requested = QtCore.pyqtSignal()
    save_as_template_requested = QtCore.pyqtSignal()
    comparison_requested = QtCore.pyqtSignal()
    resource_selected = QtCore.pyqtSignal(object)       # StrategyResource

    # Sinais de importação/remoção de recurso (repassados do ResourceTreeWidget)
    add_dataset_requested = QtCore.pyqtSignal()
    add_split_config_requested = QtCore.pyqtSignal()
    add_mining_config_requested = QtCore.pyqtSignal()
    add_pipeline_config_requested = QtCore.pyqtSignal()
    remove_resource_requested = QtCore.pyqtSignal(object)  # StrategyResource
    edit_dataset_in_manager_requested = QtCore.pyqtSignal(str)  # path absoluto

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(380)
        self._project_root: Optional[Path] = None
        self._definitions: List[LRStrategyDefinition] = []
        self._embedding_dimensions: Optional[int] = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Splitter principal: estratégias | recursos
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        self.strategy_list = StrategyListWidget()
        self.resource_tree = ResourceTreeWidget()
        self.resource_info = ResourceInfoWidget()

        # Estilo comum para os frames do painel esquerdo
        _frame_style = """
            QFrame#%(name)s {
                border: 1px solid #c0c8d0;
                border-radius: 5px;
                background: white;
            }
        """

        # Frame de estratégias
        strategy_frame = QtWidgets.QFrame(parent=self)
        strategy_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        strategy_frame.setObjectName("strategyFrame")
        strategy_frame.setStyleSheet(_frame_style % {"name": "strategyFrame"})

        strategy_frame_layout = QtWidgets.QVBoxLayout(strategy_frame)
        strategy_frame_layout.setContentsMargins(0, 0, 0, 0)
        strategy_frame_layout.setSpacing(0)
        strategy_frame_layout.addWidget(self.strategy_list)

        # Frame de recursos: moldura visual agrupando árvore + info tabulada
        resource_frame = QtWidgets.QFrame(parent=self)
        resource_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        resource_frame.setObjectName("resourceFrame")
        resource_frame.setStyleSheet(_frame_style % {"name": "resourceFrame"})

        resource_layout = QtWidgets.QVBoxLayout(resource_frame)
        resource_layout.setContentsMargins(0, 0, 0, 0)
        resource_layout.setSpacing(0)

        resource_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        resource_splitter.setChildrenCollapsible(False)
        resource_splitter.setHandleWidth(1)
        resource_splitter.setStyleSheet(
            "QSplitter::handle { background: #d0d8e0; }"
        )
        resource_splitter.addWidget(self.resource_tree)
        resource_splitter.addWidget(self.resource_info)
        # Árvore e abas com proporções equilibradas
        resource_splitter.setSizes([220, 260])
        resource_splitter.setStretchFactor(0, 1)
        resource_splitter.setStretchFactor(1, 1)

        resource_layout.addWidget(resource_splitter, 1)

        # Barra de ações (botões de recurso) como rodapé do frame
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(4, 3, 4, 3)
        btn_row.setSpacing(4)
        btn_row.addWidget(self.resource_tree.btn_add)
        btn_row.addWidget(self.resource_tree.btn_remove)
        resource_layout.addLayout(btn_row)

        splitter.addWidget(strategy_frame)
        splitter.addWidget(resource_frame)

        # Proporção inicial: 35% estratégias / 65% recursos
        splitter.setSizes([200, 370])

        layout.addWidget(splitter)

        # Conexões internas
        self.strategy_list.strategy_selected.connect(self.strategy_selected)
        self.strategy_list.strategy_opened.connect(self.strategy_opened)
        self.strategy_list.new_strategy_requested.connect(self.new_strategy_requested)
        self.strategy_list.duplicate_requested.connect(self.duplicate_requested)
        self.strategy_list.rename_requested.connect(self.rename_requested)
        self.strategy_list.remove_requested.connect(self.remove_requested)
        self.strategy_list.run_strategy_requested.connect(self.run_strategy_requested)
        self.strategy_list.save_as_template_requested.connect(self.save_as_template_requested)
        self.strategy_list.comparison_requested.connect(self.comparison_requested)

        self.resource_tree.resource_selected.connect(self._on_resource_selected)
        self.resource_tree.resource_deselected.connect(self.resource_info.clear)
        self.resource_tree.add_dataset_requested.connect(self.add_dataset_requested)
        self.resource_tree.add_split_config_requested.connect(self.add_split_config_requested)
        self.resource_tree.add_mining_config_requested.connect(self.add_mining_config_requested)
        self.resource_tree.add_pipeline_config_requested.connect(self.add_pipeline_config_requested)
        self.resource_tree.remove_resource_requested.connect(self.remove_resource_requested)
        self.resource_info.edit_dataset_in_manager_requested.connect(
            self.edit_dataset_in_manager_requested
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def populate_strategies(
        self,
        definitions: List[LRStrategyDefinition],
        selected_strategy_id: Optional[str] = None,
    ) -> None:
        self._definitions = list(definitions)
        self.strategy_list.populate(
            definitions, selected_strategy_id=selected_strategy_id
        )

    def select_strategy_by_id(self, strategy_id: Optional[str]) -> None:
        """Marca visualmente a estratégia indicada como selecionada."""
        self.strategy_list.select_strategy_by_id(strategy_id)

    def populate_resources(
        self,
        resources: List[StrategyResource],
        resource_to_strategy: Optional[dict] = None,
    ) -> None:
        self.resource_tree.populate(resources, resource_to_strategy=resource_to_strategy)

    def set_project_root(self, root: Optional[Path]) -> None:
        self._project_root = root

    def set_embedding_dimensions(self, dimensions: Optional[int]) -> None:
        """Define as dimensões do embedding da estratégia selecionada."""
        self._embedding_dimensions = dimensions

    def set_add_resource_enabled(self, enabled: bool) -> None:
        """Habilita/desabilita o botão de adicionar recurso na árvore."""
        self.resource_tree.btn_add.setEnabled(enabled)

    def set_strategy_actions_enabled(self, enabled: bool) -> None:
        """Habilita/desabilita botões Executar e Salvar como Template."""
        self.strategy_list.set_strategy_actions_enabled(enabled)

    def clear(self) -> None:
        self._definitions.clear()
        self._project_root = None
        self._embedding_dimensions = None
        self.strategy_list.clear()
        self.resource_tree.clear()
        self.resource_info.clear()
        self.resource_tree.btn_add.setEnabled(False)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_resource_selected(self, resource: StrategyResource) -> None:
        self.resource_info.show_resource(
            resource=resource,
            project_root=self._project_root,
            definitions=self._definitions,
            embedding_dimensions=self._embedding_dimensions,
        )
        self.resource_selected.emit(resource)
