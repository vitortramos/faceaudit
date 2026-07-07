# interface/tabs/strategy_manager_v2/left_panel/resource_tree_widget.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.strategy_resource import StrategyResource


# Custom item role pra guardar o nome da estratégia no top-level item da
# árvore de derivados. Usado pra preservar o estado de colapso/expansão
# através de rebuilds (UserRole+0 é reservado pro StrategyResource das
# folhas; o lookup `_find_item_for_resource` quebraria se mexer nele).
_STRATEGY_NAME_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


# Rótulos legíveis por tipo. DATASET cobre tanto primário ("Datasets")
# quanto derivado ("Datasets") — a aba já cinde a categoria, então o
# rótulo pode ser o mesmo nas duas abas.
_TYPE_LABELS: Dict[StrategyResourceType, str] = {
    StrategyResourceType.DATASET: "Datasets",
    StrategyResourceType.SIMILARITY_SCORES: "Scores de Similaridade",
    StrategyResourceType.CALIBRATED_SCORES: "Scores Calibrados",
    StrategyResourceType.VALIDATION_RESULT: "Resultados de Validação",
    StrategyResourceType.CASEWORK_VISUALIZATION: "Visualizações de Caso",
    StrategyResourceType.TRAINED_PIPELINE: "Pipelines Treinados",
    StrategyResourceType.TRAINED_SIMILARITY_MODEL: "Modelos de Similaridade",
    StrategyResourceType.TRAINED_CALIBRATION_MODEL: "Modelos de Calibração",
    StrategyResourceType.TRAJECTORY_DATA: "Trajetórias 3D",
    StrategyResourceType.PIPELINE_CONFIG: "Configs de Pipeline",
    StrategyResourceType.SPLIT_CONFIG: "Configs de Split",
    StrategyResourceType.MINING_CONFIG: "Configs de Mineração",
    StrategyResourceType.MERGE_CONFIG: "Configs de Merge",
    StrategyResourceType.SIMILARITY_MODEL_CONFIG: "Configs de Similaridade",
    StrategyResourceType.CALIBRATION_MODEL_CONFIG: "Configs de Calibração",
    StrategyResourceType.HUBNESS_REPORT: "Relatórios de Hubness",
    StrategyResourceType.HUBNESS_ABLATION_RESULT: "Ablações de Hubness",
    StrategyResourceType.DATASET_STATS: "Estatísticas de Dataset",
    StrategyResourceType.ARTIFACT: "Artefatos",
}


def _tree_qss(t) -> str:
    return (
        "QTreeWidget {"
        "  font-size: 11px;"
        "  border: none;"
        "  background: " + t("SURFACE") + ";"
        "}"
        "QTreeWidget::item:selected {"
        "  background: " + t("PRIMARY_BG") + ";"
        "  color: inherit;"
        "}"
        "QTreeWidget::item:selected:!active {"
        "  background: " + t("PRIMARY_BG") + ";"
        "  color: inherit;"
        "}"
        "QTreeWidget::item:hover:!selected {"
        "  background: " + t("PRIMARY_BG") + ";"
        "}"
        "QHeaderView::section {"
        "  font-weight: bold;"
        "  font-size: 12px;"
        "  color: " + t("PRIMARY_STRONG") + ";"
        "  background: " + t("SURFACE_ALT") + ";"
        "  border: none;"
        "  border-bottom: 1px solid " + t("BORDER") + ";"
        "  padding: 5px 8px;"
        "}"
    )


class ResourceTreeWidget(QtWidgets.QWidget):
    """
    Painel de recursos do projeto com 2 abas (Primários / Derivados).

    - Aba **Primários**: recursos importados pelo usuário via
      "Adicionar Recurso" — datasets, pipeline/split/mining configs.
      Lista plana (compartilhados entre estratégias).
    - Aba **Derivados**: recursos produzidos por execução de instrução —
      datasets derivados, scores, modelos treinados, validation results,
      casework visualizations, hubness reports/ablations, trajetórias.
      Sub-agrupados por estratégia.

    `category` (PRIMARY/DERIVED) no `StrategyResource` decide em qual aba
    o recurso aparece — o eixo "tipo" (DATASET, TRAINED_PIPELINE, …) é
    secundário e fica como agrupamento dentro de cada aba.

    O botão "Adicionar Recurso" só aparece quando a aba Primários está
    ativa — não há como "adicionar" um recurso derivado, eles só nascem
    como output de uma instrução.

    Sinais (preservados — controller V2 já está conectado a eles):
        resource_selected(StrategyResource)
        resource_deselected()
        add_dataset_requested
        add_split_config_requested
        add_mining_config_requested
        add_pipeline_config_requested
        remove_resource_requested(StrategyResource)
    """

    resource_selected = QtCore.pyqtSignal(object)
    resource_deselected = QtCore.pyqtSignal()
    add_dataset_requested = QtCore.pyqtSignal()
    add_split_config_requested = QtCore.pyqtSignal()
    add_mining_config_requested = QtCore.pyqtSignal()
    add_pipeline_config_requested = QtCore.pyqtSignal()
    remove_resource_requested = QtCore.pyqtSignal(object)
    # Fase 4 — renomeação de `name` (o id técnico nunca muda).
    # Args: (StrategyResource, novo_name).
    rename_requested = QtCore.pyqtSignal(object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_resource: Optional[StrategyResource] = None
        # Estado de colapso por nome de estratégia, preservado entre
        # rebuilds da árvore de derivados.
        self._collapsed_strategy_names: set = set()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget(parent=self)
        apply_themed_stylesheet(
            self._tabs,
            lambda t: (
                "QTabBar::tab { padding: 6px 14px; font-size: 11px; }"
                "QTabBar::tab:selected { background: " + t("SURFACE") + "; font-weight: bold; }"
            ),
        )
        # NÃO conectar `currentChanged` aqui — `addTab` dispara o sinal
        # quando o índice muda de -1 pra 0, e nesse ponto `btn_add` ainda
        # não foi criado. A conexão é feita no fim do `_init_ui`, depois
        # que todos os widgets que o slot referencia já existem.
        self.tree_primary = self._build_tree()
        self.tree_derived = self._build_tree()
        self._tabs.addTab(self.tree_primary, "Primários")
        self._tabs.addTab(self.tree_derived, "Derivados")
        layout.addWidget(self._tabs)

        # ── Botões "Adicionar"/"Remover" ──
        # Não vão ao layout — posicionados externamente pelo left_panel
        # (compat com a estrutura existente). `btn_add` é escondido
        # quando o usuário troca pra aba Derivados.
        self.btn_add = QtWidgets.QToolButton(parent=self)
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
        apply_themed_stylesheet(
            self.btn_add,
            lambda t: (
                "QToolButton {"
                "  border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                "  background: " + t("SURFACE") + "; padding: 3px 8px; font-size: 11px;"
                "}"
                "QToolButton:hover { background: " + t("PRIMARY_BG") + "; border-color: " + t("PRIMARY") + "; }"
                "QToolButton:disabled { background: " + t("SURFACE_ALT") + "; color: " + t("TEXT_MUTED") + "; border-color: " + t("BORDER") + "; }"
                "QToolButton::menu-button {"
                "  border-left: 1px solid " + t("BORDER") + "; width: 16px; border-radius: 0 4px 4px 0;"
                "}"
            ),
        )
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
        apply_themed_stylesheet(
            self.btn_remove,
            lambda t: (
                "QPushButton {"
                "  border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                "  background: " + t("SURFACE") + "; padding: 3px 8px; font-size: 11px;"
                "}"
                "QPushButton:hover { background: " + t("DANGER_BG") + "; border-color: " + t("DANGER") + "; color: " + t("DANGER") + "; }"
                "QPushButton:disabled { background: " + t("SURFACE_ALT") + "; color: " + t("TEXT_MUTED") + "; border-color: " + t("BORDER") + "; }"
            ),
        )
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        # Conexão tardia — `btn_add` já existe nesse ponto, então o slot
        # pode acessá-lo sem AttributeError quando o sinal disparar.
        self._tabs.currentChanged.connect(self._on_tab_changed)
        # Estado inicial: aba Primários ativa → btn_add visível. O sinal
        # `currentChanged` não dispara em runtime quando o widget aparece
        # (já está no índice 0); sincroniza explicitamente.
        self._on_tab_changed(self._tabs.currentIndex())

    def _build_tree(self) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget(parent=self)
        tree.setHeaderLabels(["Recursos"])
        tree.setAlternatingRowColors(True)
        apply_themed_stylesheet(tree, _tree_qss)
        tree.itemClicked.connect(self._on_item_clicked)
        # Context menu pra renomear name (Fase 4 da migração id/name).
        tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        tree.customContextMenuRequested.connect(
            lambda pos, t=tree: self._on_tree_context_menu(t, pos)
        )
        return tree

    def _on_tree_context_menu(
        self, tree: QtWidgets.QTreeWidget, pos: QtCore.QPoint,
    ) -> None:
        """Menu de contexto sobre nó de recurso — opção Renomear name.

        Só ativa em folhas (StrategyResource no UserRole); nós internos
        (grupo por estratégia, agrupador por tipo) são ignorados.
        """
        from faceaudit.models.strategy.strategy_resource import StrategyResource
        item = tree.itemAt(pos)
        if item is None:
            return
        res = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(res, StrategyResource):
            return
        menu = QtWidgets.QMenu(self)
        rename_action = menu.addAction("✎  Renomear…")
        rename_action.triggered.connect(
            lambda _checked, r=res: self._prompt_rename(r)
        )
        menu.exec(tree.viewport().mapToGlobal(pos))

    def _prompt_rename(self, res) -> None:
        """Abre prompt pra editar `res.name` e emite `rename_requested`.

        O id técnico (DS-xxxxxxxx) NÃO muda — só o display name. Nenhuma
        ref interna nas estratégias quebra.
        """
        current = res.name or res.resource_id or ""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Renomear recurso",
            f"Novo nome para o recurso:\n\n"
            f"ID técnico (não muda): {res.id or '(legado)'}\n"
            f"Nome atual: {current}",
            QtWidgets.QLineEdit.EchoMode.Normal,
            current,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == current:
            return
        self.rename_requested.emit(res, new_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def populate(
        self,
        resources: List[StrategyResource],
        resource_to_strategy: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Distribui os recursos entre as 2 abas pela `category`:
          - Primários: lista plana `<tipo> / <recurso>`.
          - Derivados: aninhamento `<estratégia> / <tipo> / <recurso>`
            (quando `resource_to_strategy` é fornecido).

        **Filtro na aba Derivados**: placeholders sem `artifact_ref` E sem
        `payload` ficam ocultos. São declarações de output que ainda não
        produziram nada — esses só fazem sentido na coluna de configuração
        do editor (3ª coluna), não na árvore de recursos do projeto, que
        deve refletir "o que existe". Quando o usuário executa a instrução,
        o placeholder ganha artifact_ref e reaparece automaticamente.
        """
        # Preserva o estado de colapso por estratégia ANTES do rebuild.
        # Sem isso, qualquer execução que dispare repopulação volta todos
        # os grupos de estratégia pra expandido, atrapalhando o usuário
        # que tinha colapsado as outras estratégias pra ver só a atual.
        collapsed_strategies: set = set()
        for i in range(self.tree_derived.topLevelItemCount()):
            top = self.tree_derived.topLevelItem(i)
            if top is None or top.isExpanded():
                continue
            name = top.data(0, _STRATEGY_NAME_ROLE)
            if name:
                collapsed_strategies.add(name)
        self._collapsed_strategy_names = collapsed_strategies

        self.tree_primary.clear()
        self.tree_derived.clear()
        resource_to_strategy = resource_to_strategy or {}

        primary = [r for r in resources if r.category == ResourceCategory.PRIMARY]
        derived_all = [r for r in resources if r.category == ResourceCategory.DERIVED]
        derived = [
            r for r in derived_all
            if r.has_artifact or r.has_payload
        ]

        self._populate_tree(
            self.tree_primary, primary,
            resource_to_strategy={},  # primários não agrupam por estratégia
            group_by_strategy=False,
        )
        self._populate_tree(
            self.tree_derived, derived,
            resource_to_strategy=resource_to_strategy,
            group_by_strategy=True,
        )

        # Atualiza contadores nas abas pra o usuário ver quantidade de cada lado.
        # Derivados conta só os visíveis (com artefato/payload), não placeholders
        # pendentes — alinhado com o filtro acima.
        self._tabs.setTabText(0, f"Primários ({len(primary)})")
        self._tabs.setTabText(1, f"Derivados ({len(derived)})")

        # Se havia um recurso selecionado antes do rebuild E ele ainda
        # existe na lista nova (mesma instância — `_refresh_left_panel`
        # passa as instâncias mutadas in-place), re-emite o signal pra
        # que o ResourceInfoWidget no left panel re-renderize com os
        # dados atualizados. Sem isso, após re-executar uma instrução
        # o painel de propriedades continuava mostrando hash/tamanho/
        # métricas antigos até o usuário clicar de novo no recurso.
        if self._current_resource is not None:
            still_present = (
                self._current_resource in primary
                or self._current_resource in derived
            )
            if still_present:
                # Re-localiza o QTreeWidgetItem correspondente (foi
                # recriado durante o rebuild) e re-aplica `setCurrentItem`,
                # preservando o highlight azul que o usuário tinha antes.
                # Decide a tree certa pela categoria do resource.
                target_tree = (
                    self.tree_primary
                    if self._current_resource in primary
                    else self.tree_derived
                )
                item = self._find_item_for_resource(
                    target_tree, self._current_resource,
                )
                if item is not None:
                    # Garante a aba certa visível (Primários ou Derivados).
                    self._tabs.setCurrentWidget(target_tree)
                    target_tree.setCurrentItem(item)
                self.resource_selected.emit(self._current_resource)
            else:
                self._current_resource = None
                self.btn_remove.setEnabled(False)

    @staticmethod
    def _find_item_for_resource(
        tree: QtWidgets.QTreeWidget,
        resource: StrategyResource,
    ) -> Optional[QtWidgets.QTreeWidgetItem]:
        """Busca recursiva pelo QTreeWidgetItem cujo UserRole bate com
        `resource` (identity-equal). Retorna None se não achar."""
        target_id = id(resource)

        def _walk(parent: QtWidgets.QTreeWidgetItem) -> Optional[QtWidgets.QTreeWidgetItem]:
            for i in range(parent.childCount()):
                child = parent.child(i)
                data = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                if data is not None and id(data) == target_id:
                    return child
                found = _walk(child)
                if found is not None:
                    return found
            return None

        return _walk(tree.invisibleRootItem())

    def clear(self) -> None:
        self.tree_primary.clear()
        self.tree_derived.clear()
        self._current_resource = None
        self.btn_remove.setEnabled(False)
        self._tabs.setTabText(0, "Primários")
        self._tabs.setTabText(1, "Derivados")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _populate_tree(
        self,
        tree: QtWidgets.QTreeWidget,
        resources: List[StrategyResource],
        resource_to_strategy: Dict[int, str],
        group_by_strategy: bool,
    ) -> None:
        if group_by_strategy and resource_to_strategy:
            # Derivados: estrutura `<estratégia> / <tipo> / <recurso>` —
            # cada estratégia é seu próprio "container" e a árvore se
            # parece com o que o usuário vê na lista de estratégias.
            self._populate_tree_by_strategy_first(
                tree, resources, resource_to_strategy,
            )
            return

        # Primários: lista plana `<tipo> / <recurso>` — recursos
        # compartilhados entre estratégias, sem nível intermediário.
        grouped: Dict[StrategyResourceType, List[StrategyResource]] = {}
        for res in resources:
            grouped.setdefault(res.resource_type, []).append(res)

        for res_type, label in _TYPE_LABELS.items():
            group_resources = grouped.get(res_type, [])
            if not group_resources:
                continue

            root_item = QtWidgets.QTreeWidgetItem(
                [f"{label} ({len(group_resources)})"]
            )
            root_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            tree.addTopLevelItem(root_item)

            for res in group_resources:
                self._add_resource_child(root_item, res)

            root_item.setExpanded(True)

    def _populate_tree_by_strategy_first(
        self,
        tree: QtWidgets.QTreeWidget,
        resources: List[StrategyResource],
        resource_to_strategy: Dict[int, str],
    ) -> None:
        """Layout `<estratégia> / <tipo> / <recurso>` pra a aba Derivados.

        Dois passos: primeiro indexa por estratégia → tipo → list[res];
        depois renderiza, mantendo a ordem dos tipos consistente com
        `_TYPE_LABELS` pra que tipos apareçam na mesma ordem dentro de
        cada estratégia.

        Lookup do owner via `id(res)` (não por `resource_id`-string) —
        duplicação de estratégia via deep copy gera instâncias distintas
        com o mesmo `resource_id`, e cada uma pertence a uma estratégia
        diferente. A identidade Python (`id`) é o único discriminador
        garantido pós-duplicação.
        """
        by_strategy: Dict[str, Dict[StrategyResourceType, List[StrategyResource]]] = {}
        for res in resources:
            strategy_name = resource_to_strategy.get(
                id(res), "(sem estratégia)",
            )
            by_strategy.setdefault(strategy_name, {}).setdefault(
                res.resource_type, [],
            ).append(res)

        for strategy_name in sorted(by_strategy.keys()):
            types_dict = by_strategy[strategy_name]
            total = sum(len(v) for v in types_dict.values())
            strategy_item = QtWidgets.QTreeWidgetItem(
                [f"📋 {strategy_name} ({total})"]
            )
            strategy_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
            # Nome da estratégia gravado num role separado pra que o
            # rebuild possa restaurar o estado de colapso.
            strategy_item.setData(0, _STRATEGY_NAME_ROLE, strategy_name)
            tree.addTopLevelItem(strategy_item)

            # Itera tipos na ordem de `_TYPE_LABELS` pra consistência visual
            # entre estratégias (ex: Datasets sempre antes de Scores etc.).
            for res_type, label in _TYPE_LABELS.items():
                type_resources = types_dict.get(res_type, [])
                if not type_resources:
                    continue
                type_item = QtWidgets.QTreeWidgetItem(
                    [f"{label} ({len(type_resources)})"]
                )
                type_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, None)
                strategy_item.addChild(type_item)
                for res in type_resources:
                    self._add_resource_child(type_item, res)
                type_item.setExpanded(True)

            # Restaura estado de colapso (default: expandido).
            strategy_item.setExpanded(
                strategy_name not in self._collapsed_strategy_names
            )

    def _add_resource_child(
        self,
        parent_item: QtWidgets.QTreeWidgetItem,
        res: StrategyResource,
    ) -> None:
        path_str = res.artifact_ref.path if res.artifact_ref else "(sem arquivo)"
        display_name = res.name if res.name else res.resource_id
        # Display padrão: `nome (ID)`. O ID (DS-xxxxxxxx) sempre acompanha
        # o nome — garante distinguibilidade quando há recursos com mesmo
        # nome em pastas diferentes. Pré-Fase 2, id era None — fallback
        # gracioso pra só name pra recursos sem id.
        if res.id:
            display_label = f"{display_name}  ({res.id})"
        else:
            display_label = display_name
        child = QtWidgets.QTreeWidgetItem([display_label])
        child.setData(0, QtCore.Qt.ItemDataRole.UserRole, res)
        child.setToolTip(
            0,
            f"ID técnico: {res.id or '(sem id)'}\n"
            f"Resource ID (legado): {res.resource_id}\n"
            f"Nome: {res.name}\n"
            f"Path: {path_str}",
        )
        parent_item.addChild(child)

    def _on_item_clicked(self, item: QtWidgets.QTreeWidgetItem, _col: int) -> None:
        # Limpa seleção na OUTRA árvore — Qt não desincroniza automaticamente
        # quando o usuário clica numa árvore enquanto a outra tem item
        # selecionado, e isso causa ambiguidade no `_current_resource`.
        active_tree = self.sender()
        for tree in (self.tree_primary, self.tree_derived):
            if tree is not active_tree:
                tree.clearSelection()

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

    def _on_tab_changed(self, index: int) -> None:
        """Esconde btn_add quando o usuário troca pra aba Derivados —
        não há como adicionar manualmente um recurso derivado."""
        # Defensivo: o sinal pode disparar durante o setup do QTabWidget
        # (addTab muda o índice de -1 pra 0 antes de `btn_add` existir).
        btn_add = getattr(self, "btn_add", None)
        if btn_add is None:
            return
        btn_add.setVisible(index == 0)
