# interface/tabs/strategy_manager_v2/dialogs/add_resources_from_library/
# add_resources_from_library_view.py

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Dict, List

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)


class AddResourcesFromLibraryView(QtWidgets.QDialog):
    """
    Diálogo de adição em lote de recursos da biblioteca de configurações
    ao projeto atual.

    Mostra 4 abas — Datasets, Split Configs, Mining Configs, Pipeline
    Configs — cada uma com uma **árvore hierárquica** dos recursos da
    biblioteca, agrupados por subpasta. Multi-seleção via Ctrl/Shift.
    Confirmar copia todos os selecionados para as pastas do projeto via
    `LocalProjectArtifactStorage.copy_artifact_to_project`.

    Subpastas:
    O usuário pode organizar a biblioteca em subdiretórios (ex:
    `datasets/cfp_frontal/`, `datasets/cfp_lateral/`). Esta view detecta
    subpastas automaticamente a partir do path relativo de cada ref e
    renderiza uma árvore navegável. Pastas vazias são omitidas.

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

    # UserRole: payload do nó folha (StrategyArtifactReference).
    # Nós internos (pastas) carregam None — distinção semântica.
    _REF_ROLE = QtCore.Qt.ItemDataRole.UserRole

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adicionar Recursos ao Projeto")
        self.setMinimumSize(720, 520)
        self.setModal(True)
        self._trees: Dict[str, QtWidgets.QTreeWidget] = {}
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QtWidgets.QLabel(
            "Selecione os recursos da biblioteca para adicionar ao projeto. "
            "Use <b>Ctrl+clique</b> ou <b>Shift+clique</b> para multi-selecionar. "
            "Subpastas da biblioteca aparecem como nós navegáveis.",
            parent=self,
        )
        header.setWordWrap(True)
        apply_themed_stylesheet(
            header,
            lambda t: "color: " + t("TEXT") + "; font-size: 11px;",
        )
        layout.addWidget(header)

        self.tabs = QtWidgets.QTabWidget(parent=self)
        for key in self._KEYS:
            tree = QtWidgets.QTreeWidget(parent=self)
            tree.setHeaderLabels(["Recurso"])
            tree.setSelectionMode(
                QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection,
            )
            tree.setAlternatingRowColors(True)
            tree.setUniformRowHeights(True)
            tree.itemSelectionChanged.connect(self._update_counts)
            # Expandir tudo por padrão — caso típico (poucas subpastas)
            # fica imediatamente útil. Caso com muitas subpastas, o user
            # pode colapsar manualmente.
            self._trees[key] = tree
            self.tabs.addTab(tree, self._LABELS[key])
        layout.addWidget(self.tabs, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()

        self.btn_ok = QtWidgets.QPushButton("Adicionar", parent=self)
        apply_themed_stylesheet(
            self.btn_ok,
            lambda t: (
                "QPushButton { border: 1px solid " + t("PRIMARY") + ";"
                " border-radius: 4px;"
                " background: " + t("PRIMARY") + ";"
                " color: " + t("HIGHLIGHT_TEXT") + "; padding: 5px 14px; "
                " font-weight: bold; }"
                "QPushButton:hover { background: " + t("PRIMARY_STRONG") + "; }"
                "QPushButton:disabled { background: " + t("BORDER") + ";"
                " color: " + t("TEXT_MUTED") + "; "
                " border: 1px solid " + t("BORDER") + "; }"
            ),
        )
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)

        self.btn_cancel = QtWidgets.QPushButton("Cancelar", parent=self)
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
        """Popula as 4 árvores com refs da biblioteca global, agrupadas por subpasta."""
        mapping = {
            "datasets": datasets,
            "splits": splits,
            "minings": minings,
            "pipelines": pipelines,
        }
        for key, refs in mapping.items():
            tree = self._trees[key]
            tree.clear()
            self._populate_tree(tree, refs)
            tree.expandAll()
        self._update_counts()

    def _populate_tree(
        self,
        tree: QtWidgets.QTreeWidget,
        refs: List[StrategyArtifactReference],
    ) -> None:
        """Constrói a árvore: o primeiro componente do path é a pasta da
        categoria (ex: 'datasets/') e é descartado; os componentes
        intermediários viram nós internos; o último componente (com
        extensão) vira folha.

        Implementação: indexa nós internos por path acumulado pra
        merge correto quando múltiplos refs compartilham subpastas.
        """
        root = tree.invisibleRootItem()
        # Mapa subpath_acumulado → QTreeWidgetItem (pastas).
        folder_nodes: Dict[str, QtWidgets.QTreeWidgetItem] = {}

        # Ordena pra determinismo + ordem alfabética dentro de cada subpasta.
        # PurePosixPath garante que '/' funcione cross-platform em sorting.
        def _sort_key(ref: StrategyArtifactReference) -> tuple:
            p = PurePosixPath(ref.path.replace("\\", "/"))
            # Tuple de partes desconsiderando o primeiro (pasta da categoria).
            parts = p.parts[1:] if len(p.parts) > 1 else p.parts
            return parts

        for ref in sorted(refs, key=_sort_key):
            # path relativo à raiz da biblioteca: ex. "datasets/grupo_a/foo.parquet"
            p = PurePosixPath(ref.path.replace("\\", "/"))
            parts = list(p.parts)
            if len(parts) <= 1:
                # ref malformado — pula
                continue
            # Pula a primeira componente (pasta da categoria — datasets/, etc.)
            sub_parts = parts[1:]
            # Last is the file (with ext); the rest are subpasta nodes.
            file_part = sub_parts[-1]
            folder_parts = sub_parts[:-1]

            # Garante que cada nível intermediário tem um nó pai criado.
            parent_node = root
            accumulated = ""
            for folder_name in folder_parts:
                accumulated = (
                    f"{accumulated}/{folder_name}" if accumulated else folder_name
                )
                node = folder_nodes.get(accumulated)
                if node is None:
                    node = QtWidgets.QTreeWidgetItem(parent_node)
                    node.setText(0, folder_name)
                    node.setIcon(0, self._folder_icon())
                    # Pastas: não selecionáveis individualmente, mas
                    # expandíveis. Aceitar Ctrl+clique no nó-pasta seleciona
                    # apenas folhas descendentes via signal handler abaixo.
                    node.setFlags(
                        node.flags()
                        & ~QtCore.Qt.ItemFlag.ItemIsSelectable
                    )
                    folder_nodes[accumulated] = node
                parent_node = node

            # Folha = arquivo. Stem como display, ref no UserRole.
            leaf_text = Path(file_part).stem
            leaf = QtWidgets.QTreeWidgetItem(parent_node)
            leaf.setText(0, leaf_text)
            leaf.setIcon(0, self._file_icon())
            leaf.setData(0, self._REF_ROLE, ref)
            # Tooltip mostra o path relativo completo (útil quando há
            # múltiplos arquivos com mesmo stem em subpastas distintas).
            leaf.setToolTip(0, ref.path)

    @staticmethod
    def _folder_icon() -> QtGui.QIcon:
        return QtWidgets.QApplication.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_DirIcon
        )

    @staticmethod
    def _file_icon() -> QtGui.QIcon:
        return QtWidgets.QApplication.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        )

    def selected_refs(self) -> Dict[str, List[StrategyArtifactReference]]:
        """Retorna dict com a chave do tipo e lista de refs selecionadas.

        Como pastas não são selecionáveis (flags negam ItemIsSelectable),
        a iteração só pega folhas — i.e., itens com payload `ref` setado.
        """
        result: Dict[str, List[StrategyArtifactReference]] = {}
        for key, tree in self._trees.items():
            picked = []
            for item in tree.selectedItems():
                ref = item.data(0, self._REF_ROLE)
                if ref is not None:
                    picked.append(ref)
            result[key] = picked
        return result

    def _update_counts(self) -> None:
        """Atualiza o título de cada aba com `(selecionados/total)` e
        habilita o botão Adicionar quando pelo menos um está selecionado.

        Total = nº de folhas (arquivos), ignora pastas intermediárias.
        """
        total_selected = 0
        for idx, key in enumerate(self._KEYS):
            tree = self._trees[key]
            n_sel = sum(
                1 for item in tree.selectedItems()
                if item.data(0, self._REF_ROLE) is not None
            )
            n_total = self._count_leaves(tree)
            total_selected += n_sel
            self.tabs.setTabText(
                idx, f"{self._LABELS[key]} ({n_sel}/{n_total})"
            )
        self.btn_ok.setText(
            f"Adicionar ({total_selected})" if total_selected
            else "Adicionar"
        )
        self.btn_ok.setEnabled(total_selected > 0)

    @staticmethod
    def _count_leaves(tree: QtWidgets.QTreeWidget) -> int:
        """Conta nós folha (com ref payload) em qualquer nível da árvore."""
        count = 0
        stack = [tree.invisibleRootItem().child(i) for i in range(tree.topLevelItemCount())]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.data(0, AddResourcesFromLibraryView._REF_ROLE) is not None:
                count += 1
            for i in range(node.childCount()):
                stack.append(node.child(i))
        return count
