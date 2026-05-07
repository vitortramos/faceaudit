"""View do diálogo de Auditoria & Sanitização do dataset.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Auditoria Forense — meu_dataset                                   │
  ├─────────────────────┬───────────────────────────────────────────┤
  │ ▼ Integridade Física│ [emoji] Audit label                        │
  │   🔴 47 Paths órf...│   summary text                            │
  │   🔴 12 Hash conf...│ ┌─────────────────────────────────────────┐│
  │   🔴 3 Sem embe...  │ │ Tabela de issues (path, descrição, ...)  ││
  │ ▼ Lógica Rótulos    │ │                                          ││
  │   🟠 18 Singletons  │ └─────────────────────────────────────────┘│
  │ ▼ Qualidade         │ [✓ Selecionar tudo] [Excluir selecionadas]│
  │   ⚪ 5 sem OFIQ     │                                            │
  │   🟠 8 FIQA<30      │                                            │
  ├─────────────────────┴───────────────────────────────────────────┤
  │ [🔍 Mineração Pesada] [🔄 Reanalisar] [Fechar]                   │
  └──────────────────────────────────────────────────────────────────┘

Toda interação real (filtrar, executar ações) vive no controller. A view
só renderiza estado e emite sinais semânticos.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity


_FINDING_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class DatasetAnalysisView(QtWidgets.QDialog):
    """View do diálogo de auditoria. Stateless o quanto possível —
    o controller aplica as ações."""

    # Sinais semânticos pro controller
    finding_selected = QtCore.pyqtSignal(object)              # AuditFinding
    delete_issues_requested = QtCore.pyqtSignal(list)         # List[str] de image_paths
    rerun_requested = QtCore.pyqtSignal()
    run_heavy_mining_requested = QtCore.pyqtSignal()

    def __init__(self, dataset_name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Auditoria Forense — {dataset_name}")
        self.resize(1100, 700)

        self._findings: List[AuditFinding] = []
        self._build_ui()

    # ── Construção da UI ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Header geral
        self.lbl_header = QtWidgets.QLabel("Carregando auditoria…")
        self.lbl_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a3d6e;"
        )
        outer.addWidget(self.lbl_header)

        # Splitter principal
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # ── Esquerda: árvore de findings ──
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Auditoria", "Total"])
        self.tree.setColumnWidth(0, 230)
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        splitter.addWidget(self.tree)

        # ── Direita: painel de detalhes ──
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Header do finding selecionado
        self.lbl_finding_title = QtWidgets.QLabel("Selecione uma auditoria à esquerda.")
        self.lbl_finding_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #333;"
        )
        right_layout.addWidget(self.lbl_finding_title)

        self.lbl_finding_summary = QtWidgets.QLabel("")
        self.lbl_finding_summary.setWordWrap(True)
        self.lbl_finding_summary.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(self.lbl_finding_summary)

        # Tabela de issues
        self.table_issues = QtWidgets.QTableWidget()
        self.table_issues.setColumnCount(3)
        self.table_issues.setHorizontalHeaderLabels(["Locator", "Descrição", "Detalhes"])
        self.table_issues.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_issues.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table_issues.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.table_issues.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        right_layout.addWidget(self.table_issues, 1)

        # Barra de ações batch
        actions_row = QtWidgets.QHBoxLayout()
        self.btn_select_all = QtWidgets.QPushButton("☑ Selecionar tudo")
        self.btn_select_all.clicked.connect(self.table_issues.selectAll)

        self.btn_select_none = QtWidgets.QPushButton("☐ Limpar seleção")
        self.btn_select_none.clicked.connect(self.table_issues.clearSelection)

        self.btn_delete = QtWidgets.QPushButton("🗑 Excluir face(s) selecionada(s)")
        self.btn_delete.setStyleSheet(
            "background-color: #c62828; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_delete.setEnabled(False)

        actions_row.addWidget(self.btn_select_all)
        actions_row.addWidget(self.btn_select_none)
        actions_row.addStretch()
        actions_row.addWidget(self.btn_delete)
        right_layout.addLayout(actions_row)

        splitter.addWidget(right_widget)
        splitter.setSizes([280, 820])

        # ── Bottom: comandos globais ──
        bottom = QtWidgets.QHBoxLayout()
        self.btn_run_mining = QtWidgets.QPushButton("🔍 Mineração Biométrica (pesada)")
        self.btn_run_mining.setToolTip(
            "Roda hard mining O(N²) — pares de mesma identidade com baixa "
            "similaridade e pares de identidades distintas com alta similaridade. "
            "Pode levar minutos em datasets grandes."
        )
        self.btn_run_mining.clicked.connect(self.run_heavy_mining_requested.emit)

        self.btn_rerun = QtWidgets.QPushButton("🔄 Reanalisar")
        self.btn_rerun.setToolTip(
            "Reexecuta as auditorias rápidas (após edição manual no dataset)."
        )
        self.btn_rerun.clicked.connect(self.rerun_requested.emit)

        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)

        bottom.addWidget(self.btn_run_mining)
        bottom.addWidget(self.btn_rerun)
        bottom.addStretch()
        bottom.addWidget(self.btn_close)
        outer.addLayout(bottom)

    # ── API pública (chamada pelo controller) ────────────────────────────

    def set_findings(self, findings: Sequence[AuditFinding]) -> None:
        """Atualiza a árvore com a lista nova de findings.

        Agrupa por `category` preservando a ordem de inserção (os baratos
        rodam primeiro, os pesados são adicionados depois). Severity é
        propagada do maior pro menor por categoria via emoji do nó pai.
        """
        self._findings = list(findings)
        previous_id = self._currently_selected_audit_id()

        self.tree.blockSignals(True)
        self.tree.clear()

        # Agrupa preservando ordem de aparição
        category_order: List[str] = []
        by_category: dict = {}
        for f in self._findings:
            if f.category not in by_category:
                by_category[f.category] = []
                category_order.append(f.category)
            by_category[f.category].append(f)

        node_to_select: Optional[QtWidgets.QTreeWidgetItem] = None

        for cat in category_order:
            children = by_category[cat]
            cat_item = QtWidgets.QTreeWidgetItem(self.tree)
            cat_item.setText(0, cat)
            cat_item.setText(1, str(sum(f.count for f in children)))
            cat_item.setExpanded(True)

            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setFont(1, font)

            for finding in children:
                child = QtWidgets.QTreeWidgetItem(cat_item)
                emoji = finding.severity.emoji
                child.setText(0, f"{emoji}  {finding.audit_label}")
                child.setText(1, str(finding.count))
                if finding.count > 0:
                    child.setForeground(0, QtGui.QColor(_severity_color(finding.severity)))
                else:
                    child.setForeground(0, QtGui.QColor("#777"))
                child.setData(0, _FINDING_ROLE, finding)
                if previous_id is not None and finding.audit_id == previous_id:
                    node_to_select = child

        # Resumo no header
        n_total = sum(f.count for f in self._findings)
        n_critical = sum(
            f.count for f in self._findings
            if f.severity == AuditSeverity.CRITICAL
        )
        self.lbl_header.setText(
            f"Auditoria — {n_total} issue(s) detectada(s) | "
            f"{n_critical} crítica(s)"
        )

        self.tree.blockSignals(False)

        # Seleciona o nó previamente ativo (ou o primeiro com count > 0)
        if node_to_select is None:
            node_to_select = self._first_finding_with_issues()
        if node_to_select is not None:
            self.tree.setCurrentItem(node_to_select)
        else:
            self._render_empty_finding()

    def append_findings(self, extra: Sequence[AuditFinding]) -> None:
        """Adiciona novos findings à lista atual sem perder os existentes.

        Usado quando o hard mining termina e queremos juntar seus resultados
        à árvore sem rerunar as auditorias baratas."""
        self.set_findings(list(self._findings) + list(extra))

    # ── Slots internos ───────────────────────────────────────────────────

    def _on_tree_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._render_empty_finding()
            return
        finding = items[0].data(0, _FINDING_ROLE)
        if finding is None:
            self._render_empty_finding()
            return
        self._render_finding(finding)
        self.finding_selected.emit(finding)

    def _on_delete_clicked(self) -> None:
        paths = self._selected_locators()
        if not paths:
            return
        confirm = QtWidgets.QMessageBox.question(
            self, "Excluir faces",
            f"Remover {len(paths)} face(s) do dataset?\n\n"
            "Os dados originais permanecem em disco — apenas a entrada no "
            "dataset será removida. Para tornar permanente, salve o dataset.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            defaultButton=QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            self.delete_issues_requested.emit(paths)

    # ── Renderização ────────────────────────────────────────────────────

    def _render_finding(self, finding: AuditFinding) -> None:
        self.lbl_finding_title.setText(
            f"{finding.severity.emoji}  {finding.audit_label}"
        )
        self.lbl_finding_summary.setText(finding.summary)

        self.table_issues.setRowCount(len(finding.issues))
        for row, issue in enumerate(finding.issues):
            # Coluna 0: locator(s) — joga o primeiro path, mostra "(+N)" se mais
            primary = issue.locators[0] if issue.locators else ""
            extra = (f"  (+{len(issue.locators) - 1})"
                     if len(issue.locators) > 1 else "")
            cell0 = QtWidgets.QTableWidgetItem(primary + extra)
            cell0.setData(_PATH_ROLE, list(issue.locators))
            cell0.setToolTip("\n".join(issue.locators) if issue.locators else "")
            self.table_issues.setItem(row, 0, cell0)

            # Coluna 1: descrição
            cell1 = QtWidgets.QTableWidgetItem(issue.description)
            self.table_issues.setItem(row, 1, cell1)

            # Coluna 2: detalhes (metadata curta)
            details_txt = _format_metadata(issue.metadata)
            cell2 = QtWidgets.QTableWidgetItem(details_txt)
            cell2.setToolTip(details_txt)
            self.table_issues.setItem(row, 2, cell2)

        self.btn_delete.setEnabled(len(finding.issues) > 0)

    def _render_empty_finding(self) -> None:
        self.lbl_finding_title.setText("Nenhuma auditoria selecionada.")
        self.lbl_finding_summary.setText("")
        self.table_issues.setRowCount(0)
        self.btn_delete.setEnabled(False)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _selected_locators(self) -> List[str]:
        """Retorna a lista deduplicada de image_paths das linhas selecionadas."""
        seen = set()
        out: List[str] = []
        for idx in self.table_issues.selectedIndexes():
            if idx.column() != 0:
                continue
            paths = self.table_issues.item(idx.row(), 0).data(_PATH_ROLE) or []
            for p in paths:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    def _currently_selected_audit_id(self) -> Optional[str]:
        items = self.tree.selectedItems()
        if not items:
            return None
        finding = items[0].data(0, _FINDING_ROLE)
        return getattr(finding, "audit_id", None) if finding else None

    def _first_finding_with_issues(self) -> Optional[QtWidgets.QTreeWidgetItem]:
        """Itera a árvore retornando o primeiro nó-folha com count > 0."""
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            for j in range(cat.childCount()):
                child = cat.child(j)
                finding = child.data(0, _FINDING_ROLE)
                if finding and finding.count > 0:
                    return child
        # Fallback: primeira folha mesmo que vazia
        if self.tree.topLevelItemCount() > 0:
            cat = self.tree.topLevelItem(0)
            if cat.childCount() > 0:
                return cat.child(0)
        return None


def _severity_color(severity: AuditSeverity) -> str:
    return {
        AuditSeverity.CRITICAL: "#c62828",
        AuditSeverity.WARNING: "#ef6c00",
        AuditSeverity.INFO: "#1565c0",
    }.get(severity, "#333")


def _format_metadata(meta: dict) -> str:
    if not meta:
        return ""
    items = []
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            v_str = ", ".join(str(x) for x in v[:5])
            if len(v) > 5:
                v_str += f"… (+{len(v) - 5})"
        else:
            v_str = str(v)
        items.append(f"{k}={v_str}")
    return " | ".join(items)
