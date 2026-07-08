"""Diálogo de Relatório de Schema (PR 24 — Tema 5.3).

Mostra a cobertura por coluna do dataset atual (% NaN, distinct, min/max,
dim) lendo de `DatasetStats.column_coverage` — que já é populado pelo
`DatasetStatsService` em todos os 4 caminhos.

Inclui também um dialog de export (Tema 5.5) acessível via botão no
rodapé. Export aceita seleção de colunas, escolha de delimitador, formato
CSV ou Excel, e opção de excluir embeddings (que viram strings gigantes
inúteis em CSV).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.dataset_stats_service import DatasetStatsService
from interface.theme import current_token_fn


def _coverage_color(pct: float) -> QColor:
    # Cores de cobertura como estado (verde=100 / amarelo=parcial /
    # vermelho=<50 / neutro=0) — lidas dos tokens no momento da pintura.
    t = current_token_fn()
    if pct >= 1.0:
        return QColor(t("SUCCESS_BG"))     # 100%
    if pct <= 0.0:
        return QColor(t("BORDER"))         # 0% (neutro)
    if pct >= 0.5:
        return QColor(t("WARNING_BG"))     # 50–99%
    return QColor(t("DANGER_BG"))          # < 50%


def _format_pct(pct: float) -> str:
    return f"{pct * 100:.1f}%"


def _format_value(v: Optional[float]) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


class SchemaReportDialog(QtWidgets.QDialog):
    """Dialog de Colunas — Relatório de cobertura + controle de
    visibilidade na tabela (PR — Tema 5.3 ampliado).

    Primeira coluna do grid (checkbox "Visível") aplica live via
    `on_visibility_change(name, visible)`. Demais colunas mostram
    metadados + cobertura, como antes."""

    _COLUMNS = [
        ("Visível", 60),
        ("Coluna", 140),
        ("Rótulo", 130),
        ("Role", 90),
        ("Source", 80),
        ("Tipo", 60),
        ("Não-nulos", 80),
        ("Total", 70),
        ("Cobertura", 90),
        ("Min", 70),
        ("Max", 70),
        ("Média", 70),
        ("Distintos", 80),
        ("Dim", 50),
    ]
    # Índice da coluna do checkbox — usado pelo handler de itemChanged
    _COL_VISIBLE = 0
    _COL_NAME = 1

    def __init__(
        self,
        dataset: FaceDataset,
        parent=None,
        *,
        on_export: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_visibility_change: Optional[Callable[[str, bool], None]] = None,
    ):
        super().__init__(parent)
        self._dataset = dataset
        # PR 25 (Tema 5.4) — callback opcional chamado quando o usuário
        # exporta o dataset via ExportCsvDialog. Recebe um dict com
        # `format`, `rows`, `cols`, `path`. Default None preserva o
        # comportamento standalone (sem audit log).
        self._on_export = on_export
        # Callback do toggle de visibilidade — `(name, visible)`. Live
        # (cada toggle dispara imediatamente). None desabilita os
        # checkboxes da coluna "Visível".
        self._on_visibility_change = on_visibility_change
        self.setWindowTitle(f"Colunas — {dataset.name}")
        self.setModal(True)
        self.resize(1150, 600)
        self._build_ui()
        self._populate()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # Header explicativo
        _muted = current_token_fn()("TEXT_MUTED")
        header = QtWidgets.QLabel(
            "<b>Painel de colunas</b> — marque/desmarque <i>Visível</i> "
            "pra controlar quais colunas aparecem na tabela do dataset "
            "(aplicação imediata). Demais colunas são informativas: "
            "cobertura (% não-nulos), distintos, min/max etc. "
            f"<span style='color:{_muted};font-size:11px;'>Verde = 100%, "
            "amarelo = parcial, vermelho = &lt; 50%.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Toolbar de visibilidade — Marcar/Desmarcar tudo em batch
        vis_bar = QtWidgets.QHBoxLayout()
        self.btn_show_all = QtWidgets.QPushButton("☑ Mostrar todas")
        self.btn_show_all.clicked.connect(
            lambda: self._toggle_all_visibility(True)
        )
        self.btn_hide_all = QtWidgets.QPushButton("☐ Ocultar todas")
        self.btn_hide_all.clicked.connect(
            lambda: self._toggle_all_visibility(False)
        )
        vis_bar.addWidget(self.btn_show_all)
        vis_bar.addWidget(self.btn_hide_all)
        vis_bar.addStretch()
        layout.addLayout(vis_bar)

        # Tabela
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels([h for h, _ in self._COLUMNS])
        for i, (_, w) in enumerate(self._COLUMNS):
            self.table.setColumnWidth(i, w)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        # Conecta itemChanged pra capturar toggle dos checkboxes —
        # bloqueado durante populate.
        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table)

        # Footer com botões
        footer = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("📤 Exportar para CSV/Excel…")
        self.btn_export.clicked.connect(self._on_export)
        footer.addWidget(self.btn_export)
        footer.addStretch()
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        footer.addWidget(self.btn_close)
        layout.addLayout(footer)

    # ── Populate ────────────────────────────────────────────────────────────

    def _populate(self):
        service = DatasetStatsService()
        stats = service.compute_from_dataset(self._dataset)
        coverage = stats.column_coverage or {}
        specs = self._dataset.column_specs or {}

        # Ordena por (categoria, role.value, name) — agrupa visualmente.
        # `column_specs` mantém ordem do dict (post-PR 20 drag-reorder),
        # então respeitar essa ordem é o esperado pelo usuário.
        names: List[str] = []
        for name in specs.keys():
            names.append(name)
        # Inclui colunas que estão em coverage mas não em specs (parquets
        # legados ou colunas técnicas não declaradas).
        for name in coverage.keys():
            if name not in names:
                names.append(name)

        # Bloqueia signals — sem isso, o setItem do checkbox abaixo
        # dispara `itemChanged` por cada linha e tenta aplicar
        # visibilidade no model (com estado ainda transiente).
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(names))
            for row, name in enumerate(names):
                spec = specs.get(name)
                cs = coverage.get(name)
                self._fill_row(row, name, spec, cs)
        finally:
            self.table.blockSignals(False)

        self.table.resizeRowsToContents()

    def _fill_row(self, row: int, name: str, spec: Any, cs: Any):
        label = getattr(spec, "label", name) if spec else name
        role = getattr(getattr(spec, "role", None), "value", "—") if spec else "—"
        source = getattr(getattr(spec, "source", None), "value", "—") if spec else "—"
        dtype = getattr(getattr(spec, "dtype", None), "value", "—") if spec else "—"

        non_null = getattr(cs, "non_null_count", 0) if cs else 0
        total = getattr(cs, "total_count", 0) if cs else 0
        pct = getattr(cs, "coverage", 0.0) if cs else 0.0
        min_val = getattr(cs, "min_value", None) if cs else None
        max_val = getattr(cs, "max_value", None) if cs else None
        mean_val = getattr(cs, "mean_value", None) if cs else None
        distinct = getattr(cs, "distinct_count", None) if cs else None
        dim = getattr(cs, "dim", None) if cs else None

        # Coluna 0 — checkbox de visibilidade. Estado lido de
        # `spec.visible_default` (mutado pelo model.set_column_visibility
        # ao longo da sessão, então reflete estado corrente).
        visible_now = (
            bool(getattr(spec, "visible_default", True))
            if spec else False
        )
        vis_item = QtWidgets.QTableWidgetItem()
        vis_item.setFlags(
            (QtCore.Qt.ItemFlag.ItemIsEnabled
             | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            if (spec is not None and self._on_visibility_change is not None)
            else QtCore.Qt.ItemFlag.ItemIsEnabled
        )
        vis_item.setCheckState(
            QtCore.Qt.CheckState.Checked
            if visible_now
            else QtCore.Qt.CheckState.Unchecked
        )
        # Guarda o nome da coluna no UserRole pra recuperar no handler
        vis_item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
        vis_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, self._COL_VISIBLE, vis_item)

        # Demais colunas — informativas (read-only)
        cells = [
            name, label, role, source, dtype,
            f"{non_null:,}", f"{total:,}", _format_pct(pct),
            _format_value(min_val), _format_value(max_val), _format_value(mean_val),
            "—" if distinct is None else f"{distinct:,}",
            "—" if dim is None else str(dim),
        ]
        for offset, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            self.table.setItem(row, self._COL_VISIBLE + 1 + offset, item)

        # Pinta a linha inteira com a cor da cobertura
        color = _coverage_color(pct)
        brush = QBrush(color)
        for col in range(self.table.columnCount()):
            item = self.table.item(row, col)
            if item is not None:
                item.setBackground(brush)

    # ── Visibilidade ────────────────────────────────────────────────────────

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Disparado por toggle do checkbox da coluna Visível. Ignora
        toggles de outras colunas (no caso, nenhuma editable além do
        checkbox, mas é defensive). Chama o callback do controller."""
        if item is None or item.column() != self._COL_VISIBLE:
            return
        if self._on_visibility_change is None:
            return
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not name:
            return
        visible = (item.checkState() == QtCore.Qt.CheckState.Checked)
        try:
            self._on_visibility_change(name, visible)
        except Exception:
            # Callback do controller é best-effort — não trava a UI
            # se algo falhar lá.
            pass

    def _toggle_all_visibility(self, visible: bool) -> None:
        """Marca/desmarca todos os checkboxes em batch e dispara o
        callback do controller pra cada mudança efetiva."""
        if self._on_visibility_change is None:
            return
        target_state = (
            QtCore.Qt.CheckState.Checked
            if visible else QtCore.Qt.CheckState.Unchecked
        )
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, self._COL_VISIBLE)
                if item is None:
                    continue
                # Só linhas com checkbox editável (specs presentes)
                if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsUserCheckable):
                    continue
                if item.checkState() == target_state:
                    continue
                item.setCheckState(target_state)
                name = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if name:
                    try:
                        self._on_visibility_change(name, visible)
                    except Exception:
                        pass
        finally:
            self.table.blockSignals(False)

    # ── Export ──────────────────────────────────────────────────────────────

    def _on_export(self):
        dlg = ExportCsvDialog(
            self._dataset, parent=self, on_export_complete=self._on_export_callback,
        )
        dlg.exec()

    def _on_export_callback(self, meta: Dict[str, Any]) -> None:
        """Chamado pelo ExportCsvDialog após save bem-sucedido.
        Repassa pra fora se houver um callback registrado."""
        if self._on_export is not None:
            try:
                self._on_export(meta)
            except Exception:
                pass  # callback de auditoria é best-effort


# ═══════════════════════════════════════════════════════════════════════════
# Export Dialog (PR 24 — Tema 5.5)
# ═══════════════════════════════════════════════════════════════════════════

_DELIMITER_OPTIONS = [
    (",", "Vírgula  ( , )"),
    (";", "Ponto-e-vírgula  ( ; )"),
    ("\t", "Tab"),
    ("|", "Pipe  ( | )"),
]

_FORMAT_OPTIONS = [
    ("csv", "CSV (.csv)"),
    ("xlsx", "Excel (.xlsx)"),
]


class ExportCsvDialog(QtWidgets.QDialog):
    """Coleta opções de export e salva via pandas. Tema 5.5 (PR 24).

    UX:
      - Lista de colunas com checkbox (default: visíveis no schema).
      - Botões helper: "Todas", "Apenas visíveis", "Sem embeddings".
      - Format CSV/XLSX, delimitador (apenas CSV).
      - Excluir embeddings por default (ficam strings gigantes inúteis).
    """

    def __init__(
        self,
        dataset: FaceDataset,
        parent=None,
        *,
        on_export_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        super().__init__(parent)
        self._dataset = dataset
        self._on_export_complete = on_export_complete
        self.setWindowTitle("Exportar Dataset")
        self.setModal(True)
        self.resize(520, 520)
        self._build_ui()
        self._populate_columns()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Seleção de colunas ────────────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("<b>Colunas a exportar</b>"))
        self.list_columns = QtWidgets.QListWidget()
        self.list_columns.setMinimumHeight(220)
        layout.addWidget(self.list_columns)

        helper_row = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Todas")
        self.btn_all.clicked.connect(lambda: self._toggle_all(True))
        self.btn_none = QtWidgets.QPushButton("Nenhuma")
        self.btn_none.clicked.connect(lambda: self._toggle_all(False))
        self.btn_visible = QtWidgets.QPushButton("Apenas visíveis")
        self.btn_visible.clicked.connect(self._select_visible_only)
        self.btn_no_emb = QtWidgets.QPushButton("Sem embeddings")
        self.btn_no_emb.clicked.connect(self._unselect_embeddings)
        for btn in (self.btn_all, self.btn_none, self.btn_visible, self.btn_no_emb):
            btn.setStyleSheet("padding: 4px 10px;")
            helper_row.addWidget(btn)
        helper_row.addStretch()
        layout.addLayout(helper_row)

        # ── Formato + delimitador ─────────────────────────────────────────
        form = QtWidgets.QFormLayout()
        self.cmb_format = QtWidgets.QComboBox()
        for value, label in _FORMAT_OPTIONS:
            self.cmb_format.addItem(label, userData=value)
        self.cmb_format.currentIndexChanged.connect(self._on_format_changed)
        form.addRow("Formato:", self.cmb_format)

        self.cmb_delim = QtWidgets.QComboBox()
        for value, label in _DELIMITER_OPTIONS:
            self.cmb_delim.addItem(label, userData=value)
        form.addRow("Delimitador:", self.cmb_delim)

        layout.addLayout(form)

        # ── Botões ────────────────────────────────────────────────────────
        layout.addStretch()
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Exportar…")
        self.buttons.accepted.connect(self._on_export)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _populate_columns(self):
        """Popula a lista a partir do df do dataset. Default checked:
        visíveis e não-embedding."""
        df = self._dataset.to_pandas()
        if df is None or df.empty:
            return
        specs = self._dataset.column_specs or {}

        self.list_columns.clear()
        for col in df.columns:
            spec = specs.get(col)
            label = getattr(spec, "label", col) if spec else col
            visible = bool(getattr(spec, "visible_default", True)) if spec else True
            is_emb = self._is_embedding_column(col, spec)
            display = f"{col}  [{label}]" if label != col else col
            if is_emb:
                display += "  ⚠ embedding (gigante em CSV)"
            item = QtWidgets.QListWidgetItem(display)
            item.setData(Qt.ItemDataRole.UserRole, col)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check = visible and not is_emb
            item.setCheckState(Qt.CheckState.Checked if check else Qt.CheckState.Unchecked)
            self.list_columns.addItem(item)

    @staticmethod
    def _is_embedding_column(col: str, spec: Any) -> bool:
        if spec is not None:
            role = getattr(getattr(spec, "role", None), "value", None)
            if role == "embedding":
                return True
        return col.startswith("embedding")

    def _toggle_all(self, checked: bool):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_columns.count()):
            self.list_columns.item(i).setCheckState(state)

    def _select_visible_only(self):
        specs = self._dataset.column_specs or {}
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            col = item.data(Qt.ItemDataRole.UserRole)
            spec = specs.get(col)
            visible = bool(getattr(spec, "visible_default", True)) if spec else True
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)

    def _unselect_embeddings(self):
        specs = self._dataset.column_specs or {}
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            col = item.data(Qt.ItemDataRole.UserRole)
            if self._is_embedding_column(col, specs.get(col)):
                item.setCheckState(Qt.CheckState.Unchecked)

    def _on_format_changed(self, _idx: int):
        # Delimitador só é relevante pra CSV.
        is_csv = self.cmb_format.currentData() == "csv"
        self.cmb_delim.setEnabled(is_csv)

    def _selected_columns(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list_columns.count()):
            item = self.list_columns.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole))
        return out

    def _on_export(self):
        cols = self._selected_columns()
        if not cols:
            QtWidgets.QMessageBox.warning(
                self, "Export", "Nenhuma coluna selecionada.",
            )
            return

        fmt = self.cmb_format.currentData()
        delim = self.cmb_delim.currentData()
        ext = "csv" if fmt == "csv" else "xlsx"
        filt = "CSV (*.csv)" if fmt == "csv" else "Excel (*.xlsx)"
        suggested = f"{self._dataset.name or 'dataset'}.{ext}"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar para…", suggested, filt,
        )
        if not path:
            return

        try:
            df = self._dataset.to_pandas()[cols]
        except KeyError as e:
            QtWidgets.QMessageBox.critical(
                self, "Coluna ausente",
                f"Coluna selecionada não está no DataFrame: {e}",
            )
            return

        try:
            if fmt == "csv":
                df.to_csv(path, sep=delim, index=False)
            else:
                # to_excel requer openpyxl; quando ausente, mostra dica.
                try:
                    df.to_excel(path, index=False)
                except ImportError:
                    QtWidgets.QMessageBox.critical(
                        self, "Dependência ausente",
                        "Export para Excel requer 'openpyxl'. "
                        "Instale com: pip install openpyxl",
                    )
                    return
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Erro ao exportar", str(e),
            )
            return

        # PR 25 (Tema 5.4) — notifica o caller (Schema Report → controller)
        # pra que o evento EXPORT seja registrado no audit log.
        if self._on_export_complete is not None:
            try:
                self._on_export_complete({
                    "format": fmt,
                    "delimiter": delim if fmt == "csv" else None,
                    "path": str(path),
                    "rows": int(len(df)),
                    "cols": int(len(cols)),
                    "column_names": list(cols),
                })
            except Exception:
                pass  # best-effort

        QtWidgets.QMessageBox.information(
            self, "Export",
            f"Salvo em:\n{path}\n\n{len(df):,} linhas, {len(cols)} colunas.",
        )
        self.accept()
