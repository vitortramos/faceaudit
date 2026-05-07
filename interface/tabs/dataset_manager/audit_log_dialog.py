"""Diálogo de Audit Log forense (PR 25, Tema 5.4).

Mostra a cadeia de custódia (lista de `AuditLogEntry` lida do sidecar
`<dataset_stem>.audit.json`) numa tabela ordenada cronologicamente, e
permite verificar a integridade da cadeia (hashes batem, parents
ligam) com um clique.

UX:
  - Tabela: Timestamp / Operador / Operação / Descrição / Hash do dataset / Hash da entrada
  - Botão "Verificar integridade" — roda `verify_chain`, mostra resultado
  - Botão "Exportar log" — salva uma cópia do sidecar JSON no destino
    escolhido pelo usuário (cadeia de custódia portátil)
  - Sem opção de editar/remover — log é append-only por design forense.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.models.audit.audit_log_entry import AuditLogEntry
from faceaudit.services.audit_log_service import (
    AuditChainProblem,
    AuditLogService,
)


_BG_OK = QColor("#d4edda")
_BG_PROBLEM = QColor("#f8d7da")


class AuditLogDialog(QtWidgets.QDialog):
    """Dialog de visualização do audit log de um dataset.

    Construa com o path do dataset (`.parquet`); o sidecar é localizado
    automaticamente via `AuditLogService.sidecar_path_for`.
    """

    _COLUMNS = [
        ("Timestamp (UTC)", 170),
        ("Operador", 180),
        ("Operação", 180),
        ("Descrição", 360),
        ("Hash do dataset", 110),
        ("Hash da entrada", 110),
    ]

    def __init__(
        self,
        dataset_path: str | Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._dataset_path = Path(dataset_path)
        self._entries: List[AuditLogEntry] = []
        self._problem_indices: set[int] = set()

        self.setWindowTitle(f"Audit Log — {self._dataset_path.stem}")
        self.setModal(True)
        self.resize(1180, 600)
        self._build_ui()
        self._reload()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        sidecar = AuditLogService.sidecar_path_for(self._dataset_path)
        header = QtWidgets.QLabel(
            "<b>Cadeia de custódia forense</b> — todas as operações que tocam "
            "o dataset, em ordem cronológica. Cada entrada referencia o hash "
            "da anterior, formando uma cadeia detectável de adulterações."
            f"<br><span style='color:#666;font-size:11px;'>Sidecar: "
            f"{sidecar.name}</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

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
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)

        # Status line
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("padding: 4px 0; font-weight: 500;")
        layout.addWidget(self.status_label)

        # Footer
        footer = QtWidgets.QHBoxLayout()
        self.btn_verify = QtWidgets.QPushButton("🔒 Verificar integridade")
        self.btn_verify.clicked.connect(self._on_verify)
        footer.addWidget(self.btn_verify)

        self.btn_export = QtWidgets.QPushButton("📤 Exportar log")
        self.btn_export.clicked.connect(self._on_export)
        footer.addWidget(self.btn_export)

        footer.addStretch()
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        footer.addWidget(self.btn_close)
        layout.addLayout(footer)

    # ── Dados ───────────────────────────────────────────────────────────────

    def _reload(self):
        self._entries = AuditLogService.load(self._dataset_path)
        self._problem_indices = set()
        self._populate_table()
        self._update_status_initial()

    def _populate_table(self):
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            self._fill_row(row, entry)
        self.table.resizeRowsToContents()

    def _fill_row(self, row: int, entry: AuditLogEntry):
        op = entry.operation
        op_text = f"{op.emoji}  {op.label_pt}"
        ds_hash = (entry.dataset_hash or "—")
        if len(ds_hash) > 12:
            ds_hash = ds_hash[:8] + "…" + ds_hash[-3:]
        eh = entry.entry_hash[:8] + "…" + entry.entry_hash[-3:] if entry.entry_hash else "—"

        cells = [
            entry.timestamp,
            entry.operator,
            op_text,
            entry.description,
            ds_hash,
            eh,
        ]
        for col, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)
            item.setToolTip(text)
            self.table.setItem(row, col, item)

        # Tooltip rico no campo "Descrição" mostra metadata
        if entry.metadata:
            try:
                meta_str = json.dumps(entry.metadata, indent=2, ensure_ascii=False)
                self.table.item(row, 3).setToolTip(
                    f"{entry.description}\n\n--- metadata ---\n{meta_str}"
                )
            except (TypeError, ValueError):
                pass

        # Tooltip rico nos hashes mostra valor completo
        if entry.dataset_hash:
            self.table.item(row, 4).setToolTip(f"SHA-256: {entry.dataset_hash}")
        self.table.item(row, 5).setToolTip(f"SHA-256: {entry.entry_hash}")

    def _update_status_initial(self):
        n = len(self._entries)
        if n == 0:
            self.status_label.setText(
                "📜 Nenhuma entrada registrada ainda. "
                "O log é criado quando você abre, salva ou modifica o dataset."
            )
            self.status_label.setStyleSheet("color: #666; padding: 4px 0;")
            self.btn_verify.setEnabled(False)
            self.btn_export.setEnabled(False)
        else:
            self.status_label.setText(
                f"📜 {n} entrada(s). Clique em 'Verificar integridade' para "
                "validar a cadeia."
            )
            self.status_label.setStyleSheet("color: #444; padding: 4px 0;")
            self.btn_verify.setEnabled(True)
            self.btn_export.setEnabled(True)

    # ── Ações ───────────────────────────────────────────────────────────────

    def _on_verify(self):
        ok, problems = AuditLogService.verify_chain(self._entries)
        self._problem_indices = {p.entry_index for p in problems}

        # Repinta linhas com cor de status
        for row in range(self.table.rowCount()):
            color = _BG_PROBLEM if row in self._problem_indices else _BG_OK
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                if item:
                    item.setBackground(QBrush(color))

        if ok:
            self.status_label.setText(
                f"✅ Cadeia íntegra — {len(self._entries)} entrada(s) verificada(s) com sucesso."
            )
            self.status_label.setStyleSheet(
                "color: #155724; background:#d4edda; padding: 6px; "
                "border-radius: 3px; font-weight: 600;"
            )
            QtWidgets.QMessageBox.information(
                self, "Integridade OK",
                f"Cadeia de custódia íntegra: {len(self._entries)} entrada(s).",
            )
        else:
            lines = "\n".join(
                f"  • [{p.entry_index}] {p.code}: {p.description}"
                for p in problems
            )
            self.status_label.setText(
                f"❌ Cadeia comprometida — {len(problems)} problema(s) detectado(s)."
            )
            self.status_label.setStyleSheet(
                "color: #721c24; background:#f8d7da; padding: 6px; "
                "border-radius: 3px; font-weight: 600;"
            )
            QtWidgets.QMessageBox.critical(
                self, "Falha de integridade",
                f"Problema(s) detectado(s) na cadeia:\n\n{lines}\n\n"
                "Possível adulteração ou remoção de entrada.",
            )

        # Audit-meta: registra o resultado da verificação no próprio log
        # (autoreferência). Pula se o log estava vazio.
        try:
            if self._entries:
                AuditLogService.append(
                    self._dataset_path,
                    AuditOperation.INTEGRITY_VERIFIED if ok else AuditOperation.INTEGRITY_FAILED,
                    (
                        "Verificação de cadeia: OK"
                        if ok
                        else f"Verificação de cadeia: {len(problems)} problema(s)"
                    ),
                    metadata={
                        "n_entries_verified": len(self._entries),
                        "n_problems": len(problems),
                        "problem_codes": sorted({p.code for p in problems}),
                    },
                )
                # Recarrega pra mostrar a entrada nova
                self._reload()
        except Exception:
            pass  # best-effort

    def _on_export(self):
        sidecar = AuditLogService.sidecar_path_for(self._dataset_path)
        if not sidecar.exists():
            QtWidgets.QMessageBox.warning(
                self, "Sem log",
                "Sidecar de audit log não existe ainda — execute uma "
                "operação primeiro.",
            )
            return
        suggested = f"{self._dataset_path.stem}-audit-log.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar audit log", suggested, "JSON (*.json)",
        )
        if not path:
            return
        try:
            shutil.copy2(sidecar, path)
            QtWidgets.QMessageBox.information(
                self, "Exportado",
                f"Log exportado em:\n{path}\n\n{len(self._entries)} entrada(s).",
            )
        except OSError as e:
            QtWidgets.QMessageBox.critical(
                self, "Erro ao exportar", str(e),
            )

    def _on_row_double_clicked(self, item: QtWidgets.QTableWidgetItem):
        """Mostra detalhes completos da entrada (metadata, hashes) num dialog."""
        row = item.row()
        if row < 0 or row >= len(self._entries):
            return
        entry = self._entries[row]
        details = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Entrada #{row + 1} — {entry.operation.label_pt}")
        dlg.resize(700, 500)
        lay = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QPlainTextEdit(details)
        text.setReadOnly(True)
        text.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(text)
        btn = QtWidgets.QPushButton("Fechar")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)
        dlg.exec()
