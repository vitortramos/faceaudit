"""Diálogo de confirmação pra "Selar projeto" (PR L2.b).

Mostra resumo do que será feito (quantos datasets, quanto pode
ocupar) e oferece a opção de simular antes de selar de verdade.

UX:
  - Texto explicativo + warning sobre destrutividade
  - Tabela compacta listando os datasets que serão processados
  - Checkbox "Apenas simular (dry-run)"
  - Botões Cancelar / Selar
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.face_dataset import FaceDataset


class SealProjectDialog(QtWidgets.QDialog):
    """Confirma a operação de selar com preview dos datasets."""

    def __init__(self, project, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._project = project
        self.setWindowTitle(f"Selar Projeto — {project.metadata.name}")
        self.setModal(True)
        self.resize(680, 520)
        self._build_ui()
        self._populate_preview()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Selar projeto</b> — copia todas as imagens referenciadas "
            "pelos datasets pra dentro do <code>.fa</code> e reescreve os "
            "parquets pra paths relativos. Após selar, o projeto fica "
            "totalmente autocontido — pode ser zipado e levado pra outra "
            "máquina sem perder integridade."
            "<br><br>"
            "<span style='color:#a05500;'>⚠ Operação <b>destrutiva</b>: "
            "os parquets do projeto são reescritos in-place. Faça um "
            "backup antes se necessário. O audit log de cada parquet "
            "preserva a trilha (quem/quando/o quê).</span>"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Tabela com datasets a processar ──
        layout.addWidget(QtWidgets.QLabel("<b>Datasets a processar:</b>"))
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "ID", "Faces", "Modo atual", "Ação",
        ])
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 110)
        self.table.setColumnWidth(3, 200)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection,
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # Resumo
        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(self.lbl_summary)

        # ── Dry-run ──
        self.chk_dry_run = QtWidgets.QCheckBox(
            "Apenas simular (dry-run) — não escreve nada, só lista o "
            "que seria feito"
        )
        self.chk_dry_run.setChecked(False)
        layout.addWidget(self.chk_dry_run)

        # ── Botões ──
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)\
            .setText("🔒 Selar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    # ── Preview ────────────────────────────────────────────────────────────

    def _populate_preview(self):
        """Lista os datasets do projeto e marca cada um como
        'localizar' ou 'pular' (já em RELATIVE)."""
        artifacts = self._project.storage.list_artifacts(
            StrategyArtifactType.GLOBAL_DATASET,
        )
        rows: List[Tuple[str, int, str, str]] = []  # (id, n_faces, mode, action)
        n_seal = 0
        n_skip = 0
        n_unreadable = 0

        for ref in artifacts:
            ds_id = Path(ref.path).stem
            abs_path = self._project.storage.project_root / ref.path
            try:
                ds = FaceDataset.from_parquet(abs_path)
                n_faces = len(ds.faces)
                mode_str = ds.path_mode.value
                if ds.path_mode == PathMode.RELATIVE:
                    action = "Pular (já localizado)"
                    n_skip += 1
                else:
                    action = "Localizar imagens + reescrever paths"
                    n_seal += 1
            except Exception as e:
                n_faces = 0
                mode_str = "?"
                action = f"⚠ erro ao ler: {e}"
                n_unreadable += 1
            rows.append((ds_id, n_faces, mode_str, action))

        self.table.setRowCount(len(rows))
        for r, (ds_id, n_faces, mode_str, action) in enumerate(rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(ds_id))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(n_faces)))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(mode_str))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(action))

        # Resumo
        bits = [f"<b>{len(rows)}</b> dataset(s) total"]
        if n_seal:
            bits.append(f"<b>{n_seal}</b> serão selados")
        if n_skip:
            bits.append(f"{n_skip} pulados")
        if n_unreadable:
            bits.append(f"<span style='color:#a00;'>{n_unreadable} ilegíveis</span>")
        self.lbl_summary.setText(" · ".join(bits))

        # Desabilita Selar se não há nada a fazer
        if n_seal == 0 and n_unreadable == 0:
            ok_btn = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
            ok_btn.setEnabled(False)
            ok_btn.setToolTip("Nenhum dataset precisa ser selado.")

    @property
    def dry_run(self) -> bool:
        return self.chk_dry_run.isChecked()
