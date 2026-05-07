"""Diálogo de aplicação de pipeline de imagens em todo o dataset
(PR I5).

Single-file (controller + view juntos, padrão do
schema_report_dialog.py / audit_log_dialog.py / localize_dataset_
dialog.py).

UX:
  - Combo: pipelines disponíveis na biblioteca global (lê
    `<library>/image_pipelines/*.json`).
  - Lista read-only de steps do pipeline selecionado.
  - Seletor de pasta destino (default = `<parquet_dir>/
    <dataset>__<pipeline>/`).
  - Checkbox skip_existing (default ON).
  - Botões Cancel / 🌀 Processar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.global_library_folder import GlobalLibraryFolder
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)


class ApplyImagePipelineDialog(QtWidgets.QDialog):
    """Modal 580×520. Coleta pipeline + target_dir + skip_existing.

    Após `exec() == Accepted`, chame:
      - `selected_pipeline()` → `ImageProcessingPipelineConfig` ou None
      - `target_dir()` → `Path`
      - `skip_existing()` → `bool`
    """

    def __init__(
        self,
        library_root: Path,
        dataset_name: str,
        suggested_target_parent: Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._library_root = Path(library_root)
        self._dataset_name = dataset_name
        self._suggested_target_parent = Path(suggested_target_parent)
        self._loaded_pipelines: list[
            tuple[str, ImageProcessingPipelineConfig]
        ] = []

        self.setWindowTitle("Aplicar Pipeline de Imagens")
        self.setModal(True)
        self.resize(580, 520)
        self._build_ui()
        self._load_pipelines()
        self._refresh_target_default()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Aplicar pipeline em todo o dataset</b> — gera um dataset "
            "derivado enxuto (sem embeddings, só image_path/hash/file_name/"
            "identity). Perito roda <i>Enriquecer</i> depois pra recalcular "
            "métricas no novo dataset."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Pipeline ──
        pipeline_box = QtWidgets.QGroupBox("Pipeline da biblioteca global")
        pipe_layout = QtWidgets.QVBoxLayout(pipeline_box)

        self.cmb_pipeline = QtWidgets.QComboBox()
        self.cmb_pipeline.currentIndexChanged.connect(
            self._on_pipeline_selected
        )
        pipe_layout.addWidget(self.cmb_pipeline)

        self.lbl_pipeline_meta = QtWidgets.QLabel("")
        self.lbl_pipeline_meta.setStyleSheet(
            "color: #555; padding: 4px; font-size: 11px;"
        )
        self.lbl_pipeline_meta.setWordWrap(True)
        pipe_layout.addWidget(self.lbl_pipeline_meta)

        self.list_steps = QtWidgets.QListWidget()
        self.list_steps.setMaximumHeight(120)
        self.list_steps.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection,
        )
        pipe_layout.addWidget(self.list_steps)

        layout.addWidget(pipeline_box)

        # ── Destino ──
        target_box = QtWidgets.QGroupBox("Pasta destino")
        target_layout = QtWidgets.QHBoxLayout(target_box)
        self.input_target = QtWidgets.QLineEdit()
        target_layout.addWidget(self.input_target, 1)
        self.btn_browse = QtWidgets.QPushButton("📁 Escolher…")
        self.btn_browse.clicked.connect(self._on_browse)
        target_layout.addWidget(self.btn_browse)
        layout.addWidget(target_box)

        # ── Idempotência ──
        self.chk_skip = QtWidgets.QCheckBox(
            "Pular imagens cujo PNG destino já existe (idempotente — "
            "permite re-rodar sem refazer trabalho)"
        )
        self.chk_skip.setChecked(True)
        layout.addWidget(self.chk_skip)

        layout.addStretch()

        # Botões
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("🌀 Processar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _load_pipelines(self):
        """Lê `<library>/image_pipelines/*.json` e popula combo."""
        pipelines_dir = (
            self._library_root / GlobalLibraryFolder.IMAGE_PIPELINES.value
        )
        if not pipelines_dir.exists():
            self._set_empty_state(
                f"Pasta {pipelines_dir} não existe. "
                "Salve um pipeline no Editor primeiro."
            )
            return
        files = sorted(pipelines_dir.glob("*.json"))
        if not files:
            self._set_empty_state(
                f"Nenhum pipeline em {pipelines_dir}. "
                "Salve um pipeline no Editor primeiro."
            )
            return

        self._loaded_pipelines.clear()
        self.cmb_pipeline.clear()
        for f in files:
            try:
                cfg = ImageProcessingPipelineConfig.from_json_string(
                    f.read_text(encoding="utf-8"),
                )
            except Exception as e:
                continue  # ignora arquivos corrompidos
            self._loaded_pipelines.append((f.stem, cfg))
            self.cmb_pipeline.addItem(f.stem, userData=cfg)

        if self._loaded_pipelines:
            self._on_pipeline_selected(0)

    def _set_empty_state(self, msg: str):
        self.cmb_pipeline.clear()
        self.cmb_pipeline.addItem("(vazio)")
        self.cmb_pipeline.setEnabled(False)
        self.lbl_pipeline_meta.setText(msg)
        self.list_steps.clear()
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setEnabled(False)

    def _on_pipeline_selected(self, idx: int):
        if idx < 0 or idx >= len(self._loaded_pipelines):
            return
        _, cfg = self._loaded_pipelines[idx]
        meta_lines = [
            f"<b>Origem:</b> {cfg.input_source.label_pt}",
            f"<b>Seed:</b> {cfg.seed}",
            f"<b>Steps:</b> {len(cfg.steps)}",
        ]
        if cfg.description:
            meta_lines.append(f"<b>Descrição:</b> {cfg.description}")
        self.lbl_pipeline_meta.setText(" · ".join(meta_lines))

        self.list_steps.clear()
        for i, step in enumerate(cfg.steps):
            self.list_steps.addItem(
                f"{i + 1}.  {step.step_type.label_pt}"
            )

        self._refresh_target_default()

    def _refresh_target_default(self):
        """Sugere `<parent>/<dataset>__<pipeline>/`."""
        cfg = self.selected_pipeline()
        if cfg is None:
            self.input_target.setText("")
            return
        # Sanitização básica (igual do service)
        safe_ds = self._safe_name(self._dataset_name)
        safe_pipe = self._safe_name(cfg.name)
        target = (
            self._suggested_target_parent / f"{safe_ds}__{safe_pipe}"
        )
        self.input_target.setText(str(target))

    @staticmethod
    def _safe_name(raw: str) -> str:
        bad = '\\/:*?"<>| '
        out = raw or "untitled"
        for ch in bad:
            out = out.replace(ch, "_")
        return out

    def _on_browse(self):
        current = self.input_target.text().strip() or str(
            self._suggested_target_parent
        )
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pasta destino", current,
        )
        if chosen:
            self.input_target.setText(chosen)

    # ── API pública ────────────────────────────────────────────────────────

    def selected_pipeline(self) -> Optional[ImageProcessingPipelineConfig]:
        idx = self.cmb_pipeline.currentIndex()
        if idx < 0 or idx >= len(self._loaded_pipelines):
            return None
        return self._loaded_pipelines[idx][1]

    def target_dir(self) -> Path:
        return Path(self.input_target.text().strip())

    def skip_existing(self) -> bool:
        return self.chk_skip.isChecked()
