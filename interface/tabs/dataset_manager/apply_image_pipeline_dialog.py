"""Diálogo de aplicação de pipeline de imagens em todo o dataset
(PR I5).

Single-file (controller + view juntos, padrão do
schema_report_dialog.py / audit_log_dialog.py / localize_dataset_
dialog.py).

UX:
  - Combo: pipelines disponíveis na biblioteca global (lê
    `<library>/image_pipelines/*.json`).
  - Lista read-only de steps do pipeline selecionado.
  - Pasta de destino (qualquer) + nome do dataset. O dataset nasce
    autocontido em `<pasta>/<nome>/`, com parquet, cadeia de custódia,
    manifesto e `images/` dentro.
  - Checkbox skip_existing (default ON).
  - Botões Cancel / 🌀 Processar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet

from faceaudit.enums.global_library_folder import GlobalLibraryFolder
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.dataset_localization_service import (
    self_contained_dataset_dir,
)


class ApplyImagePipelineDialog(QtWidgets.QDialog):
    """Modal 580×520. Coleta pipeline + target_dir + skip_existing.

    Após `exec() == Accepted`, chame:
      - `selected_pipeline()` → `ImageProcessingPipelineConfig` ou None
      - `target_dir()` → `Path` (pasta-base escolhida)
      - `dataset_name()` → `str`
      - `dataset_dir()` → `Path` (`<pasta>/<nome>/`, o que será criado)
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
            "derivado com as imagens transformadas. Ele leva a identidade "
            "e as colunas de template, mas não embeddings nem métricas de "
            "qualidade: rode <i>Enriquecer</i> no novo dataset depois.",
            parent=self,
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
        apply_themed_stylesheet(
            self.lbl_pipeline_meta,
            lambda t: "color: " + t("TEXT") + "; padding: 4px; font-size: 11px;",
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

        # ── Destino do dataset ──
        target_box = QtWidgets.QGroupBox("Destino do dataset", parent=self)
        target_form = QtWidgets.QFormLayout(target_box)

        target_row = QtWidgets.QHBoxLayout()
        self.input_target = QtWidgets.QLineEdit(parent=target_box)
        self.input_target.textChanged.connect(self._refresh_preview)
        target_row.addWidget(self.input_target, 1)
        self.btn_browse = QtWidgets.QPushButton("📁 Escolher…", parent=target_box)
        self.btn_browse.clicked.connect(self._on_browse)
        target_row.addWidget(self.btn_browse)
        target_form.addRow("Pasta:", target_row)

        self.input_filename = QtWidgets.QLineEdit(parent=target_box)
        self.input_filename.setPlaceholderText("dataset__pipeline")
        self.input_filename.textChanged.connect(self._refresh_preview)
        target_form.addRow("Nome do dataset:", self.input_filename)

        self.lbl_preview = QtWidgets.QLabel("", parent=target_box)
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        target_form.addRow(self.lbl_preview)

        self.lbl_exists = QtWidgets.QLabel("", parent=target_box)
        self.lbl_exists.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_exists,
            lambda t: "color: " + t("WARNING") + "; font-size: 11px;",
        )
        self.lbl_exists.setVisible(False)
        target_form.addRow(self.lbl_exists)

        target_help = QtWidgets.QLabel(
            "O dataset ganha uma pasta própria dentro da pasta escolhida, "
            "com o parquet, a cadeia de custódia, o <code>manifest.json</code> "
            "e as imagens em <code>images/</code>. A pasta pode ser movida "
            "inteira sem quebrar o dataset. Falhas vão para "
            "<code>&lt;nome&gt;__errors.parquet</code>, na mesma pasta.",
            parent=target_box,
        )
        target_help.setWordWrap(True)
        apply_themed_stylesheet(
            target_help,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 10px;",
        )
        target_form.addRow(target_help)

        layout.addWidget(target_box)

        # ── Idempotência ──
        self.chk_skip = QtWidgets.QCheckBox(
            "Pular imagens cujo PNG destino já existe (idempotente — "
            "permite re-rodar sem refazer trabalho)"
        )
        self.chk_skip.setChecked(True)
        layout.addWidget(self.chk_skip)

        # ── Após processar ──
        after_box = QtWidgets.QGroupBox("Após processar")
        after_layout = QtWidgets.QVBoxLayout(after_box)
        self.rbn_keep_current = QtWidgets.QRadioButton(
            "Manter dataset atual aberto"
        )
        self.rbn_keep_current.setToolTip(
            "O novo dataset fica salvo em disco; o atual continua "
            "aberto na tela. Você abre o derivado depois via 'Abrir "
            "Dataset' se quiser."
        )
        self.rbn_keep_current.setChecked(True)
        self.rbn_open_new = QtWidgets.QRadioButton(
            "Abrir o novo dataset (fecha o atual)"
        )
        self.rbn_open_new.setToolTip(
            "Após o processamento, fecha o dataset atual e abre o "
            "derivado automaticamente. Se houver alterações não "
            "salvas, você será perguntado se quer salvar antes."
        )
        after_layout.addWidget(self.rbn_keep_current)
        after_layout.addWidget(self.rbn_open_new)
        layout.addWidget(after_box)

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
        if cfg.questioned is not None:
            meta_lines.append(
                f"<b>Questionada:</b> {cfg.questioned.file_name} — o resultado "
                "traz a comparação das medidas com ela"
            )
        self.lbl_pipeline_meta.setText(" · ".join(meta_lines))

        self.list_steps.clear()
        for i, step in enumerate(cfg.steps):
            self.list_steps.addItem(
                f"{i + 1}.  {step.step_type.label_pt}"
            )

        self._refresh_target_default()

    def _refresh_target_default(self):
        """Sugere a pasta-base e o nome `<dataset>__<pipeline>`."""
        cfg = self.selected_pipeline()
        if cfg is None:
            self.input_target.setText("")
            self.input_filename.setText("")
            return
        if not self.input_target.text().strip():
            self.input_target.setText(str(self._suggested_target_parent))
        safe_ds = self._safe_name(self._dataset_name)
        safe_pipe = self._safe_name(cfg.name)
        self.input_filename.setText(f"{safe_ds}__{safe_pipe}")
        self._refresh_preview()

    def _refresh_preview(self, *_):
        """Mostra a pasta que será criada e avisa se ela já existe."""
        ok = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        base = self.input_target.text().strip()
        nome = self.dataset_name()
        if not base or not nome:
            self.lbl_preview.setText("<i>Escolha a pasta e o nome do dataset.</i>")
            self.lbl_exists.setVisible(False)
            ok.setEnabled(False)
            return
        destino = self.dataset_dir()
        self.lbl_preview.setText(f"Será criado: <b>{destino}</b>")
        try:
            existe = destino.is_dir() and any(destino.iterdir())
        except OSError:
            existe = False
        if existe:
            self.lbl_exists.setText(
                "⚠ Esta pasta já existe e não está vazia. O parquet será "
                "substituído; com a opção abaixo marcada, os PNGs já "
                "gravados são reaproveitados."
            )
        self.lbl_exists.setVisible(existe)
        ok.setEnabled(self.selected_pipeline() is not None)

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
            self, "Pasta onde o dataset será criado", current,
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

    def dataset_name(self) -> str:
        """Nome do dataset derivado — também nome da pasta e do parquet.
        Espaços e acentos são mantidos; o serviço só troca caracteres
        proibidos em nome de pasta."""
        raw = self.input_filename.text().strip()
        if raw.lower().endswith(".parquet"):
            raw = raw[:-len(".parquet")]
        return raw

    def dataset_dir(self) -> Path:
        """Pasta que será criada: `<pasta escolhida>/<nome>/`."""
        return self_contained_dataset_dir(self.target_dir(), self.dataset_name())

    def open_new_after(self) -> bool:
        """True quando o usuário escolheu abrir o derivado e fechar o
        atual; False mantém o estado atual."""
        return self.rbn_open_new.isChecked()
