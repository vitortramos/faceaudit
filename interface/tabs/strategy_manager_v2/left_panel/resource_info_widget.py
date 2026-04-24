# interface/tabs/strategy_manager_v2/left_panel/resource_info_widget.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.services.hash_service import HashService


class ResourceInfoWidget(QtWidgets.QWidget):
    """
    Painel tabulado de informações do recurso selecionado na árvore.

    Exibe duas abas:
    - Propriedades: tipo, ID, path, hash, tamanho, stats do dataset ou
      detalhes do pipeline (conforme o tipo do recurso)
    - Utilização: em quais estratégias/instruções o recurso aparece
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._hash_service = HashService()
        self._init_ui()
        self.clear()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── QTabWidget com 2 abas ────────────────────────────────────
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                border-top: 1px solid #d0d8e0;
                background: white;
            }
            QTabBar::tab {
                padding: 4px 10px;
                font-size: 11px;
                background: transparent;
                border: none;
                border-bottom: 2px solid transparent;
                margin-right: 2px;
                color: #666;
            }
            QTabBar::tab:selected {
                color: #2c3e50;
                font-weight: bold;
                border-bottom: 2px solid #4a90d9;
            }
            QTabBar::tab:hover:!selected {
                color: #333;
                border-bottom: 2px solid #ccc;
            }
        """)

        # ── Tab 1: Propriedades ──────────────────────────────────────
        props_scroll = QtWidgets.QScrollArea()
        props_scroll.setWidgetResizable(True)
        props_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        props_container = QtWidgets.QWidget()
        props_layout = QtWidgets.QVBoxLayout(props_container)
        props_layout.setContentsMargins(8, 8, 8, 8)
        props_layout.setSpacing(6)

        # Metadados em grade
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setSpacing(4)

        self.lbl_id = QtWidgets.QLabel("—")
        self.lbl_type = QtWidgets.QLabel("—")
        self.lbl_path = QtWidgets.QLabel("—")
        self.lbl_path.setWordWrap(True)
        self.lbl_hash = QtWidgets.QLabel("—")
        self.lbl_hash.setWordWrap(True)
        self.lbl_size = QtWidgets.QLabel("—")

        for lbl in (self.lbl_id, self.lbl_type, self.lbl_path, self.lbl_hash, self.lbl_size):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        _lbl_style = "font-size: 11px; color: #666; font-weight: bold;"
        for text, value_lbl in [
            ("ID:", self.lbl_id),
            ("Tipo:", self.lbl_type),
            ("Path:", self.lbl_path),
            ("Hash:", self.lbl_hash),
            ("Tamanho:", self.lbl_size),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            form.addRow(lbl, value_lbl)

        props_layout.addLayout(form)

        # Estatísticas do dataset (visível apenas para DATASET)
        self._stats_section = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(self._stats_section)
        stats_layout.setContentsMargins(0, 4, 0, 0)
        stats_layout.setSpacing(4)

        stats_sep = QtWidgets.QFrame()
        stats_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        stats_sep.setStyleSheet("color: #ddd;")
        stats_layout.addWidget(stats_sep)

        stats_header = QtWidgets.QLabel("Estatísticas do Dataset")
        stats_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        stats_layout.addWidget(stats_header)

        stats_form = QtWidgets.QFormLayout()
        stats_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        stats_form.setSpacing(4)

        self.lbl_n_imgs = QtWidgets.QLabel("—")
        self.lbl_n_ids = QtWidgets.QLabel("—")
        self.lbl_emb_cols = QtWidgets.QLabel("—")

        for lbl in (self.lbl_n_imgs, self.lbl_n_ids, self.lbl_emb_cols):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        self.lbl_emb_cols.setWordWrap(True)

        for text, value_lbl in [
            ("Amostras:", self.lbl_n_imgs),
            ("Identidades:", self.lbl_n_ids),
            ("Embeddings:", self.lbl_emb_cols),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            stats_form.addRow(lbl, value_lbl)

        stats_layout.addLayout(stats_form)
        props_layout.addWidget(self._stats_section)

        # Detalhes do pipeline (visível apenas para PIPELINE_CONFIG)
        self._pipeline_section = QtWidgets.QWidget()
        pipe_layout = QtWidgets.QVBoxLayout(self._pipeline_section)
        pipe_layout.setContentsMargins(0, 4, 0, 0)
        pipe_layout.setSpacing(4)

        pipe_sep = QtWidgets.QFrame()
        pipe_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        pipe_sep.setStyleSheet("color: #ddd;")
        pipe_layout.addWidget(pipe_sep)

        pipe_header = QtWidgets.QLabel("Detalhes do Pipeline")
        pipe_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        pipe_layout.addWidget(pipe_header)

        pipe_form = QtWidgets.QFormLayout()
        pipe_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        pipe_form.setSpacing(4)

        self.lbl_pipe_steps = QtWidgets.QLabel("—")
        self.lbl_pipe_steps.setWordWrap(True)
        self.lbl_pipe_output_dim = QtWidgets.QLabel("—")
        self.lbl_pipe_flags = QtWidgets.QLabel("—")

        for lbl in (self.lbl_pipe_steps, self.lbl_pipe_output_dim, self.lbl_pipe_flags):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        for text, value_lbl in [
            ("Etapas:", self.lbl_pipe_steps),
            ("Dim. saída:", self.lbl_pipe_output_dim),
            ("Flags:", self.lbl_pipe_flags),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            pipe_form.addRow(lbl, value_lbl)

        pipe_layout.addLayout(pipe_form)
        props_layout.addWidget(self._pipeline_section)

        props_layout.addStretch()

        props_scroll.setWidget(props_container)
        self.tabs.addTab(props_scroll, "Propriedades")

        # ── Tab 2: Utilização ────────────────────────────────────────
        usage_container = QtWidgets.QWidget()
        usage_layout = QtWidgets.QVBoxLayout(usage_container)
        usage_layout.setContentsMargins(8, 8, 8, 8)
        usage_layout.setSpacing(4)

        self.usage_list = QtWidgets.QListWidget()
        self.usage_list.setStyleSheet(
            "font-size: 11px; border: none; background: white;"
        )
        self.usage_list.setAlternatingRowColors(True)
        usage_layout.addWidget(self.usage_list)

        self.tabs.addTab(usage_container, "Utilização")

        layout.addWidget(self.tabs)

    # ── Public API ────────────────────────────────────────────────────────────

    def show_resource(
        self,
        resource: StrategyResource,
        project_root: Optional[Path],
        definitions: Optional[List[LRStrategyDefinition]] = None,
        embedding_dimensions: Optional[int] = None,
    ) -> None:
        # Metadados
        type_label = resource.resource_type.value
        self.lbl_id.setText(resource.resource_id)
        self.lbl_type.setText(type_label)

        abs_path: Optional[Path] = None

        if resource.artifact_ref:
            self.lbl_path.setText(resource.artifact_ref.path)

            if project_root:
                abs_path = project_root / resource.artifact_ref.path

            # Hash: usa o armazenado; se None, computa do disco
            hash_val = resource.artifact_ref.hash
            if not hash_val and abs_path and abs_path.exists():
                hash_val = self._hash_service.calculate_file_hash(abs_path)
            if hash_val:
                self.lbl_hash.setText(hash_val[:16] + "…")
            else:
                self.lbl_hash.setText("—")

            if abs_path and abs_path.exists():
                size_bytes = abs_path.stat().st_size
                self.lbl_size.setText(self._format_size(size_bytes))
            else:
                self.lbl_size.setText("arquivo não encontrado")
        else:
            self.lbl_path.setText("(sem arquivo vinculado)")
            self.lbl_hash.setText("—")
            self.lbl_size.setText("—")

        # Estatísticas do dataset
        is_dataset = resource.resource_type == StrategyResourceType.DATASET
        self._stats_section.setVisible(is_dataset)
        if is_dataset and abs_path:
            self._load_stats(abs_path)
        else:
            self._clear_stats()

        # Detalhes do pipeline
        is_pipeline = resource.resource_type == StrategyResourceType.PIPELINE_CONFIG
        self._pipeline_section.setVisible(is_pipeline)
        if is_pipeline and abs_path:
            self._load_pipeline_details(abs_path, embedding_dimensions)
        else:
            self._clear_pipeline()

        # Uso nas estratégias
        self.usage_list.clear()
        if definitions:
            for defn in definitions:
                for instr in defn.instructions:
                    role = None
                    if resource.resource_id in getattr(instr, "input_resource_ids", []):
                        role = "input"
                    if resource.resource_id in getattr(instr, "output_resource_ids", []):
                        role = "output" if role is None else "input/output"
                    if role:
                        self.usage_list.addItem(
                            f"{defn.name}  ›  {instr.instruction_type.value}  [{role}]"
                        )

        if self.usage_list.count() == 0:
            self.usage_list.addItem("(não utilizado em nenhuma instrução)")

        # Atualiza badge de contagem na aba Utilização
        usage_count = self.usage_list.count()
        if usage_count == 1 and self.usage_list.item(0).text().startswith("(não"):
            self.tabs.setTabText(1, "Utilização")
        else:
            self.tabs.setTabText(1, f"Utilização ({usage_count})")

    def clear(self) -> None:
        for lbl in (self.lbl_id, self.lbl_type, self.lbl_path, self.lbl_hash, self.lbl_size):
            lbl.setText("—")
        self._clear_stats()
        self._clear_pipeline()
        self._stats_section.setVisible(False)
        self._pipeline_section.setVisible(False)
        self.usage_list.clear()
        self.tabs.setTabText(1, "Utilização")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_stats(self, dataset_abs_path: Path) -> None:
        """Tenta carregar stats do JSON paralelo ao parquet."""
        stats_path = dataset_abs_path.parent / (dataset_abs_path.stem + "_stats.json")
        if stats_path.exists():
            try:
                data = json.loads(stats_path.read_text(encoding="utf-8"))
                self.lbl_n_imgs.setText(str(data.get("n_imgs", "—")))
                self.lbl_n_ids.setText(str(data.get("n_ids", "—")))
                emb_cols = data.get("embedding_columns", [])
                if emb_cols:
                    lines = [
                        f"{col.get('name', f'col_{i}')} ({col.get('dim', '?')}d)"
                        for i, col in enumerate(emb_cols)
                    ]
                    self.lbl_emb_cols.setText("\n".join(lines))
                else:
                    self.lbl_emb_cols.setText("—")
                return
            except Exception:
                pass
        self._clear_stats()

    def _clear_stats(self) -> None:
        self.lbl_n_imgs.setText("—")
        self.lbl_n_ids.setText("—")
        self.lbl_emb_cols.setText("—")

    def _load_pipeline_details(
        self, pipeline_abs_path: Path, embedding_dimensions: Optional[int]
    ) -> None:
        """Carrega e exibe detalhes de um pipeline config JSON."""
        try:
            cfg = TransformationPipelineConfig.from_json(pipeline_abs_path)
        except Exception:
            self.lbl_pipe_steps.setText("erro ao carregar")
            self.lbl_pipe_output_dim.setText("—")
            self.lbl_pipe_flags.setText("—")
            return

        # Etapas: reutiliza to_summary() existente
        summary = cfg.to_summary()
        self.lbl_pipe_steps.setText(summary.text)

        # Dimensão de saída
        output_dim = cfg.output_dimension
        if output_dim is not None:
            self.lbl_pipe_output_dim.setText(f"{output_dim}D")
        elif embedding_dimensions is not None:
            self.lbl_pipe_output_dim.setText(
                f"{embedding_dimensions}D (embedding original)"
            )
        else:
            self.lbl_pipe_output_dim.setText("definida em runtime")

        # Flags
        flags = []
        if cfg.has_pca:
            flags.append("PCA/Whitening")
        if cfg.has_lda:
            flags.append("LDA")
        self.lbl_pipe_flags.setText(", ".join(flags) if flags else "nenhuma redução")

    def _clear_pipeline(self) -> None:
        self.lbl_pipe_steps.setText("—")
        self.lbl_pipe_output_dim.setText("—")
        self.lbl_pipe_flags.setText("—")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes //= 1024
        return f"{size_bytes:.1f} TB"
