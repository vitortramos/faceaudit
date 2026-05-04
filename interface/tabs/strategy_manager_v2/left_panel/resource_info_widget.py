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

    def __init__(self, parent=None, compact: bool = False):
        """
        Args:
            compact: quando True, oculta a aba "Utilização", a barra de
                abas e a seção de metadados básicos (ID/Tipo/Path/Hash/
                Tamanho), deixando apenas as seções ricas
                (Stats / Pipeline / Sim Model / Calib Model / Scores).
                Útil pra embedar em outros painéis (ex: 3ª coluna do
                editor de instruções) sem duplicar info que já está visível.
        """
        super().__init__(parent)
        self._hash_service = HashService()
        self._compact = bool(compact)
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

        # Encapsula o form em um widget próprio para que possa ser ocultado
        # em modo compact (a 3ª coluna do editor já mostra ID/Tipo na linha
        # do nome do recurso e Path/Hash/Tamanho são pouco úteis lá).
        self._basic_info_section = QtWidgets.QWidget()
        basic_layout = QtWidgets.QVBoxLayout(self._basic_info_section)
        basic_layout.setContentsMargins(0, 0, 0, 0)
        basic_layout.setSpacing(0)
        basic_layout.addLayout(form)
        props_layout.addWidget(self._basic_info_section)
        if self._compact:
            self._basic_info_section.setVisible(False)

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
        self.lbl_balance = QtWidgets.QLabel("—")
        self.lbl_samples_per_id = QtWidgets.QLabel("—")

        for lbl in (
            self.lbl_n_imgs, self.lbl_n_ids, self.lbl_emb_cols,
            self.lbl_balance, self.lbl_samples_per_id,
        ):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        self.lbl_emb_cols.setWordWrap(True)
        self.lbl_balance.setWordWrap(True)
        self.lbl_samples_per_id.setWordWrap(True)

        for text, value_lbl in [
            ("Amostras:", self.lbl_n_imgs),
            ("Identidades:", self.lbl_n_ids),
            ("Embeddings:", self.lbl_emb_cols),
            ("Balanceamento:", self.lbl_balance),
            ("Amostras/ID:", self.lbl_samples_per_id),
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

        # Detalhes do Modelo de Similaridade (visível apenas para
        # TRAINED_SIMILARITY_MODEL)
        self._sim_model_section = QtWidgets.QWidget()
        sim_layout = QtWidgets.QVBoxLayout(self._sim_model_section)
        sim_layout.setContentsMargins(0, 4, 0, 0)
        sim_layout.setSpacing(4)

        sim_sep = QtWidgets.QFrame()
        sim_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sim_sep.setStyleSheet("color: #ddd;")
        sim_layout.addWidget(sim_sep)

        sim_header = QtWidgets.QLabel("Detalhes do Modelo de Similaridade")
        sim_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        sim_layout.addWidget(sim_header)

        sim_form = QtWidgets.QFormLayout()
        sim_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        sim_form.setSpacing(4)

        self.lbl_sim_type = QtWidgets.QLabel("—")
        self.lbl_sim_input_dim = QtWidgets.QLabel("—")
        self.lbl_sim_latent_dim = QtWidgets.QLabel("—")
        self.lbl_sim_extra = QtWidgets.QLabel("—")
        self.lbl_sim_extra.setWordWrap(True)

        for lbl in (self.lbl_sim_type, self.lbl_sim_input_dim,
                    self.lbl_sim_latent_dim, self.lbl_sim_extra):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        for text, value_lbl in [
            ("Tipo:", self.lbl_sim_type),
            ("Dim. entrada:", self.lbl_sim_input_dim),
            ("Dim. latente:", self.lbl_sim_latent_dim),
            ("Extra:", self.lbl_sim_extra),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            sim_form.addRow(lbl, value_lbl)

        sim_layout.addLayout(sim_form)
        props_layout.addWidget(self._sim_model_section)

        # Detalhes do Modelo de Calibração (visível apenas para
        # TRAINED_CALIBRATION_MODEL)
        self._calib_model_section = QtWidgets.QWidget()
        calib_layout = QtWidgets.QVBoxLayout(self._calib_model_section)
        calib_layout.setContentsMargins(0, 4, 0, 0)
        calib_layout.setSpacing(4)

        calib_sep = QtWidgets.QFrame()
        calib_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        calib_sep.setStyleSheet("color: #ddd;")
        calib_layout.addWidget(calib_sep)

        calib_header = QtWidgets.QLabel("Detalhes do Modelo de Calibração")
        calib_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        calib_layout.addWidget(calib_header)

        calib_form = QtWidgets.QFormLayout()
        calib_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        calib_form.setSpacing(4)

        self.lbl_calib_type = QtWidgets.QLabel("—")
        self.lbl_calib_hyperparams = QtWidgets.QLabel("—")
        self.lbl_calib_hyperparams.setWordWrap(True)
        self.lbl_calib_clipping = QtWidgets.QLabel("—")
        self.lbl_calib_extra = QtWidgets.QLabel("—")
        self.lbl_calib_extra.setWordWrap(True)

        for lbl in (self.lbl_calib_type, self.lbl_calib_hyperparams,
                    self.lbl_calib_clipping, self.lbl_calib_extra):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        for text, value_lbl in [
            ("Tipo:", self.lbl_calib_type),
            ("Hyperparâmetros:", self.lbl_calib_hyperparams),
            ("Clipping log-LR:", self.lbl_calib_clipping),
            ("Extra:", self.lbl_calib_extra),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            calib_form.addRow(lbl, value_lbl)

        calib_layout.addLayout(calib_form)
        props_layout.addWidget(self._calib_model_section)

        # Detalhes de Score Dataset (visível para SIMILARITY_SCORES e
        # CALIBRATED_SCORES)
        self._scores_section = QtWidgets.QWidget()
        scores_layout = QtWidgets.QVBoxLayout(self._scores_section)
        scores_layout.setContentsMargins(0, 4, 0, 0)
        scores_layout.setSpacing(4)

        scores_sep = QtWidgets.QFrame()
        scores_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        scores_sep.setStyleSheet("color: #ddd;")
        scores_layout.addWidget(scores_sep)

        scores_header = QtWidgets.QLabel("Detalhes do Score Dataset")
        scores_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        scores_layout.addWidget(scores_header)

        scores_form = QtWidgets.QFormLayout()
        scores_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        scores_form.setSpacing(4)

        self.lbl_scores_total = QtWidgets.QLabel("—")
        self.lbl_scores_genuine = QtWidgets.QLabel("—")
        self.lbl_scores_impostor = QtWidgets.QLabel("—")
        self.lbl_scores_identities = QtWidgets.QLabel("—")
        self.lbl_scores_calibrated = QtWidgets.QLabel("—")

        for lbl in (self.lbl_scores_total, self.lbl_scores_genuine,
                    self.lbl_scores_impostor, self.lbl_scores_identities,
                    self.lbl_scores_calibrated):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        for text, value_lbl in [
            ("Total de pares:", self.lbl_scores_total),
            ("Genuínos (Hp):", self.lbl_scores_genuine),
            ("Impostores (Hd):", self.lbl_scores_impostor),
            ("Identidades:", self.lbl_scores_identities),
            ("Calibrados (log-LR):", self.lbl_scores_calibrated),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            scores_form.addRow(lbl, value_lbl)

        scores_layout.addLayout(scores_form)
        props_layout.addWidget(self._scores_section)

        # Detalhes de Pipeline TREINADO (visível para TRAINED_PIPELINE)
        self._pipeline_trained_section = QtWidgets.QWidget()
        pt_layout = QtWidgets.QVBoxLayout(self._pipeline_trained_section)
        pt_layout.setContentsMargins(0, 4, 0, 0)
        pt_layout.setSpacing(4)

        pt_sep = QtWidgets.QFrame()
        pt_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        pt_sep.setStyleSheet("color: #ddd;")
        pt_layout.addWidget(pt_sep)

        pt_header = QtWidgets.QLabel("Detalhes do Pipeline Treinado")
        pt_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        pt_layout.addWidget(pt_header)

        pt_form = QtWidgets.QFormLayout()
        pt_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        pt_form.setSpacing(4)

        self.lbl_pt_mode = QtWidgets.QLabel("—")
        self.lbl_pt_n_steps = QtWidgets.QLabel("—")
        self.lbl_pt_steps = QtWidgets.QLabel("—")
        self.lbl_pt_fitted = QtWidgets.QLabel("—")
        for lbl in (self.lbl_pt_mode, self.lbl_pt_n_steps, self.lbl_pt_steps, self.lbl_pt_fitted):
            lbl.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_pt_steps.setWordWrap(True)

        for text, value_lbl in [
            ("Modo:", self.lbl_pt_mode),
            ("Nº de steps:", self.lbl_pt_n_steps),
            ("Cadeia:", self.lbl_pt_steps),
            ("Estado:", self.lbl_pt_fitted),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            pt_form.addRow(lbl, value_lbl)

        pt_layout.addLayout(pt_form)
        props_layout.addWidget(self._pipeline_trained_section)

        # Detalhes de Hubness Report (visível para HUBNESS_REPORT)
        self._hubness_report_section = QtWidgets.QWidget()
        hr_layout = QtWidgets.QVBoxLayout(self._hubness_report_section)
        hr_layout.setContentsMargins(0, 4, 0, 0)
        hr_layout.setSpacing(4)

        hr_sep = QtWidgets.QFrame()
        hr_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        hr_sep.setStyleSheet("color: #ddd;")
        hr_layout.addWidget(hr_sep)

        hr_header = QtWidgets.QLabel("Detalhes do Relatório de Hubness")
        hr_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        hr_layout.addWidget(hr_header)

        hr_form = QtWidgets.QFormLayout()
        hr_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        hr_form.setSpacing(4)

        self.lbl_hr_dataset = QtWidgets.QLabel("—")
        self.lbl_hr_n_samples = QtWidgets.QLabel("—")
        self.lbl_hr_k = QtWidgets.QLabel("—")
        self.lbl_hr_extras = QtWidgets.QLabel("—")
        for lbl in (self.lbl_hr_dataset, self.lbl_hr_n_samples, self.lbl_hr_k, self.lbl_hr_extras):
            lbl.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_hr_extras.setWordWrap(True)

        for text, value_lbl in [
            ("Dataset:", self.lbl_hr_dataset),
            ("Amostras:", self.lbl_hr_n_samples),
            ("K (vizinhos):", self.lbl_hr_k),
            ("Estudos auxiliares:", self.lbl_hr_extras),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            hr_form.addRow(lbl, value_lbl)

        hr_layout.addLayout(hr_form)
        props_layout.addWidget(self._hubness_report_section)

        # Detalhes de Hubness Ablation (visível para HUBNESS_ABLATION_RESULT)
        self._hubness_ablation_section = QtWidgets.QWidget()
        ha_layout = QtWidgets.QVBoxLayout(self._hubness_ablation_section)
        ha_layout.setContentsMargins(0, 4, 0, 0)
        ha_layout.setSpacing(4)

        ha_sep = QtWidgets.QFrame()
        ha_sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        ha_sep.setStyleSheet("color: #ddd;")
        ha_layout.addWidget(ha_sep)

        ha_header = QtWidgets.QLabel("Detalhes da Ablação")
        ha_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        ha_layout.addWidget(ha_header)

        ha_form = QtWidgets.QFormLayout()
        ha_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ha_form.setSpacing(4)

        self.lbl_ha_n_scenarios = QtWidgets.QLabel("—")
        self.lbl_ha_k = QtWidgets.QLabel("—")
        self.lbl_ha_scenarios = QtWidgets.QLabel("—")
        for lbl in (self.lbl_ha_n_scenarios, self.lbl_ha_k, self.lbl_ha_scenarios):
            lbl.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_ha_scenarios.setWordWrap(True)

        for text, value_lbl in [
            ("Nº de cenários:", self.lbl_ha_n_scenarios),
            ("K (vizinhos):", self.lbl_ha_k),
            ("Cenários:", self.lbl_ha_scenarios),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            ha_form.addRow(lbl, value_lbl)

        ha_layout.addLayout(ha_form)
        props_layout.addWidget(self._hubness_ablation_section)

        props_layout.addStretch()

        props_scroll.setWidget(props_container)
        self.tabs.addTab(props_scroll, "Propriedades")

        # ── Tab 2: Utilização ────────────────────────────────────────
        # Em modo compact, a aba "Utilização" não é adicionada — quem
        # embeda esse widget na 3ª coluna do editor não precisa dessa
        # informação (e ela exigiria as definitions do projeto, que o
        # caller compact não passa).
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

        if not self._compact:
            self.tabs.addTab(usage_container, "Utilização")
        else:
            # Esconde a barra de abas (sobra só o conteúdo da aba 0).
            self.tabs.tabBar().setVisible(False)

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

        # Estatísticas do dataset — também aplicáveis a INTERMEDIATE_DATASET
        # (datasets produzidos por instruções; o JSON paralelo é gerado sob
        # demanda na primeira seleção).
        is_dataset = resource.resource_type in (
            StrategyResourceType.DATASET,
            StrategyResourceType.INTERMEDIATE_DATASET,
        )
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

        # Detalhes do modelo de similaridade treinado
        is_sim_model = resource.resource_type == StrategyResourceType.TRAINED_SIMILARITY_MODEL
        self._sim_model_section.setVisible(is_sim_model)
        if is_sim_model:
            self._load_similarity_model_details(resource, abs_path)
        else:
            self._clear_similarity_model()

        # Detalhes do modelo de calibração treinado
        is_calib_model = resource.resource_type == StrategyResourceType.TRAINED_CALIBRATION_MODEL
        self._calib_model_section.setVisible(is_calib_model)
        if is_calib_model:
            self._load_calibration_model_details(resource, abs_path)
        else:
            self._clear_calibration_model()

        # Detalhes de score dataset (similarity / calibrated)
        is_scores = resource.resource_type in (
            StrategyResourceType.SIMILARITY_SCORES,
            StrategyResourceType.CALIBRATED_SCORES,
        )
        self._scores_section.setVisible(is_scores)
        if is_scores:
            self._load_scores_details(resource, abs_path)
        else:
            self._clear_scores()

        # Detalhes do pipeline TREINADO (TRAINED_PIPELINE)
        is_pipeline_trained = resource.resource_type == StrategyResourceType.TRAINED_PIPELINE
        self._pipeline_trained_section.setVisible(is_pipeline_trained)
        if is_pipeline_trained:
            self._load_pipeline_trained_details(resource, abs_path)
        else:
            self._clear_pipeline_trained()

        # Detalhes do relatório de hubness (HUBNESS_REPORT)
        is_hubness_report = resource.resource_type == StrategyResourceType.HUBNESS_REPORT
        self._hubness_report_section.setVisible(is_hubness_report)
        if is_hubness_report:
            self._load_hubness_report_details(resource, abs_path)
        else:
            self._clear_hubness_report()

        # Detalhes da ablação (HUBNESS_ABLATION_RESULT)
        is_hubness_ablation = resource.resource_type == StrategyResourceType.HUBNESS_ABLATION_RESULT
        self._hubness_ablation_section.setVisible(is_hubness_ablation)
        if is_hubness_ablation:
            self._load_hubness_ablation_details(resource, abs_path)
        else:
            self._clear_hubness_ablation()

        # Uso nas estratégias — só aplicável quando a aba Utilização existe
        # (modo não-compact, embedado no left panel principal).
        if not self._compact:
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
        self._clear_similarity_model()
        self._clear_calibration_model()
        self._clear_scores()
        self._stats_section.setVisible(False)
        self._pipeline_section.setVisible(False)
        self._sim_model_section.setVisible(False)
        self._calib_model_section.setVisible(False)
        self._scores_section.setVisible(False)
        self._pipeline_trained_section.setVisible(False)
        self._hubness_report_section.setVisible(False)
        self._hubness_ablation_section.setVisible(False)
        self._clear_pipeline_trained()
        self._clear_hubness_report()
        self._clear_hubness_ablation()
        # Em modo compact a aba "Utilização" não foi adicionada, então o
        # usage_list é um widget órfão e pode já ter sido GC'd pelo Qt.
        if not self._compact:
            self.usage_list.clear()
            self.tabs.setTabText(1, "Utilização")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_stats(self, dataset_abs_path: Path) -> None:
        """
        Tenta carregar stats do JSON paralelo ao parquet. Se não existir
        (caso típico de INTERMEDIATE_DATASET produzido por instruções),
        computa via pyarrow metadata e cacheia o JSON ao lado do parquet.

        Stale-cache check: se o parquet for mais recente que o sidecar,
        recomputa (caso típico: re-execução de instrução com parâmetros
        diferentes, mantendo o mesmo path do output).
        """
        stats_path = dataset_abs_path.parent / (dataset_abs_path.stem + "_stats.json")
        is_stale = False
        if stats_path.exists():
            try:
                if dataset_abs_path.stat().st_mtime > stats_path.stat().st_mtime + 1.0:
                    is_stale = True
            except Exception:
                pass

        # Cache antigo sem balance_metrics → força recomputação (mesmo padrão
        # usado em _load_scores_details para n_identities).
        needs_recompute = not stats_path.exists() or is_stale
        if not needs_recompute:
            try:
                _peek = json.loads(stats_path.read_text(encoding="utf-8"))
                if "balance_metrics" not in _peek:
                    needs_recompute = True
            except Exception:
                needs_recompute = True

        if needs_recompute:
            self._compute_and_cache_stats(dataset_abs_path, stats_path)

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
                self._render_balance(data.get("balance_metrics"))
                return
            except Exception:
                pass
        self._clear_stats()

    def _compute_and_cache_stats(self, parquet_path: Path, stats_path: Path) -> None:
        """
        Computa stats leves do parquet (n_imgs, n_ids, colunas de embedding e
        suas dimensões) via pyarrow metadata, sem materializar embeddings, e
        salva em JSON ao lado. Silencioso em caso de falha — stats são opcionais.
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            from faceaudit.services.dataset_stats_service import DatasetStatsService

            svc = DatasetStatsService()
            stats = svc._read_parquet_stats(parquet_path)

            embedding_columns = []
            try:
                schema = pq.ParquetFile(parquet_path).schema_arrow
                emb_indices = [
                    i for i, field in enumerate(schema)
                    if pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
                ]
                if emb_indices:
                    pq_file = pq.ParquetFile(parquet_path)
                    for batch in pq_file.iter_batches(batch_size=1):
                        for pos, idx in enumerate(emb_indices):
                            val = batch.column(idx)[0].as_py()
                            dim = len(val) if (val is not None and hasattr(val, "__len__")) else 0
                            col_name = schema.field(idx).name
                            embedding_columns.append({
                                "index": pos,
                                "parquet_index": idx,
                                "name": col_name,
                                "dim": dim,
                            })
                        break
            except Exception:
                pass

            payload = {
                "n_imgs": stats.n_imgs,
                "n_ids": stats.n_ids,
                "embedding_columns": embedding_columns,
            }

            # Métricas de balanceamento (None quando inaplicável: 0 ou 1
            # identidade). Persistido como dict simples — JSON-friendly.
            bm = stats.balance_metrics
            if bm is not None:
                payload["balance_metrics"] = {
                    "n_identities": bm.n_identities,
                    "min_samples": bm.min_samples,
                    "max_samples": bm.max_samples,
                    "mean_samples": bm.mean_samples,
                    "median_samples": bm.median_samples,
                    "std_samples": bm.std_samples,
                    "cv": bm.cv,
                    "imbalance_ratio": (
                        bm.imbalance_ratio
                        if bm.imbalance_ratio != float("inf") else None
                    ),
                    "verdict": bm.verdict,
                }
            else:
                payload["balance_metrics"] = None

            stats_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _clear_stats(self) -> None:
        self.lbl_n_imgs.setText("—")
        self.lbl_n_ids.setText("—")
        self.lbl_emb_cols.setText("—")
        self.lbl_balance.setText("—")
        self.lbl_samples_per_id.setText("—")

    def _render_balance(self, bm: Optional[dict]) -> None:
        """
        Pinta as duas linhas de balanceamento a partir do dict serializado
        no sidecar (ou None quando inaplicável: 0 ou 1 identidade).
        """
        if not bm:
            self.lbl_balance.setText("n/a (≤ 1 identidade)")
            self.lbl_samples_per_id.setText("—")
            return

        verdict = bm.get("verdict") or "—"
        cv = bm.get("cv")
        ratio = bm.get("imbalance_ratio")

        ratio_txt = f"{ratio:.1f}:1" if isinstance(ratio, (int, float)) else "—"
        cv_txt = f"{cv:.3f}" if isinstance(cv, (int, float)) else "—"
        self.lbl_balance.setText(f"{verdict}  (CV={cv_txt}, max/min={ratio_txt})")

        mn = bm.get("min_samples")
        mx = bm.get("max_samples")
        mean = bm.get("mean_samples")
        median = bm.get("median_samples")
        if mn is not None and mx is not None and mean is not None and median is not None:
            self.lbl_samples_per_id.setText(
                f"{int(mn)}–{int(mx)}  (média {float(mean):.1f}, mediana {float(median):.1f})"
            )
        else:
            self.lbl_samples_per_id.setText("—")

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

    def _load_similarity_model_details(
        self, resource: StrategyResource, abs_path: Optional[Path]
    ) -> None:
        """
        Carrega e exibe detalhes de um modelo de similaridade treinado:
        tipo, dimensão de entrada (D), dimensão latente (PLDA) e
        informações extras como nº de iterações EM ou amostras vistas.

        Usa o payload em memória se disponível; caso contrário, carrega o
        pickle do disco via ResourceLoaderService (lazy).
        """
        payload = resource.payload
        if payload is None and abs_path and abs_path.exists():
            try:
                from faceaudit.services.strategy_resource_loader_service import (
                    StrategyResourceLoaderService,
                )
                # Reusa a instância única se houver, senão cria sob demanda.
                loader = getattr(self, "_loader", None)
                if loader is None:
                    self._loader = StrategyResourceLoaderService(artifact_storage=None)
                    loader = self._loader
                # Hidrata diretamente via _load_pickle (sabemos que é pickle).
                payload = loader._load_pickle(abs_path)
            except Exception:
                payload = None

        if payload is None:
            self.lbl_sim_type.setText("erro ao carregar")
            self.lbl_sim_input_dim.setText("—")
            self.lbl_sim_latent_dim.setText("—")
            self.lbl_sim_extra.setText("—")
            return

        type_name = type(payload).__name__
        self.lbl_sim_type.setText(type_name)

        input_dim = getattr(payload, "input_dimension", None)
        if input_dim is not None:
            self.lbl_sim_input_dim.setText(f"{int(input_dim)}D")
        else:
            self.lbl_sim_input_dim.setText("—")

        # latent_dim só existe no PLDA; em outros modelos fica oculto.
        cfg = getattr(payload, "config", None)
        latent_dim = getattr(cfg, "latent_dim", None) if cfg is not None else None
        if latent_dim is not None:
            self.lbl_sim_latent_dim.setText(f"{int(latent_dim)}D")
        else:
            self.lbl_sim_latent_dim.setText("n/a")

        # Linha extra: nº de iterações EM (PLDA), nº de amostras vistas
        # (Cosine), modo PLDA, etc — varia por implementação.
        extras = []
        em_iters = getattr(cfg, "em_iterations", None) if cfg is not None else None
        if em_iters is not None:
            extras.append(f"EM iters={em_iters}")
        plda_mode = getattr(cfg, "plda_mode", None) if cfg is not None else None
        if plda_mode is not None:
            mode_str = (
                plda_mode.value if hasattr(plda_mode, "value") else str(plda_mode)
            )
            extras.append(f"modo={mode_str}")
        n_seen = getattr(payload, "n_samples_seen", None)
        if n_seen is not None:
            extras.append(f"N treino={n_seen:,}".replace(",", "."))
        is_fitted = getattr(payload, "is_fitted", None)
        if is_fitted is False:
            extras.append("⚠ não treinado")
        self.lbl_sim_extra.setText(" | ".join(extras) if extras else "—")

    def _clear_similarity_model(self) -> None:
        self.lbl_sim_type.setText("—")
        self.lbl_sim_input_dim.setText("—")
        self.lbl_sim_latent_dim.setText("—")
        self.lbl_sim_extra.setText("—")

    def _load_calibration_model_details(
        self, resource: StrategyResource, abs_path: Optional[Path]
    ) -> None:
        """
        Carrega e exibe detalhes de um modelo de calibração treinado:
        tipo (LogisticRegression / KDE), hyperparâmetros principais, faixa
        de clipping de log-LR (se aplicável) e métricas auxiliares como
        coeficiente do LogReg ou bandwidth efetivo do KDE.

        Usa o payload em memória se disponível; caso contrário, hidrata do
        disco via _load_pickle.
        """
        payload = resource.payload
        if payload is None and abs_path and abs_path.exists():
            try:
                from faceaudit.services.strategy_resource_loader_service import (
                    StrategyResourceLoaderService,
                )
                loader = getattr(self, "_loader", None)
                if loader is None:
                    self._loader = StrategyResourceLoaderService(artifact_storage=None)
                    loader = self._loader
                payload = loader._load_pickle(abs_path)
            except Exception:
                payload = None

        if payload is None:
            self.lbl_calib_type.setText("erro ao carregar")
            self.lbl_calib_hyperparams.setText("—")
            self.lbl_calib_clipping.setText("—")
            self.lbl_calib_extra.setText("—")
            return

        type_name = type(payload).__name__
        self.lbl_calib_type.setText(type_name)

        cfg = getattr(payload, "config", None)
        is_logreg = "LogisticRegression" in type_name
        is_kde = "KDE" in type_name

        # Hyperparâmetros principais por família.
        hyperparams = []
        if is_logreg and cfg is not None:
            best_c = getattr(payload, "best_C_", None)
            if best_c is not None:
                hyperparams.append(f"C (efetivo)={best_c:g}")
            elif getattr(cfg, "C", None) is not None:
                hyperparams.append(f"C={cfg.C:g}")
            if getattr(cfg, "use_gridsearch", False):
                cv = getattr(cfg, "cv_folds", None)
                scoring = getattr(cfg, "scoring", None)
                scoring_str = (
                    scoring.value if hasattr(scoring, "value") else str(scoring)
                )
                hyperparams.append(f"GridSearchCV (cv={cv}, scoring={scoring_str})")
            hyperparams.append("class_weight=balanced (Brümmer π=0.5)")
        elif is_kde and cfg is not None:
            kernel = getattr(cfg, "kernel", None)
            if kernel:
                hyperparams.append(f"kernel={kernel}")
            kde_gen = getattr(payload, "kde_genuine", None)
            kde_imp = getattr(payload, "kde_impostor", None)
            bw_gen = (
                getattr(kde_gen, "bandwidth", None)
                if kde_gen is not None else None
            )
            bw_imp = (
                getattr(kde_imp, "bandwidth", None)
                if kde_imp is not None else None
            )
            if bw_gen is not None:
                hyperparams.append(f"bw_gen={bw_gen:.4f}")
            if bw_imp is not None:
                hyperparams.append(f"bw_imp={bw_imp:.4f}")
            if getattr(cfg, "use_gridsearch", False):
                hyperparams.append("GridSearch (bandwidth)")
        self.lbl_calib_hyperparams.setText(
            " | ".join(hyperparams) if hyperparams else "—"
        )

        # Clipping de log-LR (KDE expõe min/max no config; LogReg não tem).
        if is_kde and cfg is not None:
            min_lr = getattr(cfg, "min_log_lr", None)
            max_lr = getattr(cfg, "max_log_lr", None)
            if min_lr is not None and max_lr is not None:
                self.lbl_calib_clipping.setText(f"[{min_lr:g}, {max_lr:g}]")
            else:
                self.lbl_calib_clipping.setText("—")
        else:
            self.lbl_calib_clipping.setText("n/a")

        # Linha extra: para LogReg, coef + intercept; para KDE, n samples
        # vistas no fit; aviso se não treinado.
        extras = []
        is_fitted = getattr(payload, "is_fitted", None)
        if is_fitted is False:
            extras.append("⚠ não treinado")

        if is_logreg:
            sk = getattr(payload, "logreg", None)
            try:
                if sk is not None and getattr(sk, "coef_", None) is not None:
                    coef = float(sk.coef_[0][0])
                    bias = float(sk.intercept_[0])
                    extras.append(f"slope={coef:.4f}, intercept={bias:.4f}")
            except Exception:
                pass
        elif is_kde:
            n_gen = getattr(payload, "n_genuine_seen", None)
            n_imp = getattr(payload, "n_impostor_seen", None)
            if n_gen is not None and n_imp is not None:
                extras.append(
                    f"N_gen={n_gen:,} | N_imp={n_imp:,}".replace(",", ".")
                )
            fisher_eps = getattr(cfg, "fisher_epsilon", None) if cfg else None
            if fisher_eps is not None:
                extras.append(f"fisher_eps={fisher_eps:g}")

        self.lbl_calib_extra.setText(" | ".join(extras) if extras else "—")

    def _clear_calibration_model(self) -> None:
        self.lbl_calib_type.setText("—")
        self.lbl_calib_hyperparams.setText("—")
        self.lbl_calib_clipping.setText("—")
        self.lbl_calib_extra.setText("—")

    def _load_scores_details(
        self, resource: StrategyResource, abs_path: Optional[Path]
    ) -> None:
        """
        Mostra contagem total/gen/imp e indicador de log-LR para
        SIMILARITY_SCORES e CALIBRATED_SCORES.

        Estratégia de leitura (em ordem de preferência):
        1. payload em memória → usa total_genuines / total_impostors (cached).
        2. JSON sidecar `<stem>_scores_stats.json` ao lado do parquet.
        3. pyarrow.read_table com colunas mínimas (`same_person`, `log_lr` se
           houver) — fast path sem hidratar o dataset inteiro. Cacheia em
           sidecar JSON para o próximo acesso.
        """
        # Caminho 1: payload já em memória.
        payload = resource.payload
        if payload is not None:
            ds_md = getattr(payload, "metadata", None)
            is_casework = bool(isinstance(ds_md, dict) and ds_md.get("is_casework"))

            n_total = len(getattr(payload, "items", []) or [])
            if is_casework:
                # Casework: total real, gen/imp irrelevantes.
                n_gen = n_imp = None
            else:
                n_gen = getattr(payload, "total_genuines", None)
                n_imp = getattr(payload, "total_impostors", None)
                if n_gen is not None and n_imp is not None:
                    n_total = (n_gen or 0) + (n_imp or 0)
            has_log_lr = self._payload_has_log_lr(payload)

            # Identidades únicas: itera samples coletando probe/reference identity.
            n_ids_total = n_probe_ids = n_ref_ids = None
            try:
                items = getattr(payload, "items", None) or []
                probe_set = set()
                ref_set = set()
                for s in items:
                    pid = getattr(getattr(s, "probe", None), "identity", None)
                    rid = getattr(getattr(s, "reference", None), "identity", None)
                    if pid is not None:
                        probe_set.add(pid)
                    if rid is not None:
                        ref_set.add(rid)
                n_probe_ids = len(probe_set)
                n_ref_ids = len(ref_set)
                n_ids_total = len(probe_set | ref_set)
            except Exception:
                pass

            self._render_scores_counts(
                n_total, n_gen, n_imp, has_log_lr,
                n_identities=n_ids_total,
                n_probe_ids=n_probe_ids,
                n_ref_ids=n_ref_ids,
                is_casework=is_casework,
            )
            return

        # Caminho 2/3: precisa do parquet.
        if not abs_path or not abs_path.exists():
            self.lbl_scores_total.setText("(artefato ausente)")
            self.lbl_scores_genuine.setText("—")
            self.lbl_scores_impostor.setText("—")
            self.lbl_scores_calibrated.setText("—")
            return

        # Sidecar de cache.
        cache_path = abs_path.parent / (abs_path.stem + "_scores_stats.json")
        if cache_path.exists():
            try:
                # Stale-cache check: invalida o cache quando o parquet foi
                # reescrito após a última computação dos stats (caso típico:
                # re-execução de PAIR_MINING com max_global_pairs diferente,
                # mantendo o mesmo resource_id/path do output).
                parquet_mtime = abs_path.stat().st_mtime
                cache_mtime = cache_path.stat().st_mtime
                if parquet_mtime > cache_mtime + 1.0:  # 1s de folga
                    raise ValueError("cache mais antigo que o parquet")

                data = json.loads(cache_path.read_text(encoding="utf-8"))
                # Garante que o cache antigo (sem identidades) seja revalidado
                # — se faltar a chave, força recomputação caindo fora.
                if "n_identities" in data:
                    self._render_scores_counts(
                        data.get("n_total"),
                        data.get("n_genuine"),
                        data.get("n_impostor"),
                        bool(data.get("has_log_lr", False)),
                        n_identities=data.get("n_identities"),
                        n_probe_ids=data.get("n_probe_ids"),
                        n_ref_ids=data.get("n_ref_ids"),
                        is_casework=bool(data.get("is_casework", False)),
                    )
                    return
            except Exception:
                pass

        # Computa lendo apenas as colunas mínimas.
        try:
            import pyarrow.parquet as pq

            schema_columns = set()
            try:
                schema_columns = set(pq.ParquetFile(abs_path).schema_arrow.names)
            except Exception:
                schema_columns = set()

            cols_to_read = []
            if "same_person" in schema_columns:
                cols_to_read.append("same_person")
            elif "label" in schema_columns:
                cols_to_read.append("label")
            else:
                self.lbl_scores_total.setText("(coluna de classe ausente)")
                return

            has_probe_id_col = "probe_identity" in schema_columns
            has_ref_id_col = "reference_identity" in schema_columns
            if has_probe_id_col:
                cols_to_read.append("probe_identity")
            if has_ref_id_col:
                cols_to_read.append("reference_identity")

            has_log_lr_col = "log_lr" in schema_columns

            tbl = pq.read_table(abs_path, columns=cols_to_read)
            n_total = tbl.num_rows

            if "same_person" in cols_to_read:
                arr = tbl.column("same_person").to_pylist()
                n_gen = sum(1 for v in arr if v is True)
                n_imp = sum(1 for v in arr if v is False)
                # Detecção de casework: nenhum sample com ground truth
                # → todos same_person == None.
                is_casework = (n_total > 0 and n_gen == 0 and n_imp == 0)
            else:
                arr = tbl.column("label").to_pylist()
                # label==-1 (sentinela) significa same_person=None (casework)
                n_gen = sum(1 for v in arr if int(v) == 1)
                n_imp = sum(1 for v in arr if int(v) == 0)
                n_unknown = sum(1 for v in arr if int(v) == -1)
                is_casework = (n_total > 0 and n_unknown == n_total)

            # Em casework, gen/imp deixam de fazer sentido — não exibir.
            if is_casework:
                n_gen = None
                n_imp = None

            n_ids_total = n_probe_ids = n_ref_ids = None
            try:
                probe_set = (
                    set(tbl.column("probe_identity").to_pylist())
                    if has_probe_id_col else set()
                )
                ref_set = (
                    set(tbl.column("reference_identity").to_pylist())
                    if has_ref_id_col else set()
                )
                probe_set.discard(None)
                ref_set.discard(None)
                if probe_set or ref_set:
                    n_probe_ids = len(probe_set)
                    n_ref_ids = len(ref_set)
                    n_ids_total = len(probe_set | ref_set)
            except Exception:
                pass

            self._render_scores_counts(
                n_total, n_gen, n_imp, has_log_lr_col,
                n_identities=n_ids_total,
                n_probe_ids=n_probe_ids,
                n_ref_ids=n_ref_ids,
                is_casework=is_casework,
            )

            # Cacheia.
            try:
                cache_path.write_text(
                    json.dumps({
                        "n_total": int(n_total),
                        "n_genuine": int(n_gen) if n_gen is not None else None,
                        "n_impostor": int(n_imp) if n_imp is not None else None,
                        "has_log_lr": bool(has_log_lr_col),
                        "n_identities": (
                            int(n_ids_total) if n_ids_total is not None else None
                        ),
                        "n_probe_ids": (
                            int(n_probe_ids) if n_probe_ids is not None else None
                        ),
                        "n_ref_ids": (
                            int(n_ref_ids) if n_ref_ids is not None else None
                        ),
                        "is_casework": bool(is_casework),
                    }),
                    encoding="utf-8",
                )
            except Exception:
                pass

        except Exception:
            self.lbl_scores_total.setText("erro ao ler parquet")
            self.lbl_scores_genuine.setText("—")
            self.lbl_scores_impostor.setText("—")
            self.lbl_scores_calibrated.setText("—")

    def _payload_has_log_lr(self, payload) -> bool:
        """True se ao menos um sample do dataset tem `metadata['log_lr']`."""
        try:
            items = getattr(payload, "items", []) or []
            for s in items[:1000]:
                md = getattr(s, "metadata", None)
                if isinstance(md, dict) and md.get("log_lr") is not None:
                    return True
        except Exception:
            pass
        return False

    def _render_scores_counts(
        self,
        n_total: Optional[int],
        n_gen: Optional[int],
        n_imp: Optional[int],
        has_log_lr: bool,
        n_identities: Optional[int] = None,
        n_probe_ids: Optional[int] = None,
        n_ref_ids: Optional[int] = None,
        is_casework: bool = False,
    ) -> None:
        def fmt(n):
            if n is None:
                return "—"
            return f"{int(n):,}".replace(",", ".")

        if is_casework:
            # Modo casework: substitui rótulo "Total de pares" e suprime
            # gen/imp (que são sempre 0 sem ground truth).
            self.lbl_scores_total.setText(
                f"{fmt(n_total)} (caso forense — sem ground truth)"
            )
            self.lbl_scores_genuine.setText("n/a (casework)")
            self.lbl_scores_impostor.setText("n/a (casework)")
        else:
            self.lbl_scores_total.setText(fmt(n_total))

            if n_total and n_gen is not None and n_total > 0:
                pct_gen = 100.0 * n_gen / n_total
                self.lbl_scores_genuine.setText(f"{fmt(n_gen)}  ({pct_gen:.2f}%)")
            else:
                self.lbl_scores_genuine.setText(fmt(n_gen))

            if n_total and n_imp is not None and n_total > 0:
                pct_imp = 100.0 * n_imp / n_total
                self.lbl_scores_impostor.setText(f"{fmt(n_imp)}  ({pct_imp:.2f}%)")
            else:
                self.lbl_scores_impostor.setText(fmt(n_imp))

            # Razão imp:gen — útil para entender desbalanceamento (típico cross-comp).
            if n_gen and n_imp and n_gen > 0:
                ratio = n_imp / n_gen
                extra = f"  (razão imp:gen = {ratio:.0f}:1)"
                self.lbl_scores_impostor.setText(
                    self.lbl_scores_impostor.text() + extra
                )

        # Identidades únicas. Se probe e reference compartilham o mesmo
        # conjunto, mostra só o total. Se diferem (cross-comparison entre
        # datasets distintos), mostra a discriminação.
        if n_identities is not None:
            label = fmt(n_identities) + " únicas"
            if (
                n_probe_ids is not None
                and n_ref_ids is not None
                and n_probe_ids != n_ref_ids
            ):
                label += (
                    f"  (probe: {fmt(n_probe_ids)} / "
                    f"reference: {fmt(n_ref_ids)})"
                )
            self.lbl_scores_identities.setText(label)
        else:
            self.lbl_scores_identities.setText("—")

        if has_log_lr:
            self.lbl_scores_calibrated.setText("sim — coluna log_lr presente")
        else:
            self.lbl_scores_calibrated.setText("não (raw scores)")

    def _clear_scores(self) -> None:
        self.lbl_scores_total.setText("—")
        self.lbl_scores_genuine.setText("—")
        self.lbl_scores_impostor.setText("—")
        self.lbl_scores_identities.setText("—")
        self.lbl_scores_calibrated.setText("—")

    # ── TRAINED_PIPELINE ────────────────────────────────────────────────────

    def _load_pipeline_trained_details(
        self,
        resource: StrategyResource,
        abs_path: Optional[Path],
    ) -> None:
        payload = getattr(resource, "payload", None)
        if payload is None and abs_path and abs_path.exists():
            try:
                from faceaudit.services.strategy_resource_loader_service import (
                    StrategyResourceLoaderService,
                )
                loader = getattr(self, "_loader", None)
                if loader is None:
                    self._loader = StrategyResourceLoaderService(artifact_storage=None)
                    loader = self._loader
                payload = loader._load_pickle(abs_path)
            except Exception:
                payload = None

        if payload is None:
            self._clear_pipeline_trained()
            return

        mode = getattr(payload, "mode", None)
        steps = getattr(payload, "steps", []) or []
        fitted = getattr(payload, "_fitted", None)

        mode_str = mode.to_str() if (mode is not None and hasattr(mode, "to_str")) else (
            str(mode) if mode is not None else "—"
        )
        self.lbl_pt_mode.setText(mode_str)
        self.lbl_pt_n_steps.setText(str(len(steps)))

        names = []
        for step in steps:
            name = getattr(step, "name", None) or type(step).__name__
            names.append(str(name))
        self.lbl_pt_steps.setText(" → ".join(names) if names else "(sem steps)")

        if fitted is True:
            self.lbl_pt_fitted.setText("✅ treinado")
        elif fitted is False:
            self.lbl_pt_fitted.setText("⚠ não treinado")
        else:
            self.lbl_pt_fitted.setText("—")

    def _clear_pipeline_trained(self) -> None:
        self.lbl_pt_mode.setText("—")
        self.lbl_pt_n_steps.setText("—")
        self.lbl_pt_steps.setText("—")
        self.lbl_pt_fitted.setText("—")

    # ── HUBNESS_REPORT ─────────────────────────────────────────────────────

    def _load_hubness_report_details(
        self,
        resource: StrategyResource,
        abs_path: Optional[Path],
    ) -> None:
        payload = getattr(resource, "payload", None)
        if payload is None and abs_path and abs_path.exists():
            try:
                from faceaudit.services.strategy_resource_loader_service import (
                    StrategyResourceLoaderService,
                )
                loader = getattr(self, "_loader", None)
                if loader is None:
                    self._loader = StrategyResourceLoaderService(artifact_storage=None)
                    loader = self._loader
                payload = loader._load_pickle(abs_path)
            except Exception:
                payload = None

        if payload is None:
            self._clear_hubness_report()
            return

        self.lbl_hr_dataset.setText(str(getattr(payload, "dataset_name", "") or "—"))
        self.lbl_hr_n_samples.setText(str(getattr(payload, "n_samples", 0) or "—"))
        self.lbl_hr_k.setText(str(getattr(payload, "k_neighbors", 0) or "—"))

        extras = []
        if getattr(payload, "stratification", None):
            extras.append(f"{len(payload.stratification)} faixas yaw")
        if getattr(payload, "paired_study", None) is not None:
            extras.append("estudo pareado")
        if getattr(payload, "quality_diagnostics", None) is not None:
            extras.append("diagnóstico de qualidade")
        if getattr(payload, "optimization_study", None) is not None:
            extras.append("sweet spot")
        self.lbl_hr_extras.setText(", ".join(extras) if extras else "(somente análise base)")

    def _clear_hubness_report(self) -> None:
        self.lbl_hr_dataset.setText("—")
        self.lbl_hr_n_samples.setText("—")
        self.lbl_hr_k.setText("—")
        self.lbl_hr_extras.setText("—")

    # ── HUBNESS_ABLATION_RESULT ────────────────────────────────────────────

    def _load_hubness_ablation_details(
        self,
        resource: StrategyResource,
        abs_path: Optional[Path],
    ) -> None:
        payload = getattr(resource, "payload", None)
        if payload is None and abs_path and abs_path.exists():
            try:
                from faceaudit.services.strategy_resource_loader_service import (
                    StrategyResourceLoaderService,
                )
                loader = getattr(self, "_loader", None)
                if loader is None:
                    self._loader = StrategyResourceLoaderService(artifact_storage=None)
                    loader = self._loader
                payload = loader._load_pickle(abs_path)
            except Exception:
                payload = None

        if payload is None:
            self._clear_hubness_ablation()
            return

        scenarios = getattr(payload, "scenarios", []) or []
        self.lbl_ha_n_scenarios.setText(str(len(scenarios)))
        self.lbl_ha_k.setText(str(getattr(payload, "k_neighbors", 0) or "—"))

        names = []
        for s in scenarios:
            sn = getattr(s, "scenario_name", None)
            mx = getattr(s, "max_nk", None)
            if sn is not None and mx is not None:
                names.append(f"{sn} (max Nk={mx})")
            elif sn is not None:
                names.append(str(sn))
        self.lbl_ha_scenarios.setText("; ".join(names) if names else "(sem cenários)")

    def _clear_hubness_ablation(self) -> None:
        self.lbl_ha_n_scenarios.setText("—")
        self.lbl_ha_k.setText("—")
        self.lbl_ha_scenarios.setText("—")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes //= 1024
        return f"{size_bytes:.1f} TB"
