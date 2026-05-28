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

    # Disparado quando o usuário clica em "Abrir no Gerenciador" no
    # painel de propriedades de um DATASET primário. O controller V2
    # propaga até MainGUIController, que troca de aba e carrega o
    # dataset. Path é absoluto.
    edit_dataset_in_manager_requested = QtCore.pyqtSignal(str)

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
        # WA_DontShowOnScreen impede o Qt de exibir este widget como
        # janela top-level caso seja criado sem parent legítimo (callers
        # antigos podem ter feito isso). A flag é automaticamente
        # neutralizada quando o widget vai ser mostrado dentro da árvore
        # do parent legítimo. Crítico pra evitar o flash "janelinha
        # piscando" durante criação lazy de cards de instrução.
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True
        )
        self._hash_service = HashService()
        self._compact = bool(compact)
        self._init_ui()
        self.clear()

    def showEvent(self, event):
        # Remove a flag uma vez que o widget está sendo realmente
        # exibido dentro do parent — daí em diante comportamento normal.
        if self.testAttribute(
            QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen
        ):
            self.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False
            )
        super().showEvent(event)

    def _add_section_header(
        self,
        layout: QtWidgets.QVBoxLayout,
        text: str,
    ) -> None:
        """Adiciona separador HLine + título bold ao layout da seção.

        No-op em modo compact — em embeddings (3ª coluna do editor) o tipo
        do recurso já é evidente pelo chip `[TYPE]` no header do row, então
        repetir "Detalhes do ..." apenas polui visualmente.
        """
        if self._compact:
            return
        sep = QtWidgets.QFrame(parent=self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ddd;")
        layout.addWidget(sep)
        header = QtWidgets.QLabel(text)
        header.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #444;"
        )
        layout.addWidget(header)

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── QTabWidget (somente modo não-compact) ────────────────────
        # Em compact (embedado no OutputRowWidget) não criamos tabs/scroll:
        # QScrollArea com setWidgetResizable(True) força o inner widget a
        # esticar verticalmente pra preencher o viewport, gerando "espaço
        # vazio" perceptível no card. Sem scroll area, o props_container
        # respeita seu sizeHint natural.
        if not self._compact:
            self.tabs = QtWidgets.QTabWidget(parent=self)
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

        # ── Container das Propriedades (sempre criado) ───────────────
        props_container = QtWidgets.QWidget(parent=self)
        props_layout = QtWidgets.QVBoxLayout(props_container)
        props_layout.setContentsMargins(8, 8, 8, 8)
        props_layout.setSpacing(6)

        # Placeholder mostrado quando não há recurso selecionado — evita
        # que o usuário veja uma fileira de "—" nos campos da Propriedades.
        # Toggle de visibilidade controlado por `show_resource`/`clear`.
        self._empty_placeholder = QtWidgets.QLabel(
            "Selecione um recurso para ver as propriedades."
        )
        self._empty_placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._empty_placeholder.setWordWrap(True)
        self._empty_placeholder.setStyleSheet(
            "color: #888; font-style: italic; font-size: 12px; padding: 16px;"
        )
        props_layout.addWidget(self._empty_placeholder)

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
        # Em modo compact a linha do ID é omitida — o ID já aparece como
        # nome do recurso no header do OutputRowWidget que embarca este widget.
        rows = [
            ("Tipo:", self.lbl_type),
            ("Path:", self.lbl_path),
            ("Hash:", self.lbl_hash),
            ("Tamanho:", self.lbl_size),
        ]
        if not self._compact:
            rows.insert(0, ("ID:", self.lbl_id))
        for text, value_lbl in rows:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            form.addRow(lbl, value_lbl)

        self._basic_info_section = QtWidgets.QWidget(parent=self)
        basic_layout = QtWidgets.QVBoxLayout(self._basic_info_section)
        basic_layout.setContentsMargins(0, 0, 0, 0)
        basic_layout.setSpacing(4)
        basic_layout.addLayout(form)

        # Botão "Abrir no Gerenciador de Datasets" — visível só pra
        # DATASET PRIMARY com arquivo existente no projeto. Modo compact
        # esconde (a 3ª coluna do editor é informativa, não navegacional).
        self.btn_open_in_dataset_manager = QtWidgets.QPushButton(
            "📂 Abrir no Gerenciador de Datasets"
        )
        self.btn_open_in_dataset_manager.setStyleSheet(
            "QPushButton {"
            "  border: 1px solid #1565c0; border-radius: 4px;"
            "  background: #e3f2fd; color: #1565c0;"
            "  padding: 4px 10px; font-size: 11px;"
            "}"
            "QPushButton:hover { background: #bbdefb; }"
        )
        self.btn_open_in_dataset_manager.setToolTip(
            "Abre o dataset no Gerenciador de Datasets. Modificações "
            "salvas lá são automaticamente refletidas no projeto."
        )
        self.btn_open_in_dataset_manager.clicked.connect(
            self._on_open_in_dataset_manager
        )
        self.btn_open_in_dataset_manager.setVisible(False)
        basic_layout.addWidget(self.btn_open_in_dataset_manager)
        # Path absoluto atual exposto pra `_on_open_in_dataset_manager`.
        self._current_resource_abs_path: Optional[Path] = None

        props_layout.addWidget(self._basic_info_section)

        # Estatísticas do dataset (visível apenas para DATASET)
        self._stats_section = QtWidgets.QWidget(parent=self)
        stats_layout = QtWidgets.QVBoxLayout(self._stats_section)
        stats_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        stats_layout.setSpacing(4)

        self._add_section_header(stats_layout, "Estatísticas do Dataset")

        stats_form = QtWidgets.QFormLayout()
        stats_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        stats_form.setSpacing(4)

        self.lbl_n_imgs = QtWidgets.QLabel("—")
        self.lbl_n_ids = QtWidgets.QLabel("—")
        self.lbl_emb_cols = QtWidgets.QLabel("—")
        self.lbl_balance = QtWidgets.QLabel("—")
        self.lbl_samples_per_id = QtWidgets.QLabel("—")
        # PR 7: cobertura de métricas de qualidade lida do column_coverage
        # do sidecar JSON (que por sua vez vem da metadata do parquet pós-PR 3).
        self.lbl_quality_coverage = QtWidgets.QLabel("—")

        for lbl in (
            self.lbl_n_imgs, self.lbl_n_ids, self.lbl_emb_cols,
            self.lbl_balance, self.lbl_samples_per_id,
            self.lbl_quality_coverage,
        ):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        self.lbl_emb_cols.setWordWrap(True)
        self.lbl_balance.setWordWrap(True)
        self.lbl_samples_per_id.setWordWrap(True)
        self.lbl_quality_coverage.setWordWrap(True)

        for text, value_lbl in [
            ("Amostras:", self.lbl_n_imgs),
            ("Identidades:", self.lbl_n_ids),
            ("Embeddings:", self.lbl_emb_cols),
            ("Balanceamento:", self.lbl_balance),
            ("Amostras/ID:", self.lbl_samples_per_id),
            ("Qualidade:", self.lbl_quality_coverage),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            stats_form.addRow(lbl, value_lbl)

        stats_layout.addLayout(stats_form)
        props_layout.addWidget(self._stats_section)

        # Detalhes do pipeline (visível apenas para PIPELINE_CONFIG)
        self._pipeline_section = QtWidgets.QWidget(parent=self)
        pipe_layout = QtWidgets.QVBoxLayout(self._pipeline_section)
        pipe_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        pipe_layout.setSpacing(4)

        self._add_section_header(pipe_layout, "Detalhes do Pipeline")

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
        self._sim_model_section = QtWidgets.QWidget(parent=self)
        sim_layout = QtWidgets.QVBoxLayout(self._sim_model_section)
        sim_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        sim_layout.setSpacing(4)

        self._add_section_header(sim_layout, "Detalhes do Modelo de Similaridade")

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
        self._calib_model_section = QtWidgets.QWidget(parent=self)
        calib_layout = QtWidgets.QVBoxLayout(self._calib_model_section)
        calib_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        calib_layout.setSpacing(4)

        self._add_section_header(calib_layout, "Detalhes do Modelo de Calibração")

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
        self._scores_section = QtWidgets.QWidget(parent=self)
        scores_layout = QtWidgets.QVBoxLayout(self._scores_section)
        scores_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        scores_layout.setSpacing(4)

        self._add_section_header(scores_layout, "Detalhes do Score Dataset")

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

        # Tabela compacta de stats por classe (N, mean, std, P5..P95).
        # Renderizada sob demanda em `_render_class_stats_tables`. Para
        # CALIBRATED_SCORES, mostra DUAS tabelas (raw_score e log-LR);
        # pra SIMILARITY_SCORES (incl. PAIR_MINING), só uma (raw_score).
        # Container vazio até `_load_scores_details` chamar o render.
        self._class_stats_container = QtWidgets.QWidget(parent=self._scores_section)
        self._class_stats_layout = QtWidgets.QVBoxLayout(self._class_stats_container)
        self._class_stats_layout.setContentsMargins(0, 6, 0, 0)
        self._class_stats_layout.setSpacing(4)
        scores_layout.addWidget(self._class_stats_container)

        props_layout.addWidget(self._scores_section)

        # Detalhes de Pipeline TREINADO (visível para TRAINED_PIPELINE)
        self._pipeline_trained_section = QtWidgets.QWidget(parent=self)
        pt_layout = QtWidgets.QVBoxLayout(self._pipeline_trained_section)
        pt_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        pt_layout.setSpacing(4)

        self._add_section_header(pt_layout, "Detalhes do Pipeline Treinado")

        pt_form = QtWidgets.QFormLayout()
        pt_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        pt_form.setSpacing(4)

        self.lbl_pt_mode = QtWidgets.QLabel("—")
        self.lbl_pt_n_steps = QtWidgets.QLabel("—")
        self.lbl_pt_steps = QtWidgets.QLabel("—")
        self.lbl_pt_output_dim = QtWidgets.QLabel("—")
        self.lbl_pt_fitted = QtWidgets.QLabel("—")
        for lbl in (
            self.lbl_pt_mode, self.lbl_pt_n_steps, self.lbl_pt_steps,
            self.lbl_pt_output_dim, self.lbl_pt_fitted,
        ):
            lbl.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_pt_steps.setWordWrap(True)
        # Habilita HTML pra exibir "Source: ... <br> Target: ..." em DUAL.
        self.lbl_pt_steps.setTextFormat(QtCore.Qt.TextFormat.RichText)

        for text, value_lbl in [
            ("Modo:", self.lbl_pt_mode),
            ("Nº de steps:", self.lbl_pt_n_steps),
            ("Cadeia:", self.lbl_pt_steps),
            ("Dim. saída:", self.lbl_pt_output_dim),
            ("Estado:", self.lbl_pt_fitted),
        ]:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(_lbl_style)
            pt_form.addRow(lbl, value_lbl)

        pt_layout.addLayout(pt_form)
        props_layout.addWidget(self._pipeline_trained_section)

        # Detalhes de Hubness Report (visível para HUBNESS_REPORT)
        self._hubness_report_section = QtWidgets.QWidget(parent=self)
        hr_layout = QtWidgets.QVBoxLayout(self._hubness_report_section)
        hr_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        hr_layout.setSpacing(4)

        self._add_section_header(hr_layout, "Detalhes do Relatório de Hubness")

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
        self._hubness_ablation_section = QtWidgets.QWidget(parent=self)
        ha_layout = QtWidgets.QVBoxLayout(self._hubness_ablation_section)
        ha_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        ha_layout.setSpacing(4)

        self._add_section_header(ha_layout, "Detalhes da Ablação")

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

        # Detalhes do Modelo de Tipicidade (visível para TRAINED_TYPICALITY_MODEL)
        self._typicality_section = QtWidgets.QWidget(parent=self)
        ty_layout = QtWidgets.QVBoxLayout(self._typicality_section)
        ty_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        ty_layout.setSpacing(4)

        self._add_section_header(ty_layout, "Detalhes do Modelo de Tipicidade")

        ty_form = QtWidgets.QFormLayout()
        ty_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        ty_form.setSpacing(4)

        self.lbl_ty_mode = QtWidgets.QLabel("—", parent=self._typicality_section)
        self.lbl_ty_k = QtWidgets.QLabel("—", parent=self._typicality_section)
        self.lbl_ty_n = QtWidgets.QLabel("—", parent=self._typicality_section)
        self.lbl_ty_input_faces = QtWidgets.QLabel(
            "—", parent=self._typicality_section,
        )
        self.lbl_ty_singletons = QtWidgets.QLabel(
            "—", parent=self._typicality_section,
        )
        self.lbl_ty_dim = QtWidgets.QLabel("—", parent=self._typicality_section)
        self.lbl_ty_ood = QtWidgets.QLabel("—", parent=self._typicality_section)
        self.lbl_ty_loo_median = QtWidgets.QLabel(
            "—", parent=self._typicality_section,
        )
        self.lbl_ty_loo_range = QtWidgets.QLabel(
            "—", parent=self._typicality_section,
        )

        for lbl in (self.lbl_ty_mode, self.lbl_ty_k,
                    self.lbl_ty_n, self.lbl_ty_input_faces, self.lbl_ty_singletons,
                    self.lbl_ty_dim, self.lbl_ty_ood,
                    self.lbl_ty_loo_median, self.lbl_ty_loo_range):
            lbl.setStyleSheet("font-size: 11px; color: #333;")

        for text, value_lbl in [
            ("Modo de Referência:", self.lbl_ty_mode),
            ("k (vizinhos):", self.lbl_ty_k),
            ("Pontos de referência:", self.lbl_ty_n),
            ("Faces de origem:", self.lbl_ty_input_faces),
            ("Identidades singleton:", self.lbl_ty_singletons),
            ("Dim. do embedding:", self.lbl_ty_dim),
            ("Threshold OOD:", self.lbl_ty_ood),
            ("T_k mediano (LOO):", self.lbl_ty_loo_median),
            ("T_k min / max:", self.lbl_ty_loo_range),
        ]:
            lbl = QtWidgets.QLabel(text, parent=self._typicality_section)
            lbl.setStyleSheet(_lbl_style)
            ty_form.addRow(lbl, value_lbl)

        ty_layout.addLayout(ty_form)
        props_layout.addWidget(self._typicality_section)

        # Detalhes do Resultado de Validação (visível para VALIDATION_RESULT)
        self._validation_result_section = QtWidgets.QWidget(parent=self)
        vr_layout = QtWidgets.QVBoxLayout(self._validation_result_section)
        vr_layout.setContentsMargins(0, 0 if self._compact else 4, 0, 0)
        vr_layout.setSpacing(4)

        self._add_section_header(vr_layout, "Detalhes do Resultado de Validação")

        # Layout em flow horizontal — campos separados por " · " que
        # quebram pra linha de baixo quando estouram a largura. Single
        # QLabel rich text + word wrap dá esse comportamento sem precisar
        # de FlowLayout custom — Qt wraps em separadores de espaço.
        self.lbl_vr_summary = QtWidgets.QLabel("—", parent=self)
        self.lbl_vr_summary.setStyleSheet("font-size: 11px; color: #333;")
        self.lbl_vr_summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_vr_summary.setWordWrap(True)
        self.lbl_vr_summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        vr_layout.addWidget(self.lbl_vr_summary)
        props_layout.addWidget(self._validation_result_section)

        if self._compact:
            # Embeda direto no layout principal — sem QScrollArea, sem
            # QTabWidget, sem addStretch. Cada seção (basic_info, stats,
            # etc.) já tem sizePolicy default (Preferred), o que faria o
            # layout esticar; setamos Maximum no props_container para que
            # ele respeite o sizeHint natural (soma das seções visíveis +
            # spacings) e o card pare exatamente ali.
            props_container.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
            layout.addWidget(props_container)
        else:
            props_layout.addStretch()

            props_scroll = QtWidgets.QScrollArea(parent=self)
            props_scroll.setWidgetResizable(True)
            props_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            props_scroll.setWidget(props_container)
            self.tabs.addTab(props_scroll, "Propriedades")

            # Aba Utilização — só faz sentido com `definitions` passadas
            # pelo left panel principal.
            usage_container = QtWidgets.QWidget(parent=self)
            usage_layout = QtWidgets.QVBoxLayout(usage_container)
            usage_layout.setContentsMargins(8, 8, 8, 8)
            usage_layout.setSpacing(4)

            self.usage_list = QtWidgets.QListWidget(parent=self)
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
        # Esconde o placeholder e mostra a seção básica (Tipo/Path/Hash/
        # Tamanho — ID é só no modo não-compact).
        self._empty_placeholder.setVisible(False)
        self._basic_info_section.setVisible(True)

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

        # Atualiza o botão "Abrir no Gerenciador de Datasets":
        # só visível pra DATASET + PRIMARY + arquivo presente no projeto.
        # Modo compact (3ª coluna do editor) NUNCA mostra — não é fluxo
        # de navegação.
        from faceaudit.enums.resource_category import ResourceCategory as _RC
        from faceaudit.enums.strategy_resource_type import (
            StrategyResourceType as _RT,
        )
        show_open_btn = (
            not self._compact
            and resource.resource_type == _RT.DATASET
            and resource.category == _RC.PRIMARY
            and abs_path is not None
            and abs_path.exists()
        )
        self.btn_open_in_dataset_manager.setVisible(show_open_btn)
        self._current_resource_abs_path = abs_path if show_open_btn else None

        # Estatísticas do dataset — também aplicáveis a INTERMEDIATE_DATASET
        # (datasets produzidos por instruções; o JSON paralelo é gerado sob
        # demanda na primeira seleção).
        is_dataset = resource.resource_type in (
            StrategyResourceType.DATASET,
            StrategyResourceType.DATASET,
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

        # Detalhes do modelo de tipicidade (TRAINED_TYPICALITY_MODEL)
        is_typicality = (
            resource.resource_type == StrategyResourceType.TRAINED_TYPICALITY_MODEL
        )
        self._typicality_section.setVisible(is_typicality)
        if is_typicality:
            self._load_typicality_details(resource, abs_path)
        else:
            self._clear_typicality()

        # Detalhes do resultado de validação (VALIDATION_RESULT)
        is_validation_result = (
            resource.resource_type == StrategyResourceType.VALIDATION_RESULT
        )
        self._validation_result_section.setVisible(is_validation_result)
        if is_validation_result:
            self._load_validation_result_details(resource, abs_path)
        else:
            self._clear_validation_result()

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
                                f"{defn.name}  ›  {instr.instruction_type.label_pt}  [{role}]"
                            )

            if self.usage_list.count() == 0:
                self.usage_list.addItem("(não utilizado em nenhuma instrução)")

            # Atualiza badge de contagem na aba Utilização
            usage_count = self.usage_list.count()
            if usage_count == 1 and self.usage_list.item(0).text().startswith("(não"):
                self.tabs.setTabText(1, "Utilização")
            else:
                self.tabs.setTabText(1, f"Utilização ({usage_count})")

    def _on_open_in_dataset_manager(self) -> None:
        """Slot do botão — emite signal com path absoluto do dataset."""
        if self._current_resource_abs_path is not None:
            self.edit_dataset_in_manager_requested.emit(
                str(self._current_resource_abs_path)
            )

    def clear(self) -> None:
        # Mostra o placeholder e esconde TODAS as seções de conteúdo —
        # estado "sem recurso selecionado". Usuário vê uma mensagem
        # única em vez de uma fileira de "—" nos campos.
        self._empty_placeholder.setVisible(True)
        self._basic_info_section.setVisible(False)
        self.btn_open_in_dataset_manager.setVisible(False)
        self._current_resource_abs_path = None

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
        self._typicality_section.setVisible(False)
        self._validation_result_section.setVisible(False)
        self._clear_pipeline_trained()
        self._clear_hubness_report()
        self._clear_hubness_ablation()
        self._clear_typicality()
        self._clear_validation_result()
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

        # Cache antigo sem balance_metrics ou column_coverage → força
        # recomputação. Cada nova chave no sidecar entra nessa lista de
        # required keys até estabilizar o schema.
        needs_recompute = not stats_path.exists() or is_stale
        if not needs_recompute:
            try:
                _peek = json.loads(stats_path.read_text(encoding="utf-8"))
                required_keys = ("balance_metrics", "column_coverage")
                if any(k not in _peek for k in required_keys):
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
                self._render_quality_coverage(data.get("column_coverage"))
                return
            except Exception:
                pass
        self._clear_stats()

    def _compute_and_cache_stats(self, parquet_path: Path, stats_path: Path) -> None:
        """
        Computa stats leves do parquet via DatasetStatsService (que já cobre
        column_coverage por pyarrow vetorizado) e salva em JSON ao lado do
        parquet. Silencioso em caso de falha — stats são opcionais.

        Em PR 3 isso vai pra metadata do parquet diretamente; o sidecar
        permanece como fallback durante a transição.
        """
        try:
            from faceaudit.services.dataset_stats_service import DatasetStatsService

            svc = DatasetStatsService()
            stats = svc._read_parquet_stats(parquet_path)

            # embedding_columns no formato rico (mantém retrocompat com a UI
            # que já lê esse formato em _load_stats).
            embedding_columns = []
            for pos, (name, cs) in enumerate(stats.column_coverage.items()):
                if cs.dim is None:
                    continue
                embedding_columns.append({
                    "index": pos,
                    "name": name,
                    "dim": cs.dim,
                })

            payload = {
                "n_imgs": stats.n_imgs,
                "n_ids": stats.n_ids,
                "embedding_columns": embedding_columns,
                # NOVO: cobertura por coluna (substitui probing colunar do
                # QualityMetricAvailabilityService em PR 4).
                "column_coverage": {
                    name: cs.to_dict() for name, cs in stats.column_coverage.items()
                },
                "last_modified": stats.last_modified,
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
        self.lbl_quality_coverage.setText("—")

    # Métricas conhecidas que aparecem no resumo (tier "primary" do
    # ColumnSpec.quality_subtype). EMBEDDING_NORM é calculado on-the-fly
    # — sempre disponível, então não precisa de linha aqui.
    _QUALITY_COLUMNS_TO_SHOW = (
        ("fiqa_score", "FIQA"),
        ("ofiq_score", "OFIQ"),
        ("serfiq_score", "SER-FIQ"),
    )

    def _render_quality_coverage(self, coverage: Optional[dict]) -> None:
        """Renderiza um resumo das colunas primárias de quality presentes
        no dataset, com cobertura percentual.

        Lê de `column_coverage` (sidecar JSON pós-PR 2 / metadata pós-PR 3).
        Quando todas as três métricas primárias estão ausentes, mostra
        apenas "Embedding Norm (sempre disponível)".
        """
        if not isinstance(coverage, dict) or not coverage:
            self.lbl_quality_coverage.setText("Embedding Norm (sempre)")
            return

        lines = []
        for col_name, label in self._QUALITY_COLUMNS_TO_SHOW:
            cs = coverage.get(col_name)
            if not cs:
                continue
            non_null = int(cs.get("non_null_count", 0))
            total = int(cs.get("total_count", 0))
            if total == 0 or non_null == 0:
                continue
            pct = (non_null / total) * 100.0
            lines.append(f"{label}: {pct:.1f}% ({non_null}/{total})")

        # EMBEDDING_NORM é sempre disponível (calc on-the-fly).
        lines.append("Embedding Norm: sempre")

        self.lbl_quality_coverage.setText("\n".join(lines))

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

        # Detalhes específicos por variante de PLDA. As variantes two-cov não
        # fazem redução (B e W são full-rank D×D) e não usam EM/plda_mode/
        # latent_dim; mostrar esses campos confundiria o usuário.
        cfg = getattr(payload, "config", None)
        plda_variant = getattr(cfg, "plda_variant", None) if cfg is not None else None
        variant_value = getattr(plda_variant, "value", None)
        is_two_cov = variant_value == "TWO_COVARIANCE"
        is_shrinkage = variant_value == "SHRINKAGE_TWO_COVARIANCE"
        is_any_two_cov = is_two_cov or is_shrinkage
        is_simple_plda = variant_value == "SIMPLE"

        if is_any_two_cov:
            # Em duas covariâncias o "latente" coincide com a entrada (sem
            # redução); mostrar D explícito ajuda a deixar isso claro.
            d_in = getattr(payload, "input_dimension", None) or getattr(
                payload, "trained_dim", None
            )
            if d_in is not None:
                self.lbl_sim_latent_dim.setText(f"{int(d_in)}D (sem redução)")
            else:
                self.lbl_sim_latent_dim.setText("n/a")
        elif is_simple_plda:
            latent_dim = getattr(cfg, "latent_dim", None)
            if latent_dim is not None:
                self.lbl_sim_latent_dim.setText(f"{int(latent_dim)}D")
            else:
                self.lbl_sim_latent_dim.setText("n/a")
        else:
            # Modelos não-PLDA (ex: CosineSimilarityModel) não têm latente.
            self.lbl_sim_latent_dim.setText("n/a")

        extras: list[str] = []
        if is_any_two_cov:
            extras.append(f"variante={variant_value}")
            reg = getattr(cfg, "regularization", None)
            if reg is not None:
                extras.append(f"regularização={float(reg):g}")
            seed = getattr(cfg, "random_seed", None)
            if seed is not None:
                extras.append(f"seed={int(seed)}")

            if is_shrinkage:
                # Metadados de shrinkage vêm do estado treinado (two_cov_state),
                # não do config — refletem o α efetivamente aplicado (especialmente
                # útil em LEDOIT_WOLF onde o config não fixa o valor).
                two_cov_state = getattr(payload, "two_cov_state", None)
                shrink_mode = getattr(two_cov_state, "shrinkage_mode", None)
                shrink_alpha_w = getattr(two_cov_state, "shrinkage_alpha_w", None)
                shrink_target = getattr(two_cov_state, "shrinkage_target", None)
                if shrink_mode is not None:
                    if shrink_alpha_w is not None:
                        extras.append(f"shrinkage={shrink_mode} (α_w={float(shrink_alpha_w):.4f})")
                    else:
                        extras.append(f"shrinkage={shrink_mode}")
                if shrink_target is not None:
                    extras.append(f"target={shrink_target}")
        elif is_simple_plda:
            em_iters = getattr(cfg, "em_iterations", None)
            if em_iters is not None:
                extras.append(f"EM iters={em_iters}")
            plda_mode = getattr(cfg, "plda_mode", None)
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
        is_kde = "KDE" in type_name and "BiGauss" not in type_name
        is_bigauss = "BiGauss" in type_name

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
        elif is_bigauss and cfg is not None:
            target_metric = getattr(cfg, "target_metric", None)
            initial = getattr(cfg, "initial_calibrator", None)
            inv_method = getattr(cfg, "inversion_method", None)
            shrinkage = getattr(cfg, "sigma_shrinkage_factor", None)
            tm_str = target_metric.value if hasattr(target_metric, "value") else str(target_metric)
            init_str = initial.value if hasattr(initial, "value") else str(initial)
            inv_str = inv_method.value if hasattr(inv_method, "value") else str(inv_method)
            hyperparams.append(f"metric={tm_str}")
            hyperparams.append(f"initial={init_str}")
            hyperparams.append(f"inv={inv_str}")
            if shrinkage is not None and shrinkage != 1.0:
                hyperparams.append(f"shrinkage={shrinkage:g}")
        self.lbl_calib_hyperparams.setText(
            " | ".join(hyperparams) if hyperparams else "—"
        )

        # Clipping de log-LR (KDE e BiGauss expõem min/max no config; LogReg não tem).
        # BiGauss tem toggle enable_clipping (default False); KDE sempre clipa.
        if (is_kde or is_bigauss) and cfg is not None:
            bigauss_clip_off = (
                is_bigauss and not getattr(cfg, "enable_clipping", True)
            )
            if bigauss_clip_off:
                self.lbl_calib_clipping.setText("desligado")
            else:
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
            md_z = getattr(payload, "metadata", {}) or {}
            mu = md_z.get("score_mean")
            sigma = md_z.get("score_std")
            if mu is not None and sigma is not None:
                extras.append(f"z-score(μ={mu:.4g}, σ={sigma:.4g})")
        elif is_bigauss:
            sigma = getattr(payload, "sigma_target", None)
            if sigma is not None:
                extras.append(f"σ_alvo={sigma:.4f}")
            md = getattr(payload, "metadata", {}) or {}
            cllr = md.get("cllr_at_fit")
            eer = md.get("eer_at_fit")
            if cllr is not None:
                extras.append(f"Cllr_treino={cllr:.4f}")
            if eer is not None:
                extras.append(f"EER_treino={eer:.4f}")
            n_gen = md.get("n_genuine")
            n_imp = md.get("n_impostor")
            if n_gen is not None and n_imp is not None:
                extras.append(
                    f"N_gen={n_gen:,} | N_imp={n_imp:,}".replace(",", ".")
                )

        # Telemetria CV (in-sample + out-of-fold) — comum aos 3 calibradores.
        # Anexada pelo CALIBRATION_MODEL_TRAINING executor em trained_model.metadata.
        cv_md = getattr(payload, "metadata", {}) or {}
        cllr_train = cv_md.get("calibrator_cllr_train")
        cllr_cv = cv_md.get("calibrator_cllr_cv")
        if cllr_train is not None:
            extras.append(f"Cllr_train={cllr_train:.4f}")
        if cllr_cv is not None:
            mode_str = cv_md.get("cv_split_mode", "?")
            k_eff = cv_md.get("cv_folds_effective", "?")
            extras.append(f"Cllr_CV={cllr_cv:.4f} ({mode_str}, k={k_eff})")
            gap = cv_md.get("calibrator_cv_gap")
            if gap is not None:
                extras.append(f"cv_gap={gap:+.4f}")

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
            # Stats por classe (preferência: sidecar v3 stale-check > live).
            stats = self._load_class_stats(abs_path)
            if stats is None:
                stats = self._compute_class_stats_live(payload)
            self._render_class_stats_tables(stats)
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
                    self._render_class_stats_tables(
                        self._load_class_stats(abs_path)
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
            # Stats por classe vêm do sidecar `_dist.json` (v3+) gerado
            # pela persistência. Se não existir (parquet antigo, sidecar
            # ausente), fica oculta — a tabela só aparece quando o usuário
            # re-executar a instrução com o código novo.
            self._render_class_stats_tables(self._load_class_stats(abs_path))

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
        self._render_class_stats_tables(None)

    # ── class_stats: tabela de N/média/std/percentis por classe ────────────

    def _load_class_stats(self, abs_path: Optional[Path]) -> Optional[dict]:
        """Lê `class_stats` do sidecar `_dist.json` (schema v3+).

        Retorna `None` se o sidecar é ausente, mais antigo que o parquet
        (stale), ou de schema v<3 (sem class_stats). Reaproveita o mesmo
        sidecar usado pelo widget de distribuição — sem custo extra de I/O
        quando o usuário já abriu o plot.
        """
        if not abs_path or not abs_path.exists():
            return None
        sidecar = abs_path.parent / (abs_path.stem + "_dist.json")
        if not sidecar.exists():
            return None
        try:
            if abs_path.stat().st_mtime > sidecar.stat().st_mtime + 1.0:
                return None
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            return data.get("class_stats")
        except Exception:
            return None

    def _compute_class_stats_live(self, payload: Any) -> Optional[dict]:
        """Computa class_stats em tempo real do payload em memória.

        Útil quando o sidecar não existe ainda (recurso recém-produzido,
        ainda não persistido) ou está stale. Iteração O(N) — pra datasets
        com 5M+ samples leva ~1s, aceitável no contexto de info widget.
        """
        if payload is None:
            return None
        try:
            items = getattr(payload, "items", None) or []
            if not items:
                return None
            gen_raw, imp_raw, case_raw = [], [], []
            gen_lr, imp_lr, case_lr = [], [], []
            has_log_lr = False
            for s in items:
                sp = s.same_person
                if sp is True:
                    raw_arr, lr_arr = gen_raw, gen_lr
                elif sp is False:
                    raw_arr, lr_arr = imp_raw, imp_lr
                else:
                    raw_arr, lr_arr = case_raw, case_lr
                raw_arr.append(float(s.raw_score))
                md = getattr(s, "metadata", None)
                if isinstance(md, dict):
                    v = md.get("log_lr")
                    if v is not None:
                        has_log_lr = True
                        lr_arr.append(float(v))
            from faceaudit.models.similarity.similarity_score_dataset import (
                _per_class_stats,
            )
            import numpy as np
            out = {
                "raw": {
                    "gen": _per_class_stats(np.asarray(gen_raw, dtype=np.float64)),
                    "imp": _per_class_stats(np.asarray(imp_raw, dtype=np.float64)),
                    "case": _per_class_stats(np.asarray(case_raw, dtype=np.float64)),
                },
            }
            if has_log_lr:
                out["log_lr"] = {
                    "gen": _per_class_stats(np.asarray(gen_lr, dtype=np.float64)),
                    "imp": _per_class_stats(np.asarray(imp_lr, dtype=np.float64)),
                    "case": _per_class_stats(np.asarray(case_lr, dtype=np.float64)),
                }
            return out
        except Exception:
            return None

    def _render_class_stats_tables(self, class_stats: Optional[dict]) -> None:
        """Reconstrói o conteúdo do container de stats. Limpa antes."""
        # Limpa widgets anteriores.
        while self._class_stats_layout.count():
            item = self._class_stats_layout.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()
        if not class_stats:
            return
        # Itera scales na ordem fixa (raw primeiro, log_lr depois).
        for scale in ("raw", "log_lr"):
            data = class_stats.get(scale)
            if not data:
                continue
            self._class_stats_layout.addWidget(
                self._build_class_stats_table(scale, data)
            )

    def _build_class_stats_table(self, scale: str, data: dict) -> QtWidgets.QWidget:
        """Constrói uma tabela compacta 4-col (Métrica × Gen/Imp/Caso).

        `scale` é "raw" ou "log_lr". `data` é o dict com chaves 'gen',
        'imp', 'case', cada uma com {n, mean, std, p5, p25, p50, p75, p95}.
        """
        container = QtWidgets.QWidget(parent=self._scores_section)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        title_text = "raw_score" if scale == "raw" else "log-LR (calibrado)"
        title = QtWidgets.QLabel(
            f"<b>Em {title_text}</b>", parent=container,
        )
        title.setStyleSheet("font-size: 10px; color: #555;")
        v.addWidget(title)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(8, 0, 0, 0)

        header_style = "font-size: 10px; color: #666; font-weight: bold;"
        cell_style = "font-size: 10px; color: #333;"
        name_style = "font-size: 10px; color: #555;"

        for col, name in enumerate(["", "Gen (Hp)", "Imp (Hd)", "Caso"]):
            lbl = QtWidgets.QLabel(name, parent=container)
            lbl.setStyleSheet(header_style)
            if col > 0:
                lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            grid.addWidget(lbl, 0, col)

        rows = [
            ("N",     "n",   "{:,d}"),
            ("Média", "mean", "{:.3f}"),
            ("Std",   "std",  "{:.3f}"),
            ("P5",    "p5",   "{:.3f}"),
            ("P25",   "p25",  "{:.3f}"),
            ("P50",   "p50",  "{:.3f}"),
            ("P75",   "p75",  "{:.3f}"),
            ("P95",   "p95",  "{:.3f}"),
        ]
        for r, (label, key, fmt) in enumerate(rows, start=1):
            nm = QtWidgets.QLabel(label, parent=container)
            nm.setStyleSheet(name_style)
            grid.addWidget(nm, r, 0)
            for c, cls in enumerate(("gen", "imp", "case"), start=1):
                cls_data = data.get(cls, {}) or {}
                n = cls_data.get("n", 0)
                if n == 0 or key not in cls_data:
                    txt = "—"
                elif key == "n":
                    txt = f"{int(cls_data[key]):,}".replace(",", ".")
                else:
                    txt = fmt.format(float(cls_data[key]))
                val = QtWidgets.QLabel(txt, parent=container)
                val.setStyleSheet(cell_style)
                val.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                grid.addWidget(val, r, c)

        v.addLayout(grid)
        return container

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
        fitted = getattr(payload, "_fitted", None)

        mode_str = mode.to_str() if (mode is not None and hasattr(mode, "to_str")) else (
            str(mode) if mode is not None else "—"
        )
        self.lbl_pt_mode.setText(mode_str)

        # DUAL: source_steps + target_steps separados. LINEAR: steps. A
        # versão antiga lia só `payload.steps` que vinha vazia em DUAL
        # (steps próprio guarda só a chain linear); por isso o painel
        # mostrava "Nº de steps: 0" e "(sem steps)" pra pipelines DUAL.
        is_dual = (
            mode_str == "DUAL"
            or (hasattr(mode, "name") and mode.name == "DUAL")
        )

        def _step_name(step) -> str:
            n = getattr(step, "name", None) or type(step).__name__
            return str(n)

        if is_dual:
            source_steps = getattr(payload, "source_steps", []) or []
            target_steps = getattr(payload, "target_steps", []) or []
            self.lbl_pt_n_steps.setText(
                f"{len(source_steps)} (source) + {len(target_steps)} (target)"
            )
            src_chain = (
                " → ".join(_step_name(s) for s in source_steps)
                or "(vazio)"
            )
            tgt_chain = (
                " → ".join(_step_name(s) for s in target_steps)
                or "(vazio)"
            )
            self.lbl_pt_steps.setText(
                f"<b>Source:</b> {src_chain}<br/><b>Target:</b> {tgt_chain}"
            )

            src_dim = self._compute_steps_output_dim(source_steps)
            tgt_dim = self._compute_steps_output_dim(target_steps)
            if src_dim is not None and tgt_dim is not None:
                if src_dim == tgt_dim:
                    dim_str = f"{src_dim}D"
                else:
                    dim_str = f"source={src_dim}D • target={tgt_dim}D"
            elif src_dim is not None:
                dim_str = f"source={src_dim}D • target=preserva"
            elif tgt_dim is not None:
                dim_str = f"source=preserva • target={tgt_dim}D"
            else:
                dim_str = "preserva (nenhum step redutor)"
            self.lbl_pt_output_dim.setText(dim_str)
        else:
            steps = getattr(payload, "steps", []) or []
            self.lbl_pt_n_steps.setText(str(len(steps)))
            self.lbl_pt_steps.setText(
                " → ".join(_step_name(s) for s in steps) or "(sem steps)"
            )
            dim = self._compute_steps_output_dim(steps)
            self.lbl_pt_output_dim.setText(
                f"{dim}D" if dim is not None else "preserva (nenhum step redutor)"
            )

        if fitted is True:
            self.lbl_pt_fitted.setText("✅ treinado")
        elif fitted is False:
            self.lbl_pt_fitted.setText("⚠ não treinado")
        else:
            self.lbl_pt_fitted.setText("—")

    @staticmethod
    def _compute_steps_output_dim(steps) -> Optional[int]:
        """
        Inspeciona as matrizes fitadas dos steps (LDA, Whitening) pra
        determinar a dim de saída do pipeline. Itera de trás pra frente
        — a última matriz redutora define a dim final.

        Retorna None quando nenhum step da chain altera dim (ex: só
        CORAL + Centering + L2Norm — preserva). Os transformators que
        reduzem dim guardam `_lda_matrix` ou `_whitening_matrix`
        após `fit`, com `.shape[1]` = output_dim.
        """
        import numpy as _np
        for step in reversed(list(steps or [])):
            for attr in ("_lda_matrix", "_whitening_matrix"):
                m = getattr(step, attr, None)
                if isinstance(m, _np.ndarray) and m.ndim == 2:
                    try:
                        return int(m.shape[1])
                    except Exception:
                        continue
        return None

    def _clear_pipeline_trained(self) -> None:
        self.lbl_pt_mode.setText("—")
        self.lbl_pt_n_steps.setText("—")
        self.lbl_pt_steps.setText("—")
        self.lbl_pt_output_dim.setText("—")
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

    # ── TRAINED_TYPICALITY_MODEL ───────────────────────────────────────────

    def _load_typicality_details(
        self,
        resource: StrategyResource,
        abs_path: Optional[Path],
    ) -> None:
        """Popula labels do `_typicality_section` a partir de
        `KNNTypicalityModel.summary()`. Lazy-load via pickle quando o
        payload não está em memória (modelo grande, ~20 MB pra N=10k)."""
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
            self._clear_typicality()
            return

        summary = getattr(payload, "summary", None)
        if not callable(summary):
            self._clear_typicality()
            return

        try:
            s = summary()
        except Exception:
            self._clear_typicality()
            return

        if not isinstance(s, dict) or s.get("status") != "fitted":
            self.lbl_ty_mode.setText("(não treinado)")
            return

        mode = s.get("reference_mode", "PER_FACE")
        mode_label = (
            "Por identidade (centroide)"
            if mode == "PER_IDENTITY" else "Por face"
        )
        self.lbl_ty_mode.setText(mode_label)
        self.lbl_ty_k.setText(str(s.get("k", "—")))

        n = s.get("sample_count")
        unit = "identidades" if mode == "PER_IDENTITY" else "faces"
        self.lbl_ty_n.setText(
            f"{int(n):,} {unit}".replace(",", ".") if n is not None else "—"
        )

        nin = s.get("n_input_faces")
        self.lbl_ty_input_faces.setText(
            f"{int(nin):,} faces".replace(",", ".") if nin is not None else "—"
        )

        if mode == "PER_IDENTITY":
            ns = s.get("n_singleton_identities", 0)
            ns_pct = (
                100.0 * int(ns) / max(int(n), 1) if n is not None else 0.0
            )
            self.lbl_ty_singletons.setText(
                f"{int(ns):,} ({ns_pct:.1f}%)".replace(",", ".")
            )
        else:
            self.lbl_ty_singletons.setText("(n/a no modo Por face)")

        d = s.get("embedding_dim")
        self.lbl_ty_dim.setText(str(int(d)) if d is not None else "—")

        ood = s.get("ood_threshold")
        ood_q = s.get("ood_quantile")
        if ood is not None and ood_q is not None:
            self.lbl_ty_ood.setText(f"{float(ood):.4f}  (quantil {float(ood_q):.4g})")
        elif ood is not None:
            self.lbl_ty_ood.setText(f"{float(ood):.4f}")
        else:
            self.lbl_ty_ood.setText("—")

        med = s.get("loo_T_k_median")
        self.lbl_ty_loo_median.setText(f"{float(med):.4f}" if med is not None else "—")

        lo = s.get("loo_T_k_min")
        hi = s.get("loo_T_k_max")
        if lo is not None and hi is not None:
            self.lbl_ty_loo_range.setText(f"{float(lo):.4f} / {float(hi):.4f}")
        else:
            self.lbl_ty_loo_range.setText("—")

    def _clear_typicality(self) -> None:
        for lbl in (
            self.lbl_ty_mode, self.lbl_ty_k,
            self.lbl_ty_n, self.lbl_ty_input_faces, self.lbl_ty_singletons,
            self.lbl_ty_dim, self.lbl_ty_ood,
            self.lbl_ty_loo_median, self.lbl_ty_loo_range,
        ):
            lbl.setText("—")

    def _load_validation_result_details(
        self, resource: StrategyResource, abs_path: Optional[Path]
    ) -> None:
        """Popula as labels do `_validation_result_section` a partir do
        payload (StrategyValidationResult) ou, se ele estiver ausente,
        do sidecar `<stem>_metrics.json` gravado pelo executor.

        Métricas mostradas: contagens (n_genuines, n_impostors), EER,
        EER threshold, AUC, Cllr, Cllr_min e cal_loss (derivada).
        """
        metrics = self._extract_validation_metrics_dict(resource, abs_path)
        if metrics is None:
            self._clear_validation_result()
            return

        def _fmt_int(v) -> str:
            return f"{int(v):,}".replace(",", ".") if v is not None else "—"

        def _fmt_float(v, fmt: str) -> str:
            return fmt.format(float(v)) if v is not None else "—"

        # Constrói flow HTML "label=value · label=value · …" — uma única
        # QLabel rich-text + wordWrap fluem horizontalmente até estourar a
        # largura, então quebram. Mesmo padrão visual do antigo
        # Summary header do _TabbedPngSnapshotWidget.
        parts: list[str] = []

        def _add(label: str, value: Optional[str]) -> None:
            if value is None or value == "—":
                return
            parts.append(f"<b>{label}</b>={value}")

        n_gen_v = metrics.get("n_genuines")
        n_imp_v = metrics.get("n_impostors")
        if n_gen_v is not None and n_imp_v is not None:
            parts.append(
                f"{_fmt_int(n_gen_v)} genuínos / {_fmt_int(n_imp_v)} impostores"
            )

        _add("EER", _fmt_float(metrics.get("eer"), "{:.2%}"))
        _add("EER thr", _fmt_float(metrics.get("eer_threshold"), "{:.4f}"))
        _add("AUC", _fmt_float(metrics.get("auc"), "{:.4f}"))
        _add("Falso+", _fmt_float(metrics.get("wrong_sign_imp_rate"), "{:.2%}"))
        _add("Falso−", _fmt_float(metrics.get("wrong_sign_gen_rate"), "{:.2%}"))
        _add("Cllr", _fmt_float(metrics.get("cllr"), "{:.4f}"))
        _add("Cllr_min", _fmt_float(metrics.get("cllr_min"), "{:.4f}"))
        _add("cal_loss", _fmt_float(metrics.get("cal_loss"), "{:.4f}"))
        _add("Cllr_affine", _fmt_float(metrics.get("cllr_affine"), "{:.4f}"))

        slope_val = metrics.get("calibration_slope")
        if slope_val is not None:
            arrow = "≈" if 0.95 <= float(slope_val) <= 1.05 else (
                "↑" if float(slope_val) > 1 else "↓"
            )
            parts.append(f"<b>slope</b>={float(slope_val):.3f} {arrow}")

        self.lbl_vr_summary.setText(" &nbsp;·&nbsp; ".join(parts) or "—")

    @staticmethod
    def _extract_validation_metrics_dict(
        resource: StrategyResource, abs_path: Optional[Path]
    ) -> Optional[dict]:
        """Devolve métricas como dict. Prioridade: payload em memória >
        sidecar `_metrics.json`. Retorna None se nenhuma fonte estiver
        disponível.
        """
        payload = resource.payload
        if payload is not None:
            try:
                m = getattr(payload, "metrics", None)
                if m is not None:
                    cllr = getattr(m, "cllr", None)
                    cllr_min = getattr(m, "cllr_min", None)
                    cal_loss = (
                        max(0.0, float(cllr) - float(cllr_min))
                        if cllr is not None and cllr_min is not None
                        else None
                    )
                    return {
                        "n_genuines": int(getattr(payload, "n_genuines", 0) or 0),
                        "n_impostors": int(getattr(payload, "n_impostors", 0) or 0),
                        "eer": getattr(m, "eer", None),
                        "eer_threshold": getattr(m, "eer_threshold", None),
                        "auc": getattr(m, "auc", None),
                        "cllr": cllr,
                        "cllr_min": cllr_min,
                        "cal_loss": cal_loss,
                        # Tier 1 forense extras (podem ser None em pickles antigos)
                        "wrong_sign_gen_rate": getattr(m, "wrong_sign_gen_rate", None),
                        "wrong_sign_imp_rate": getattr(m, "wrong_sign_imp_rate", None),
                        "cllr_affine": getattr(m, "cllr_affine", None),
                        "calibration_slope": getattr(m, "calibration_slope", None),
                    }
            except Exception:
                pass

        # Fallback pro sidecar — escrito pelo executor no persist.
        if abs_path is None or not abs_path.exists():
            return None
        try:
            from faceaudit.models.validation_result_plot import (
                try_load_validation_metrics_sidecar,
            )
            return try_load_validation_metrics_sidecar(abs_path)
        except Exception:
            return None

    def _clear_validation_result(self) -> None:
        self.lbl_vr_summary.setText("—")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes //= 1024
        return f"{size_bytes:.1f} TB"
