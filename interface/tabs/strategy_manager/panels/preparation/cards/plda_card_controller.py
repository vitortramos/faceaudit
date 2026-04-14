from pathlib import Path
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from typing import Optional, Dict, Any, TYPE_CHECKING

# View
from .plda_card_view import PLDACardView

# Enums e Models
from faceaudit.enums.lr_model_type import LRModelType
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.models.plda_config import PLDAConfig
from faceaudit.parameters.plda_parameters import PLDAParameters
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.models.plda_validation_result import ValidationSeverity, PLDAValidationResult
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.models.split_simulation_result import SplitSimulationResult # <--- [NOVO] IMPORT

# IMPORT DO DIALOG
from interface.tabs.strategy_manager.log_progress_dialog.log_progress_dialog import LogProgressDialog

# Controllers Auxiliares
from interface.tabs.strategy_manager.pipeline_creator.pipeline_creator_controller import PipelineCreatorController
from interface.tabs.strategy_manager.panels.preparation.cards.transformation_trajectory_viewer.transformation_trajectory_viewer_controller import TransformationTrajectoryViewerController

# Workers
from interface.tabs.strategy_manager.panels.preparation.workers.plda_preparation_worker import PLDAPreparationWorker
from interface.tabs.strategy_manager.panels.preparation.workers.plda_training_worker import PLDATrainingWorker
from interface.tabs.strategy_manager.panels.preparation.workers.transformation_trajectory_visualization_worker import TransformationTrajectoryVisualizationWorker
from interface.tabs.strategy_manager.workers.plda_split_simulation_worker import PLDASplitSimulationWorker 

if TYPE_CHECKING:
    from faceaudit.manager.plda_manager import PLDAManager
    from faceaudit.models.system_config import AppConfig


class PLDACardController(QtCore.QObject):
    
    request_destruction = QtCore.pyqtSignal(object)
    preparation_finished = QtCore.pyqtSignal()
    
    def __init__(self, view_idx: int, strategy_id: str, manager: 'PLDAManager', global_config: 'AppConfig', parent=None):
        super().__init__(parent)
        self.card_id = view_idx
        self.strategy_id = strategy_id
        self.manager = manager        
        self.global_config = global_config
        
        self.view = PLDACardView(view_idx, parent)
        
        self.prep_worker: Optional[PLDAPreparationWorker] = None
        self.train_worker: Optional[PLDATrainingWorker] = None
        self.viz_worker: Optional[TransformationTrajectoryVisualizationWorker] = None
        self.split_worker: Optional[PLDASplitSimulationWorker] = None
        self.progress_dialog: Optional[LogProgressDialog] = None
        
        # --- TRAVA DE IDENTIDADE VISUAL ---
        nome_formatado = f"ESTRATÉGIA {self.strategy_id.upper()}"
        self.view.lbl_title.setText(nome_formatado)

        # Sequência de Boot
        self._init_data()
        self._connect_signals()
        
        # Mágica de Restauração
        self._sync_ui_from_config()

    def get_widget(self) -> QtWidgets.QWidget:
        return self.view

    def get_config(self) -> PLDAConfig:
        """Devolve o DTO tipado e puro gerado pela interface."""
        return self._build_config_from_ui()

    def update_global_config(self, new_config: 'AppConfig'):
        self.global_config = new_config
        self._init_data() 
        self._trigger_validation()
    
    def _handle_visualization(self):
        self.view.btn_visualize.setEnabled(False)
        self._show_progress("Calculando Trajetória 3D (Isso pode levar alguns instantes)...")
        
        self.viz_worker = TransformationTrajectoryVisualizationWorker(self.manager)
        
        if hasattr(self.viz_worker, 'progress_log'):
            self.viz_worker.progress_log.connect(self._update_progress_log)
            
        self.viz_worker.finished_success.connect(self._on_viz_success)
        self.viz_worker.finished_error.connect(self._on_viz_error)
        self.viz_worker.start()

    def _on_viz_success(self, frames):
        if self.progress_dialog:
            self.progress_dialog.finish(success=True)
            
        self.view.btn_visualize.setEnabled(True)
        try:
            viz_controller = TransformationTrajectoryViewerController(frames, parent=self.view)
            viz_controller.show()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.view, "Erro", str(e))

    def _on_viz_error(self, msg):
        if self.progress_dialog:
            self.progress_dialog.finish(success=False)
        self.view.btn_visualize.setEnabled(True)
        QtWidgets.QMessageBox.critical(self.view, "Erro na Visualização", msg)

    # =========================================================================
    # RESTAURAÇÃO DE ESTADO E INJEÇÃO DE DADOS
    # =========================================================================

    def _set_signals_blocked(self, block: bool):
        """Bloqueia ou desbloqueia a emissão de sinais de todos os inputs da UI."""
        widgets = [
            self.view.combo_strategy, self.view.combo_embedding, self.view.combo_plda_mode,
            self.view.combo_train_data, self.view.combo_calib_data,
            self.view.combo_pipe_source, self.view.combo_pipe_target,
            self.view.combo_split_config, self.view.combo_mining_config,
            self.view.spin_iter, self.view.spin_latent_dim, self.view.spin_logreg_c
        ]
        for widget in widgets:
            widget.blockSignals(block)

    def _inject_saved_data(self, cfg: PLDAConfig):
        """Injeta os valores da configuração do modelo diretamente nos widgets."""
        self.view.combo_strategy.setCurrentText(cfg.strategy.label if hasattr(cfg.strategy, 'label') else str(cfg.strategy))
        self.view.combo_embedding.setCurrentText(cfg.embedding_type.display_name if hasattr(cfg.embedding_type, 'display_name') else str(cfg.embedding_type))
        self.view.combo_plda_mode.setCurrentText(cfg.parameters.plda_mode.label if hasattr(cfg.parameters.plda_mode, 'label') else str(cfg.parameters.plda_mode))

        self.view.spin_iter.setValue(cfg.parameters.em_iterations)
        self.view.spin_latent_dim.setValue(cfg.parameters.latent_dim)
        self.view.spin_logreg_c.setValue(cfg.parameters.logreg_C)

        self._set_combo_safely(self.view.combo_train_data, cfg.source_dataset_name)
        self._set_combo_safely(self.view.combo_calib_data, cfg.target_dataset_name)
        self._set_combo_safely(self.view.combo_pipe_source, cfg.source_pipeline_name)
        self._set_combo_safely(self.view.combo_pipe_target, cfg.target_pipeline_name)
        self._set_combo_safely(self.view.combo_split_config, cfg.split_config_name) 
        self._set_combo_safely(self.view.combo_mining_config, cfg.mining_config_name) 

    def _sync_ui_from_config(self):
        """Orquestra a reconstrução visual do Card a partir dos dados do backend."""
        cfg = self.manager.get_config()
        if not cfg or not isinstance(cfg, PLDAConfig):
            self._on_strategy_changed()
            self._on_dataset_changed()
            self._check_pipeline_ui_logic() 
            self._trigger_validation()
            return

        # 1. Popula as listas disponíveis lendo o Pendrive
        self._init_data()

        # 2. Isola a UI para injeção limpa (Sem disparar validações intermediárias)
        self._set_signals_blocked(True)
        self._inject_saved_data(cfg)
        self._set_signals_blocked(False)
        
        # 3. Restaura os estados secundários dependentes e textos de UI
        self.view._on_plda_mode_changed()
        self.view.update_dataset_labels(self.view.combo_strategy.currentData())
        
        # Chama as funções oficiais que atualizam labels e botões de edição 
        self._on_dataset_changed()
        self._check_pipeline_ui_logic()

        # [NOVO] 4. Restaura visualmente a label de simulação ao carregar o projeto
        sim_result = self.manager.get_split_simulation()
        self._update_simulation_label(sim_result)


    # =========================================================================
    # LÓGICA DE AGRUPAMENTO (MODELOS E UI)
    # =========================================================================

    def _get_clean_combo_text(self, combo: QtWidgets.QComboBox) -> str:
        """Puxa o dado limpo armazenado (sem as indentações ou headers visuais)"""
        idx = combo.currentIndex()
        if idx < 0: return "Selecione..."
        
        data = combo.itemData(idx, QtCore.Qt.ItemDataRole.UserRole)
        if data:
            return str(data)
            
        text = combo.itemText(idx).strip()
        if text.startswith("──") or text == "Selecione...":
            return "Selecione..."
        return text

    def _set_combo_safely(self, combo: QtWidgets.QComboBox, text: str):
        if not text or text == "Selecione...":
            combo.setCurrentIndex(0)
            return
            
        for i in range(combo.count()):
            item_data = combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole)
            if item_data == text or combo.itemText(i).strip() == text.strip():
                combo.setCurrentIndex(i)
                return
                
        combo.addItem(text, text)
        combo.setCurrentIndex(combo.count() - 1)

    def _populate_combo_grouped(self, combo: QtWidgets.QComboBox, global_items: list, local_items: list, default=None):
        combo.blockSignals(True)
        model = QtGui.QStandardItemModel()
        
        empty = QtGui.QStandardItem("Selecione...")
        empty.setData(None, QtCore.Qt.ItemDataRole.UserRole)
        model.appendRow(empty)
        
        def add_group(title, items):
            if not items: return
            header = QtGui.QStandardItem(f"── {title} ──")
            header.setFlags(QtCore.Qt.ItemFlag.NoItemFlags) 
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setForeground(QtCore.Qt.GlobalColor.darkGray)
            model.appendRow(header)
            
            for i in items:
                clean_val = str(i).split(" (")[0] 
                item = QtGui.QStandardItem(f"  {str(i)}")
                item.setData(clean_val, QtCore.Qt.ItemDataRole.UserRole) 
                model.appendRow(item)

        add_group("Projeto Ativo (Local)", local_items)
        add_group("Sistema (Global)", global_items)
        
        combo.setModel(model)
        
        if default:
            self._set_combo_safely(combo, default)
        else:
            combo.setCurrentIndex(0)
            
        combo.blockSignals(False)

    def _list_datasets_grouped(self):
        return self.manager.get_available_datasets()

    def _list_pipelines_grouped(self):
        return self.manager.get_available_pipelines()

    def _init_data(self):
        self.view.combo_strategy.blockSignals(True)
        if self.view.combo_strategy.count() == 0:
            for strat in PLDATrainingStrategy:
                self.view.combo_strategy.addItem(strat.label, strat)
            self.view.combo_strategy.setCurrentIndex(2)
        self.view.combo_strategy.blockSignals(False)

        self.view.combo_embedding.blockSignals(True)
        if self.view.combo_embedding.count() == 0:
            for emb_type in EmbeddingType:
                self.view.combo_embedding.addItem(emb_type.display_name, emb_type)
        self.view.combo_embedding.blockSignals(False)

        self.view.combo_plda_mode.blockSignals(True)
        if self.view.combo_plda_mode.count() == 0:
            for mode in PLDAMode:
                self.view.combo_plda_mode.addItem(mode.label, mode)
            self.view.combo_plda_mode.setCurrentIndex(0) 
        self.view.combo_plda_mode.blockSignals(False)
        self.view._on_plda_mode_changed() 

        global_ds, local_ds = self._list_datasets_grouped()
        global_pipe, local_pipe = self._list_pipelines_grouped()
        global_split, local_split = self.manager.get_available_splits() 
        global_mining, local_mining = self.manager.get_available_mining_configs() 

        self._populate_combo_grouped(self.view.combo_train_data, global_ds, local_ds)
        self._populate_combo_grouped(self.view.combo_calib_data, global_ds, local_ds)
        self._populate_combo_grouped(self.view.combo_pipe_source, global_pipe, local_pipe)
        self._populate_combo_grouped(self.view.combo_pipe_target, global_pipe, local_pipe)
        self._populate_combo_grouped(self.view.combo_split_config, global_split, local_split) 
        self._populate_combo_grouped(self.view.combo_mining_config, global_mining, local_mining) 


    def _refresh_pipelines(self):
        current_src = self._get_clean_combo_text(self.view.combo_pipe_source)
        current_tgt = self._get_clean_combo_text(self.view.combo_pipe_target)

        global_pipe, local_pipe = self._list_pipelines_grouped()

        self._populate_combo_grouped(self.view.combo_pipe_source, global_pipe, local_pipe)
        self._populate_combo_grouped(self.view.combo_pipe_target, global_pipe, local_pipe)

        self._set_combo_safely(self.view.combo_pipe_source, current_src)
        self._set_combo_safely(self.view.combo_pipe_target, current_tgt)
        
        self._check_pipeline_ui_logic()

    # =========================================================================
    # SINAIS E SLOTS
    # =========================================================================

    def _connect_signals(self):
        self.view.remove_requested.connect(lambda: self.request_destruction.emit(self))
        self.view.combo_strategy.currentIndexChanged.connect(self._on_strategy_changed)
        self.view.combo_embedding.currentIndexChanged.connect(self._reset_prep_and_validate)
        
        self.view.combo_train_data.currentTextChanged.connect(self._on_dataset_changed)
        self.view.combo_train_data.currentTextChanged.connect(self._check_pipeline_ui_logic) 
        self.view.combo_calib_data.currentTextChanged.connect(self._on_dataset_changed)
        
        self.view.combo_pipe_source.currentTextChanged.connect(self._check_pipeline_ui_logic)
        self.view.combo_pipe_target.currentTextChanged.connect(self._check_pipeline_ui_logic)
        
        self.view.combo_split_config.currentTextChanged.connect(self._trigger_validation) 
        self.view.combo_mining_config.currentTextChanged.connect(self._trigger_validation) 
        
        self.view.combo_plda_mode.currentIndexChanged.connect(self._trigger_validation)
        self.view.spin_latent_dim.valueChanged.connect(self._trigger_validation)
        self.view.spin_iter.valueChanged.connect(self._trigger_validation)
        self.view.spin_logreg_c.valueChanged.connect(self._trigger_validation)
        
        self.view.run_prep_requested.connect(self._handle_preparation)
        self.view.simulate_split_requested.connect(self._handle_simulate_split) 
        
        self.view.edit_source_pipeline_requested.connect(lambda: self._handle_edit_pipeline(self.view.combo_pipe_source))
        self.view.edit_target_pipeline_requested.connect(lambda: self._handle_edit_pipeline(self.view.combo_pipe_target))
        
        # [NOVO] Conexão dos botões de edição do Grupo 4
        self.view.edit_split_requested.connect(self._handle_edit_split)
        self.view.edit_mining_requested.connect(self._handle_edit_mining)
        
        self.view.train_requested.connect(self._handle_train)
        self.view.visualize_requested.connect(self._handle_visualization)

    def _reset_prep_and_validate(self):
        self.view.btn_visualize.setVisible(False)
        self._trigger_validation()

    def _on_strategy_changed(self):
        plda_training_strategy = self.view.combo_strategy.currentData()
        self.view.update_dataset_labels(plda_training_strategy)
        self._reset_prep_and_validate()

    def _on_dataset_changed(self):
        t_name = self._get_clean_combo_text(self.view.combo_train_data)
        c_name = self._get_clean_combo_text(self.view.combo_calib_data)
        
        st_t = self.manager.get_dataset_stats(t_name)
        st_c = self.manager.get_dataset_stats(c_name)
        
        self.view.set_stats_text(self.view.lbl_stats_train, st_t.n_imgs if st_t else 0, st_t.n_ids if st_t else 0)
        self.view.set_stats_text(self.view.lbl_stats_calib, st_c.n_imgs if st_c else 0, st_c.n_ids if st_c else 0)
        self._reset_prep_and_validate()

    def _check_pipeline_ui_logic(self):
        p_src_name = self._get_clean_combo_text(self.view.combo_pipe_source).split(" (")[0]
        p_tgt_name = self._get_clean_combo_text(self.view.combo_pipe_target).split(" (")[0]
        
        # Controle visual dos botões de edição
        self.view.btn_edit_source.setEnabled(p_src_name != "Selecione...")
        self.view.btn_edit_target.setEnabled(p_tgt_name != "Selecione...")
        
        summary_src = self.manager.get_pipeline_details(p_src_name)
        summary_tgt = self.manager.get_pipeline_details(p_tgt_name)
        
        self.view.lbl_pipe_source_details.setText(summary_src.text)
        self.view.lbl_pipe_target_details.setText(summary_tgt.text)
            
        self._trigger_validation()

    def _trigger_validation(self):
        current_cfg = self._build_config_from_ui()
        self.manager.update_config(current_cfg)
        result: PLDAValidationResult = self.manager.validate_all()
        self._update_validation_labels(result)
        self._update_buttons_state(result)

    def _update_validation_labels(self, res: PLDAValidationResult):
        msg_data = f"✅ Datasets Válidos (Overlap: {res.metrics.get('overlap', 0)})" if res.data_severity == ValidationSeverity.OK else "\n".join(res.messages)
        if res.data_severity == ValidationSeverity.NEUTRAL: msg_data = "ℹ️ Selecione Datasets"
        self._set_label(self.view.lbl_val_data, res.data_severity, msg_data)
        
        msg_pipe = res.pipeline_message or "OK"
        if res.pipeline_severity == ValidationSeverity.NEUTRAL: msg_pipe = "ℹ️ Selecione Pipelines"
        self._set_label(self.view.lbl_val_pipe, res.pipeline_severity, msg_pipe)
        
        self._set_label(self.view.lbl_val_config, res.plda_severity, res.plda_message)

        # Estados lógicos
        is_model_loaded = self.manager.is_model_loaded()
        has_model_disk = self.manager.has_model_on_disk()
        is_prepared = self.manager.is_prepared()
        has_cache = hasattr(self.manager, 'has_trajectory_data') and self.manager.has_trajectory_data()
        has_pipelines = hasattr(self.manager, 'has_fitted_pipelines') and self.manager.has_fitted_pipelines()

        # Prioridade visual
        if is_model_loaded:
            self._set_label(self.view.lbl_val_postprep, ValidationSeverity.OK, "✅ Modelo PLDA treinado e alocado na memória.\nPronto para Inferência Forense!")
        elif has_model_disk:
            self._set_label(self.view.lbl_val_postprep, ValidationSeverity.OK, "✅ Modelo PLDA treinado encontrado no disco.\nPronto para Inferência Forense!")
        elif is_prepared:
            self._set_label(self.view.lbl_val_postprep, ValidationSeverity.OK, "✅ Dados alocados na memória (RAM).\nPronto para Treino e Visualização 3D.")
        elif has_pipelines:
            self._set_label(self.view.lbl_val_postprep, ValidationSeverity.NEUTRAL, "ℹ️ Pipelines treinados e estatísticas encontrados no disco.\nVocê pode Treinar o PLDA ou gerar a trajetória 3D diretamente.")
        elif has_cache:
            self._set_label(self.view.lbl_val_postprep, ValidationSeverity.NEUTRAL, "ℹ️ Projeto carregado do disco. Gráfico 3D disponível via cache.\nExecute a Preparação para treinar um novo modelo.")
        else:
            can_prep = (res.data_severity != ValidationSeverity.CRITICAL and res.pipeline_severity != ValidationSeverity.CRITICAL)
            if can_prep:
                self._set_label(self.view.lbl_val_postprep, ValidationSeverity.NEUTRAL, "ℹ️ Configuração carregada. Clique em 'Executar Preparação' para carregar os dados na memória.")
            else:
                self._set_label(self.view.lbl_val_postprep, res.post_prep_severity, res.post_prep_message)

    def _update_buttons_state(self, res: PLDAValidationResult):
        can_prep = (res.data_severity != ValidationSeverity.CRITICAL and res.pipeline_severity != ValidationSeverity.CRITICAL)
        self.view.btn_run_prep.setEnabled(can_prep)
        
        is_model_loaded = self.manager.is_model_loaded()
        has_model_disk = self.manager.has_model_on_disk()
        is_prepared = self.manager.is_prepared()
        has_pipelines = hasattr(self.manager, 'has_fitted_pipelines') and self.manager.has_fitted_pipelines()
        has_cache = hasattr(self.manager, 'has_trajectory_data') and self.manager.has_trajectory_data()
        
        # Simulação só ativa se os combos estiverem preenchidos
        has_split = self._get_clean_combo_text(self.view.combo_split_config) != "Selecione..."
        has_mining = self._get_clean_combo_text(self.view.combo_mining_config) != "Selecione..."
        self.view.btn_simulate_split.setEnabled(has_split)
        
        # O Treino é liberado se tiver RAM ou Pipelines ou se o Modelo já existir (para re-treino)
        # E A GORA PRECISA DE SPLIT E MINING CONFIGURADOS
        can_train = (is_prepared or has_pipelines or is_model_loaded or has_model_disk) and \
                    (res.plda_severity != ValidationSeverity.CRITICAL) and has_split and has_mining
        
        self.view.btn_train.setEnabled(can_train)
        
        # Feedback visual do botão
        if is_model_loaded or has_model_disk:
            self.view.btn_train.setText("🔄 Atualizar Modelo")
            self.view.btn_train.setToolTip("O modelo já existe. Clique para recalcular os parâmetros matemáticos.")
        else:
            self.view.btn_train.setText("🧠 Treinar Estratégia")
            if not (is_prepared or has_pipelines):
                self.view.btn_train.setToolTip("Os dados precisam estar na memória ou os pipelines no disco. Execute a preparação primeiro.")
            elif not has_split or not has_mining:
                self.view.btn_train.setToolTip("Selecione regras de Split e Mineração antes de treinar.")
            elif not can_train:
                self.view.btn_train.setToolTip("Configuração PLDA inválida (verifique parâmetros).")
            else:
                self.view.btn_train.setToolTip("")

        # Visualização
        can_visualize = is_prepared or has_cache or has_pipelines or is_model_loaded or has_model_disk
        self.view.btn_visualize.setEnabled(can_visualize)
        self.view.btn_visualize.setVisible(can_visualize)

        if has_cache and not is_prepared:
            self.view.btn_visualize.setText("👁️ Ver Trajetória 3D (Carregar Cache)")
        elif has_pipelines and not is_prepared:
            self.view.btn_visualize.setText("👁️ Calcular Trajetória 3D (Carregar Pipelines)")
        else:
            self.view.btn_visualize.setText("👁️ Visualizar Trajetória 3D")

    def _set_label(self, lbl, severity, text):
        lbl.setText(text)
        lbl.setVisible(True)
        bg, color = "#e9ecef", "#495057"
        if severity == ValidationSeverity.OK: bg, color = "#d4edda", "#155724"
        elif severity == ValidationSeverity.WARNING: bg, color = "#fff3cd", "#856404"
        elif severity == ValidationSeverity.CRITICAL: bg, color = "#f8d7da", "#721c24"
        lbl.setStyleSheet(f"background-color: {bg}; color: {color}; border-radius: 4px; padding: 5px;")

    # =========================================================================
    # HANDLERS DE EXECUÇÃO
    # =========================================================================

    def _update_simulation_label(self, res: Optional[SplitSimulationResult]):
        """[NOVO] Isola a lógica visual de desenhar os dados do Split."""
        if res and res.success:
            msg = (
                f"✅ Simulação Matemática Concluída\n\n"
                f"🎯 Treinamento PLDA: {res.plda_identities_count} Pessoas | {res.plda_images_count} Fotos\n"
                f"    ↳ Combinatória: {res.plda_genuine_pairs:,} Genuínos / {res.plda_impostor_pairs:,} Impostores\n\n"
                f"⚖️ Calibração LogReg: {res.calib_identities_count} Pessoas | {res.calib_images_count} Fotos\n"
                f"    ↳ Poder de Fogo: {res.calib_genuine_pairs:,} Genuínos / {res.calib_impostor_pairs:,} Impostores"
            ).replace(",", ".")
            
            self.view.lbl_split_result.setText(msg)
            self.view.lbl_split_result.setStyleSheet("background-color: #d4edda; color: #155724; border-radius: 4px; padding: 5px; font-weight: bold; font-size: 11px;")
        else:
            self.view.lbl_split_result.setText("Aguardando simulação...")
            self.view.lbl_split_result.setStyleSheet("background-color: #f8f9fa; border-radius: 4px; padding: 5px; font-size: 11px; margin-top: 5px; color: #6c757d;")


    def _handle_preparation(self):
        self.manager.update_config(self._build_config_from_ui())
        self.view.set_prep_loading(True)
        self._show_progress("Preparando dados e pipelines...")
        self.prep_worker = PLDAPreparationWorker(self.manager)
        self.prep_worker.progress_log.connect(self._update_progress_log)
        self.prep_worker.finished_success.connect(self._on_prep_success)
        self.prep_worker.finished_error.connect(self._on_prep_error)
        self.prep_worker.start()

    def _on_prep_success(self, result_info):
        if self.progress_dialog:
            self.progress_dialog.finish(success=True)
            
        self.view.set_prep_loading(False)
        self._trigger_validation()
        self.preparation_finished.emit()

    def _on_prep_error(self, error_msg):
        if self.progress_dialog:
            self.progress_dialog.finish(success=False)
            
        self.view.set_prep_loading(False)
        QtWidgets.QMessageBox.critical(self.view, "Erro na Preparação", error_msg)

    def _handle_simulate_split(self):
        # Atualiza o manager com as seleções de combobox
        self.manager.update_config(self._build_config_from_ui())
        
        self.view.btn_simulate_split.setEnabled(False)
        self.view.lbl_split_result.setText("Iniciando simulação...")
        self.view.lbl_split_result.setStyleSheet("background-color: #fff3cd; color: #856404; border-radius: 4px; padding: 5px; font-size: 11px;")
        
        self._show_progress("Simulando divisão de identidades e contando pares...")
        
        self.split_worker = PLDASplitSimulationWorker(self.manager)
        self.split_worker.progress_log.connect(self._update_progress_log)
        self.split_worker.finished_success.connect(self._on_simulate_success)
        self.split_worker.finished_error.connect(self._on_simulate_error)
        self.split_worker.start()

    def _on_simulate_success(self, res):
        if self.progress_dialog:
            self.progress_dialog.finish(success=True)
            
        self.view.btn_simulate_split.setEnabled(True)
        
        # [MODIFICADO] Chama a função visual isolada
        self._update_simulation_label(res)

        self._trigger_validation() # Avisa o Gatekeeper que a simulação acabou e ele pode reavaliar as dimensões!

    def _on_simulate_error(self, error_msg):
        if self.progress_dialog:
            self.progress_dialog.finish(success=False)
            
        self.view.btn_simulate_split.setEnabled(True)
        self._update_simulation_label(None) # Limpa a label em caso de falha
        QtWidgets.QMessageBox.critical(self.view, "Erro na Simulação", error_msg)

    def _handle_train(self):
        self.manager.update_config(self._build_config_from_ui())
        
        self.view.btn_train.setEnabled(False)
        self.view.btn_train.setText("Treinando...")
        
        self._show_progress("Treinando classificador probabilístico PLDA...")
        
        self.train_worker = PLDATrainingWorker(self.manager)
        self.train_worker.progress_log.connect(self._update_progress_log)
        self.train_worker.finished_success.connect(self._on_train_finish_success)
        self.train_worker.finished_error.connect(self._on_train_finish_error)
        self.train_worker.start()

    def _on_train_finish_success(self, result_path):
        if self.progress_dialog:
            self.progress_dialog.finish(success=True)
            
        self.view.btn_train.setEnabled(True)
        
        QtWidgets.QMessageBox.information(
            self.view, 
            "Sucesso", 
            "Modelo treinado e assinado digitalmente na pasta do projeto com sucesso!"
        )
        self._trigger_validation()

    def _on_train_finish_error(self, error):
        if self.progress_dialog:
            self.progress_dialog.finish(success=False)
            
        self.view.btn_train.setEnabled(True)
        QtWidgets.QMessageBox.critical(self.view, "Erro", f"Falha no treino: {error}")
        self._trigger_validation()

    # --- EDIÇÃO E CRIAÇÃO DE PIPELINE (ARQUITETURA DTO) ---
    def _handle_edit_pipeline(self, combo_widget: QtWidgets.QComboBox):
        raw_name = self._get_clean_combo_text(combo_widget)
        if raw_name == "Selecione...": return

        dto_config = self.manager.get_pipeline_config(raw_name)
        if not dto_config:
            QtWidgets.QMessageBox.critical(self.view, "Erro", "Pipeline não encontrado para edição.")
            return

        try:
            creator_ui = PipelineCreatorController(self.view, existing_config=dto_config)
            edited_config = creator_ui.run()
            
            if edited_config:
                self.manager.save_or_update_pipeline(edited_config, logger_callback=lambda x: None)
                self._refresh_pipelines()
                self._set_combo_safely(combo_widget, edited_config.name)
                self._trigger_validation()
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.view, "Erro de Edição", str(e))

    def _handle_edit_split(self):
        """[NOVO] Invoca a edição de regras de Split e salva no projeto ativo."""
        raw_name = self._get_clean_combo_text(self.view.combo_split_config)
        if raw_name == "Selecione...": return
        
        dto_config = self.manager.get_split_config(raw_name)
        if not dto_config:
            QtWidgets.QMessageBox.critical(self.view, "Erro", "Configuração de Split não encontrada.")
            return
            
        QtWidgets.QMessageBox.information(self.view, "Em Breve", f"Interface de Edição de Split '{raw_name}' em desenvolvimento.")

    def _handle_edit_mining(self):
        """[NOVO] Invoca a edição de regras de Mineração e salva no projeto ativo."""
        raw_name = self._get_clean_combo_text(self.view.combo_mining_config)
        if raw_name == "Selecione...": return
        
        dto_config = self.manager.get_mining_config(raw_name)
        if not dto_config:
            QtWidgets.QMessageBox.critical(self.view, "Erro", "Configuração de Mineração não encontrada.")
            return
            
        QtWidgets.QMessageBox.information(self.view, "Em Breve", f"Interface de Edição de Mineração '{raw_name}' em desenvolvimento.")


    def _build_config_from_ui(self) -> PLDAConfig:
        emb_type = self.view.combo_embedding.currentData()
        plda_mode = self.view.combo_plda_mode.currentData()
        
        # Puxa o dado limpo livre de formatações
        return PLDAConfig(
            id=self.strategy_id,
            type=LRModelType.PLDA_STRATEGY,
            strategy=self.view.combo_strategy.currentData(),
            embedding_type=emb_type if emb_type else EmbeddingType.SEPAEL_V2,
            parameters=PLDAParameters(
                em_iterations=self.view.spin_iter.value(),
                plda_mode=plda_mode if plda_mode else PLDAMode.FULL_RANK,
                latent_dim=self.view.spin_latent_dim.value(),
                logreg_C=self.view.spin_logreg_c.value()
            ),
            source_dataset_name=self._get_clean_combo_text(self.view.combo_train_data).split(" (")[0],
            target_dataset_name=self._get_clean_combo_text(self.view.combo_calib_data).split(" (")[0],
            source_pipeline_name=self._get_clean_combo_text(self.view.combo_pipe_source).split(" (")[0],
            target_pipeline_name=self._get_clean_combo_text(self.view.combo_pipe_target).split(" (")[0],
            split_config_name=self._get_clean_combo_text(self.view.combo_split_config).split(" (")[0], 
            mining_config_name=self._get_clean_combo_text(self.view.combo_mining_config).split(" (")[0] 
        )

    def _show_progress(self, msg):
        self.progress_dialog = LogProgressDialog(title=msg, parent=self.view)
        self.progress_dialog.show()

    def _update_progress_log(self, msg):
        if self.progress_dialog:
            self.progress_dialog.append_log(msg)

    def _close_progress(self):
        pass