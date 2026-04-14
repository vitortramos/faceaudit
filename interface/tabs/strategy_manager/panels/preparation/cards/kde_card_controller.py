# interface/tabs/strategy_manager/panels/preparation/cards/kde_card/kde_card_controller.py
from pathlib import Path
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QMessageBox
from typing import Optional, Dict, Any, TYPE_CHECKING

from faceaudit.enums.lr_model_type import LRModelType
from faceaudit.enums.project_file_type import ProjectFileType
from interface.tabs.strategy_manager.new_mining_config_dialog.new_mining_config_dialog_controller import NewMiningConfigDialogController 

# View
from .kde_card_view import KDECardView

# Dialogs
from interface.tabs.strategy_manager.log_progress_dialog.log_progress_dialog import LogProgressDialog
from interface.tabs.strategy_manager.panels.preparation.cards.kde_plot_dialog.kde_plot_dialog import KDEPlotDialog

# Workers
from interface.tabs.strategy_manager.panels.preparation.workers.kde_training_worker import KDETrainingWorker
from interface.tabs.strategy_manager.panels.preparation.workers.kde_preparation_worker import KDEPreparationWorker 

# Enums
from faceaudit.enums.strategy_type import StrategyType
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.kde_kernel_type import KDEKernelType

# Models
from faceaudit.models.kde_config import KDEConfig
from faceaudit.parameters.kde_parameters import KDEParameters
from faceaudit.models.kde_validation_result import KDEValidationResult, ValidationSeverity

# --- NOVO IMPORT DO DTO ---
from faceaudit.models.aggregation_config import AggregationConfig

if TYPE_CHECKING:
    from faceaudit.manager.kde_manager import KDEManager
    from faceaudit.models.system_config import AppConfig

class KDECardController(QtCore.QObject):
    
    request_destruction = QtCore.pyqtSignal(object)
    preparation_finished = QtCore.pyqtSignal()
    
    def __init__(self, view_idx: int, strategy_id: str, manager: 'KDEManager', global_config: 'AppConfig', parent=None):
        super().__init__(parent)
        self.strategy_id = strategy_id
        self.manager = manager
        self.global_config = global_config
        
        self.prep_worker: Optional[KDEPreparationWorker] = None 
        self.train_worker: Optional[KDETrainingWorker] = None
        self._log_dialog: Optional[LogProgressDialog] = None

        self.view = KDECardView(view_idx, parent)
        nome_formatado = f"ESTRATÉGIA {self.strategy_id.upper()}"
        self.view.lbl_title.setText(nome_formatado)
        
        self._init_data()
        self._connect_signals()
        self._sync_ui_from_config()

    def get_widget(self) -> QtWidgets.QWidget:
        return self.view

    def get_config(self) -> KDEConfig:
        return self._build_config_from_ui()

    def update_global_config(self, new_config: 'AppConfig'):
        self.global_config = new_config
        self._init_data()
        self._trigger_validation()

    def _sync_ui_from_config(self):
        cfg = self.manager.get_config()
        if not cfg or not isinstance(cfg, KDEConfig):
            self._trigger_validation()
            return

        p = cfg.parameters
        agg = cfg.aggregation # Pode ser None

        self.view.combo_agg_method.blockSignals(True)
        self.view.combo_quality_type.blockSignals(True)
        self.view.combo_mining_config.blockSignals(True) 
        self.view.combo_kernel.blockSignals(True)
        self.view.combo_probe_data.blockSignals(True)
        self.view.combo_ref_data.blockSignals(True)

        # --- A LEITURA DO NOVO DTO DE AGREGAÇÃO ---
        if agg:
            self.view.chk_aggregation.setChecked(True)
            self._set_combo_enum_safely(self.view.combo_agg_method, agg.method)
            self.view.spin_group_size.setValue(agg.combination_group_size)
            self.view.chk_weighted.setChecked(agg.use_quality_weighting)
            self._set_combo_enum_safely(self.view.combo_quality_type, agg.quality_weight_type)
        else:
            self.view.chk_aggregation.setChecked(False)
            # Valores padrão visuais quando desmarcado
            self.view.combo_agg_method.setCurrentIndex(0)
            self.view.spin_group_size.setValue(3)
            self.view.chk_weighted.setChecked(False)
            self.view.combo_quality_type.setCurrentIndex(0)

        # --- PARÂMETROS MATEMÁTICOS NORMAIS ---
        self._set_combo_safely(self.view.combo_mining_config, getattr(p, "mining_config_name", ""))
        self._set_combo_enum_safely(self.view.combo_kernel, p.kernel)
        self.view.chk_gridsearch.setChecked(p.use_gridsearch)
        self.view.spin_max_grid.setValue(p.max_samples_for_gridsearch)
        self.view.spin_max_store.setValue(p.max_stored_samples)
        self.view.spin_bw_gen.setValue(p.bandwidth_genuines)
        self.view.spin_bw_imp.setValue(p.bandwidth_impostors)
        
        self._set_combo_safely(self.view.combo_probe_data, cfg.probe_dataset_name)
        self._set_combo_safely(self.view.combo_ref_data, cfg.reference_dataset_name)

        self.view.combo_agg_method.blockSignals(False)
        self.view.combo_quality_type.blockSignals(False)
        self.view.combo_mining_config.blockSignals(False) 
        self.view.combo_kernel.blockSignals(False)
        self.view.combo_probe_data.blockSignals(False)
        self.view.combo_ref_data.blockSignals(False)

        self.view._update_agg_visibility()
        self.view._update_bw_visibility()
        
        p_name = self._get_clean_combo_text(self.view.combo_probe_data)
        r_name = self._get_clean_combo_text(self.view.combo_ref_data)
        
        st_p = self.manager.get_dataset_stats(p_name)
        st_r = self.manager.get_dataset_stats(r_name)
        
        self.view.set_stats_text(self.view.lbl_stats_probe, st_p.n_imgs if st_p else 0, st_p.n_ids if st_p else 0)
        self.view.set_stats_text(self.view.lbl_stats_ref, st_r.n_imgs if st_r else 0, st_r.n_ids if st_r else 0)

        self._check_mining_ui_logic() 
        res: KDEValidationResult = self.manager.validate_all()
        self._update_validation_labels(res)
        self._update_buttons_state(res)

    def _build_config_from_ui(self) -> KDEConfig:
        agg_active = self.view.chk_aggregation.isChecked()
        use_grid = self.view.chk_gridsearch.isChecked()
        
        # --- A CRIAÇÃO DO NOVO DTO DE AGREGAÇÃO ---
        agg_config = None
        if agg_active:
            agg_config = AggregationConfig(
                method=self.view.combo_agg_method.currentData(),
                combination_group_size=self.view.spin_group_size.value(),
                # O Max templates não estava na UI do KDECard (só na tela do PLDA Creator), 
                # então assumimos o default ou usamos a constante do DTO.
                max_synthetic_templates_per_id=100, 
                use_quality_weighting=self.view.chk_weighted.isChecked(),
                quality_weight_type=self.view.combo_quality_type.currentData()
            )

        # O KDEParameters agora é puramente matemático
        params = KDEParameters(
            mining_config_name=self._get_clean_combo_text(self.view.combo_mining_config),
            kernel=self.view.combo_kernel.currentData(),
            use_gridsearch=use_grid,
            bandwidth_genuines=self.view.spin_bw_gen.value(),
            bandwidth_impostors=self.view.spin_bw_imp.value(),
            max_samples_for_gridsearch=self.view.spin_max_grid.value(), 
            max_stored_samples=self.view.spin_max_store.value(),        
            impostors_min_count=100, 
            genuines_min_count=100
        )
        
        return KDEConfig(
            id=self.strategy_id,
            type=LRModelType.KDE_STRATEGY,
            probe_dataset_name=self._get_clean_combo_text(self.view.combo_probe_data),
            reference_dataset_name=self._get_clean_combo_text(self.view.combo_ref_data),
            embedding_type=EmbeddingType.SEPAEL_V2,
            aggregation=agg_config, # <--- DTO INJETADO
            parameters=params
        )

    # --- LÓGICA DE AGRUPAMENTO ---
    def _get_clean_combo_text(self, combo: QtWidgets.QComboBox) -> str:
        idx = combo.currentIndex()
        if idx < 0: return "Selecione..."
        data = combo.itemData(idx, QtCore.Qt.ItemDataRole.UserRole)
        if data: return str(data)
        text = combo.itemText(idx).strip()
        if text.startswith("──") or text == "Selecione...": return "Selecione..."
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

    def _set_combo_enum_safely(self, combo: QtWidgets.QComboBox, enum_val):
        if not enum_val: return
        idx = combo.findData(enum_val)
        if idx >= 0: combo.setCurrentIndex(idx)

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
        if default: self._set_combo_safely(combo, default)
        else: combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _init_data(self):
        self.view.combo_agg_method.blockSignals(True)
        self.view.combo_quality_type.blockSignals(True)
        self.view.combo_mining_config.blockSignals(True) 
        self.view.combo_kernel.blockSignals(True)

        global_ds, local_ds = self.manager.get_available_datasets()
        self._populate_combo_grouped(self.view.combo_probe_data, global_ds, local_ds)
        self._populate_combo_grouped(self.view.combo_ref_data, global_ds, local_ds)

        global_mining, local_mining = self.manager.get_available_mining_configs()
        self._populate_combo_grouped(self.view.combo_mining_config, global_mining, local_mining)

        self.view.combo_agg_method.clear()
        for agg in AggregationMethod:
            label = getattr(agg.info, 'label', agg.value.replace("_", " ").title()) if hasattr(agg, 'info') else agg.value
            self.view.combo_agg_method.addItem(label, agg)

        self.view.combo_quality_type.clear()
        for qual in QualityScoreType:
            label = getattr(qual, 'display_name', qual.name)
            self.view.combo_quality_type.addItem(label, qual)

        self.view.combo_kernel.clear()
        for kern in KDEKernelType:
            label = getattr(kern.info, 'label', kern.value) if hasattr(kern, 'info') else kern.value
            self.view.combo_kernel.addItem(label, kern)
            
        self.view.combo_kernel.setCurrentIndex(0) 
        self.view.combo_agg_method.setCurrentIndex(0)

        self.view.combo_agg_method.blockSignals(False)
        self.view.combo_quality_type.blockSignals(False)
        self.view.combo_mining_config.blockSignals(False) 
        self.view.combo_kernel.blockSignals(False)

    def _refresh_mining_configs(self):
        current_mining = self._get_clean_combo_text(self.view.combo_mining_config)
        global_mining, local_mining = self.manager.get_available_mining_configs()
        self._populate_combo_grouped(self.view.combo_mining_config, global_mining, local_mining)
        self._set_combo_safely(self.view.combo_mining_config, current_mining)
        self._check_mining_ui_logic()

    def _connect_signals(self):
        self.view.remove_requested.connect(lambda: self.request_destruction.emit(self))
        self.view.combo_probe_data.currentTextChanged.connect(self._on_dataset_changed)
        self.view.combo_ref_data.currentTextChanged.connect(self._on_dataset_changed)
        self.view.chk_aggregation.toggled.connect(self._trigger_validation)
        self.view.combo_agg_method.currentIndexChanged.connect(self._trigger_validation)
        self.view.spin_group_size.valueChanged.connect(self._trigger_validation)
        self.view.chk_weighted.toggled.connect(self._trigger_validation)
        self.view.combo_mining_config.currentIndexChanged.connect(self._check_mining_ui_logic) 
        self.view.chk_gridsearch.toggled.connect(self._trigger_validation)
        self.view.spin_bw_gen.valueChanged.connect(self._trigger_validation)
        self.view.spin_bw_imp.valueChanged.connect(self._trigger_validation)
        self.view.spin_max_grid.valueChanged.connect(self._trigger_validation)
        self.view.spin_max_store.valueChanged.connect(self._trigger_validation)
        self.view.run_prep_requested.connect(self._handle_preparation)
        self.view.train_requested.connect(self._handle_train)
        self.view.view_plot_requested.connect(self._handle_view_plot)
        
        # Apenas edição conectada
        self.view.edit_mining_requested.connect(self._handle_edit_mining)

    def _check_mining_ui_logic(self):
        m_name = self._get_clean_combo_text(self.view.combo_mining_config).split(" (")[0]
        self.view.btn_edit_mining.setEnabled(m_name != "Selecione...")
        self._trigger_validation()

    def _on_dataset_changed(self):
        p_name = self._get_clean_combo_text(self.view.combo_probe_data)
        r_name = self._get_clean_combo_text(self.view.combo_ref_data)
        st_p = self.manager.get_dataset_stats(p_name)
        st_r = self.manager.get_dataset_stats(r_name)
        self.view.set_stats_text(self.view.lbl_stats_probe, st_p.n_imgs if st_p else 0, st_p.n_ids if st_p else 0)
        self.view.set_stats_text(self.view.lbl_stats_ref, st_r.n_imgs if st_r else 0, st_r.n_ids if st_r else 0)
        self._trigger_validation()

    def _trigger_validation(self):
        current_cfg = self._build_config_from_ui()
        self.manager.update_config(current_cfg)
        res: KDEValidationResult = self.manager.validate_all()
        self._update_validation_labels(res)
        self._update_buttons_state(res)

    def _update_validation_labels(self, res: KDEValidationResult):
        msg_data = "\n".join(res.messages) if res.messages else "OK"
        if res.data_severity == ValidationSeverity.NEUTRAL: msg_data = "Selecione Datasets"
        self.view.set_label_feedback(self.view.lbl_val_data, res.data_severity, msg_data)
        if self.manager.is_prepared():
             self.view.set_label_feedback(self.view.lbl_val_prep, ValidationSeverity.OK, "Dados Preparados.")
        else:
             self.view.set_label_feedback(self.view.lbl_val_prep, ValidationSeverity.NEUTRAL, "Aguardando Preparação...")
        self.view.set_label_feedback(self.view.lbl_val_config, res.params_severity, "Parâmetros OK" if res.params_severity == ValidationSeverity.OK else "Verifique Parâmetros")

    def _update_buttons_state(self, res: KDEValidationResult):
        is_data_ok = (res.data_severity != ValidationSeverity.CRITICAL)
        is_agg_checked = self.view.chk_aggregation.isChecked()
        
        self.view.btn_run_prep.setEnabled(is_data_ok and is_agg_checked)
        
        can_train = is_data_ok and (res.params_severity != ValidationSeverity.CRITICAL)
        self.view.btn_train.setEnabled(can_train)

        has_model = self.manager.is_model_loaded() or self.manager.has_model_on_disk()
        self.view.btn_view_plot.setEnabled(has_model)
        self.view.btn_view_plot.setVisible(has_model)
        
        if has_model:
            self.view.btn_train.setText("🔄 Recalibrar Modelo")
            self.view.btn_train.setToolTip("O modelo já existe. Clique para recalcular a distribuição probabilística.")
        else:
            self.view.btn_train.setText("CALIBRAR MODELO KDE")
            self.view.btn_train.setToolTip("")

    # =========================================================================
    # EDIÇÃO DE REGRA DE MINERAÇÃO (FORK & SAVE)
    # =========================================================================

    def _handle_edit_mining(self):
        raw_name = self._get_clean_combo_text(self.view.combo_mining_config)
        if raw_name == "Selecione...": return

        dto_config = self.manager.get_mining_config(raw_name)
        if not dto_config:
            QtWidgets.QMessageBox.critical(self.view, "Erro", "Regra de mineração não encontrada para edição.")
            return

        try:
            creator_ui = NewMiningConfigDialogController(self.view)
            creator_ui.load_config(dto_config)
            
            if creator_ui.exec():
                edited_config = creator_ui.get_config()
                self.manager.save_or_update_mining_config(edited_config, logger_callback=lambda x: print(x))
                
                self._refresh_mining_configs()
                self._set_combo_safely(self.view.combo_mining_config, edited_config.name)
                self._trigger_validation()
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.view, "Erro de Edição", str(e))

    # =========================================================================
    # EXECUÇÃO DE WORKERS 
    # =========================================================================

    def _handle_preparation(self):
        try:
            current_cfg = self._build_config_from_ui()
            self.manager.update_config(current_cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.view, "Erro de Configuração", str(e))
            return

        self._log_dialog = LogProgressDialog("Preparação de Dados (Agregação)", self.view)
        self._log_dialog.show()

        self.prep_worker = KDEPreparationWorker(self.manager)
        self.prep_worker.progress_log.connect(self._log_dialog.append_log)
        self.prep_worker.finished_success.connect(self._on_prep_success)
        self.prep_worker.finished_error.connect(self._on_prep_error)
        self.prep_worker.start()

    def _on_prep_success(self, stats: dict):
        if self._log_dialog: self._log_dialog.finish(success=True)
        self._trigger_validation()
        self.preparation_finished.emit()

    def _on_prep_error(self, err_msg: str):
        if self._log_dialog:
            self._log_dialog.append_log(f"\nERRO: {err_msg}")
            self._log_dialog.finish(success=False)
        self.view.set_label_feedback(self.view.lbl_val_prep, ValidationSeverity.CRITICAL, f"Erro: {err_msg}")

    def _handle_train(self):
        try:
            current_cfg = self._build_config_from_ui()
            self.manager.update_config(current_cfg)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.view, "Erro de Configuração", f"{e}")
            return

        self._log_dialog = LogProgressDialog("Calibração KDE", self.view)
        self._log_dialog.show()
        self._log_dialog.append_log("Iniciando Worker de Treinamento...")

        if self.train_worker is not None:
             if self.train_worker.isRunning(): self.train_worker.terminate()
             self.train_worker.deleteLater()
             
        self.train_worker = KDETrainingWorker(self.manager)
        self.train_worker.progress_log.connect(self._log_dialog.append_log)
        self.train_worker.finished_success.connect(self._on_worker_success)
        self.train_worker.finished_error.connect(self._on_worker_error)
        self._log_dialog.cancellation_requested.connect(self._on_cancel_requested)
        self.train_worker.start()
    
    def _on_cancel_requested(self):
        if self.train_worker and self.train_worker.isRunning():
            self._log_dialog.append_log("🛑 Forçando parada do Worker...")
            self.train_worker.terminate()
            self.train_worker.wait()
            self._log_dialog.finish(success=False)
            self._log_dialog.append_log("Cancelado pelo usuário.")

    def _on_worker_success(self, path: str):
        if self._log_dialog:
            self._log_dialog.append_log(f"Treinamento concluído. Artefatos salvos.")
            self._log_dialog.finish(success=True)
            
        QtWidgets.QMessageBox.information(self.view, "Sucesso", "Modelo e artefatos salvos e assinados no projeto.")
        self._trigger_validation()

    def _on_worker_error(self, err_msg: str):
        if self._log_dialog:
            self._log_dialog.append_log(f"ERRO FATAL: {err_msg}")
            self._log_dialog.finish(success=False)
        self._trigger_validation()

    def _handle_view_plot(self):
        try:
            model_path = self.manager.get_model_path() 
            if not model_path or not Path(model_path).exists():
                QtWidgets.QMessageBox.warning(self.view, "Aviso", "Nenhum modelo treinado foi encontrado.\nTreine ou carregue um modelo primeiro.")
                return
                
            viz_data = self.manager.build_visualization_data_from_model(Path(model_path), x_min=-0.25, x_max=1.0)
            dialog = KDEPlotDialog(self.view)
            dialog.plot_visualization(viz_data, title_suffix=f"({self._get_clean_combo_text(self.view.combo_probe_data)} vs {self._get_clean_combo_text(self.view.combo_ref_data)})")
            dialog.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc() 
            QtWidgets.QMessageBox.critical(self.view, "Erro", f"Falha ao gerar gráfico:\n{e}")