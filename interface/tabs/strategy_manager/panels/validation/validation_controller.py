from __future__ import annotations
import numpy as np
import matplotlib

# Força o Matplotlib a usar o backend do Qt6
matplotlib.use('QtAgg') 

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, TYPE_CHECKING
from PyQt6 import QtCore, QtWidgets, QtGui

from faceaudit.enums.strategy_type import StrategyType
from faceaudit.models.validation_config import ValidationConfig

from faceaudit.manager.validation_manager import ValidationManager

# No topo do arquivo:
from faceaudit.models.validation_comparison_report import ValidationComparisonReport
from interface.tabs.strategy_manager.log_progress_dialog.log_progress_dialog import LogProgressDialog
from interface.tabs.strategy_manager.workers.validation_worker import ValidationWorker

if TYPE_CHECKING:
    from faceaudit.manager.strategies_manager import StrategiesManager
    from faceaudit.models.system_config import AppConfig
    from interface.tabs.strategy_manager.panels.validation.validation_view import ValidationView

class ValidationController(QtCore.QObject):
    """
    Controlador da Arena de Validação.
    Gerencia a Seleção do Gabarito, as Estratégias e o Gatekeeper (Data Leakage).
    """

    def __init__(self, 
                 view: "ValidationView", 
                 config: "AppConfig", 
                 strategies_manager: "StrategiesManager",
                 validation_manager: ValidationManager):
        
        super().__init__()
        self.view = view
        self.global_config = config
        self.strategies_manager = strategies_manager
        self.validation_manager = validation_manager

        self._strategy_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        
        # Agora conectamos um método interno que atualiza os labels ANTES de disparar a auditoria
        self.view.combo_probe.currentIndexChanged.connect(self._on_dataset_changed)
        self.view.combo_ref.currentIndexChanged.connect(self._on_dataset_changed)
        
        self.view.execute_requested.connect(self._handle_execute)

        print("[ValidationController] Arena Inicializada.")
        
        self.sync_ui_from_backend()

    # ---------------------------------------------------------------------
    # INTEGRAÇÃO COM A COLUNA 1 (PREPARAÇÃO) E BOOT
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # INTEGRAÇÃO COM A COLUNA 1 (PREPARAÇÃO) E BOOT
    # ---------------------------------------------------------------------

    def sync_ui_from_backend(self):
        """Popula a UI com os dados do Manager sem sujar o estado do projeto."""
        # 1. Salva o estado de modificação atual para não "sujar" o projeto no boot/load
        was_dirty = self.validation_manager.is_dirty()
        
        self._populate_datasets()
        self._refresh_strategies_list()
        
        # 2. INJETA O ESTADO SALVO NA UI (Para o Load Project funcionar)
        config = self.validation_manager.get_config()
        if config:
            self._set_combo_safely(self.view.combo_probe, config.probe_dataset_name)
            self._set_combo_safely(self.view.combo_ref, config.reference_dataset_name)
            
            for sid, chk in self._strategy_checkboxes.items():
                if sid in config.strategy_ids:
                    chk.blockSignals(True)
                    chk.setChecked(True)
                    chk.blockSignals(False)

        # 3. Força a atualização das estatísticas e o disparo da auditoria visual
        self._on_dataset_changed() 
        
        # 4. Restaura a flag para o que era antes da UI mexer
        self.validation_manager.set_dirty(was_dirty)

    def _set_combo_safely(self, combo: QtWidgets.QComboBox, text: str):
        """Seleciona o texto no combo lidando com os headers visuais (Projeto/Global)."""
        if not text or text == "Selecione...":
            combo.setCurrentIndex(0)
            return
        for i in range(combo.count()):
            item_data = combo.itemData(i, QtCore.Qt.ItemDataRole.UserRole)
            if item_data == text or combo.itemText(i).strip() == text.strip():
                combo.setCurrentIndex(i)
                return
        # Se não achar, não faz nada para não poluir o combo

    def _populate_datasets(self):
        self.view.combo_probe.blockSignals(True)
        self.view.combo_ref.blockSignals(True)
        
        global_ds, local_ds = self.validation_manager.get_available_datasets()
        
        def populate_grouped(combo, globals_, locals_):
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

            add_group("Projeto Ativo (Local)", locals_)
            add_group("Sistema (Global)", globals_)
            
            combo.setModel(model)
            combo.setCurrentIndex(0)

        populate_grouped(self.view.combo_probe, global_ds, local_ds)
        populate_grouped(self.view.combo_ref, global_ds, local_ds)
        
        self.view.combo_probe.blockSignals(False)
        self.view.combo_ref.blockSignals(False)

    @QtCore.pyqtSlot(str, object)
    def on_strategy_created(self, strategy_id: str, strategy_type: StrategyType):
        self._refresh_strategies_list()

    @QtCore.pyqtSlot(str, object)
    def on_strategy_removed(self, strategy_id: str, strategy_type: StrategyType):
        self._refresh_strategies_list()
        self._trigger_audit()

    @QtCore.pyqtSlot()
    def on_workspace_cleared(self):
        self.view.clear_strategies_list()
        self._strategy_checkboxes.clear()
        self.view.combo_probe.setCurrentIndex(0)
        self.view.combo_ref.setCurrentIndex(0)
        self._on_dataset_changed()

    def _refresh_strategies_list(self):
        self.view.clear_strategies_list()
        self._strategy_checkboxes.clear()

        strategies_info = self.validation_manager.get_active_strategies_info()
        
        if not strategies_info:
            lbl = QtWidgets.QLabel("Nenhuma estratégia criada na Coluna 1.")
            lbl.setStyleSheet("color: gray; font-style: italic;")
            self.view.strats_layout.addWidget(lbl)
            return

        for info in strategies_info:
            chk = QtWidgets.QCheckBox(f"{info['name']} ({info['type']})")
            chk.setEnabled(info['is_trained']) 
            if not info['is_trained']:
                chk.setToolTip("A estratégia precisa ser preparada e treinada na Coluna 1.")
            
            chk.toggled.connect(self._trigger_audit)
            
            self._strategy_checkboxes[info['id']] = chk
            self.view.strats_layout.addWidget(chk)

    # ---------------------------------------------------------------------
    # O GATEKEEPER FORENSE EM AÇÃO
    # ---------------------------------------------------------------------

    def _get_clean_combo_text(self, combo: QtWidgets.QComboBox) -> str:
        idx = combo.currentIndex()
        if idx < 0: return "Selecione..."
        data = combo.itemData(idx, QtCore.Qt.ItemDataRole.UserRole)
        if data: return str(data)
        text = combo.itemText(idx).strip()
        if text.startswith("──") or text == "Selecione...": return "Selecione..."
        return text

    def _on_dataset_changed(self):
        """Busca as estatísticas e atualiza os labels antes de invocar o Gatekeeper."""
        p_name = self._get_clean_combo_text(self.view.combo_probe)
        r_name = self._get_clean_combo_text(self.view.combo_ref)
        
        st_p = self.validation_manager.get_dataset_stats(p_name)
        st_r = self.validation_manager.get_dataset_stats(r_name)
        
        self.view.set_stats_text(self.view.lbl_stats_probe, st_p.n_imgs if st_p else 0, st_p.n_ids if st_p else 0)
        self.view.set_stats_text(self.view.lbl_stats_ref, st_r.n_imgs if st_r else 0, st_r.n_ids if st_r else 0)
        
        self._trigger_audit()

    def _build_config_from_ui(self) -> ValidationConfig:
        cfg = ValidationConfig()
        cfg.probe_dataset_name = self._get_clean_combo_text(self.view.combo_probe)
        cfg.reference_dataset_name = self._get_clean_combo_text(self.view.combo_ref)
        
        # Ignora "Selecione..." para não mandar lixo pro Manager
        if cfg.probe_dataset_name == "Selecione...": cfg.probe_dataset_name = ""
        if cfg.reference_dataset_name == "Selecione...": cfg.reference_dataset_name = ""
        
        cfg.strategy_ids = [sid for sid, chk in self._strategy_checkboxes.items() if chk.isChecked()]
        return cfg

    def _trigger_audit(self):
        config = self._build_config_from_ui()
        
        self.validation_manager.update_config(config, run_audit=True)
        
        audit_result = self.validation_manager.get_last_audit_result()
        if not audit_result:
            return

        messages = "\n".join(audit_result.arena_messages)
        for strat_id, strat_audit in audit_result.strategy_audits.items():
            if not strat_audit.is_pure:
                messages += f"\n\n[{strat_id.upper()}]: " + " ".join(strat_audit.messages)

        if audit_result.severity.name == "NEUTRAL":
            messages = "ℹ️ Selecione o Gabarito e pelo menos um competidor treinado para iniciar."
            
        self.view.set_gatekeeper_feedback(audit_result.is_pure, messages)

        can_execute = audit_result.is_pure and len(config.strategy_ids) > 0 and audit_result.severity.name != "NEUTRAL"
        self.view.btn_execute.setEnabled(can_execute)

    # ---------------------------------------------------------------------
    # EXECUÇÃO E RENDERIZAÇÃO DE RESULTADOS
    # ---------------------------------------------------------------------
    
    def _handle_execute(self):
        """Inicia a QThread de validação bloqueando a interface com o LogDialog."""
        if not self.validation_manager.get_config().is_audit_passed:
            QtWidgets.QMessageBox.warning(self.view, "Auditoria Falhou", "Resolva os avisos de segurança antes de executar.")
            return

        self.log_dialog = LogProgressDialog(title="Arena de Validação em Execução", parent=self.view)
        
        self.worker = ValidationWorker(self.validation_manager)
        self.worker.progress_log.connect(self.log_dialog.append_log)
        self.worker.finished_success.connect(self._on_validation_success)
        self.worker.finished_error.connect(self._on_validation_error)
        
        self.log_dialog.cancellation_requested.connect(self.worker.cancel)

        self.worker.start()
        self.log_dialog.exec()

    def _on_validation_error(self, message: str):
        self.log_dialog.finish(success=False)
        QtWidgets.QMessageBox.critical(self.view, "Erro na Validação", message)

    def _on_validation_success(self, report: ValidationComparisonReport):
        self.log_dialog.finish(success=True)
        
        # 1. Salva automaticamente o relatório recém-gerado no Pendrive
        # Usamos um nome genérico base, que pode ser sobrescrito pelo usuário futuramente
        try:
            self.validation_manager.save_report(report, "latest_run")
        except Exception as e:
            print(f"⚠️ Aviso: Não foi possível persistir o laudo de validação no disco: {e}")

        # 2. Exibe os resultados na UI
        self.view.results_widget.setVisible(True)
        self._populate_leaderboard(report)
        self._plot_results(report)

    def _populate_leaderboard(self, report: ValidationComparisonReport):
        """Preenche a tabela com as métricas extraídas do DTO."""
        table = self.view.table_results
        table.setRowCount(0)
        
        # Ordena as estratégias pelo Cllr (menor para maior) para formar um ranking
        sorted_results = sorted(report.results.values(), key=lambda r: r.metrics.cllr)
        
        for result in sorted_results:
            row = table.rowCount()
            table.insertRow(row)
            
            # Formatação numérica
            cllr_str = f"{result.metrics.cllr:.3f}"
            min_cllr_str = f"{result.metrics.cllr_min:.3f}"
            eer_str = f"{(result.metrics.eer * 100):.2f}%"
            
            # Insere os dados
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{result.strategy_name} ({result.strategy_type})"))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(cllr_str))
            table.setItem(row, 2, QtWidgets.QTableWidgetItem(min_cllr_str))
            table.setItem(row, 3, QtWidgets.QTableWidgetItem(eer_str))
            
            # Centraliza as colunas de números
            for col in range(1, 4):
                table.item(row, col).setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def _clear_layout(self, layout):
        """Limpa gráficos antigos das abas antes de desenhar os novos."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def _plot_results(self, report: ValidationComparisonReport):
        """Renderiza as curvas de desempenho (DET, Tippett, ECE e PDF)."""
        
        # 1. Limpa os layouts das abas
        self._clear_layout(self.view.layout_det)
        self._clear_layout(self.view.layout_tippett)
        self._clear_layout(self.view.layout_ece)
        self._clear_layout(self.view.layout_dist)

        # 2. Prepara as figuras do Matplotlib
        fig_det = Figure(figsize=(6, 4), dpi=100)
        ax_det = fig_det.add_subplot(111)
        
        fig_tip = Figure(figsize=(6, 4), dpi=100)
        ax_tip = fig_tip.add_subplot(111)
        
        fig_ece = Figure(figsize=(6, 4), dpi=100)
        ax_ece = fig_ece.add_subplot(111)

        fig_dist = Figure(figsize=(6, 4), dpi=100)
        ax_dist = fig_dist.add_subplot(111)

        # Paleta de cores dinâmica: Garante que cada Estratégia tenha uma cor ÚNICA em todos os gráficos!
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # 3. Itera sobre os resultados
        for idx, (strat_id, result) in enumerate(report.results.items()):
            name = result.strategy_name
            color = palette[idx % len(palette)] # Vincula a estratégia a uma cor específica
            
            # --- A. CURVA DET ---
            fpr = result.plot_data.fpr_array
            fnr = result.plot_data.fnr_array
            valid_idx = (fpr > 0) & (fnr > 0)
            ax_det.plot(fpr[valid_idx], fnr[valid_idx], lw=2, color=color, label=f'{name} (EER: {result.metrics.eer*100:.2f}%)')
            
            # --- B. TIPPETT PLOT ---
            gen_lr_10 = result.plot_data.genuine_log_lrs / np.log(10)
            imp_lr_10 = result.plot_data.impostor_log_lrs / np.log(10)
            
            gen_sorted = np.sort(gen_lr_10)
            imp_sorted = np.sort(imp_lr_10)
            gen_prop = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
            imp_prop = 1.0 - np.arange(1, len(imp_sorted) + 1) / len(imp_sorted)
            
            # Linha contínua para Genuínos, tracejada para Impostores (mesma cor da estratégia)
            ax_tip.plot(gen_sorted, gen_prop, lw=2, color=color, label=f'{name} (Genuínos)')
            ax_tip.plot(imp_sorted, imp_prop, lw=2, linestyle='--', color=color, label=f'{name} (Impostores)')

            # --- C. ECE PLOT (Calibração) ---
            if len(result.plot_data.ece_prior_log_odds) > 0:
                x_odds = result.plot_data.ece_prior_log_odds
                y_real = result.plot_data.ece_values
                y_min = result.plot_data.ece_min_values
                
                ax_ece.plot(x_odds, y_real, lw=2, color=color, label=f'{name} (Real)')
                ax_ece.plot(x_odds, y_min, lw=2, linestyle='--', color=color, alpha=0.7, label=f'{name} (Min/PAVA)')

            # --- D. DISTRIBUIÇÃO DE SCORES (PDF) ---
            # histtype='step' = Apenas a linha de contorno, sem preenchimento colorido!
            ax_dist.hist(gen_lr_10, bins=50, density=True, histtype='step', lw=2, color=color, label=f'{name} (Genuínos)')
            ax_dist.hist(imp_lr_10, bins=50, density=True, histtype='step', lw=2, linestyle='--', color=color, label=f'{name} (Impostores)')

        # Desenha a Referência Neutra no ECE Plot
        if len(report.results) > 0:
            sample_result = list(report.results.values())[0]
            if len(sample_result.plot_data.ece_prior_log_odds) > 0:
                x_odds = sample_result.plot_data.ece_prior_log_odds
                P = 10**x_odds / (1 + 10**x_odds)
                P = np.clip(P, 1e-15, 1 - 1e-15) 
                ece_null = -P * np.log2(P) - (1-P) * np.log2(1-P)
                ax_ece.plot(x_odds, ece_null, lw=1.5, color='black', linestyle=':', label='Neutra (Null System)')

        # --- Formatação do DET ---
        ax_det.set_xscale('log')
        ax_det.set_yscale('log')
        ax_det.set_xlabel('False Positive Rate (FPR)')
        ax_det.set_ylabel('False Negative Rate (FNR)')
        ax_det.set_title('Detection Error Tradeoff (DET)')
        ax_det.grid(True, which="both", linestyle='--', alpha=0.5)
        ax_det.legend(fontsize=9)

        # --- Formatação do Tippett ---
        ax_tip.set_xlabel(r'$\log_{10}(LR)$')
        ax_tip.set_ylabel('Proporção')
        ax_tip.set_title('Tippett Plot')
        ax_tip.axvline(0, color='red', linestyle=':', alpha=0.8) 
        ax_tip.grid(True, linestyle='--', alpha=0.6)
        ax_tip.legend(fontsize=9)

        # --- Formatação do ECE ---
        ax_ece.set_xlabel(r'Prior Log-Odds ($\log_{10} O$)')
        ax_ece.set_ylabel('Empirical Cross-Entropy (Bits)')
        ax_ece.set_title('Gráfico de Calibração (ECE)')
        ax_ece.axvline(0, color='gray', linestyle='-', alpha=0.5)
        ax_ece.grid(True, linestyle='--', alpha=0.6)
        ax_ece.legend(fontsize=9)

        # --- Formatação da Distribuição (PDF) ---
        ax_dist.set_xlabel(r'$\log_{10}(LR)$')
        ax_dist.set_ylabel('Densidade')
        ax_dist.set_title('Distribuição de LRs (Genuínos vs Impostores)')
        ax_dist.axvline(0, color='black', linestyle='-', alpha=0.8, label='LR = 1 (Neutro)')
        ax_dist.grid(True, linestyle='--', alpha=0.5)
        ax_dist.legend(fontsize=9)

        fig_det.tight_layout()
        fig_tip.tight_layout()
        fig_ece.tight_layout()
        fig_dist.tight_layout()

        # Injeta as figuras no PyQt
        self.view.layout_det.addWidget(FigureCanvas(fig_det))
        self.view.layout_tippett.addWidget(FigureCanvas(fig_tip))
        self.view.layout_ece.addWidget(FigureCanvas(fig_ece))
        self.view.layout_dist.addWidget(FigureCanvas(fig_dist))