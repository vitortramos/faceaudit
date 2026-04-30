import uuid
from PyQt6 import QtCore, QtWidgets
from .new_mining_config_dialog_view import NewMiningConfigDialogView

# Importe os seus DTOs e Enums
from faceaudit.models.comparison_mining_config import (
    ComparisonMiningConfig, MiningRule, MiningStrategy, MiningFocus, GlobalSelectionMode, MixedProportions
)

class NewMiningConfigDialogController(QtCore.QObject):
    def __init__(self, parent=None, lock_name: bool = False):
        super().__init__(parent)
        self.view = NewMiningConfigDialogView(parent)
        self._editing_id = None
        self.lock_name = bool(lock_name)
        self._connect_signals()
        self._sync_ui_states()

        # lock_name: usado quando a edição vem do botão ✎ ao lado de um
        # dropdown na nova arch — o nome do recurso é o stem do arquivo no
        # disco e não pode ser alterado por aqui (renomear será operação
        # dedicada no painel esquerdo, ainda não implementada).
        if self.lock_name:
            self.view.txt_name.setReadOnly(True)
            self.view.txt_name.setToolTip(
                "O nome do recurso não pode ser alterado por aqui. "
                "Renomear a config de mineração será uma operação dedicada "
                "no painel esquerdo (ainda não disponível)."
            )
            self.view.txt_name.setStyleSheet(
                "QLineEdit { background-color: #f0f0f0; color: #555; }"
            )

    def _connect_signals(self):
        self.view.button_box.accepted.connect(self._validate_and_accept)
        self.view.button_box.rejected.connect(self.view.reject)

        self.view.imp_widgets["focus"].currentIndexChanged.connect(self._sync_ui_states)
        self.view.gen_widgets["focus"].currentIndexChanged.connect(self._sync_ui_states)
        
        self.view.imp_widgets["max_global"].valueChanged.connect(self._sync_ui_states)
        self.view.gen_widgets["max_global"].valueChanged.connect(self._sync_ui_states)
        
        # Sinais para esconder/mostrar o painel MIXED
        self.view.imp_widgets["strategy"].currentIndexChanged.connect(self._sync_ui_states)
        self.view.gen_widgets["strategy"].currentIndexChanged.connect(self._sync_ui_states)

    def _sync_ui_states(self):
        """Habilita ou desabilita campos dependendo da seleção do usuário."""
        # --- Lógica do Modo Misto (Esconde/Mostra Painel) ---
        imp_is_mixed = self.view.imp_widgets["strategy"].currentIndex() == 4
        self.view.imp_widgets["group_mixed"].setVisible(imp_is_mixed)
        
        gen_is_mixed = self.view.gen_widgets["strategy"].currentIndex() == 4
        self.view.gen_widgets["group_mixed"].setVisible(gen_is_mixed)

        # --- Lógica de Foco (Identities vs Pairs) ---
        is_imp_identities = self.view.imp_widgets["focus"].currentIndex() == 1
        self.view.imp_widgets["max_ids"].setEnabled(is_imp_identities)

        is_gen_identities = self.view.gen_widgets["focus"].currentIndex() == 1
        self.view.gen_widgets["max_ids"].setEnabled(is_gen_identities)
        
        # --- Lógica de Corte Global ---
        has_imp_global_limit = self.view.imp_widgets["max_global"].value() > 0
        self.view.imp_widgets["global_mode"].setEnabled(has_imp_global_limit)
        
        has_gen_global_limit = self.view.gen_widgets["max_global"].value() > 0
        self.view.gen_widgets["global_mode"].setEnabled(has_gen_global_limit)

    def load_config(self, config: ComparisonMiningConfig):
        """Preenche a tela com os dados de um objeto existente para edição."""
        self._editing_id = config.id
        self.view.txt_name.setText(config.name)
        self.view.txt_desc.setText(config.description)
        self.view.sp_seed.setValue(config.seed)
        self.view.chk_exclude_self.setChecked(config.exclude_self_comparisons)
        
        def _apply_rule(w_dict, rule: MiningRule):
            strategy_map = [
                MiningStrategy.RANDOM, MiningStrategy.HARD, MiningStrategy.EASY, 
                MiningStrategy.TYPICAL, MiningStrategy.MIXED
            ]
            focus_map = [MiningFocus.PAIRS, MiningFocus.IDENTITIES]
            global_mode_map = [
                GlobalSelectionMode.GLOBAL_ORDER, 
                GlobalSelectionMode.ROUND_ROBIN_BY_PIVOT, 
                GlobalSelectionMode.ROUND_ROBIN_BY_IDENTITY_PAIR
            ]
            
            try:
                w_dict["strategy"].setCurrentIndex(strategy_map.index(rule.strategy))
                w_dict["focus"].setCurrentIndex(focus_map.index(rule.focus))
                w_dict["global_mode"].setCurrentIndex(global_mode_map.index(rule.global_selection_mode))
            except ValueError:
                pass
            
            w_dict["max_global"].setValue(rule.max_global_pairs if rule.max_global_pairs else 0)
            w_dict["max_per_id"].setValue(rule.max_pairs_per_identity if rule.max_pairs_per_identity else 0)
            w_dict["max_ids"].setValue(rule.max_identities if rule.max_identities else 0)
            
            w_dict["sp_hard_pct"].setValue(rule.mixed_proportions.hard_ratio * 100)
            w_dict["sp_typical_pct"].setValue(rule.mixed_proportions.typical_ratio * 100)
            w_dict["sp_random_pct"].setValue(rule.mixed_proportions.random_ratio * 100)
            w_dict["sp_easy_pct"].setValue(rule.mixed_proportions.easy_ratio * 100)

        _apply_rule(self.view.imp_widgets, config.impostors_rule)
        _apply_rule(self.view.gen_widgets, config.genuines_rule)
        
        self.view.setWindowTitle(f"Editar Estratégia: {config.name}")
        self._sync_ui_states()

    def exec(self) -> int:
        return self.view.exec()

    def _validate_and_accept(self):
        if not self.view.txt_name.text().strip():
            QtWidgets.QMessageBox.warning(self.view, "Atenção", "Dê um nome para a Estratégia de Mineração.")
            return
            
        # Validação Matemática do Modo Misto
        def check_proportions(w_dict, tab_name):
            if w_dict["strategy"].currentIndex() == 4: # Se for MIXED
                total = (w_dict["sp_hard_pct"].value() + w_dict["sp_typical_pct"].value() + 
                         w_dict["sp_random_pct"].value() + w_dict["sp_easy_pct"].value())
                if abs(total - 100.0) > 0.1:
                    QtWidgets.QMessageBox.warning(
                        self.view, "Erro de Proporção", 
                        f"Na aba '{tab_name}', a soma das porcentagens do modo Misto deve ser exatamente 100%.\nSoma atual: {total}%"
                    )
                    return False
            return True

        if not check_proportions(self.view.imp_widgets, "Impostores"): return
        if not check_proportions(self.view.gen_widgets, "Genuínos"): return

        self.view.accept()

    def get_config(self) -> ComparisonMiningConfig:
        """Extrai os dados da tela e constrói o DTO."""
        def _build_rule(w_dict) -> MiningRule:
            strategy_map = [
                MiningStrategy.RANDOM, MiningStrategy.HARD, MiningStrategy.EASY, 
                MiningStrategy.TYPICAL, MiningStrategy.MIXED
            ]
            focus_map = [MiningFocus.PAIRS, MiningFocus.IDENTITIES]
            global_mode_map = [
                GlobalSelectionMode.GLOBAL_ORDER, 
                GlobalSelectionMode.ROUND_ROBIN_BY_PIVOT, 
                GlobalSelectionMode.ROUND_ROBIN_BY_IDENTITY_PAIR
            ]

            st_val = strategy_map[w_dict["strategy"].currentIndex()]
            focus_val = focus_map[w_dict["focus"].currentIndex()]
            global_mode_val = global_mode_map[w_dict["global_mode"].currentIndex()]

            val_global = w_dict["max_global"].value()
            val_per_id = w_dict["max_per_id"].value()
            val_ids = w_dict["max_ids"].value()
            
            # Converte porcentagem de volta para ratio (0.0 a 1.0)
            props = MixedProportions(
                hard_ratio=w_dict["sp_hard_pct"].value() / 100.0,
                typical_ratio=w_dict["sp_typical_pct"].value() / 100.0,
                random_ratio=w_dict["sp_random_pct"].value() / 100.0,
                easy_ratio=w_dict["sp_easy_pct"].value() / 100.0
            )

            return MiningRule(
                strategy=st_val,
                mixed_proportions=props,
                focus=focus_val,
                max_global_pairs=val_global if val_global > 0 else None,
                global_selection_mode=global_mode_val,
                max_pairs_per_identity=val_per_id if val_per_id > 0 else None,
                max_identities=val_ids if (val_ids > 0 and focus_val == MiningFocus.IDENTITIES) else None
            )

        config_id = self._editing_id if self._editing_id else f"mining_{uuid.uuid4().hex[:8]}"

        return ComparisonMiningConfig(
            id=config_id,
            name=self.view.txt_name.text().strip(),
            description=self.view.txt_desc.toPlainText().strip(),
            seed=self.view.sp_seed.value(),
            exclude_self_comparisons=self.view.chk_exclude_self.isChecked(),
            impostors_rule=_build_rule(self.view.imp_widgets),
            genuines_rule=_build_rule(self.view.gen_widgets)
        )