from PyQt6.QtWidgets import QMessageBox
from faceaudit.models.split_config import SplitConfig, SplitMode, GroupingStrategy
from interface.theme import apply_themed_stylesheet
from .new_split_config_dialog_view import NewSplitConfigDialogView

class NewSplitConfigDialogController:
    """
    Controller para o Diálogo de Criação/Edição de Configurações de Split (PLDA).
    Gerencia as validações, as travas dinâmicas da interface e a extração do DTO.
    """
    def __init__(
        self,
        parent=None,
        config_to_edit: SplitConfig = None,
        lock_name: bool = False,
    ):
        self.view = NewSplitConfigDialogView(parent)
        self.config_to_edit = config_to_edit
        self.lock_name = bool(lock_name)
        self.final_config = None

        self._setup_connections()

        # Se um objeto foi passado, estamos em modo de EDIÇÃO.
        if self.config_to_edit:
            self._load_config()
            self.view.setWindowTitle("Editar Configuração de Split")
        else:
            # Em modo de CRIAÇÃO, apenas força o estado inicial das travas
            self._update_ui_state()

        # lock_name: usado quando a edição vem do botão ✎ ao lado de um
        # dropdown na nova arch — o nome do recurso é o stem do arquivo
        # em disco e não pode ser alterado por aqui (renomear será
        # operação dedicada no painel esquerdo, ainda não implementada).
        if self.lock_name:
            self.view.name_input.setReadOnly(True)
            self.view.name_input.setToolTip(
                "O nome do recurso não pode ser alterado por aqui. "
                "Renomear a config de split será uma operação dedicada "
                "no painel esquerdo (ainda não disponível)."
            )
            apply_themed_stylesheet(
                self.view.name_input,
                lambda t: (
                    "QLineEdit { background-color: " + t("SURFACE_ALT")
                    + "; color: " + t("TEXT") + "; }"
                ),
            )

    def _setup_connections(self):
        """Conecta os sinais (events) da view aos métodos do controller."""
        self.view.btn_save.clicked.connect(self._handle_save)
        self.view.btn_cancel.clicked.connect(self.view.reject)
        
        # Dinâmica de UI: Quando o usuário muda um combobox, ativamos/desativamos os spinboxes
        self.view.grouping_combo.currentIndexChanged.connect(self._update_ui_state)
        self.view.mode_combo.currentIndexChanged.connect(self._update_ui_state)

    def _update_ui_state(self):
        """Bloqueia ou desbloqueia campos da view baseados nas escolhas de Enums."""
        # 1. Estratégia de Agrupamento
        strategy = self.view.grouping_combo.currentData()
        is_cluster = (strategy == GroupingStrategy.PRESERVE_LOOKALIKES)
        
        # Só permite editar o tamanho do pacote se a estratégia for de preservação
        self.view.cluster_size_spin.setEnabled(is_cluster)
        
        # 2. Modo de Corte
        mode = self.view.mode_combo.currentData()
        is_ratio = (mode == SplitMode.RATIO)
        
        # Só permite editar a porcentagem se o modo não for Absoluto
        self.view.ratio_spin.setEnabled(is_ratio)

    def _load_config(self):
        """Preenche a UI com os dados do DTO se estivermos editando uma configuração existente."""
        self.view.name_input.setText(self.config_to_edit.name)
        
        # Usa string vazia como fallback se a descrição for None
        self.view.desc_input.setText(self.config_to_edit.description or "")
        
        self.view.grouping_combo.setCurrentText(self.config_to_edit.grouping_strategy.value)
        self.view.cluster_size_spin.setValue(self.config_to_edit.cluster_size)
        
        self.view.mode_combo.setCurrentText(self.config_to_edit.mode.value)
        self.view.ratio_spin.setValue(self.config_to_edit.calibration_ratio)
        self.view.min_pairs_spin.setValue(self.config_to_edit.min_calibration_pairs)
        self.view.min_ids_spin.setValue(self.config_to_edit.min_calibration_ids)
        
        self.view.min_samples_spin.setValue(self.config_to_edit.min_samples_per_id)
        self.view.seed_spin.setValue(self.config_to_edit.seed)
        
        # Força a atualização visual das travas baseada nos dados carregados
        self._update_ui_state()

    def _handle_save(self):
        """Valida a entrada do usuário e extrai os dados para o objeto DTO final."""
        name = self.view.name_input.text().strip()
        desc = self.view.desc_input.text().strip()
        
        # O nome é o único campo obrigatoriamente text-based (usado para salvar o JSON)
        if not name:
            QMessageBox.warning(self.view, "Aviso", "O nome da configuração é obrigatório.")
            return

        # Popula a variável de classe com o novo DTO
        self.final_config = SplitConfig(
            name=name,
            description=desc if desc else None,
            grouping_strategy=self.view.grouping_combo.currentData(),
            cluster_size=self.view.cluster_size_spin.value(),
            mode=self.view.mode_combo.currentData(),
            calibration_ratio=self.view.ratio_spin.value(),
            min_calibration_pairs=self.view.min_pairs_spin.value(),
            min_calibration_ids=self.view.min_ids_spin.value(),
            min_samples_per_id=self.view.min_samples_spin.value(),
            seed=self.view.seed_spin.value()
        )
        
        # Fecha a janela sinalizando sucesso (QDialog.Accepted)
        self.view.accept()

    def exec(self):
        """
        Roda o dialog de forma modal e retorna a configuração gerada.
        Returns:
            SplitConfig: O objeto configurado se o usuário clicou em Salvar.
            None: Se o usuário cancelou ou fechou a janela.
        """
        if self.view.exec():
            return self.final_config
        return None