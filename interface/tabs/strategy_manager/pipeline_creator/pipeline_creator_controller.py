# interface/tabs/strategy_manager/panels/preparation/pipeline_creator/pipeline_creator_controller.py
from PyQt6 import QtWidgets, QtCore
from typing import Optional, Dict, Any, Type
from dataclasses import asdict

from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.enums.transformation_type import TransformationType

# Importação dos Enums que montam a UI
from faceaudit.enums.coral_estimator_type import CoralEstimatorType
from faceaudit.enums.whitening_style import WhiteningStyle
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.wccn_estimator_type import WccnEstimatorType
from faceaudit.enums.wccn_grid_search_metric import WccnGridSearchMetric
from faceaudit.enums.lda_estimator_type import LdaEstimatorType
from faceaudit.enums.lda_grid_search_metric import LdaGridSearchMetric

from interface.theme import apply_themed_stylesheet

from .pipeline_creator_view import PipelineCreatorView

ROLE_DTO = QtCore.Qt.ItemDataRole.UserRole + 1

class PipelineCreatorController(QtCore.QObject):
    
    def __init__(
        self,
        parent,
        existing_config: Optional[TransformationPipelineConfig] = None,
        lock_name: bool = False,
    ):
        super().__init__(parent)
        self.view = PipelineCreatorView(parent)
        self.result_config: Optional[TransformationPipelineConfig] = None

        # --- Recebe o Objeto DTO diretamente ---
        self.existing_config = existing_config
        self.lock_name = bool(lock_name)

        if self.existing_config:
            self.view.setWindowTitle("Editar Pipeline de Transformação")

        self._populate_tools()
        self._connect_signals()

        # Preenche a tela se houver um DTO injetado
        if self.existing_config:
            self._load_existing_pipeline()

        # lock_name: usado quando a edição vem de um botão "✎" ao lado de um
        # dropdown na nova arch — o nome do recurso é o stem do arquivo no
        # disco e não pode ser alterado por aqui (renomear é operação separada
        # via left panel, ainda não implementada).
        if self.lock_name:
            self.view.input_name.setReadOnly(True)
            self.view.input_name.setToolTip(
                "O nome do recurso não pode ser alterado por aqui. "
                "Renomear o pipeline será uma operação dedicada no painel "
                "esquerdo (ainda não disponível)."
            )
            apply_themed_stylesheet(
                self.view.input_name,
                lambda t: (
                    "QLineEdit { background-color: " + t("SURFACE_ALT")
                    + "; color: " + t("TEXT") + "; }"
                ),
            )

    def run(self) -> Optional[TransformationPipelineConfig]:
        if self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return self.result_config
        return None

    # --- Carregamento Limpo via DTO ---
    def _load_existing_pipeline(self):
        try:
            self.view.input_name.setText(self.existing_config.name)
            if self.existing_config.description:
                self.view.input_desc.setText(self.existing_config.description)
            
            self.view.input_name.setToolTip("Altere o nome para criar uma cópia (Fork). Mantenha o nome para atualizar o atual.")
            
            for idx, transformator in enumerate(self.existing_config.transformators):
                t_type = transformator.type
                
                if isinstance(t_type, str):
                    try:
                        t_type = TransformationType(t_type)
                    except ValueError:
                        continue 
                
                label = getattr(t_type, 'label', t_type.value)
                icon = getattr(t_type, 'icon', '🔹')
                display_text = f"{icon}   {label}"
                
                new_item = QtWidgets.QListWidgetItem(display_text)
                new_item.setData(QtCore.Qt.ItemDataRole.UserRole, t_type)
                
                # INJEÇÃO: Em vez de params, salvamos o DTO instanciado na memória da UI
                new_item.setData(ROLE_DTO, transformator.config)
                
                self.view.list_sequence.addItem(new_item)

            self._renumber_sequence()
            
        except Exception as e:
            self.view.show_warning(f"Erro ao renderizar dados do pipeline: {e}")

    def _populate_tools(self):
        for t_type in TransformationType:
            icon = getattr(t_type, 'icon', '🔹')
            label = getattr(t_type, 'label', t_type.value)
            desc = getattr(t_type, 'description', "")
            display_text = f"{icon}   {label}"
            self.view.add_available_item(display_text, desc, t_type)

    def _connect_signals(self):
        self.view.btn_add.clicked.connect(self._add_item)
        self.view.btn_remove.clicked.connect(self._remove_item)
        self.view.btn_up.clicked.connect(self._move_up)
        self.view.btn_down.clicked.connect(self._move_down)
        self.view.buttons.accepted.connect(self._validate_and_close)
        self.view.list_sequence.itemSelectionChanged.connect(self._on_sequence_selection_changed)

    # --- Lógica de Sequência ---

    def _add_item(self):
        curr = self.view.list_available.currentItem()
        if not curr: return
        
        t_type = curr.data(QtCore.Qt.ItemDataRole.UserRole)
        new_item = QtWidgets.QListWidgetItem(curr.text())
        new_item.setData(QtCore.Qt.ItemDataRole.UserRole, t_type)
        
        # MÁGICA DO FACTORY: Se o Enum tiver um DTO mapeado, nós o instanciamos vazio (defaults)
        dto_instance = None
        if t_type.dto_class is not None:
            dto_instance = t_type.dto_class()
            
        new_item.setData(ROLE_DTO, dto_instance)
        new_item.setToolTip(curr.toolTip())
        
        self.view.list_sequence.addItem(new_item)
        self._renumber_sequence()
        self.view.list_sequence.setCurrentItem(new_item)

    def _remove_item(self):
        row = self.view.list_sequence.currentRow()
        if row >= 0:
            self.view.list_sequence.takeItem(row)
            self._renumber_sequence()
            self._on_sequence_selection_changed()

    def _move_up(self): self._move_item(-1)
    def _move_down(self): self._move_item(1)

    def _move_item(self, direction: int):
        row = self.view.list_sequence.currentRow()
        target = row + direction
        if 0 <= target < self.view.list_sequence.count():
            item = self.view.list_sequence.takeItem(row)
            self.view.list_sequence.insertItem(target, item)
            self.view.list_sequence.setCurrentRow(target)
            self._renumber_sequence()

    def _renumber_sequence(self):
        for i in range(self.view.list_sequence.count()):
            item = self.view.list_sequence.item(i)
            raw_text = item.text()
            if ". " in raw_text and raw_text.split(". ")[0].isdigit():
                raw_text = raw_text.split(". ", 1)[1]
            item.setText(f"{i + 1}. {raw_text}")

    # --- Lógica de Propriedades (Inspector) ---

    def _on_sequence_selection_changed(self):
        items = self.view.list_sequence.selectedItems()
        if not items:
            self.view.set_properties_widget(None)
            return

        item = items[0]
        t_type = item.data(QtCore.Qt.ItemDataRole.UserRole)

        # Recupera o DTO (pode ser None para L2/Centering)
        dto_config = item.data(ROLE_DTO)

        # Dispatch table: tipo → builder. Adicionar nova transformação
        # com params editáveis = adicionar uma linha aqui. Tipos stateless
        # (L2, Centering) ficam fora — caem no fallback "sem params".
        form_builders = {
            TransformationType.CORAL_TRANSFORMATION:   self._build_coral_form,
            TransformationType.LDA_TRANSFORMATION:     self._build_lda_form,
            TransformationType.WHITENING_TRANSFORMATION: self._build_whitening_form,
            TransformationType.WCCN_TRANSFORMATION:    self._build_wccn_form,
            TransformationType.IDENTITY_AGGREGATION:   self._build_aggregation_form,
            TransformationType.RADIAL_GAUSSIANIZATION: self._build_radial_gauss_form,
        }

        builder = form_builders.get(t_type)
        if builder is not None:
            widget = builder(dto_config, item)
        else:
            label = getattr(t_type, "label", str(t_type))
            widget = QtWidgets.QLabel(f"{label}\nnão possui parâmetros editáveis.")
            apply_themed_stylesheet(
                widget,
                lambda t: (
                    "color: " + t("TEXT_MUTED")
                    + "; font-style: italic; margin-top: 20px;"
                ),
            )
            widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.view.set_properties_widget(widget)

    def _update_dto_param(self, item: QtWidgets.QListWidgetItem, key: str, value: Any):
        """Atualiza uma propriedade diretamente no objeto DTO e salva no ListWidget."""
        dto_config = item.data(ROLE_DTO)
        if dto_config:
            setattr(dto_config, key, value)
            item.setData(ROLE_DTO, dto_config)

    # --- Construtores de Formulário ---

    def _build_coral_form(self, dto, item):
        w = QtWidgets.QWidget(); layout = QtWidgets.QFormLayout(w); layout.setContentsMargins(0,0,0,0)
        combo_est = QtWidgets.QComboBox()
        for est in CoralEstimatorType:
            combo_est.addItem(est.label, est)
            combo_est.setItemData(combo_est.count()-1, est.description, QtCore.Qt.ItemDataRole.ToolTipRole)

        curr_est = dto.estimator
        idx = combo_est.findData(curr_est)
        if idx >= 0: combo_est.setCurrentIndex(idx)
        combo_est.currentIndexChanged.connect(lambda: self._update_dto_param(item, 'estimator', combo_est.currentData()))
        layout.addRow("Estimador:", combo_est)

        spin_reg = QtWidgets.QDoubleSpinBox(); spin_reg.setDecimals(6); spin_reg.setRange(0.0, 1.0); spin_reg.setSingleStep(1e-6)
        spin_reg.setValue(dto.regularization)
        spin_reg.valueChanged.connect(lambda v: self._update_dto_param(item, 'regularization', v))
        lbl_reg = QtWidgets.QLabel("Regularização:"); layout.addRow(lbl_reg, spin_reg)
        
        chk_gs = QtWidgets.QCheckBox("Ativar Grid Search"); chk_gs.setChecked(dto.use_grid_search)
        chk_gs.toggled.connect(lambda v: self._update_dto_param(item, 'use_grid_search', v))
        layout.addRow("", chk_gs)

        def update_ui_state():
            est = combo_est.currentData()
            is_standard = (est == CoralEstimatorType.STANDARD)
            for x in [chk_gs, lbl_reg, spin_reg]: x.setVisible(is_standard)
            if is_standard: spin_reg.setEnabled(not chk_gs.isChecked())
        
        combo_est.currentIndexChanged.connect(update_ui_state); chk_gs.toggled.connect(update_ui_state); update_ui_state()
        return w

    def _build_lda_form(self, dto, item):
        """Form do LDA: n_components + estimator + shrinkage + ridge +
        grid search opcional. Paridade total com WCCN."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        # n_components com toggle auto (max = n_classes - 1).
        chk_auto = QtWidgets.QCheckBox("Automático (Max)")
        spin_n = QtWidgets.QSpinBox()
        spin_n.setRange(1, 4096)
        current_n = dto.n_components
        is_auto = current_n is None
        chk_auto.setChecked(is_auto)
        spin_n.setEnabled(not is_auto)
        if not is_auto:
            spin_n.setValue(current_n)

        def toggle_auto(checked):
            spin_n.setEnabled(not checked)
            self._update_dto_param(
                item, 'n_components', None if checked else spin_n.value()
            )
        spin_n.valueChanged.connect(
            lambda v: self._update_dto_param(
                item, 'n_components', v if not chk_auto.isChecked() else None
            )
        )
        chk_auto.toggled.connect(toggle_auto)
        layout.addRow("Dimensões:", chk_auto)
        layout.addRow(" ", spin_n)

        # Estimador de Sw (STANDARD / LEDOIT_WOLF / OAS).
        combo_est = QtWidgets.QComboBox()
        for est in LdaEstimatorType:
            combo_est.addItem(est.label, est)
            combo_est.setItemData(
                combo_est.count() - 1,
                est.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        idx = combo_est.findData(dto.estimator)
        if idx >= 0:
            combo_est.setCurrentIndex(idx)
        combo_est.currentIndexChanged.connect(
            lambda: self._update_dto_param(item, 'estimator', combo_est.currentData())
        )
        layout.addRow("Estimador:", combo_est)

        # Shrinkage explícito (combinável com qualquer estimator).
        spin_shrink = QtWidgets.QDoubleSpinBox()
        spin_shrink.setDecimals(4)
        spin_shrink.setRange(0.0, 1.0)
        spin_shrink.setSingleStep(0.01)
        spin_shrink.setValue(dto.shrinkage)
        spin_shrink.valueChanged.connect(
            lambda v: self._update_dto_param(item, 'shrinkage', v)
        )
        layout.addRow("Shrinkage (α):", spin_shrink)

        # Regularização (desabilita quando grid search ativo).
        spin_reg = QtWidgets.QDoubleSpinBox()
        spin_reg.setDecimals(8)
        spin_reg.setRange(1e-12, 1.0)
        spin_reg.setSingleStep(1e-4)
        spin_reg.setValue(dto.regularization)
        spin_reg.valueChanged.connect(
            lambda v: self._update_dto_param(item, 'regularization', v)
        )
        lbl_reg = QtWidgets.QLabel("Regularização:")
        layout.addRow(lbl_reg, spin_reg)

        # Grid search toggle.
        chk_gs = QtWidgets.QCheckBox("Ativar Grid Search (busca ε via holdout)")
        chk_gs.setChecked(dto.use_grid_search)
        chk_gs.toggled.connect(
            lambda v: self._update_dto_param(item, 'use_grid_search', v)
        )
        layout.addRow("", chk_gs)

        # Métrica do grid search.
        combo_metric = QtWidgets.QComboBox()
        for m in LdaGridSearchMetric:
            combo_metric.addItem(m.label, m)
            combo_metric.setItemData(
                combo_metric.count() - 1,
                m.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        idx = combo_metric.findData(dto.grid_search_metric)
        if idx >= 0:
            combo_metric.setCurrentIndex(idx)
        combo_metric.currentIndexChanged.connect(
            lambda: self._update_dto_param(
                item, 'grid_search_metric', combo_metric.currentData(),
            )
        )
        lbl_metric = QtWidgets.QLabel("Métrica (GS):")
        layout.addRow(lbl_metric, combo_metric)

        def update_ui_state():
            gs = chk_gs.isChecked()
            spin_reg.setEnabled(not gs)
            combo_metric.setEnabled(gs)
            lbl_metric.setEnabled(gs)
        chk_gs.toggled.connect(update_ui_state)
        update_ui_state()
        return w

    def _build_whitening_form(self, dto, item):
        w = QtWidgets.QWidget(); layout = QtWidgets.QFormLayout(w); layout.setContentsMargins(0,0,0,0)
        combo_style = QtWidgets.QComboBox()
        
        for s in WhiteningStyle:
            label = getattr(s, 'name', str(s))
            combo_style.addItem(label, s)
        
        idx = combo_style.findData(dto.style)
        if idx >= 0: combo_style.setCurrentIndex(idx)
        combo_style.currentIndexChanged.connect(lambda: self._update_dto_param(item, 'style', combo_style.currentData()))
        layout.addRow("Estilo:", combo_style)

        chk_auto = QtWidgets.QCheckBox("Todas as Dimensões"); spin_n = QtWidgets.QSpinBox(); spin_n.setRange(1, 4096)
        is_auto = dto.n_components is None
        chk_auto.setChecked(is_auto); spin_n.setEnabled(not is_auto)
        
        if not is_auto: spin_n.setValue(dto.n_components)
        
        def toggle_auto(checked):
            spin_n.setEnabled(not checked)
            self._update_dto_param(item, 'n_components', None if checked else spin_n.value())
            
        spin_n.valueChanged.connect(lambda v: self._update_dto_param(item, 'n_components', v if not chk_auto.isChecked() else None))
        chk_auto.toggled.connect(toggle_auto)
        layout.addRow("Componentes:", chk_auto); layout.addRow(" ", spin_n)
        return w

    def _build_wccn_form(self, dto, item):
        """Form de WCCN: estimator + shrinkage + regularization + grid search opcional.

        Layout idêntico ao CORAL (estimator + regularization + grid search
        toggle), mais o `shrinkage` (extra do WCCN: blend explícito com I)
        e a métrica do grid search (FISHER_RATIO ou COSINE_AUC).
        """
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        # Estimator (STANDARD / LEDOIT_WOLF / OAS).
        combo_est = QtWidgets.QComboBox()
        for est in WccnEstimatorType:
            combo_est.addItem(est.label, est)
            combo_est.setItemData(
                combo_est.count() - 1,
                est.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        idx = combo_est.findData(dto.estimator)
        if idx >= 0:
            combo_est.setCurrentIndex(idx)
        combo_est.currentIndexChanged.connect(
            lambda: self._update_dto_param(item, 'estimator', combo_est.currentData())
        )
        layout.addRow("Estimador:", combo_est)

        # Shrinkage explícito (sempre visível — combinável com qualquer estimator).
        spin_shrink = QtWidgets.QDoubleSpinBox()
        spin_shrink.setDecimals(4)
        spin_shrink.setRange(0.0, 1.0)
        spin_shrink.setSingleStep(0.01)
        spin_shrink.setValue(dto.shrinkage)
        spin_shrink.valueChanged.connect(
            lambda v: self._update_dto_param(item, 'shrinkage', v)
        )
        layout.addRow("Shrinkage (α):", spin_shrink)

        # Regularização (desabilitada quando grid search está ligado — ε vai
        # ser escolhido automaticamente da `reg_grid`).
        spin_reg = QtWidgets.QDoubleSpinBox()
        spin_reg.setDecimals(8)
        spin_reg.setRange(1e-12, 1.0)
        spin_reg.setSingleStep(1e-6)
        spin_reg.setValue(dto.regularization)
        spin_reg.valueChanged.connect(
            lambda v: self._update_dto_param(item, 'regularization', v)
        )
        lbl_reg = QtWidgets.QLabel("Regularização:")
        layout.addRow(lbl_reg, spin_reg)

        # Toggle de grid search.
        chk_gs = QtWidgets.QCheckBox("Ativar Grid Search (busca ε via holdout)")
        chk_gs.setChecked(dto.use_grid_search)
        chk_gs.toggled.connect(
            lambda v: self._update_dto_param(item, 'use_grid_search', v)
        )
        layout.addRow("", chk_gs)

        # Métrica do grid search (FISHER_RATIO / COSINE_AUC) — só faz
        # sentido quando grid search está ligado.
        combo_metric = QtWidgets.QComboBox()
        for m in WccnGridSearchMetric:
            combo_metric.addItem(m.label, m)
            combo_metric.setItemData(
                combo_metric.count() - 1,
                m.description,
                QtCore.Qt.ItemDataRole.ToolTipRole,
            )
        idx = combo_metric.findData(dto.grid_search_metric)
        if idx >= 0:
            combo_metric.setCurrentIndex(idx)
        combo_metric.currentIndexChanged.connect(
            lambda: self._update_dto_param(
                item, 'grid_search_metric', combo_metric.currentData(),
            )
        )
        lbl_metric = QtWidgets.QLabel("Métrica (GS):")
        layout.addRow(lbl_metric, combo_metric)

        def update_ui_state():
            gs = chk_gs.isChecked()
            spin_reg.setEnabled(not gs)
            combo_metric.setEnabled(gs)
            lbl_metric.setEnabled(gs)
        chk_gs.toggled.connect(update_ui_state)
        update_ui_state()
        return w

    def _build_aggregation_form(self, dto, item):
        """O NOVO form de Agregação baseado no AggregationConfig"""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Combo Method (Seleção da Estratégia) ---
        combo_method = QtWidgets.QComboBox()
        for m in AggregationMethod:
            # CORREÇÃO: Acessando via propriedade .info.label conforme definido no Enum
            label = m.info.label
            combo_method.addItem(label, m)
            # Opcional: Adiciona a descrição como ToolTip no combo
            combo_method.setItemData(combo_method.count()-1, m.info.description, QtCore.Qt.ItemDataRole.ToolTipRole)
            
        idx = combo_method.findData(dto.method)
        if idx >= 0: combo_method.setCurrentIndex(idx)
        combo_method.currentIndexChanged.connect(lambda: self._update_dto_param(item, 'method', combo_method.currentData()))
        layout.addRow("Estratégia:", combo_method)

        # --- Checkbox e Combo de Qualidade ---
        chk_quality = QtWidgets.QCheckBox("Usar pesos de qualidade")
        chk_quality.setChecked(dto.use_quality_weighting)
        
        combo_quality = QtWidgets.QComboBox()
        for q in QualityScoreType:
            # Aqui assumimos que QualityScoreType segue padrão similar ou tem .name
            q_label = getattr(q, 'display_name', q.name)
            combo_quality.addItem(q_label, q)
            
        idx = combo_quality.findData(dto.quality_weight_type)
        if idx >= 0: combo_quality.setCurrentIndex(idx)
        
        combo_quality.setEnabled(dto.use_quality_weighting)

        def toggle_quality(checked):
            combo_quality.setEnabled(checked)
            self._update_dto_param(item, 'use_quality_weighting', checked)
            
        chk_quality.toggled.connect(toggle_quality)
        combo_quality.currentIndexChanged.connect(lambda: self._update_dto_param(item, 'quality_weight_type', combo_quality.currentData()))
        
        layout.addRow("", chk_quality)
        layout.addRow("Métrica Qual.:", combo_quality)

        # --- Parâmetros Combinatórios (K e Max) ---
        spin_k = QtWidgets.QSpinBox()
        spin_k.setRange(2, 50)
        spin_k.setValue(dto.combination_group_size)
        spin_k.valueChanged.connect(lambda v: self._update_dto_param(item, 'combination_group_size', v))
        lbl_k = QtWidgets.QLabel("Tamanho do Grupo (K):")
        
        spin_max = QtWidgets.QSpinBox()
        spin_max.setRange(1, 9999)
        spin_max.setValue(dto.max_synthetic_templates_per_id)
        spin_max.valueChanged.connect(lambda v: self._update_dto_param(item, 'max_synthetic_templates_per_id', v))
        lbl_max = QtWidgets.QLabel("Máx Templates / ID:")

        layout.addRow(lbl_k, spin_k)
        layout.addRow(lbl_max, spin_max)

        def update_combo_vis():
            method = combo_method.currentData()
            is_combo = (method == AggregationMethod.COMBINATORIAL)
            lbl_k.setVisible(is_combo)
            spin_k.setVisible(is_combo)
            lbl_max.setVisible(is_combo)
            spin_max.setVisible(is_combo)
            
        combo_method.currentIndexChanged.connect(update_combo_vis)
        update_combo_vis()

        return w

    def _build_radial_gauss_form(self, dto, item):
        """
        Form pra `RadialGaussianizationConfig` (Lyu & Simoncelli 2009).

        Três parâmetros:
        - `eps_clip`: clamp dos quantis empíricos antes do `chi.ppf`.
          Sem isso, u=1.0 produz r=∞. Default 1e-6 (~ 1 amostra a cada 1M).
        - `assume_whitened`: quando True, o transformator emite WARNING
          se a covariância do source desviar muito de I (proxy de "não
          whitened"). Não impede a execução.
        - `whitening_check_tolerance`: tolerância pra detecção de desvio
          (±10% por padrão na média da diagonal da covariância).
        """
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        # eps_clip — escala log porque o valor útil varia de 1e-9 a 1e-3.
        spin_eps = QtWidgets.QDoubleSpinBox()
        spin_eps.setDecimals(9)
        spin_eps.setRange(1e-12, 1e-2)
        spin_eps.setSingleStep(1e-7)
        spin_eps.setValue(float(dto.eps_clip))
        spin_eps.setToolTip(
            "Clamp dos quantis empíricos antes do chi.ppf — evita "
            "r=∞ no inverso da CDF. Default 1e-6."
        )
        spin_eps.valueChanged.connect(
            lambda v: self._update_dto_param(item, "eps_clip", float(v))
        )
        layout.addRow("Clip de quantil (ε):", spin_eps)

        # assume_whitened
        chk_assume = QtWidgets.QCheckBox("Esperar dados whitened")
        chk_assume.setChecked(bool(dto.assume_whitened))
        chk_assume.setToolTip(
            "Quando marcado, emite WARNING no fit se a covariância "
            "desvia da identidade (proxy de 'não whitened'). RG só "
            "remove dependências em fontes elipticamente simétricas."
        )

        # whitening_check_tolerance — só relevante quando assume_whitened.
        spin_tol = QtWidgets.QDoubleSpinBox()
        spin_tol.setDecimals(3)
        spin_tol.setRange(0.0, 1.0)
        spin_tol.setSingleStep(0.01)
        spin_tol.setValue(float(dto.whitening_check_tolerance))
        spin_tol.setToolTip(
            "Tolerância pra detectar desvio de whitening (média da "
            "diagonal da covariância vs 1.0). 0.1 = ±10%."
        )
        spin_tol.valueChanged.connect(
            lambda v: self._update_dto_param(
                item, "whitening_check_tolerance", float(v),
            )
        )

        def toggle_assume(checked: bool):
            self._update_dto_param(item, "assume_whitened", bool(checked))
            spin_tol.setEnabled(checked)

        chk_assume.toggled.connect(toggle_assume)
        spin_tol.setEnabled(chk_assume.isChecked())

        layout.addRow("", chk_assume)
        layout.addRow("Tolerância whitening:", spin_tol)

        return w

    # --- Persistência (Validar e Fechar - SEM HD) ---
    def _validate_and_close(self):
        """Apenas constrói o objeto em memória e fecha a janela."""
        name = self.view.input_name.text().strip()
        if not name or self.view.list_sequence.count() == 0:
            self.view.show_warning("Valide o nome e a sequência do pipeline.")
            return

        transformers = []

        for i in range(self.view.list_sequence.count()):
            item = self.view.list_sequence.item(i)
            t_type = item.data(QtCore.Qt.ItemDataRole.UserRole)
            
            # Aqui pegamos o DTO pronto e blindado, ou None
            dto_config = item.data(ROLE_DTO)
            
            step_type_name = t_type.value if hasattr(t_type, 'value') else str(t_type)

            # Injetamos direto no atributo config da Factory
            transformers.append(TransformatorConfig(
                type=t_type,
                config=dto_config, 
                name=f"Step_{i+1}_{step_type_name}" 
            ))

        self.result_config = TransformationPipelineConfig(
            name=name,
            description=self.view.input_desc.text().strip(),
            transformators=transformers
        )

        self.view.accept()