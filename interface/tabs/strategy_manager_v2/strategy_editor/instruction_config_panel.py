# interface/tabs/strategy_manager_v2/strategy_editor/instruction_config_panel.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from PyQt6 import QtWidgets, QtCore

from faceaudit.models.instructions.pipeline_application_instruction import (
    PipelineApplicationInstruction,
)
from faceaudit.models.instructions.similarity_scoring_instruction import (
    SimilarityScoringInstruction,
)
from faceaudit.models.instructions.score_calibration_instruction import (
    ScoreCalibrationInstruction,
)
from faceaudit.models.instructions.strategy_evaluation_instruction import (
    StrategyEvaluationInstruction,
)
from faceaudit.models.instructions.casework_visualization_instruction import (
    CaseworkVisualizationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)

from .nested_dto_widget import (
    NESTED_DTO_ATTRS,
    NestedDTODialogTrigger,
    NestedDTOInlineWidget,
    get_inline_field_labels,
    get_inline_field_tooltips,
    get_inline_intro_text,
    get_inline_title,
    make_default_dto,
    open_mining_config_dialog,
    summarize_mining_config,
)
from .output_row_widget import OutputRowWidget
from .section_header import SectionHeader


_PAIR_SUFFIXES = ("_source", "_target")


_HEADER_TOOLTIPS: Dict[str, str] = {
    "training_policy":
        "Controla como os steps do pipeline usam o dataset target durante o "
        "treino DUAL. Os defaults já são adequados para o uso clássico de CORAL "
        "— modifique apenas para variantes de pesquisa específicas.",
    "application_policy":
        "Controla artefatos opcionais persistidos durante a aplicação do "
        "pipeline (checkpoints, dataset transformado, dados de trajetória).",
}


class InstructionConfigPanel(QtWidgets.QWidget):
    """
    Painel de configuração da instrução selecionada na lista.

    Layout em 3 seções verticais:
    - IDENTIFICAÇÃO: nome, descrição
    - CONFIGURAÇÃO: choice fields, resource inputs (com pareamento Source/Target),
      DTOs aninhados (inline ou via dialog)
    - SAÍDAS: uma OutputRowWidget por output declarado, com status + resumo

    Sinais:
        config_changed(StrategyInstructionDefinition) — alteração aplicada
        validate_requested(StrategyInstructionDefinition) — usuário clicou em Validar
    """

    config_changed = QtCore.pyqtSignal(object)
    instruction_applied = QtCore.pyqtSignal(object)
    validate_requested = QtCore.pyqtSignal(object)
    view_trajectory_requested = QtCore.pyqtSignal(object)
    view_score_distribution_requested = QtCore.pyqtSignal(object)
    view_validation_result_requested = QtCore.pyqtSignal(object)
    # Emitido quando o usuário clica no botão "Visualizar Caso" de uma
    # CaseworkVisualizationInstruction. O controller resolve, hidrata o
    # CaseworkVisualizationResult do disco se necessário, e abre o
    # CaseworkVisualizationViewerDialog.
    view_casework_visualization_requested = QtCore.pyqtSignal(object)
    # Emitido quando o usuário clica no botão ✎ ao lado de um dropdown de
    # config editável (PIPELINE_CONFIG, MINING_CONFIG, ...). O controller é
    # responsável por hidratar o payload, abrir o dialog de edição apropriado
    # ao tipo, atualizar o payload e re-persistir o artefato.
    resource_edit_requested = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, outputs_in_results_panel: bool = False):
        """
        Args:
            parent: pai Qt opcional.
            outputs_in_results_panel: quando True, a seção "Saídas" e os
                botões de visualização (📊 Trajetória, 📊 Distribuição,
                📊 Validação, 📊 Caso) ficam ocultos — usado quando o
                editor delega esses elementos ao `InstructionResultsPanel`
                (3ª coluna). Default False preserva o layout antigo
                (instâncias usadas isoladamente em outros contextos).
        """
        super().__init__(parent)
        self._outputs_in_results_panel = bool(outputs_in_results_panel)

        self._instruction: Optional[StrategyInstructionDefinition] = None
        self._instruction_snapshot: Optional[dict] = None
        self._available_resources: List[StrategyResource] = []

        # Estado de runtime para colorir as Saídas após execução
        self._latest_definition: Optional[LRStrategyDefinition] = None
        self._latest_executed_ids: Set[str] = set()
        self._latest_context: Optional[StrategyExecutionContextProtocol] = None

        # Widgets internos do form (recriados em cada _rebuild_form)
        self._field_name: Optional[QtWidgets.QLineEdit] = None
        self._field_desc: Optional[QtWidgets.QPlainTextEdit] = None
        self._choice_field_widgets: Dict[str, QtWidgets.QComboBox] = {}
        self._choice_specs: Dict[str, object] = {}
        self._resource_field_widgets: Dict[str, QtWidgets.QWidget] = {}
        self._value_field_widgets: Dict[str, QtWidgets.QWidget] = {}
        self._value_specs: Dict[str, ValueFieldSpec] = {}
        self._param_fields: Dict[str, QtWidgets.QWidget] = {}
        self._nested_inline_widgets: Dict[str, NestedDTOInlineWidget] = {}
        self._nested_dialog_widgets: Dict[str, NestedDTODialogTrigger] = {}
        self._output_widgets: Dict[str, OutputRowWidget] = {}

        # Layouts das três seções (criados uma vez em _init_ui)
        self._form_id: Optional[QtWidgets.QFormLayout] = None
        self._form_cfg: Optional[QtWidgets.QFormLayout] = None
        self._outputs_layout: Optional[QtWidgets.QVBoxLayout] = None

        self._init_ui()
        self.show_empty()

    # ── UI build ──────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        self._root_layout = QtWidgets.QVBoxLayout(self)
        self._root_layout.setContentsMargins(8, 8, 8, 8)
        self._root_layout.setSpacing(6)

        self.stack = QtWidgets.QStackedWidget()
        self._root_layout.addWidget(self.stack, 1)

        # Página 0 — vazia
        empty_page = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_page)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl = QtWidgets.QLabel("Selecione uma instrução para configurar.")
        lbl.setStyleSheet("color: #888; font-size: 13px;")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl)
        self.stack.addWidget(empty_page)   # index 0

        # Página 1 — formulário em 3 seções
        self.form_page = QtWidgets.QWidget()
        form_root = QtWidgets.QVBoxLayout(self.form_page)
        form_root.setContentsMargins(0, 0, 0, 0)
        form_root.setSpacing(4)

        # Header bar (tipo + id)
        self._header_lbl = QtWidgets.QLabel()
        self._header_lbl.setStyleSheet(
            "padding: 4px 8px; background: #2c3e50; color: white;"
            " font-size: 11px; font-weight: bold; border-radius: 3px;"
        )
        form_root.addWidget(self._header_lbl)

        # Scroll area para o conteúdo das seções
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(2)
        scroll_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # ─── Seção 1: Identificação ───────────────────────────────────────
        scroll_layout.addWidget(SectionHeader("Identificação", "identification"))
        self._form_id = QtWidgets.QFormLayout()
        self._form_id.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._form_id.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._form_id.setContentsMargins(8, 6, 8, 8)
        self._form_id.setSpacing(6)
        wrapper_id = QtWidgets.QWidget()
        wrapper_id.setLayout(self._form_id)
        scroll_layout.addWidget(wrapper_id)

        # ─── Seção 2: Configuração ────────────────────────────────────────
        scroll_layout.addWidget(SectionHeader("Configuração", "configuration"))
        self._form_cfg = QtWidgets.QFormLayout()
        self._form_cfg.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._form_cfg.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._form_cfg.setContentsMargins(8, 6, 8, 8)
        self._form_cfg.setSpacing(6)
        wrapper_cfg = QtWidgets.QWidget()
        wrapper_cfg.setLayout(self._form_cfg)
        scroll_layout.addWidget(wrapper_cfg)

        # ─── Seção 3: Saídas ──────────────────────────────────────────────
        # Quando o painel de resultados (3ª coluna) está ativo, esta seção
        # fica oculta — outputs são renderizados lá. Mas o layout/dict
        # ainda é criado para preservar o caminho antigo (chamadas a
        # update_runtime_state via legacy callers ou tests).
        if not self._outputs_in_results_panel:
            scroll_layout.addWidget(SectionHeader("Saídas", "outputs"))
            self._outputs_layout = QtWidgets.QVBoxLayout()
            self._outputs_layout.setContentsMargins(8, 6, 8, 8)
            self._outputs_layout.setSpacing(6)
            wrapper_out = QtWidgets.QWidget()
            wrapper_out.setLayout(self._outputs_layout)
            scroll_layout.addWidget(wrapper_out)
        else:
            # Layout vazio (não embedded em widget) — usado pelos métodos
            # internos pra evitar branches; fica orphan e sem efeito visual.
            self._outputs_layout = QtWidgets.QVBoxLayout()

        scroll_layout.addStretch(1)  # ANCORA conteúdo ao topo

        scroll.setWidget(scroll_content)
        form_root.addWidget(scroll, 1)

        # Botões: ações específicas à esquerda (ex: Visualizar Trajetória 3D)
        # e ações genéricas (Validar / Aplicar) à direita.
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 6, 0, 0)

        # Quando o painel de resultados (3ª coluna) está ativo, os 4 botões
        # de visualização migram pra lá — neste painel só ficam Validar/Aplicar.
        # Os botões ainda são instanciados (e mantidos como atributos) mas
        # invisíveis, preservando referências internas que outros métodos
        # consultam.
        _vis_buttons_visible = not self._outputs_in_results_panel

        # Botão de visualização da trajetória — visível apenas em
        # PipelineApplicationInstruction; habilitado quando há trajetória
        # disponível em memória ou em disco.
        self.btn_view_trajectory = QtWidgets.QPushButton("📊 Visualizar Trajetória 3D")
        self.btn_view_trajectory.setMinimumWidth(220)
        self.btn_view_trajectory.setStyleSheet(
            "QPushButton { border: 1px solid #E8A020; border-radius: 4px;"
            " background: #FFF8E8; color: #8B5E00;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #FFEFC4; }"
            "QPushButton:disabled { color: #aaa; background: #f5f5f5;"
            " border: 1px solid #ccc; }"
        )
        self.btn_view_trajectory.setVisible(False)
        self.btn_view_trajectory.setEnabled(False)
        self.btn_view_trajectory.clicked.connect(self._on_view_trajectory)
        if _vis_buttons_visible:
            btn_row.addWidget(self.btn_view_trajectory)

        # Botão de visualização da distribuição de scores — visível apenas em
        # SimilarityScoringInstruction; habilitado quando há scores disponíveis.
        self.btn_view_distribution = QtWidgets.QPushButton(
            "📊 Visualizar Distribuição de Scores"
        )
        self.btn_view_distribution.setMinimumWidth(240)
        self.btn_view_distribution.setStyleSheet(
            "QPushButton { border: 1px solid #E8A020; border-radius: 4px;"
            " background: #FFF8E8; color: #8B5E00;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #FFEFC4; }"
            "QPushButton:disabled { color: #aaa; background: #f5f5f5;"
            " border: 1px solid #ccc; }"
        )
        self.btn_view_distribution.setVisible(False)
        self.btn_view_distribution.setEnabled(False)
        self.btn_view_distribution.clicked.connect(self._on_view_distribution)
        if _vis_buttons_visible:
            btn_row.addWidget(self.btn_view_distribution)

        # Botão de visualização do resultado de validação — visível apenas
        # em StrategyEvaluationInstruction; habilitado quando o recurso de
        # validação está disponível (em memória ou em disco).
        self.btn_view_validation_result = QtWidgets.QPushButton(
            "📊 Visualizar Resultado de Validação"
        )
        self.btn_view_validation_result.setMinimumWidth(260)
        self.btn_view_validation_result.setStyleSheet(
            "QPushButton { border: 1px solid #E8A020; border-radius: 4px;"
            " background: #FFF8E8; color: #8B5E00;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #FFEFC4; }"
            "QPushButton:disabled { color: #aaa; background: #f5f5f5;"
            " border: 1px solid #ccc; }"
        )
        self.btn_view_validation_result.setVisible(False)
        self.btn_view_validation_result.setEnabled(False)
        self.btn_view_validation_result.clicked.connect(self._on_view_validation_result)
        if _vis_buttons_visible:
            btn_row.addWidget(self.btn_view_validation_result)

        # Botão de visualização forense de caso — visível apenas em
        # CaseworkVisualizationInstruction; habilitado quando o recurso de
        # CaseworkVisualizationResult está disponível (em memória ou disco).
        self.btn_view_casework_visualization = QtWidgets.QPushButton(
            "📊 Visualizar Caso"
        )
        self.btn_view_casework_visualization.setMinimumWidth(220)
        self.btn_view_casework_visualization.setStyleSheet(
            "QPushButton { border: 1px solid #E8A020; border-radius: 4px;"
            " background: #FFF8E8; color: #8B5E00;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #FFEFC4; }"
            "QPushButton:disabled { color: #aaa; background: #f5f5f5;"
            " border: 1px solid #ccc; }"
        )
        self.btn_view_casework_visualization.setVisible(False)
        self.btn_view_casework_visualization.setEnabled(False)
        self.btn_view_casework_visualization.clicked.connect(
            self._on_view_casework_visualization
        )
        if _vis_buttons_visible:
            btn_row.addWidget(self.btn_view_casework_visualization)

        btn_row.addStretch(1)

        self.btn_validate = QtWidgets.QPushButton("Validar")
        self.btn_validate.setMinimumWidth(120)
        self.btn_validate.setStyleSheet(
            "QPushButton { border: 1px solid #1565c0; border-radius: 4px;"
            " background: #e3f2fd; color: #1565c0;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #bbdefb; }"
        )
        self.btn_validate.clicked.connect(self._on_validate)
        btn_row.addWidget(self.btn_validate)

        self.btn_apply = QtWidgets.QPushButton("Aplicar alterações")
        self.btn_apply.setMinimumWidth(160)
        self.btn_apply.setStyleSheet(
            "QPushButton { border: 1px solid #2e7d32; border-radius: 4px;"
            " background: #e8f5e9; color: #2e7d32;"
            " padding: 5px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #c8e6c9; }"
        )
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self.btn_apply)

        form_root.addLayout(btn_row)

        self.stack.addWidget(self.form_page)   # index 1

    # ── Public API ────────────────────────────────────────────────────────────

    def set_available_resources(self, resources: List[StrategyResource]) -> None:
        self._available_resources = list(resources)
        if self._instruction is not None:
            self._rebuild_form()

    def load_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        self._instruction = instruction
        # Snapshot da instrução no momento do load — usado em _on_apply para
        # detectar quando o usuário clica "Aplicar Alterações" SEM ter
        # alterado nada de fato (no-op): nesse caso não emitimos sinais que
        # invalidariam os outputs persistidos.
        self._instruction_snapshot = self._snapshot_instruction(instruction)
        self._rebuild_form()
        self.stack.setCurrentIndex(1)

    def show_empty(self) -> None:
        self._instruction = None
        self._instruction_snapshot = None
        self.stack.setCurrentIndex(0)

    @staticmethod
    def _snapshot_instruction(instr: StrategyInstructionDefinition):
        """
        Tira um snapshot serializável da instrução para comparação posterior.
        Falha de serialização cai para None — no caso, _on_apply trata como
        "sempre modificado" (comportamento conservador).
        """
        try:
            return instr.to_dict()
        except Exception:
            return None

    def update_runtime_state(
        self,
        definition: LRStrategyDefinition,
        executed_ids: Set[str],
        context: Optional[StrategyExecutionContextProtocol] = None,
    ) -> None:
        """Atualiza o estado visual das Saídas com base no runtime atual."""
        self._latest_definition = definition
        self._latest_executed_ids = executed_ids or set()
        self._latest_context = context

        if self._instruction is None:
            return

        executed = self._instruction.instruction_id in self._latest_executed_ids
        for row in self._output_widgets.values():
            row.update_state(
                definition=definition,
                instruction_id=self._instruction.instruction_id,
                executed=executed,
                context=context,
            )

        self._refresh_view_trajectory_button_state()
        self._refresh_view_distribution_button_state()
        self._refresh_view_validation_result_button_state()
        self._refresh_view_casework_visualization_button_state()

    def _refresh_view_distribution_button_state(self) -> None:
        """
        Atualiza a habilitação do botão "Visualizar Distribuição de Scores"
        com base na disponibilidade do recurso de scores (em memória ou em disco).

        Aplicável a SIMILARITY_SCORING (raw_score) e a SCORE_CALIBRATION
        (log_lr calibrado) — ambos produzem SimilarityScoreDataset.
        """
        if not isinstance(
            self._instruction,
            (SimilarityScoringInstruction, ScoreCalibrationInstruction),
        ):
            self.btn_view_distribution.setEnabled(False)
            return

        scores_id = self._resolve_scores_resource_id()
        if scores_id is None:
            self.btn_view_distribution.setEnabled(False)
            return

        has_data = False
        if self._latest_context is not None and self._latest_context.has_resource(scores_id):
            ctx_resource = self._latest_context.get_resource(scores_id)
            if ctx_resource and ctx_resource.has_payload:
                has_data = True

        if not has_data and self._latest_definition is not None:
            existing = self._latest_definition.get_resource_by_id(scores_id)
            if existing and (existing.has_artifact or existing.has_payload):
                has_data = True

        self.btn_view_distribution.setEnabled(has_data)

    def _resolve_scores_resource_id(self) -> Optional[str]:
        """
        Retorna o resource_id do output de scores da instrução atual ou None.

        Para SIMILARITY_SCORING → output_similarity_scores_resource_id (raw).
        Para SCORE_CALIBRATION → output_calibrated_scores_resource_id (log-LR).
        """
        if isinstance(self._instruction, SimilarityScoringInstruction):
            return self._instruction.output_similarity_scores_resource_id or None
        if isinstance(self._instruction, ScoreCalibrationInstruction):
            return self._instruction.output_calibrated_scores_resource_id or None
        return None

    def _refresh_view_validation_result_button_state(self) -> None:
        """
        Atualiza a habilitação do botão "Visualizar Resultado de Validação"
        com base na disponibilidade do recurso VALIDATION_RESULT (em memória
        ou em disco). Aplicável apenas a StrategyEvaluationInstruction.
        """
        if not isinstance(self._instruction, StrategyEvaluationInstruction):
            self.btn_view_validation_result.setEnabled(False)
            return

        result_id = self._resolve_validation_result_resource_id()
        if result_id is None:
            self.btn_view_validation_result.setEnabled(False)
            return

        has_data = False
        if self._latest_context is not None and self._latest_context.has_resource(result_id):
            ctx_resource = self._latest_context.get_resource(result_id)
            if ctx_resource and ctx_resource.has_payload:
                has_data = True

        if not has_data and self._latest_definition is not None:
            existing = self._latest_definition.get_resource_by_id(result_id)
            if existing and (existing.has_artifact or existing.has_payload):
                has_data = True

        self.btn_view_validation_result.setEnabled(has_data)

    def _resolve_validation_result_resource_id(self) -> Optional[str]:
        """Retorna o resource_id do output de validação ou None."""
        if not isinstance(self._instruction, StrategyEvaluationInstruction):
            return None
        return self._instruction.output_validation_result_resource_id or None

    def _refresh_view_trajectory_button_state(self) -> None:
        """
        Atualiza a habilitação do botão "Visualizar Trajetória 3D" com base
        na disponibilidade do recurso de trajetória (em memória ou em disco).
        """
        if not isinstance(self._instruction, PipelineApplicationInstruction):
            self.btn_view_trajectory.setEnabled(False)
            return

        trajectory_id = self._resolve_trajectory_resource_id()
        if trajectory_id is None:
            self.btn_view_trajectory.setEnabled(False)
            return

        has_data = False

        # Em memória — recurso no contexto com payload.
        if self._latest_context is not None and self._latest_context.has_resource(trajectory_id):
            ctx_resource = self._latest_context.get_resource(trajectory_id)
            if ctx_resource and ctx_resource.has_payload:
                has_data = True

        # Em disco — recurso na definição com artifact_ref persistido.
        if not has_data and self._latest_definition is not None:
            existing = self._latest_definition.get_resource_by_id(trajectory_id)
            if existing and (existing.has_artifact or existing.has_payload):
                has_data = True

        self.btn_view_trajectory.setEnabled(has_data)

    def _resolve_trajectory_resource_id(self) -> Optional[str]:
        """Retorna o resource_id da trajetória 3D da instrução atual ou None."""
        if not isinstance(self._instruction, PipelineApplicationInstruction):
            return None
        explicit = self._instruction.trajectory_data_resource_id
        if explicit:
            return explicit
        # Fallback à convenção usada pelo executor quando o id não foi atribuído.
        return f"{self._instruction.instruction_id}_trajectory"

    # ── Form building ─────────────────────────────────────────────────────────

    def _clear_form_layout(self, form_layout: QtWidgets.QFormLayout) -> None:
        while form_layout.rowCount():
            form_layout.removeRow(0)

    def _clear_outputs_layout(self) -> None:
        while self._outputs_layout.count():
            item = self._outputs_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_form(self) -> None:
        instr = self._instruction
        if not instr:
            return

        # Bloqueia sinais dos combos antigos para evitar efeitos colaterais
        for combo in self._choice_field_widgets.values():
            combo.blockSignals(True)

        # Reset widget caches
        self._choice_field_widgets = {}
        self._choice_specs = {}
        self._resource_field_widgets = {}
        self._value_field_widgets = {}
        self._value_specs = {}
        self._param_fields = {}
        self._nested_inline_widgets = {}
        self._nested_dialog_widgets = {}
        self._output_widgets = {}

        # Limpa as três seções
        self._clear_form_layout(self._form_id)
        self._clear_form_layout(self._form_cfg)
        self._clear_outputs_layout()

        # ─── Header bar ──────────────────────────────────────────────────
        self._header_lbl.setText(
            f"  {instr.instruction_type.value}    "
            f"<span style='color:#aac7e0; font-weight:normal;'>id: {instr.instruction_id}</span>"
        )

        # ─── Seção Identificação ────────────────────────────────────────
        self._field_name = QtWidgets.QLineEdit(instr.name or "")
        self._form_id.addRow("Nome:", self._field_name)

        self._field_desc = QtWidgets.QPlainTextEdit(instr.description or "")
        self._field_desc.setMaximumHeight(54)
        self._form_id.addRow("Descrição:", self._field_desc)

        # ─── Seção Configuração ─────────────────────────────────────────
        self._build_configuration_section(instr)

        # ─── Seção Saídas ────────────────────────────────────────────────
        self._build_outputs_section(instr)

        # Visibilidade do botão "Visualizar Trajetória 3D" — apenas em PIPELINE_APPLICATION.
        self.btn_view_trajectory.setVisible(
            isinstance(instr, PipelineApplicationInstruction)
        )

        # Visibilidade do botão "Visualizar Distribuição de Scores" —
        # SIMILARITY_SCORING (raw_score) e SCORE_CALIBRATION (log_lr calibrado).
        self.btn_view_distribution.setVisible(
            isinstance(instr, (SimilarityScoringInstruction, ScoreCalibrationInstruction))
        )

        # Visibilidade do botão "Visualizar Resultado de Validação" —
        # apenas em STRATEGY_EVALUATION.
        self.btn_view_validation_result.setVisible(
            isinstance(instr, StrategyEvaluationInstruction)
        )

        # Visibilidade do botão "Visualizar Caso" — apenas em
        # CASEWORK_VISUALIZATION.
        self.btn_view_casework_visualization.setVisible(
            isinstance(instr, CaseworkVisualizationInstruction)
        )
        self._refresh_view_casework_visualization_button_state()

        # Aplica estado de runtime se houver
        if self._latest_definition is not None:
            executed = instr.instruction_id in self._latest_executed_ids
            for row in self._output_widgets.values():
                row.update_state(
                    definition=self._latest_definition,
                    instruction_id=instr.instruction_id,
                    executed=executed,
                    context=self._latest_context,
                )

        self._refresh_view_trajectory_button_state()
        self._refresh_view_distribution_button_state()
        self._refresh_view_validation_result_button_state()
        self._refresh_view_casework_visualization_button_state()

    def _build_configuration_section(self, instr: StrategyInstructionDefinition) -> None:
        # Ajuda contextual da instrução (texto explicativo opcional no topo
        # da seção). Renderizada como caixa âmbar — mesmo padrão visual da
        # descrição contextual de choice fields, para coerência cromática.
        help_text = instr.get_configuration_help() if hasattr(instr, "get_configuration_help") else None
        if help_text:
            help_box = QtWidgets.QFrame()
            help_box.setStyleSheet(
                "QFrame { background-color: #FFF8E8;"
                " border-left: 3px solid #E8A020;"
                " border-radius: 2px;"
                " margin: 0px 0px 6px 0px; }"
            )
            help_layout = QtWidgets.QHBoxLayout(help_box)
            help_layout.setContentsMargins(8, 6, 8, 6)
            help_lbl = QtWidgets.QLabel(help_text)
            help_lbl.setStyleSheet(
                "background: transparent; border: none;"
                " color: #5a4500; font-size: 11px;"
            )
            help_lbl.setWordWrap(True)
            help_layout.addWidget(help_lbl, 1)
            self._form_cfg.addRow(help_box)

        # Choice fields (ex: pipeline_mode LINEAR/DUAL)
        choice_specs = instr.get_choice_field_specs()
        for cspec in choice_specs:
            combo = QtWidgets.QComboBox()
            combo.addItems(cspec.choices)

            raw_val = getattr(instr, cspec.field_name, None)
            if raw_val is not None:
                if hasattr(raw_val, "name"):
                    current_str = raw_val.name
                else:
                    current_str = str(raw_val)
                    current_str = current_str.split(".")[-1] if "." in current_str else current_str

                choices_upper = {c.upper(): c for c in cspec.choices}
                match = choices_upper.get(current_str.upper())
                if match:
                    combo.setCurrentText(match)

            self._choice_field_widgets[cspec.field_name] = combo
            self._choice_specs[cspec.field_name] = cspec
            self._form_cfg.addRow(f"{cspec.label}:", combo)
            combo.currentTextChanged.connect(
                lambda _text, fn=cspec.field_name: self._on_choice_changed(fn)
            )

            # Linha de ajuda contextual logo abaixo do combo, atualizada
            # imediatamente conforme a seleção (antes do rebuild completo).
            if cspec.descriptions:
                desc_box = QtWidgets.QFrame()
                desc_box.setStyleSheet(
                    "QFrame { background-color: #FFF8E8;"
                    " border-left: 3px solid #E8A020;"
                    " border-radius: 2px;"
                    " margin: 0px 0px 4px 4px; }"
                )
                desc_layout = QtWidgets.QHBoxLayout(desc_box)
                desc_layout.setContentsMargins(8, 4, 8, 4)
                desc_layout.setSpacing(4)

                desc_lbl = QtWidgets.QLabel(
                    cspec.descriptions.get(combo.currentText(), "")
                )
                desc_lbl.setStyleSheet(
                    "background: transparent; border: none;"
                    " color: #5a4500; font-size: 11px;"
                )
                desc_lbl.setWordWrap(True)
                desc_layout.addWidget(desc_lbl, 1)

                # Spans both columns for full width — limpo e compacto.
                self._form_cfg.addRow(desc_box)
                combo.currentTextChanged.connect(
                    lambda text, lbl=desc_lbl, ds=cspec.descriptions:
                    lbl.setText(ds.get(text, ""))
                )

        # Resource fields — somente os de input (outputs vão para a seção Saídas)
        specs = instr.get_resource_field_specs()
        input_specs = [s for s in specs if not s.is_output]

        # Detecta pares Source/Target pelo sufixo do field_name
        rendered_field_names: Set[str] = set()
        i = 0
        while i < len(input_specs):
            spec = input_specs[i]
            if spec.field_name in rendered_field_names:
                i += 1
                continue

            pair_partner = self._find_pair_partner(spec, input_specs)
            if pair_partner is not None:
                self._render_paired_inputs(spec, pair_partner)
                rendered_field_names.add(spec.field_name)
                rendered_field_names.add(pair_partner.field_name)
            else:
                combo = self._make_input_resource_combo(instr, spec)
                self._resource_field_widgets[spec.field_name] = combo
                row_widget = self._wrap_combo_with_edit_button_if_needed(combo, spec)
                self._form_cfg.addRow(f"{spec.label}:", row_widget)
                rendered_field_names.add(spec.field_name)

            i += 1

        # DTOs aninhados (PipelineApplicationPolicy / ComparisonMiningConfig /
        # configs de modelo de similaridade e calibração).
        for attr_name, render_strategy in NESTED_DTO_ATTRS:
            if not hasattr(instr, attr_name):
                continue

            # Configs de modelo só renderizam quando o tipo correspondente
            # está selecionado na instrução.
            if attr_name == "similarity_model_config" and not getattr(
                instr, "uses_plda", False
            ):
                continue
            if attr_name == "logreg_config" and not getattr(
                instr, "uses_logistic_regression", False
            ):
                continue
            if attr_name == "kde_config" and not getattr(
                instr, "uses_kde", False
            ):
                continue

            current_dto = getattr(instr, attr_name, None)

            if render_strategy == "inline":
                if current_dto is None:
                    current_dto = make_default_dto(attr_name)
                    setattr(instr, attr_name, current_dto)
                widget = NestedDTOInlineWidget(
                    title=get_inline_title(attr_name),
                    dto=current_dto,
                    field_labels=get_inline_field_labels(attr_name),
                    field_tooltips=get_inline_field_tooltips(attr_name),
                    header_tooltip=_HEADER_TOOLTIPS.get(attr_name),
                    intro_text=get_inline_intro_text(attr_name),
                )
                widget.value_changed.connect(
                    lambda new_dto, an=attr_name: self._on_nested_changed(an, new_dto)
                )
                self._nested_inline_widgets[attr_name] = widget
                self._form_cfg.addRow(widget)

            elif render_strategy == "dialog":
                widget = NestedDTODialogTrigger(
                    title=get_inline_title(attr_name),
                    dto=current_dto,
                    summary_fn=summarize_mining_config,
                    open_dialog_fn=open_mining_config_dialog,
                )
                widget.value_changed.connect(
                    lambda new_dto, an=attr_name: self._on_nested_changed(an, new_dto)
                )
                self._nested_dialog_widgets[attr_name] = widget
                self._form_cfg.addRow(widget)

        # Campos primitivos do DTO declarados via get_value_field_specs
        # (ex: split_ratio, random_seed, identity_list, output_dataset_a_name).
        for vspec in instr.get_value_field_specs():
            current = getattr(instr, vspec.field_name, None)
            widget = self._make_value_widget(vspec, current)
            if widget is None:
                continue
            self._value_field_widgets[vspec.field_name] = widget
            self._value_specs[vspec.field_name] = vspec
            self._form_cfg.addRow(f"{vspec.label}:", widget)

            # Conexão de rebuild para campos cujo valor altera o conjunto de
            # specs subsequentes (ex: bool que habilita outputs condicionais).
            if vspec.triggers_rebuild:
                if isinstance(widget, QtWidgets.QCheckBox):
                    widget.toggled.connect(
                        lambda _checked, fn=vspec.field_name:
                        self._on_value_triggers_rebuild(fn)
                    )
                elif isinstance(widget, QtWidgets.QComboBox):
                    widget.currentTextChanged.connect(
                        lambda _text, fn=vspec.field_name:
                        self._on_value_triggers_rebuild(fn)
                    )
                elif isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                    widget.valueChanged.connect(
                        lambda _v, fn=vspec.field_name:
                        self._on_value_triggers_rebuild(fn)
                    )
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.editingFinished.connect(
                        lambda fn=vspec.field_name:
                        self._on_value_triggers_rebuild(fn)
                    )

        # Parâmetros genéricos (parameters dict)
        for key, val in (instr.parameters or {}).items():
            if isinstance(val, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(val)
            elif isinstance(val, (int, float)):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setValue(float(val))
            else:
                widget = QtWidgets.QLineEdit(str(val) if val is not None else "")
            self._param_fields[key] = widget
            self._form_cfg.addRow(f"{key}:", widget)

    def _make_value_widget(
        self, spec: ValueFieldSpec, current: Any
    ) -> Optional[QtWidgets.QWidget]:
        """Cria o widget apropriado para um ValueFieldSpec."""
        vt = spec.value_type

        if vt == "bool":
            w = QtWidgets.QCheckBox()
            w.setChecked(bool(current))
            return w

        if vt == "int":
            w = QtWidgets.QSpinBox()
            lo = int(spec.min_value) if spec.min_value is not None else -(10 ** 9)
            hi = int(spec.max_value) if spec.max_value is not None else (10 ** 9)
            w.setRange(lo, hi)
            if current is not None:
                try:
                    w.setValue(int(current))
                except Exception:
                    pass
            return w

        if vt == "float":
            w = QtWidgets.QDoubleSpinBox()
            lo = float(spec.min_value) if spec.min_value is not None else -1e9
            hi = float(spec.max_value) if spec.max_value is not None else 1e9
            w.setRange(lo, hi)
            if spec.decimals is not None:
                w.setDecimals(spec.decimals)
                w.setSingleStep(10 ** -spec.decimals)
            if current is not None:
                try:
                    w.setValue(float(current))
                except Exception:
                    pass
            return w

        if vt == "str":
            w = QtWidgets.QLineEdit(str(current) if current else "")
            if spec.placeholder:
                w.setPlaceholderText(spec.placeholder)
            return w

        if vt in ("list_str", "list_int"):
            text = ""
            if isinstance(current, (list, tuple)):
                text = ", ".join(str(x) for x in current)
            w = QtWidgets.QLineEdit(text)
            w.setPlaceholderText(spec.placeholder or "valores separados por vírgula")
            return w

        return None

    def _read_value_widget(
        self, widget: QtWidgets.QWidget, spec: ValueFieldSpec
    ) -> Any:
        """Lê o valor atual do widget conforme o value_type do spec."""
        vt = spec.value_type

        if vt == "bool" and isinstance(widget, QtWidgets.QCheckBox):
            return widget.isChecked()

        if vt == "int" and isinstance(widget, QtWidgets.QSpinBox):
            return int(widget.value())

        if vt == "float" and isinstance(widget, QtWidgets.QDoubleSpinBox):
            return float(widget.value())

        if vt == "str" and isinstance(widget, QtWidgets.QLineEdit):
            return widget.text().strip() or None

        if vt == "list_str" and isinstance(widget, QtWidgets.QLineEdit):
            text = widget.text().strip()
            if not text:
                return []
            return [tok.strip() for tok in text.split(",") if tok.strip()]

        if vt == "list_int" and isinstance(widget, QtWidgets.QLineEdit):
            text = widget.text().strip()
            if not text:
                return []
            result: list = []
            for tok in text.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    result.append(int(tok))
                except ValueError:
                    pass
            return result

        return None

    def _find_pair_partner(
        self, spec: ResourceFieldSpec, all_specs: List[ResourceFieldSpec]
    ) -> Optional[ResourceFieldSpec]:
        """Retorna o spec irmão se field_name termina com _source/_target e há um par."""
        for suffix in _PAIR_SUFFIXES:
            if spec.field_name.endswith(suffix):
                base = spec.field_name[: -len(suffix)]
                other_suffix = "_target" if suffix == "_source" else "_source"
                target_name = base + other_suffix
                for other in all_specs:
                    if other.field_name == target_name and other.accepted_types == spec.accepted_types:
                        # Retorna apenas se este spec for o de Source (para ordenação consistente)
                        return other if suffix == "_source" else None
        return None

    def _render_paired_inputs(
        self, source_spec: ResourceFieldSpec, target_spec: ResourceFieldSpec
    ) -> None:
        """Renderiza Source/Target lado a lado em duas colunas."""
        if not self._instruction:
            return

        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        for spec in (source_spec, target_spec):
            col = QtWidgets.QWidget()
            cv = QtWidgets.QVBoxLayout(col)
            cv.setContentsMargins(0, 0, 0, 0)
            cv.setSpacing(2)

            lbl = QtWidgets.QLabel(spec.label)
            lbl.setStyleSheet("font-size: 10px; color: #555;")
            cv.addWidget(lbl)

            combo = self._make_input_resource_combo(self._instruction, spec)
            row_widget = self._wrap_combo_with_edit_button_if_needed(combo, spec)
            cv.addWidget(row_widget)

            self._resource_field_widgets[spec.field_name] = combo
            h.addWidget(col, 1)

        # Adiciona como um row sem label (ocupa toda a largura)
        self._form_cfg.addRow(container)

    def _make_input_resource_combo(
        self, instr: StrategyInstructionDefinition, spec: ResourceFieldSpec
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.addItem("(nenhum)", userData=None)
        for res in self._available_resources:
            if res.resource_type in spec.accepted_types:
                display = res.name if res.name else res.resource_id
                combo.addItem(display, userData=res.resource_id)
                combo.setItemData(
                    combo.count() - 1,
                    f"ID: {res.resource_id}\nTipo: {res.resource_type.value}",
                    QtCore.Qt.ItemDataRole.ToolTipRole,
                )

        current_val = getattr(instr, spec.field_name, None)
        if current_val:
            for idx in range(combo.count()):
                if combo.itemData(idx) == current_val:
                    combo.setCurrentIndex(idx)
                    break
        return combo

    # Tipos de recurso que ganham um botão ✎ ao lado do combo, abrindo um
    # editor dedicado. O controller resolve qual dialog abrir com base no
    # resource_type do recurso selecionado.
    _EDITABLE_RESOURCE_TYPES = (
        StrategyResourceType.PIPELINE_CONFIG,
        StrategyResourceType.MINING_CONFIG,
    )

    _EDIT_TOOLTIPS = {
        StrategyResourceType.PIPELINE_CONFIG: "Editar a configuração de pipeline selecionada",
        StrategyResourceType.MINING_CONFIG: "Editar a configuração de mineração selecionada",
    }

    def _wrap_combo_with_edit_button_if_needed(
        self, combo: QtWidgets.QComboBox, spec: ResourceFieldSpec
    ) -> QtWidgets.QWidget:
        """
        Quando o spec aceita um tipo editável (PIPELINE_CONFIG ou
        MINING_CONFIG), envolve o combo num container horizontal com um
        botão ✎ que solicita ao controller a edição do recurso selecionado.
        """
        editable = next(
            (t for t in self._EDITABLE_RESOURCE_TYPES if t in spec.accepted_types),
            None,
        )
        if editable is None:
            return combo

        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(combo, 1)

        edit_btn = QtWidgets.QToolButton()
        edit_btn.setText("✎")
        edit_btn.setToolTip(self._EDIT_TOOLTIPS.get(editable, "Editar configuração"))
        edit_btn.setFixedWidth(28)
        edit_btn.clicked.connect(
            lambda _checked=False, c=combo: self._on_edit_resource(c)
        )
        edit_btn.setEnabled(combo.currentData() is not None)
        combo.currentIndexChanged.connect(
            lambda _idx, b=edit_btn, c=combo: b.setEnabled(c.currentData() is not None)
        )

        h.addWidget(edit_btn, 0)
        return container

    def _on_edit_resource(self, combo: QtWidgets.QComboBox) -> None:
        """
        Solicita ao controller que abra o dialog de edição apropriado para
        o recurso selecionado no combo. O controller resolve por
        resource_type (PIPELINE_CONFIG → PipelineCreatorController,
        MINING_CONFIG → NewMiningConfigDialogController).
        """
        resource_id = combo.currentData()
        if not resource_id:
            return
        self.resource_edit_requested.emit(str(resource_id))

    def _build_outputs_section(self, instr: StrategyInstructionDefinition) -> None:
        specs = instr.get_resource_field_specs()
        output_specs = [s for s in specs if s.is_output]

        if not output_specs:
            empty = QtWidgets.QLabel("(esta instrução não declara saídas)")
            empty.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
            self._outputs_layout.addWidget(empty)
            return

        for spec in output_specs:
            current_id = getattr(instr, spec.field_name, None)
            resource_type = (
                spec.accepted_types[0] if spec.accepted_types else None
            )
            if resource_type is None:
                continue

            row = OutputRowWidget(
                field_name=spec.field_name,
                label=spec.label,
                resource_id=current_id,
                resource_type=resource_type,
            )
            # Sincronia imediata: cada edição do nome do output grava no DTO
            # logo no instante em que o usuário digita, sem depender de o painel
            # invocar _on_apply antes da validação/execução.
            row.name_changed.connect(self._on_output_name_changed)
            self._output_widgets[spec.field_name] = row
            self._outputs_layout.addWidget(row)

    # ── Apply / Validate ──────────────────────────────────────────────────────

    def _on_apply(self) -> None:
        if not self._instruction:
            return

        instr = self._instruction

        # Identificação
        instr.name = self._field_name.text().strip() or instr.name
        instr.description = self._field_desc.toPlainText().strip() or None

        # Resource fields ANTES dos choice fields (specs dependem do choice atual)
        specs = instr.get_resource_field_specs()
        for spec in specs:
            if spec.is_output:
                widget = self._output_widgets.get(spec.field_name)
                val = widget.current_value() if widget else None
                setattr(instr, spec.field_name, val or None)
            else:
                widget = self._resource_field_widgets.get(spec.field_name)
                if widget is None:
                    continue
                if isinstance(widget, QtWidgets.QComboBox):
                    setattr(instr, spec.field_name, widget.currentData())

        # DTOs aninhados — já estão sincronizados via signals, mas garante atribuição
        for attr_name, widget in self._nested_inline_widgets.items():
            setattr(instr, attr_name, widget.value())
        for attr_name, widget in self._nested_dialog_widgets.items():
            setattr(instr, attr_name, widget.value())

        # Value fields — também ANTES dos choice fields, pois algumas specs
        # podem depender do choice atual (ex: split_ratio só existe em modos *_RATIO).
        for field_name, widget in self._value_field_widgets.items():
            spec = self._value_specs.get(field_name)
            if spec is None:
                continue
            value = self._read_value_widget(widget, spec)
            setattr(instr, field_name, value)

        if hasattr(instr, "_sync_declared_io"):
            instr._sync_declared_io()

        # Choice fields — DEPOIS dos resource fields
        for field_name, combo in self._choice_field_widgets.items():
            new_val = combo.currentText()
            new_val = self._convert_choice_value(field_name, new_val)
            setattr(instr, field_name, new_val)

        # Parâmetros genéricos
        for key, widget in self._param_fields.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                instr.parameters[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                instr.parameters[key] = widget.value()
            elif isinstance(widget, QtWidgets.QLineEdit):
                instr.parameters[key] = widget.text().strip()

        # Sincroniza io novamente após choice fields (modo pode ter mudado)
        if hasattr(instr, "_sync_declared_io"):
            instr._sync_declared_io()

        # No-op detection: se nada mudou desde o último load/apply, não
        # invalidamos outputs nem marcamos a estratégia como modificada.
        new_snapshot = self._snapshot_instruction(instr)
        baseline = getattr(self, "_instruction_snapshot", None)
        if baseline is not None and new_snapshot == baseline:
            return

        # Atualiza snapshot para refletir o estado pós-apply (a próxima
        # chamada com os mesmos valores será detectada como no-op).
        self._instruction_snapshot = new_snapshot

        self.config_changed.emit(instr)
        # Sinal específico do Aplicar — usado para invalidar outputs
        # persistidos da instrução cuja configuração mudou.
        self.instruction_applied.emit(instr)

    def _on_validate(self) -> None:
        if self._instruction:
            self._on_apply()
            self.validate_requested.emit(self._instruction)

    def _on_view_trajectory(self) -> None:
        if isinstance(self._instruction, PipelineApplicationInstruction):
            self.view_trajectory_requested.emit(self._instruction)

    def _on_view_distribution(self) -> None:
        if isinstance(
            self._instruction,
            (SimilarityScoringInstruction, ScoreCalibrationInstruction),
        ):
            self.view_score_distribution_requested.emit(self._instruction)

    def _on_view_validation_result(self) -> None:
        if isinstance(self._instruction, StrategyEvaluationInstruction):
            self.view_validation_result_requested.emit(self._instruction)

    def _on_view_casework_visualization(self) -> None:
        if isinstance(self._instruction, CaseworkVisualizationInstruction):
            self.view_casework_visualization_requested.emit(self._instruction)

    def _refresh_view_casework_visualization_button_state(self) -> None:
        """
        Habilita o botão "Visualizar Caso" quando o recurso de output da
        instrução existe (em memória ou em disco). Mesmo padrão de
        _refresh_view_validation_result_button_state.
        """
        if not isinstance(self._instruction, CaseworkVisualizationInstruction):
            self.btn_view_casework_visualization.setEnabled(False)
            return
        instr = self._instruction
        rid = instr.output_visualization_resource_id
        if not rid:
            self.btn_view_casework_visualization.setEnabled(False)
            return
        # Verifica se o recurso está vivo no contexto OU tem artifact_ref no disco
        enabled = False
        ctx = self._latest_context
        if ctx is not None:
            try:
                if ctx.has_resource(rid):
                    res = ctx.get_resource(rid)
                    if res is not None and (
                        getattr(res, "has_payload", lambda: res.payload is not None)()
                        if callable(getattr(res, "has_payload", None))
                        else (res.payload is not None or res.artifact_ref is not None)
                    ):
                        enabled = True
            except Exception:
                pass
        if not enabled and self._latest_definition is not None:
            for r in self._latest_definition.resources:
                if r.resource_id == rid and r.artifact_ref is not None:
                    enabled = True
                    break
        self.btn_view_casework_visualization.setEnabled(enabled)

    def _on_output_name_changed(self, field_name: str, new_value: str) -> None:
        """
        Atualiza o DTO assim que o usuário edita um nome de recurso de saída,
        sem precisar clicar em Aplicar. Garante que validações/execuções
        subsequentes vejam o valor digitado mesmo que o foco ainda esteja no
        campo. Não emite config_changed para evitar reset do contexto a cada
        keystroke — quem decide salvar é o Aplicar/Validar.
        """
        if not self._instruction:
            return
        setattr(self._instruction, field_name, new_value or None)
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()

    def _on_choice_changed(self, field_name: str) -> None:
        """Aplica o valor do campo de escolha e agenda reconstrução do formulário."""
        if not self._instruction:
            return
        combo = self._choice_field_widgets.get(field_name)
        if combo is None:
            return
        new_val = self._convert_choice_value(field_name, combo.currentText())
        setattr(self._instruction, field_name, new_val)
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()
        # Adia o rebuild para depois do processamento do sinal atual,
        # evitando crash por destruição do widget que originou o sinal.
        QtCore.QTimer.singleShot(0, self._rebuild_and_notify)

    def _on_value_triggers_rebuild(self, field_name: str) -> None:
        """
        Aplica o valor atual do widget de ValueFieldSpec ao DTO e dispara
        rebuild do formulário. Usado para campos com triggers_rebuild=True
        (ex: bool que controla outputs condicionais).
        """
        if not self._instruction:
            return
        widget = self._value_field_widgets.get(field_name)
        spec = self._value_specs.get(field_name)
        if widget is None or spec is None:
            return
        new_val = self._read_value_widget(widget, spec)
        setattr(self._instruction, field_name, new_val)
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()
        QtCore.QTimer.singleShot(0, self._rebuild_and_notify)

    def _on_nested_changed(self, attr_name: str, new_dto: Any) -> None:
        if not self._instruction:
            return
        setattr(self._instruction, attr_name, new_dto)
        self.config_changed.emit(self._instruction)

    def _rebuild_and_notify(self) -> None:
        if not self._instruction:
            return
        self._rebuild_form()
        self.config_changed.emit(self._instruction)

    def _convert_choice_value(self, field_name: str, text: str) -> Any:
        cspec = self._choice_specs.get(field_name)
        if cspec is not None and getattr(cspec, "enum_type", None) is not None:
            enum_cls = cspec.enum_type
            # Preferir from_str se existir (semântica custom do projeto).
            if hasattr(enum_cls, "from_str"):
                try:
                    return enum_cls.from_str(text)
                except Exception:
                    pass
            # Fallback genérico: tenta construtor (vale para Enum por value),
            # depois lookup por nome (`enum_cls[name]`).
            try:
                return enum_cls(text)
            except Exception:
                pass
            try:
                return enum_cls[text]
            except Exception:
                pass
        return text
