# interface/tabs/strategy_manager_v2/strategy_editor/instruction_config_panel.py

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
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
from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
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
from .section_header import SectionHeader, ColumnHeader


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


from .widget_helpers import (
    MIN_FIELD_WIDTH_PX as _MIN_FIELD_WIDTH_PX,
    make_combo_shrinkable as _make_combo_shrinkable,
    make_lineedit_shrinkable as _make_lineedit_shrinkable,
)


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
    # Emitido quando o usuário edita o nome de um recurso de saída no
    # campo "Saídas". Permite que outros painéis (ex: results panel na 3ª
    # coluna) sincronizem o nome exibido em tempo real.
    output_name_changed = QtCore.pyqtSignal(str, str)  # (field_name, new_value)

    # Emitido quando o usuário clica em 📤 numa linha de regra. Carrega o
    # config bruto da regra clicada; o controller é responsável por abrir
    # o `SaveValidationRuleTemplateDialog` e persistir.
    save_validation_rule_as_template_requested = QtCore.pyqtSignal(object)
    # Emitido quando o usuário clica em ➕ Adicionar Regra no header da
    # seção de regras. O controller responde abrindo
    # `AddValidationRuleDialog` com a lista de templates já listada do
    # disco — quando o usuário aceitar, o controller anexa o novo
    # ValidationRuleConfig em instruction.validation_rule_configs.
    add_validation_rule_requested = QtCore.pyqtSignal(object)

    # Emitido quando um campo com `triggers_rebuild=True` muda — ex:
    # `apply_to_training_datasets` em PIPELINE_TRAINING, que adiciona
    # ou remove campos de output. O editor escuta esse sinal e invalida
    # o card cacheado no `InstructionResultsPanel` (3ª coluna) pra que
    # ele seja recriado com os novos specs. Sem isso, a 3ª coluna fica
    # com a lista de outputs do estado anterior — usuário só vê a
    # mudança após save/reload do projeto.
    instruction_specs_rebuilt = QtCore.pyqtSignal(object)

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
        # Definition usada pra filtrar os dropdowns de input por ordem de
        # execução das instruções (recursos produzidos por instruções
        # posteriores não aparecem). Setada via `set_strategy_definition`
        # pelo editor a cada `load_strategy` / `update_available_resources`.
        # Independente de `_latest_definition` (visual runtime state).
        self._strategy_definition: Optional[LRStrategyDefinition] = None

        # Raiz do projeto atual — usada para resolver `artifact_ref.path`
        # (relativo) em path absoluto para leitura do sidecar de stats
        # exibido sob dropdowns de DATASET. Propagado pelo controller via
        # `set_project_root` após abrir o projeto.
        self._project_root: Optional[Path] = None

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
        # Margens 4px iguais às colunas 1 (InstructionListWidget) e 3
        # (InstructionResultsPanel) — sem isso o ColumnHeader desta coluna
        # ficava ~4px mais embaixo que os outros, dando impressão de
        # desalinhamento no topo.
        self._root_layout.setContentsMargins(4, 4, 4, 4)
        self._root_layout.setSpacing(4)

        # Cabeçalho principal da coluna — padrão unificado com Instruções
        # e Saídas. Fica FORA do stack pra que apareça mesmo quando nenhuma
        # instrução está selecionada (página vazia). Tipo + id da instrução
        # aparecem como campos dentro da seção Identificação.
        self._root_layout.addWidget(ColumnHeader("Configurações"))

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

        # ─── Seção 3: Configurações de saída ──────────────────────────────
        # A seção fica SEMPRE no painel de configuração (2ª coluna): o nome
        # do recurso de saída é decisão de configuração. O painel de
        # resultados (3ª coluna) renderiza a mesma OutputRowWidget em modo
        # read-only com foco em estado de runtime + summary + plot.
        scroll_layout.addWidget(SectionHeader("Configurações de saída", "outputs"))
        self._outputs_layout = QtWidgets.QVBoxLayout()
        self._outputs_layout.setContentsMargins(8, 6, 8, 8)
        self._outputs_layout.setSpacing(6)
        wrapper_out = QtWidgets.QWidget()
        wrapper_out.setLayout(self._outputs_layout)
        scroll_layout.addWidget(wrapper_out)

        # ─── Seção 4: Regras de Validação ─────────────────────────────────
        # Lista as `validation_rule_configs` da instrução. Pluggable via
        # __post_init__ + _sync_declared_io das instruções que conhecem
        # seus defaults. O usuário pode remover regras individualmente
        # (🗑) ou restaurar os defaults via botão na header da seção.
        # Não há UI de adição nesta fase — virá na Fase 3.
        self._validation_section_header = SectionHeader(
            "Regras de Validação", "validation"
        )
        # Botão ➕ — abre AddValidationRuleDialog (Phase 3, template-only).
        self._btn_add_validation_rule = QtWidgets.QPushButton(
            "➕ Adicionar Regra"
        )
        self._btn_add_validation_rule.setToolTip(
            "Adiciona uma regra de validação à instrução. Tipos "
            "configuráveis ou templates salvos na biblioteca."
        )
        self._btn_add_validation_rule.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        _validation_btn_style = (
            "QPushButton { background: transparent; border: 1px solid #E8A020;"
            " border-radius: 3px; color: #8B5E00; font-size: 10px;"
            " padding: 2px 8px; }"
            "QPushButton:hover { background: #FFEFC4; }"
        )
        self._btn_add_validation_rule.setStyleSheet(_validation_btn_style)
        self._btn_add_validation_rule.clicked.connect(
            self._on_add_validation_rule_clicked
        )
        self._validation_section_header.addActionWidget(
            self._btn_add_validation_rule
        )

        self._btn_restore_validation_defaults = QtWidgets.QPushButton(
            "🔄 Restaurar padrões"
        )
        self._btn_restore_validation_defaults.setToolTip(
            "Limpa a lista atual e repopula com as regras padrão "
            "recomendadas para o tipo + estado da instrução."
        )
        self._btn_restore_validation_defaults.setCursor(
            QtCore.Qt.CursorShape.PointingHandCursor
        )
        self._btn_restore_validation_defaults.setStyleSheet(_validation_btn_style)
        self._btn_restore_validation_defaults.clicked.connect(
            self._on_restore_default_validation_rules
        )
        self._validation_section_header.addActionWidget(
            self._btn_restore_validation_defaults
        )
        scroll_layout.addWidget(self._validation_section_header)

        self._validation_layout = QtWidgets.QVBoxLayout()
        self._validation_layout.setContentsMargins(8, 6, 8, 8)
        self._validation_layout.setSpacing(4)
        wrapper_val = QtWidgets.QWidget()
        wrapper_val.setLayout(self._validation_layout)
        scroll_layout.addWidget(wrapper_val)

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

    def set_strategy_definition(
        self, definition: Optional[LRStrategyDefinition]
    ) -> None:
        """Define a estratégia aberta no editor. Usada pelos dropdowns de
        input pra filtrar derivados por ordem de execução (instruções
        posteriores não devem aparecer como inputs de uma anterior).
        Sem chamar isso, o filtro é desativado e mostra todos os
        derivados — degradação segura.
        """
        self._strategy_definition = definition
        if self._instruction is not None:
            self._rebuild_form()

    def set_project_root(self, project_root: Optional[Path]) -> None:
        """
        Define a raiz do projeto atual. Usado para resolver paths relativos
        de artefatos (`StrategyArtifactReference.path`) ao ler o sidecar
        `<stem>_stats.json` que alimenta o info label de DATASETs.
        """
        self._project_root = Path(project_root) if project_root else None

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
        # ─── Seção Identificação ────────────────────────────────────────
        # Tipo + ID exibidos como campos read-only dentro da Identificação,
        # em vez de num header competindo com o título da coluna.
        type_lbl = QtWidgets.QLabel(instr.instruction_type.label_pt)
        type_lbl.setStyleSheet(
            "color: #1A2942; font-size: 11px; font-weight: bold;"
            " padding: 2px 0px;"
        )
        type_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._form_id.addRow("Tipo:", type_lbl)

        id_lbl = QtWidgets.QLabel(str(instr.instruction_id))
        id_lbl.setStyleSheet(
            "color: #555E6D; font-size: 10px; font-family: monospace;"
            " padding: 2px 0px;"
        )
        id_lbl.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._form_id.addRow("ID:", id_lbl)

        self._field_name = QtWidgets.QLineEdit(instr.name or "")
        _make_lineedit_shrinkable(self._field_name)
        self._form_id.addRow("Nome:", self._field_name)

        self._field_desc = QtWidgets.QPlainTextEdit(instr.description or "")
        self._field_desc.setMaximumHeight(54)
        self._field_desc.setMinimumWidth(_MIN_FIELD_WIDTH_PX)
        self._form_id.addRow("Descrição:", self._field_desc)

        # ─── Seção Configuração ─────────────────────────────────────────
        self._build_configuration_section(instr)

        # ─── Seção Saídas ────────────────────────────────────────────────
        self._build_outputs_section(instr)

        # ─── Seção Regras de Validação ───────────────────────────────────
        self._rebuild_validation_section()

        # Visibilidade dos botões de visualização — só faz sentido aplicar
        # quando os botões foram realmente adicionados ao layout deste painel.
        # Quando outputs_in_results_panel=True, os botões são instanciados
        # mas NÃO são parentados a nenhum layout (ficam invisíveis no init);
        # chamar setVisible(True) num widget sem parent faz ele virar uma
        # top-level window flutuante (a "janelinha pequena" no macOS).
        if not self._outputs_in_results_panel:
            self.btn_view_trajectory.setVisible(
                isinstance(instr, PipelineApplicationInstruction)
            )
            self.btn_view_distribution.setVisible(
                isinstance(instr, (SimilarityScoringInstruction, ScoreCalibrationInstruction))
            )
            self.btn_view_validation_result.setVisible(
                isinstance(instr, StrategyEvaluationInstruction)
            )
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
            _make_combo_shrinkable(combo)

            # `dynamic_choices` permite que o DTO filtre as opções a cada
            # rebuild, com base no estado atual de outras seleções e nos
            # recursos disponíveis (ex: DATASET_BALANCING filtra
            # quality_metric pelas métricas presentes no dataset).
            effective_choices = cspec.choices
            if cspec.dynamic_choices is not None:
                try:
                    filtered = cspec.dynamic_choices(self._available_resources)
                except Exception:
                    filtered = None
                if filtered:
                    effective_choices = filtered

            combo.addItems(effective_choices)

            raw_val = getattr(instr, cspec.field_name, None)
            if raw_val is not None:
                if hasattr(raw_val, "name"):
                    current_str = raw_val.name
                else:
                    current_str = str(raw_val)
                    current_str = current_str.split(".")[-1] if "." in current_str else current_str

                choices_upper = {c.upper(): c for c in effective_choices}
                match = choices_upper.get(current_str.upper())
                if match:
                    combo.setCurrentText(match)

            self._choice_field_widgets[cspec.field_name] = combo
            self._choice_specs[cspec.field_name] = cspec
            tooltip = self._get_field_tooltip(instr, cspec.field_name)
            if tooltip:
                combo.setToolTip(tooltip)
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
                row_widget = self._wrap_with_dataset_info_label_if_needed(
                    combo, spec, row_widget
                )
                self._form_cfg.addRow(f"{spec.label}:", row_widget)
                rendered_field_names.add(spec.field_name)

                # Wire triggers_rebuild também no caminho não-pareado.
                # Antes só o `_render_paired_inputs` conectava esse signal,
                # então specs single como `DATASET_BALANCING.dataset_resource_id`
                # e `TYPICALITY_PAIR_SCORING.reference_dataset_resource_id`
                # tinham `triggers_rebuild=True` mas nunca disparavam rebuild
                # — o usuário precisava clicar em "Aplicar alterações" pra
                # ver dropdowns dependentes aparecerem.
                if getattr(spec, "triggers_rebuild", False):
                    combo.currentIndexChanged.connect(
                        lambda _idx, fn=spec.field_name:
                            self._on_resource_triggers_rebuild(fn)
                    )

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
            if attr_name == "bigauss_config" and not getattr(
                instr, "uses_bigauss", False
            ):
                continue
            if attr_name == "typicality_model_config" and not (
                getattr(instr, "uses_knn", False)
                or getattr(instr, "uses_movmf", False)
            ):
                continue

            current_dto = getattr(instr, attr_name, None)

            # Reconcilia config polimórfico antes da renderização inline.
            # Quando o usuário troca KNN ↔ MoVMF no dropdown, _sync_declared_io
            # já chama _reconcile_config_type, mas defensivamente garantimos
            # aqui que o DTO carregado bate com o backend escolhido — assim
            # JSONs híbridos antigos abertos pela 1ª vez também se acertam.
            if attr_name == "typicality_model_config" and hasattr(
                instr, "_reconcile_config_type"
            ):
                instr._reconcile_config_type()
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
            tooltip = self._get_field_tooltip(instr, vspec.field_name)
            if tooltip:
                widget.setToolTip(tooltip)
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
                _make_lineedit_shrinkable(widget)
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
            _make_lineedit_shrinkable(w)
            return w

        if vt in ("list_str", "list_int"):
            text = ""
            if isinstance(current, (list, tuple)):
                text = ", ".join(str(x) for x in current)
            w = QtWidgets.QLineEdit(text)
            w.setPlaceholderText(spec.placeholder or "valores separados por vírgula")
            _make_lineedit_shrinkable(w)
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
            row_widget = self._wrap_with_dataset_info_label_if_needed(
                combo, spec, row_widget
            )
            cv.addWidget(row_widget)

            self._resource_field_widgets[spec.field_name] = combo

            # Quando o spec marcou triggers_rebuild=True, mudanças no
            # dropdown disparam um rebuild completo do form. Útil pra
            # specs cujos `dynamic_choices` dependem do recurso escolhido
            # (ex: DATASET_BALANCING.quality_metric filtra pelo dataset).
            if getattr(spec, "triggers_rebuild", False):
                combo.currentIndexChanged.connect(
                    lambda _idx, fn=spec.field_name:
                        self._on_resource_triggers_rebuild(fn)
                )

            h.addWidget(col, 1)

        # Adiciona como um row sem label (ocupa toda a largura)
        self._form_cfg.addRow(container)

    def _make_input_resource_combo(
        self, instr: StrategyInstructionDefinition, spec: ResourceFieldSpec
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        _make_combo_shrinkable(combo)
        combo.addItem("(nenhum)", userData=None)

        # IDs dos outputs declarados pela própria instrução — não podem
        # aparecer como opção de input (auto-feedback loop). Mesmo se a
        # instrução já tiver sido executada antes e o output exista em
        # disco como artefato, semanticamente um PIPELINE_APPLICATION
        # não pode consumir o próprio dataset transformado como entrada.
        own_output_ids: set = set()
        try:
            for own_spec in instr.get_resource_field_specs():
                if own_spec.is_output:
                    val = getattr(instr, own_spec.field_name, None)
                    if val:
                        own_output_ids.add(val)
        except Exception:
            pass

        # Outputs declarados por instruções ANTERIORES a `instr` na ordem
        # de execução (preparation → validation → casework; dentro da
        # fase, ordem da lista). Outputs de instruções POSTERIORES não
        # entram — não estão produzidos no momento em que `instr` roda.
        # Sem isso, derivados herdados por duplicate (estratégia clonada
        # de outra com 20+ instruções já configuradas) apareciam todos
        # no dropdown, mesmo aqueles produzidos por instruções que
        # rodam DEPOIS — confunde o usuário e permite seleções inválidas.
        upstream_output_ids: Optional[Set[str]] = None
        defn_for_filter = (
            self._strategy_definition or self._latest_definition
        )
        if defn_for_filter is not None and instr is not None:
            upstream_output_ids = set()
            try:
                done = False
                for phase_list in defn_for_filter._phase_lists():
                    if done:
                        break
                    for prior in phase_list:
                        if prior.instruction_id == instr.instruction_id:
                            done = True
                            break
                        try:
                            for ph in prior.get_output_resource_placeholders():
                                upstream_output_ids.add(ph.resource_id)
                        except Exception:
                            pass
            except Exception:
                upstream_output_ids = None

        for res in self._available_resources:
            if res.resource_type not in spec.accepted_types:
                continue
            if res.resource_id in own_output_ids:
                continue
            # Filtragem por ordem de execução: PRIMARY sempre passa;
            # DERIVED só passa se produzido por instrução anterior.
            if (
                upstream_output_ids is not None
                and res.category == ResourceCategory.DERIVED
                and res.resource_id not in upstream_output_ids
            ):
                continue
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
        StrategyResourceType.SPLIT_CONFIG,
    )

    _EDIT_TOOLTIPS = {
        StrategyResourceType.PIPELINE_CONFIG: "Editar a configuração de pipeline selecionada",
        StrategyResourceType.MINING_CONFIG: "Editar a configuração de mineração selecionada",
        StrategyResourceType.SPLIT_CONFIG: "Editar a configuração de split selecionada",
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

    # Tipos de recurso pra os quais mostramos um info label abaixo do combo
    # com (N° Amostras, N° Identidades, N° Dimensões do Embedding). Inclui
    # apenas DATASET — INTERMEDIATE_DATASET foi unificado no DATASET.
    _DATASET_INFO_TYPES = (
        StrategyResourceType.DATASET,
    )

    def _wrap_with_dataset_info_label_if_needed(
        self,
        combo: QtWidgets.QComboBox,
        spec: ResourceFieldSpec,
        outer_widget: QtWidgets.QWidget,
    ) -> QtWidgets.QWidget:
        """
        Se o spec aceita DATASET, envelopa `outer_widget` (combo ou container
        com botão ✎) num container vertical [combo / info_label], onde o
        info_label exibe "N° Amostras, N° Identidades, N° Dimensões" do
        dataset atualmente selecionado. Sem dataset selecionado o label fica
        escondido. Reage ao `currentIndexChanged` do combo.
        """
        if not any(t in spec.accepted_types for t in self._DATASET_INFO_TYPES):
            return outer_widget

        # Container subclass: dispara `_update_dataset_info_label` na PRIMEIRA
        # exibição (showEvent). Antes essa chamada era sincrona dentro de
        # `_wrap_...` ou via QTimer.singleShot(0) — ambos sofriam do mesmo
        # bug: setVisible(True) no info_lbl enquanto o container ainda não
        # estava parenteado fazia o Qt exibi-lo brevemente como janela
        # flutuante. showEvent só dispara quando o widget está realmente
        # sendo desenhado dentro da árvore correta — não há risco de top-level.
        panel = self

        class _LazyInitContainer(QtWidgets.QWidget):
            def __init__(self, lbl: QtWidgets.QLabel, target_combo: QtWidgets.QComboBox):
                # Parent provisório = panel (self do enclosing scope) — Qt
                # vai reparentá-lo via `addWidget` quando o caller inseri-lo
                # no layout final. Crítico: NÃO criar parentless mesmo
                # provisoriamente. Widgets criados sem parent viram top-level
                # window e podem flashear brevemente em algumas plataformas
                # (especialmente Windows) durante o ciclo de criação/reparent.
                super().__init__(parent=panel)
                # WA_DontShowOnScreen é um cinto-de-segurança adicional —
                # garante que mesmo se Qt tentasse exibir o widget durante
                # o reparent, nada apareceria como janela.
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
                self._lbl = lbl
                self._target_combo = target_combo
                self._initialized = False

            def showEvent(self, event):
                # Remove a flag agora que o widget está sendo exibido
                # dentro da árvore correta — caso ainda esteja setada.
                if self.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen):
                    self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)
                super().showEvent(event)
                if not self._initialized:
                    self._initialized = True
                    panel._update_dataset_info_label(self._target_combo, self._lbl)

        info_lbl = QtWidgets.QLabel("")
        info_lbl.setStyleSheet(
            "QLabel { color: #555; font-size: 11px; font-style: italic;"
            " padding-left: 4px; }"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setVisible(False)

        container = _LazyInitContainer(info_lbl, combo)
        v = QtWidgets.QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        v.addWidget(outer_widget)
        v.addWidget(info_lbl)

        combo.currentIndexChanged.connect(
            lambda _idx, c=combo, lbl=info_lbl:
                self._update_dataset_info_label(c, lbl)
        )
        return container

    def _update_dataset_info_label(
        self, combo: QtWidgets.QComboBox, info_lbl: QtWidgets.QLabel
    ) -> None:
        """
        Atualiza o info label sob um dropdown de DATASET com as estatísticas
        do recurso selecionado. Fonte preferida: sidecar `<stem>_stats.json`
        ao lado do parquet (gerado on-demand pelo `ResourceInfoWidget` ou
        pela mesma rotina embutida abaixo). Fallback: payload em memória.
        """
        resource_id = combo.currentData()
        if not resource_id:
            info_lbl.setVisible(False)
            info_lbl.setText("")
            return

        resource = next(
            (r for r in self._available_resources if r.resource_id == resource_id),
            None,
        )
        if resource is None:
            info_lbl.setVisible(False)
            return

        n_imgs: Optional[int] = None
        n_ids: Optional[int] = None
        emb_dim: Optional[int] = None

        # 1) Sidecar JSON (rápido — apenas metadata, single I/O). Se
        # ausente ou stale, computa via DatasetStatsService (pyarrow
        # metadata) — mesmo padrão usado pelo ResourceInfoWidget — pra
        # primeira seleção não precisar do usuário clicar antes no recurso
        # no left panel.
        artifact_ref = getattr(resource, "artifact_ref", None)
        if artifact_ref and self._project_root is not None:
            try:
                abs_path = self._project_root / artifact_ref.path
                stats_path = abs_path.parent / (abs_path.stem + "_stats.json")
                if not stats_path.exists() and abs_path.exists():
                    self._try_compute_dataset_stats_sidecar(abs_path, stats_path)
                if stats_path.exists():
                    data = json.loads(stats_path.read_text(encoding="utf-8"))
                    n_imgs = data.get("n_imgs")
                    n_ids = data.get("n_ids")
                    emb_cols = data.get("embedding_columns", []) or []
                    if emb_cols and isinstance(emb_cols, list):
                        first = emb_cols[0] if isinstance(emb_cols[0], dict) else None
                        if first:
                            emb_dim = first.get("dim")
            except Exception:
                pass

        # 2) Fallback: payload em memória (datasets ainda não persistidos).
        payload = getattr(resource, "payload", None)
        if payload is not None:
            if n_imgs is None:
                n_imgs = getattr(payload, "n_samples", None) or len(
                    getattr(payload, "items", None) or []
                ) or None
            if n_ids is None:
                ids_attr = getattr(payload, "identities", None)
                if ids_attr is not None:
                    try:
                        n_ids = len(ids_attr)
                    except Exception:
                        n_ids = None
            if emb_dim is None:
                items = getattr(payload, "items", None) or []
                if items:
                    try:
                        emb_dim = int(items[0].embedding.shape[0])
                    except Exception:
                        emb_dim = None

        if n_imgs is None and n_ids is None and emb_dim is None:
            info_lbl.setVisible(False)
            return

        n_imgs_txt = f"{n_imgs:,}".replace(",", ".") if n_imgs is not None else "—"
        n_ids_txt = f"{n_ids:,}".replace(",", ".") if n_ids is not None else "—"
        emb_dim_txt = f"{emb_dim}D" if emb_dim is not None else "—"
        info_lbl.setText(
            f"N° Amostras: <b>{n_imgs_txt}</b>  •  "
            f"N° Identidades: <b>{n_ids_txt}</b>  •  "
            f"Dim. Embedding: <b>{emb_dim_txt}</b>"
        )
        info_lbl.setVisible(True)

    @staticmethod
    def _try_compute_dataset_stats_sidecar(
        parquet_path: Path, stats_path: Path
    ) -> None:
        """
        Computa o sidecar `<stem>_stats.json` via `DatasetStatsService` (leitura
        de metadata pyarrow — não materializa o parquet inteiro). Silencioso
        em caso de falha — o info label simplesmente fica oculto.
        """
        try:
            from faceaudit.services.dataset_stats_service import DatasetStatsService

            stats = DatasetStatsService()._read_parquet_stats(parquet_path)
            embedding_columns = [
                {"index": pos, "name": name, "dim": cs.dim}
                for pos, (name, cs) in enumerate(stats.column_coverage.items())
                if cs.dim is not None
            ]
            payload = {
                "n_imgs": stats.n_imgs,
                "n_ids": stats.n_ids,
                "embedding_columns": embedding_columns,
                "column_coverage": {
                    name: cs.to_dict() for name, cs in stats.column_coverage.items()
                },
                "last_modified": stats.last_modified,
            }
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

    # =========================================================================
    # SEÇÃO: REGRAS DE VALIDAÇÃO
    # =========================================================================

    # Sumarizador de params por tipo de regra — devolve texto compacto
    # exibido sob o nome da regra. Cobertura completa dos 8 tipos da Fase 1.
    _RULE_PARAM_SUMMARIZERS = {
        ValidationRuleType.FISHER: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • margem: {p.get('safety_margin_ratio', 0.2)}"
        ),
        ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • mínimo: {p.get('min_required', 2)}"
            f" • alerta abaixo: {p.get('safety_threshold', 10)}"
        ),
        ValidationRuleType.MIN_CALIBRATION_PAIRS: lambda p: (
            f"scores: <b>{p.get('scores_resource_id', '—')}</b>"
            f" • min_gen: {p.get('min_genuines', 100)}"
            f" • min_imp: {p.get('min_impostors', 500)}"
        ),
        ValidationRuleType.DATASET_ISOLATION: lambda p: (
            f"A: <b>{p.get('dataset_a_resource_id', '—')}</b>"
            f" • B: <b>{p.get('dataset_b_resource_id', '—')}</b>"
        ),
        ValidationRuleType.DATASET_OVERLAP_REQUIRED: lambda p: (
            f"A: <b>{p.get('dataset_a_resource_id', '—')}</b>"
            f" • B: <b>{p.get('dataset_b_resource_id', '—')}</b>"
            f" • min: {p.get('min_overlap', 1)}"
        ),
        ValidationRuleType.HOLDOUT_LEAKAGE: lambda p: (
            f"holdout: {len(p.get('holdout_resource_ids') or [])} recurso(s)"
            f" • treino: {len(p.get('train_resource_ids') or [])} recurso(s)"
        ),
        ValidationRuleType.DUAL_PIPELINE_DIM_COMPAT: lambda p: (
            f"source: <b>{p.get('source_config_resource_id', '—')}</b>"
            f" • target: <b>{p.get('target_config_resource_id', '—')}</b>"
        ),
        ValidationRuleType.DUAL_PIPELINE_SUBSEQUENCE: lambda p: (
            f"source: <b>{p.get('source_config_resource_id', '—')}</b>"
            f" • target: <b>{p.get('target_config_resource_id', '—')}</b>"
        ),
        ValidationRuleType.MIN_IDENTITIES: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • mínimo: {p.get('min_required', 2)}"
            f" • alerta abaixo: {p.get('safety_threshold', 5)}"
        ),
        ValidationRuleType.MIN_SAMPLES: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • mínimo: {p.get('min_required', 2)}"
            f" • alerta abaixo: {p.get('safety_threshold', 100)}"
        ),
        ValidationRuleType.MODEL_FITTED: lambda p: (
            f"modelo: <b>{p.get('model_resource_id', '—')}</b>"
        ),
        ValidationRuleType.EMBEDDING_DIM_MATCHES_MODEL: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" × modelo: <b>{p.get('model_resource_id', '—')}</b>"
        ),
        ValidationRuleType.SCORE_CALIBRATION_STATE: lambda p: (
            f"scores: <b>{p.get('scores_resource_id', '—')}</b>"
            f" • esperado: {p.get('expected', 'RAW')}"
        ),
        ValidationRuleType.CASEWORK_STATE: lambda p: (
            f"scores: <b>{p.get('scores_resource_id', '—')}</b>"
            f" • is_casework esperado: {p.get('expected', False)}"
        ),
        ValidationRuleType.SCORE_TYPE_MATCH: lambda p: (
            f"referência: <b>{p.get('reference_resource_id', '—')}</b>"
            f" • caso: <b>{p.get('case_resource_id', '—')}</b>"
        ),
        ValidationRuleType.DATASETS_EMBEDDING_COMPAT: lambda p: (
            f"datasets: {len(p.get('dataset_resource_ids') or [])}"
            + (
                f" • coluna: <b>{p.get('embedding_column')}</b>"
                if p.get('embedding_column') else ""
            )
        ),
        ValidationRuleType.QUALITY_METRIC_AVAILABLE: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • métrica: <b>{p.get('quality_metric', '—')}</b>"
            f" • cobertura mín: {float(p.get('min_coverage', 0.5)):.0%}"
        ),
        ValidationRuleType.HAS_POSE_METADATA: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
        ),
        ValidationRuleType.MAX_SAMPLES_WARNING: lambda p: (
            f"dataset: <b>{p.get('dataset_resource_id', '—')}</b>"
            f" • limite: {int(p.get('max_samples', 100_000)):,}"
        ),
    }

    @classmethod
    def _summarize_rule_params(cls, rule_type: ValidationRuleType, params: dict) -> str:
        """
        Formata os parâmetros de uma regra como string compacta (HTML-friendly,
        usado no QLabel da linha). Fallback genérico para tipos não cobertos
        no mapa explícito.
        """
        summarizer = cls._RULE_PARAM_SUMMARIZERS.get(rule_type)
        if summarizer:
            try:
                return summarizer(params or {})
            except Exception:
                pass
        items = ", ".join(f"{k}={v}" for k, v in (params or {}).items())
        return items or "(sem parâmetros)"

    def _clear_validation_layout(self) -> None:
        while self._validation_layout.count():
            item = self._validation_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_validation_section(self) -> None:
        """
        Limpa e redesenha a lista de regras de validação. Renderiza dois
        grupos:
        1. **Configuráveis** (persistentes em `validation_rule_configs`):
           podem ser removidas (🗑) e restauradas (🔄 padrões).
        2. **Intrínsecas** (dinâmicas, retornadas por
           `instruction.build_validation_rules()`): exibidas em read-only
           com 🔒 — mostram ao usuário que essas regras vão rodar mas não
           são removíveis (sem sentido desabilitar checks de
           pré-condições matemáticas da própria instrução).

        Chamada pelo `_rebuild_form` e por handlers de exclusão /
        restauração / adição.
        """
        self._clear_validation_layout()

        instr = self._instruction
        configs = getattr(instr, "validation_rule_configs", None) or [] if instr else []

        # Regras dinâmicas são construídas a cada render — sempre refletem
        # o estado atual da instrução (ex: PIPELINE_TRAINING DUAL só
        # retorna rules quando source+target estão preenchidos).
        dynamic_rules = []
        if instr is not None:
            try:
                dynamic_rules = list(instr.build_validation_rules() or [])
            except Exception:
                dynamic_rules = []

        # Estado vazio só quando NÃO há nem configuráveis nem intrínsecas.
        if not configs and not dynamic_rules:
            empty = QtWidgets.QLabel(
                "(nenhuma regra ativa) — defaults são populados conforme "
                "você preenche os recursos da instrução. Clique em "
                "<b>Restaurar padrões</b> se já preencheu mas não viu "
                "regras aparecerem."
            )
            empty.setStyleSheet(
                "color: #888; font-size: 11px; font-style: italic;"
            )
            empty.setWordWrap(True)
            self._validation_layout.addWidget(empty)
            return

        for idx, cfg in enumerate(configs):
            self._validation_layout.addWidget(self._make_rule_row(idx, cfg))

        for rule in dynamic_rules:
            self._validation_layout.addWidget(self._make_rule_row_locked(rule))

    def _make_rule_row(self, index: int, cfg) -> QtWidgets.QWidget:
        """
        Constrói uma linha visual para um `ValidationRuleConfig`:
            [icon] [Rule label]   [params summary]            [🗑]
        """
        row = QtWidgets.QFrame()
        row.setStyleSheet(
            "QFrame { background: #FFFCF2; border: 1px solid #F3E0B0;"
            " border-radius: 3px; }"
        )
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(8, 4, 6, 4)
        h.setSpacing(8)

        icon = QtWidgets.QLabel("🛡️")
        icon.setStyleSheet("font-size: 13px;")
        h.addWidget(icon, 0)

        # Nome + sumário ficam empilhados num layout vertical compacto.
        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        rule_type = cfg.rule_type
        name_lbl = QtWidgets.QLabel(rule_type.label)
        name_lbl.setStyleSheet(
            "color: #5D4500; font-size: 11px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        v.addWidget(name_lbl)

        summary = self._summarize_rule_params(rule_type, cfg.params)
        summary_lbl = QtWidgets.QLabel(summary)
        summary_lbl.setStyleSheet(
            "color: #6E5A2E; font-size: 10px;"
            " background: transparent; border: none;"
        )
        summary_lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        summary_lbl.setWordWrap(True)
        v.addWidget(summary_lbl)

        # Tooltip rico (descrição da regra) ajuda o usuário a entender o
        # que cada regra checa sem precisar abrir documentação.
        tooltip = rule_type.description
        if tooltip:
            row.setToolTip(tooltip)

        col_wrapper = QtWidgets.QWidget()
        col_wrapper.setLayout(v)
        h.addWidget(col_wrapper, 1)

        # 📤 — salvar esta regra como template (Global library e/ou projeto).
        btn_save = QtWidgets.QToolButton()
        btn_save.setText("📤")
        btn_save.setToolTip(
            "Salvar esta regra como template (biblioteca global ou "
            "no projeto). Templates aparecem no diálogo 'Adicionar Regra'."
        )
        btn_save.setFixedWidth(28)
        btn_save.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_save.setStyleSheet(
            "QToolButton { background: transparent; border: 1px solid #6E5A2E;"
            " border-radius: 3px; color: #6E5A2E; }"
            "QToolButton:hover { background: #F5EBD0; }"
        )
        btn_save.clicked.connect(
            lambda _checked=False, c=cfg: self.save_validation_rule_as_template_requested.emit(c)
        )
        h.addWidget(btn_save, 0)

        btn_del = QtWidgets.QToolButton()
        btn_del.setText("🗑")
        btn_del.setToolTip(
            "Remover esta regra. Pode ser restaurada via "
            "'Restaurar padrões' (se for uma regra padrão)."
        )
        btn_del.setFixedWidth(28)
        btn_del.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn_del.setStyleSheet(
            "QToolButton { background: transparent; border: 1px solid #E8A020;"
            " border-radius: 3px; color: #8B5E00; }"
            "QToolButton:hover { background: #FFEFC4; }"
        )
        btn_del.clicked.connect(
            lambda _checked=False, i=index: self._on_delete_validation_rule(i)
        )
        h.addWidget(btn_del, 0)

        return row

    def _make_rule_row_locked(self, rule) -> QtWidgets.QWidget:
        """
        Constrói uma linha read-only para uma regra de validação intrínseca
        (retornada por `instruction.build_validation_rules()`). Visualmente
        distinta da linha editável: fundo cinza-azulado, 🔒 em vez de 🗑,
        tooltip explicando o porquê.

        O sumário usa `rule.description` (já populada com nomes dos
        recursos pelo construtor da regra concreta) — evita re-mapear de
        atributos privados pra dict de params.
        """
        row = QtWidgets.QFrame()
        row.setStyleSheet(
            "QFrame { background: #F5F5F8; border: 1px solid #D8D8E0;"
            " border-radius: 3px; }"
        )
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(8, 4, 6, 4)
        h.setSpacing(8)

        icon = QtWidgets.QLabel("🛡️")
        icon.setStyleSheet("font-size: 13px;")
        h.addWidget(icon, 0)

        v = QtWidgets.QVBoxLayout()
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        name_lbl = QtWidgets.QLabel(getattr(rule, "name", "Regra intrínseca"))
        name_lbl.setStyleSheet(
            "color: #34344A; font-size: 11px; font-weight: bold;"
            " background: transparent; border: none;"
        )
        v.addWidget(name_lbl)

        desc = getattr(rule, "description", None) or ""
        if desc:
            desc_lbl = QtWidgets.QLabel(desc)
            desc_lbl.setStyleSheet(
                "color: #555566; font-size: 10px;"
                " background: transparent; border: none;"
            )
            desc_lbl.setWordWrap(True)
            v.addWidget(desc_lbl)

        col_wrapper = QtWidgets.QWidget()
        col_wrapper.setLayout(v)
        h.addWidget(col_wrapper, 1)

        row.setToolTip(
            "Regra intrínseca — não pode ser removida. É uma pré-condição "
            "matemática da instrução (ex: compatibilidade dimensional, "
            "modelo treinado, scores no estado certo). Falha aqui = "
            "impossibilidade de execução, não opinião de configuração."
        )

        # Indicador visual de "trancado": label simples ao invés de botão
        # — reforça que não é clicável. Mesma largura do 🗑 pra alinhar
        # verticalmente com as linhas editáveis acima.
        lock = QtWidgets.QLabel("🔒")
        lock.setFixedWidth(28)
        lock.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lock.setStyleSheet(
            "QLabel { color: #888; background: transparent; border: none;"
            " font-size: 12px; }"
        )
        h.addWidget(lock, 0)

        return row

    def _on_delete_validation_rule(self, index: int) -> None:
        """
        Remove a regra de validação no índice indicado e rebuilda a seção.
        Emite `config_changed` para que o controller persista a alteração.

        Marca `validation_rules_user_managed=True` para sinalizar que o
        usuário assumiu o controle — impede que `_ensure_default_validation_
        rules` re-popule defaults no próximo reload do projeto, preservando
        a intenção do usuário entre sessões.
        """
        instr = self._instruction
        if instr is None:
            return

        configs = getattr(instr, "validation_rule_configs", None)
        if not configs or index < 0 or index >= len(configs):
            return

        del configs[index]
        instr.validation_rules_user_managed = True
        self._rebuild_validation_section()
        self.config_changed.emit(instr)

    def _on_restore_default_validation_rules(self) -> None:
        """
        Limpa a lista atual e dispara `_ensure_default_validation_rules`
        (via `_sync_declared_io`) para repopular com os defaults do tipo
        + estado atual da instrução. Útil para reverter exclusões acidentais
        ou para aplicar regras quando o usuário acabou de configurar um
        dataset compatível.

        Reseta `validation_rules_user_managed=False` — devolve o controle
        ao DTO. O auto-management volta a ficar ativo: na próxima vez que
        a lista ficar vazia (sem deleção explícita), defaults se repopulam.
        """
        instr = self._instruction
        if instr is None:
            return

        instr.validation_rule_configs = []
        instr.validation_rules_user_managed = False
        if hasattr(instr, "_sync_declared_io"):
            instr._sync_declared_io()

        self._rebuild_validation_section()
        self.config_changed.emit(instr)

    def _on_add_validation_rule_clicked(self) -> None:
        """
        Encaminha o pedido pro controller. O controller é quem tem acesso
        aos storages (global + projeto) pra carregar templates e abrir o
        `AddValidationRuleDialog`. Quando o usuário aceitar, o controller
        chama de volta `append_validation_rule_config(cfg)` neste painel.
        """
        if self._instruction is not None:
            self.add_validation_rule_requested.emit(self._instruction)

    def append_validation_rule_config(self, cfg) -> None:
        """
        Anexa um `ValidationRuleConfig` à instrução atual. Chamado pelo
        controller após o `AddValidationRuleDialog` aceitar.

        Marca `validation_rules_user_managed=True` — qualquer adição
        manual sinaliza que o usuário assumiu controle, impedindo
        repopulação automática dos defaults na próxima abertura.
        """
        instr = self._instruction
        if instr is None or cfg is None:
            return
        if instr.validation_rule_configs is None:
            instr.validation_rule_configs = []
        instr.validation_rule_configs.append(cfg)
        instr.validation_rules_user_managed = True
        self._rebuild_validation_section()
        self.config_changed.emit(instr)

    @staticmethod
    def _get_field_tooltip(instr: StrategyInstructionDefinition, field_name: str) -> Optional[str]:
        """
        Lê o `metadata['help']` do dataclass field para uso como tooltip.

        Strings que começam com '<html>' são renderizadas como rich text
        pelo Qt automaticamente — listas, negrito, código e quebras de
        linha funcionam direto.
        """
        try:
            for f in dataclasses.fields(instr):
                if f.name == field_name and f.metadata:
                    help_text = f.metadata.get("help")
                    if help_text:
                        return str(help_text)
            return None
        except (TypeError, AttributeError):
            return None

    # ── Apply / Validate ──────────────────────────────────────────────────────

    def apply_pending(self) -> None:
        """Aplica mudanças pendentes do form ao DTO. Idempotente — se nada
        mudou desde o último apply (snapshot check em `_on_apply`), no-op.

        Chamado externamente pelo `StrategyEditorWidget` antes de propagar
        Run: garante que widgets editados mas não-aplicados sejam commitados
        no DTO em vez de serem silenciosamente descartados (Run usaria o DTO
        no estado salvo, rodando com input antigo).
        """
        self._on_apply()

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

        # Limpa campos não usados na configuração atual (ex: PIPELINE_
        # APPLICATION com pipeline_mode alterado entre LINEAR/DUAL deixa
        # lixo do modo anterior pendurado). Default no-op no base; DTOs
        # com choice fields que controlam quais campos são relevantes
        # sobrescrevem. Chamado SÓ no Apply pra que digitação intermediária
        # no panel não perca dados.
        if hasattr(instr, "prune_unused_fields"):
            instr.prune_unused_fields()

        # Sincroniza io novamente após choice fields + prune (modo pode
        # ter mudado e campos do modo oposto foram zerados acima).
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
        self.output_name_changed.emit(field_name, new_value or "")

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

    def _on_resource_triggers_rebuild(self, field_name: str) -> None:
        """
        Aplica o resource_id atualmente selecionado ao DTO e dispara
        rebuild. Usado por ResourceFieldSpec com triggers_rebuild=True
        (ex: DATASET_BALANCING.dataset_resource_id, cujo valor afeta as
        opções do dropdown quality_metric via dynamic_choices).
        """
        if not self._instruction:
            return
        combo = self._resource_field_widgets.get(field_name)
        if combo is None:
            return
        new_val = combo.currentData()  # resource_id ou None
        setattr(self._instruction, field_name, new_val)
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()
        QtCore.QTimer.singleShot(0, self._rebuild_and_notify)

    def _on_nested_changed(self, attr_name: str, new_dto: Any) -> None:
        if not self._instruction:
            return
        setattr(self._instruction, attr_name, new_dto)
        # Permite que a instrução manifeste que a mudança no DTO aninhado
        # afetou a visibilidade dos resource specs do panel (ex: mudar a
        # variante PLDA esconde/mostra o dropdown de dataset secundário).
        hook = getattr(self._instruction, "should_rebuild_on_nested_change", None)
        needs_rebuild = bool(hook(attr_name, new_dto)) if callable(hook) else False
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()
        if needs_rebuild:
            QtCore.QTimer.singleShot(0, self._rebuild_and_notify)
        else:
            self.config_changed.emit(self._instruction)

    def _rebuild_and_notify(self) -> None:
        if not self._instruction:
            return
        self._rebuild_form()
        # `instruction_specs_rebuilt` PRIMEIRO — força a 3ª coluna a
        # invalidar o card cacheado, que está com a lista de outputs
        # do estado anterior do triggers_rebuild. Em seguida emite
        # `config_changed` (fluxo normal de save).
        self.instruction_specs_rebuilt.emit(self._instruction)
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
