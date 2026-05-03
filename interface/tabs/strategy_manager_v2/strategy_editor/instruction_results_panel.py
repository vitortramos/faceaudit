# interface/tabs/strategy_manager_v2/strategy_editor/instruction_results_panel.py

"""
Painel de Resultados — terceira coluna do editor de estratégias.

Exibe, para a instrução atualmente selecionada na coluna de instruções:
- A lista de outputs declarados com estado runtime (⬜ placeholder /
  🟡 em memória / 🟢 salvo em disco) e resumo do payload (delegado ao
  `OutputRowWidget` existente).
- Quando aplicável, um gráfico inline 2D (distribuição de scores,
  resultado de validação 4-em-1, ou visualização forense de caso),
  ou um botão para abrir a visualização em dialog (caso 3D — trajetória).
- Botão "🔍 Expandir" para abrir o gráfico inline em dialog separado
  com toolbar matplotlib completa.

Cache: para cada instrução vista, mantém um "results card" com seus
widgets de gráfico já renderizados. Trocar de instrução = swap do card
ativo via `QStackedWidget`. Re-execução de uma instrução invalida o
card daquela instrução (drop do cache + redraw na próxima exibição).

Hidratação preguiçosa: payloads ainda no disco (após reabertura de
projeto) são carregados sob demanda quando o usuário seleciona a
instrução. Para datasets de scores muito grandes, o painel mostra um
botão "Carregar e visualizar" em vez de hidratar automaticamente —
evita travar a main thread com leituras de parquets de 5M+ pares.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)

from .output_row_widget import OutputRowWidget
from .section_header import SectionHeader
from .result_widgets import (
    CaseworkVisualizationWidget,
    ScoreDistributionWidget,
    ValidationResultWidget,
)


# Tipos de plot inline (não inclui trajetória — fica como botão).
_PLOT_KIND_SCORE_DISTRIBUTION = "score_distribution"
_PLOT_KIND_VALIDATION_RESULT = "validation_result"
_PLOT_KIND_CASEWORK_VISUALIZATION = "casework_visualization"
_PLOT_KIND_TRAJECTORY_BUTTON = "trajectory_button"  # pseudo-kind: só botão
_PLOT_KIND_HUBNESS_REPORT = "hubness_report"
_PLOT_KIND_HUBNESS_ABLATION = "hubness_ablation"


def _instruction_plot_kind(
    instruction: StrategyInstructionDefinition,
) -> Optional[str]:
    """Decide qual gráfico inline (se houver) cabe nessa instrução."""
    if instruction is None:
        return None
    t = instruction.instruction_type
    if t in (
        StrategyInstructionType.SIMILARITY_SCORING,
        StrategyInstructionType.PAIR_MINING,
        StrategyInstructionType.SCORE_CALIBRATION,
    ):
        return _PLOT_KIND_SCORE_DISTRIBUTION
    if t == StrategyInstructionType.STRATEGY_EVALUATION:
        return _PLOT_KIND_VALIDATION_RESULT
    if t == StrategyInstructionType.CASEWORK_VISUALIZATION:
        return _PLOT_KIND_CASEWORK_VISUALIZATION
    # Trajetória 3D só é produzida pelo PIPELINE_APPLICATION quando
    # application_policy.save_trajectory_data=True. PIPELINE_TRAINING
    # não computa trajetória mesmo com apply_to_training_datasets ligado
    # (usa PipelineApplicationPolicy default, sem checkpoints/trajetória).
    if t == StrategyInstructionType.PIPELINE_APPLICATION:
        return _PLOT_KIND_TRAJECTORY_BUTTON
    if t == StrategyInstructionType.HUBNESS_ANALYSIS:
        return _PLOT_KIND_HUBNESS_REPORT
    if t == StrategyInstructionType.HUBNESS_ABLATION:
        return _PLOT_KIND_HUBNESS_ABLATION
    return None


def _output_field_label(field_name: str) -> str:
    """Heurística simples para gerar label legível a partir do nome do campo."""
    return field_name.replace("_resource_id", "").replace("_", " ").title()


class _ResultsCard(QtWidgets.QWidget):
    """
    Container para uma instrução: outputs + (opcional) gráfico inline.

    Mantido em cache pelo `InstructionResultsPanel`. Não é um QDialog —
    é um QWidget que vai como página de um QStackedWidget.
    """

    # Re-emitidos pelo parent panel. Usados pelo botão expandir.
    view_trajectory_requested = QtCore.pyqtSignal(object)
    view_score_distribution_requested = QtCore.pyqtSignal(object)
    view_validation_result_requested = QtCore.pyqtSignal(object)
    view_casework_visualization_requested = QtCore.pyqtSignal(object)
    view_hubness_report_requested = QtCore.pyqtSignal(object)
    view_hubness_ablation_requested = QtCore.pyqtSignal(object)

    # Pedido de hidratação assíncrona — emitido pelo botão "Carregar e
    # visualizar" para datasets grandes ainda no disco.
    hydrate_requested = QtCore.pyqtSignal(str)  # resource_id

    def __init__(
        self,
        instruction: StrategyInstructionDefinition,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._instruction = instruction
        self._plot_kind = _instruction_plot_kind(instruction)
        self._plot_widget: Optional[QtWidgets.QWidget] = None
        self._output_widgets: Dict[str, OutputRowWidget] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        scroll_content = QtWidgets.QWidget()
        self._scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        self._scroll_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_layout.setSpacing(6)

        # ── Seção: Saídas ──────────────────────────────────────────────────
        self._scroll_layout.addWidget(SectionHeader("Saídas", "outputs"))

        self._outputs_layout = QtWidgets.QVBoxLayout()
        self._outputs_layout.setContentsMargins(8, 6, 8, 4)
        self._outputs_layout.setSpacing(6)
        out_wrapper = QtWidgets.QWidget()
        out_wrapper.setLayout(self._outputs_layout)
        self._scroll_layout.addWidget(out_wrapper)

        self._build_output_rows()

        # ── Seção: Visualização (opcional) ─────────────────────────────────
        if self._plot_kind is not None:
            self._scroll_layout.addWidget(SectionHeader("Visualização", "plots"))
            self._build_plot_section()

        self._scroll_layout.addStretch(1)

        self._scroll.setWidget(scroll_content)
        layout.addWidget(self._scroll, 1)

    # ── Construção de widgets ─────────────────────────────────────────────────

    def _build_output_rows(self) -> None:
        """Cria uma OutputRowWidget para cada output declarado pela instrução."""
        try:
            specs = self._instruction.get_resource_field_specs() or []
        except Exception:
            specs = []

        any_output = False
        for spec in specs:
            if not spec.is_output:
                continue
            field_name = spec.field_name
            resource_id = getattr(self._instruction, field_name, None) or ""
            resource_type = (
                spec.accepted_types[0]
                if spec.accepted_types
                else StrategyResourceType.UNKNOWN
            )
            row = OutputRowWidget(
                field_name=field_name,
                label=spec.label,
                resource_id=resource_id,
                resource_type=resource_type,
                parent=self,
                read_only=True,
            )
            self._outputs_layout.addWidget(row)
            self._output_widgets[field_name] = row
            any_output = True

        if not any_output:
            placeholder = QtWidgets.QLabel(
                "(esta instrução não declara recursos de saída)"
            )
            placeholder.setStyleSheet(
                "color: #888; font-style: italic; font-size: 11px; padding: 4px;"
            )
            self._outputs_layout.addWidget(placeholder)

    def _build_plot_section(self) -> None:
        """Cria o widget de gráfico inline (ou botão de trajetória)."""
        if self._plot_kind == _PLOT_KIND_TRAJECTORY_BUTTON:
            # Trajetória 3D fica como botão (abre dialog GraphViewerDialog).
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(8, 4, 8, 4)
            self._btn_trajectory = QtWidgets.QPushButton(
                "📊 Visualizar Trajetória 3D"
            )
            self._btn_trajectory.setStyleSheet(
                "QPushButton { border: 1px solid #E8A020; border-radius: 4px;"
                " background: #FFF8E8; color: #8B5E00;"
                " padding: 6px 14px; font-weight: bold; }"
                "QPushButton:hover { background: #FFEFC4; }"
                "QPushButton:disabled { color: #aaa; background: #f5f5f5;"
                " border: 1px solid #ccc; }"
            )
            self._btn_trajectory.setEnabled(True)
            self._btn_trajectory.clicked.connect(
                lambda: self.view_trajectory_requested.emit(self._instruction)
            )
            note = QtWidgets.QLabel(
                "Trajetória 3D abre em janela própria por exigir espaço para "
                "rotação/zoom."
            )
            note.setStyleSheet(
                "font-size: 10px; color: #666; font-style: italic;"
            )
            note.setWordWrap(True)
            row.addWidget(self._btn_trajectory)
            row.addStretch(1)
            self._scroll_layout.addLayout(row)
            self._scroll_layout.addWidget(note)
            return

        # Gráficos 2D inline + botão expandir.
        # Container vai num QStackedWidget interno: index 0 é placeholder/botão
        # de hidratação, index 1 é o widget renderizado.
        self._plot_stack = QtWidgets.QStackedWidget()
        self._scroll_layout.addWidget(self._plot_stack, 1)

        # ── Página 0: placeholder/loading ──────────────────────────────────
        placeholder_page = QtWidgets.QWidget()
        ph_layout = QtWidgets.QVBoxLayout(placeholder_page)
        ph_layout.setContentsMargins(8, 12, 8, 8)
        ph_layout.setSpacing(6)
        ph_layout.addStretch(1)

        self._placeholder_label = QtWidgets.QLabel("Execute para ver os resultados")
        self._placeholder_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setStyleSheet(
            "color: #666; font-style: italic; font-size: 12px;"
        )
        ph_layout.addWidget(self._placeholder_label)

        self._hydrate_button_row = QtWidgets.QHBoxLayout()
        self._hydrate_button_row.addStretch(1)
        self._btn_hydrate = QtWidgets.QPushButton("📊 Carregar e visualizar")
        self._btn_hydrate.setStyleSheet(
            "QPushButton { border: 1px solid #1565c0; border-radius: 4px;"
            " background: #e3f2fd; color: #1565c0;"
            " padding: 6px 14px; font-weight: bold; }"
            "QPushButton:hover { background: #bbdefb; }"
        )
        self._btn_hydrate.setVisible(False)
        self._btn_hydrate.clicked.connect(self._on_hydrate_clicked)
        self._hydrate_button_row.addWidget(self._btn_hydrate)
        self._hydrate_button_row.addStretch(1)
        ph_layout.addLayout(self._hydrate_button_row)

        ph_layout.addStretch(2)
        self._plot_stack.addWidget(placeholder_page)  # index 0

        # ── Página 1: widget de gráfico inline + botão expandir ────────────
        plot_page = QtWidgets.QWidget()
        pp_layout = QtWidgets.QVBoxLayout(plot_page)
        pp_layout.setContentsMargins(0, 0, 0, 0)
        pp_layout.setSpacing(2)

        # Linha do botão expandir, no topo da página de plot.
        expand_row = QtWidgets.QHBoxLayout()
        expand_row.setContentsMargins(8, 2, 8, 0)
        expand_row.addStretch(1)
        self._btn_expand = QtWidgets.QPushButton("🔍 Expandir")
        self._btn_expand.setStyleSheet(
            "QPushButton { border: 1px solid #888; border-radius: 3px;"
            " background: #f5f5f5; color: #333;"
            " padding: 3px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #eaeaea; }"
        )
        self._btn_expand.clicked.connect(self._on_expand_clicked)
        expand_row.addWidget(self._btn_expand)
        pp_layout.addLayout(expand_row)

        # Widget concreto inline (sem toolbar, sem help — modo compacto).
        if self._plot_kind == _PLOT_KIND_SCORE_DISTRIBUTION:
            self._plot_widget = ScoreDistributionWidget(
                parent=plot_page, score_dataset=None,
                show_toolbar=False,
            )
        elif self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
            self._plot_widget = ValidationResultWidget(
                parent=plot_page, result=None,
                show_toolbar=False, show_help=False,
            )
        elif self._plot_kind == _PLOT_KIND_CASEWORK_VISUALIZATION:
            self._plot_widget = CaseworkVisualizationWidget(
                parent=plot_page, result=None,
                show_toolbar=False, show_help=False,
            )
        elif self._plot_kind == _PLOT_KIND_HUBNESS_REPORT:
            from .result_widgets import HubnessReportWidget
            self._plot_widget = HubnessReportWidget(
                parent=plot_page, result=None,
                show_header=False,  # cabeçalho âmbar pesa muito inline; o
                                    # OutputRowWidget já mostra o resumo
                show_toolbar=False, show_help=False,
            )
        elif self._plot_kind == _PLOT_KIND_HUBNESS_ABLATION:
            from .result_widgets import HubnessAblationWidget
            self._plot_widget = HubnessAblationWidget(
                parent=plot_page, result=None,
                show_header=False, show_toolbar=False, show_help=False,
            )

        if self._plot_widget is not None:
            pp_layout.addWidget(self._plot_widget, 1)

        self._plot_stack.addWidget(plot_page)  # index 1

        # Default: placeholder (será trocado quando payload chegar).
        self._plot_stack.setCurrentIndex(0)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_state(
        self,
        definition: LRStrategyDefinition,
        executed_ids: Set[str],
        context: Optional[StrategyExecutionContextProtocol],
        resource_loader: Any = None,
    ) -> None:
        """
        Atualiza estados visuais. Atualiza a tabela de outputs com `OutputRowWidget.update_state`
        e tenta popular o gráfico inline se o payload está disponível.
        """
        executed = self._instruction.instruction_id in (executed_ids or set())

        for row in self._output_widgets.values():
            row.update_state(definition, self._instruction.instruction_id,
                             executed, context)

        if self._plot_widget is None:
            return  # sem plot inline; nada a fazer

        # Tenta resolver o payload do recurso de output relevante para o plot.
        payload, resource_id, in_memory = self._resolve_plot_payload(
            definition=definition, context=context,
        )

        if payload is not None:
            self._render_plot(payload)
            return

        # Sem payload em memória. Verifica se está no disco — então oferece
        # botão de carregamento. Caso contrário, fica em placeholder vazio.
        has_artifact = self._has_artifact_for_plot(definition)
        if has_artifact and resource_id:
            self._show_hydrate_button(resource_id)
        else:
            self._show_placeholder()

    def invalidate(self) -> None:
        """
        Marca o card como precisando de re-render do gráfico (ex.: quando a
        instrução foi re-executada). Próxima chamada a update_state vai
        atualizar com novo payload.
        """
        if self._plot_widget is None:
            return
        # Reset do widget de plot — força redraw.
        # Importa Hubness widgets aqui (lazy) pra evitar dep circular.
        from .result_widgets import HubnessReportWidget, HubnessAblationWidget
        try:
            if isinstance(self._plot_widget, ScoreDistributionWidget):
                self._plot_widget.set_dataset(None)
            elif isinstance(self._plot_widget, ValidationResultWidget):
                self._plot_widget.set_result(None)
            elif isinstance(self._plot_widget, CaseworkVisualizationWidget):
                self._plot_widget.set_result(None)
            elif isinstance(self._plot_widget, HubnessReportWidget):
                self._plot_widget.set_result(None)
            elif isinstance(self._plot_widget, HubnessAblationWidget):
                self._plot_widget.set_result(None)
        except Exception:
            pass
        self._show_placeholder()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _show_placeholder(self) -> None:
        if hasattr(self, "_plot_stack"):
            self._btn_hydrate.setVisible(False)
            self._placeholder_label.setText("Execute para ver os resultados")
            self._plot_stack.setCurrentIndex(0)

    def _show_hydrate_button(self, resource_id: str) -> None:
        if hasattr(self, "_plot_stack"):
            self._placeholder_label.setText(
                f"Recurso '{resource_id}' está em disco — clique para carregar"
            )
            self._btn_hydrate.setVisible(True)
            self._pending_hydrate_id = resource_id
            self._plot_stack.setCurrentIndex(0)

    def _on_hydrate_clicked(self) -> None:
        rid = getattr(self, "_pending_hydrate_id", None)
        if rid:
            self._btn_hydrate.setEnabled(False)
            self._btn_hydrate.setText("Carregando…")
            self.hydrate_requested.emit(rid)

    def _on_expand_clicked(self) -> None:
        """Abre o gráfico atual em dialog (delegado ao controller via signal)."""
        if self._plot_kind == _PLOT_KIND_SCORE_DISTRIBUTION:
            self.view_score_distribution_requested.emit(self._instruction)
        elif self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
            self.view_validation_result_requested.emit(self._instruction)
        elif self._plot_kind == _PLOT_KIND_CASEWORK_VISUALIZATION:
            self.view_casework_visualization_requested.emit(self._instruction)
        elif self._plot_kind == _PLOT_KIND_HUBNESS_REPORT:
            self.view_hubness_report_requested.emit(self._instruction)
        elif self._plot_kind == _PLOT_KIND_HUBNESS_ABLATION:
            self.view_hubness_ablation_requested.emit(self._instruction)

    def _render_plot(self, payload: Any) -> None:
        """Atualiza o widget de plot com o payload recebido."""
        if self._plot_widget is None:
            return
        from .result_widgets import HubnessReportWidget, HubnessAblationWidget
        try:
            if isinstance(self._plot_widget, ScoreDistributionWidget):
                use_log_lr = self._is_calibrated_scores_output()
                self._plot_widget.set_dataset(
                    payload, use_log_lr=use_log_lr
                )
            elif isinstance(self._plot_widget, ValidationResultWidget):
                self._plot_widget.set_result(payload)
            elif isinstance(self._plot_widget, CaseworkVisualizationWidget):
                self._plot_widget.set_result(payload)
            elif isinstance(self._plot_widget, HubnessReportWidget):
                self._plot_widget.set_result(payload)
            elif isinstance(self._plot_widget, HubnessAblationWidget):
                self._plot_widget.set_result(payload)
        except Exception:
            self._show_placeholder()
            return
        if hasattr(self, "_plot_stack"):
            self._btn_hydrate.setVisible(False)
            self._plot_stack.setCurrentIndex(1)

    def _is_calibrated_scores_output(self) -> bool:
        """True se o output relevante é CALIBRATED_SCORES (usa log_lr)."""
        try:
            specs = self._instruction.get_resource_field_specs() or []
        except Exception:
            return False
        for spec in specs:
            if not spec.is_output:
                continue
            if StrategyResourceType.CALIBRATED_SCORES in spec.accepted_types:
                return True
        return False

    def _resolve_plot_payload(
        self,
        definition: LRStrategyDefinition,
        context: Optional[StrategyExecutionContextProtocol],
    ) -> Tuple[Any, Optional[str], bool]:
        """
        Encontra o payload do recurso de output do tipo certo para o plot.

        Retorna (payload, resource_id, in_memory). `payload` é None se não
        estiver imediatamente disponível em memória. `in_memory=True`
        indica que veio direto do contexto vivo; False = veio da definition
        com payload já populado.
        """
        target_types = self._plot_target_types()
        if not target_types:
            return None, None, False

        target_resource_id: Optional[str] = None
        try:
            specs = self._instruction.get_resource_field_specs() or []
        except Exception:
            specs = []
        for spec in specs:
            if not spec.is_output:
                continue
            if any(t in target_types for t in spec.accepted_types):
                target_resource_id = getattr(self._instruction, spec.field_name, None)
                break

        if not target_resource_id:
            return None, None, False

        # 1) Contexto vivo
        if context is not None and context.has_resource(target_resource_id):
            ctx_res = context.get_resource(target_resource_id)
            if ctx_res is not None and ctx_res.has_payload:
                return ctx_res.payload, target_resource_id, True

        # 2) Definition (payload já carregado no recurso)
        if definition is not None:
            existing = definition.get_resource_by_id(target_resource_id)
            if existing is not None and existing.has_payload:
                return existing.payload, target_resource_id, False

        return None, target_resource_id, False

    def _has_artifact_for_plot(self, definition: LRStrategyDefinition) -> bool:
        """True se há artefato em disco para o output do plot."""
        target_types = self._plot_target_types()
        if not target_types or definition is None:
            return False
        try:
            specs = self._instruction.get_resource_field_specs() or []
        except Exception:
            return False
        for spec in specs:
            if not spec.is_output:
                continue
            if not any(t in target_types for t in spec.accepted_types):
                continue
            rid = getattr(self._instruction, spec.field_name, None)
            if not rid:
                continue
            existing = definition.get_resource_by_id(rid)
            if existing is not None and existing.has_artifact:
                return True
        return False

    def _plot_target_types(self) -> Tuple[StrategyResourceType, ...]:
        if self._plot_kind == _PLOT_KIND_SCORE_DISTRIBUTION:
            return (
                StrategyResourceType.SIMILARITY_SCORES,
                StrategyResourceType.CALIBRATED_SCORES,
            )
        if self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
            return (StrategyResourceType.VALIDATION_RESULT,)
        if self._plot_kind == _PLOT_KIND_CASEWORK_VISUALIZATION:
            return (StrategyResourceType.CASEWORK_VISUALIZATION,)
        if self._plot_kind == _PLOT_KIND_HUBNESS_REPORT:
            return (StrategyResourceType.HUBNESS_REPORT,)
        if self._plot_kind == _PLOT_KIND_HUBNESS_ABLATION:
            return (StrategyResourceType.HUBNESS_ABLATION_RESULT,)
        return ()


class InstructionResultsPanel(QtWidgets.QWidget):
    """
    Terceira coluna do editor de estratégias.

    Public API:
        load_instruction(instruction): mostra o card daquela instrução
        show_empty(): volta ao estado vazio (nenhuma instrução selecionada)
        update_runtime_state(definition, executed_ids, context): atualiza
            estados de outputs e popula gráficos inline quando aplicável
        invalidate_instruction(instruction_id): drop do card cacheado
            (próxima exibição é fresca; usar quando instrução é re-executada)
        set_resource_loader(loader): registra o loader pra que cards possam
            pedir hidratação on-demand de payloads ainda no disco

    Signals:
        view_trajectory_requested(StrategyInstructionDefinition)
        view_score_distribution_requested(StrategyInstructionDefinition)
        view_validation_result_requested(StrategyInstructionDefinition)
        view_casework_visualization_requested(StrategyInstructionDefinition)
        view_hubness_report_requested(StrategyInstructionDefinition)
        view_hubness_ablation_requested(StrategyInstructionDefinition)
        hydrate_requested(StrategyInstructionDefinition, str): instrução +
            resource_id solicitado para carregamento
    """

    view_trajectory_requested = QtCore.pyqtSignal(object)
    view_score_distribution_requested = QtCore.pyqtSignal(object)
    view_validation_result_requested = QtCore.pyqtSignal(object)
    view_casework_visualization_requested = QtCore.pyqtSignal(object)
    view_hubness_report_requested = QtCore.pyqtSignal(object)
    view_hubness_ablation_requested = QtCore.pyqtSignal(object)
    hydrate_requested = QtCore.pyqtSignal(object, str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._cards: Dict[str, _ResultsCard] = {}
        self._current_instruction: Optional[StrategyInstructionDefinition] = None
        self._latest_definition: Optional[LRStrategyDefinition] = None
        self._latest_executed_ids: Set[str] = set()
        self._latest_context: Optional[StrategyExecutionContextProtocol] = None
        self._resource_loader: Any = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack, 1)

        # ── Index 0: empty state ──────────────────────────────────────────────
        empty_page = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_page)
        empty_layout.addStretch(1)
        empty_lbl = QtWidgets.QLabel("Selecione uma instrução para ver os resultados")
        empty_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        empty_lbl.setWordWrap(True)
        empty_layout.addWidget(empty_lbl)
        empty_layout.addStretch(2)
        self._stack.addWidget(empty_page)  # index 0

    # ── Public API ────────────────────────────────────────────────────────────

    def show_empty(self) -> None:
        self._current_instruction = None
        self._stack.setCurrentIndex(0)

    def load_instruction(
        self,
        instruction: Optional[StrategyInstructionDefinition],
    ) -> None:
        if instruction is None:
            self.show_empty()
            return
        self._current_instruction = instruction
        card = self._cards.get(instruction.instruction_id)
        if card is None:
            card = self._build_card(instruction)
            self._cards[instruction.instruction_id] = card
            self._stack.addWidget(card)
        self._stack.setCurrentWidget(card)
        # Popula com último estado conhecido.
        if self._latest_definition is not None:
            card.update_state(
                definition=self._latest_definition,
                executed_ids=self._latest_executed_ids,
                context=self._latest_context,
                resource_loader=self._resource_loader,
            )

    def update_runtime_state(
        self,
        definition: Optional[LRStrategyDefinition],
        executed_ids: Optional[Set[str]],
        context: Optional[StrategyExecutionContextProtocol] = None,
    ) -> None:
        self._latest_definition = definition
        self._latest_executed_ids = executed_ids or set()
        self._latest_context = context
        if self._current_instruction is None:
            return
        card = self._cards.get(self._current_instruction.instruction_id)
        if card is None or definition is None:
            return
        card.update_state(
            definition=definition,
            executed_ids=self._latest_executed_ids,
            context=context,
            resource_loader=self._resource_loader,
        )

    def invalidate_instruction(self, instruction_id: str) -> None:
        """
        Drop do card cacheado (cleanup gráfico). Próxima exibição é fresca.
        Usado quando uma instrução é re-executada.
        """
        card = self._cards.pop(instruction_id, None)
        if card is None:
            return
        # Remove do stack
        idx = self._stack.indexOf(card)
        if idx >= 0:
            self._stack.removeWidget(card)
        card.deleteLater()
        if (
            self._current_instruction is not None
            and self._current_instruction.instruction_id == instruction_id
        ):
            # Recria imediatamente o card ativo, com estado mais recente.
            self.load_instruction(self._current_instruction)

    def invalidate_all(self) -> None:
        """Limpa todos os cards. Usado em troca de estratégia/projeto."""
        for instr_id in list(self._cards.keys()):
            self.invalidate_instruction(instr_id)
        self.show_empty()

    def update_output_name(self, field_name: str, new_value: str) -> None:
        """Sincroniza o nome exibido (read-only) no card ativo quando o
        usuário edita o nome no painel de configuração."""
        if self._current_instruction is None:
            return
        instr_id = getattr(self._current_instruction, "instruction_id", None)
        if instr_id is None:
            return
        card = self._cards.get(instr_id)
        if card is None:
            return
        row = card._output_widgets.get(field_name)
        if row is not None:
            row.set_resource_name(new_value)

    def set_resource_loader(self, loader: Any) -> None:
        self._resource_loader = loader

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_card(
        self, instruction: StrategyInstructionDefinition
    ) -> _ResultsCard:
        card = _ResultsCard(instruction, parent=self)
        # Re-emite sinais do card para o panel.
        card.view_trajectory_requested.connect(self.view_trajectory_requested.emit)
        card.view_score_distribution_requested.connect(
            self.view_score_distribution_requested.emit
        )
        card.view_validation_result_requested.connect(
            self.view_validation_result_requested.emit
        )
        card.view_casework_visualization_requested.connect(
            self.view_casework_visualization_requested.emit
        )
        card.view_hubness_report_requested.connect(
            self.view_hubness_report_requested.emit
        )
        card.view_hubness_ablation_requested.connect(
            self.view_hubness_ablation_requested.emit
        )
        # Hydrate com instrução + resource_id (parent enriquece).
        card.hydrate_requested.connect(
            lambda rid, instr=instruction: self.hydrate_requested.emit(instr, rid)
        )
        return card
