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

from PyQt6 import QtCore, QtGui, QtWidgets


def _with_wait_cursor():
    """
    Context manager local que ativa o cursor de espera durante operações
    bloqueantes do painel de resultados (renderização de gráficos custosos:
    KDE de 5M+ pares, ValidationResultWidget com 4 abas matplotlib, etc.).
    Garante restoreOverrideCursor mesmo em caso de exceção e força repaint
    via processEvents() antes de bloquear.
    """
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            QtWidgets.QApplication.processEvents()
            yield
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    return _cm()

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
from .section_header import SectionHeader, ColumnHeader
from .result_widgets import (
    CaseworkVisualizationWidget,
    ScoreDistributionWidget,
    ValidationResultWidget,
)
from .result_widgets.png_snapshot_widget import PngSnapshotWidget


# Alias mantido pra preservar todos os call sites internos do
# instruction_results_panel sem mexer em dezenas de linhas. A versão
# pública vive em `result_widgets/png_snapshot_widget.py` — usada
# também pela janela de comparação de estratégias.
_PngSnapshotWidget = PngSnapshotWidget


class _TabbedPngSnapshotWidget(QtWidgets.QWidget):
    """QTabWidget de `_PngSnapshotWidget`s — usado pra STRATEGY_EVALUATION,
    que tem 4 plots (distribuição de LRs, Tippett, DET, ECE) em vez de 1.

    Espelha visualmente o `ValidationResultWidget` (4 abas) mas substitui
    cada FigureCanvas+plot vivo por um QPixmap pré-renderizado. Render
    instantâneo, sem matplotlib na main thread. Plot vivo continua
    disponível pelo botão 🔍 Expandir → `ValidationResultViewerDialog`.
    """

    def __init__(self, tab_keys, tab_titles: Dict[str, str],
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QtWidgets.QTabWidget()
        self._snapshots: Dict[str, _PngSnapshotWidget] = {}
        for key in tab_keys:
            snap = _PngSnapshotWidget()
            self._snapshots[key] = snap
            self._tabs.addTab(snap, tab_titles.get(key, key))
        layout.addWidget(self._tabs, 1)

    def show_images(self, paths_dict: Dict[str, Any]) -> int:
        """Carrega os PNGs em cada aba. Retorna número de PNGs carregados
        com sucesso (algumas abas podem ficar em placeholder se o PNG
        correspondente não existir / falhar)."""
        loaded = 0
        for key, snap in self._snapshots.items():
            target = paths_dict.get(key)
            if target is not None and snap.show_image(target):
                loaded += 1
            else:
                snap.show_placeholder("(sem PNG disponível)")
        return loaded

    def show_placeholder(self, text: str) -> None:
        for snap in self._snapshots.values():
            snap.show_placeholder(text)


# Tipos de plot inline (não inclui trajetória — fica como botão).
_PLOT_KIND_SCORE_DISTRIBUTION = "score_distribution"
_PLOT_KIND_VALIDATION_RESULT = "validation_result"
_PLOT_KIND_CASEWORK_VISUALIZATION = "casework_visualization"
_PLOT_KIND_TRAJECTORY_BUTTON = "trajectory_button"  # pseudo-kind: só botão
_PLOT_KIND_HUBNESS_REPORT = "hubness_report"
_PLOT_KIND_HUBNESS_ABLATION = "hubness_ablation"
_PLOT_KIND_TYPICALITY_DISTRIBUTION = "typicality_distribution"


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
        # Tipicidade pareada produz SimilarityScoreDataset com `raw_score`
        # = combinador(t_probe, t_ref) — mesmo schema, mesmo widget.
        StrategyInstructionType.TYPICALITY_PAIR_SCORING,
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
    if t == StrategyInstructionType.KNN_TYPICALITY_MODEL_TRAINING:
        return _PLOT_KIND_TYPICALITY_DISTRIBUTION
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
    view_typicality_distribution_requested = QtCore.pyqtSignal(object)

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
        # Cache do último payload renderizado — evita redesenhar gráficos
        # caros (KDE em 5M+ pares, etc.) quando update_state é chamado
        # repetidamente com o mesmo payload.
        self._last_rendered_payload_id: Optional[int] = None
        self._last_render_args: Optional[tuple] = None
        # True quando o plot já foi desenhado pelo menos uma vez. Usado
        # pra evitar voltar pra placeholder quando o payload some
        # transitoriamente do contexto/definition (ex: contexto resetado
        # após edição) — o canvas matplotlib já tem a figura em memória.
        self._plot_rendered_once: bool = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # parent=self (o card) em TODOS os widgets internos pra evitar
        # criação como top-level window mesmo brevemente. Antes vários
        # eram parentless até addWidget/setWidget reparentá-los — causa
        # do flash "janelinha" na primeira seleção de cada instrução
        # (criação lazy do card).
        self._scroll = QtWidgets.QScrollArea(parent=self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        scroll_content = QtWidgets.QWidget(parent=self._scroll)
        self._scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        self._scroll_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_layout.setSpacing(6)

        # NOTA: o sub-header "Saídas" foi removido — o cabeçalho principal
        # da coluna (ColumnHeader "Saídas" no InstructionResultsPanel) já
        # cumpre esse papel; ter os dois ao mesmo tempo era redundante.

        self._outputs_layout = QtWidgets.QVBoxLayout()
        self._outputs_layout.setContentsMargins(8, 6, 8, 4)
        self._outputs_layout.setSpacing(6)
        out_wrapper = QtWidgets.QWidget(parent=scroll_content)
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
            # Empacotado num container próprio (`_trajectory_container`)
            # pra que `update_state` possa esconder/mostrar conforme
            # estado da instrução (executada + save_trajectory_data ativo).
            self._trajectory_container = QtWidgets.QWidget(parent=self)
            tc_layout = QtWidgets.QVBoxLayout(self._trajectory_container)
            tc_layout.setContentsMargins(0, 0, 0, 0)
            tc_layout.setSpacing(2)

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
            tc_layout.addLayout(row)
            tc_layout.addWidget(note)

            # Mensagens alternativas mostradas QUANDO o botão não aparece —
            # explica ao usuário POR QUE a trajetória está indisponível
            # ("ainda não executou" vs "save_trajectory_data está OFF").
            self._trajectory_disabled_msg = QtWidgets.QLabel()
            self._trajectory_disabled_msg.setWordWrap(True)
            self._trajectory_disabled_msg.setStyleSheet(
                "font-size: 11px; color: #888; font-style: italic;"
                " padding: 8px 12px;"
            )
            self._trajectory_disabled_msg.setVisible(False)

            self._scroll_layout.addWidget(self._trajectory_container)
            self._scroll_layout.addWidget(self._trajectory_disabled_msg)
            return

        # Gráficos 2D inline + botão expandir.
        # Container vai num QStackedWidget interno: index 0 é placeholder/botão
        # de hidratação, index 1 é o widget renderizado.
        self._plot_stack = QtWidgets.QStackedWidget(parent=self)
        self._scroll_layout.addWidget(self._plot_stack, 1)

        # ── Página 0: placeholder/loading ──────────────────────────────────
        placeholder_page = QtWidgets.QWidget(parent=self._plot_stack)
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
        plot_page = QtWidgets.QWidget(parent=self._plot_stack)
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
            # Pra SIMILARITY_SCORING e PAIR_MINING usamos o snapshot PNG
            # gerado durante a persistência — render instantâneo via
            # QPixmap (sem reconstruir a figura matplotlib na main thread,
            # sem rodar gaussian_kde de novo). Ambas produzem
            # `SIMILARITY_SCORES`, então o executor já cuida do PNG no
            # `persist_produced_resource`. SCORE_CALIBRATION (output
            # `CALIBRATED_SCORES`) tem DUAS vistas em abas: distribuição
            # de log-LR + curva de calibração (raw_score × log-LR).
            if self._uses_png_snapshot_inline():
                if (
                    self._instruction.instruction_type
                    == StrategyInstructionType.SCORE_CALIBRATION
                ):
                    self._plot_widget = _TabbedPngSnapshotWidget(
                        tab_keys=("distribution", "curve"),
                        tab_titles={
                            "distribution": "Distribuição",
                            "curve": "Curva de Calibração",
                        },
                        parent=plot_page,
                    )
                else:
                    self._plot_widget = _PngSnapshotWidget(
                        parent=plot_page,
                    )
            else:
                self._plot_widget = ScoreDistributionWidget(
                    parent=plot_page, score_dataset=None,
                    show_toolbar=False,
                )
        elif self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
            # STRATEGY_EVALUATION: 4 PNGs (distribuição de LRs, Tippett,
            # DET, ECE) gerados pelo executor durante a persistência.
            # Tabbed snapshot widget reproduz o layout do
            # `ValidationResultWidget` mas sem reconstruir 4 figuras
            # matplotlib. Expandir abre o widget vivo completo.
            from faceaudit.models.validation_result_plot import (
                PLOT_KEYS, PLOT_TAB_TITLE,
            )
            self._plot_widget = _TabbedPngSnapshotWidget(
                tab_keys=PLOT_KEYS,
                tab_titles=PLOT_TAB_TITLE,
                parent=plot_page,
            )
        elif self._plot_kind == _PLOT_KIND_CASEWORK_VISUALIZATION:
            # CASEWORK_VISUALIZATION: mesma estratégia das instruções de
            # scoring — executor gera PNG em `<project>/graphics/` durante
            # a persistência, painel mostra via QPixmap. Expandir abre o
            # plot vivo (CaseworkVisualizationWidget completo).
            self._plot_widget = _PngSnapshotWidget(
                parent=plot_page,
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
        elif self._plot_kind == _PLOT_KIND_TYPICALITY_DISTRIBUTION:
            from .result_widgets.typicality_distribution_widget import (
                TypicalityDistributionWidget,
            )
            self._plot_widget = TypicalityDistributionWidget(
                parent=plot_page, model=None, show_toolbar=False,
            )

        if self._plot_widget is not None:
            # Wrap em "card" branco padronizado pra que todos os tipos de
            # plot inline (snapshot PNG simples, tabbed PNG, ScoreDistribution
            # vivo, Hubness) tenham a mesma moldura visual. Sem isso, plots
            # baseados em _PngSnapshotWidget puro ficavam direto no fundo
            # cinza-escuro do painel, enquanto STRATEGY_EVALUATION (QTabWidget)
            # já trazia branco da Qt default — daí a inconsistência.
            plot_card = QtWidgets.QFrame(parent=plot_page)
            plot_card.setObjectName("plot_card")
            plot_card.setStyleSheet(
                "QFrame#plot_card {"
                "  background-color: white;"
                "  border: 1px solid #d0d8e0;"
                "  border-radius: 4px;"
                "}"
            )
            # Largura: expande pra preencher a coluna. Altura: derivada do
            # heightForWidth do plot widget interno (que escala o PNG pela
            # largura preservando aspect ratio). Mínimo pra placeholder.
            card_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            card_policy.setHeightForWidth(True)
            plot_card.setSizePolicy(card_policy)
            plot_card.setMinimumHeight(140)
            card_layout = QtWidgets.QVBoxLayout(plot_card)
            card_layout.setContentsMargins(0, 0, 0, 0)
            card_layout.setSpacing(0)
            card_layout.addWidget(self._plot_widget, 1)
            pp_layout.addWidget(plot_card, 1)

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

        # Extrai project_root do storage agregado pelo resource_loader. Usado
        # pelas OutputRowWidgets para popular ResourceInfoWidget(compact=True)
        # com path/hash/tamanho do artifact em disco.
        project_root = None
        try:
            storage = getattr(resource_loader, "_storage", None) if resource_loader else None
            if storage is not None:
                project_root = getattr(storage, "project_root", None)
        except Exception:
            project_root = None

        for row in self._output_widgets.values():
            row.update_state(
                definition,
                self._instruction.instruction_id,
                executed,
                context,
                project_root=project_root,
            )

        # Caminho do botão de Trajetória 3D (PIPELINE_APPLICATION): só
        # exibe quando a instrução já foi executada E
        # `application_policy.save_trajectory_data=True`. Caso contrário,
        # mostra mensagem explicando o porquê.
        if self._plot_kind == _PLOT_KIND_TRAJECTORY_BUTTON:
            self._update_trajectory_button_visibility(executed)
            return

        if self._plot_widget is None:
            return  # sem plot inline; nada a fazer

        # Caminho dedicado pros plots com snapshot PNG (SIMILARITY_SCORING,
        # PAIR_MINING, CASEWORK_VISUALIZATION → `_PngSnapshotWidget` simples;
        # STRATEGY_EVALUATION → `_TabbedPngSnapshotWidget` com 4 abas). O
        # inline mostra os PNGs gravados pelo executor, sem reconstruir
        # matplotlib.
        if isinstance(
            self._plot_widget,
            (_PngSnapshotWidget, _TabbedPngSnapshotWidget),
        ):
            self._update_snapshot_state(
                definition=definition,
                context=context,
                resource_loader=resource_loader,
            )
            return

        # Tenta resolver o payload do recurso de output relevante para o plot.
        payload, resource_id, in_memory = self._resolve_plot_payload(
            definition=definition, context=context,
        )

        if payload is not None:
            self._render_plot(payload)
            return

        # Caminho rápido pra ScoreDistributionWidget: tenta usar sidecar
        # `<stem>_dist.json` pré-computado em vez de hidratar o parquet
        # de 5M+ pares. Funciona sem o usuário clicar "Carregar e
        # visualizar" — gráfico aparece imediatamente.
        if (
            self._plot_kind == _PLOT_KIND_SCORE_DISTRIBUTION
            and resource_id is not None
            and isinstance(self._plot_widget, ScoreDistributionWidget)
        ):
            summary = self._try_load_dist_summary(definition, resource_id, resource_loader)
            if summary is not None:
                self._render_plot_from_summary(summary)
                return

        # Payload sumiu transitoriamente do contexto/definition (ex:
        # contexto resetado por uma edição em outra parte da UI). Se o
        # gráfico já foi desenhado anteriormente nesta sessão, mantém
        # ele visível — o canvas matplotlib preserva a figura em memória
        # e voltar pra placeholder/hydrate forçaria o usuário a clicar
        # "Carregar e visualizar" de novo apesar do gráfico existir.
        if self._plot_rendered_once and hasattr(self, "_plot_stack"):
            self._plot_stack.setCurrentIndex(1)
            self._btn_hydrate.setVisible(False)
            return

        # Primeira exibição sem payload em memória. Verifica se está no
        # disco — então oferece botão de carregamento. Caso contrário,
        # fica em placeholder vazio.
        has_artifact = self._has_artifact_for_plot(definition)
        if has_artifact and resource_id:
            self._show_hydrate_button(resource_id)
        else:
            self._show_placeholder()

    def _try_load_dist_summary(
        self,
        definition: LRStrategyDefinition,
        resource_id: str,
        resource_loader: Any,
    ) -> Optional[dict]:
        """
        Tenta carregar o sidecar de distribuição pré-computado pro
        `resource_id`. Retorna o dict pronto pra plotar, ou None se
        ausente / inválido / sem caminho resolvível.
        """
        if definition is None or resource_loader is None:
            return None
        existing = definition.get_resource_by_id(resource_id)
        if existing is None or not existing.has_artifact:
            return None
        try:
            storage = getattr(resource_loader, "_storage", None)
            project_root = getattr(storage, "project_root", None) if storage else None
            if project_root is None:
                return None
            from pathlib import Path
            from faceaudit.models.similarity.similarity_score_dataset import (
                SimilarityScoreDataset,
            )
            parquet_abs = Path(project_root) / existing.artifact_ref.path
            if not parquet_abs.exists():
                return None
            return SimilarityScoreDataset.try_load_distribution_summary(parquet_abs)
        except Exception:
            return None

    def _render_plot_from_summary(self, summary: dict) -> None:
        """
        Versão do `_render_plot` que usa um sidecar de distribuição em
        vez do payload em memória. Mesmo cache de identidade — `id()`
        do dict serve como chave de fingerprint.
        """
        if not isinstance(self._plot_widget, ScoreDistributionWidget):
            return
        summary_id = id(summary)
        # Args derivados — pra distinguir de uma render anterior baseada
        # em payload (use_log_lr) e evitar cache hit indevido.
        extra_args = ("summary", summary.get("score_type", "raw"))
        if (
            self._last_rendered_payload_id == summary_id
            and self._last_render_args == extra_args
            and self._plot_rendered_once
        ):
            if hasattr(self, "_plot_stack"):
                self._btn_hydrate.setVisible(False)
                self._plot_stack.setCurrentIndex(1)
            return
        with _with_wait_cursor():
            try:
                self._plot_widget.set_distribution_summary(summary)
            except Exception:
                self._show_placeholder()
                return
            self._last_rendered_payload_id = summary_id
            self._last_render_args = extra_args
            self._plot_rendered_once = True
            if hasattr(self, "_plot_stack"):
                self._btn_hydrate.setVisible(False)
                self._plot_stack.setCurrentIndex(1)

    def _update_trajectory_button_visibility(self, executed: bool) -> None:
        """Decide se mostra o botão Visualizar Trajetória 3D.

        Regras (PIPELINE_APPLICATION):
        - Instrução NÃO executada → botão escondido + mensagem
          orientando a executar.
        - Executada + `application_policy.save_trajectory_data=False` →
          botão escondido + mensagem explicando que precisa marcar a
          opção na config e re-executar.
        - Executada + `save_trajectory_data=True` → botão visível.

        Sem isso, o botão aparecia sempre e clicar antes de executar
        (ou com a opção off) abria um diálogo vazio/erro.
        """
        # Checagem defensiva — só PIPELINE_APPLICATION tem
        # application_policy. Se outro tipo cair aqui por engano,
        # esconde tudo.
        policy = getattr(self._instruction, "application_policy", None)
        save_trajectory = bool(
            policy is not None
            and getattr(policy, "save_trajectory_data", False)
        )

        if not executed:
            self._trajectory_container.setVisible(False)
            self._trajectory_disabled_msg.setText(
                "ℹ️ Execute esta instrução para visualizar a trajetória 3D."
            )
            self._trajectory_disabled_msg.setVisible(True)
        elif not save_trajectory:
            self._trajectory_container.setVisible(False)
            self._trajectory_disabled_msg.setText(
                "ℹ️ Trajetória 3D não disponível: ative <b>Salvar dados de "
                "trajetória</b> na configuração da instrução e re-execute."
            )
            self._trajectory_disabled_msg.setVisible(True)
        else:
            self._trajectory_container.setVisible(True)
            self._trajectory_disabled_msg.setVisible(False)

    def _update_snapshot_state(
        self,
        definition: LRStrategyDefinition,
        context: Optional[StrategyExecutionContextProtocol],
        resource_loader: Any,
    ) -> None:
        """Atualiza o snapshot PNG inline. Mostra a(s) imagem(ns) se
        existem; senão exibe placeholder. Cobre tanto o
        `_PngSnapshotWidget` (1 PNG: scoring/pair_mining/casework) quanto
        o `_TabbedPngSnapshotWidget` (4 PNGs: validation)."""
        # Resolve o resource_id do output relevante.
        target_types = self._plot_target_types()
        target_resource_id: Optional[str] = None
        try:
            specs = self._instruction.get_resource_field_specs() or []
        except Exception:
            specs = []
        for spec in specs:
            if not spec.is_output:
                continue
            if any(t in target_types for t in spec.accepted_types):
                target_resource_id = getattr(
                    self._instruction, spec.field_name, None,
                )
                break

        widget = self._plot_widget
        loaded = False

        if isinstance(widget, _TabbedPngSnapshotWidget):
            # Multi-PNG: STRATEGY_EVALUATION (4 PNGs) ou SCORE_CALIBRATION
            # (2 PNGs: distribuição + curva de calibração).
            paths_dict = self._resolve_png_paths_dict(
                definition, target_resource_id or "", resource_loader,
            )
            if paths_dict:
                loaded = widget.show_images(paths_dict) > 0
            if not loaded:
                if self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
                    placeholder_msg = "Execute para gerar visualização da validação"
                else:
                    placeholder_msg = "Execute para gerar visualização da calibração"
                widget.show_placeholder(placeholder_msg)
        elif isinstance(widget, _PngSnapshotWidget):
            # Plot único.
            png_path = self._resolve_png_path(
                definition, target_resource_id or "", resource_loader,
            )
            if png_path is not None and widget.show_image(png_path):
                loaded = True
            else:
                widget.show_placeholder(
                    "Execute para gerar visualização da distribuição"
                )
        else:
            return

        if hasattr(self, "_plot_stack"):
            self._btn_hydrate.setVisible(False)
            self._plot_stack.setCurrentIndex(1)
        if loaded:
            self._plot_rendered_once = True

    def _resolve_png_paths_dict(
        self,
        definition: LRStrategyDefinition,
        resource_id: str,
        resource_loader: Any,
    ):
        """Versão "multi-PNG" do `_resolve_png_path` — usada pelo tabbed
        snapshot widget. Retorna dict `{plot_key: Path}` ou None se
        faltam dados.

        Hoje cobre dois casos: STRATEGY_EVALUATION (4 PNGs) e
        SCORE_CALIBRATION (2 PNGs: distribuição + curva de calibração).
        """
        if resource_loader is None or definition is None or not resource_id:
            return None
        storage = getattr(resource_loader, "_storage", None)
        project_root = getattr(storage, "project_root", None) if storage else None
        strategy_id = getattr(definition, "strategy_id", None)
        if project_root is None or not strategy_id:
            return None
        if self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
            from faceaudit.models.validation_result_plot import (
                validation_result_png_paths,
            )
            return validation_result_png_paths(
                project_root, strategy_id, resource_id,
            )
        if (
            self._plot_kind == _PLOT_KIND_SCORE_DISTRIBUTION
            and self._instruction.instruction_type
            == StrategyInstructionType.SCORE_CALIBRATION
        ):
            from faceaudit.models.similarity.distribution_plot import (
                score_distribution_png_path,
            )
            from faceaudit.models.similarity.calibration_curve_plot import (
                calibration_curve_png_path,
            )
            return {
                "distribution": score_distribution_png_path(
                    project_root, strategy_id, resource_id,
                    calibrated=True,
                ),
                "curve": calibration_curve_png_path(
                    project_root, strategy_id, resource_id,
                ),
            }
        return None

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
            if isinstance(self._plot_widget, _TabbedPngSnapshotWidget):
                if self._plot_kind == _PLOT_KIND_VALIDATION_RESULT:
                    placeholder_msg = "Execute para gerar visualização da validação"
                else:
                    placeholder_msg = "Execute para gerar visualização da calibração"
                self._plot_widget.show_placeholder(placeholder_msg)
            elif isinstance(self._plot_widget, _PngSnapshotWidget):
                self._plot_widget.show_placeholder(
                    "Execute para gerar visualização"
                )
            elif isinstance(self._plot_widget, ScoreDistributionWidget):
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
        # Limpa cache pra que próxima render aconteça mesmo se o id() do
        # payload bater (re-execução pode produzir um objeto novo com mesmo
        # conteúdo mas id diferente — defensivo).
        self._last_rendered_payload_id = None
        self._last_render_args = None
        self._plot_rendered_once = False
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
        elif self._plot_kind == _PLOT_KIND_TYPICALITY_DISTRIBUTION:
            self.view_typicality_distribution_requested.emit(self._instruction)

    def _render_plot(self, payload: Any) -> None:
        """Atualiza o widget de plot com o payload recebido.

        Cache de identidade: se o último payload desenhado foi o mesmo
        objeto (mesmo `id()`) e os parâmetros derivados não mudaram,
        pula o redraw. Resolve lentidão crescente — `update_state` é
        chamado em vários lugares (load_instruction, refresh do panel,
        etc.) e antes redesenhava o KDE de 5M+ pares toda vez.
        """
        if self._plot_widget is None:
            return
        from .result_widgets import HubnessReportWidget, HubnessAblationWidget

        # Argumentos derivados que afetam o plot (além do payload).
        if isinstance(self._plot_widget, ScoreDistributionWidget):
            extra_args = (self._is_calibrated_scores_output(),)
        else:
            extra_args = ()

        # Skip redraw se mesmo payload + mesmos args.
        payload_id = id(payload)
        if (
            self._last_rendered_payload_id == payload_id
            and self._last_render_args == extra_args
            and self._plot_rendered_once
        ):
            # Garante apenas que a página correta está visível.
            if hasattr(self, "_plot_stack"):
                self._btn_hydrate.setVisible(False)
                self._plot_stack.setCurrentIndex(1)
            return

        # Cursor de espera durante o render — desenhar histograma+KDE de
        # milhões de pares ou popular as 4 abas do ValidationResultWidget
        # bloqueia a main thread por segundos. Sem isso o usuário vê a UI
        # "travada" sem feedback nenhum após clicar.
        with _with_wait_cursor():
            try:
                if isinstance(self._plot_widget, ScoreDistributionWidget):
                    use_log_lr = extra_args[0]
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
                else:
                    # KNN_TYPICALITY_MODEL_TRAINING: widget aceita o modelo
                    # treinado direto via set_model.
                    from .result_widgets.typicality_distribution_widget import (
                        TypicalityDistributionWidget,
                    )
                    if isinstance(self._plot_widget, TypicalityDistributionWidget):
                        self._plot_widget.set_model(payload)
            except Exception:
                self._show_placeholder()
                return
            # Marca como renderizado e cacheia identidade do payload.
            self._last_rendered_payload_id = payload_id
            self._last_render_args = extra_args
            self._plot_rendered_once = True
            if hasattr(self, "_plot_stack"):
                self._btn_hydrate.setVisible(False)
                self._plot_stack.setCurrentIndex(1)

    def _uses_png_snapshot_inline(self) -> bool:
        """True se a instrução deve renderizar o snapshot PNG inline em
        vez do widget vivo. Aplica-se aos tipos cujo executor já gera o
        PNG durante a persistência: SIMILARITY_SCORING e PAIR_MINING
        (produzem `SIMILARITY_SCORES`); SCORE_CALIBRATION (produz
        `CALIBRATED_SCORES`, plot usa log-LR no eixo X)."""
        try:
            return self._instruction.instruction_type in (
                StrategyInstructionType.SIMILARITY_SCORING,
                StrategyInstructionType.PAIR_MINING,
                StrategyInstructionType.SCORE_CALIBRATION,
            )
        except Exception:
            return False

    def _resolve_png_path(
        self,
        definition: LRStrategyDefinition,
        resource_id: str,
        resource_loader: Any,
    ):
        """Path absoluto do PNG snapshot deste output, ou None se faltam
        dados (project_root indisponível, etc.). Não verifica existência.

        Dispatch pelo `_plot_kind`: score distribution usa
        `score_distribution_png_path`; casework usa
        `casework_visualization_png_path`. Mesma convenção `<project>/
        graphics/<strategy_id>__<resource_id>_<sufixo>.png`, com sufixo
        distinto por tipo (`_dist` vs `_casework`)."""
        if resource_loader is None or definition is None or not resource_id:
            return None
        storage = getattr(resource_loader, "_storage", None)
        project_root = getattr(storage, "project_root", None) if storage else None
        strategy_id = getattr(definition, "strategy_id", None)
        if project_root is None or not strategy_id:
            return None
        if self._plot_kind == _PLOT_KIND_CASEWORK_VISUALIZATION:
            from faceaudit.models.casework_visualization_plot import (
                casework_visualization_png_path,
            )
            return casework_visualization_png_path(
                project_root, strategy_id, resource_id,
            )
        # Score distribution: subfolder muda se for output de
        # SCORE_CALIBRATION (CALIBRATED_SCORES) — log-LR no eixo X.
        is_calibrated = False
        try:
            is_calibrated = (
                self._instruction.instruction_type
                == StrategyInstructionType.SCORE_CALIBRATION
            )
        except Exception:
            pass
        from faceaudit.models.similarity.distribution_plot import (
            score_distribution_png_path,
        )
        return score_distribution_png_path(
            project_root, strategy_id, resource_id,
            calibrated=is_calibrated,
        )

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
        if self._plot_kind == _PLOT_KIND_TYPICALITY_DISTRIBUTION:
            return (StrategyResourceType.TRAINED_TYPICALITY_MODEL,)
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
        view_typicality_distribution_requested(StrategyInstructionDefinition)
        hydrate_requested(StrategyInstructionDefinition, str): instrução +
            resource_id solicitado para carregamento
    """

    view_trajectory_requested = QtCore.pyqtSignal(object)
    view_score_distribution_requested = QtCore.pyqtSignal(object)
    view_validation_result_requested = QtCore.pyqtSignal(object)
    view_casework_visualization_requested = QtCore.pyqtSignal(object)
    view_hubness_report_requested = QtCore.pyqtSignal(object)
    view_hubness_ablation_requested = QtCore.pyqtSignal(object)
    view_typicality_distribution_requested = QtCore.pyqtSignal(object)
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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Cabeçalho principal da coluna — padrão unificado com Instruções
        # e Configurações.
        layout.addWidget(ColumnHeader("Saídas"))

        self._stack = QtWidgets.QStackedWidget()
        layout.addWidget(self._stack, 1)

        # ── Index 0: empty state ──────────────────────────────────────────────
        # Centralizado vertical e horizontalmente — alinha com o comportamento
        # do empty state da coluna 2 (InstructionConfigPanel) que usa
        # AlignCenter no layout.
        empty_page = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_page)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_lbl = QtWidgets.QLabel("Selecione uma instrução para ver os resultados.")
        empty_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_lbl.setStyleSheet("color: #888; font-style: italic; font-size: 13px;")
        empty_lbl.setWordWrap(True)
        empty_layout.addWidget(empty_lbl)
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
        card.view_typicality_distribution_requested.connect(
            self.view_typicality_distribution_requested.emit
        )
        # Hydrate com instrução + resource_id (parent enriquece).
        card.hydrate_requested.connect(
            lambda rid, instr=instruction: self.hydrate_requested.emit(instr, rid)
        )
        return card
