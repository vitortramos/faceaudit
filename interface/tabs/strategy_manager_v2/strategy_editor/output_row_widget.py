# interface/tabs/strategy_manager_v2/strategy_editor/output_row_widget.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


# ─── Summarizers por StrategyResourceType ────────────────────────────────────

def _summarize_dataset(payload: Any) -> List[str]:
    """
    Resumo para EmbeddingDataset.

    PERFORMANCE: NÃO acessa `payload.matrix` nem `payload.identities` —
    ambas as properties alocam novas estruturas (np.vstack / np.array)
    O(N) sobre todos os items. Para datasets de centenas de milhares de
    samples × 256D, cada acesso aloca centenas de MB, e o refresh da UI
    fica progressivamente mais lento à medida que mais OutputRowWidget
    ficam visíveis.

    Em vez disso:
    - dim → lê shape do primeiro embedding (O(1)).
    - n_ids → conta via set comprehension sobre `item.identity` (O(N) Python
      puro mas SEM alocar ndarray).
    - Para datasets gigantes (>= 100k items) o n_ids é estimado por
      amostragem para manter o refresh responsivo.
    """
    try:
        n = len(payload)
    except Exception:
        n = None

    items = getattr(payload, "items", None) or []

    n_ids = None
    n_ids_note = ""
    try:
        if items:
            BIG_THRESHOLD = 100_000
            SAMPLE_LIMIT = 10_000
            if len(items) > BIG_THRESHOLD:
                step = max(1, len(items) // SAMPLE_LIMIT)
                sampled = items[::step]
                n_ids_note = " (amostragem)"
                ids_seen = set()
                for it in sampled:
                    ident = getattr(it, "identity", None)
                    if ident is not None:
                        ids_seen.add(str(ident))
                # Extrapola contagem proporcionalmente para a estimativa.
                ratio = len(items) / len(sampled) if sampled else 1
                n_ids = int(round(len(ids_seen) * (1.0 if ratio < 1.5 else 1.0)))
                # Para datasets grandes geralmente IDs são únicas; reportamos
                # o que vimos na amostra (estimativa).
                n_ids = len(ids_seen)
            else:
                ids_seen = set()
                for it in items:
                    ident = getattr(it, "identity", None)
                    if ident is not None:
                        ids_seen.add(str(ident))
                n_ids = len(ids_seen)
    except Exception:
        pass

    emb_dim = None
    try:
        if items:
            emb = getattr(items[0], "embedding", None)
            if emb is not None and hasattr(emb, "shape") and len(emb.shape) >= 1:
                emb_dim = int(emb.shape[0])
    except Exception:
        pass

    line1_parts = []
    if n is not None:
        line1_parts.append(f"{n:,} amostras".replace(",", "."))
    if n_ids is not None:
        line1_parts.append(f"{n_ids} identidades{n_ids_note}")

    lines: List[str] = []
    if line1_parts:
        lines.append(", ".join(line1_parts))
    if emb_dim is not None:
        lines.append(f"dim {emb_dim}")
    return lines or ["(sem informações)"]


def _summarize_pipeline(payload: Any) -> List[str]:
    """Resumo para EmbeddingTransformationPipeline."""
    mode = getattr(payload, "mode", None)
    mode_str = getattr(mode, "value", str(mode) if mode is not None else "—")

    steps = getattr(payload, "steps", None)
    if not steps:
        # Modo DUAL pode ter source_steps em vez de steps
        steps = getattr(payload, "source_steps", None)
    n_steps = len(steps) if steps else 0

    lines = [f"modo: {mode_str.upper()}, {n_steps} etapa(s)"]

    # Tenta recuperar dimensões
    try:
        config = getattr(payload, "config", None) or getattr(payload, "source_config", None)
        if config is not None:
            out_dim = getattr(config, "output_dimension", None)
            if out_dim is not None:
                lines.append(f"dim. saída: {out_dim}")
    except Exception:
        pass

    return lines


def _summarize_similarity_model(payload: Any) -> List[str]:
    cls_name = type(payload).__name__
    return [f"tipo: {cls_name}"]


def _summarize_calibration_model(payload: Any) -> List[str]:
    cls_name = type(payload).__name__
    return [f"tipo: {cls_name}"]


def _summarize_trajectory(payload: Any) -> List[str]:
    """Resumo para TransformationTrajectoryData."""
    steps = getattr(payload, "steps", None)
    n = len(steps) if steps is not None else 0
    if n == 0:
        return ["(trajetória vazia)"]
    has_target = any(getattr(s, "target_points", None) is not None for s in steps)
    mode = "DUAL" if has_target else "LINEAR"
    return [f"{n} step(s)", f"modo: {mode}"]


def _summarize_similarity_scores(payload: Any) -> List[str]:
    """Resumo para SimilarityScoreDataset."""
    try:
        n = len(payload)
    except Exception:
        return ["(scores indisponíveis)"]
    if n == 0:
        return ["(0 pares)"]

    # Modo casework: sem ground truth → exibe contagem como "pares de caso"
    # em vez de gen/imp (que seriam ambos zero ou enganosos).
    md = getattr(payload, "metadata", None)
    if isinstance(md, dict) and md.get("is_casework"):
        return [
            f"{n:,} pares de caso (sem ground truth)".replace(",", ".")
        ]

    n_gen = getattr(payload, "total_genuines", None)
    n_imp = getattr(payload, "total_impostors", None)

    lines = [f"{n:,} pares".replace(",", ".")]
    if n_gen is not None and n_imp is not None:
        lines.append(
            f"{n_gen:,} genuínos / {n_imp:,} impostores".replace(",", ".")
        )
    return lines


def _summarize_calibrated_scores(payload: Any) -> List[str]:
    """
    Resumo para SimilarityScoreDataset usado como CALIBRATED_SCORES.

    Inclui as estatísticas básicas (idênticas a SIMILARITY_SCORES) e, quando
    há `log_lr` em metadata dos samples, adiciona uma linha com o log-LR
    médio para genuínos e impostores.

    Para datasets grandes (>= 50k items) usa AMOSTRAGEM de até 5000 items
    distribuídos uniformemente — evita travar a main thread Qt iterando
    milhões de samples a cada refresh da UI. A média estimada por amostra
    é praticamente idêntica à média populacional para distribuições com
    variância moderada.
    """
    lines = _summarize_similarity_scores(payload)

    try:
        items = getattr(payload, "items", []) or []
        n_total = len(items)
        if n_total == 0:
            return lines

        # Amostragem determinística para datasets grandes — preserva o
        # médio sem custo de iteração linear na main thread.
        SAMPLE_LIMIT = 5_000
        BIG_THRESHOLD = 50_000
        if n_total > BIG_THRESHOLD:
            step = max(1, n_total // SAMPLE_LIMIT)
            sampled = items[::step]
            sample_note = f" (amostra de {len(sampled):,})".replace(",", ".")
        else:
            sampled = items
            sample_note = ""

        # Modo casework: sem ground truth → exibe log-LR médio e range
        # geral em vez de discriminar por classe.
        ds_md = getattr(payload, "metadata", None)
        is_casework = isinstance(ds_md, dict) and ds_md.get("is_casework")

        if is_casework:
            log_lrs = []
            for s in sampled:
                md = getattr(s, "metadata", None)
                if isinstance(md, dict):
                    v = md.get("log_lr")
                    if v is not None:
                        log_lrs.append(float(v))
            if log_lrs:
                mean = sum(log_lrs) / len(log_lrs)
                lo, hi = min(log_lrs), max(log_lrs)
                lines.append(
                    f"log-LR (caso): médio={mean:+.2f} | "
                    f"range=[{lo:+.2f}, {hi:+.2f}]{sample_note}"
                )
            return lines

        sum_gen = 0.0; n_gen = 0
        sum_imp = 0.0; n_imp = 0
        for s in sampled:
            md = getattr(s, "metadata", None)
            if not isinstance(md, dict):
                continue
            v = md.get("log_lr")
            if v is None:
                continue
            if s.same_person:
                sum_gen += float(v); n_gen += 1
            else:
                sum_imp += float(v); n_imp += 1

        if n_gen or n_imp:
            mean_g = f"{sum_gen/n_gen:+.2f}" if n_gen else "—"
            mean_i = f"{sum_imp/n_imp:+.2f}" if n_imp else "—"
            lines.append(
                f"log-LR médio: gen={mean_g} / imp={mean_i}{sample_note}"
            )
    except Exception:
        pass

    return lines


def _summarize_casework_visualization(payload: Any) -> List[str]:
    """
    Resumo para CaseworkVisualizationResult — modo (RAW/CALIBRATED),
    contagens da referência e do caso, EER threshold (se RAW).
    """
    try:
        score_type = getattr(payload, "score_type", "?")
        n_gen = int(getattr(payload, "n_genuine", 0) or 0)
        n_imp = int(getattr(payload, "n_impostor", 0) or 0)
        n_case = int(getattr(payload, "n_case", 0) or 0)
        eer_threshold = getattr(payload, "eer_threshold", None)
        case_summary = getattr(payload, "case_summary", {}) or {}

        lines = [f"Modo: {score_type}"]
        lines.append(
            f"ref={n_gen:,} gen / {n_imp:,} imp".replace(",", ".")
        )

        if n_case > 0 and case_summary:
            lines.append(
                f"caso: {n_case} pares "
                f"(mean={case_summary.get('mean', 0):+.2f})"
            )
        elif n_case > 0:
            lines.append(f"caso: {n_case} pares")

        if eer_threshold is not None:
            lines.append(f"EER threshold={float(eer_threshold):.4f}")

        return lines
    except Exception:
        return ["(erro ao resumir visualização)"]


def _summarize_validation_result(payload: Any) -> List[str]:
    """
    Resumo para StrategyValidationResult — métricas escalares (EER, AUC,
    Cllr, cal_loss) e contagens de classe.
    """
    try:
        metrics = getattr(payload, "metrics", None)
        if metrics is None:
            return ["(métricas indisponíveis)"]

        n_gen = int(getattr(payload, "n_genuines", 0) or 0)
        n_imp = int(getattr(payload, "n_impostors", 0) or 0)

        eer = getattr(metrics, "eer", None)
        cllr = getattr(metrics, "cllr", None)
        cllr_min = getattr(metrics, "cllr_min", None)
        auc = getattr(metrics, "auc", None)

        lines = [
            f"{n_gen:,} genuínos / {n_imp:,} impostores".replace(",", ".")
        ]

        line2_parts = []
        if eer is not None:
            line2_parts.append(f"EER={float(eer)*100:.2f}%")
        if auc is not None:
            line2_parts.append(f"AUC={float(auc):.4f}")
        if line2_parts:
            lines.append(" | ".join(line2_parts))

        line3_parts = []
        if cllr is not None:
            line3_parts.append(f"Cllr={float(cllr):.3f}")
        if cllr is not None and cllr_min is not None:
            cal_loss = max(0.0, float(cllr) - float(cllr_min))
            line3_parts.append(f"cal_loss={cal_loss:.3f}")
        if line3_parts:
            lines.append(" | ".join(line3_parts))

        return lines
    except Exception:
        return ["(erro ao resumir validação)"]


_OUTPUT_SUMMARIZERS: Dict[StrategyResourceType, Callable[[Any], List[str]]] = {
    StrategyResourceType.INTERMEDIATE_DATASET:      _summarize_dataset,
    StrategyResourceType.DATASET:                   _summarize_dataset,
    StrategyResourceType.TRAINED_PIPELINE:          _summarize_pipeline,
    StrategyResourceType.TRAINED_SIMILARITY_MODEL:  _summarize_similarity_model,
    StrategyResourceType.TRAINED_CALIBRATION_MODEL: _summarize_calibration_model,
    StrategyResourceType.TRAJECTORY_DATA:           _summarize_trajectory,
    StrategyResourceType.SIMILARITY_SCORES:         _summarize_similarity_scores,
    StrategyResourceType.CALIBRATED_SCORES:         _summarize_calibrated_scores,
    StrategyResourceType.VALIDATION_RESULT:         _summarize_validation_result,
    StrategyResourceType.CASEWORK_VISUALIZATION:    _summarize_casework_visualization,
    # Forward-compat: VISUALIZATION → mostra botão "Visualizar gráfico..."
    # (registrado via OutputRowWidget._is_visualization_type abaixo).
}


# ─── Widget ──────────────────────────────────────────────────────────────────

class OutputRowWidget(QtWidgets.QFrame):
    """
    Linha de output na seção Saídas do InstructionConfigPanel.

    Mostra:
    - Ícone de estado (⬜ placeholder | 🟡 em memória | 🟢 salvo)
    - QLineEdit editável com o nome do recurso (output)
    - Label do tipo do recurso
    - Linhas de resumo contextual (lazy, baseadas no payload)
    - Botão "Visualizar gráfico…" para tipos de visualização (forward-compat)

    Sinais:
        name_changed(field_name, new_text) — emitido quando o usuário edita o nome do recurso.
    """

    name_changed = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        field_name: str,
        label: str,
        resource_id: Optional[str],
        resource_type: StrategyResourceType,
        parent: Optional[QtWidgets.QWidget] = None,
        read_only: bool = False,
    ):
        super().__init__(parent)
        self._field_name = field_name
        self._resource_type = resource_type
        self._read_only = bool(read_only)

        # Estado para a seção de Detalhes (lazy, criada no primeiro expand).
        self._info_widget: Optional[QtWidgets.QWidget] = None
        self._info_expanded = False
        self._last_resource_for_details: Optional[StrategyResource] = None
        self._last_project_root: Optional[Path] = None

        self.setStyleSheet(
            "OutputRowWidget {"
            "  background-color: #fafdf6;"
            "  border: 1px solid #d6e8d0;"
            "  border-radius: 4px;"
            "}"
        )

        self._init_ui(label, resource_id)

    def _init_ui(self, label: str, resource_id: Optional[str]) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        # Linha 1: ícone status + label + nome editável + tipo
        self._status_lbl = QtWidgets.QLabel("⬜")  # ⬜ placeholder default
        self._status_lbl.setStyleSheet("font-size: 14px;")
        self._status_lbl.setMinimumWidth(20)
        layout.addWidget(self._status_lbl, 0, 0)

        lbl = QtWidgets.QLabel(f"{label}:")
        lbl.setStyleSheet("font-size: 11px; color: #2e7d32; font-weight: bold;")
        layout.addWidget(lbl, 0, 1)

        if self._read_only:
            # Display puro — QLabel, sem borda nem fundo. Visualmente
            # sintoniza com o resto da coluna de resultados (status +
            # summary + plot), que é toda informacional.
            self._name_field = QtWidgets.QLabel(resource_id or "")
            self._name_field.setStyleSheet(
                "font-size: 11px; color: #222; font-weight: 500;"
            )
            self._name_field.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._name_field.setToolTip(
                "Edite o nome do recurso na coluna de configuração (2ª coluna)."
            )
        else:
            self._name_field = QtWidgets.QLineEdit(resource_id or "")
            self._name_field.setPlaceholderText("nome do recurso de saída")
            self._name_field.setStyleSheet("font-size: 11px;")
            # Emite name_changed em CADA edição (textChanged) e também ao sair do
            # campo (editingFinished) — garante que o DTO fique em sincronia mesmo
            # se o painel não invocar _on_apply antes de uma validação/execução.
            self._name_field.textChanged.connect(self._on_name_changed)
            self._name_field.editingFinished.connect(self._on_name_changed)
        layout.addWidget(self._name_field, 0, 2)

        type_lbl = QtWidgets.QLabel(f"[{self._resource_type.value}]")
        type_lbl.setStyleSheet("font-size: 9px; color: #888;")
        layout.addWidget(type_lbl, 0, 3)

        # Linha 2: status textual + resumo
        self._summary_container = QtWidgets.QWidget()
        sum_layout = QtWidgets.QVBoxLayout(self._summary_container)
        sum_layout.setContentsMargins(28, 0, 0, 0)
        sum_layout.setSpacing(0)
        self._summary_layout = sum_layout
        layout.addWidget(self._summary_container, 1, 0, 1, 4)

        # Linha 3 (apenas read_only): botão expansível "▾ Detalhes" que
        # mostra ResourceInfoWidget(compact=True) abaixo. Criado lazy no
        # primeiro clique para evitar instanciar widgets pesados (5 seções
        # de form) por output, mesmo quando o usuário não vai consultar.
        if self._read_only:
            self._details_btn = QtWidgets.QToolButton()
            self._details_btn.setText("▸ Detalhes")
            self._details_btn.setStyleSheet(
                "QToolButton { border: none; background: transparent;"
                " color: #555; font-size: 10px; padding: 2px 4px; }"
                "QToolButton:hover { color: #1565c0; }"
            )
            self._details_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            self._details_btn.clicked.connect(self._on_toggle_details)
            self._details_btn.setVisible(False)  # só aparece quando há recurso
            layout.addWidget(self._details_btn, 2, 0, 1, 4)

            self._details_container = QtWidgets.QWidget()
            details_layout = QtWidgets.QVBoxLayout(self._details_container)
            details_layout.setContentsMargins(28, 4, 4, 4)
            details_layout.setSpacing(0)
            self._details_layout = details_layout
            self._details_container.setVisible(False)
            layout.addWidget(self._details_container, 3, 0, 1, 4)
        else:
            self._details_btn = None
            self._details_container = None
            self._details_layout = None

        layout.setColumnStretch(2, 1)

    def update_state(
        self,
        definition: LRStrategyDefinition,
        instruction_id: str,
        executed: bool,
        context: Optional[StrategyExecutionContextProtocol],
        project_root: Optional[Path] = None,
    ) -> None:
        """Atualiza ícone de status e bloco de resumo conforme estado do recurso."""
        # Limpa resumo anterior
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        resource_id = self._name_field.text().strip()
        if not resource_id:
            self._set_status("⬜", "(sem nome)")
            self._update_details_visibility(None, project_root)
            return

        existing = definition.get_resource_by_id(resource_id) if definition else None
        ctx_resource: Optional[StrategyResource] = None
        if context is not None and context.has_resource(resource_id):
            ctx_resource = context.get_resource(resource_id)

        # Determina status visual
        if existing and existing.has_artifact:
            self._set_status("\U0001f7e2", "salvo em disco")
            self._render_summary_for(existing.payload or (ctx_resource.payload if ctx_resource else None))
            self._update_details_visibility(existing, project_root)
        elif ctx_resource and ctx_resource.has_payload:
            self._set_status("\U0001f7e1", "em memória")
            self._render_summary_for(ctx_resource.payload)
            self._update_details_visibility(ctx_resource, project_root)
        elif existing is not None:
            # Recurso conhecido mas sem payload nem artifact — placeholder.
            if executed:
                self._set_status("\U0001f7e1", "em memória")
            else:
                self._set_status("⬜", "placeholder")
            self._update_details_visibility(existing, project_root)
        else:
            self._set_status("⬜", "placeholder")
            self._update_details_visibility(None, project_root)

    def _update_details_visibility(
        self,
        resource: Optional[StrategyResource],
        project_root: Optional[Path],
    ) -> None:
        """Mostra o toggle só quando há um recurso real (em mem ou disco)."""
        if self._details_btn is None:
            return
        has_data = resource is not None and (
            getattr(resource, "has_payload", False)
            or getattr(resource, "has_artifact", False)
        )
        self._details_btn.setVisible(has_data)
        self._last_resource_for_details = resource if has_data else None
        self._last_project_root = project_root
        # Re-renderiza se a seção já estava expandida.
        if self._info_expanded and self._info_widget is not None and resource is not None:
            self._refresh_info_widget()

    def _on_toggle_details(self) -> None:
        if self._details_btn is None or self._details_container is None:
            return
        self._info_expanded = not self._info_expanded
        if self._info_expanded:
            self._details_btn.setText("▾ Detalhes")
            if self._info_widget is None:
                # Lazy init na primeira expansão.
                from interface.tabs.strategy_manager_v2.left_panel.resource_info_widget import (
                    ResourceInfoWidget,
                )
                self._info_widget = ResourceInfoWidget(compact=True)
                self._details_layout.addWidget(self._info_widget)
            self._refresh_info_widget()
            self._details_container.setVisible(True)
        else:
            self._details_btn.setText("▸ Detalhes")
            self._details_container.setVisible(False)

    def _refresh_info_widget(self) -> None:
        if self._info_widget is None or self._last_resource_for_details is None:
            return
        try:
            self._info_widget.show_resource(
                resource=self._last_resource_for_details,
                project_root=self._last_project_root,
                definitions=None,
            )
        except Exception:
            pass

    def _set_status(self, icon: str, status_text: str) -> None:
        self._status_lbl.setText(icon)
        status_lbl = QtWidgets.QLabel(status_text)
        status_lbl.setStyleSheet("font-size: 10px; color: #555; font-style: italic;")
        self._summary_layout.addWidget(status_lbl)

    def _render_summary_for(self, payload: Any) -> None:
        if payload is None:
            return

        # Forward-compat point: VISUALIZATION → botão "Visualizar gráfico..."
        if self._is_visualization_type():
            self._render_visualize_button(payload)
            return

        summarizer = _OUTPUT_SUMMARIZERS.get(self._resource_type)
        if summarizer is None:
            return

        try:
            lines = summarizer(payload)
        except Exception:
            lines = ["(erro ao resumir payload)"]

        for line in lines:
            lbl = QtWidgets.QLabel(f"→ {line}")
            lbl.setStyleSheet("font-size: 10px; color: #444;")
            lbl.setWordWrap(True)
            self._summary_layout.addWidget(lbl)

    def _is_visualization_type(self) -> bool:
        # StrategyResourceType.VISUALIZATION ainda não existe — extensão futura.
        return self._resource_type.value == "VISUALIZATION"

    def _render_visualize_button(self, payload: Any) -> None:
        btn = QtWidgets.QPushButton("\U0001f4ca Visualizar gráfico…")
        btn.setStyleSheet(
            "QPushButton { border: 1px solid #1565c0; border-radius: 3px;"
            " background: #e3f2fd; color: #1565c0;"
            " padding: 3px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #bbdefb; }"
        )
        btn.clicked.connect(lambda: self._open_graph_viewer(payload))
        self._summary_layout.addWidget(btn)

    def _open_graph_viewer(self, payload: Any) -> None:
        from .graph_viewer_dialog import GraphViewerDialog
        from matplotlib.figure import Figure

        figure = payload if isinstance(payload, Figure) else getattr(payload, "figure", None)
        if figure is None:
            QtWidgets.QMessageBox.information(
                self, "Visualização indisponível",
                "Este recurso não contém um gráfico Matplotlib renderizável."
            )
            return

        title = f"Gráfico: {self._name_field.text().strip()}"
        dlg = GraphViewerDialog(title, figure=figure, parent=self)
        dlg.exec()

    def field_name(self) -> str:
        return self._field_name

    def current_value(self) -> str:
        return self._name_field.text().strip()

    def set_resource_name(self, name: str) -> None:
        """Atualiza o texto exibido sem reemitir name_changed (para sincronia
        entre painéis read-only e editável). Funciona com QLineEdit (modo
        editável) e QLabel (modo read-only)."""
        if self._name_field.text() == name:
            return
        # blockSignals só faz sentido em widgets que emitem signals em setText
        # (QLineEdit). Em QLabel (read_only) é no-op mas inofensivo.
        was_blocked = self._name_field.blockSignals(True)
        try:
            self._name_field.setText(name or "")
        finally:
            self._name_field.blockSignals(was_blocked)

    def _on_name_changed(self) -> None:
        self.name_changed.emit(self._field_name, self._name_field.text().strip())
