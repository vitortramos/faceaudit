# interface/tabs/strategy_manager_v2/strategy_editor/add_validation_rule_dialog.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig
from faceaudit.models.validation.validation_rule_template import (
    ValidationRuleTemplate,
)


# =============================================================================
# SCHEMA — fields editáveis por tipo de regra
# =============================================================================

@dataclass(frozen=True)
class _RuleParamSpec:
    """
    Descrição de UM parâmetro editável de uma regra de validação.

    Atributos:
        key: chave usada no dict `params` do `ValidationRuleConfig`.
        label: rótulo legível no form.
        kind: "resource" | "resource_multi" | "int" | "float" | "enum_choice".
        accepted_types: para kind=resource/resource_multi, tipos aceitos.
        default: valor padrão quando o usuário não tocou no campo.
        min_value/max_value: para kind=int/float.
        required: se True, o botão "Adicionar" fica desabilitado enquanto
                  vazio.
        choices: para kind=enum_choice, lista de strings selecionáveis.
        help: tooltip do widget.
    """

    key: str
    label: str
    kind: str
    accepted_types: Optional[Tuple[StrategyResourceType, ...]] = None
    default: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = True
    choices: Optional[Tuple[str, ...]] = None
    help: Optional[str] = None


# Tipos persistentes-configuráveis exposíveis no diálogo (8 originais + 3 das
# regras de cobertura mínima da Phase 4 que aceitam threshold tunável).
# Os outros 8 tipos da Phase 4 são intrínsecos-dinâmicos (MODEL_FITTED,
# EMBEDDING_DIM_MATCHES_MODEL, SCORE_CALIBRATION_STATE, CASEWORK_STATE,
# SCORE_TYPE_MATCH, DATASETS_EMBEDDING_COMPAT, DUAL_PIPELINE_*) — usuário
# não adiciona manualmente; instruções os constroem em `build_validation_rules`.
_RULE_PARAM_SCHEMAS: Dict[ValidationRuleType, Tuple[_RuleParamSpec, ...]] = {
    ValidationRuleType.FISHER: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset de treino", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "safety_margin_ratio", "Margem de segurança", "float",
            default=0.2, min_value=0.0, max_value=1.0, required=False,
            help="Fração extra exigida acima de embedding_dim.",
        ),
    ),
    ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset de treino", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "min_required", "Mínimo de identidades não-singleton", "int",
            default=2, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "safety_threshold", "Limiar de aviso", "int",
            default=10, min_value=1, required=False,
            help="Abaixo deste valor a regra emite WARNING.",
        ),
    ),
    ValidationRuleType.MIN_CALIBRATION_PAIRS: (
        _RuleParamSpec(
            "scores_resource_id", "Score dataset", "resource",
            accepted_types=(
                StrategyResourceType.SIMILARITY_SCORES,
                StrategyResourceType.CALIBRATED_SCORES,
            ),
        ),
        _RuleParamSpec(
            "min_genuines", "Mínimo de genuínos", "int",
            default=100, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "min_impostors", "Mínimo de impostores", "int",
            default=500, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "warn_ratio_threshold", "Razão crítica gen/imp", "float",
            default=0.02, min_value=0.0, required=False,
        ),
    ),
    ValidationRuleType.DATASET_ISOLATION: (
        _RuleParamSpec(
            "dataset_a_resource_id", "Dataset A", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "dataset_b_resource_id", "Dataset B", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
    ),
    ValidationRuleType.DATASET_OVERLAP_REQUIRED: (
        _RuleParamSpec(
            "dataset_a_resource_id", "Dataset A", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "dataset_b_resource_id", "Dataset B", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "min_overlap", "Mínimo de identidades em comum", "int",
            default=1, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "warn_below_ratio", "Razão de aviso", "float",
            default=0.1, min_value=0.0, max_value=1.0, required=False,
        ),
    ),
    ValidationRuleType.HOLDOUT_LEAKAGE: (
        _RuleParamSpec(
            "holdout_resource_ids", "Datasets de holdout", "resource_multi",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "train_resource_ids", "Datasets de treino/calibração",
            "resource_multi",
            accepted_types=(
                StrategyResourceType.DATASET,
                StrategyResourceType.SIMILARITY_SCORES,
            ),
        ),
    ),
    ValidationRuleType.MIN_IDENTITIES: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "min_required", "Mínimo de identidades", "int",
            default=2, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "safety_threshold", "Limiar de aviso", "int",
            default=5, min_value=1, required=False,
        ),
    ),
    ValidationRuleType.MIN_SAMPLES: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "min_required", "Mínimo de amostras", "int",
            default=2, min_value=1, required=False,
        ),
        _RuleParamSpec(
            "safety_threshold", "Limiar de aviso", "int",
            default=100, min_value=1, required=False,
        ),
    ),
    ValidationRuleType.QUALITY_METRIC_AVAILABLE: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "quality_metric", "Métrica de qualidade", "enum_choice",
            choices=("fiqa_score", "ofiq_score", "serfiq_score", "embedding_norm"),
            default="fiqa_score",
        ),
        _RuleParamSpec(
            "min_coverage", "Cobertura mínima (0–1)", "float",
            default=0.5, min_value=0.0, max_value=1.0, required=False,
        ),
    ),
    ValidationRuleType.HAS_POSE_METADATA: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
    ),
    ValidationRuleType.MAX_SAMPLES_WARNING: (
        _RuleParamSpec(
            "dataset_resource_id", "Dataset", "resource",
            accepted_types=(StrategyResourceType.DATASET,),
        ),
        _RuleParamSpec(
            "max_samples", "Threshold de aviso", "int",
            default=100_000, min_value=1, required=False,
        ),
    ),
}


# =============================================================================
# DIÁLOGO
# =============================================================================


class AddValidationRuleDialog(QtWidgets.QDialog):
    """
    Modal pra adicionar uma nova regra de validação a uma instrução.

    Dois fluxos no mesmo combo:
    1. **Tipo de regra**: usuário pica um dos 11 tipos configuráveis.
       Form dinâmico aparece — preenche params, clica Adicionar.
    2. **Template salvo**: usuário pica um template salvo (Global library
       ou projeto). Pré-visualiza os params, clica Adicionar — config
       é embed na instrução.

    Combo organizado em dois grupos visuais (separator label):
        ── Tipos de regra ──
        FISHER, MIN_IDENTITIES, ...
        ── Templates salvos ──
        my_template_1 (FISHER), my_template_2 (MIN_CALIBRATION_PAIRS), ...

    Não bloqueia duplicatas (ex: 2 FISHER no mesmo recurso) — pode ser
    intencional (ex: FISHER + um threshold diferente). Usuário tem que
    saber o que está fazendo.
    """

    def __init__(
        self,
        available_resources: List[StrategyResource],
        templates: List[ValidationRuleTemplate],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adicionar Regra de Validação")
        self.setModal(True)
        self.resize(560, 460)

        self._resources = list(available_resources or [])
        self._templates = list(templates or [])
        # Caches: mapa de widget→spec pra leitura, lista de widgets pra
        # `_recompute_ok_enabled`. Resetados a cada `_rebuild_param_form`.
        self._widget_map: Dict[str, Tuple[_RuleParamSpec, QtWidgets.QWidget]] = {}
        self._result: Optional[ValidationRuleConfig] = None

        self._build_ui()
        self._populate_combo()
        self._on_combo_changed()  # Setup inicial da label/form

    # ── API pública ───────────────────────────────────────────────────────

    def get_config(self) -> Optional[ValidationRuleConfig]:
        """Retorna o config aceito (ou None se cancelado)."""
        return self._result

    # ── Construção da UI ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        # ── Header / combo principal ──────────────────────────────────────
        header_lbl = QtWidgets.QLabel("Tipo de regra ou template:", parent=self)
        header_lbl.setStyleSheet("font-size: 11px; font-weight: bold;")
        root.addWidget(header_lbl)

        self.combo = QtWidgets.QComboBox(parent=self)
        self.combo.setStyleSheet("font-size: 11px;")
        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        root.addWidget(self.combo)

        # Descrição do tipo selecionado.
        self._desc_box = QtWidgets.QLabel(parent=self)
        self._desc_box.setWordWrap(True)
        self._desc_box.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._desc_box.setStyleSheet(
            "QLabel { background: #FDF6E6; border: 1px solid #F3E0B0;"
            " border-radius: 3px; padding: 8px; color: #5D4500;"
            " font-size: 11px; }"
        )
        root.addWidget(self._desc_box)

        # ── Param form ────────────────────────────────────────────────────
        params_group = QtWidgets.QGroupBox("Parâmetros", parent=self)
        params_group.setStyleSheet(
            "QGroupBox { font-size: 11px; font-weight: bold;"
            " border: 1px solid #ccc; border-radius: 4px; margin-top: 8px;"
            " padding-top: 12px; }"
            "QGroupBox::title { left: 8px; subcontrol-position: top left; }"
        )
        self._form_layout = QtWidgets.QFormLayout(params_group)
        self._form_layout.setContentsMargins(10, 10, 10, 10)
        self._form_layout.setSpacing(6)
        root.addWidget(params_group, 1)

        # ── Bottom buttons ────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        self.btn_cancel = QtWidgets.QPushButton("Cancelar", parent=self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_ok = QtWidgets.QPushButton("Adicionar", parent=self)
        self.btn_ok.setStyleSheet(
            "QPushButton { background: #2e7d32; color: white;"
            " padding: 5px 18px; border-radius: 3px; font-weight: bold; }"
            "QPushButton:disabled { background: #aaa; color: #eee; }"
        )
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_ok)
        btn_row.addWidget(self.btn_ok)

        root.addLayout(btn_row)

    def _populate_combo(self) -> None:
        """
        Popula o combo com dois grupos: tipos configuráveis e templates
        salvos. `userData` distingue: para tipos é a chave canônica
        `"type:{enum.value}"`, para templates é `"tpl:{template_id}"`.
        """
        self.combo.blockSignals(True)
        self.combo.clear()

        # Grupo 1: tipos
        self.combo.addItem("── Tipos de regra ──")
        self.combo.model().item(self.combo.count() - 1).setEnabled(False)
        for rule_type in _RULE_PARAM_SCHEMAS.keys():
            self.combo.addItem(rule_type.label, userData=f"type:{rule_type.value}")

        # Grupo 2: templates (só se existirem)
        if self._templates:
            self.combo.addItem("── Templates salvos ──")
            self.combo.model().item(self.combo.count() - 1).setEnabled(False)
            for tpl in self._templates:
                label = f"{tpl.name}  ({tpl.rule_config.rule_type.label})"
                self.combo.addItem(label, userData=f"tpl:{tpl.template_id}")

        # Seleção default no primeiro tipo (índice 1, porque 0 é o
        # separator desabilitado).
        if self.combo.count() > 1:
            self.combo.setCurrentIndex(1)
        self.combo.blockSignals(False)

    # ── Reação à mudança do combo ─────────────────────────────────────────

    def _on_combo_changed(self) -> None:
        """Reage a mudança do combo — rebuilda form/descrição."""
        data = self.combo.currentData()
        if not isinstance(data, str):
            self._clear_form()
            self._desc_box.setText("(selecione um tipo ou template)")
            self._recompute_ok_enabled()
            return

        if data.startswith("type:"):
            type_value = data.split(":", 1)[1]
            rule_type = ValidationRuleType.from_str(type_value)
            self._render_type(rule_type, prefill=None)
        elif data.startswith("tpl:"):
            tpl_id = data.split(":", 1)[1]
            template = next(
                (t for t in self._templates if t.template_id == tpl_id), None,
            )
            if template is not None:
                self._render_type(
                    template.rule_config.rule_type,
                    prefill=template.rule_config.params or {},
                )

    def _render_type(
        self,
        rule_type: ValidationRuleType,
        prefill: Optional[Dict[str, Any]],
    ) -> None:
        """Configura o desc box + form pra o `rule_type` indicado.

        Quando `prefill` é fornecido (caso de template importado), pré-popula
        os widgets com os valores do template — usuário pode ajustar antes
        de aceitar.
        """
        self._desc_box.setText(
            f"<b>{rule_type.label}</b><br/>{rule_type.description}"
        )
        self._build_param_form(rule_type, prefill or {})
        self._recompute_ok_enabled()

    def _clear_form(self) -> None:
        """Remove todos os widgets do form."""
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._widget_map.clear()

    def _build_param_form(
        self,
        rule_type: ValidationRuleType,
        prefill: Dict[str, Any],
    ) -> None:
        """Constrói o form a partir do schema do tipo selecionado."""
        self._clear_form()
        self._current_rule_type = rule_type

        specs = _RULE_PARAM_SCHEMAS.get(rule_type, ())
        if not specs:
            empty = QtWidgets.QLabel(
                "(esta regra não tem parâmetros configuráveis)",
                parent=self,
            )
            empty.setStyleSheet("color: #888; font-style: italic;")
            self._form_layout.addRow(empty)
            return

        for spec in specs:
            widget = self._make_field_widget(spec, prefill.get(spec.key))
            label = QtWidgets.QLabel(
                spec.label + (" *" if spec.required else ""), parent=self,
            )
            label.setStyleSheet("font-size: 11px;")
            if spec.help:
                label.setToolTip(spec.help)
                widget.setToolTip(spec.help)
            self._form_layout.addRow(label, widget)
            self._widget_map[spec.key] = (spec, widget)

    def _make_field_widget(
        self,
        spec: _RuleParamSpec,
        prefill_value: Any,
    ) -> QtWidgets.QWidget:
        """Constrói o widget Qt apropriado pra o kind do spec."""
        if spec.kind == "resource":
            return self._make_resource_combo(spec, prefill_value)
        if spec.kind == "resource_multi":
            return self._make_resource_multi(spec, prefill_value)
        if spec.kind == "int":
            return self._make_int_spinbox(spec, prefill_value)
        if spec.kind == "float":
            return self._make_float_spinbox(spec, prefill_value)
        if spec.kind == "enum_choice":
            return self._make_enum_combo(spec, prefill_value)
        raise ValueError(f"_RuleParamSpec.kind desconhecido: '{spec.kind}'.")

    def _make_resource_combo(
        self,
        spec: _RuleParamSpec,
        prefill: Any,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent=self)
        combo.setStyleSheet("font-size: 11px;")
        combo.addItem("(selecione...)", userData=None)
        # Filtra recursos pelos accepted_types do spec.
        filtered = [
            r for r in self._resources
            if spec.accepted_types is None
            or r.resource_type in spec.accepted_types
        ]
        for r in filtered:
            combo.addItem(r.resource_id, userData=r.resource_id)

        # Pré-popula se houver valor — mesmo que o resource não exista
        # mais no contexto atual (template importado de outra estratégia),
        # ainda adicionamos como item órfão pra preservar a intenção.
        if isinstance(prefill, str) and prefill:
            idx = combo.findData(prefill)
            if idx < 0:
                combo.addItem(f"{prefill}  (indisponível)", userData=prefill)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)

        combo.currentIndexChanged.connect(self._recompute_ok_enabled)
        return combo

    def _make_resource_multi(
        self,
        spec: _RuleParamSpec,
        prefill: Any,
    ) -> QtWidgets.QListWidget:
        lst = QtWidgets.QListWidget(parent=self)
        lst.setStyleSheet("font-size: 11px;")
        lst.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        lst.setMaximumHeight(110)

        filtered = [
            r for r in self._resources
            if spec.accepted_types is None
            or r.resource_type in spec.accepted_types
        ]
        prefill_set = set(prefill) if isinstance(prefill, (list, tuple)) else set()
        for r in filtered:
            item = QtWidgets.QListWidgetItem(r.resource_id)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if r.resource_id in prefill_set
                else QtCore.Qt.CheckState.Unchecked
            )
            item.setData(QtCore.Qt.ItemDataRole.UserRole, r.resource_id)
            lst.addItem(item)

        # Items "órfãos" do prefill que não constam nos recursos disponíveis.
        existing_ids = {r.resource_id for r in filtered}
        for prefilled_id in prefill_set - existing_ids:
            item = QtWidgets.QListWidgetItem(f"{prefilled_id}  (indisponível)")
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, prefilled_id)
            lst.addItem(item)

        lst.itemChanged.connect(self._recompute_ok_enabled)
        return lst

    def _make_int_spinbox(
        self,
        spec: _RuleParamSpec,
        prefill: Any,
    ) -> QtWidgets.QSpinBox:
        sb = QtWidgets.QSpinBox(parent=self)
        sb.setStyleSheet("font-size: 11px;")
        sb.setMinimum(int(spec.min_value) if spec.min_value is not None else 0)
        sb.setMaximum(int(spec.max_value) if spec.max_value is not None else 10**9)
        value = prefill if isinstance(prefill, (int, float)) else (spec.default or 0)
        sb.setValue(int(value))
        return sb

    def _make_float_spinbox(
        self,
        spec: _RuleParamSpec,
        prefill: Any,
    ) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox(parent=self)
        sb.setStyleSheet("font-size: 11px;")
        sb.setDecimals(4)
        sb.setSingleStep(0.01)
        sb.setMinimum(float(spec.min_value) if spec.min_value is not None else -1e9)
        sb.setMaximum(float(spec.max_value) if spec.max_value is not None else 1e9)
        value = prefill if isinstance(prefill, (int, float)) else (spec.default or 0.0)
        sb.setValue(float(value))
        return sb

    def _make_enum_combo(
        self,
        spec: _RuleParamSpec,
        prefill: Any,
    ) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(parent=self)
        combo.setStyleSheet("font-size: 11px;")
        for choice in (spec.choices or ()):
            combo.addItem(choice, userData=choice)
        value = prefill if isinstance(prefill, str) else (spec.default or "")
        if value:
            idx = combo.findData(value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.currentIndexChanged.connect(self._recompute_ok_enabled)
        return combo

    # ── Recompute do estado do botão Adicionar ────────────────────────────

    def _recompute_ok_enabled(self) -> None:
        """
        Habilita "Adicionar" só quando todos os campos `required` têm valor.
        """
        if not self._widget_map:
            # Sem campos = não habilita (significa que combo está em
            # separator ou tipo sem schema).
            self.btn_ok.setEnabled(False)
            return

        for spec, widget in self._widget_map.values():
            if not spec.required:
                continue
            val = self._read_widget(spec, widget)
            if spec.kind in ("resource", "enum_choice"):
                if not val:
                    self.btn_ok.setEnabled(False)
                    return
            elif spec.kind == "resource_multi":
                if not val:
                    self.btn_ok.setEnabled(False)
                    return

        self.btn_ok.setEnabled(True)

    def _read_widget(
        self,
        spec: _RuleParamSpec,
        widget: QtWidgets.QWidget,
    ) -> Any:
        """Extrai o valor atual de um widget conforme o spec."""
        if spec.kind == "resource":
            return widget.currentData()
        if spec.kind == "resource_multi":
            ids: List[str] = []
            for i in range(widget.count()):
                item = widget.item(i)
                if item.checkState() == QtCore.Qt.CheckState.Checked:
                    ids.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
            return ids
        if spec.kind == "int":
            return int(widget.value())
        if spec.kind == "float":
            return float(widget.value())
        if spec.kind == "enum_choice":
            return widget.currentData()
        return None

    # ── Aceitar ────────────────────────────────────────────────────────────

    def _on_ok(self) -> None:
        """Coleta valores → constrói ValidationRuleConfig → accept()."""
        params: Dict[str, Any] = {}
        for spec, widget in self._widget_map.values():
            val = self._read_widget(spec, widget)
            # Pula campos opcionais com valor default — manter os params
            # mais enxutos no JSON. O construtor da regra usa o default
            # próprio se a chave não estiver presente.
            if not spec.required and val == spec.default:
                continue
            params[spec.key] = val

        self._result = ValidationRuleConfig(
            rule_type=self._current_rule_type,
            params=params,
        )
        self.accept()
