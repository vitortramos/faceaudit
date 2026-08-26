# interface/tabs/strategy_manager_v2/strategy_editor/nested_dto_widget.py

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt6 import QtWidgets, QtCore

from interface.theme import current_token_fn
from .widget_helpers import make_combo_shrinkable, make_lineedit_shrinkable


def _btn_style() -> str:
    """QSS do botão de edição, resolvido no tema atual no momento da chamada."""
    t = current_token_fn()
    return (
        "QPushButton { border: 1px solid " + t("BORDER") + "; border-radius: 3px;"
        " background: " + t("SURFACE") + "; padding: 3px 8px; font-size: 11px; }"
        "QPushButton:hover { background: " + t("PRIMARY_BG") + "; }"
    )


class NestedDTOInlineWidget(QtWidgets.QWidget):
    """
    Widget colapsável para edição inline de DTOs aninhados rasos
    (ex: PipelineTrainingPolicy, PipelineApplicationPolicy).

    Mostra um cabeçalho clicável (▶/▼) que expande/colapsa um
    formulário de checkboxes para cada campo bool do DTO.

    Sinais:
        value_changed(dto) — emitido quando qualquer campo muda.
    """

    value_changed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        title: str,
        dto: Any,
        field_labels: Optional[Dict[str, str]] = None,
        field_tooltips: Optional[Dict[str, str]] = None,
        header_tooltip: Optional[str] = None,
        intro_text: Optional[str] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._dto = dto
        self._field_labels = field_labels or {}
        self._field_tooltips = field_tooltips or {}
        self._field_widgets: Dict[str, QtWidgets.QWidget] = {}
        # Mapas auxiliares para a descrição contextual de campos enum.
        self._enum_desc_labels: Dict[str, QtWidgets.QLabel] = {}
        self._enum_member_descriptions: Dict[str, Dict[str, str]] = {}
        self._expanded = True
        # Guarda o intro_text pra reuso em _rebuild_form (re-adicionado
        # ao topo do layout após limpeza).
        self._intro_text = intro_text

        self._init_ui(title, header_tooltip, intro_text)
        self._populate_fields()

    def _init_ui(
        self,
        title: str,
        header_tooltip: Optional[str] = None,
        intro_text: Optional[str] = None,
    ) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header clicável
        t = current_token_fn()
        self._header_btn = QtWidgets.QToolButton()
        self._header_btn.setText(f"▼  {title}")
        self._header_btn.setStyleSheet(
            "QToolButton {"
            "  text-align: left; padding: 4px 6px;"
            "  border: 1px solid " + t("BORDER") + "; border-radius: 3px;"
            "  background: " + t("SURFACE_RAISED") + "; color: " + t("PRIMARY_STRONG") + ";"
            "  font-size: 11px; font-weight: bold;"
            "}"
            "QToolButton:hover { background: " + t("PRIMARY_BG") + "; }"
        )
        self._header_btn.setCheckable(True)
        self._header_btn.setChecked(True)
        if header_tooltip:
            self._header_btn.setToolTip(header_tooltip)
        self._header_btn.clicked.connect(self._on_header_clicked)
        self._header_title = title
        layout.addWidget(self._header_btn)

        # Container de campos
        self._fields_container = QtWidgets.QWidget()
        self._fields_layout = QtWidgets.QFormLayout(self._fields_container)
        self._fields_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._fields_layout.setContentsMargins(20, 4, 4, 4)
        self._fields_layout.setSpacing(4)

        # Texto introdutório opcional — visível sempre que o bloco está
        # expandido. Útil para explicar o propósito/contexto da config para
        # usuários casuais antes que precisem hover por cada tooltip.
        if intro_text:
            intro_lbl = QtWidgets.QLabel(intro_text)
            intro_lbl.setWordWrap(True)
            intro_lbl.setStyleSheet(
                "QLabel {"
                "  background: " + t("ACCENT_BG") + "; border-left: 3px solid " + t("ACCENT") + ";"
                "  border-radius: 2px; padding: 6px 8px;"
                "  color: " + t("ACCENT_TEXT") + "; font-size: 11px; margin-bottom: 4px;"
                "}"
            )
            self._fields_layout.addRow(intro_lbl)

        layout.addWidget(self._fields_container)

    def _populate_fields(self) -> None:
        if self._dto is None or not is_dataclass(self._dto):
            empty = QtWidgets.QLabel("(sem configuração)")
            empty.setStyleSheet(
                "color: " + current_token_fn()("TEXT_MUTED")
                + "; font-size: 11px; font-style: italic;"
            )
            self._fields_layout.addRow(empty)
            return

        # Hooks opcionais para visibilidade dinâmica de fields:
        # - get_hidden_fields(): retorna set de nomes de fields a esconder
        #   condicionalmente (ex: PLDA esconde plda_mode/latent_dim quando
        #   variant=TWO_COVARIANCE).
        # - REACTIVE_FIELDS: tupla de nomes de fields cuja mudança deve
        #   disparar rebuild do form (ex: plda_variant). O reset é feito
        #   por _rebuild_form() abaixo.
        hidden_fields = set()
        if hasattr(self._dto, "get_hidden_fields"):
            try:
                hidden_fields = set(self._dto.get_hidden_fields()) or set()
            except Exception:
                hidden_fields = set()
        # Campos visíveis mas trancados (ex: latent_dim em PLDA SIMPLE +
        # FULL_RANK — valor é ignorado em runtime). Mais informativo que
        # esconder: usuário VÊ o campo, mas com tooltip explicando que
        # o valor real vem do dataset.
        readonly_fields = set()
        if hasattr(self._dto, "get_readonly_fields"):
            try:
                readonly_fields = set(self._dto.get_readonly_fields()) or set()
            except Exception:
                readonly_fields = set()
        reactive_fields = set(getattr(self._dto, "REACTIVE_FIELDS", ()) or ())

        for f in fields(self._dto):
            if f.name in hidden_fields:
                continue
            value = getattr(self._dto, f.name, None)
            label_text = self._field_labels.get(f.name, f.name.replace("_", " ").capitalize())

            if isinstance(value, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(value)

                def _on_bool_changed(checked, fn=f.name,
                                     is_reactive=(f.name in reactive_fields)):
                    self._on_field_changed(fn, checked)
                    # `REACTIVE_FIELDS` era honrado APENAS no ramo Enum, então
                    # um bool listado ali não reconstruía o form e os campos que
                    # ele governa em `get_hidden_fields()` não apareciam/sumiam
                    # ao vivo. Casos afetados: `use_quality_weighting` do
                    # AggregationConfig (esconde a métrica de qualidade) e
                    # `enable_clipping` do BiGaussianizedCalibrationModelConfig,
                    # que já declarava reatividade sem efeito nenhum.
                    if is_reactive:
                        QtCore.QTimer.singleShot(0, self._rebuild_form)

                widget.toggled.connect(_on_bool_changed)
            elif isinstance(value, Enum):
                # Enum field — render as combo. Usa label rico se disponível
                # (atributo .label do enum), senão o nome.
                widget = QtWidgets.QComboBox()
                make_combo_shrinkable(widget)
                enum_cls = type(value)
                member_desc: Dict[str, str] = {}
                for member in enum_cls:
                    display = getattr(member, "label", member.name)
                    widget.addItem(display, userData=member.name)
                    desc = getattr(member, "description", None)
                    if desc:
                        member_desc[member.name] = desc
                # Seleciona o membro atual.
                for idx in range(widget.count()):
                    if widget.itemData(idx) == value.name:
                        widget.setCurrentIndex(idx)
                        break

                # Linha contextual abaixo do combo, atualizada conforme a
                # seleção. Só renderiza se o enum carrega `description`
                # (padrão dos enums "ricos" do projeto).
                desc_label: Optional[QtWidgets.QLabel] = None
                if member_desc:
                    desc_label = QtWidgets.QLabel(
                        member_desc.get(value.name, "")
                    )
                    desc_label.setWordWrap(True)
                    _t = current_token_fn()
                    desc_label.setStyleSheet(
                        "QLabel {"
                        "  background: " + _t("ACCENT_BG") + "; border-left: 2px solid " + _t("ACCENT") + ";"
                        "  border-radius: 2px; padding: 4px 6px;"
                        "  color: " + _t("ACCENT_TEXT") + "; font-size: 10px;"
                        "}"
                    )
                    self._enum_desc_labels[f.name] = desc_label
                    self._enum_member_descriptions[f.name] = member_desc

                def _on_enum_changed(_i, fn=f.name, w=widget, ec=enum_cls,
                                     is_reactive=(f.name in reactive_fields)):
                    selected_name = w.itemData(w.currentIndex())
                    self._on_field_changed(fn, ec[selected_name])
                    lbl = self._enum_desc_labels.get(fn)
                    descs = self._enum_member_descriptions.get(fn, {})
                    if lbl is not None:
                        lbl.setText(descs.get(selected_name, ""))
                    # Field marcado como reativo (ex: plda_variant) — força
                    # rebuild do form pra refletir mudanças em get_hidden_fields().
                    if is_reactive:
                        QtCore.QTimer.singleShot(0, self._rebuild_form)

                widget.currentIndexChanged.connect(_on_enum_changed)
            elif isinstance(value, int) and not isinstance(value, bool):
                widget = QtWidgets.QSpinBox()
                widget.setRange(-(10**9), 10**9)
                widget.setValue(value)
                widget.valueChanged.connect(
                    lambda v, fn=f.name: self._on_field_changed(fn, v)
                )
            elif isinstance(value, float):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setDecimals(4)
                widget.setValue(value)
                widget.valueChanged.connect(
                    lambda v, fn=f.name: self._on_field_changed(fn, v)
                )
            elif isinstance(value, str):
                widget = QtWidgets.QLineEdit(value)
                make_lineedit_shrinkable(widget)
                widget.editingFinished.connect(
                    lambda fn=f.name, w=widget:
                    self._on_field_changed(fn, w.text().strip())
                )
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            ):
                # Lista numérica (list[int] ou list[float]) — entrada via texto
                # separado por vírgula. Mantém o tipo dos elementos no parse:
                # se TODOS forem int, persiste como list[int]; senão list[float].
                is_all_int = all(isinstance(v, int) for v in value)
                text = ", ".join(str(v) for v in value)
                widget = QtWidgets.QLineEdit(text)
                make_lineedit_shrinkable(widget)
                widget.setPlaceholderText("ex: 0.01, 0.1, 1.0, 10")

                def _on_list_edit(fn=f.name, w=widget, force_int=is_all_int):
                    raw = w.text().strip()
                    if not raw:
                        self._on_field_changed(fn, [])
                        return
                    parsed = []
                    for tok in raw.split(","):
                        tok = tok.strip()
                        if not tok:
                            continue
                        try:
                            parsed.append(int(tok) if force_int else float(tok))
                        except ValueError:
                            # token inválido — ignora silenciosamente.
                            pass
                    self._on_field_changed(fn, parsed)

                widget.editingFinished.connect(_on_list_edit)
            elif isinstance(value, list) and not value:
                # Lista vazia — não conhecemos o tipo dos elementos. Tratar
                # como lista de strings (caso degenerado raro).
                widget = QtWidgets.QLineEdit("")
                make_lineedit_shrinkable(widget)
                widget.setPlaceholderText("itens separados por vírgula")
                widget.editingFinished.connect(
                    lambda fn=f.name, w=widget: self._on_field_changed(
                        fn, [t.strip() for t in w.text().split(",") if t.strip()]
                    )
                )
            elif is_dataclass(value):
                # Dataclass aninhada — renderiza recursivamente como
                # sub-seção dentro deste form. A sub-instância tem seu
                # próprio header colapsável (▶/▼); mudanças propagam pra
                # cá via signal value_changed.
                widget = NestedDTOInlineWidget(
                    title=label_text,
                    dto=value,
                    parent=self,
                )
                widget.value_changed.connect(
                    lambda new_dto, fn=f.name:
                    self._on_field_changed(fn, new_dto)
                )
                # Atribui em linha cheia (sem coluna de label) — o header
                # do sub-widget já carrega o título.
                self._field_widgets[f.name] = widget
                self._fields_layout.addRow(widget)
                continue
            else:
                # Tipos não-primitivos: ignora silenciosamente (DTO aninhado profundo
                # deve usar NestedDTODialogTrigger, não inline).
                continue

            help_text = self._field_tooltips.get(f.name) or (
                f.metadata.get("help") if f.metadata else None
            )
            if help_text:
                widget.setToolTip(help_text)

            # Campos read-only (via get_readonly_fields) ficam visíveis
            # mas desabilitados. Sobrescreve o tooltip pra explicar por
            # que está trancado.
            if f.name in readonly_fields:
                widget.setEnabled(False)
                widget.setToolTip(
                    "Este campo é determinado automaticamente pelo dataset "
                    "de treinamento no momento da execução — o valor exibido "
                    "é apenas referência."
                )

            self._field_widgets[f.name] = widget
            self._fields_layout.addRow(f"{label_text}:", widget)

            # Se este campo é um Enum com descrição, adiciona a linha
            # contextual ABAIXO do combo, sem coluna de label.
            enum_desc_label = self._enum_desc_labels.get(f.name)
            if enum_desc_label is not None:
                self._fields_layout.addRow(enum_desc_label)

    def _on_header_clicked(self) -> None:
        self._expanded = self._header_btn.isChecked()
        self._fields_container.setVisible(self._expanded)
        arrow = "▼" if self._expanded else "▶"
        self._header_btn.setText(f"{arrow}  {self._header_title}")

    def _on_field_changed(self, field_name: str, new_value: Any) -> None:
        if self._dto is None:
            return
        setattr(self._dto, field_name, new_value)
        self.value_changed.emit(self._dto)

    def _rebuild_form(self) -> None:
        """
        Recria o formulário com a visibilidade atual dos fields.
        Usado quando um campo `REACTIVE_FIELDS` muda (ex: plda_variant)
        e a função `get_hidden_fields()` retorna outro conjunto.
        """
        # Limpa widgets e mapas auxiliares.
        self._field_widgets.clear()
        self._enum_desc_labels.clear()
        self._enum_member_descriptions.clear()
        # Remove todas as rows do form layout (do fim pro começo).
        while self._fields_layout.rowCount() > 0:
            self._fields_layout.removeRow(self._fields_layout.rowCount() - 1)
        # Re-adiciona o intro_text (foi removido no clear acima).
        if self._intro_text:
            intro_lbl = QtWidgets.QLabel(self._intro_text)
            intro_lbl.setWordWrap(True)
            _t = current_token_fn()
            intro_lbl.setStyleSheet(
                "QLabel {"
                "  background: " + _t("ACCENT_BG") + "; border-left: 3px solid " + _t("ACCENT") + ";"
                "  border-radius: 2px; padding: 6px 8px;"
                "  color: " + _t("ACCENT_TEXT") + "; font-size: 11px; margin-bottom: 4px;"
                "}"
            )
            self._fields_layout.addRow(intro_lbl)
        # Repopula com as novas hidden_fields.
        self._populate_fields()

    def value(self) -> Any:
        return self._dto


class NestedDTODialogTrigger(QtWidgets.QWidget):
    """
    Widget para edição de DTOs aninhados profundos via dialog separado
    (ex: ComparisonMiningConfig).

    Mostra:
    - Linha-resumo do DTO (extraída via summary_fn)
    - Botão que abre um dialog de edição (provido via open_dialog_fn)

    Sinais:
        value_changed(dto) — emitido quando o dialog retorna OK e o DTO mudou.
    """

    value_changed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        title: str,
        dto: Any,
        summary_fn: Callable[[Any], str],
        open_dialog_fn: Callable[[Any, QtWidgets.QWidget], Optional[Any]],
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        """
        Args:
            title: rótulo curto exibido acima do botão.
            dto: instância atual do DTO (pode ser None).
            summary_fn: produz a linha-resumo a partir do DTO.
            open_dialog_fn: abre o dialog de edição. Recebe (dto, parent_widget) e
                retorna o DTO atualizado em caso de OK ou None se cancelado.
        """
        super().__init__(parent)
        self._dto = dto
        self._summary_fn = summary_fn
        self._open_dialog_fn = open_dialog_fn

        self._init_ui(title)
        self._refresh_summary()

    def _init_ui(self, title: str) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        t = current_token_fn()
        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setStyleSheet(
            "font-size: 11px; color: " + t("PRIMARY_STRONG") + "; font-weight: bold;"
        )
        layout.addWidget(title_lbl)

        self._summary_lbl = QtWidgets.QLabel("(sem configuração)")
        self._summary_lbl.setStyleSheet(
            "font-size: 10px; color: " + t("TEXT_MUTED") + "; padding-left: 4px;"
        )
        self._summary_lbl.setWordWrap(True)
        layout.addWidget(self._summary_lbl)

        self._btn_edit = QtWidgets.QPushButton("⚙ Editar…")
        self._btn_edit.setMaximumWidth(180)
        self._btn_edit.setStyleSheet(_btn_style())
        self._btn_edit.clicked.connect(self._on_edit_clicked)
        layout.addWidget(self._btn_edit)

    def _refresh_summary(self) -> None:
        text = self._summary_fn(self._dto) if self._dto is not None else "(sem configuração)"
        self._summary_lbl.setText(text)

    def _on_edit_clicked(self) -> None:
        new_dto = self._open_dialog_fn(self._dto, self)
        if new_dto is not None:
            self._dto = new_dto
            self._refresh_summary()
            self.value_changed.emit(new_dto)

    def value(self) -> Any:
        return self._dto


# ─── Helpers de detecção e dispatch para DTOs aninhados ──────────────────────

# Atributos da instrução que carregam DTO aninhado e como devem ser renderizados.
# Cada entrada: (attr_name, render_strategy)
#
# `training_policy` foi removido da UI: os defaults aplicados pelo executor
# (PipelineTrainingPolicy.default_linear / default_dual) cobrem o uso esperado
# pelos peritos. Variantes de pesquisa podem ser editadas via JSON do template.
NESTED_DTO_ATTRS: Tuple[Tuple[str, str], ...] = (
    ("application_policy", "inline"),
    ("mining_config", "dialog"),
    # Configs embutidas das instruções de modelo de similaridade/calibração —
    # exibidas inline quando o tipo correspondente está selecionado.
    # O painel filtra qual aparecer via uses_plda / uses_logistic_regression /
    # uses_kde / uses_bigauss.
    ("similarity_model_config", "inline"),
    ("logreg_config", "inline"),
    ("kde_config", "inline"),
    ("bigauss_config", "inline"),
    # Polimórfico: KNNTypicalityModelConfig ou MoVMFTypicalityModelConfig.
    # O painel gate pela escolha de backend (uses_knn / uses_movmf).
    ("typicality_model_config", "inline"),
    # Regras de fusão da EMBEDDING_AGGREGATION. O próprio DTO declara
    # `get_hidden_fields()` e `REACTIVE_FIELDS`, então k/teto/métrica aparecem
    # e somem conforme o método e a ponderação escolhidos.
    ("aggregation_config", "inline"),
)


_PIPELINE_TRAINING_POLICY_LABELS = {
    "use_target_for_coral":         "CORAL alinha covariância ao target  (recomendado)",
    "use_target_for_whitening":     "Whitening conjunto (source ∪ target)  (raro)",
    "use_target_for_lda":           "LDA conjunto (source ∪ target)  (raro)",
    "copy_shared_state_to_target":  "Replicar estado de steps compartilhados  (recomendado)",
}

_PIPELINE_APPLICATION_POLICY_LABELS = {
    "save_checkpoints":          "Salvar checkpoints intermediários",
    "save_transformed_dataset":  "Salvar dataset final transformado",
    "save_trajectory_data":      "Salvar dados de trajetória (visualização 3D)",
}


_PIPELINE_TRAINING_POLICY_TOOLTIPS = {
    "use_target_for_coral":
        "Padrão: ATIVADO.\n\n"
        "ATIVADO: CORAL alinha a covariância do source à do target — "
        "uso clássico de CORAL para adaptação de domínio entre populações.\n\n"
        "DESATIVADO: CORAL é treinado apenas com o source, perdendo o efeito "
        "de adaptação. Útil apenas para comparar com/sem alinhamento.",
    "use_target_for_whitening":
        "Padrão: DESATIVADO.\n\n"
        "ATIVADO: Whitening é fit em source ∪ target (whitening conjunto). "
        "Variante de pesquisa pouco comum.\n\n"
        "DESATIVADO: Whitening é fit apenas no source — comportamento padrão "
        "da literatura.",
    "use_target_for_lda":
        "Padrão: DESATIVADO.\n\n"
        "ATIVADO: LDA é fit usando identidades de source + target combinados. "
        "Exige rótulos em ambos os datasets. Variante exótica.\n\n"
        "DESATIVADO: LDA é fit apenas no source — comportamento padrão.",
    "copy_shared_state_to_target":
        "Padrão: ATIVADO em DUAL.\n\n"
        "ATIVADO: steps comuns aos dois ramos (Centering, Whitening, L2) "
        "treinados no source têm seu estado replicado no ramo target — "
        "necessário para que a transformação seja consistente entre ramos.\n\n"
        "DESATIVADO: cada ramo treinaria seu próprio estado independente, "
        "quebrando a comparabilidade no scoring DUAL.",
}

_PIPELINE_APPLICATION_POLICY_TOOLTIPS = {
    "save_checkpoints":
        "Persiste em disco o estado intermediário do pipeline a cada step "
        "durante a aplicação. Útil para auditoria/debug. Aumenta uso de disco.",
    "save_transformed_dataset":
        "Persiste em disco o dataset final após todas as transformações. "
        "Permite reusá-lo em outras instruções sem reaplicar o pipeline.",
    "save_trajectory_data":
        "Captura dados intermediários necessários para visualização 3D da "
        "trajetória de transformação via PipelineTrajectoryService.",
}


_PLDA_CONFIG_LABELS = {
    "plda_variant":   "Variante",
    "em_iterations":  "Iterações EM",
    "plda_mode":      "Modo PLDA",
    "latent_dim":     "Dimensão Latente",
    "regularization": "Regularização (TwoCov)",
    "random_seed":    "Semente Aleatória",
}

_PLDA_CONFIG_TOOLTIPS = {
    "plda_variant":
        "SIMPLE (padrão): formulação SpeechBrain com subspace F entre-classes "
        "(rank configurável). Treinada via EM iterativo. Usa em_iterations, "
        "plda_mode e latent_dim.\n\n"
        "TWO_COVARIANCE: formulação Brümmer 2010 com covariâncias completas "
        "B (entre-classes) e W (intra-classe). Treinada em forma fechada — "
        "sem EM. Recomendada na literatura forense (FoCal toolkit). "
        "em_iterations / plda_mode / latent_dim são IGNORADOS; usa "
        "`regularization` opcional.",
    "em_iterations":
        "Número de iterações do algoritmo Expectation-Maximization no treino "
        "do PLDA. Mais iterações = treino mais lento mas potencialmente mais "
        "estável. Padrão 20 funciona bem na maioria dos casos. "
        "Usado apenas em variant=SIMPLE.",
    "plda_mode":
        "FULL_RANK: usa o maior rank permitido pelos dados (latent_dim ignorado).\n"
        "LOW_RANK: usa latent_dim como limite do espaço latente. Reduz "
        "dimensionalidade e geralmente é mais robusto.\n\n"
        "Usado apenas em variant=SIMPLE.",
    "latent_dim":
        "Dimensão do espaço latente quando plda_mode=LOW_RANK. Valores típicos: "
        "64–256. Aumentar pode capturar mais variabilidade mas exige mais dados "
        "para evitar overfitting. Usado apenas em variant=SIMPLE.",
    "regularization":
        "Termo ε ≥ 0 somado a B e W como ε·I antes da inversão no scoring. "
        "Estabiliza numericamente quando D é alto comparado ao número de "
        "classes/amostras. Padrão 0 = sem regularização. "
        "Usado apenas em variant=TWO_COVARIANCE.",
    "random_seed":
        "Semente aleatória para reprodutibilidade do treino.",
}


_LOGREG_CONFIG_LABELS = {
    "C":                     "C (regularização inversa)",
    "use_gridsearch":        "Otimizar C via GridSearch",
    "c_grid":                "Grade de C (GridSearch)",
    "cv_folds":              "Folds do CV (GridSearch)",
    "scoring":               "Métrica do GridSearch",
    "standardize_features":  "Standardizar features (score vectorization)",
    "k_folds":               "🧪 Cllr CV diagnóstico — k-folds (0 = desativado)",
    "cv_split_mode":         "🧪 Modo de partição do CV diagnóstico",
}

_LOGREG_CONFIG_TOOLTIPS = {
    "C":
        "Parâmetro de regularização inversa da Regressão Logística. "
        "C menor → mais regularização (mais robusto a outliers, menos "
        "overfitting). C maior → menos regularização. Padrão 1.0. "
        "Ignorado quando 'Otimizar C via GridSearch' está ATIVO.",
    "use_gridsearch":
        "Se ATIVO, ignora o C fixo e busca o melhor valor em c_grid via "
        "cross-validation estratificada. Recomendado quando há scores e "
        "labels suficientes (≥ algumas centenas de pares por classe).",
    "c_grid":
        "Grade de valores de C testados pelo GridSearch. Lista separada "
        "por vírgula (ex: 0.01, 0.1, 1.0, 10, 100). Ignorado quando "
        "use_gridsearch=False.",
    "cv_folds":
        "Número de folds da cross-validation estratificada (StratifiedKFold). "
        "Padrão 5. Reduzido automaticamente se a classe minoritária tem "
        "menos amostras. Ignorado quando use_gridsearch=False.",
    "scoring":
        "Métrica usada pelo GridSearch:\n"
        "- 'neg_log_loss' (padrão): penaliza miscalibração diretamente. "
        "Ideal para calibradores.\n"
        "- 'roc_auc': otimiza separabilidade, não calibração. Útil "
        "quando o objetivo é apenas ranking.",
    "standardize_features":
        "Quando ATIVO, aplica z-score por feature (μ, σ aprendidos no "
        "treino) antes do LogReg. Necessário em score vectorization quando "
        "os sistemas têm escalas muito diferentes (ex: Cosseno em [-0.5, "
        "+0.5] + PLDA em [-200, +60]). Sem standardização, a regularização "
        "L2 penaliza pesos uniformemente, distorcendo a contribuição "
        "relativa das features.\n\n"
        "Só ativa quando X tem shape (N, D) com D > 1 (multivariado). "
        "Para calibração univariada normal (D=1), o flag é ignorado. "
        "μ/σ são persistidos com o modelo e re-aplicados automaticamente "
        "em validação e casework.\n\n"
        "Default: desativado (preserva comportamento legado).",
    "k_folds":
        "K-fold CV DIAGNÓSTICO do calibrador (opt-in). NÃO altera o "
        "calibrador final — apenas calcula Cllr out-of-fold pra auditoria.\n\n"
        "0 (padrão) → desativado, comportamento idêntico ao anterior.\n"
        "5 → roda 5 folds estratificados após o fit principal usando "
        "C congelado (best_C_) — anexa Cllr CV out-of-fold ao metadata, "
        "propagado pra ValidationMetrics.\n\n"
        "Cllr CV mede generalização DENTRO do domínio de calibração — "
        "NÃO é a performance em validação externa.",
    "cv_split_mode":
        "Estratégia de partição dos folds (só aplica quando k_folds > 0).\n\n"
        "STRATIFIED_PAIR (padrão): sklearn.StratifiedKFold. Pares da mesma "
        "identidade podem aparecer em treino e held-out → otimista.\n\n"
        "STRATIFIED_GROUP: sklearn.StratifiedGroupKFold por probe identity. "
        "Forense-defensável, mais pessimista, recomendado em laudos.\n"
        "LIMITAÇÃO: agrupa só por probe_identity — a identidade da REFERÊNCIA "
        "pode ainda vazar entre treino e held-out. Adequado quando o protocolo "
        "trata probe como questionada e reference como padrão.",
}


_KDE_CONFIG_LABELS = {
    "bandwidth_genuine":         "Bandwidth (genuínos)",
    "bandwidth_impostor":        "Bandwidth (impostores)",
    "use_gridsearch":            "Otimizar bandwidth via GridSearch",
    "max_samples_for_gridsearch": "Máx. amostras (GridSearch)",
    "max_stored_samples":        "Máx. amostras armazenadas",
    "min_bandwidth_abs":         "Bandwidth mínimo absoluto",
    "max_grid_expansions":       "Máx. expansões do grid",
    "min_log_lr":                "log-LR mínimo",
    "max_log_lr":                "log-LR máximo",
    "interpolation_points":      "Pontos da interpolação",
    "random_seed":               "Semente Aleatória",
    "kernel":                    "Kernel KDE",
    "k_folds":                   "🧪 Cllr CV diagnóstico — k-folds (0 = desativado)",
    "cv_split_mode":             "🧪 Modo de partição do CV diagnóstico",
}

_KDE_CONFIG_TOOLTIPS = {
    "bandwidth_genuine":
        "Bandwidth do KDE para a distribuição de scores genuínos. "
        "Ignorado quando use_gridsearch=True.",
    "bandwidth_impostor":
        "Bandwidth do KDE para a distribuição de scores impostores. "
        "Ignorado quando use_gridsearch=True.",
    "use_gridsearch":
        "Se ATIVO, otimiza o bandwidth via GridSearch + cross-validation, "
        "ignorando bandwidth_genuine/impostor. Recomendado para a maioria dos casos.",
    "max_samples_for_gridsearch":
        "Limite de amostras usadas no GridSearch para acelerar. O ajuste final "
        "ainda usa todas as amostras (até max_stored_samples).",
    "max_stored_samples":
        "Máximo de amostras retidas por classe (genuínos/impostores) para "
        "ajuste final do KDE.",
    "min_bandwidth_abs":
        "Bandwidth mínimo absoluto no espaço z-score — evita "
        "que o GridSearch escolha bandwidth degenerado próximo de zero.",
    "max_grid_expansions":
        "Número máximo de vezes que o grid de busca pode ser expandido se o "
        "melhor bandwidth cair na borda do grid atual.",
    "min_log_lr":
        "Limite inferior do log-LR calibrado (clipping de saída).",
    "max_log_lr":
        "Limite superior do log-LR calibrado (clipping de saída).",
    "interpolation_points":
        "Quantidade de pontos usados para construir a curva interpolada "
        "score → LR. Mais pontos = mais preciso, levemente mais lento.",
    "random_seed":
        "Semente aleatória usada em amostragens internas, quando aplicável.",
    "kernel":
        "Kernel do sklearn.neighbors.KernelDensity. Padrão 'gaussian'.",
    "k_folds":
        "K-fold CV DIAGNÓSTICO do calibrador (opt-in). NÃO altera o "
        "calibrador final — apenas calcula Cllr out-of-fold pra auditoria.\n\n"
        "0 (padrão) → desativado, comportamento idêntico ao anterior.\n"
        "5 → roda 5 folds estratificados após o fit principal usando os "
        "bandwidths congelados — anexa Cllr CV out-of-fold ao metadata, "
        "propagado pra ValidationMetrics.\n\n"
        "KDE tende a mostrar gap CV − in-sample mais alto que LogReg "
        "(calibrador de alta capacidade overfita mais). Cllr CV mede "
        "generalização DENTRO do domínio de calibração — NÃO é a "
        "performance em validação externa.",
    "cv_split_mode":
        "Estratégia de partição dos folds (só aplica quando k_folds > 0).\n\n"
        "STRATIFIED_PAIR (padrão): sklearn.StratifiedKFold. Pares da mesma "
        "identidade podem aparecer em treino e held-out → otimista.\n\n"
        "STRATIFIED_GROUP: sklearn.StratifiedGroupKFold por probe identity. "
        "Forense-defensável, mais pessimista, recomendado em laudos.\n"
        "LIMITAÇÃO: agrupa só por probe_identity — a identidade da REFERÊNCIA "
        "pode ainda vazar entre treino e held-out.",
}


_BIGAUSS_CONFIG_LABELS = {
    "target_metric":           "Métrica do σ alvo",
    "initial_calibrator":      "Calibrador inicial",
    "inversion_method":        "Método de inversão da CDF",
    "sigma_shrinkage_factor":  "Fator de encolhimento (σ)",
    "enable_clipping":         "Habilitar Clipping",
    "min_log_lr":              "log-LR mínimo (clipping)",
    "max_log_lr":              "log-LR máximo (clipping)",
    "interpolation_points":    "Pontos da interpolação (INTERP1D)",
    "brent_xtol":              "Tolerância do Brent",
    "brent_search_range":      "Range simétrico do Brent",
    "initial_logreg_C":        "C da LogReg inicial",
    "initial_kde_bandwidth":   "Bandwidth do KDE inicial (0=auto)",
    "random_seed":             "Semente aleatória",
    "gx_method":                  "Método de G(x)",
    "gx_kde_bandwidth_gen":       "Bandwidth G(x) genuínos (0=auto)",
    "gx_kde_bandwidth_imp":       "Bandwidth G(x) impostores (0=auto)",
    "gx_clamp_to_training_range": "Clampar G(x) ao range do treino",
    "gx_grid_points":             "Pontos do grid G(x)",
    "k_folds":                    "🧪 Cllr CV diagnóstico — k-folds (0 = desativado)",
    "cv_split_mode":              "🧪 Modo de partição do CV diagnóstico",
}

_BIGAUSS_CONFIG_TOOLTIPS = {
    "target_metric":
        "CLLR (padrão): numérico, fit ln(Cllr) = b·(exp(-c·σ²)-1) — "
        "Eq. (8) do paper. Usa toda a informação dos scores; recomendado "
        "quando há ≥ 50 fontes.\n\n"
        "EER: analítico, σ = -2·Φ⁻¹(EER) — Eq. (10). Mais ruidoso com "
        "poucas same-source samples (discretiza em passos 1/N).",
    "initial_calibrator":
        "NONE (padrão): scores já em espaço de ln(Λ) (caso PLDA log-LR). "
        "Cllr/EER calculados direto.\n"
        "LOGREG: calibração linear inicial, rápida.\n"
        "KDE: §3.6 do paper — robusto a outliers, ~15× mais lento.\n\n"
        "Só afeta a derivação do σ² alvo; o mapeamento final sempre usa "
        "a CDF empírica.",
    "inversion_method":
        "INTERP1D (padrão): grid pré-computado + interpolação linear. "
        "O(N) total. Preciso o bastante na prática.\n\n"
        "BRENT_ROOT: root-finding por ponto via Brent. Mais preciso nas "
        "caudas, ~50–100× mais lento.",
    "sigma_shrinkage_factor":
        "§3.5 do paper. 1.0 = sem encolhimento; 0.9 = 90% do σ calculado, "
        "puxa LRs grandes pra perto de 1. Útil quando há pouco treino e "
        "preocupa-se com extrapolação.",
    "enable_clipping":
        "Aplica clipping no log-LR calibrado a [min_log_lr, max_log_lr]. "
        "Padrão: desligado — o calibrador não trunca a saída. Ligar pode "
        "ser útil em laudos onde se quer reportar log-LR com magnitude "
        "limitada, mas introduz perda de informação e satura métricas "
        "como cllr_min.\n\n"
        "⚠ IMPORTANTE: este toggle controla APENAS o clip final do log-LR. "
        "Mesmo desligado, o BiGauss tem dois clamps internos que produzem "
        "floors/ceilings naturais e NÃO são afetados:\n"
        "• gx_clamp_to_training_range (default True): scores fora do range "
        "do treino mapeiam pra G(x_min)/G(x_max) → todos caem no mesmo "
        "log-LR → aparece como patamar nas caudas da curva.\n"
        "• Clamp de probabilidade u ∈ [ε, 1−ε] (hardcoded): evita F⁻¹ "
        "infinito nas bordas. Define o piso/teto matemático que o BiGauss "
        "consegue emitir (depende de σ alvo).\n\n"
        "Pra eliminar o patamar visual, é preciso desligar também "
        "gx_clamp_to_training_range (disponível quando gx_method=KDE_SMOOTHED), "
        "mas isso permite extrapolação confiante — não recomendado em forense.",
    "min_log_lr":
        "Limite inferior do log-LR calibrado (clipping de saída). "
        "Só aplica se enable_clipping=True.",
    "max_log_lr":
        "Limite superior do log-LR calibrado (clipping de saída). "
        "Só aplica se enable_clipping=True.",
    "interpolation_points":
        "Quantidade de pontos do grid usado pra construir G(x) (CDF empírica) "
        "e F⁻¹ (inversa da mistura). Só aplica se inversion_method=INTERP1D.",
    "brent_xtol":
        "Tolerância absoluta do scipy.optimize.brentq. Só aplica se "
        "inversion_method=BRENT_ROOT.",
    "brent_search_range":
        "Range simétrico [-R, +R] em ln(Λ) usado como bracket inicial do "
        "Brent. Default 50 cobre LR de até e^50.",
    "initial_logreg_C":
        "Parâmetro C (regularização inversa) da LogReg inicial. "
        "Só aplica se initial_calibrator=LOGREG.",
    "initial_kde_bandwidth":
        "Bandwidth do KDE inicial. 0.0 = automático via Silverman (padrão "
        "do paper). Só aplica se initial_calibrator=KDE.",
    "random_seed":
        "Semente para amostragens internas (reprodutibilidade).",
    "gx_method":
        "EMPIRICAL_LINEAR (padrão, paper §2.7): soma cumulativa nos breakpoints "
        "+ interpolação linear. Preserva exatamente os dados; pode produzir "
        "stairsteps visíveis quando há poucos genuínos no treino.\n\n"
        "KDE_SMOOTHED: variante deste projeto. G(x) = 0.5·F̂_g(x) + 0.5·F̂_i(x) "
        "via integração analítica de KDEs Gaussianos por classe. Elimina "
        "stairsteps mas introduz bandwidth como hiperparâmetro.",
    "gx_kde_bandwidth_gen":
        "Bandwidth do KDE dos genuínos pra construir G(x). 0.0 = Silverman "
        "automático. Só aplica se gx_method=KDE_SMOOTHED.",
    "gx_kde_bandwidth_imp":
        "Bandwidth do KDE dos impostores pra construir G(x). 0.0 = Silverman "
        "automático. Só aplica se gx_method=KDE_SMOOTHED.",
    "gx_clamp_to_training_range":
        "Preserva a propriedade do paper §2.7: scores fora do range do treino "
        "recebem G(x_min) ou G(x_max). Recomendado True. False permite que o "
        "KDE extrapole suavemente nas caudas. Só aplica se gx_method=KDE_SMOOTHED.",
    "gx_grid_points":
        "Quantidade de pontos do grid usado pra cachear G(x). 2000 é suficiente. "
        "Só aplica se gx_method=KDE_SMOOTHED.",
    "k_folds":
        "K-fold CV DIAGNÓSTICO do calibrador BiGauss completo (opt-in). "
        "NÃO altera o calibrador final — apenas calcula Cllr out-of-fold "
        "pra auditoria.\n\n"
        "DIFERENTE do sigma_cv_folds — aquele é CV interno só pra estimar "
        "σ². Este re-roda o pipeline inteiro (initial calibrator + σ² + "
        "G(x) + F⁻¹) em k-folds.\n\n"
        "0 (padrão) → desativado, comportamento idêntico ao anterior.\n"
        "5 → roda 5 folds estratificados após o fit principal usando todos "
        "os hiperparâmetros congelados — anexa Cllr CV out-of-fold ao "
        "metadata, propagado pra ValidationMetrics.\n\n"
        "Cllr CV mede generalização DENTRO do domínio de calibração — "
        "NÃO é a performance em validação externa.",
    "cv_split_mode":
        "Estratégia de partição dos folds (só aplica quando k_folds > 0).\n\n"
        "STRATIFIED_PAIR (padrão): sklearn.StratifiedKFold. Pares da mesma "
        "identidade podem aparecer em treino e held-out → otimista.\n\n"
        "STRATIFIED_GROUP: sklearn.StratifiedGroupKFold por probe identity. "
        "Forense-defensável, mais pessimista, recomendado em laudos.\n"
        "LIMITAÇÃO: agrupa só por probe_identity — a identidade da REFERÊNCIA "
        "pode ainda vazar entre treino e held-out.",
}


_AGGREGATION_CONFIG_LABELS = {
    "method": "Método de fusão",
    "combination_group_size": "Faces por template (k)",
    "max_synthetic_templates_per_id": "Teto de templates por identidade",
    "combination_sampling": "Seleção acima do teto",
    "use_quality_weighting": "Ponderar pela qualidade",
    "quality_weight_type": "Métrica de qualidade",
}


def get_inline_field_labels(attr_name: str) -> Dict[str, str]:
    """Retorna o mapa de field_name → label legível para o DTO inline conhecido."""
    if attr_name == "training_policy":
        return _PIPELINE_TRAINING_POLICY_LABELS
    if attr_name == "application_policy":
        return _PIPELINE_APPLICATION_POLICY_LABELS
    if attr_name == "similarity_model_config":
        return _PLDA_CONFIG_LABELS
    if attr_name == "logreg_config":
        return _LOGREG_CONFIG_LABELS
    if attr_name == "kde_config":
        return _KDE_CONFIG_LABELS
    if attr_name == "bigauss_config":
        return _BIGAUSS_CONFIG_LABELS
    if attr_name == "aggregation_config":
        return _AGGREGATION_CONFIG_LABELS
    return {}


def get_inline_field_tooltips(attr_name: str) -> Dict[str, str]:
    """Retorna o mapa de field_name → tooltip explicativo para o DTO inline."""
    if attr_name == "training_policy":
        return _PIPELINE_TRAINING_POLICY_TOOLTIPS
    if attr_name == "application_policy":
        return _PIPELINE_APPLICATION_POLICY_TOOLTIPS
    if attr_name == "similarity_model_config":
        return _PLDA_CONFIG_TOOLTIPS
    if attr_name == "logreg_config":
        return _LOGREG_CONFIG_TOOLTIPS
    if attr_name == "kde_config":
        return _KDE_CONFIG_TOOLTIPS
    if attr_name == "bigauss_config":
        return _BIGAUSS_CONFIG_TOOLTIPS
    return {}


_INLINE_INTRO_TEXTS = {
    "logreg_config": (
        "A Regressão Logística é o calibrador padrão para converter scores brutos "
        "em probabilidades (e, daí, em LRs). O hiperparâmetro mais sensível é o "
        "C — controla o quanto o modelo é regularizado. C pequeno = mais "
        "regularização, fronteira de decisão mais suave; C grande = menos "
        "regularização, fronteira mais ajustada aos dados de treino.\n\n"
        "💡 Como o C ótimo varia muito entre datasets, recomenda-se deixar "
        "'Otimizar C via GridSearch' ATIVO (padrão). Os parâmetros abaixo "
        "controlam a busca; passe o mouse sobre cada um para ver detalhes."
    ),
    "kde_config": (
        "O calibrador KDE estima duas densidades de probabilidade não-paramétricas "
        "(uma para genuínos, outra para impostores) usando Kernel Density Estimation, "
        "com transformação de Fisher para estabilizar scores próximos de ±1. "
        "O parâmetro mais crítico é o bandwidth — passe o mouse sobre cada campo "
        "para ver detalhes específicos."
    ),
    "similarity_model_config": (
        "Hyperparâmetros do PLDA (Probabilistic Linear Discriminant Analysis).\n\n"
        "Escolha primeiro a VARIANTE:\n"
        "• SIMPLE (SpeechBrain): subspace F entre-classes, treino EM. "
        "FULL_RANK preserva o máximo de informação; LOW_RANK reduz para "
        "latent_dim, geralmente mais estável com poucos dados.\n"
        "• TWO_COVARIANCE (Brümmer): covariâncias completas B/W em forma "
        "fechada — sem EM. Padrão da literatura forense; ignora plda_mode "
        "e latent_dim, mas aceita `regularization` para estabilizar."
    ),
    "bigauss_config": (
        "Calibrador Bi-Gaussianized (Morrison 2024, Law, Probability and Risk). "
        "Mapeia scores para um sistema bi-gaussiano perfeitamente calibrado com "
        "σ² alvo derivado de Cllr ou EER do treino, via CDF empírica + inversão "
        "da mistura 0.5·N(+σ²/2,σ²) + 0.5·N(-σ²/2,σ²).\n\n"
        "💡 Defaults sensatos: target_metric=CLLR, initial_calibrator=NONE "
        "(adequado para PLDA log-LR), inversion_method=INTERP1D. Mude apenas "
        "se souber o que está fazendo — passe o mouse sobre cada campo para "
        "ver detalhes."
    ),
}


def get_inline_intro_text(attr_name: str) -> Optional[str]:
    """Retorna o texto introdutório (visível ao expandir) do bloco DTO inline."""
    return _INLINE_INTRO_TEXTS.get(attr_name)


def get_inline_title(attr_name: str) -> str:
    if attr_name == "training_policy":
        return "Política de Treinamento"
    if attr_name == "application_policy":
        return "Política de Aplicação"
    if attr_name == "similarity_model_config":
        return "Config PLDA"
    if attr_name == "logreg_config":
        return "Config Logistic Regression"
    if attr_name == "kde_config":
        return "Config KDE"
    if attr_name == "bigauss_config":
        return "Config Bi-Gaussianized"
    if attr_name == "typicality_model_config":
        # Genérico — o backend específico (k-NN ou MoVMF) já está
        # selecionado no dropdown acima.
        return "Hiperparâmetros do Modelo"
    if attr_name == "aggregation_config":
        return "Regras de Agregação"
    return attr_name.replace("_", " ").title()


def make_default_dto(attr_name: str) -> Any:
    """Constrói um DTO default para o atributo conhecido quando o valor é None."""
    if attr_name == "training_policy":
        from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy
        return PipelineTrainingPolicy()
    if attr_name == "application_policy":
        from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy
        return PipelineApplicationPolicy()
    if attr_name == "similarity_model_config":
        from faceaudit.models.similarity.plda_similarity_model_config import (
            PLDASimilarityModelConfig,
        )
        return PLDASimilarityModelConfig()
    if attr_name == "logreg_config":
        from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
            LogisticRegressionCalibrationModelConfig,
        )
        return LogisticRegressionCalibrationModelConfig()
    if attr_name == "kde_config":
        from faceaudit.models.calibration.kde_calibration_model_config import (
            KDECalibrationModelConfig,
        )
        return KDECalibrationModelConfig()
    if attr_name == "bigauss_config":
        from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
            BiGaussianizedCalibrationModelConfig,
        )
        return BiGaussianizedCalibrationModelConfig()
    if attr_name == "typicality_model_config":
        # Fallback genérico. O caso real é tratado pelo painel: quando o
        # config nested vem `None`, o panel usa
        # `instr._reconcile_config_type()` antes de chamar este helper —
        # então este fallback raramente é alcançado. Mantemos KNN como
        # default seguro (backend estável; MoVMF é EXPERIMENTAL).
        from faceaudit.models.typicality.knn_typicality_model_config import (
            KNNTypicalityModelConfig,
        )
        return KNNTypicalityModelConfig()
    if attr_name == "aggregation_config":
        from faceaudit.models.aggregation_config import AggregationConfig
        return AggregationConfig()
    return None


# ─── Dialog wrappers (estratégia "dialog") ───────────────────────────────────

def summarize_mining_config(cfg: Any) -> str:
    if cfg is None:
        return "(nenhuma configuração)"
    name = getattr(cfg, "name", "—")
    parts: List[str] = [f"nome: {name}"]
    seed = getattr(cfg, "seed", None)
    if seed is not None:
        parts.append(f"seed: {seed}")

    imp_rule = getattr(cfg, "impostors_rule", None)
    gen_rule = getattr(cfg, "genuines_rule", None)
    if imp_rule is not None:
        strat = getattr(imp_rule.strategy, "value", str(imp_rule.strategy))
        parts.append(f"imp: {strat}")
    if gen_rule is not None:
        strat = getattr(gen_rule.strategy, "value", str(gen_rule.strategy))
        parts.append(f"gen: {strat}")
    return " · ".join(parts)


def open_mining_config_dialog(
    current: Any, parent: QtWidgets.QWidget
) -> Optional[Any]:
    """Abre o NewMiningConfigDialogController, hidratado com o config atual."""
    from interface.tabs.strategy_manager.new_mining_config_dialog.new_mining_config_dialog_controller import (
        NewMiningConfigDialogController,
    )
    from faceaudit.models.comparison_mining_config import ComparisonMiningConfig

    controller = NewMiningConfigDialogController(parent=parent)
    if isinstance(current, ComparisonMiningConfig):
        controller.load_config(current)

    if controller.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        return controller.get_config()
    return None
