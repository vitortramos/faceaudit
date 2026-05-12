# interface/tabs/strategy_manager_v2/strategy_editor/output_row_widget.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class OutputRowWidget(QtWidgets.QFrame):
    """
    Linha de output no painel de resultados (3ª coluna do editor) e na seção
    Saídas do InstructionConfigPanel.

    Modo `read_only=True` (painel de resultados): mostra status + nome em
    QLabel + ResourceInfoWidget(compact=True) sempre expandido com as mesmas
    seções do painel de Propriedades do recurso.

    Modo `read_only=False` (painel de configuração): mostra status + nome
    editável em QLineEdit. Sem props embedadas — é só edição.

    Sinais:
        name_changed(field_name, new_text) — emitido quando o usuário edita
        o nome do recurso (modo editável apenas).
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

        # ResourceInfoWidget embedado (modo read_only). Eagerly criado em
        # `_init_ui` e sempre visível quando há recurso com dados. Cache
        # via signature evita `show_resource` redundante em chamadas
        # repetidas de update_state.
        self._info_widget: Optional[QtWidgets.QWidget] = None
        self._info_container: Optional[QtWidgets.QWidget] = None
        self._last_info_signature: Optional[tuple] = None

        self.setStyleSheet(
            "OutputRowWidget {"
            "  background-color: #fafdf6;"
            "  border: 1px solid #d6e8d0;"
            "  border-radius: 4px;"
            "}"
        )

        # Vertical Maximum: o widget cresce até o sizeHint do conteúdo e
        # NÃO além — sem isso o ResourceInfoWidget embedado (que tem
        # QTabWidget+QScrollArea com policy Expanding) puxa o row inteiro
        # pra cima/baixo, deixando uma faixa vazia abaixo do conteúdo.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )

        self._init_ui(label, resource_id)

    def _init_ui(self, label: str, resource_id: Optional[str]) -> None:
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(2)

        # Linha 1: ícone status + label + nome + tipo
        self._status_lbl = QtWidgets.QLabel("⬜")
        self._status_lbl.setStyleSheet("font-size: 14px;")
        self._status_lbl.setMinimumWidth(20)
        layout.addWidget(self._status_lbl, 0, 0)

        lbl = QtWidgets.QLabel(f"{label}:")
        lbl.setStyleSheet("font-size: 11px; color: #2e7d32; font-weight: bold;")
        layout.addWidget(lbl, 0, 1)

        if self._read_only:
            # Display puro — QLabel, visualmente sintoniza com o painel
            # informativo da 3ª coluna.
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
            self._name_field.textChanged.connect(self._on_name_changed)
            self._name_field.editingFinished.connect(self._on_name_changed)
        layout.addWidget(self._name_field, 0, 2)

        type_lbl = QtWidgets.QLabel(f"[{self._resource_type.value}]")
        type_lbl.setStyleSheet("font-size: 9px; color: #888;")
        layout.addWidget(type_lbl, 0, 3)

        # Linha 2: status textual (e.g. "salvo em disco", "em memória",
        # "placeholder"). Container vazio quando não há texto.
        self._summary_container = QtWidgets.QWidget()
        sum_layout = QtWidgets.QVBoxLayout(self._summary_container)
        sum_layout.setContentsMargins(28, 0, 0, 0)
        sum_layout.setSpacing(0)
        self._summary_layout = sum_layout
        layout.addWidget(self._summary_container, 1, 0, 1, 4)

        # Linha 3 (apenas read_only): ResourceInfoWidget(compact=True) sempre
        # expandido. Mostra as mesmas seções do painel de Propriedades —
        # sem toggle, "informativo direto" como combinado.
        if self._read_only:
            from interface.tabs.strategy_manager_v2.left_panel.resource_info_widget import (
                ResourceInfoWidget,
            )
            self._info_widget = ResourceInfoWidget(compact=True)
            # Em compact o widget tem QTabWidget+QScrollArea (policy
            # Expanding default) — sem Maximum vertical o widget puxa o
            # row inteiro e gera faixa vazia.
            self._info_widget.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
            self._info_container = QtWidgets.QWidget()
            info_layout = QtWidgets.QVBoxLayout(self._info_container)
            info_layout.setContentsMargins(28, 4, 4, 4)
            info_layout.setSpacing(0)
            info_layout.addWidget(self._info_widget)
            self._info_container.setVisible(False)
            layout.addWidget(self._info_container, 2, 0, 1, 4)

        layout.setColumnStretch(2, 1)

    def update_state(
        self,
        definition: LRStrategyDefinition,
        instruction_id: str,
        executed: bool,
        context: Optional[StrategyExecutionContextProtocol],
        project_root: Optional[Path] = None,
    ) -> None:
        """Atualiza ícone de status e props embedadas conforme estado do recurso."""
        while self._summary_layout.count():
            item = self._summary_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        resource_id = self._name_field.text().strip()
        if not resource_id:
            self._set_status("⬜", "(sem nome)")
            self._update_info_widget(None, project_root)
            return

        existing = definition.get_resource_by_id(resource_id) if definition else None
        ctx_resource: Optional[StrategyResource] = None
        if context is not None and context.has_resource(resource_id):
            ctx_resource = context.get_resource(resource_id)

        # Prioridade: salvo em disco > em memória (ctx ou direto no resource)
        # > placeholder. ResourceInfoWidget lida com hidratação lazy via
        # artifact_ref quando payload está vazio.
        if existing and existing.has_artifact:
            self._set_status("\U0001f7e2", "salvo em disco")
            self._update_info_widget(existing, project_root)
        elif ctx_resource and ctx_resource.has_payload:
            self._set_status("\U0001f7e1", "em memória")
            self._update_info_widget(ctx_resource, project_root)
        elif existing is not None and existing.has_payload:
            self._set_status("\U0001f7e1", "em memória")
            self._update_info_widget(existing, project_root)
        elif existing is not None:
            if executed:
                self._set_status("\U0001f7e1", "em memória")
            else:
                self._set_status("⬜", "placeholder")
            self._update_info_widget(None, project_root)
        else:
            self._set_status("⬜", "placeholder")
            self._update_info_widget(None, project_root)

    def _update_info_widget(
        self,
        resource: Optional[StrategyResource],
        project_root: Optional[Path],
    ) -> None:
        """Atualiza o ResourceInfoWidget embedado. No-op em modo editável.

        Cache via (resource_id, has_payload, has_artifact, project_root)
        evita re-render em chamadas redundantes — show_resource pode
        recarregar pickles em algumas seções (similarity model, calibration
        model), então re-disparar à toa é caro.
        """
        if self._info_widget is None or self._info_container is None:
            return

        if resource is None:
            if self._last_info_signature is not None:
                self._info_widget.clear()
                self._last_info_signature = None
            self._info_container.setVisible(False)
            return

        signature = (
            getattr(resource, "resource_id", None),
            bool(getattr(resource, "has_payload", False)),
            bool(getattr(resource, "has_artifact", False)),
            str(project_root) if project_root else None,
        )
        if signature == self._last_info_signature:
            self._info_container.setVisible(True)
            return

        try:
            self._info_widget.show_resource(
                resource=resource,
                project_root=project_root,
                definitions=None,
            )
            self._last_info_signature = signature
            self._info_container.setVisible(True)
        except Exception:
            self._info_container.setVisible(False)
            self._last_info_signature = None

    def _set_status(self, icon: str, status_text: str) -> None:
        self._status_lbl.setText(icon)
        status_lbl = QtWidgets.QLabel(status_text)
        status_lbl.setStyleSheet(
            "font-size: 10px; color: #555; font-style: italic;"
        )
        self._summary_layout.addWidget(status_lbl)

    def field_name(self) -> str:
        return self._field_name

    def current_value(self) -> str:
        return self._name_field.text().strip()

    def set_resource_name(self, name: str) -> None:
        """Atualiza o texto exibido sem reemitir name_changed.

        Funciona com QLineEdit (modo editável) e QLabel (modo read_only).
        Usado pelo InstructionResultsPanel para espelhar mudanças do nome
        feitas na col 2 sem disparar loop de signals.
        """
        if self._name_field.text() == name:
            return
        was_blocked = self._name_field.blockSignals(True)
        try:
            self._name_field.setText(name or "")
        finally:
            self._name_field.blockSignals(was_blocked)

    def _on_name_changed(self) -> None:
        self.name_changed.emit(
            self._field_name, self._name_field.text().strip()
        )
