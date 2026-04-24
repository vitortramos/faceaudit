# interface/tabs/strategy_manager_v2/strategy_editor/produced_resources_widget.py

from __future__ import annotations

from typing import List, Optional, Set

from PyQt6 import QtWidgets, QtCore

from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class ProducedResourcesWidget(QtWidgets.QWidget):
    """
    Exibe os recursos produzidos pelas instruções de uma fase.

    Cada recurso mostra:
    - Ícone de estado: placeholder | em memória | salvo em disco
    - Nome (resource_id)
    - Tipo (StrategyResourceType)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(2)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ddd;")
        layout.addWidget(sep)

        header = QtWidgets.QLabel("Recursos Produzidos")
        header.setStyleSheet("font-weight: bold; font-size: 10px; color: #555;")
        layout.addWidget(header)

        self._list = QtWidgets.QListWidget()
        self._list.setStyleSheet("font-size: 10px;")
        self._list.setMaximumHeight(90)
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        layout.addWidget(self._list)

    def refresh(
        self,
        instructions: List[StrategyInstructionDefinition],
        definition: LRStrategyDefinition,
        executed_ids: Set[str],
        context: Optional[StrategyExecutionContextProtocol] = None,
    ) -> None:
        """
        Atualiza a lista de recursos produzidos pelas instruções da fase.

        Estados visuais:
        - Cinza: placeholder (declarado mas não executado)
        - Amarelo: em memória (executado, payload presente no contexto)
        - Verde: salvo em disco (artifact_ref presente na definição)
        """
        self._list.clear()

        for instr in instructions:
            placeholders = instr.get_output_resource_placeholders()
            for placeholder in placeholders:
                rid = placeholder.resource_id
                rtype = placeholder.resource_type.value

                # Determinar estado
                existing = definition.get_resource_by_id(rid)
                if existing and existing.has_artifact:
                    icon = "\U0001f7e2"  # green circle
                    status = "salvo"
                elif context and context.has_resource(rid):
                    ctx_resource = context.get_resource(rid)
                    if ctx_resource and ctx_resource.has_payload:
                        icon = "\U0001f7e1"  # yellow circle
                        status = "em memória"
                    else:
                        icon = "\u2b1c"  # white square
                        status = "placeholder"
                elif instr.instruction_id in executed_ids:
                    icon = "\U0001f7e1"  # yellow circle
                    status = "em memória"
                else:
                    icon = "\u2b1c"  # white square
                    status = "placeholder"

                self._list.addItem(f"{icon} {rid} [{rtype}] — {status}")

        if self._list.count() == 0:
            self._list.addItem("(nenhum recurso produzido nesta fase)")

    def clear(self) -> None:
        self._list.clear()
