# interface/tabs/strategy_manager_v2/dialogs/new_strategy/new_strategy_controller.py

from __future__ import annotations

import copy
import json
import uuid
from typing import Dict, List, Optional

from PyQt6 import QtWidgets

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.storage.global_library_storage import GlobalLibraryStorage

from .new_strategy_view import NewStrategyView


class NewStrategyController:
    """
    Controlador do dialog de criação de nova estratégia.

    Uso:
        ctrl = NewStrategyController(
            parent=self.view,
            global_storage=self._global_storage,
        )
        definition = ctrl.exec()
        if definition:
            ...
    """

    def __init__(
        self,
        parent=None,
        global_storage: Optional[GlobalLibraryStorage] = None,
    ) -> None:
        self._parent = parent
        self._global_storage = global_storage
        self._templates: Dict[str, LRStrategyDefinition] = {}

    def exec(self) -> Optional[LRStrategyDefinition]:
        """
        Exibe o dialog e retorna a LRStrategyDefinition criada, ou None se cancelado.
        """
        template_names = self._load_template_names()
        view = NewStrategyView(template_names=template_names, parent=self._parent)

        while True:
            if view.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None

            name = view.get_name()
            if not name:
                QtWidgets.QMessageBox.warning(
                    view, "Campo obrigatório", "O nome da estratégia não pode estar vazio."
                )
                continue

            description = view.get_description() or None

            if view.use_template():
                template_name = view.selected_template_name()
                base = self._templates.get(template_name)
                if base is None:
                    QtWidgets.QMessageBox.warning(
                        view, "Template inválido", f"Template '{template_name}' não encontrado."
                    )
                    continue
                # Template preserva o embedding_column gravado no JSON do
                # template (se houver). O usuário pode ajustar depois no
                # editor, onde o combo lista a interseção real.
                definition = self._clone_from_template(base, name, description)
            else:
                # Estratégia do zero: embedding_column = None (auto-detect).
                # O combo do editor seleciona a única opção comum quando
                # houver, ou o usuário escolhe explicitamente.
                definition = LRStrategyDefinition(
                    strategy_id=str(uuid.uuid4()),
                    name=name,
                    description=description,
                    resources=[],
                )

            return definition

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_template_names(self) -> List[str]:
        """Carrega os templates da GlobalLibraryStorage."""
        self._templates.clear()
        if not self._global_storage:
            return []

        try:
            from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
            refs = self._global_storage.list_artifacts(StrategyArtifactType.STRATEGY_TEMPLATE)
            for ref in refs:
                abs_path = self._global_storage.resolve_artifact(ref)
                if abs_path and abs_path.exists():
                    try:
                        data = json.loads(abs_path.read_text(encoding="utf-8"))
                        defn = LRStrategyDefinition.from_dict(data)
                        if defn:
                            self._templates[defn.name] = defn
                    except Exception:
                        continue
        except Exception:
            pass

        return list(self._templates.keys())

    @staticmethod
    def _clone_from_template(
        template: LRStrategyDefinition,
        new_name: str,
        description: Optional[str],
    ) -> LRStrategyDefinition:
        """
        Clona um template zerando os artifact_refs dos recursos (Pendrive Paradigm).
        A nova estratégia parte com as instruções configuradas mas sem arquivos vinculados.
        """
        cloned = copy.deepcopy(template)
        cloned.strategy_id = str(uuid.uuid4())
        cloned.name = new_name
        cloned.description = description

        for resource in cloned.resources:
            resource.artifact_ref = None

        return cloned
