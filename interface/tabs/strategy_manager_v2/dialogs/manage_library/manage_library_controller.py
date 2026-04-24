# interface/tabs/strategy_manager_v2/dialogs/manage_library/manage_library_controller.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets

from faceaudit.storage.global_library_storage import GlobalLibraryStorage

from .manage_library_view import ManageLibraryView


class ManageLibraryController:
    """
    Controlador do dialog de gestão da biblioteca global.

    Uso:
        ctrl = ManageLibraryController(parent=view, global_storage=storage)
        ctrl.exec()
    """

    def __init__(
        self,
        parent=None,
        global_storage: Optional[GlobalLibraryStorage] = None,
    ) -> None:
        self._parent = parent
        self._storage = global_storage

    def exec(self) -> None:
        if not self._storage:
            QtWidgets.QMessageBox.information(
                self._parent,
                "Biblioteca não configurada",
                "Nenhuma biblioteca global configurada. "
                "Verifique o config.json (paths.library).",
            )
            return

        view = ManageLibraryView(parent=self._parent)
        self._populate(view)
        view.btn_remove.clicked.connect(lambda: self._on_remove(view))
        view.exec()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _populate(self, view: ManageLibraryView) -> None:
        try:
            refs = self._storage.list_artifacts()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._parent, "Erro", f"Não foi possível listar a biblioteca:\n{exc}"
            )
            return

        rows = []
        for ref in refs:
            rows.append((
                ref.artifact_type.value if ref.artifact_type else "—",
                ref.path,
                (ref.hash or "—")[:16] + "…" if ref.hash and len(ref.hash) > 16 else (ref.hash or "—"),
                ref,
            ))
        view.populate(rows)

    def _on_remove(self, view: ManageLibraryView) -> None:
        ref = view.selected_ref()
        if not ref:
            return

        confirm = QtWidgets.QMessageBox.question(
            view,
            "Confirmar remoção",
            f"Remover o arquivo:\n{ref.path}\n\nEssa ação não pode ser desfeita.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        try:
            abs_path = self._storage.resolve_artifact(ref)
            if abs_path and abs_path.exists():
                abs_path.unlink()
            self._populate(view)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                view, "Erro ao remover", f"Não foi possível remover o arquivo:\n{exc}"
            )
