# interface/tabs/strategy_manager_v2/dialogs/new_project_v2/new_project_v2_controller.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets

from faceaudit.services.project_service import FaceAuditProject, FaceAuditProjectService

from .new_project_v2_view import NewProjectV2View


class NewProjectV2Controller:
    """
    Controlador do dialog de criação de novo projeto FaceAudit.

    Uso:
        ctrl = NewProjectV2Controller(parent=self.view)
        project = ctrl.exec()
        if project:
            ...  # projeto criado com sucesso
    """

    def __init__(self, parent=None, initial_folder: Optional[Path] = None) -> None:
        self._parent = parent
        self._initial_folder = initial_folder
        self._project: Optional[FaceAuditProject] = None

    def exec(self) -> Optional[FaceAuditProject]:
        """
        Exibe o dialog e cria o projeto se o usuário confirmar.

        Returns:
            FaceAuditProject recém-criado, ou None se cancelado.
        """
        view = NewProjectV2View(parent=self._parent)

        if self._initial_folder:
            view.field_folder.setText(str(self._initial_folder))

        view.btn_browse.clicked.connect(lambda: self._browse_folder(view))

        while True:
            if view.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return None

            name = view.get_name()
            folder = view.get_folder()

            if not name:
                QtWidgets.QMessageBox.warning(
                    view, "Campo obrigatório", "O nome do projeto não pode estar vazio."
                )
                continue

            if not folder:
                QtWidgets.QMessageBox.warning(
                    view, "Campo obrigatório", "Selecione uma pasta de destino."
                )
                continue

            project_folder = Path(folder) / name

            try:
                project = FaceAuditProjectService.create(
                    folder=project_folder,
                    name=name,
                    description=view.get_description(),
                )
                return project

            except FileExistsError:
                QtWidgets.QMessageBox.warning(
                    view,
                    "Projeto já existe",
                    f"Já existe um projeto FaceAudit em:\n{project_folder}\n\n"
                    "Escolha outro nome ou pasta.",
                )

            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    view,
                    "Erro ao criar projeto",
                    f"Não foi possível criar o projeto:\n{exc}",
                )
                return None

    @staticmethod
    def _browse_folder(view: NewProjectV2View) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            view,
            "Selecionar pasta de destino",
            view.get_folder() or str(Path.home()),
        )
        if folder:
            view.field_folder.setText(folder)
