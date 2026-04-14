from PyQt6 import QtCore, QtWidgets
from typing import Tuple

from .save_project_view import SaveProjectView

class SaveProjectController(QtCore.QObject):
    """
    Controlador do Diálogo de Salvamento.
    Captura as escolhas do usuário sobre Deep Copy e Garbage Collection.
    """
    def __init__(self, parent_view=None):
        super().__init__()
        self.view = SaveProjectView(parent_view)
        self._connect_signals()

    def _connect_signals(self):
        self.view.buttons.accepted.connect(self.view.accept)
        self.view.buttons.rejected.connect(self.view.reject)

    def exec(self) -> bool:
        """Exibe o diálogo modalmente e retorna True se o usuário confirmou."""
        return self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted

    def get_options(self) -> Tuple[bool, bool]:
        """
        Retorna as opções selecionadas.
        Returns:
            Tuple[bool, bool]: (wants_deep_copy, wants_cleanup)
        """
        return self.view.chk_images.isChecked(), self.view.chk_clean.isChecked()