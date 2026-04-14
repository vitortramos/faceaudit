import os
import re
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from .new_project_view import NewProjectView

class NewProjectController(QtCore.QObject):
    def __init__(self, parent_view=None):
        super().__init__()
        self.view = NewProjectView(parent_view)
        
        # Sugere a pasta de Documentos do utilizador como padrão
        self.current_location = Path(os.path.expanduser("~/Documents"))
        self.view.input_location.setText(str(self.current_location))
        
        self._connect_signals()
        self._update_preview()

    def _connect_signals(self):
        self.view.btn_browse.clicked.connect(self._browse_location)
        self.view.input_name.textChanged.connect(self._update_preview)
        self.view.buttons.accepted.connect(self._validate_and_accept)
        self.view.buttons.rejected.connect(self.view.reject)

    def _browse_location(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self.view, "Selecione a diretoria base", str(self.current_location)
        )
        if path:
            self.current_location = Path(path)
            self.view.input_location.setText(str(self.current_location))
            self._update_preview()

    def _sanitize_name(self, name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    def _update_preview(self):
        name = self._sanitize_name(self.view.input_name.text())
        if not name:
            self.view.lbl_preview.setText("Caminho Final: Aguardando nome...")
            return

        final_path = self.current_location / name / f"{name}.fap"
        self.view.lbl_preview.setText(f"Caminho Final:\n{final_path}")

    def _validate_and_accept(self):
        name = self._sanitize_name(self.view.input_name.text())
        if not name:
            QtWidgets.QMessageBox.warning(self.view, "Atenção", "O nome do projeto não pode ficar vazio.")
            return

        project_folder = self.current_location / name
        if project_folder.exists() and any(project_folder.iterdir()):
            reply = QtWidgets.QMessageBox.question(
                self.view, 
                "Aviso de Sobrescrita", 
                f"A pasta '{name}' já existe e não está vazia.\nDeseja criar o projeto aqui mesmo?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if reply == QtWidgets.QMessageBox.StandardButton.No:
                return

        self.view.accept()

    def get_fap_path(self) -> Path:
        name = self._sanitize_name(self.view.input_name.text())
        return self.current_location / name / f"{name}.fap"
        
    def wants_deep_copy(self) -> bool:
        """Retorna se o utilizador quer empacotar as imagens."""
        return self.view.chk_images.isChecked()
        
    def exec(self) -> bool:
        return self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted