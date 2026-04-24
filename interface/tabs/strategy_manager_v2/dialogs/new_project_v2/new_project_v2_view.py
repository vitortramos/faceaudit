# interface/tabs/strategy_manager_v2/dialogs/new_project_v2/new_project_v2_view.py

from PyQt6 import QtWidgets, QtCore


class NewProjectV2View(QtWidgets.QDialog):
    """
    Dialog para criação de novo projeto FaceAudit (nova arquitetura).

    Campos: nome, descrição e pasta de destino.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Novo Projeto FaceAudit")
        self.setMinimumWidth(480)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        # Nome
        self.field_name = QtWidgets.QLineEdit()
        self.field_name.setPlaceholderText("Ex: Caso 2024-001")
        form.addRow("Nome do projeto *:", self.field_name)

        # Descrição
        self.field_desc = QtWidgets.QPlainTextEdit()
        self.field_desc.setPlaceholderText("Descrição opcional do projeto…")
        self.field_desc.setMaximumHeight(80)
        form.addRow("Descrição:", self.field_desc)

        # Pasta
        folder_widget = QtWidgets.QWidget()
        folder_row = QtWidgets.QHBoxLayout(folder_widget)
        folder_row.setContentsMargins(0, 0, 0, 0)
        self.field_folder = QtWidgets.QLineEdit()
        self.field_folder.setPlaceholderText("Selecione a pasta de destino…")
        self.field_folder.setReadOnly(True)
        self.btn_browse = QtWidgets.QPushButton("Procurar…")
        folder_row.addWidget(self.field_folder, 1)
        folder_row.addWidget(self.btn_browse)
        form.addRow("Pasta de destino *:", folder_widget)

        layout.addLayout(form)

        # Rodapé
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Criar Projeto")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    # ── Read-only accessors ───────────────────────────────────────────────────

    def get_name(self) -> str:
        return self.field_name.text().strip()

    def get_description(self) -> str:
        return self.field_desc.toPlainText().strip()

    def get_folder(self) -> str:
        return self.field_folder.text().strip()
