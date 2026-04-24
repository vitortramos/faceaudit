# interface/tabs/strategy_manager_v2/dialogs/new_strategy/new_strategy_view.py

from typing import List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.enums.embedding_type import EmbeddingType


class NewStrategyView(QtWidgets.QDialog):
    """
    Dialog para criação de nova estratégia — do zero ou a partir de template.
    """

    def __init__(self, template_names: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nova Estratégia")
        self.setMinimumWidth(440)
        self.setModal(True)
        self._init_ui(template_names)

    def _init_ui(self, template_names: List[str]) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        self.field_name = QtWidgets.QLineEdit()
        self.field_name.setPlaceholderText("Ex: KDE + LDA")
        form.addRow("Nome *:", self.field_name)

        self.field_desc = QtWidgets.QPlainTextEdit()
        self.field_desc.setPlaceholderText("Descrição opcional…")
        self.field_desc.setMaximumHeight(60)
        form.addRow("Descrição:", self.field_desc)

        self.combo_embedding = QtWidgets.QComboBox()
        self.combo_embedding.addItem("(auto-detect)", userData=None)
        for emb in EmbeddingType:
            self.combo_embedding.addItem(
                f"{emb.display_name}  ({emb.dimensions}d)", userData=emb
            )
        form.addRow("Embedding:", self.combo_embedding)

        layout.addLayout(form)

        # Grupo: origem
        group = QtWidgets.QGroupBox("Origem")
        group_layout = QtWidgets.QVBoxLayout(group)

        self.radio_scratch = QtWidgets.QRadioButton("Criar do zero")
        self.radio_scratch.setChecked(True)
        self.radio_template = QtWidgets.QRadioButton("A partir de template")
        group_layout.addWidget(self.radio_scratch)
        group_layout.addWidget(self.radio_template)

        self.combo_template = QtWidgets.QComboBox()
        self.combo_template.addItems(template_names if template_names else ["(nenhum template disponível)"])
        self.combo_template.setEnabled(False)
        group_layout.addWidget(self.combo_template)

        self.radio_template.toggled.connect(self.combo_template.setEnabled)

        layout.addWidget(group)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Criar")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_name(self) -> str:
        return self.field_name.text().strip()

    def get_description(self) -> str:
        return self.field_desc.toPlainText().strip()

    def use_template(self) -> bool:
        return self.radio_template.isChecked()

    def selected_template_name(self) -> str:
        return self.combo_template.currentText()

    def get_embedding_type(self) -> Optional[EmbeddingType]:
        return self.combo_embedding.currentData()
