from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMessageBox

# Importa a view local
from .split_dataset_dialog_view import SplitDatasetDialogView

class SplitDatasetDialogController(QtCore.QObject):
    """
    Controlador para a janela de Divisão de Datasets.
    """
    def __init__(self, original_dataset, parent_view=None):
        super().__init__()
        self.dataset = original_dataset
        self.view = SplitDatasetDialogView(parent_view)
        
        self._init_data()
        self._connect_signals()

    def _init_data(self):
        """Popula os campos da View com os dados do dataset atual."""
        # Info
        qtd_imgs = len(self.dataset)
        qtd_ids = self.dataset.number_of_identities
        self.view.lbl_dataset_info.setText(
            f"Nome: {self.dataset.name}\nTotal de Imagens: {qtd_imgs}\nIdentidades Únicas: {qtd_ids}"
        )

        # Nomes Padrão
        base_name = self.dataset.name if self.dataset.name else "Dataset"
        self.view.txt_name_a.setText(f"{base_name}_A")
        self.view.txt_name_b.setText(f"{base_name}_B")

        # Popula a Lista Manual (Lado A começa com todas as identidades)
        identities = self.dataset.unique_identities
        self.view.list_a.addItems(identities)

    def _connect_signals(self):
        # Alternar abas do StackedWidget
        self.view.cmb_mode.currentIndexChanged.connect(self.view.stack.setCurrentIndex)

        # Mover itens nas listas
        self.view.btn_move_right.clicked.connect(self._move_to_b)
        self.view.btn_move_left.clicked.connect(self._move_to_a)
        
        # Duplo clique também move itens
        self.view.list_a.itemDoubleClicked.connect(self._move_to_b_item)
        self.view.list_b.itemDoubleClicked.connect(self._move_to_a_item)

        # Botões finais
        self.view.button_box.accepted.connect(self._validate_and_accept)
        self.view.button_box.rejected.connect(self.view.reject)

    def _move_to_b(self):
        items = self.view.list_a.selectedItems()
        for item in items:
            self.view.list_a.takeItem(self.view.list_a.row(item))
            self.view.list_b.addItem(item)

    def _move_to_a(self):
        items = self.view.list_b.selectedItems()
        for item in items:
            self.view.list_b.takeItem(self.view.list_b.row(item))
            self.view.list_a.addItem(item)

    def _move_to_b_item(self, item):
        self.view.list_a.takeItem(self.view.list_a.row(item))
        self.view.list_b.addItem(item)

    def _move_to_a_item(self, item):
        self.view.list_b.takeItem(self.view.list_b.row(item))
        self.view.list_a.addItem(item)

    def _validate_and_accept(self):
        """Valida as entradas antes de fechar a janela."""
        name_a = self.view.txt_name_a.text().strip()
        name_b = self.view.txt_name_b.text().strip()

        if not name_a or not name_b:
            QMessageBox.warning(self.view, "Erro de Validação", "Os nomes dos datasets de saída não podem estar vazios.")
            return

        if name_a == name_b:
            QMessageBox.warning(self.view, "Erro de Validação", "Os nomes dos datasets devem ser diferentes.")
            return

        # Validação do Modo Manual
        if self.view.cmb_mode.currentIndex() == 1:
            if self.view.list_a.count() == 0 or self.view.list_b.count() == 0:
                QMessageBox.warning(
                    self.view, 
                    "Aviso", 
                    "No modo manual, ambas as listas de identidades devem conter pelo menos um indivíduo."
                )
                return

        self.view.accept()

    def exec(self):
        return self.view.exec()

    def get_split_config(self) -> dict:
        """
        Retorna um dicionário com os parâmetros escolhidos pelo usuário.
        """
        mode_idx = self.view.cmb_mode.currentIndex()
        
        # Pega as identidades do Dataset B
        ids_for_b = set()
        for i in range(self.view.list_b.count()):
            ids_for_b.add(self.view.list_b.item(i).text())

        return {
            "name_a": self.view.txt_name_a.text().strip(),
            "name_b": self.view.txt_name_b.text().strip(),
            "mode": "ratio" if mode_idx == 0 else "manual",
            "ratio": self.view.spin_ratio.value(),
            "manual_ids_b": ids_for_b,
            "post_action": self.view.cmb_action.currentIndex() # 0: Manter, 1: Load A, 2: Load B
        }