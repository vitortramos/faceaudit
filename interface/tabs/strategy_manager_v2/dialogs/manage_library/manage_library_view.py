# interface/tabs/strategy_manager_v2/dialogs/manage_library/manage_library_view.py

from PyQt6 import QtWidgets, QtCore


class ManageLibraryView(QtWidgets.QDialog):
    """
    Dialog para visualizar e remover recursos da biblioteca global.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gerenciar Biblioteca Global")
        self.setMinimumSize(640, 420)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Tabela
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Tipo", "Path", "Hash"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

        # Botões inferiores
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_remove = QtWidgets.QPushButton("🗑 Remover selecionado")
        self.btn_remove.setEnabled(False)
        self.btn_remove.setStyleSheet("""
            QPushButton { border: 1px solid #c62828; border-radius: 4px;
                          background: #ffebee; color: #c62828; padding: 4px 10px; }
            QPushButton:hover { background: #ffcdd2; }
            QPushButton:disabled { color: #aaa; border-color: #ddd; background: #f5f5f5; }
        """)

        btn_close = QtWidgets.QPushButton("Fechar")
        btn_close.clicked.connect(self.accept)

        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.table.selectionModel().selectionChanged.connect(
            lambda: self.btn_remove.setEnabled(bool(self.table.selectedItems()))
        )

    def populate(self, rows: list) -> None:
        """
        Popula a tabela.

        Args:
            rows: Lista de tuplas (tipo_str, path_str, hash_str, ref_object)
        """
        self.table.setRowCount(0)
        for tipo, path, hash_val, ref in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(tipo))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(path))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(hash_val or "—"))
            # Armazena ref no item da coluna 0 para recuperação posterior
            self.table.item(row, 0).setData(QtCore.Qt.ItemDataRole.UserRole, ref)

    def selected_ref(self):
        """Retorna a referência do item selecionado, ou None."""
        items = self.table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        return self.table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
