from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt

from interface.theme import current_token_fn

class StatsDialogView(QDialog):
    """
    View do Diálogo de Estatísticas.
    Responsável apenas por renderizar as abas e tabelas com dados pré-processados.
    """
    def __init__(self, title_suffix="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Estatísticas do Dataset - {title_suffix}")
        self.resize(600, 700)
        
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Botão Fechar padrão
        btn_close = QtWidgets.QPushButton("Fechar")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

    def add_statistics_tab(self, title: str, icon: str, total_unique: int, data_rows: list):
        """
        Cria uma nova aba com uma tabela.
        
        Args:
            title: Título da aba (ex: "Gênero")
            icon: Ícone da aba
            total_unique: Número total de itens únicos para exibir no topo
            data_rows: Lista de tuplas (label, count, percentage_float)
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Label de Resumo
        layout.addWidget(QLabel(f"Total de {title} Únicos: <b>{total_unique}</b>"))

        # Configuração da Tabela
        table = QTableWidget()
        table.setColumnCount(3) 
        table.setHorizontalHeaderLabels([title, "Quantidade", "% Total"])
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(table)
        self.tabs.addTab(tab, f"{icon} {title}")

        # População da Tabela
        if not data_rows:
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem("Nenhum dado disponível"))
            return

        table.setRowCount(len(data_rows))
        
        for row_idx, (label, count, pct) in enumerate(data_rows):
            # Coluna 0: Label
            table.setItem(row_idx, 0, QTableWidgetItem(str(label)))
            
            # Coluna 1: Quantidade (Numérico para ordenação correta)
            item_count = QTableWidgetItem()
            item_count.setData(Qt.ItemDataRole.DisplayRole, int(count))
            item_count.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 1, item_count)

            # Coluna 2: Porcentagem
            item_pct = QTableWidgetItem(f"{pct:.1f}%")
            item_pct.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            # Lógica Visual: Destacar valores muito baixos em vermelho
            if pct < 1.0:
                item_pct.setForeground(QtGui.QColor(current_token_fn()("DANGER")))
            
            table.setItem(row_idx, 2, item_pct)