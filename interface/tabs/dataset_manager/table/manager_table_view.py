from PyQt6.QtWidgets import QTableView, QAbstractItemView, QHeaderView
from PyQt6.QtCore import Qt

class ManagerTableView(QTableView):
    """
    View da Tabela de Dados.
    Configurada para seleção de células e performance.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        # Comportamento de Seleção (Células, não linhas inteiras, para permitir edição fina)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Menu de Contexto (Controller irá conectar)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        
        # Ordenação
        self.setSortingEnabled(True)
        
        # Estilo Visual
        self.alternatingRowColors = True
        self.setAlternatingRowColors(True)
        self.setShowGrid(False) # Grid falso deixa mais limpo (estilo moderno)
        self.setStyleSheet("QTableView { border: none; selection-background-color: #0078d7; }")

        # --- CABEÇALHO (AQUI ESTÁ A MUDANÇA) ---
        header = self.horizontalHeader()
        
        # FALSE: A última coluna terá o tamanho do seu conteúdo, não da tela inteira.
        header.setStretchLastSection(False) 
        
        # INTERACTIVE: Usuário pode redimensionar qualquer coluna manualmente
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)