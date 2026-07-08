from PyQt6.QtWidgets import QTableView, QAbstractItemView, QHeaderView
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet

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
        apply_themed_stylesheet(
            self,
            lambda t: "QTableView { border: none; selection-background-color: " + t("PRIMARY") + "; }",
        )

        # --- CABEÇALHO (AQUI ESTÁ A MUDANÇA) ---
        header = self.horizontalHeader()
        
        # FALSE: A última coluna terá o tamanho do seu conteúdo, não da tela inteira.
        header.setStretchLastSection(False) 
        
        # INTERACTIVE: Usuário pode redimensionar qualquer coluna manualmente
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # PR 20 (Tema 4.1) — drag-reorder: arrastar o header reordena colunas.
        # O controller escuta `sectionMoved` para persistir a nova ordem em
        # `column_specs` (mantém Python dict insertion order), de modo que
        # save → reopen do parquet preserva a escolha do usuário.
        header.setSectionsMovable(True)