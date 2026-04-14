# interface/tabs/strategy_manager/panels/preparation/preparation_view.py
from PyQt6 import QtWidgets, QtCore, QtGui

class PreparationView(QtWidgets.QWidget):
    """
    View da Coluna de Preparação (Coluna 1).
    Atua como um container passivo para os cards de cada estratégia.
    A criação e gestão de estratégias ocorre exclusivamente via Toolbar.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        # Layout Principal: Vertical sem margens desnecessárias
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # A repetição do cabeçalho ("1. PREPARAÇÃO E PIPELINES") foi removida daqui, 
        # pois o StrategyManagerView (pai) já possui o título da coluna.

        # 1. Área de Scroll (Onde os Cards serão injetados)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet("QScrollArea { background-color: #ffffff; }")
        
        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.cards_layout.setSpacing(15)
        self.cards_layout.setContentsMargins(10, 15, 10, 15)
        
        self.scroll.setWidget(self.cards_container)
        self.main_layout.addWidget(self.scroll)

        # 2. Estado Vazio (Placeholder)
        self._setup_empty_state()

    def _setup_empty_state(self):
        """Widget exibido quando não há cards no projeto atual."""
        self.empty_widget = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self.empty_widget)
        empty_layout.setContentsMargins(0, 50, 0, 0)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignHCenter)
        
        icon_label = QtWidgets.QLabel("📥")
        icon_label.setStyleSheet("font-size: 32px; color: #dee2e6;")
        icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        text_label = QtWidgets.QLabel("Nenhum experimento.\nUse a barra superior para começar.")
        text_label.setStyleSheet("color: #adb5bd; font-size: 12px; line-height: 1.4;")
        text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        empty_layout.addWidget(icon_label)
        empty_layout.addSpacing(10)
        empty_layout.addWidget(text_label)
        
        self.cards_layout.addWidget(self.empty_widget)

    # --- MÉTODOS DE MANIPULAÇÃO (Chamados pelo Controller) ---

    def add_card_widget(self, card_widget: QtWidgets.QWidget):
        """Injeta um card de estratégia na lista visual."""
        if self.empty_widget.isVisible():
            self.empty_widget.hide()
        self.cards_layout.addWidget(card_widget)
    
    def remove_card_widget(self, card_widget: QtWidgets.QWidget):
        """Remove um card específico da visualização."""
        self.cards_layout.removeWidget(card_widget)
        card_widget.deleteLater()
        
        # Se removeu o último, volta o placeholder
        if self.cards_layout.count() <= 1: 
            self.empty_widget.show()
    
    def clear_cards(self):
        """Limpa todos os cards (ex: ao fechar projeto)."""
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w and w != self.empty_widget:
                w.deleteLater()
        self.empty_widget.show()