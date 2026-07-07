from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QFrame
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet

class DatasetManagerView(QWidget):
    """
    View Principal (Container) do Gerenciador de Datasets.
    Versão Final de Produção.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        # Layout Mestre
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Container da Toolbar
        self.toolbar_container = QWidget()
        # Define uma altura mínima para garantir que a toolbar não suma
        self.toolbar_container.setMinimumHeight(50) 
        
        self.toolbar_layout = QVBoxLayout(self.toolbar_container)
        self.toolbar_layout.setContentsMargins(0, 0, 0, 0)
        
        main_layout.addWidget(self.toolbar_container)

        # 2. Splitter Central
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        apply_themed_stylesheet(
            self.splitter,
            lambda t: "QSplitter::handle { background-color: " + t("BORDER") + "; }",
        )
        
        main_layout.addWidget(self.splitter, 1)

        # 2.1 Container Tabela
        self.table_container = QWidget()
        self.table_layout = QVBoxLayout(self.table_container)
        self.table_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter.addWidget(self.table_container)

        # 2.2 Container Detalhes
        self.detail_container = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_container)
        self.detail_layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter.addWidget(self.detail_container)

        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)

        # Impede que o painel de detalhes (direita) seja colapsado a 0 quando
        # a tabela fica larga (datasets com muitas colunas, ex.: video_*).
        # Sem isso, o splitter dava toda a largura à tabela e o painel
        # (imagem + infos) "sumia". Largura mínima + não-colapsável + tamanhos
        # iniciais garantem que ele sempre apareça.
        self.detail_container.setMinimumWidth(260)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setSizes([720, 320])