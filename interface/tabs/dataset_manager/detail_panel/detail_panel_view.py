from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTabWidget, QScrollArea, QProgressBar, QFormLayout
)
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet
from .quality_metric_widget import QualityMetricWidget


class DetailPanelView(QWidget):
    """
    View do Painel de Detalhes.
    Gerencia as abas (Geral, Qualidade) e a exibição da imagem.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Área da Imagem
        self.lbl_image = QLabel("Selecione uma face")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumHeight(150)
        self.lbl_image.setMaximumHeight(280)
        apply_themed_stylesheet(
            self.lbl_image,
            lambda t: "background-color: " + t("SURFACE_ALT")
            + "; border-bottom: 1px solid " + t("BORDER") + ";",
        )
        layout.addWidget(self.lbl_image)

        # Abas
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Aba 1: Geral
        self.tab_general = QWidget()
        self.scroll_general = QScrollArea() 
        self.scroll_general.setWidgetResizable(True)
        self.scroll_general.setWidget(self.tab_general)
        
        self.form_general = QFormLayout(self.tab_general)
        self.form_general.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_general.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        self.tabs.addTab(self.scroll_general, "Geral / Bio")
        
        # Aba 2: Qualidade ISO (OFIQ)
        self.scroll_ofiq = QScrollArea()
        self.scroll_ofiq.setWidgetResizable(True)
        self.container_ofiq = QWidget()
        
        self.layout_ofiq = QVBoxLayout(self.container_ofiq)
        self.layout_ofiq.setContentsMargins(5, 5, 5, 5)
        self.layout_ofiq.setSpacing(2)
        
        self.scroll_ofiq.setWidget(self.container_ofiq)
        self.tabs.addTab(self.scroll_ofiq, "Qualidade ISO")

    # --- API de Atualização Visual (Chamada pelo Controller) ---

    def set_image(self, pixmap: QtGui.QPixmap):
        """Atualiza o label com a imagem redimensionada."""
        if not pixmap.isNull():
            w = self.lbl_image.width()
            # Escala suave mantendo aspect ratio
            scaled = pixmap.scaled(w, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
        else:
            self.reset_image_placeholder("Imagem inválida")

    def reset_image_placeholder(self, text="Selecione uma face"):
        """Reseta a área de imagem para o estado padrão (texto)."""
        self.lbl_image.clear()
        self.lbl_image.setText(text)

    def clear_general_info(self):
        """Remove todas as linhas do formulário geral."""
        while self.form_general.count():
            item = self.form_general.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_general_info(self, label: str, value: str):
        """Adiciona uma linha chave-valor na aba Geral."""
        lbl = QLabel(str(value))
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.form_general.addRow(f"<b>{label}:</b>", lbl)

    def clear_quality_metrics(self):
        """Remove todas as métricas da aba de qualidade."""
        while self.layout_ofiq.count():
            item = self.layout_ofiq.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_quality_metric(self, name: str, value: float):
        """Adiciona uma barra de qualidade."""
        self.layout_ofiq.addWidget(QualityMetricWidget(name, value))
        
    def add_quality_label(self, text: str):
        """Adiciona um texto simples na aba de qualidade (ex: avisos)."""
        self.layout_ofiq.addWidget(QLabel(text))
        
    def add_quality_stretch(self):
        """Empurra o conteúdo para o topo."""
        self.layout_ofiq.addStretch()