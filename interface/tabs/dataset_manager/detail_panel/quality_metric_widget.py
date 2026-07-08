from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QProgressBar
)
from PyQt6.QtCore import Qt

from interface.theme import current_token_fn

class QualityMetricWidget(QWidget):
    """
    Widget visual auxiliar para exibir uma barra de progresso colorida.
    """
    def __init__(self, label, value):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        lbl_name = QLabel(label)
        lbl_name.setFixedWidth(140)
        
        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setTextVisible(False)
        pbar.setFixedHeight(10)
        
        lbl_val = QLabel(f"{value:.2f}" if value is not None else "N/A")
        lbl_val.setFixedWidth(50)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        if value is not None:
            # Lógica visual apenas (cores e escala)
            display_val = value if value > 1.0 else value * 100
            pbar.setValue(int(display_val))
            
            # Cores baseadas no valor (Visual)
            t = current_token_fn()
            if display_val < 40:
                pbar.setStyleSheet("QProgressBar::chunk { background-color: " + t("DANGER") + "; }") # Vermelho
            elif display_val < 70:
                pbar.setStyleSheet("QProgressBar::chunk { background-color: " + t("WARNING") + "; }") # Laranja
            else:
                pbar.setStyleSheet("QProgressBar::chunk { background-color: " + t("SUCCESS") + "; }") # Verde
        else:
            pbar.setValue(0)

        layout.addWidget(lbl_name)
        layout.addWidget(pbar)
        layout.addWidget(lbl_val)