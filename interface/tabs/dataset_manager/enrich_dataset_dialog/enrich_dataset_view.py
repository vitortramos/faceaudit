from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, 
    QDialogButtonBox, QGroupBox, QWidget
)

class EnrichDatasetView(QDialog):
    """
    Janela de seleção de opções de Enriquecimento.
    Exibe checkboxes para cada pipeline (FF, OFIQ, SER-FIQ) junto com
    um status visual do quanto aquele dataset já possui desses dados.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enriquecer Dataset")
        self.resize(450, 350)
        
        # Armazena referências para acesso fácil pelo Controller
        self.checkboxes = {}    # { 'ofiq': QCheckBox, ... }
        self.status_labels = {} # { 'ofiq': QLabel, ... }
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # 1. Texto Explicativo
        lbl_info = QLabel(
            "Selecione quais processos deseja executar nas imagens deste dataset.\n"
            "O sistema analisou os dados atuais para sugerir o que falta."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(lbl_info)
        
        # 2. Grupo de Opções
        group = QGroupBox("Processadores Disponíveis")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(15)

        # Adiciona as linhas (Chave, Título, Descrição)
        self._add_option(group_layout, "ff", "Forensic Face", "Biometria, Detecção, Pose, Idade, Gênero")
        self._add_option(group_layout, "ofiq", "OFIQ (Qualidade ISO)", "Métricas vetoriais de qualidade (ISO/IEC 29794-5)")
        self._add_option(group_layout, "serfiq", "SER-FIQ (Robustez)", "Score estocástico de estabilidade do embedding")
        
        group_layout.addStretch()
        main_layout.addWidget(group)

        # 3. Botões
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Iniciar Processamento")
        
        main_layout.addWidget(self.button_box)

    def _add_option(self, layout, key, title, subtitle):
        """Cria uma linha com: Checkbox (Título+Subtítulo) + Label de Status à direita."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        
        # Coluna da Esquerda (Checkbox e Textos)
        chk = QCheckBox()
        self.checkboxes[key] = chk
        
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0,0,0,0)
        text_layout.setSpacing(2)
        
        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("font-size: 10px; color: #888;")
        
        text_layout.addWidget(lbl_title)
        text_layout.addWidget(lbl_sub)
        
        # Coluna da Direita (Status)
        lbl_status = QLabel("Carregando...")
        lbl_status.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        lbl_status.setStyleSheet("font-weight: bold; color: #999;")
        self.status_labels[key] = lbl_status
        
        # Montagem
        row.addWidget(chk)
        row.addWidget(text_container, stretch=1) # O texto ocupa o espaço livre
        row.addWidget(lbl_status)
        
        # Ao clicar no texto, marca o checkbox (melhor UX)
        # (Opcional, mas requer subclassing ou event filter, vamos deixar simples por enquanto)
        
        layout.addWidget(container)

    # --- API para o Controller ---

    def set_status(self, key: str, text: str, color: str):
        """Atualiza o texto e cor do status de uma opção."""
        if key in self.status_labels:
            self.status_labels[key].setText(text)
            self.status_labels[key].setStyleSheet(f"font-weight: bold; color: {color};")

    def set_checked(self, key: str, checked: bool):
        if key in self.checkboxes:
            self.checkboxes[key].setChecked(checked)

    def is_checked(self, key: str) -> bool:
        if key in self.checkboxes:
            return self.checkboxes[key].isChecked()
        return False