from PyQt6 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
# Import necessário para projeção 3D
from mpl_toolkits.mplot3d import Axes3D 

class TransformationTrajectoryViewerView(QtWidgets.QDialog):
    """
    Janela de visualização 3D (Dual Plot).
    Mostra dois gráficos lado a lado:
    - Esquerda: Source (Evolução da Adaptação)
    - Direita: Target (Referência/Galeria)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auditoria Visual de Trajetória (Source vs Target)")
        self.resize(1300, 800) # Aumentei um pouco para garantir conforto
        self.setStyleSheet("background-color: white;")
        
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # 1. Header (Apenas Título agora)
        header_layout = QtWidgets.QHBoxLayout()
        
        self.lbl_title = QtWidgets.QLabel("Visualização do Espaço Latente")
        self.lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #212529;")
        
        # Removi o lbl_info daqui. Ele ficava escondido no canto.
        
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # 2. Área dos Gráficos (Matplotlib)
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        # Cria 2 Subplots 3D lado a lado
        self.ax1 = self.figure.add_subplot(121, projection='3d')
        self.ax2 = self.figure.add_subplot(122, projection='3d')
        
        # Ajusta o espaçamento
        self.figure.subplots_adjust(left=0.02, right=0.98, wspace=0.05, hspace=0)
        
        # Barra de Ferramentas
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, stretch=1)

        # 3. Controles (Slider e Informações Detalhadas)
        control_group = QtWidgets.QGroupBox("Auditoria do Passo")
        control_group.setStyleSheet("""
            QGroupBox { font-weight: bold; margin-top: 10px; border: 1px solid #ddd; border-radius: 5px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        control_layout = QtWidgets.QVBoxLayout(control_group)
        
        # A) Nome do Passo (Destaque Azul)
        self.lbl_step_name = QtWidgets.QLabel("-")
        self.lbl_step_name.setStyleSheet("font-weight: 900; color: #0d6efd; font-size: 16px; margin-bottom: 2px;")
        self.lbl_step_name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        # B) Métricas (O novo lugar das informações)
        self.lbl_metrics = QtWidgets.QLabel("Aguardando dados...")
        self.lbl_metrics.setStyleSheet("color: #495057; font-size: 13px; font-family: Consolas, monospace; margin-bottom: 10px;")
        self.lbl_metrics.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_metrics.setWordWrap(True) # Permite que o texto quebre linha se for muito longo
        
        # C) Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0) 
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        
        control_layout.addWidget(self.lbl_step_name)
        control_layout.addWidget(self.lbl_metrics) # Adicionado aqui
        control_layout.addWidget(self.slider)
        
        layout.addWidget(control_group)
        
        # Botão Fechar
        self.btn_close = QtWidgets.QPushButton("Fechar Visualização")
        self.btn_close.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.btn_close.setFixedWidth(200)
        self.btn_close.setStyleSheet("""
            QPushButton { background-color: #6c757d; color: white; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #5a6268; }
        """)
        self.btn_close.clicked.connect(self.accept)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_step_label(self, text: str):
        self.lbl_step_name.setText(text)

    def set_step_info(self, text: str):
        # Agora atualiza o lbl_metrics que está centralizado e visível
        self.lbl_metrics.setText(text)