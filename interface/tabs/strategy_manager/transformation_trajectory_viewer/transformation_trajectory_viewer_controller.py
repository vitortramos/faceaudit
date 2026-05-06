import numpy as np
from PyQt6 import QtCore, QtWidgets

from .transformation_trajectory_viewer_view import TransformationTrajectoryViewerView
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData
from faceaudit.services.embedding_math_service import EmbeddingMathService

# Importamos o novo pintor!
from faceaudit.services.trajectory_plotter_service import TrajectoryPlotterService

class TransformationTrajectoryViewerController(QtCore.QObject):
    """
    Controlador para a visualização 3D.
    Agora delega o desenho pesado para o TrajectoryPlotterService.
    """
    
    def __init__(self, 
                 data: TransformationTrajectoryData, 
                 math_service: EmbeddingMathService = None, 
                 parent=None):
        super().__init__(parent)
        
        self.data = data
        self.view = TransformationTrajectoryViewerView(parent)
        self.math_service = math_service or EmbeddingMathService()
        self.steps = self.data.steps 
        self.current_step_idx = 0
        
        # Instancia nosso serviço pintor
        self.plotter_service = TrajectoryPlotterService(self.math_service)

        self._setup_view()
        self._connect_signals()
        self._update_plot()

    def show(self):
        self.view.exec()

    def _setup_view(self):
        n_steps = len(self.steps)
        if n_steps > 0:
            self.view.slider.setMinimum(0)
            self.view.slider.setMaximum(n_steps - 1)
            self.view.slider.setSingleStep(1)
            self.view.slider.setPageStep(1)
            self.view.slider.setValue(0)
        else:
            self.view.slider.setMaximum(0)
            
    def _connect_signals(self):
        self.view.slider.valueChanged.connect(self._on_slider_changed)

    def _on_slider_changed(self, value):
        self.current_step_idx = value
        self._update_plot()

    def _update_plot(self):
        if not self.steps: return

        step = self.steps[self.current_step_idx]
        
        # O plotter limpa os eixos, desenha tudo e nos devolve o texto de auditoria!
        audit_text = self.plotter_service.plot_step(
            ax_source=self.view.ax1, 
            ax_target=self.view.ax2, 
            step=step
        )
        
        # Atualiza apenas os elementos nativos da UI do PyQt
        self.view.set_step_label(f"Passo {self.current_step_idx + 1} de {len(self.steps)}")
        self.view.set_step_info(audit_text.replace("\n", " | ")) 
        self.view.canvas.draw()