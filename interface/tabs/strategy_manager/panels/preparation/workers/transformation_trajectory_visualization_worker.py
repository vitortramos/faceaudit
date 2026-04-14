from PyQt6 import QtCore
from typing import Any

# Import do Protocolo para tipagem
from faceaudit.protocols.strategy_manager_protocol import StrategyManagerProtocol
# Import do DTO de Dados
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData

class TransformationTrajectoryVisualizationWorker(QtCore.QThread):
    """
    Worker responsável por gerar os dados da visualização 3D em background.
    """
    
    # Sinais alinhados com o Controller:
    finished_success = QtCore.pyqtSignal(object)  # Sucesso (retorna DTO)
    finished_error = QtCore.pyqtSignal(str)       # Erro (retorna mensagem)
    
    # --- CORREÇÃO AQUI ---
    # Renomeado de 'log' para 'progress_log' para parear com a UI e os outros workers
    progress_log = QtCore.pyqtSignal(str)         

    def __init__(self, manager: StrategyManagerProtocol):
        super().__init__()
        self.manager = manager

    def run(self):
        try:
            self.progress_log.emit("⏳ Iniciando cálculo da trajetória de visualização...")
            
            # Chama o Manager -> Controller -> MathService
            viz_data: TransformationTrajectoryData = self.manager.get_visualization_data(
                logger_callback=self.progress_log.emit
            )
            
            self.progress_log.emit("✅ Visualização gerada com sucesso.")
            self.finished_success.emit(viz_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Emite o sinal com o nome correto
            self.finished_error.emit(str(e))