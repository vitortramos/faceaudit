# interface/tabs/strategy_manager/workers/plda_split_simulation_worker.py
import traceback
from PyQt6 import QtCore
from typing import TYPE_CHECKING

from faceaudit.models.split_simulation_result import SplitSimulationResult

if TYPE_CHECKING:
    from faceaudit.manager.plda_manager import PLDAManager

class PLDASplitSimulationWorker(QtCore.QThread):
    """
    Worker assíncrono para executar a simulação (dry-run) de divisão de dados do PLDA.
    Impede que a interface (UI) congele durante cálculos combinatórios ou leitura de Parquets do disco.
    """
    
    # Sinais de comunicação com a Thread Principal (UI)
    progress_log = QtCore.pyqtSignal(str)
    finished_success = QtCore.pyqtSignal(SplitSimulationResult)
    finished_error = QtCore.pyqtSignal(str)

    def __init__(self, manager: 'PLDAManager', parent=None):
        super().__init__(parent)
        self.manager = manager

    def run(self):
        try:
            self.progress_log.emit("Iniciando Worker de Simulação de Split...")
            
            # Chama o método que acabamos de criar no Manager, 
            # delegando o log_callback para o sinal da interface.
            result: SplitSimulationResult = self.manager.simulate_data_split(
                logger_callback=self.progress_log.emit
            )
            
            # Avalia a resposta do DTO
            if result.success:
                self.progress_log.emit("Simulação finalizada com sucesso.")
                self.finished_success.emit(result)
            else:
                self.progress_log.emit(f"Falha na simulação: {result.error_message}")
                self.finished_error.emit(result.error_message)
                
        except Exception as e:
            traceback.print_exc()
            error_msg = f"Erro crítico não tratado no Worker de Simulação: {str(e)}"
            self.progress_log.emit(error_msg)
            self.finished_error.emit(error_msg)