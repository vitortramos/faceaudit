from pathlib import Path
from PyQt6 import QtCore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faceaudit.manager.plda_manager import PLDAManager

class PLDATrainingWorker(QtCore.QThread):
    """
    Worker Proxy: Move a execução do manager.run_training() para background.
    """
    finished_success = QtCore.pyqtSignal(str) # Retorna caminho do arquivo salvo
    finished_error = QtCore.pyqtSignal(str)
    progress_log = QtCore.pyqtSignal(str) 

    def __init__(self, manager: 'PLDAManager'): # <--- REMOVIDO: output_path
        super().__init__()
        self.manager = manager

    def run(self):
        try:
            # O Manager assume total controle do destino físico (Pendrive Paradigm)
            saved_path = self.manager.run_training(
                logger_callback=self.progress_log.emit # <--- REMOVIDO: output_path
            )
            
            self.finished_success.emit(saved_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_error.emit(str(e))