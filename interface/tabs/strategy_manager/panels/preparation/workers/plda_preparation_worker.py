# interface/tabs/strategy_manager/panels/preparation/workers/plda_preparation_worker.py
from PyQt6 import QtCore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faceaudit.manager.plda_manager import PLDAManager

class PLDAPreparationWorker(QtCore.QThread):
    """
    Worker Proxy: Move a execução do manager.run_preparation() para background.
    """
    # Retorna (final_dim, n_src, n_tgt) ou dict de info
    finished_success = QtCore.pyqtSignal(dict) 
    finished_error = QtCore.pyqtSignal(str)
    progress_log = QtCore.pyqtSignal(str) 

    def __init__(self, manager: 'PLDAManager'):
        super().__init__()
        self.manager = manager

    def run(self):
        try:
            # O Manager faz todo o trabalho pesado agora.
            # Passamos o self.progress_log.emit como callback de log.
            result_info = self.manager.run_preparation(logger_callback=self.progress_log.emit)
            
            self.finished_success.emit(result_info)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_error.emit(str(e))