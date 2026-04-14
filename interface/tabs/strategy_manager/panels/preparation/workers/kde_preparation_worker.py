from PyQt6 import QtCore
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faceaudit.manager.kde_manager import KDEManager

class KDEPreparationWorker(QtCore.QThread):
    """
    Worker Proxy: Move a execução do manager.run_preparation() do KDE para background.
    Garante que a agregação pesada de embeddings não trave a interface.
    """
    finished_success = QtCore.pyqtSignal(dict) 
    finished_error = QtCore.pyqtSignal(str)
    progress_log = QtCore.pyqtSignal(str) 

    def __init__(self, manager: 'KDEManager'):
        super().__init__()
        self.manager = manager

    def run(self):
        try:
            result_info = self.manager.run_preparation(logger_callback=self.progress_log.emit)
            self.finished_success.emit(result_info)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_error.emit(str(e))