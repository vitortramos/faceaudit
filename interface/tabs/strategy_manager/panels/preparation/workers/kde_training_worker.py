# interface/tabs/strategy_manager/panels/preparation/workers/kde_training_worker.py
from PyQt6 import QtCore
import traceback
from pathlib import Path

class KDETrainingWorker(QtCore.QThread):
    """
    Worker responsável por executar o treinamento (Calibração) do KDE em background.
    Evita o congelamento da interface durante operações pesadas (M x N + GridSearch).
    """
    
    # Sinais para comunicação com a GUI
    progress_log = QtCore.pyqtSignal(str)   # Mensagens de log
    finished_success = QtCore.pyqtSignal(str) # Caminho do arquivo salvo
    finished_error = QtCore.pyqtSignal(str)   # Mensagem de erro

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self._is_running = True

    def run(self):
        try:
            self.progress_log.emit(f"🚀 Iniciando Worker KDE no Thread: {QtCore.QThread.currentThreadId()}")
            
            # Função de callback para redirecionar prints do Manager para o sinal do Qt
            def log_callback(msg):
                if self._is_running:
                    self.progress_log.emit(msg)

            # Executa o treino (Bloqueante no Thread, livre na GUI)
            saved_path = self.manager.run_training(
                logger_callback=log_callback
            )
            
            if self._is_running:
                self.finished_success.emit(str(saved_path))
                
        except Exception as e:
            traceback.print_exc()
            if self._is_running:
                self.finished_error.emit(str(e))

    def stop(self):
        self._is_running = False
        self.quit()
        self.wait()