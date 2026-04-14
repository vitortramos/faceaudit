from PyQt6 import QtCore
import traceback

# Import do Modelo
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig

class NewDatasetWorker(QtCore.QObject):
    """
    Worker específico para criação de datasets.
    Recebe o Pipeline e o Objeto de Configuração (DTO).
    """
    
    # Sinais
    progress_update = QtCore.pyqtSignal(str, int)  # Msg, %
    finished = QtCore.pyqtSignal(object)           # Retorna o FaceDataset
    error = QtCore.pyqtSignal(str)                 # Retorna mensagem de erro
    cancelled = QtCore.pyqtSignal()                # Sinaliza cancelamento

    def __init__(self, pipeline, config: DatasetIngestionConfig):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self._is_interrupted = False

    def stop(self):
        """Acionado pelo clique do botão Cancelar na interface"""
        self._is_interrupted = True

    def run(self):
        try:
            def callback_wrapper(msg, pct):
                # O coração do cancelamento: corta o fluxo da IA
                if self._is_interrupted:
                    raise InterruptedError("Cancelado pelo usuário")
                self.progress_update.emit(msg, pct)

            # Executa o pipeline pesado
            result = self.pipeline.execute(
                config=self.config,
                progress_callback=callback_wrapper
            )
            
            # Checagem de segurança: 
            # Como o pipeline às vezes engole o InterruptedError para salvar 
            # os dados parciais, nós garantimos que a UI saiba que foi cancelado.
            if self._is_interrupted:
                self.cancelled.emit()
                self.finished.emit(None)
            else:
                self.finished.emit(result)

        except InterruptedError:
            self.cancelled.emit()
            # FALTAVA ISTO: Garante que a thread saiba que acabou
            self.finished.emit(None) 
            
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            # FALTAVA ISTO TAMBÉM: Evita que erros travem a tela
            self.finished.emit(None)