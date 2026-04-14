# interface/tabs/dataset_manager/workers/enrich_worker.py
from PyQt6 import QtCore
from faceaudit.models import FaceDataset
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig

class EnrichWorker(QtCore.QObject):
    """
    Worker assíncrono para enriquecimento de datasets existentes.
    Executa os serviços de IA (FF, OFIQ, SER-FIQ) em background.
    """
    # Sinais para comunicação com a GUI
    finished = QtCore.pyqtSignal(object)       # Retorna o FaceDataset atualizado
    error = QtCore.pyqtSignal(str)             # Retorna mensagem de erro
    progress_update = QtCore.pyqtSignal(str, int) # Retorna (mensagem, porcentagem)

    def __init__(self, pipeline, dataset: FaceDataset, config: DatasetIngestionConfig):
        super().__init__()
        self.pipeline = pipeline
        self.dataset = dataset
        self.config = config
        self._is_running = True

    def run(self):
        """
        Método principal executado na thread separada.
        """
        try:
            # Chama o método novo que criamos no Pipeline
            # Passa self._callback para receber atualizações do pipeline
            dataset_enriquecido = self.pipeline.enrich_existing_dataset(
                dataset=self.dataset,
                config=self.config,
                progress_callback=self._callback
            )
            
            # Se chegou aqui, deu tudo certo
            self.finished.emit(dataset_enriquecido)

        except InterruptedError:
            # Cancelamento controlado (usuário apertou Cancelar)
            self.finished.emit(None) # Retorna None para indicar cancelamento
            
        except Exception as e:
            # Erro inesperado
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def stop(self):
        """Chamado pelo botão Cancelar da janela de progresso."""
        self._is_running = False

    def _callback(self, msg: str, pct: int):
        """
        Função passada para o Pipeline.
        Verifica se deve continuar rodando e emite o progresso.
        """
        if not self._is_running:
            raise InterruptedError("Operação cancelada pelo usuário.")
            
        self.progress_update.emit(msg, pct)