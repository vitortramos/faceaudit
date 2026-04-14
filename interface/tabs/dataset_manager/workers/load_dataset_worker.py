import traceback
from PyQt6 import QtCore
from faceaudit.models import FaceDataset

class LoadDatasetWorker(QtCore.QObject):
    """
    Worker responsável por carregar o dataset (arquivo Parquet) do disco.
    """
    finished = QtCore.pyqtSignal(object)  # Retorna o objeto FaceDataset carregado
    error = QtCore.pyqtSignal(str)        # Retorna mensagem de erro se falhar

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            # O processo pesado de leitura/deserialização
            dataset = FaceDataset.from_parquet(self.path)
            self.finished.emit(dataset)
        except Exception as e:
            error_trace = traceback.format_exc()
            self.error.emit(f"Erro ao ler arquivo: {str(e)}\n\n{error_trace}")