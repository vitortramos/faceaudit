import traceback
from PyQt6 import QtCore
from faceaudit.models import FaceDataset

class LoadDatasetWorker(QtCore.QObject):
    """
    Worker responsável por carregar o dataset (arquivo Parquet) do disco.
    """
    finished = QtCore.pyqtSignal(object)  # emite (FaceDataset, display_df) — df metadata-only
    error = QtCore.pyqtSignal(str)        # Retorna mensagem de erro se falhar

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            # O processo pesado de leitura/deserialização
            dataset = FaceDataset.from_parquet(self.path)
            # Constrói o DataFrame de EXIBIÇÃO (metadata-only, sem colunas de
            # embedding) AQUI, na thread do worker. Antes isso rodava na
            # thread da GUI (via to_pandas em carregar_dataset_na_tela),
            # congelando a interface por minutos em datasets grandes e
            # re-serializando todos os vetores para listas Python. Emitimos os
            # dois: o dataset (fonte de verdade, com embeddings nos FaceInfo)
            # e o df leve pronto pra tabela.
            display_df = dataset.to_pandas(include_embeddings=False)
            self.finished.emit((dataset, display_df))
        except Exception as e:
            error_trace = traceback.format_exc()
            self.error.emit(f"Erro ao ler arquivo: {str(e)}\n\n{error_trace}")