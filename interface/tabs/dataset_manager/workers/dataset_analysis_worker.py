import traceback
import pandas as pd
from PyQt6 import QtCore

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.comparison_service import ComparisonService

class DatasetAnalysisWorker(QtCore.QObject):
    """
    Worker em background para realizar a Mineração Biométrica Profunda (Hard Mining).
    Executa a autocomparação (N x N) sem travar a interface gráfica.
    """
    # Sinais para comunicação com a UI principal
    finished = QtCore.pyqtSignal(object, object) # Emite: (df_genuines, df_impostors)
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    def __init__(self, dataset: FaceDataset, comparison_service: ComparisonService, top_n: int = 100):
        super().__init__()
        self.dataset = dataset
        self.comparison_service = comparison_service
        self.top_n = top_n
        self._is_running = True

    def run(self):
        try:
            self.progress_update.emit("Convertendo para vetores matemáticos...", 10)
            emb_dataset = self.dataset.to_embedding_dataset()
            
            n = len(emb_dataset)
            if n < 2:
                raise ValueError("O dataset precisa de pelo menos 2 faces para realizar comparações.")
            
            # --- O NOVO FLUXO VETORIZADO ---
            self.progress_update.emit(f"Minerando extremos em {n * n} combinações...", 40)
            
            df_genuines, df_impostors = self.comparison_service.mine_hard_samples(
                dataset=emb_dataset,
                top_n=self.top_n
            )
            
            if not self._is_running: return
            
            self.progress_update.emit("Finalizando renderização...", 100)
            self.finished.emit(df_genuines, df_impostors)
            
        except MemoryError:
            self.error.emit(
                "Falta de Memória RAM!\n\nMesmo com a otimização vetorizada, "
                "a matriz de similaridade é grande demais para a sua RAM livre."
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(f"Falha na mineração biométrica: {str(e)}")

    def stop(self):
        """Sinaliza para o processo parar caso o usuário cancele."""
        self._is_running = False