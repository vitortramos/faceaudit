from PyQt6 import QtCore

class FilterWorker(QtCore.QObject):
    """
    Worker responsável por aplicar filtros no Dataset em background.
    Específico para a lógica de negócio do DatasetManager.
    """
    finished = QtCore.pyqtSignal(object)  # Retorna o dataset filtrado
    progress = QtCore.pyqtSignal(int)     # Retorna % de progresso
    error = QtCore.pyqtSignal(str)        # ✅ NOVO: reporta erro

    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Callback para atualizar a barra de progresso na UI
            if hasattr(self.dataset, 'apply_filter'):
                result = self.dataset.apply_filter(self.config, callback=self.progress.emit)
                self.finished.emit(result)
            else:
                # Fallback caso o método não exista (segurança)
                self.finished.emit(self.dataset)
        except Exception as e:
            self.error.emit(str(e))
