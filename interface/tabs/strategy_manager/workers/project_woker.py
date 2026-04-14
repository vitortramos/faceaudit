# interface/tabs/strategy_manager/workers/project_woker.py
from PyQt6 import QtCore
from typing import Union
from pathlib import Path

class ProjectWorker(QtCore.QThread):
    """
    Worker assíncrono para operações de salvamento e carregamento de projetos (.fap).
    Permite que cálculos de hash, I/O de disco e cópia de imagens ocorram sem travar a interface principal.
    """
    progress_log = QtCore.pyqtSignal(str)
    finished_success = QtCore.pyqtSignal(str) 
    finished_error = QtCore.pyqtSignal(str)

    def __init__(self, 
                 project_manager, 
                 operation: str, 
                 file_path: Union[str, Path], 
                 copy_images: bool = False,
                 cleanup_orphans: bool = False,
                 parent=None):
        super().__init__(parent)
        self.project_manager = project_manager
        self.operation = operation
        self.file_path = file_path
        self.copy_images = copy_images
        self.cleanup_orphans = cleanup_orphans
        self._is_cancelled = False

    def run(self):
        try:
            def logger_bridge(msg: str):
                if not self._is_cancelled:
                    self.progress_log.emit(msg)

            if self.operation == "save":
                # Executa o salvamento passando todas as intenções do usuário
                success = self.project_manager.save_project(
                    filepath=self.file_path, 
                    copy_dependencies=True,  # Sempre empacota datasets(parquet) e pipelines(json)
                    copy_images=self.copy_images, # Condicional baseado no checkbox do Deep Copy
                    cleanup_orphans=self.cleanup_orphans, # Condicional baseado no checkbox do Garbage Collector
                    logger_callback=logger_bridge
                )
                if success:
                    self.finished_success.emit(f"Projeto salvo com sucesso em: {self.file_path}")
                else:
                    self.finished_error.emit("Falha no salvamento. Verifique as permissões de disco ou os logs.")

            elif self.operation == "load":
                # Executa o carregamento (lê o .fap e reconstrói a sessão)
                success = self.project_manager.load_project(
                    filepath=self.file_path, 
                    logger_callback=logger_bridge
                )
                if success:
                    self.finished_success.emit("Projeto carregado com sucesso!")
                else:
                    self.finished_error.emit("Falha ao carregar o projeto. O arquivo pode estar corrompido ou inacessível.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_error.emit(f"Erro crítico no ProjectWorker: {str(e)}")

    def cancel(self):
        """
        Sinaliza a interrupção. 
        Nota: Como o I/O de Python/JSON é atômico, a interrupção será sentida 
        principalmente entre as etapas de cálculo de Hash ou cópia de arquivos.
        """
        self._is_cancelled = True