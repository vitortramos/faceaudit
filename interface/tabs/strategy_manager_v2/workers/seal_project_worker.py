"""SealProjectWorker — roda ProjectSealService em thread separada
(PR L2.b).

Sinais Qt seguindo o padrão dos outros workers do strategy_manager_v2.
"""

from __future__ import annotations

from PyQt6 import QtCore

from faceaudit.models.seal_result import SealResult
from faceaudit.services.project_seal_service import ProjectSealService


class SealProjectWorker(QtCore.QObject):
    """Worker assíncrono pra "Selar projeto".

    Emite progress por dataset processado. Pra projetos com muitos
    datasets, throttle não é necessário — granularidade já é
    naturalmente baixa (1 progresso por dataset).
    """

    finished = QtCore.pyqtSignal(object)  # SealResult ou None
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    def __init__(self, project, dry_run: bool = False):
        super().__init__()
        self.project = project
        self.dry_run = dry_run
        self._is_running = True

    def run(self):
        try:
            label = "Simulando" if self.dry_run else "Selando"
            self.progress_update.emit(
                f"{label} projeto '{self.project.metadata.name}'…", 0,
            )
            service = ProjectSealService()
            result = service.seal(
                self.project,
                dry_run=self.dry_run,
                progress_callback=self._on_progress,
            )
            if not self._is_running:
                self.finished.emit(None)
                return
            self.progress_update.emit(result.summary(), 100)
            self.finished.emit(result)
        except InterruptedError:
            self.finished.emit(None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def stop(self):
        """Cancelamento cooperativo. Granularidade limitada — service
        completa o dataset atual e ignora o resultado."""
        self._is_running = False

    def _on_progress(self, idx: int, total: int, dataset_id: str):
        if not self._is_running:
            return
        pct = int(((idx + 1) / max(total, 1)) * 100)
        self.progress_update.emit(
            f"[{idx + 1}/{total}] dataset '{dataset_id}'", pct,
        )
