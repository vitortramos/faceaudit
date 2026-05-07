"""ApplyImagePipelineWorker — roda ImagePipelineBatchService em
thread separada (PR I5).

Sinais Qt seguindo padrão dos outros workers (igual
LocalizeDatasetWorker do PR L1d).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.image_pipeline_batch_service import (
    ImagePipelineBatchService,
)


class ApplyImagePipelineWorker(QtCore.QObject):
    """Worker assíncrono pra aplicar pipeline em todo dataset."""

    finished = QtCore.pyqtSignal(object)  # BatchProcessingResult ou None
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    _PROGRESS_THROTTLE_FACES = 25

    def __init__(
        self,
        dataset: FaceDataset,
        pipeline_config: ImageProcessingPipelineConfig,
        target_dir: Path,
        ff_service=None,
        skip_existing: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.pipeline_config = pipeline_config
        self.target_dir = target_dir
        self.ff_service = ff_service
        self.skip_existing = skip_existing
        self._is_running = True

    def run(self):
        try:
            self.progress_update.emit(
                f"Iniciando aplicação de '{self.pipeline_config.name}' "
                f"em '{self.dataset.name}'…", 0,
            )
            service = ImagePipelineBatchService()
            result = service.run(
                self.dataset,
                self.pipeline_config,
                self.target_dir,
                ff_service=self.ff_service,
                skip_existing=self.skip_existing,
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
        """Cancelamento cooperativo. Service não tem ponto granular —
        a face atual completa e o resto é ignorado."""
        self._is_running = False

    def _on_progress(self, idx: int, total: int, msg: str):
        if not self._is_running:
            return
        is_last = (idx + 1) >= total
        if not is_last and (idx % self._PROGRESS_THROTTLE_FACES) != 0:
            return
        pct = int(((idx + 1) / max(total, 1)) * 100)
        short = msg if len(msg) <= 80 else "…" + msg[-77:]
        self.progress_update.emit(f"[{idx + 1}/{total}] {short}", pct)
