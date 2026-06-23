"""Worker da importação/enriquecimento de um run de vídeo (Fase 5/6).

Roda `VideoRunImporter.import_run(...)` numa thread. Em modo ponta-a-ponta
(`enrich=True`), o importador chama `enrich_existing_dataset` (FF/OFIQ/
SERFIQ) e finaliza o manifesto.
"""

from __future__ import annotations

import traceback

from PyQt6 import QtCore

from faceaudit.services.video_run_importer import VideoRunImporter


class VideoImportWorker(QtCore.QObject):
    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)            # FaceDataset | None
    error = QtCore.pyqtSignal(str)

    def __init__(self, run_dir, *, enrich, pipeline=None, ingestion_config=None, dataset_name=None):
        super().__init__()
        self._run_dir = run_dir
        self._enrich = enrich
        self._pipeline = pipeline
        self._ingestion_config = ingestion_config
        self._dataset_name = dataset_name
        self._importer = VideoRunImporter()

    def run(self):
        try:
            self.progress_update.emit("Lendo run e validando hashes…", 10)

            def logger(msg):
                self.progress_update.emit(str(msg), 40)

            ds = self._importer.import_run(
                self._run_dir, dataset_name=self._dataset_name, enrich=self._enrich,
                pipeline=self._pipeline, ingestion_config=self._ingestion_config, logger=logger,
            )
            self.progress_update.emit("Concluído.", 100)
            self.finished.emit(ds)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)
