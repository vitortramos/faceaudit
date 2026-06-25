"""Worker da importação/enriquecimento de um run de vídeo (Fase 5/6).

Roda `VideoRunImporter.import_run(...)` numa thread. Em modo ponta-a-ponta
(`enrich=True`), o importador chama `enrich_existing_dataset` (FF/OFIQ/
SERFIQ) e finaliza o manifesto. Quando `dataset_dir` é informado, o worker
ainda **finaliza o dataset na pasta de datasets**: autocontido (copia os
recortes alinhados, caminhos relativos — portátil) ou por caminho absoluto
(referencia os recortes do run, sem copiar).
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PyQt6 import QtCore

from faceaudit.services.video_run_importer import VideoRunImporter


class VideoImportWorker(QtCore.QObject):
    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)            # FaceDataset | None
    error = QtCore.pyqtSignal(str)

    def __init__(self, run_dir, *, enrich, pipeline=None, ingestion_config=None,
                 dataset_name=None, dataset_dir=None, self_contained=True):
        super().__init__()
        self._run_dir = run_dir
        self._enrich = enrich
        self._pipeline = pipeline
        self._ingestion_config = ingestion_config
        self._dataset_name = dataset_name
        self._dataset_dir = dataset_dir
        self._self_contained = self_contained
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
            if ds is not None and self._enrich and self._dataset_dir:
                ds = self._finalize_to_dataset_dir(ds)
            self.progress_update.emit("Concluído.", 100)
            self.finished.emit(ds)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def _finalize_to_dataset_dir(self, ds):
        """Grava o dataset final na pasta de datasets — autocontido (copia os
        recortes) ou por caminho absoluto (referencia o run). Preserva o aviso
        de viés (provenance_metadata), os embeddings (ff_info) e is_aligned."""
        dataset_dir = Path(self._dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        name = ds.name
        prov = dict(getattr(ds, "provenance_metadata", {}) or {})
        parquet_path = dataset_dir / f"{name}.parquet"

        if self._self_contained:
            from faceaudit.services.dataset_localization_service import DatasetLocalizationService
            from faceaudit.enums.localization_mode import LocalizationMode

            self.progress_update.emit("Copiando recortes para a pasta do dataset…", 90)

            def cb(idx, total, _msg):
                if total:
                    self.progress_update.emit(f"Copiando recortes… {idx + 1}/{total}", 90)

            res = DatasetLocalizationService().localize(
                ds, dataset_dir, mode=LocalizationMode.COPY_ORIGINALS, progress_callback=cb,
            )
            final = res.dataset
            final.name = name
            final.provenance_metadata = prov            # localize não copia este campo
            final.is_aligned = ds.is_aligned
            final._parquet_dir = dataset_dir.resolve()
            final.to_parquet(parquet_path)
            return final

        # Caminho absoluto: referencia os recortes do run, sem copiar.
        from faceaudit.enums.path_mode import PathMode
        self.progress_update.emit("Gravando dataset (caminho absoluto)…", 90)
        for f in ds.faces:
            f.image_path = str(ds.resolve_image_path(f))
        ds.path_mode = PathMode.ABSOLUTE
        ds.path_root = None
        ds._parquet_dir = dataset_dir.resolve()
        ds.to_parquet(parquet_path)
        return ds
