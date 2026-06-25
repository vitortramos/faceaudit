"""Orquestrador da criação de dataset a partir de vídeo (Fase 5).

Coordena: janela de configuração → worker de extração → grade de revisão
→ (produtor) aponta a pasta / (ponta a ponta) worker de enrich → registra
o dataset no Gerenciador + audit log VIDEO_EXTRACT.

Delega ao DatasetManagerController (`manager`) a infra existente:
`_run_worker`, `carregar_dataset_na_tela`, `_audit`, `view`, `config`,
`pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.video_extraction_mode import VideoExtractionMode
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.services.video_run_importer import VideoRunImporter

from ..workers.video_extraction_worker import VideoExtractionWorker
from ..workers.video_import_worker import VideoImportWorker
from .video_dataset_setup_dialog import VideoDatasetSetupDialog
from .video_review_dialog import VideoReviewDialog


class VideoDatasetController(QtCore.QObject):
    def __init__(self, manager):
        super().__init__()
        self._mgr = manager
        self._view = manager.view
        self._config = getattr(manager, "config", None)
        self._pipeline = getattr(manager, "pipeline", None)
        self._pending = {}

    # ───────────────────────── entrada ─────────────────────────
    def run_new_from_video(self):
        # Caminho validado em CPU; esta workstation (Pascal) não tem CUDA em
        # ort>=1.20. Forçado False por segurança — revisitar quando o hardware
        # permitir (ou expor um toggle na janela).
        use_gpu = False
        start_dir = self._library_dir()
        dlg = VideoDatasetSetupDialog(self._view, use_gpu=use_gpu, model=None,
                                      datasets_dir=start_dir, output_dir=self._runs_root())
        if not dlg.exec():
            return
        service = dlg.get_service()
        targets = dlg.get_targets()
        if service is None or not targets:
            return
        name = dlg.get_dataset_name()
        config = dlg.get_config(self._runs_root())
        self._pending = {"mode": dlg.get_mode(), "name": name, "config": config,
                         "n_targets": len(targets)}

        worker = VideoExtractionWorker(service, config, targets)
        label = ("Extraindo faces do vídeo" if len(targets) == 1
                 else f"Extraindo faces do vídeo ({len(targets)} alvos)")
        self._mgr._run_worker(
            worker, label=label,
            allow_cancel=True, determinate=False, use_log_dialog=True,
            on_finished=self._on_extraction_done,
        )

    def run_import_video_run(self):
        start = self._runs_root()
        run_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self._view, "Escolher pasta do run de vídeo", start)
        if not run_dir:
            return
        if not (Path(run_dir) / "_meta").is_dir():
            QtWidgets.QMessageBox.warning(self._view, "Importar",
                                          "A pasta escolhida não parece ser um run de vídeo (sem _meta/).")
            return
        self._pending = {"mode": VideoExtractionMode.END_TO_END, "name": Path(run_dir).name,
                         "config": None}
        self._open_review_then_finish(run_dir)

    # ───────────────────────── etapas ─────────────────────────
    def _on_extraction_done(self, result):
        if result is None:
            return
        if getattr(result, "n_accepted_auto", 0) <= 0:
            QtWidgets.QMessageBox.information(
                self._view, "Extração",
                "Nenhuma face foi aceita (tente um limiar menor, det. 960, ou outra âncora).\n"
                f"Run em: {result.run_dir}")
            return
        self._open_review_then_finish(result.run_dir)

    def _open_review_then_finish(self, run_dir: str):
        dlg = VideoReviewDialog(self._view, run_dir, operator="perito")
        if not dlg.exec():
            QtWidgets.QMessageBox.information(
                self._view, "Revisão",
                f"Revisão cancelada. O run permanece em:\n{run_dir}")
            return

        mode = self._pending.get("mode", VideoExtractionMode.PRODUCER)
        if mode == VideoExtractionMode.PRODUCER:
            res = QtWidgets.QMessageBox.question(
                self._view, "Produtor",
                f"Recortes revisados em:\n{run_dir}\n\nAbrir a pasta?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if res == QtWidgets.QMessageBox.StandardButton.Yes:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(run_dir))
            return

        # Ponta a ponta: enriquecer + registrar
        name = self._pending.get("name") or Path(run_dir).name
        vcfg = self._pending.get("config")
        ingestion = DatasetIngestionConfig(
            input_path=Path(run_dir), dataset_name=name,
            compute_ff=True,
            compute_ofiq=bool(getattr(vcfg, "compute_ofiq", False)),
            compute_serfiq=bool(getattr(vcfg, "compute_serfiq", False)),
        )
        worker = VideoImportWorker(run_dir, enrich=True, pipeline=self._pipeline,
                                   ingestion_config=ingestion, dataset_name=name)
        self._mgr._run_worker(
            worker, label="Enriquecendo dataset de vídeo",
            allow_cancel=False, determinate=False, use_log_dialog=True,
            on_finished=lambda ds: self._on_import_done(ds, run_dir),
        )

    def _on_import_done(self, ds, run_dir: str):
        if ds is None or len(ds) == 0:
            QtWidgets.QMessageBox.warning(self._view, "Dataset de vídeo",
                                          "O enriquecimento terminou sem faces válidas.")
            return
        parquet = Path(run_dir) / f"{ds.name}.parquet"
        self._mgr.carregar_dataset_na_tela(ds, path=str(parquet) if parquet.exists() else None)
        try:
            meta = VideoRunImporter().audit_metadata(run_dir)
        except Exception:
            meta = {}
        try:
            self._mgr._audit(
                AuditOperation.VIDEO_EXTRACT,
                f"Dataset '{ds.name}' criado a partir de vídeo (seleção condicionada à âncora).",
                metadata=meta, hash_file=True)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self._view, "Sucesso",
            f"Dataset '{ds.name}' criado a partir de vídeo!\n\n"
            "⚠ Lembre-se: é uma seleção condicionada à similaridade com uma âncora — "
            "não representa amostra populacional neutra (vide metadados do dataset).")

    # ───────────────────────── helpers ─────────────────────────
    def _library_dir(self) -> str:
        return str(getattr(getattr(self._config, "paths", None), "library", "") or "")

    def _runs_root(self) -> str:
        lib = self._library_dir()
        return str(Path(lib) / "video_runs") if lib else "video_runs"
