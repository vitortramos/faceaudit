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
                                      datasets_dir=start_dir, output_dir=self._runs_root(),
                                      dataset_output_dir=self._datasets_dir())
        if not dlg.exec():
            return
        service = dlg.get_service()
        targets = dlg.get_targets()
        if service is None or not targets:
            return
        # Modo + dataset são decididos na janela de revisão; aqui só a extração.
        config = dlg.get_config(self._runs_root())
        self._pending = {"config": config}

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
            self._view, "Escolher pasta da extração de vídeo", start)
        if not run_dir:
            return
        if not (Path(run_dir) / "_meta").is_dir():
            QtWidgets.QMessageBox.warning(self._view, "Importar",
                                          "A pasta escolhida não parece ser uma extração de vídeo (sem _meta/).")
            return
        self._pending = {"config": None}
        # Run pré-existente (importado) — NUNCA apagar no cancelar.
        self._open_review_then_finish(run_dir, owns_run=False)

    # ───────────────────────── etapas ─────────────────────────
    def _on_extraction_done(self, result):
        if result is None:
            return
        if getattr(result, "n_accepted_auto", 0) <= 0:
            QtWidgets.QMessageBox.information(
                self._view, "Extração",
                "Nenhuma face foi aceita (tente um limiar menor, det. 960, ou outra âncora).\n"
                f"Extração em: {result.run_dir}")
            return
        # Run recém-criado NESTA sessão — pode ser descartado no cancelar (com confirmação).
        self._open_review_then_finish(result.run_dir, owns_run=True)

    def _open_review_then_finish(self, run_dir: str, *, owns_run: bool = False):
        # A janela de revisão é dona da configuração do dataset (Modo, nome,
        # embeddings, destino, OFIQ/SER-FIQ) — vale para os dois fluxos (criar
        # e importar). Lemos tudo dela após a revisão. `owns_run`: o run foi
        # criado NESTA sessão (criação) — só ele pode ser descartado no cancelar;
        # run importado (pré-existente) nunca é apagado.
        dlg = VideoReviewDialog(self._view, run_dir, operator="perito",
                                dataset_output_dir=self._datasets_dir())
        if not dlg.exec():
            self._handle_review_cancelled(run_dir, owns_run)
            return

        if dlg.get_mode() == VideoExtractionMode.PRODUCER:
            res = QtWidgets.QMessageBox.question(
                self._view, "Só recortes",
                f"Recortes revisados em:\n{run_dir}\n\nAbrir a pasta?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if res == QtWidgets.QMessageBox.StandardButton.Yes:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(run_dir))
            return

        # Recortes + dataset: enriquecer + registrar
        name = dlg.get_dataset_name() or Path(run_dir).name
        ingestion = DatasetIngestionConfig(
            input_path=Path(run_dir), dataset_name=name,
            compute_ff=True,
            compute_ofiq=dlg.compute_ofiq(),
            compute_serfiq=dlg.compute_serfiq(),
            ff_models=dlg.get_dataset_embeddings() or None,
        )
        worker = VideoImportWorker(
            run_dir, enrich=True, pipeline=self._pipeline,
            ingestion_config=ingestion, dataset_name=name,
            dataset_dir=dlg.get_dataset_output_dir() or None,
            self_contained=dlg.get_dataset_self_contained(),
            image_kind=dlg.get_image_type())
        self._mgr._run_worker(
            worker, label="Enriquecendo dataset de vídeo",
            allow_cancel=True, determinate=False, use_log_dialog=True,
            on_finished=lambda ds: self._on_import_done(ds, run_dir),
        )

    def _on_import_done(self, ds, run_dir: str):
        if ds is None or len(ds) == 0:
            QtWidgets.QMessageBox.information(
                self._view, "Dataset de vídeo",
                "Enriquecimento cancelado ou sem faces válidas.\n"
                f"A extração permanece em:\n{run_dir}\n\nVocê pode reimportá-la depois "
                "(Importar extração de vídeo) sem reprocessar.")
            return
        parquet = Path(getattr(ds, "_parquet_dir", None) or run_dir) / f"{ds.name}.parquet"
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
            f"Dataset '{ds.name}' criado a partir de vídeo!\n"
            f"Salvo em: {parquet}\n\n"
            "⚠ Lembre-se: é uma seleção condicionada à similaridade com uma âncora — "
            "não representa amostra populacional neutra (vide metadados do dataset).")

    # ───────────────────────── helpers ─────────────────────────
    def _handle_review_cancelled(self, run_dir: str, owns_run: bool):
        """Cancelamento da revisão. Run recém-criado (`owns_run`) pode ser
        descartado mediante confirmação; run importado (pré-existente) nunca é
        apagado — apenas fechamos e avisamos onde ele está."""
        if owns_run:
            res = QtWidgets.QMessageBox.question(
                self._view, "Descartar extração?",
                "Revisão cancelada.\n\nDescartar os recortes desta extração? "
                "Isso APAGA a pasta da extração:\n" + str(run_dir),
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if res == QtWidgets.QMessageBox.StandardButton.Yes:
                self._delete_run_dir(run_dir)
                return
        QtWidgets.QMessageBox.information(
            self._view, "Revisão",
            f"Revisão cancelada. A extração permanece em:\n{run_dir}\n\n"
            "Você pode revisá-la/importá-la depois (Importar extração de vídeo).")

    def _delete_run_dir(self, run_dir: str):
        """Apaga a pasta da extração. Guarda de segurança: só apaga se for mesmo
        uma extração de vídeo (tem _meta/) — evita apagar a pasta errada por engano."""
        import shutil
        p = Path(run_dir)
        if not (p / "_meta").is_dir():
            QtWidgets.QMessageBox.warning(
                self._view, "Descartar",
                "A pasta não parece uma extração de vídeo (sem _meta/) — não apaguei por segurança.")
            return
        try:
            shutil.rmtree(p)
            QtWidgets.QMessageBox.information(
                self._view, "Descartado", "Os recortes desta extração foram apagados.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self._view, "Descartar", f"Não foi possível apagar a pasta da extração:\n{e}")

    def _library_dir(self) -> str:
        return str(getattr(getattr(self._config, "paths", None), "library", "") or "")

    def _runs_root(self) -> str:
        lib = self._library_dir()
        return str(Path(lib) / "video_runs") if lib else "video_runs"

    def _datasets_dir(self) -> str:
        """Pasta global de datasets (default da 'Pasta do dataset'). Cai na
        biblioteca quando o config não expõe `paths.datasets`."""
        paths = getattr(self._config, "paths", None)
        d = getattr(paths, "datasets", None) if paths else None
        return str(d) if d else self._library_dir()
