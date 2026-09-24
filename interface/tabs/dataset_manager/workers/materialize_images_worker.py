"""Worker da geração de imagens de face pasta → pasta.

Detecção e alinhamento rodam por imagem e são caros; numa pasta com
milhares de fotos isso congelaria a interface. Segue o padrão dos demais
workers do Dataset Manager: `QObject` movido para uma thread, progresso
por sinal, cancelamento cooperativo.

Nenhuma regra aqui — só orquestra o
`FaceImageMaterializationService`, que é quem sabe produzir as imagens.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore

from faceaudit.enums.dataset_image_kind import DatasetImageKind


class MaterializeImagesWorker(QtCore.QObject):
    """Gera as imagens em thread. Emite `MaterializationResult` (ou None)."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    _PROGRESS_THROTTLE = 20

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        *,
        kind: DatasetImageKind,
        margin_factor: float,
        recursive: bool,
        ff_service,
    ) -> None:
        super().__init__()
        self._source_dir = Path(source_dir)
        self._target_dir = Path(target_dir)
        self._kind = kind
        self._margin_factor = float(margin_factor)
        self._recursive = bool(recursive)
        self._ff_service = ff_service
        self._is_running = True

    def run(self) -> None:
        from faceaudit.services.face_image_materialization_service import (
            FaceImageMaterializationService,
        )
        from faceaudit.services.ff_geometry_adapter import FFServiceGeometryAdapter

        try:
            self.progress_update.emit(
                f"Gerando imagens ({self._kind.label_pt})…", 0,
            )
            servico = FaceImageMaterializationService(
                FFServiceGeometryAdapter(self._ff_service)
            )
            resultado = servico.materialize_directory(
                self._source_dir,
                self._target_dir,
                kind=self._kind,
                margin_factor=self._margin_factor,
                recursive=self._recursive,
                progress_callback=self._on_progress,
            )
            if not self._is_running:
                self.finished.emit(None)
                return
            self.progress_update.emit(resultado.summary_pt(), 100)
            self.finished.emit(resultado)
        except InterruptedError:
            self.finished.emit(None)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.error.emit(str(exc))
            self.finished.emit(None)

    def stop(self) -> None:
        """Cancelamento cooperativo — interrompe na próxima imagem."""
        self._is_running = False

    def _on_progress(self, indice: int, total: int, caminho: str) -> None:
        if not self._is_running:
            # O serviço não tem ponto de cancelamento próprio; levantar aqui
            # interrompe o laço sem deixar imagem pela metade.
            raise InterruptedError("cancelado pelo usuário")
        if total and (indice % self._PROGRESS_THROTTLE == 0 or indice + 1 >= total):
            pct = int(100 * indice / total) if total else 0
            nome = Path(caminho).name if caminho else ""
            self.progress_update.emit(
                f"Gerando imagens… {indice}/{total}" + (f" — {nome}" if nome else ""),
                min(99, pct),
            )
