"""LocalizeDatasetWorker — roda DatasetLocalizationService em thread
separada (PR L1d).

Sinais Qt seguindo o padrão dos outros workers do dataset_manager
(EnrichWorker, FilterWorker, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.localization_result import LocalizationResult
from faceaudit.services.dataset_localization_service import (
    DatasetLocalizationService,
)


class LocalizeDatasetWorker(QtCore.QObject):
    """Worker assíncrono para localização (REWRITE_RELATIVE / COPY_ORIGINALS).

    Emite progress periodicamente — uma vez a cada `_PROGRESS_THROTTLE_FACES`
    faces ou ao chegar no final. Sem throttle, datasets de 100k faces
    saturam o event loop Qt com 100k signal emissions.
    """

    finished = QtCore.pyqtSignal(object)  # LocalizationResult ou None
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    _PROGRESS_THROTTLE_FACES = 50

    def __init__(
        self,
        dataset: FaceDataset,
        target_dir: Path,
        mode: LocalizationMode,
        naming: NamingStrategy = NamingStrategy.PRESERVE,
        skip_existing: bool = True,
        old_prefix: Optional[str] = None,
        ff_service=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.target_dir = target_dir
        self.mode = mode
        self.naming = naming
        self.skip_existing = skip_existing
        # Só relevante em REBASE_PREFIX (substituição de prefixo de path).
        self.old_prefix = old_prefix
        # PR L3 — necessário pra COPY_ALIGNED. Aceita None pros outros
        # modos (REWRITE_RELATIVE / COPY_ORIGINALS / REBASE_PREFIX).
        self.ff_service = ff_service
        self._is_running = True

    def run(self):
        """Executado na thread. Roda o service e emite finished com o
        `LocalizationResult` (ou None em cancelamento)."""
        try:
            self.progress_update.emit(
                f"Iniciando {self.mode.value} para '{self.dataset.name}'…", 0,
            )
            service = DatasetLocalizationService(ff_service=self.ff_service)
            result = service.localize(
                self.dataset,
                self.target_dir,
                mode=self.mode,
                naming=self.naming,
                skip_existing=self.skip_existing,
                old_prefix=self.old_prefix,
                progress_callback=self._on_progress,
            )
            if not self._is_running:
                self.finished.emit(None)
                return
            # Mensagem final detalhada
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
        """Sinaliza cancelamento cooperativo. O service não tem ponto
        de cancelamento granular hoje — quem rodar Cancel pode esperar
        terminar a face atual e a chamada toda completa, mas ignora o
        resultado."""
        self._is_running = False

    def _on_progress(self, idx: int, total: int, msg: str):
        """Recebe (idx, total, msg) do service e converte em
        (mensagem, pct) pro Qt signal."""
        if not self._is_running:
            # Cancelamento cooperativo — service não tem como abortar,
            # mas podemos pular emissões pra reduzir footprint.
            return
        # Throttle: emite só a cada N faces ou na última
        is_last = (idx + 1) >= total
        if not is_last and (idx % self._PROGRESS_THROTTLE_FACES) != 0:
            return
        pct = int(((idx + 1) / max(total, 1)) * 100)
        # Trim de mensagens de path muito longas pra não inflar o log
        short = msg if len(msg) <= 80 else "…" + msg[-77:]
        self.progress_update.emit(
            f"[{idx + 1}/{total}] {short}", pct,
        )
