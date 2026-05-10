"""BalanceDatasetWorker — roda DatasetBalancingService em thread.

Sinais Qt seguindo o padrão dos outros workers do dataset_manager
(EnrichWorker, FilterWorker, etc.).
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import (
    DatasetBalancingSelection,
)
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.dataset_balancing_service import (
    DatasetBalancingService,
    DatasetBalancingResult,
)


class BalanceDatasetWorker(QtCore.QObject):
    """Worker assíncrono para balanceamento por identidade."""

    # `DatasetBalancingResult` ou None (em cancelamento)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)
    progress_update = QtCore.pyqtSignal(str, int)

    def __init__(
        self,
        dataset: FaceDataset,
        mode: DatasetBalancingMode,
        target_n: int,
        selection: DatasetBalancingSelection,
        random_seed: int = 42,
        quality_metric: QualityScoreType = QualityScoreType.FIQA,
        output_name: Optional[str] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.target_n = target_n
        self.selection = selection
        self.random_seed = random_seed
        self.quality_metric = quality_metric
        self.output_name = output_name
        self._is_running = True

    def run(self):
        try:
            self.progress_update.emit(
                f"Balanceando '{self.dataset.name}' "
                f"(modo={self.mode.to_str()}, "
                f"critério={self.selection.to_str()})…", 5,
            )
            service = DatasetBalancingService()
            result = service.balance(
                dataset=self.dataset,
                mode=self.mode,
                target_n=self.target_n,
                selection=self.selection,
                random_seed=self.random_seed,
                quality_metric=self.quality_metric,
                output_name=self.output_name,
                logger=lambda msg: self.progress_update.emit(msg, 50),
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
        """Sem ponto de cancel granular — balanceamento é fast (in-memory)."""
        self._is_running = False
