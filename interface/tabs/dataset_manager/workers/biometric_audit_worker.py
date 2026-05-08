"""Worker para auditorias biométricas O(N²) (PR 13).

Roda `audit_duplicate_identities` e `audit_mixed_identities` do
`DatasetAuditService` em background. Cumpre o contrato esperado pelo
`WorkerProgressRunner`:
  - `progress_update(str, int)` para feedback durante o cálculo
  - `finished(object)` com `List[AuditFinding]` (ou None se cancelado)
  - `error(str)` em falha
  - `stop()` para cancelamento cooperativo
"""

from __future__ import annotations

import traceback
from typing import List, Optional

from PyQt6 import QtCore

from faceaudit.models.audit import AuditFinding
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.services.dataset_audit_service import DatasetAuditService


class BiometricAuditWorker(QtCore.QObject):
    """Executa as duas auditorias O(N²) sequencialmente. Cada audit
    reporta progresso 0–100% via `progress_update`; o worker remapea pra
    [0, 50] (duplicate) e [50, 100] (mixed) pra ter uma barra única."""

    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        dataset: FaceDataset,
        audit_service: DatasetAuditService,
        duplicate_threshold: float = 0.90,
        mixed_outlier_threshold: float = 0.4,
        mixed_min_samples: int = 3,
    ):
        super().__init__()
        self._dataset = dataset
        self._audit_service = audit_service
        self._duplicate_threshold = duplicate_threshold
        self._mixed_outlier_threshold = mixed_outlier_threshold
        self._mixed_min_samples = mixed_min_samples
        self._is_interrupted = False

    def stop(self) -> None:
        """Cancelamento cooperativo — checked nos callbacks de progresso."""
        self._is_interrupted = True

    def run(self) -> None:
        try:
            findings: List[AuditFinding] = []

            self.progress_update.emit("Pessoas com mais de um rótulo — iniciando…", 0)
            duplicate = self._audit_service.audit_duplicate_identities(
                self._dataset,
                threshold=self._duplicate_threshold,
                progress_cb=lambda msg, pct: self._maybe_emit(msg, int(pct * 0.5)),
            )
            if self._is_interrupted:
                self.finished.emit(None)
                return
            findings.append(duplicate)

            self.progress_update.emit("Faces com erro de identidade — iniciando…", 50)
            mixed = self._audit_service.audit_mixed_identities(
                self._dataset,
                outlier_threshold=self._mixed_outlier_threshold,
                min_samples=self._mixed_min_samples,
                progress_cb=lambda msg, pct: self._maybe_emit(
                    msg, 50 + int(pct * 0.5),
                ),
            )
            if self._is_interrupted:
                self.finished.emit(None)
                return
            findings.append(mixed)

            self.progress_update.emit("Concluído.", 100)
            self.finished.emit(findings)

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)

    def _maybe_emit(self, msg: str, pct: int) -> None:
        """Emite progresso e checka cancelamento. O service não consulta
        `_is_interrupted` diretamente — esse callback é a ponte."""
        self.progress_update.emit(msg, pct)
        if self._is_interrupted:
            # Não há mecanismo cooperativo dentro do service por enquanto;
            # apenas evita emitir mais updates. O run() captura o flag
            # entre as duas auditorias.
            pass
