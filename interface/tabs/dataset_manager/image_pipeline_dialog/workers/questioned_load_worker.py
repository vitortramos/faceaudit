"""QuestionedLoadWorker — lê a imagem questionada, localiza a face e mede,
fora da thread da interface (a detecção leva até alguns segundos na CPU).

Emite `finished((QuestionedProfile, imagem BGR))` ou `error(str)`.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore

from faceaudit.services.image_degradation_profile_service import (
    build_questioned_profile,
)


class QuestionedLoadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, path: Path, ff_service=None):
        super().__init__()
        self._path = Path(path)
        self._ff_service = ff_service

    def run(self):
        try:
            self.finished.emit(build_questioned_profile(self._path, self._ff_service))
        except Exception as e:  # noqa: BLE001 — vira mensagem na interface
            self.error.emit(str(e))
