"""SampleEvaluationWorker — roda o pipeline numa amostra da base e mede
contra a questionada, fora da thread da interface.

Recebe uma CÓPIA do config (o usuário pode continuar editando enquanto a
amostra roda). A amostra carregada é devolvida junto para o controller
reaproveitar na próxima medição.

Emite `finished((amostra, SampleEvaluation))` ou `error(str)`.
"""

from __future__ import annotations

from PyQt6 import QtCore

from faceaudit.services.image_pipeline_evaluation_service import (
    evaluate,
    load_sample,
)


class SampleEvaluationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, dataset, config, *, sample=None, ff_service=None):
        super().__init__()
        self._dataset = dataset
        self._config = config
        self._sample = sample
        self._ff_service = ff_service

    def run(self):
        try:
            amostra = self._sample
            if amostra is None:
                amostra = load_sample(
                    self._dataset, self._config.input_source, ff_service=self._ff_service)
            avaliacao = evaluate(amostra, self._config, ff_service=self._ff_service)
            self.finished.emit((amostra, avaliacao))
        except Exception as e:  # noqa: BLE001 — vira mensagem na interface
            self.error.emit(str(e))
