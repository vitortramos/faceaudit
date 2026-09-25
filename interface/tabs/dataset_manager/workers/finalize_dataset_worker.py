"""Worker da segunda metade da ingestão: copiar imagens e gravar o parquet.

Existe porque a gravação foi adiada até a revisão das imagens sem face
detectada. Copiar imagens de um dataset autocontido e escrever o parquet
são operações de disco que podem levar minutos num dataset grande — na
thread da interface, congelariam a janela logo depois de o usuário clicar
em "Concluir".

Nenhuma regra aqui: só chama `FeatureExtractionPipeline.finalize_storage_and_save`.
"""

from __future__ import annotations

import traceback

from PyQt6 import QtCore


class FinalizeDatasetWorker(QtCore.QObject):
    """Materializa o armazenamento e grava. Emite o dataset (ou None)."""

    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, pipeline, dataset, config) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._dataset = dataset
        self._config = config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            resultado = self._pipeline.finalize_storage_and_save(
                self._dataset,
                self._config,
                progress_callback=lambda msg, pct: self.progress_update.emit(msg, pct),
            )
            self.finished.emit(resultado)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            # Sem isto a tela de progresso fica aberta para sempre.
            self.finished.emit(None)
