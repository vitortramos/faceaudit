"""Worker do clique de âncora na janela de dataset de vídeo.

Um clique numa face custa detecção + alinhamento + embedding de recognition.
No macOS com CoreML o encoder exige lote fixo (padding até `batch_size`), e
cada âncora leva 2–4 s — tempo demais para a thread da interface: a janela
congela e o clique parece perdido. Este worker leva o trabalho para outra
thread e devolve um `AnchorClickResult` pronto para a janela aplicar.

Sem regra própria: só orquestra `VideoFrameFaceExtractionService`.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from PyQt6 import QtCore


@dataclass
class AnchorClickResult:
    """Resultado do clique, já na forma que a janela consome.

    - `hit`: a detecção continha o ponto e o embedding foi extraído.
    - `target`: alvo construído (1ª âncora) ou o alvo existente, já mutado.
    - `det_index`: índice da detecção clicada (None se `hit=False`).
    - `created`: True quando o clique CRIOU o alvo (1ª âncora).
    - `anchors_before`: contagem de âncoras antes do clique — a janela
      compara com a atual para avisar "face já ancorada".
    """

    hit: bool
    target: Any = None
    det_index: Optional[int] = None
    created: bool = False
    anchors_before: int = 0


class AnchorClickWorker(QtCore.QObject):
    """Executa `build_target_from_click` (1ª âncora) ou `add_anchor_from_click`
    (âncora adicional) fora da thread da interface. Emite `finished`
    (`AnchorClickResult`, ou None em erro) e `error(str)`."""

    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        service,
        frame_bgr: np.ndarray,
        x: float,
        y: float,
        *,
        model: Optional[str],
        source: dict,
        existing_target=None,
        label: str = "alvo",
        similarity_threshold: float = 0.0,
        frame_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._frame = frame_bgr
        self._x = float(x)
        self._y = float(y)
        self._model = model
        self._source = dict(source or {})
        self._existing = existing_target
        self._label = label
        self._threshold = float(similarity_threshold)
        self._frame_index = frame_index

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.finished.emit(self._compute())
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            # Sem isto a janela fica "ocupada" para sempre.
            self.finished.emit(None)

    def _compute(self) -> AnchorClickResult:
        if self._existing is not None:
            before = int(self._existing.anchor_embedding_count)
            res = self._service.add_anchor_from_click(
                self._existing, self._frame, self._x, self._y,
                model=self._model, source=self._source,
            )
            if res is None:
                return AnchorClickResult(hit=False, target=self._existing,
                                         anchors_before=before)
            det_idx, _dets = res
            return AnchorClickResult(hit=True, target=self._existing,
                                     det_index=int(det_idx), created=False,
                                     anchors_before=before)

        built = self._service.build_target_from_click(
            self._frame, self._x, self._y,
            label=self._label, similarity_threshold=self._threshold,
            model=self._model, frame_index=self._frame_index,
        )
        if built is None:
            return AnchorClickResult(hit=False)
        target, det_idx, _dets = built
        # Registra a fonte real (vídeo ou foto) preservando bbox/det_index.
        if target.anchor_sources:
            merged = dict(target.anchor_sources[-1])
            merged.update(self._source)
            target.anchor_sources[-1] = merged
        return AnchorClickResult(hit=True, target=target, det_index=int(det_idx),
                                 created=True, anchors_before=0)
