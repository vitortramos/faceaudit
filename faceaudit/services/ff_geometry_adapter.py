"""Adapta o `FFService` ao contrato mínimo de geometria facial.

O `FaceImageMaterializationService` precisa de duas coisas — localizar a
face e produzir o recorte alinhado — e não do `FFService` inteiro. Este
adaptador é a única peça que conhece as duas APIs, o que mantém a regra de
materialização testável sem carregar modelo nenhum.

Duas rotas distintas de propósito:

- `detect_faces` vai direto ao backend de detecção, sem alinhar. É o que
  o recorte com margem precisa, e evita pagar a transformação de
  similaridade que seria descartada em seguida.
- `align_face` usa `detect_and_align`, que é onde o alinhamento mora.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class FFServiceGeometryAdapter:
    """Implementa `FaceGeometryProviderProtocol` sobre um `FFService`."""

    def __init__(self, ff_service: Any, *, model: Optional[str] = None) -> None:
        self._ff = ff_service
        self._model = model

    def detect_faces(self, image: np.ndarray) -> Sequence[Any]:
        """Faces detectadas, sem alinhar. Lista vazia quando não há face."""
        try:
            engine = self._ff.app
            return engine.backend.detect_faces(image) or []
        except Exception:
            # Falha de detecção não é exceção de fluxo: o chamador conta
            # como "sem face" e registra no manifesto.
            return []

    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Recorte alinhado 112×112 em BGR, ou None se não houver face."""
        try:
            _, engine = self._ff._resolve_engine(self._model)
            item = engine.detect_and_align(image, single_face=True)
        except Exception:
            return None
        if not item:
            return None
        # `detect_and_align` devolve `aligned_face` em RGB; o resto do app
        # (e o cv2.imwrite) trabalha em BGR.
        rgb = item.get("aligned_face") if hasattr(item, "get") else None
        if rgb is None:
            return None
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
