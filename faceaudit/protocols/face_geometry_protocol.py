"""Contrato mínimo de geometria facial consumido pela materialização.

Existe para que `FaceImageMaterializationService` dependa do que
PRECISA — localizar a face e produzir o recorte alinhado — e não do
`FFService` inteiro, que carrega N modelos ONNX na construção. Além de
inverter a dependência, é o que torna a regra testável sem GPU.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class FaceGeometryProviderProtocol(Protocol):
    """Localiza faces e produz o recorte alinhado de uma imagem."""

    def detect_faces(self, image: np.ndarray) -> Sequence[Any]:
        """Faces detectadas na imagem, cada uma com `.bbox` (x1,y1,x2,y2).

        Lista vazia quando não há face — não é erro.
        """
        ...

    def align_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Recorte alinhado (112×112) da melhor face, ou None se falhar."""
        ...
