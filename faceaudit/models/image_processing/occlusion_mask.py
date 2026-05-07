"""DTOs pra máscara de oclusão livre desenhada no canvas (PR I6).

Coordenadas armazenadas em **percentuais [0..1]** independentes do
tamanho da imagem — mesmo polígono/retângulo/elipse pode ser
aplicado em imagens de tamanhos diferentes (caso típico: pipeline
configurado numa imagem de preview e aplicado em outras imagens
do dataset com tamanhos variados).

Estrutura aninhada no JSON do pipeline:

```
{
  "shapes": [
    {
      "shape_type": "polygon",
      "points": [[0.1, 0.2], [0.5, 0.1], [0.7, 0.6], ...]
    },
    {
      "shape_type": "rectangle",
      "points": [[0.2, 0.3], [0.5, 0.6]]
    },
    {
      "shape_type": "ellipse",
      "points": [[0.6, 0.6], [0.9, 0.9]]
    }
  ]
}
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from faceaudit.enums.occlusion_mask_shape import OcclusionMaskShape


# Tipo do ponto em [0..1]
Point2D = Tuple[float, float]


@dataclass(frozen=True)
class OcclusionMaskShapeDTO:
    """Uma figura individual da máscara.

    Attributes:
        shape_type: tipo da figura (POLYGON/RECTANGLE/ELLIPSE).
        points: lista de pontos em [0..1].
            - POLYGON: ≥3 vértices.
            - RECTANGLE: exatamente 2 pontos (cantos diagonais).
            - ELLIPSE: exatamente 2 pontos (bbox que circunscreve).
    """

    shape_type: OcclusionMaskShape
    points: Tuple[Point2D, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_type": self.shape_type.value,
            "points": [[float(x), float(y)] for x, y in self.points],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionMaskShapeDTO":
        return cls(
            shape_type=OcclusionMaskShape(data["shape_type"]),
            points=tuple(
                (float(p[0]), float(p[1])) for p in data.get("points", [])
            ),
        )


@dataclass(frozen=True)
class OcclusionMask:
    """Máscara composta — lista ordenada de shapes desenhados no
    canvas. Aplicação pinta tudo em preto sobre a imagem."""

    shapes: Tuple[OcclusionMaskShapeDTO, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return len(self.shapes) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {"shapes": [s.to_dict() for s in self.shapes]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionMask":
        shapes = tuple(
            OcclusionMaskShapeDTO.from_dict(s)
            for s in data.get("shapes", [])
        )
        return cls(shapes=shapes)
