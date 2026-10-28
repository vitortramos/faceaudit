"""Config de oclusão (PR I1 + I7).

I1 implementou OcclusionKind.RECTANGLE_BLACK + position={CENTER, RANDOM}.

I7 adiciona OcclusionKind.CIRCLE_BLACK + position={EYES, MOUTH} via
landmarks do FFService. Tanto CIRCLE_BLACK quanto RECTANGLE_BLACK
aceitam qualquer position (incluindo EYES/MOUTH) — em EYES o retângulo
é desenhado sobre cada olho individualmente (dois retângulos).

Campos relevantes por modo:
  - RECTANGLE_BLACK (qualquer position): usa `width_pct` + `height_pct`
    (frações independentes da largura e altura da imagem). Permite
    retângulos com proporção arbitrária (ex: barra horizontal sobre os
    olhos = width_pct alto + height_pct baixo).
  - CIRCLE_BLACK (qualquer position): usa `radius_pct` (fração do
    MENOR LADO da imagem). Raio = `radius_pct * min(H, W)`.

`position=RANDOM` requer RNG (passada via `apply(image, *, rng)`).
Sem RNG, cai em CENTER (defensivo).

`position=EYES` e `MOUTH` requerem FFService injetado no processor
(via factory). Sem ele, o apply levanta RuntimeError. Se a detecção
falhar (sem face), cai em CENTER.

Compat: JSONs antigos com `coverage_pct` são traduzidos por
`from_dict` pra `width_pct=height_pct=sqrt(coverage_pct)`, preservando
exatamente a área coberta antes da mudança.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.occlusion_mask import OcclusionMask
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
    enum_choices,
)


_POSICOES_GEOMETRICAS = (
    OcclusionPosition.CENTER,
    OcclusionPosition.RANDOM,
    OcclusionPosition.EYES,
    OcclusionPosition.MOUTH,
)


def _rotulo_da_mascara(config: "OcclusionConfig") -> str:
    if config.mask is None or config.mask.is_empty:
        return "🎨 Desenhar máscara…"
    n = len(config.mask.shapes)
    return f"🎨 Editar ({n} {'forma' if n == 1 else 'formas'})…"


@dataclass
class OcclusionConfig(ImageProcessingStepConfig):
    """Oclusão geométrica (I1 + I7) ou desenhada (I6)."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.OCCLUSION
    kind: OcclusionKind = OcclusionKind.MASK_FACE
    position: OcclusionPosition = OcclusionPosition.CENTER
    # Dimensões independentes do retângulo, como fração de W e H. O
    # default 0.5/0.5 cobre o mesmo "25% de área" que o antigo
    # `coverage_pct=0.25`.
    width_pct: float = 0.5
    height_pct: float = 0.5
    radius_pct: float = 0.10
    # PR I6 — máscara desenhada pelo usuário no canvas. Só relevante
    # quando kind=MASK_FACE. Position é ignorada nesse modo.
    mask: Optional[OcclusionMask] = None

    def __post_init__(self):
        # Clamp width/height
        for attr in ("width_pct", "height_pct"):
            v = getattr(self, attr)
            if v <= 0:
                setattr(self, attr, 0.01)
            elif v > 1.0:
                setattr(self, attr, 1.0)
        # Clamp radius (≤ 0.5 mantém círculo dentro da imagem mesmo
        # em posição CENTER de imagem quadrada)
        if self.radius_pct <= 0:
            self.radius_pct = 0.01
        elif self.radius_pct > 0.5:
            self.radius_pct = 0.5

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            **super().to_dict(),
            "kind": self.kind.value,
            "position": self.position.value,
            "width_pct": float(self.width_pct),
            "height_pct": float(self.height_pct),
            "radius_pct": float(self.radius_pct),
        }
        # PR I6 — só serializa mask quando há shapes (mantém JSON enxuto)
        if self.mask is not None and not self.mask.is_empty:
            out["mask"] = self.mask.to_dict()
        return out

    def param_specs(self) -> List[StepParamSpec]:
        mascara = lambda c: c.kind == OcclusionKind.MASK_FACE  # noqa: E731
        return [
            StepParamSpec(
                "kind", "Tipo", StepParamKind.CHOICE,
                choices=enum_choices([
                    OcclusionKind.MASK_FACE,
                    OcclusionKind.RECTANGLE_BLACK,
                    OcclusionKind.CIRCLE_BLACK,
                ]),
            ),
            StepParamSpec(
                "position", "Posição", StepParamKind.CHOICE,
                choices=enum_choices(_POSICOES_GEOMETRICAS),
                visible=lambda c: not mascara(c),
                help="Sobre os olhos / Sobre a boca usam os pontos da face "
                     "que acompanham a imagem.",
            ),
            StepParamSpec(
                "width_pct", "Largura", StepParamKind.FLOAT,
                minimum=0.01, maximum=1.0, step=0.05, decimals=2,
                visible=lambda c: c.kind == OcclusionKind.RECTANGLE_BLACK,
                help="Largura do retângulo como fração da LARGURA da imagem.",
            ),
            StepParamSpec(
                "height_pct", "Altura", StepParamKind.FLOAT,
                minimum=0.01, maximum=1.0, step=0.05, decimals=2,
                visible=lambda c: c.kind == OcclusionKind.RECTANGLE_BLACK,
                help="Altura do retângulo como fração da ALTURA da imagem.",
            ),
            StepParamSpec(
                "radius_pct", "Raio", StepParamKind.FLOAT,
                minimum=0.01, maximum=0.5, step=0.02, decimals=3,
                visible=lambda c: c.kind == OcclusionKind.CIRCLE_BLACK,
                help="Raio do círculo como fração do MENOR LADO da imagem.",
            ),
            StepParamSpec(
                "draw_mask", "Máscara", StepParamKind.ACTION,
                visible=mascara, action_label=_rotulo_da_mascara,
                help="Abre um canvas para desenhar polígonos, retângulos e "
                     "elipses sobre a imagem. Coordenadas em [0..1].",
            ),
        ]

    def summary_pt(self) -> str:
        if self.kind == OcclusionKind.MASK_FACE:
            n = 0 if self.mask is None else len(self.mask.shapes)
            return f"máscara, {n} forma(s)"
        return f"{self.kind.label_pt}, {self.position.label_pt}"

    @property
    def requires_face(self) -> bool:
        return self.kind != OcclusionKind.MASK_FACE and self.position in (
            OcclusionPosition.EYES, OcclusionPosition.MOUTH,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionConfig":
        mask = None
        if "mask" in data and data["mask"]:
            mask = OcclusionMask.from_dict(data["mask"])

        # Compat: JSON antigo armazena `coverage_pct`. Traduzimos
        # via `sqrt` (mesma derivação que o `_apply_rectangle_black`
        # legado usava) — preserva exatamente a geometria salva. Se o
        # JSON já tem `width_pct`/`height_pct`, use-os direto.
        width_pct = data.get("width_pct")
        height_pct = data.get("height_pct")
        if width_pct is None or height_pct is None:
            ratio = math.sqrt(float(data.get("coverage_pct", 0.25)))
            if width_pct is None:
                width_pct = ratio
            if height_pct is None:
                height_pct = ratio

        return cls(
            kind=OcclusionKind(
                data.get("kind", OcclusionKind.RECTANGLE_BLACK.value)
            ),
            position=OcclusionPosition(
                data.get("position", OcclusionPosition.CENTER.value)
            ),
            width_pct=float(width_pct),
            height_pct=float(height_pct),
            radius_pct=float(data.get("radius_pct", 0.10)),
            mask=mask,
        )
