"""Config de recorte (PR I1).

Modos suportados em I1:
  - CENTERED — recorta um retângulo centralizado de
    `(target_width, target_height)`. Erro se a imagem for menor que
    o destino (sem padding automático — explicitar é mais seguro
    forensicamente).
  - FACE_DETECTED — usa `FFService.align_face_only`-style detect
    (apenas detect, sem alinhamento) → bbox da maior face → corta
    bbox + padding opcional. Sem rotação — preserva pose original.

USER_RECTANGLE entra em I6 quando o canvas interativo for criado
(armazena (x, y, w, h) explícitos no config).

A margem é expressa como **fator** sobre a bbox — a mesma unidade da
extração de faces de vídeo (`VideoExtractionConfig.margin`), que usa
`forensicface.geometry.extend_bbox`. `1.0` = bbox exata, `2.0` = o dobro
em cada dimensão. Ter duas unidades para a mesma coisa no mesmo app era
armadilha: o mesmo número produzia enquadramentos diferentes conforme a
tela.

`face_padding` (fração por lado) é a unidade ANTIGA, mantida só para ler
configs gravados antes da unificação: `fator = 1 + 2 × fração`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from faceaudit.enums.crop_mode import CropMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
    enum_choices,
)


def margin_factor_from_padding(face_padding: float) -> float:
    """Converte a unidade antiga (fração por lado) em fator sobre a bbox.

    Expandir cada lado em `p` deixa a caixa com `1 + 2p` da largura
    original — daí a fórmula. `0.2` (20% por lado) vira `1.4`.
    """
    try:
        pad = max(0.0, float(face_padding))
    except (TypeError, ValueError):
        pad = 0.0
    return 1.0 + 2.0 * pad


@dataclass
class CropConfig(ImageProcessingStepConfig):
    """Recorte conforme o modo escolhido."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.CROP
    mode: CropMode = CropMode.CENTERED
    # CENTERED — alvo
    target_width: int = 224
    target_height: int = 224
    # FACE_DETECTED — margem como FATOR sobre a bbox (1.0 = bbox exata).
    # `None` significa "não informado": cai na conversão do `face_padding`,
    # que é como os configs gravados antes da unificação são lidos.
    margin_factor: Optional[float] = None
    # Unidade antiga (fração por lado). Só para migrar configs gravados
    # antes da unificação — não usar em código novo.
    face_padding: float = 0.0

    def __post_init__(self):
        if self.target_width < 1:
            self.target_width = 1
        if self.target_height < 1:
            self.target_height = 1
        if self.face_padding < 0:
            self.face_padding = 0.0
        # Config antigo (só `face_padding`) → converte para fator. Se o
        # fator veio explícito, ele manda.
        if self.margin_factor is None:
            self.margin_factor = margin_factor_from_padding(self.face_padding)
        self.margin_factor = max(1.0, float(self.margin_factor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "mode": self.mode.value,
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "margin_factor": float(self.margin_factor),
            # Gravado junto por compatibilidade com leitores antigos —
            # derivado do fator, que é o que o processador usa.
            "face_padding": round((float(self.margin_factor) - 1.0) / 2.0, 6),
        }

    def param_specs(self) -> List[StepParamSpec]:
        centrado = lambda c: c.mode == CropMode.CENTERED  # noqa: E731
        return [
            StepParamSpec(
                "mode", "Modo", StepParamKind.CHOICE,
                choices=enum_choices([CropMode.CENTERED, CropMode.FACE_DETECTED]),
            ),
            StepParamSpec(
                "target_width", "Largura", StepParamKind.INT,
                minimum=8, maximum=4096, step=1, suffix=" px", visible=centrado,
            ),
            StepParamSpec(
                "target_height", "Altura", StepParamKind.INT,
                minimum=8, maximum=4096, step=1, suffix=" px", visible=centrado,
            ),
            StepParamSpec(
                "margin_factor", "Margem", StepParamKind.FLOAT,
                minimum=1.0, maximum=5.0, step=0.1, decimals=2, suffix=" ×",
                visible=lambda c: not centrado(c),
                help="Fator sobre a caixa da face: 1,0 = caixa exata, "
                     "2,0 = o dobro em cada dimensão. Mesma unidade da "
                     "extração de faces de vídeo.",
            ),
        ]

    def summary_pt(self) -> str:
        if self.mode == CropMode.CENTERED:
            return f"{int(self.target_width)}×{int(self.target_height)} px"
        return f"face, margem {float(self.margin_factor):g}×"

    @property
    def requires_face(self) -> bool:
        return self.mode == CropMode.FACE_DETECTED

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CropConfig":
        return cls(
            mode=CropMode(data.get("mode", CropMode.CENTERED.value)),
            target_width=int(data.get("target_width", 224)),
            target_height=int(data.get("target_height", 224)),
            face_padding=float(data.get("face_padding", 0.0)),
            # Ausente = config anterior à unificação → converte do padding.
            margin_factor=(
                float(data["margin_factor"])
                if data.get("margin_factor") is not None else None
            ),
        )
