"""Modelos de pipeline de imagens que vêm com o FaceAudit.

Ponto de partida, não receita: os valores vieram de um caso real e cada
questionada tem a sua copiadora e o seu scanner. O caminho é carregar o
modelo, carregar a questionada e usar "Ajustar à questionada".

Ficam no código (versionados) e não na biblioteca global: a biblioteca é
por máquina e não está no git.
"""

from __future__ import annotations

from typing import Callable, Dict

from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)


def _xerox_de_documento() -> ImageProcessingPipelineConfig:
    return ImageProcessingPipelineConfig.from_dict({
        "name": "xerox_de_documento",
        "description": (
            "Modelo: cópia (xerox) de documento com foto, digitalizada. "
            "Copiadora: curva de tons por percentis da face (absorve a "
            "exposição de cada foto), meio-tom por difusão de erro numa "
            "resolução intermediária. Digitalização: redução ao IPD da "
            "questionada e ajuste de contraste. Valores de um caso real "
            "(questionada com IPD ~49 px) — carregue a questionada e use "
            "'Ajustar à questionada'."
        ),
        "steps": [
            {"step_type": "grayscale"},
            {"step_type": "tone_curve", "mode": "face_percentiles",
             "black_percentile": 17.5, "white_percentile": 70.0, "gamma": 1.06},
            {"step_type": "resize", "mode": "target_ipd", "target_ipd_px": 84.0},
            {"step_type": "halftone", "mode": "error_diffusion"},
            {"step_type": "resize", "mode": "target_ipd", "use_questioned_ipd": True},
            {"step_type": "tone_curve", "black_point": 24, "white_point": 223,
             "gamma": 1.09},
        ],
    })


BUILTIN_TEMPLATES: Dict[str, Callable[[], ImageProcessingPipelineConfig]] = {
    "Xerox de documento": _xerox_de_documento,
}
