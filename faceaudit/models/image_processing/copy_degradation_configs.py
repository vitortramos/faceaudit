"""Configs dos passos que reproduzem impressão, cópia e digitalização.

Pensados para adaptar uma base de referência às condições de uma imagem
questionada que passou por xerox de documento: a foto impressa no
documento é copiada (contraste estourado, toner que espalha, meio-tom) e
a cópia é digitalizada em resolução baixa. Cada etapa física é um passo
— a ordem fica com o perito, e cada um é descrevível num laudo.

  - Redimensionar: escala por distância entre os olhos alvo ou por fator.
  - Tons de cinza.
  - Curva de tons: ponto de preto, ponto de branco e gama.
  - Ganho de toner: expansão das áreas escuras.
  - Meio-tom: difusão de erro, pontilhado ordenado ou retícula de pontos.
  - Ruído de toner: pontos escuros soltos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from faceaudit.enums.halftone_mode import HalftoneMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.resize_mode import ResizeInterpolation, ResizeMode
from faceaudit.enums.tone_curve_mode import ToneCurveMode
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
    enum_choices,
)


def _faixa(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── Redimensionar ──────────────────────────────────────────────────────────

@dataclass
class ResizeConfig(ImageProcessingStepConfig):
    """Muda a resolução da imagem.

    `TARGET_IPD` escala cada imagem para que a distância entre os olhos
    fique em `target_ipd_px` — numa base em que a face tem tamanhos
    diferentes em cada foto, é o que iguala a resolução da FACE à da
    questionada. Pode aparecer duas vezes na cadeia: ampliar até a
    resolução da copiadora antes do meio-tom e reduzir à da questionada
    depois.

    `restore_size`: depois de reescalar, volta ao tamanho original —
    perde a resolução e mantém as dimensões em pixels.

    `use_questioned_ipd`: o IPD alvo é o medido na questionada do
    pipeline (resolvido pelo pipeline ao montar os passos), em vez do
    número digitado.
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.RESIZE
    mode: ResizeMode = ResizeMode.TARGET_IPD
    target_ipd_px: float = 40.0
    factor: float = 0.5
    interpolation: ResizeInterpolation = ResizeInterpolation.AUTO
    restore_size: bool = False
    use_questioned_ipd: bool = False

    def __post_init__(self):
        self.target_ipd_px = _faixa(float(self.target_ipd_px), 2.0, 2000.0)
        self.factor = _faixa(float(self.factor), 0.01, 20.0)

    @property
    def requires_face(self) -> bool:
        return self.mode == ResizeMode.TARGET_IPD

    def param_specs(self) -> List[StepParamSpec]:
        por_ipd = lambda c: c.mode == ResizeMode.TARGET_IPD  # noqa: E731
        return [
            StepParamSpec("mode", "Escala por", StepParamKind.CHOICE,
                          choices=enum_choices(list(ResizeMode))),
            StepParamSpec(
                "use_questioned_ipd", "IPD da questionada", StepParamKind.BOOL,
                visible=por_ipd,
                help="Usa a distância entre os olhos medida na imagem "
                     "questionada do pipeline. É o que se quer no "
                     "redimensionamento que leva a base à resolução da "
                     "questionada.",
            ),
            StepParamSpec(
                "target_ipd_px", "IPD alvo", StepParamKind.FLOAT,
                minimum=2.0, maximum=2000.0, step=1.0, decimals=1, suffix=" px",
                visible=lambda c: por_ipd(c) and not c.use_questioned_ipd,
                tunable=True, search_min=30.0, search_max=200.0,
                help="Distância entre os centros dos olhos, em pixels, que a "
                     "imagem terá depois do passo. Meça na imagem questionada.",
            ),
            StepParamSpec(
                "factor", "Fator", StepParamKind.FLOAT,
                minimum=0.01, maximum=20.0, step=0.05, decimals=3, suffix=" ×",
                visible=lambda c: not por_ipd(c),
                tunable=True, search_min=0.1, search_max=2.0,
                help="0,5 = metade do tamanho; 2,0 = o dobro.",
            ),
            StepParamSpec("interpolation", "Interpolação", StepParamKind.CHOICE,
                          choices=enum_choices(list(ResizeInterpolation))),
            StepParamSpec(
                "restore_size", "Voltar ao tamanho original", StepParamKind.BOOL,
                help="Depois de reescalar, volta às dimensões originais: a "
                     "imagem perde resolução sem mudar de tamanho.",
            ),
        ]

    def summary_pt(self) -> str:
        if self.mode != ResizeMode.TARGET_IPD:
            base = f"{self.factor:g}×"
        elif self.use_questioned_ipd:
            base = "IPD da questionada"
        else:
            base = f"IPD {self.target_ipd_px:g} px"
        return base + (", volta ao tamanho" if self.restore_size else "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "mode": self.mode.value,
            "target_ipd_px": float(self.target_ipd_px),
            "factor": float(self.factor),
            "interpolation": self.interpolation.value,
            "restore_size": bool(self.restore_size),
            "use_questioned_ipd": bool(self.use_questioned_ipd),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResizeConfig":
        return cls(
            mode=ResizeMode(data.get("mode", ResizeMode.TARGET_IPD.value)),
            target_ipd_px=float(data.get("target_ipd_px", 40.0)),
            factor=float(data.get("factor", 0.5)),
            interpolation=ResizeInterpolation(
                data.get("interpolation", ResizeInterpolation.AUTO.value)),
            restore_size=bool(data.get("restore_size", False)),
            use_questioned_ipd=bool(data.get("use_questioned_ipd", False)),
        )


# ── Tons de cinza ──────────────────────────────────────────────────────────

@dataclass
class GrayscaleConfig(ImageProcessingStepConfig):
    """Converte para tons de cinza (luminância ITU-R BT.601), mantendo
    três canais iguais para os passos seguintes."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.GRAYSCALE

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrayscaleConfig":
        return cls()


# ── Curva de tons ──────────────────────────────────────────────────────────

@dataclass
class ToneCurveConfig(ImageProcessingStepConfig):
    """Níveis: o que está abaixo do ponto de preto vira preto, acima do
    ponto de branco vira branco, e o meio é esticado com a gama.

    `saída = ((entrada − preto) / (branco − preto)) ^ gama`. Gama > 1
    escurece os meios-tons; < 1 clareia. Reproduz o contraste estourado
    da copiadora: preto e branco próximos fecham as sombras e queimam a
    pele.

    Em `FACE_PERCENTILES`, preto e branco são os percentis
    `black_percentile` e `white_percentile` dos níveis de cinza da região
    da face, calculados em CADA imagem: a mesma fração da face vira preto
    em todas as fotos, qualquer que seja a exposição original.
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.TONE_CURVE
    black_point: int = 40
    white_point: int = 200
    gamma: float = 1.0
    mode: ToneCurveMode = ToneCurveMode.FIXED_LEVELS
    black_percentile: float = 20.0
    white_percentile: float = 60.0

    @property
    def requires_face(self) -> bool:
        return self.mode == ToneCurveMode.FACE_PERCENTILES

    def __post_init__(self):
        self.black_point = int(_faixa(int(self.black_point), 0, 254))
        self.white_point = int(_faixa(int(self.white_point), 1, 255))
        if self.white_point <= self.black_point:
            self.white_point = min(255, self.black_point + 1)
        self.gamma = _faixa(float(self.gamma), 0.1, 5.0)
        self.black_percentile = _faixa(float(self.black_percentile), 0.0, 99.0)
        self.white_percentile = _faixa(float(self.white_percentile), 1.0, 100.0)
        if self.white_percentile <= self.black_percentile:
            self.white_percentile = min(100.0, self.black_percentile + 1.0)

    def param_specs(self) -> List[StepParamSpec]:
        fixo = lambda c: c.mode == ToneCurveMode.FIXED_LEVELS  # noqa: E731
        return [
            StepParamSpec(
                "mode", "Pontos definidos por", StepParamKind.CHOICE,
                choices=enum_choices(list(ToneCurveMode)),
                help="<b>Níveis fixos</b>: os mesmos para todas as imagens.<br>"
                     "<b>Percentis da face</b>: calculados em cada imagem, na "
                     "região da face — a mesma fração da face vira preto e "
                     "branco em todas as fotos, qualquer que seja a exposição "
                     "original.",
            ),
            StepParamSpec("black_point", "Ponto de preto", StepParamKind.INT,
                          minimum=0, maximum=254, step=1, visible=fixo,
                          tunable=True, search_min=0, search_max=150,
                          help="Nível de cinza (0–255) a partir do qual tudo "
                               "vira preto."),
            StepParamSpec("white_point", "Ponto de branco", StepParamKind.INT,
                          minimum=1, maximum=255, step=1, visible=fixo,
                          tunable=True, search_min=100, search_max=255,
                          help="Nível de cinza (0–255) a partir do qual tudo "
                               "vira branco."),
            StepParamSpec("black_percentile", "Percentil do preto", StepParamKind.FLOAT,
                          minimum=0.0, maximum=99.0, step=1.0, decimals=1,
                          suffix=" %", visible=lambda c: not fixo(c),
                          tunable=True, search_min=0.0, search_max=60.0,
                          help="Fração da face (em %) que fica abaixo do ponto "
                               "de preto: 20 = os 20% mais escuros viram preto."),
            StepParamSpec("white_percentile", "Percentil do branco", StepParamKind.FLOAT,
                          minimum=1.0, maximum=100.0, step=1.0, decimals=1,
                          suffix=" %", visible=lambda c: not fixo(c),
                          tunable=True, search_min=30.0, search_max=100.0,
                          help="Percentil a partir do qual vira branco: 60 = os "
                               "40% mais claros viram branco."),
            StepParamSpec("gamma", "Gama", StepParamKind.FLOAT,
                          minimum=0.1, maximum=5.0, step=0.05, decimals=2,
                          tunable=True, search_min=0.5, search_max=2.0,
                          help="Maior que 1 escurece os meios-tons; menor que "
                               "1 clareia."),
        ]

    def summary_pt(self) -> str:
        if self.mode == ToneCurveMode.FACE_PERCENTILES:
            return (f"face P{self.black_percentile:g}–P{self.white_percentile:g}, "
                    f"γ {self.gamma:g}")
        return f"{self.black_point}–{self.white_point}, γ {self.gamma:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "black_point": int(self.black_point),
                "white_point": int(self.white_point), "gamma": float(self.gamma),
                "mode": self.mode.value,
                "black_percentile": float(self.black_percentile),
                "white_percentile": float(self.white_percentile)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToneCurveConfig":
        return cls(black_point=int(data.get("black_point", 40)),
                   white_point=int(data.get("white_point", 200)),
                   gamma=float(data.get("gamma", 1.0)),
                   mode=ToneCurveMode(data.get("mode", ToneCurveMode.FIXED_LEVELS.value)),
                   black_percentile=float(data.get("black_percentile", 20.0)),
                   white_percentile=float(data.get("white_percentile", 60.0)))


# ── Ganho de toner ─────────────────────────────────────────────────────────

@dataclass
class TonerGainConfig(ImageProcessingStepConfig):
    """Expansão das áreas escuras: o toner espalha, e barba, sobrancelha
    e sombras engrossam. Erosão em níveis de cinza com disco de raio
    `radius_px`, misturada à imagem por `intensity`."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.TONER_GAIN
    radius_px: int = 1
    intensity: float = 1.0

    def __post_init__(self):
        self.radius_px = int(_faixa(int(self.radius_px), 0, 20))
        self.intensity = _faixa(float(self.intensity), 0.0, 1.0)

    def param_specs(self) -> List[StepParamSpec]:
        return [
            StepParamSpec("radius_px", "Raio", StepParamKind.INT,
                          minimum=0, maximum=20, step=1, suffix=" px",
                          tunable=True, search_min=0, search_max=4,
                          help="Quanto as áreas escuras avançam sobre as claras, "
                               "na resolução em que o passo roda."),
            StepParamSpec("intensity", "Intensidade", StepParamKind.FLOAT,
                          minimum=0.0, maximum=1.0, step=0.05, decimals=2,
                          tunable=True,
                          help="0 = sem efeito; 1 = expansão completa."),
        ]

    def summary_pt(self) -> str:
        return f"raio {self.radius_px} px, {self.intensity:g}"

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "radius_px": int(self.radius_px),
                "intensity": float(self.intensity)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TonerGainConfig":
        return cls(radius_px=int(data.get("radius_px", 1)),
                   intensity=float(data.get("intensity", 1.0)))


# ── Meio-tom ───────────────────────────────────────────────────────────────

@dataclass
class HalftoneConfig(ImageProcessingStepConfig):
    """Reduz a imagem a preto e branco puros simulando a impressão.

    O resultado é binário NA RESOLUÇÃO EM QUE O PASSO RODA. A cópia
    digitalizada tem cinzas porque o scanner (ou a redução) junta os
    pontos — por isso o meio-tom vem antes do Redimensionar que leva à
    resolução da questionada, e de preferência numa resolução maior.
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.HALFTONE
    mode: HalftoneMode = HalftoneMode.ERROR_DIFFUSION
    cell_size_px: float = 6.0
    angle_deg: float = 45.0

    def __post_init__(self):
        self.cell_size_px = _faixa(float(self.cell_size_px), 2.0, 64.0)
        self.angle_deg = float(self.angle_deg) % 180.0

    def param_specs(self) -> List[StepParamSpec]:
        retcula = lambda c: c.mode == HalftoneMode.AM_SCREEN  # noqa: E731
        return [
            StepParamSpec("mode", "Técnica", StepParamKind.CHOICE,
                          choices=enum_choices(list(HalftoneMode))),
            StepParamSpec("cell_size_px", "Tamanho da célula", StepParamKind.FLOAT,
                          minimum=2.0, maximum=64.0, step=0.5, decimals=1,
                          suffix=" px", visible=retcula,
                          tunable=True, search_min=2.0, search_max=12.0,
                          help="Distância entre pontos da retícula, em pixels."),
            StepParamSpec("angle_deg", "Ângulo", StepParamKind.FLOAT,
                          minimum=0.0, maximum=179.9, step=5.0, decimals=1,
                          suffix=" °", visible=retcula,
                          help="Inclinação da grade de pontos. 45° é o "
                               "padrão para preto."),
        ]

    def summary_pt(self) -> str:
        if self.mode == HalftoneMode.AM_SCREEN:
            return f"retícula {self.cell_size_px:g} px, {self.angle_deg:g}°"
        return self.mode.label_pt.split(" (")[0].lower()

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "mode": self.mode.value,
                "cell_size_px": float(self.cell_size_px),
                "angle_deg": float(self.angle_deg)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HalftoneConfig":
        return cls(mode=HalftoneMode(data.get("mode", HalftoneMode.ERROR_DIFFUSION.value)),
                   cell_size_px=float(data.get("cell_size_px", 6.0)),
                   angle_deg=float(data.get("angle_deg", 45.0)))


# ── Ruído de toner ─────────────────────────────────────────────────────────

@dataclass
class TonerNoiseConfig(ImageProcessingStepConfig):
    """Pontos escuros soltos: toner fora do lugar, poeira no vidro da
    copiadora. Posições sorteadas pela semente do pipeline."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.TONER_NOISE
    density: float = 0.002
    dot_radius_px: int = 0

    def __post_init__(self):
        self.density = _faixa(float(self.density), 0.0, 0.05)
        self.dot_radius_px = int(_faixa(int(self.dot_radius_px), 0, 5))

    def param_specs(self) -> List[StepParamSpec]:
        return [
            StepParamSpec("density", "Densidade", StepParamKind.FLOAT,
                          minimum=0.0, maximum=0.05, step=0.0005, decimals=4,
                          tunable=True, search_min=0.0, search_max=0.01,
                          help="Fração dos pixels que recebe um ponto "
                               "(0,002 = 2 por mil)."),
            StepParamSpec("dot_radius_px", "Raio do ponto", StepParamKind.INT,
                          minimum=0, maximum=5, step=1, suffix=" px",
                          help="0 = um pixel."),
        ]

    def summary_pt(self) -> str:
        return f"{self.density * 1000:g}‰"

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict(), "density": float(self.density),
                "dot_radius_px": int(self.dot_radius_px)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TonerNoiseConfig":
        return cls(density=float(data.get("density", 0.002)),
                   dot_radius_px=int(data.get("dot_radius_px", 0)))
