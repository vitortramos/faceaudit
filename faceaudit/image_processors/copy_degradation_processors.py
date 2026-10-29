"""Processadores dos passos de impressão, cópia e digitalização.

Ver `faceaudit/models/image_processing/copy_degradation_configs.py` para
o papel de cada um na cadeia da xerox de documento.

Todos operam em BGR uint8 e devolvem BGR uint8 (tons de cinza saem com
os três canais iguais, para os passos seguintes não precisarem tratar
imagem de um canal). Só o Redimensionar muda o tamanho, e é o único que
converte a geometria da face; os demais a mantêm.
"""

from __future__ import annotations

from random import Random
from typing import Optional, Tuple

import cv2
import numpy as np

from faceaudit.enums.halftone_mode import HalftoneMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.resize_mode import ResizeInterpolation, ResizeMode
from faceaudit.enums.tone_curve_mode import ToneCurveMode
from faceaudit.image_processors.image_step_base import ImageStepProcessorBase
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)

# Teto do lado maior depois de reescalar: um IPD alvo digitado errado (ou
# uma face minúscula no dataset) não pode pedir uma imagem de gigapixels.
_LADO_MAXIMO = 16000

_INTERPOLACAO = {
    ResizeInterpolation.AREA: cv2.INTER_AREA,
    ResizeInterpolation.LINEAR: cv2.INTER_LINEAR,
    ResizeInterpolation.CUBIC: cv2.INTER_CUBIC,
    ResizeInterpolation.NEAREST: cv2.INTER_NEAREST,
}


def _cinza(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ── Redimensionar ──────────────────────────────────────────────────────────

class ResizeProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.RESIZE

    def _escala(self, face: Optional[ImageFaceGeometry]) -> float:
        cfg = self._config
        if cfg.mode == ResizeMode.FACTOR:
            return float(cfg.factor)
        face = self._require_face(face, "Redimensionar pela distância entre os olhos")
        ipd = face.ipd
        if ipd is None or ipd <= 0:
            raise RuntimeError(
                "Redimensionar pela distância entre os olhos precisa dos "
                "pontos dos olhos, e eles não estão disponíveis."
            )
        return float(cfg.target_ipd_px) / ipd

    def _interp(self, ampliando: bool) -> int:
        modo = self._config.interpolation
        if modo == ResizeInterpolation.AUTO:
            return cv2.INTER_CUBIC if ampliando else cv2.INTER_AREA
        return _INTERPOLACAO[modo]

    def apply_step(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> Tuple[np.ndarray, Optional[ImageFaceGeometry]]:
        h, w = image.shape[:2]
        s = self._escala(face)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        if max(nw, nh) > _LADO_MAXIMO:
            raise ValueError(
                f"Redimensionar: a escala {s:.2f}× levaria a imagem de "
                f"{w}×{h} a {nw}×{nh} px, acima do limite de {_LADO_MAXIMO} px."
            )
        out = cv2.resize(image, (nw, nh), interpolation=self._interp(s > 1.0))
        geom = None if face is None else face.scaled(nw / w, nh / h)

        if self._config.restore_size and (nw, nh) != (w, h):
            out = cv2.resize(out, (w, h), interpolation=self._interp(s < 1.0))
            geom = None if geom is None else geom.scaled(w / nw, h / nh)
        return out, geom


# ── Tons de cinza ──────────────────────────────────────────────────────────

class GrayscaleProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.GRAYSCALE

    def _process(self, image, *, rng, face) -> np.ndarray:
        return _bgr(_cinza(image))


# ── Curva de tons ──────────────────────────────────────────────────────────

def tone_curve_lut(black: int, white: int, gamma: float,
                   input_levels: int = 256) -> np.ndarray:
    """`input_levels` < 256: entre preto e branco a imagem tinha só esse
    número de níveis quando o contraste foi esticado (digitalização de
    pouco contraste esticada depois, em 8 bits) — a saída fica com no
    máximo esse número de níveis, espaçados: o histograma em pente."""
    x = np.arange(256, dtype=np.float64)
    t = np.clip((x - black) / float(max(1, white - black)), 0.0, 1.0)
    if input_levels < 256:
        n = max(2, int(input_levels))
        t = np.round(t * (n - 1)) / (n - 1)
    return np.round(255.0 * t ** float(gamma)).astype(np.uint8)


def face_region_gray(image: np.ndarray, face: ImageFaceGeometry) -> np.ndarray:
    """Níveis de cinza da região da face (caixa presa ao quadro)."""
    gray = _cinza(image)
    caixa = face.face_box()
    if caixa is None:
        raise RuntimeError("Sem caixa nem pontos da face.")
    h, w = gray.shape
    x1, y1 = max(0, int(np.floor(caixa[0]))), max(0, int(np.floor(caixa[1])))
    x2, y2 = min(w, int(np.ceil(caixa[2]))), min(h, int(np.ceil(caixa[3])))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise RuntimeError("A caixa da face está fora da imagem.")
    return gray[y1:y2, x1:x2]


class ToneCurveProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.TONE_CURVE

    def levels_for(self, image: np.ndarray, face: Optional[ImageFaceGeometry]) -> Tuple[int, int]:
        """Ponto de preto e de branco efetivos para ESTA imagem."""
        c = self._config
        if c.mode != ToneCurveMode.FACE_PERCENTILES:
            return int(c.black_point), int(c.white_point)
        regiao = face_region_gray(image, self._require_face(face, "A curva por percentis"))
        preto, branco = np.percentile(regiao, [c.black_percentile, c.white_percentile])
        preto = int(round(preto))
        branco = max(preto + 1, int(round(branco)))
        return preto, min(255, branco)

    def _process(self, image, *, rng, face) -> np.ndarray:
        preto, branco = self.levels_for(image, face)
        return cv2.LUT(image, tone_curve_lut(preto, branco, self._config.gamma,
                                             self._config.input_levels))


# ── Ganho de toner ─────────────────────────────────────────────────────────

class TonerGainProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.TONER_GAIN

    def _process(self, image, *, rng, face) -> np.ndarray:
        r = int(self._config.radius_px)
        i = float(self._config.intensity)
        if r <= 0 or i <= 0:
            return image.copy()
        disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        erodida = cv2.erode(image, disco)
        return cv2.addWeighted(image, 1.0 - i, erodida, i, 0.0)


# ── Meio-tom ───────────────────────────────────────────────────────────────

def _bayer(n: int = 8) -> np.ndarray:
    m = np.array([[0, 2], [3, 1]], dtype=np.float64)
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size  # limiares em (0, 1)


def halftone(gray: np.ndarray, mode: HalftoneMode, cell: float, angle_deg: float) -> np.ndarray:
    """Imagem binária (0/255) a partir de uma imagem em tons de cinza."""
    h, w = gray.shape
    if mode == HalftoneMode.ERROR_DIFFUSION:
        from PIL import Image

        binaria = Image.fromarray(gray, mode="L").convert(
            "1", dither=Image.Dither.FLOYDSTEINBERG)
        return np.asarray(binaria, dtype=np.uint8) * 255
    if mode == HalftoneMode.ORDERED:
        b = _bayer(8)
        limiar = np.tile(b, (h // 8 + 1, w // 8 + 1))[:h, :w]
        return np.where(gray / 255.0 > limiar, 255, 0).astype(np.uint8)
    if mode == HalftoneMode.AM_SCREEN:
        a = np.deg2rad(angle_deg)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        u = (xx * np.cos(a) + yy * np.sin(a)) * (2 * np.pi / cell)
        v = (-xx * np.sin(a) + yy * np.cos(a)) * (2 * np.pi / cell)
        # Função de ponto: 1 no centro das células, 0 entre elas. O pixel
        # fica preto onde a função passa do claro local — o ponto cresce
        # com o escuro.
        ponto = (np.cos(u) + np.cos(v) + 2.0) / 4.0
        return np.where(ponto > gray / 255.0, 0, 255).astype(np.uint8)
    raise NotImplementedError(f"Meio-tom {mode} não implementado.")


class HalftoneProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.HALFTONE

    def _process(self, image, *, rng, face) -> np.ndarray:
        c = self._config
        return _bgr(halftone(_cinza(image), c.mode, c.cell_size_px, c.angle_deg))


# ── Ruído de toner ─────────────────────────────────────────────────────────

class TonerNoiseProcessor(ImageStepProcessorBase):
    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.TONER_NOISE

    def _process(self, image, *, rng, face) -> np.ndarray:
        h, w = image.shape[:2]
        n = int(round(float(self._config.density) * h * w))
        out = image.copy()
        if n <= 0:
            return out
        semente = rng.getrandbits(64) if rng is not None else 0
        g = np.random.default_rng(semente)
        mascara = np.zeros((h, w), np.uint8)
        mascara[g.integers(0, h, n), g.integers(0, w, n)] = 255
        r = int(self._config.dot_radius_px)
        if r > 0:
            disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            mascara = cv2.dilate(mascara, disco)
        out[mascara > 0] = 0
        return out
