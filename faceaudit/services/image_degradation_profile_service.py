"""Medidas de degradação de uma imagem de face, e comparação com a
imagem questionada que o pipeline tenta reproduzir.

Adaptar uma base às condições de uma questionada só é defensável se dá
para dizer QUANTO a base adaptada se parece com ela. Estas medidas tornam
isso objetivo: o perito ajusta o pipeline até as medidas da referência
processada ficarem próximas das da questionada, e o Processar registra,
no manifesto e na cadeia de custódia, a comparação da mediana da base
com a questionada.

As medidas são feitas na REGIÃO DA FACE (caixa da face), em tons de
cinza: o fundo varia entre fotos (papel branco de uma cópia, fundo de
documento) e dominaria as frações de preto e branco sem dizer nada sobre
a face. Sem a posição da face, a imagem inteira é medida e a medida diz
isso (`region`).

Módulo de backend, sem Qt.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)

# Faixas de nível de cinza (0–255).
PRETO_ATE = 20
BRANCO_DESDE = 235
MEIO_TOM = (60, 195)

REGIAO_FACE = "face"
REGIAO_IMAGEM = "imagem inteira"


@dataclass(frozen=True)
class DegradationMeasures:
    """Medidas de uma imagem. Frações em [0, 1]."""

    width: int
    height: int
    region: str
    is_grayscale: bool
    ipd_px: Optional[float]
    black_fraction: float
    white_fraction: float
    midtone_fraction: float
    distinct_levels: int
    grain: float
    mean_level: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["DegradationMeasures"]:
        if not data:
            return None
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in nomes})


@dataclass(frozen=True)
class MeasureSpec:
    """Como exibir e comparar uma medida."""

    key: str
    label: str
    help: str
    # Tolerância para "próximo da questionada": absoluta (`abs_tol`) ou
    # relativa (`rel_tol`, em razão: 0,3 = até 30% para mais ou menos).
    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = None
    kind: str = "float"  # float | fraction | int | bool | px

    def format(self, v: Any) -> str:
        if v is None:
            return "—"
        if self.kind == "fraction":
            return f"{100 * float(v):.1f}%".replace(".", ",")
        if self.kind == "px":
            return f"{float(v):.1f} px".replace(".", ",")
        if self.kind == "int":
            return f"{int(round(float(v)))}"
        if self.kind == "bool":
            return "sim" if v else "não"
        return f"{float(v):,.0f}".replace(",", ".")

    def tolerance_text(self) -> str:
        if self.abs_tol is not None:
            if self.kind == "fraction":
                return f"± {100 * self.abs_tol:g} pontos percentuais".replace(".", ",")
            return f"± {self.abs_tol:g}".replace(".", ",")
        if self.rel_tol is not None:
            return f"± {100 * self.rel_tol:g}%".replace(".", ",")
        return "igual"

    def is_close(self, target: Any, value: Any) -> Optional[bool]:
        if target is None or value is None:
            return None
        if self.kind == "bool":
            return bool(target) == bool(value)
        t, v = float(target), float(value)
        if self.abs_tol is not None:
            return abs(v - t) <= self.abs_tol
        if self.rel_tol is not None:
            if t <= 0 or v <= 0:
                return t == v
            return abs(math.log(v / t)) <= math.log1p(self.rel_tol)
        return t == v


MEASURES: Tuple[MeasureSpec, ...] = (
    MeasureSpec("ipd_px", "Distância entre os olhos",
                "Distância entre os centros dos olhos, em pixels: a "
                "resolução da face.", rel_tol=0.10, kind="px"),
    MeasureSpec("black_fraction", f"Pretos (≤ {PRETO_ATE})",
                "Fração da região da face em preto (ou quase).",
                abs_tol=0.05, kind="fraction"),
    MeasureSpec("white_fraction", f"Brancos (≥ {BRANCO_DESDE})",
                "Fração da região da face em branco (ou quase).",
                abs_tol=0.05, kind="fraction"),
    MeasureSpec("midtone_fraction", f"Meios-tons ({MEIO_TOM[0]}–{MEIO_TOM[1]})",
                "Fração da região da face em tons intermediários. Cópia de "
                "contraste estourado tem pouca.", abs_tol=0.05, kind="fraction"),
    MeasureSpec("grain", "Grão (variância do laplaciano)",
                "Energia de alta frequência: grão de meio-tom, ruído, nitidez. "
                "Depende da resolução — compare com a mesma distância entre "
                "os olhos.", rel_tol=0.30),
    MeasureSpec("distinct_levels", "Níveis de cinza distintos",
                "Quantos dos 256 níveis aparecem na região.",
                rel_tol=0.25, kind="int"),
    MeasureSpec("mean_level", "Nível médio",
                "Média dos níveis de cinza da região (0–255).",
                abs_tol=15.0, kind="int"),
    MeasureSpec("is_grayscale", "Tons de cinza", "Se os três canais são iguais.",
                kind="bool"),
)
MEASURES_BY_KEY = {m.key: m for m in MEASURES}


def _regiao(gray: np.ndarray, face: Optional[ImageFaceGeometry]) -> Tuple[np.ndarray, str]:
    if face is not None:
        caixa = face.face_box()
        if caixa is not None:
            h, w = gray.shape
            x1, y1 = max(0, int(math.floor(caixa[0]))), max(0, int(math.floor(caixa[1])))
            x2, y2 = min(w, int(math.ceil(caixa[2]))), min(h, int(math.ceil(caixa[3])))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                return gray[y1:y2, x1:x2], REGIAO_FACE
    return gray, REGIAO_IMAGEM


def measure(
    image_bgr: np.ndarray, face: Optional[ImageFaceGeometry] = None,
) -> DegradationMeasures:
    """Mede a imagem (BGR uint8) na região da face, se conhecida."""
    if image_bgr.ndim == 2:
        gray = image_bgr
        cinza = True
    else:
        canais = image_bgr[..., :3].astype(np.int16)
        cinza = bool(np.abs(canais[..., 0] - canais[..., 1]).max() <= 2
                     and np.abs(canais[..., 1] - canais[..., 2]).max() <= 2)
        gray = cv2.cvtColor(image_bgr[..., :3], cv2.COLOR_BGR2GRAY)
    regiao, nome = _regiao(gray, face)
    hist = np.bincount(regiao.ravel(), minlength=256) / regiao.size
    return DegradationMeasures(
        width=int(image_bgr.shape[1]),
        height=int(image_bgr.shape[0]),
        region=nome,
        is_grayscale=cinza,
        ipd_px=None if face is None or face.ipd is None else float(face.ipd),
        black_fraction=float(hist[: PRETO_ATE + 1].sum()),
        white_fraction=float(hist[BRANCO_DESDE:].sum()),
        midtone_fraction=float(hist[MEIO_TOM[0]: MEIO_TOM[1] + 1].sum()),
        distinct_levels=int((hist > 0).sum()),
        grain=float(cv2.Laplacian(regiao, cv2.CV_64F).var()),
        mean_level=float(regiao.mean()),
    )


@dataclass(frozen=True)
class MeasureComparison:
    key: str
    label: str
    target: Any
    value: Any
    close: Optional[bool]
    tolerance: str


def compare(
    target: Optional[DegradationMeasures], value: Optional[DegradationMeasures],
) -> List[MeasureComparison]:
    """Compara medida a medida; `close` None quando falta um dos lados."""
    linhas = []
    for m in MEASURES:
        t = None if target is None else getattr(target, m.key)
        v = None if value is None else getattr(value, m.key)
        linhas.append(MeasureComparison(m.key, m.label, t, v, m.is_close(t, v), m.tolerance_text()))
    return linhas


def summarize(medidas: Sequence[DegradationMeasures]) -> Dict[str, Dict[str, Any]]:
    """Mediana, P5 e P95 de cada medida numérica sobre um conjunto."""
    out: Dict[str, Dict[str, Any]] = {}
    for m in MEASURES:
        vals = [getattr(x, m.key) for x in medidas if getattr(x, m.key) is not None]
        if not vals:
            continue
        if m.kind == "bool":
            out[m.key] = {"fraction_true": round(float(np.mean([bool(v) for v in vals])), 6),
                          "n": len(vals)}
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[m.key] = {
            "median": round(float(np.median(arr)), 6),
            "p5": round(float(np.percentile(arr, 5)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "n": int(arr.size),
        }
    regioes: Dict[str, int] = {}
    for x in medidas:
        regioes[x.region] = regioes.get(x.region, 0) + 1
    out["regions"] = regioes
    return out


def compare_summary(
    target: Optional[DegradationMeasures], resumo: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Questionada × mediana do conjunto, medida a medida (JSON-safe)."""
    linhas = []
    for m in MEASURES:
        r = resumo.get(m.key)
        if r is None:
            continue
        central = (r.get("fraction_true", 0) >= 0.5) if m.kind == "bool" else r.get("median")
        t = None if target is None else getattr(target, m.key)
        linhas.append({
            "measure": m.key,
            "label": m.label,
            "questioned": t,
            "set_median": central,
            "set_p5": r.get("p5"),
            "set_p95": r.get("p95"),
            "close": m.is_close(t, central),
            "tolerance": m.tolerance_text(),
        })
    return linhas


# ── Perfil da questionada ──────────────────────────────────────────────────

def read_image_bgr(path: Path) -> np.ndarray:
    """Lê com o OpenCV (aplica a orientação EXIF, como o resto do sistema).
    Por bytes, porque `cv2.imread` falha em caminho acentuado no Windows."""
    dados = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível decodificar a imagem {Path(path).name}.")
    return img


def build_questioned_profile(path: Path, ff_service=None):
    """Lê a questionada, localiza a face e mede.

    Returns:
        `(QuestionedProfile, imagem BGR)`.
    """
    from faceaudit.models.image_processing.questioned_profile import QuestionedProfile
    from faceaudit.pipelines.image_processing_pipeline import detect_face_geometry

    path = Path(path)
    dados = path.read_bytes()
    img = read_image_bgr(path)
    face = detect_face_geometry(ff_service, img) if ff_service is not None else None
    perfil = QuestionedProfile(
        file_name=path.name,
        source_path=str(path),
        sha256=hashlib.sha256(dados).hexdigest(),
        measured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        face=face,
        face_source="detector" if face is not None else None,
        measures=measure(img, face),
    )
    return perfil, img
