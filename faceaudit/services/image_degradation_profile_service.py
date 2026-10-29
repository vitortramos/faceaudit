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

A comparação de um CONJUNTO com a questionada tem duas leituras, e as
duas são registradas: a mediana do conjunto (`compare_summary`) e a
parcela das imagens que, uma a uma, ficam dentro da tolerância
(`agreement`). Só a mediana engana: num caso real a mediana de uma base
adaptada acertou as 8 medidas da questionada enquanto apenas 5% das
imagens, individualmente, se pareciam com ela — cópias escuras demais e
claras demais se compensavam na mediana.

A textura por faixa de tom (`texture_by_tone`) separa hipóteses que o
grão total não separa: meio-tom concentra a textura nos tons médios e
deixa preto e branco chapados limpos; ruído de toner põe textura nas
áreas claras. No caso real (Q1), as duas hipóteses empatavam nas outras
oito medidas e o formato do perfil favoreceu o meio-tom em 231 de 231
imagens.

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

# Textura por faixa de tom: tom local = desfoque gaussiano (σ em px);
# textura = RMS do que sobra além dele, por faixa de tom local. Faixa
# com menos de `TEXTURA_MIN_PIXELS` pixels fica sem valor (None).
FAIXAS_DE_TOM = (0, 50, 100, 150, 200, 256)
TEXTURA_SIGMA_PX = 2.0
TEXTURA_MIN_PIXELS = 30
# Perfil só é comparável com pelo menos esta quantidade de faixas em comum.
TEXTURA_MIN_FAIXAS = 3

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
    # Uma entrada por faixa de `FAIXAS_DE_TOM`; None em perfis gravados
    # antes desta medida existir.
    texture_by_tone: Optional[Tuple[Optional[float], ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 6)
            elif isinstance(v, tuple):
                d[k] = [None if x is None else round(float(x), 4) for x in v]
        return d

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["DegradationMeasures"]:
        if not data:
            return None
        nomes = {f.name for f in fields(cls)}
        valores = {k: v for k, v in data.items() if k in nomes}
        perfil = valores.get("texture_by_tone")
        if perfil is not None:
            valores["texture_by_tone"] = tuple(
                None if x is None else float(x) for x in perfil)
        return cls(**valores)


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
    # profile: tupla por faixa de tom. `abs_tol` é a tolerância do
    # FORMATO (perfil ÷ máximo, diferença média nas faixas) e `rel_tol` a
    # do NÍVEL (razão média, como nas medidas relativas).
    kind: str = "float"  # float | fraction | int | bool | px | profile

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
        if self.kind == "profile":
            return " · ".join("—" if x is None else f"{float(x):.0f}" for x in v)
        return f"{float(v):,.0f}".replace(",", ".")

    def tolerance_text(self) -> str:
        if self.kind == "profile":
            return (f"formato ± {self.abs_tol:g}; nível ± {100 * self.rel_tol:g}%"
                    .replace(".", ","))
        if self.abs_tol is not None:
            if self.kind == "fraction":
                return f"± {100 * self.abs_tol:g} pontos percentuais".replace(".", ",")
            return f"± {self.abs_tol:g}".replace(".", ",")
        if self.rel_tol is not None:
            return f"± {100 * self.rel_tol:g}%".replace(".", ",")
        return "igual"

    def distance(self, target: Any, value: Any) -> Optional[float]:
        """Distância em múltiplos da tolerância (≤ 1 = dentro dela)."""
        if target is None or value is None:
            return None
        if self.kind == "bool":
            return 0.0 if bool(target) == bool(value) else 1e3
        if self.kind == "profile":
            return _profile_distance(target, value, self.abs_tol, self.rel_tol)
        t, v = float(target), float(value)
        if self.abs_tol is not None:
            return abs(v - t) / self.abs_tol
        if self.rel_tol is not None:
            if t <= 0 or v <= 0:
                return 0.0 if t == v else 1e3
            return abs(math.log(v / t)) / math.log1p(self.rel_tol)
        return 0.0 if t == v else 1e3

    def is_close(self, target: Any, value: Any) -> Optional[bool]:
        if target is None or value is None:
            return None
        if self.kind == "bool":
            return bool(target) == bool(value)
        if self.kind == "profile":
            d = self.distance(target, value)
            return None if d is None else d <= 1.0
        t, v = float(target), float(value)
        if self.abs_tol is not None:
            return abs(v - t) <= self.abs_tol
        if self.rel_tol is not None:
            if t <= 0 or v <= 0:
                return t == v
            return abs(math.log(v / t)) <= math.log1p(self.rel_tol)
        return t == v


def _profile_distance(target, value, tol_formato: float, tol_nivel: float) -> Optional[float]:
    """Distância entre dois perfis de textura, em múltiplos da tolerância:
    o pior entre FORMATO e NÍVEL. None com menos de `TEXTURA_MIN_FAIXAS`
    faixas válidas nos dois."""
    pares = [(float(t), float(v)) for t, v in zip(target, value)
             if t is not None and v is not None and t > 0 and v > 0]
    if len(pares) < TEXTURA_MIN_FAIXAS:
        return None
    t = np.array([p[0] for p in pares])
    v = np.array([p[1] for p in pares])
    formato = float(np.mean(np.abs(v / v.max() - t / t.max())))
    nivel = float(np.mean(np.abs(np.log(v / t))))
    return max(formato / tol_formato, nivel / math.log1p(tol_nivel))


def _rotulo_das_faixas() -> str:
    return ", ".join(f"{a}–{b - 1}" for a, b in zip(FAIXAS_DE_TOM[:-1], FAIXAS_DE_TOM[1:]))


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
    MeasureSpec("texture_by_tone", "Textura por faixa de tom",
                "Força da textura fina (RMS, em níveis de cinza, do que sobra "
                f"além do tom local — desfoque σ {TEXTURA_SIGMA_PX:g} px) em cada "
                f"faixa de tom local: {_rotulo_das_faixas()}. Meio-tom concentra "
                "a textura nos tons médios e deixa preto e branco chapados "
                "limpos; ruído de toner põe textura nas áreas claras. Compara "
                "o FORMATO (perfil ÷ máximo) e o NÍVEL. Depende da resolução — "
                "compare com a mesma distância entre os olhos.",
                abs_tol=0.12, rel_tol=0.40, kind="profile"),
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


def texture_by_tone(regiao: np.ndarray) -> Tuple[Optional[float], ...]:
    """RMS da textura fina por faixa de tom local (ver `FAIXAS_DE_TOM`)."""
    reg = regiao.astype(np.float64)
    tom = cv2.GaussianBlur(reg, (0, 0), TEXTURA_SIGMA_PX)
    residuo2 = (reg - tom) ** 2
    out: List[Optional[float]] = []
    for a, b in zip(FAIXAS_DE_TOM[:-1], FAIXAS_DE_TOM[1:]):
        m = (tom >= a) & (tom < b)
        out.append(float(np.sqrt(residuo2[m].mean()))
                   if int(m.sum()) >= TEXTURA_MIN_PIXELS else None)
    return tuple(out)


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
        texture_by_tone=texture_by_tone(regiao),
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
        if m.kind == "profile":
            out[m.key] = _summarize_profiles(vals)
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


def _summarize_profiles(perfis) -> Dict[str, Any]:
    """Mediana, P5 e P95 POR FAIXA (None na faixa sem valor em nenhuma)."""
    arr = np.array([[np.nan if x is None else float(x) for x in p] for p in perfis])

    def _por_faixa(f):
        return [None if np.all(np.isnan(col)) else round(float(f(col[~np.isnan(col)])), 4)
                for col in arr.T]

    return {
        "median": _por_faixa(np.median),
        "p5": _por_faixa(lambda c: np.percentile(c, 5)),
        "p95": _por_faixa(lambda c: np.percentile(c, 95)),
        "n": len(perfis),
    }


def agreement(
    target: Optional[DegradationMeasures], medidas: Sequence[DegradationMeasures],
) -> Optional[Dict[str, Any]]:
    """Quantas imagens, UMA A UMA, ficam dentro da tolerância da questionada.

    Returns:
        None sem questionada ou sem imagens. Senão:
        - `per_measure`: fração das imagens dentro, por medida;
        - `close_count_histogram`: quantas imagens têm k medidas dentro;
        - `all_close_fraction` / `all_but_one_close_fraction`;
        - `median_distance`: distância mediana por imagem, somada nas
          medidas e em múltiplos da tolerância.
    """
    if target is None or not medidas:
        return None
    chaves = [m.key for m in MEASURES if getattr(target, m.key) is not None]
    por_imagem = []
    distancias = []
    for x in medidas:
        dentro = {}
        soma = 0.0
        for k in chaves:
            spec = MEASURES_BY_KEY[k]
            d = spec.distance(getattr(target, k), getattr(x, k))
            dentro[k] = d is not None and d <= 1.0
            soma += min(d, 10.0) if d is not None else 10.0
        por_imagem.append(dentro)
        distancias.append(soma)
    n = len(medidas)
    contagens = [sum(d.values()) for d in por_imagem]
    hist: Dict[str, int] = {}
    for c in contagens:
        hist[str(c)] = hist.get(str(c), 0) + 1
    return {
        "n_images": n,
        "measures": chaves,
        "per_measure": {
            k: round(sum(d[k] for d in por_imagem) / n, 6) for k in chaves
        },
        "close_count_histogram": dict(sorted(hist.items(), key=lambda kv: -int(kv[0]))),
        "all_close_fraction": round(sum(c == len(chaves) for c in contagens) / n, 6),
        "all_but_one_close_fraction": round(
            sum(c >= len(chaves) - 1 for c in contagens) / n, 6),
        "median_distance": round(float(np.median(distancias)), 6),
    }


def per_image_distance(target: DegradationMeasures, x: DegradationMeasures) -> float:
    """Distância de uma imagem à questionada: soma, nas medidas que a
    questionada tem, da distância em múltiplos da tolerância (cada termo
    limitado a 10, para uma medida absurda não dominar)."""
    soma = 0.0
    for m in MEASURES:
        t = getattr(target, m.key)
        if t is None:
            continue
        d = m.distance(t, getattr(x, m.key))
        soma += 10.0 if d is None else min(d, 10.0)
    return soma


def compare_summary(
    target: Optional[DegradationMeasures],
    resumo: Dict[str, Dict[str, Any]],
    medidas: Optional[Sequence[DegradationMeasures]] = None,
) -> List[Dict[str, Any]]:
    """Questionada × mediana do conjunto, medida a medida (JSON-safe).

    Com `medidas`, cada linha traz também `images_close_fraction`: a
    parcela das imagens que, uma a uma, ficam dentro da tolerância.
    """
    concordancia = agreement(target, medidas) if medidas else None
    por_medida = (concordancia or {}).get("per_measure", {})
    linhas = []
    for m in MEASURES:
        r = resumo.get(m.key)
        if r is None:
            continue
        central = (r.get("fraction_true", 0) >= 0.5) if m.kind == "bool" else r.get("median")
        t = None if target is None else getattr(target, m.key)
        if isinstance(t, tuple):
            t = list(t)  # JSON-safe: a custódia relida tem de ser igual
        linhas.append({
            "measure": m.key,
            "label": m.label,
            "questioned": t,
            "set_median": central,
            "set_p5": r.get("p5"),
            "set_p95": r.get("p95"),
            "close": m.is_close(t, central),
            "images_close_fraction": por_medida.get(m.key),
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


def complete_questioned_measures(perfil, imagem_bgr: np.ndarray):
    """Perfil gravado antes de uma medida existir: remede a questionada
    (mesma imagem, mesma geometria gravada) e devolve o perfil com as
    medidas que faltavam, ou o próprio perfil quando nada falta. As que já
    estavam gravadas não mudam. Quem chama confere o SHA-256 antes —
    remedir outra imagem sob o mesmo perfil seria trocar o alvo."""
    import dataclasses

    gravadas = perfil.measures
    if gravadas is None:
        return perfil
    novas = measure(imagem_bgr, perfil.face)
    faltam = {m.key: getattr(novas, m.key) for m in MEASURES
              if getattr(gravadas, m.key) is None and getattr(novas, m.key) is not None}
    if not faltam:
        return perfil
    return dataclasses.replace(perfil, measures=dataclasses.replace(gravadas, **faltam))


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
