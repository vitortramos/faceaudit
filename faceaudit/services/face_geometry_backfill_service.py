"""Preenche a caixa e os pontos da face num dataset que não os tem.

A geometria nunca chegou ao parquet: existia em memória durante a ingestão
e morria na gravação. Recriar o dataset recuperaria, mas é caro e refaz
embeddings que já estão certos. Esta passada roda só o detector e escreve
`bbox` e `keypoints`, sem tocar em embedding, qualidade ou identidade.

O invariante sai de graça: a detecção roda **na imagem que o dataset
guarda**, aquela de `image_path`, então o resultado já está no sistema de
coordenadas dela, seja foto original, recorte com margem ou face alinhada.
Não há conversão nem ramificação por tipo de dataset.

Duas honestidades que o relatório carrega:

- a geometria vem de uma detecção POSTERIOR, não da que produziu o
  embedding. Nas imagens de face única dá no mesmo; num quadro com várias
  pessoas o detector pode escolher outra, então a ambiguidade é contada e
  nomeada em vez de resolvida em silêncio;
- imagem ilegível ou ausente não vira exceção de fluxo: a face fica sem
  geometria, é contada, e o dataset continua utilizável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.services.image_rotation_service import (
    caixa_girada_para_original,
    girar_imagem,
    pontos_girados_para_originais,
)

# Chave de proveniência gravada no dataset. Declara que a geometria é de
# uma detecção posterior, e não da extração original — num laudo essa
# diferença importa.
PROVENANCE_KEY = "geometry_source"
PROVENANCE_VALUE = "backfill_detector"
PROVENANCE_WHEN_KEY = "geometry_backfill_at"

# Resultados possíveis por face.
STATUS_PREENCHIDA = "preenchida"
STATUS_JA_TINHA = "ja_tinha"
STATUS_AMBIGUA = "ambigua"
STATUS_SEM_FACE = "sem_face"
STATUS_SEM_IMAGEM = "sem_imagem"


@dataclass(frozen=True)
class FaceOutcome:
    """O que aconteceu com uma face."""

    file_name: str
    status: str
    n_faces_detected: int = 0
    rotation: Optional[int] = None


@dataclass
class BackfillReport:
    """Contagens e os casos que pedem olho humano."""

    total: int = 0
    preenchidas: int = 0
    ja_tinham: int = 0
    ambiguas: int = 0
    sem_face: int = 0
    sem_imagem: int = 0
    detalhes: List[FaceOutcome] = field(default_factory=list)

    @property
    def mudou(self) -> bool:
        """True quando há algo novo a gravar."""
        return self.preenchidas > 0 or self.ambiguas > 0

    @property
    def pedem_conferencia(self) -> List[FaceOutcome]:
        """Casos que o perito tem de olhar: ambíguos e sem face."""
        return [
            d for d in self.detalhes
            if d.status in (STATUS_AMBIGUA, STATUS_SEM_FACE)
        ]

    def resumo(self) -> str:
        partes = [f"{self.total} face(s)"]
        if self.preenchidas:
            partes.append(f"{self.preenchidas} preenchida(s)")
        if self.ambiguas:
            partes.append(f"{self.ambiguas} ambígua(s)")
        if self.ja_tinham:
            partes.append(f"{self.ja_tinham} já tinha(m)")
        if self.sem_face:
            partes.append(f"{self.sem_face} sem face detectada")
        if self.sem_imagem:
            partes.append(f"{self.sem_imagem} sem imagem")
        return ", ".join(partes)


def backfill_geometry(
    dataset: Any,
    geometry_provider: Any,
    *,
    rotations: Sequence[int] = (),
    progress_callback: Optional[Callable[[str, int], None]] = None,
    overwrite: bool = False,
) -> BackfillReport:
    """Detecta na imagem de cada face e grava a geometria no `ff_info`.

    Muta o dataset no lugar e devolve o relatório. NÃO grava em disco —
    quem chama decide, e é lá que a entrada da cadeia de custódia é
    registrada.

    `geometry_provider` precisa só de `detect_faces(imagem_bgr)`, o que
    mantém a regra testável sem carregar modelo nenhum. `rotations` são as
    orientações a tentar quando a imagem como está não rende face; a
    orientação já registrada na face é sempre tentada primeiro.
    """
    from faceaudit.services.image_io import imread_unicode_safe

    relatorio = BackfillReport()
    faces = list(getattr(dataset, "faces", None) or [])
    relatorio.total = len(faces)
    n = len(faces)

    for indice, face in enumerate(faces):
        if progress_callback and (indice % 25 == 0 or indice + 1 == n):
            progress_callback(
                f"Detectando geometria ({indice + 1}/{n})…",
                int((indice + 1) / max(1, n) * 100),
            )

        ff = getattr(face, "ff_info", None)
        nome = getattr(face, "file_name", None) or str(
            getattr(face, "image_path", "") or "?"
        )

        if not overwrite and ff is not None and getattr(ff, "keypoints", None) is not None:
            relatorio.ja_tinham += 1
            relatorio.detalhes.append(FaceOutcome(nome, STATUS_JA_TINHA))
            continue

        try:
            caminho = dataset.resolve_image_path(face)
            imagem = imread_unicode_safe(str(caminho))
        except Exception:
            imagem = None
        if imagem is None:
            relatorio.sem_imagem += 1
            relatorio.detalhes.append(FaceOutcome(nome, STATUS_SEM_IMAGEM))
            continue

        detectadas, graus = _detectar(
            geometry_provider, imagem, face, rotations,
        )
        if not detectadas:
            relatorio.sem_face += 1
            relatorio.detalhes.append(
                FaceOutcome(nome, STATUS_SEM_FACE, 0, graus),
            )
            continue

        escolhida = _escolher(detectadas, ff)
        caixa, pontos = _geometria_de(escolhida)
        if caixa is None and pontos is None:
            relatorio.sem_face += 1
            relatorio.detalhes.append(
                FaceOutcome(nome, STATUS_SEM_FACE, len(detectadas), graus),
            )
            continue

        if graus:
            # A detecção rodou num quadro girado; a geometria volta ao
            # sistema do arquivo, que é onde `image_path` aponta.
            forma = girar_imagem(imagem, graus).shape[:2]
            if caixa is not None:
                caixa = caixa_girada_para_original(caixa, forma, graus)
            if pontos is not None:
                pontos = pontos_girados_para_originais(pontos, forma, graus)

        if ff is None:
            from faceaudit.models.ff_info import FFInfo

            ff = FFInfo()
            face.ff_info = ff
        if caixa is not None:
            ff.bbox = np.asarray(caixa, dtype=np.float32).reshape(-1)[:4]
        if pontos is not None:
            ff.keypoints = np.asarray(pontos, dtype=np.float32).reshape(5, 2)
        if graus:
            ff.detection_rotation = int(graus)

        if len(detectadas) > 1:
            relatorio.ambiguas += 1
            relatorio.detalhes.append(
                FaceOutcome(nome, STATUS_AMBIGUA, len(detectadas), graus),
            )
        else:
            relatorio.preenchidas += 1
            relatorio.detalhes.append(
                FaceOutcome(nome, STATUS_PREENCHIDA, 1, graus),
            )

    if relatorio.mudou:
        _marcar_proveniencia(dataset)
    return relatorio


# ── Interno ──────────────────────────────────────────────────────────────

def _detectar(
    provider: Any,
    imagem: np.ndarray,
    face: Any,
    rotations: Sequence[int],
) -> Tuple[Sequence[Any], Optional[int]]:
    """Faces detectadas e a rotação que as rendeu (None = sem girar).

    A orientação já registrada na face é tentada primeiro: foto gravada de
    lado só rende face depois de girar, e a face sabe qual giro foi.
    """
    ff = getattr(face, "ff_info", None)
    conhecida = getattr(ff, "detection_rotation", None) if ff else None

    tentativas: List[Optional[int]] = []
    if conhecida:
        tentativas.append(int(conhecida))
    tentativas.append(None)
    for g in rotations:
        if g and int(g) not in [t for t in tentativas if t]:
            tentativas.append(int(g))

    for graus in tentativas:
        quadro = imagem if not graus else girar_imagem(imagem, graus)
        try:
            encontradas = provider.detect_faces(quadro) or []
        except Exception:
            encontradas = []
        if len(encontradas) > 0:
            return encontradas, graus
    return [], None


def _escolher(detectadas: Sequence[Any], ff: Any) -> Any:
    """Qual das faces detectadas é a que gerou o embedding.

    Com uma só, não há dúvida. Com várias, a melhor pista disponível é a
    confiança de detecção que o dataset já guarda: a face escolhida na
    extração original é a de `det_score` mais próximo do gravado. Sem esse
    dado, fica a de maior área, que é a escolha que o próprio detector faz
    quando pedem uma face só. Em qualquer caso a linha é contada como
    ambígua — a dúvida é registrada, não dissolvida.
    """
    if len(detectadas) == 1:
        return detectadas[0]

    alvo = getattr(ff, "det_score", None) if ff is not None else None
    if alvo is not None:
        try:
            return min(
                detectadas,
                key=lambda f: abs(float(getattr(f, "det_score", 0.0)) - float(alvo)),
            )
        except (TypeError, ValueError):
            pass
    return max(detectadas, key=_area)


def _area(detectada: Any) -> float:
    caixa, _ = _geometria_de(detectada)
    if caixa is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in np.asarray(caixa).reshape(-1)[:4]]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _geometria_de(detectada: Any):
    """`(bbox, keypoints)` de uma face do detector, tolerante ao formato.

    O backend entrega objetos com `bbox`/`kps`; um duplo de teste pode
    entregar dicionário. Aceitar os dois é o que mantém a regra testável
    sem modelo.
    """
    caixa = None
    pontos = None
    if isinstance(detectada, dict):
        caixa = detectada.get("bbox")
        pontos = detectada.get("kps", detectada.get("keypoints"))
    else:
        caixa = getattr(detectada, "bbox", None)
        pontos = getattr(detectada, "kps", None)
        if pontos is None:
            pontos = getattr(detectada, "keypoints", None)
    if caixa is not None:
        caixa = np.asarray(caixa, dtype=np.float32).reshape(-1)
        if caixa.size < 4:
            caixa = None
    if pontos is not None:
        pontos = np.asarray(pontos, dtype=np.float32)
        if pontos.size != 10:
            pontos = None
    return caixa, pontos


def _marcar_proveniencia(dataset: Any) -> None:
    """Grava no dataset que a geometria é de uma detecção posterior."""
    from datetime import datetime, timezone

    prov = getattr(dataset, "provenance_metadata", None)
    if prov is None:
        return
    prov[PROVENANCE_KEY] = PROVENANCE_VALUE
    prov[PROVENANCE_WHEN_KEY] = datetime.now(timezone.utc).isoformat(
        timespec="seconds",
    )
