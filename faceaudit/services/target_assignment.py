# faceaudit/services/target_assignment.py
"""Atribuição detecções×alvos + tracker leve por associação entre quadros.

Módulo **puro** — sem OpenCV, Qt ou ForensicFace; só bboxes, vetores,
limiares e estado. Isso o torna trivialmente testável.

Hierarquia (a identidade é o árbitro):
  1. similaridade de cosseno com a referência de cada alvo;
  2. **gate de identidade** elimina pares abaixo do limiar do alvo;
  3. **gating espacial** atua SÓ entre os aprovados (track posição/
     velocidade → bbox prevista; IoU como bônus de desempate);
  4. assignment greedy 1-pra-1 (≤1 detecção por alvo).

Registra `margin_to_second_best` (por similaridade) e `assigned_by`
(`embedding` | `tiebreak_track` | `track_recovery`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class Candidate:
    det_index: int
    bbox: BBox
    embedding: Optional[np.ndarray] = None
    det_score: float = 0.0


@dataclass
class TargetState:
    """Alvo + estado de track (mutável; mantido pelo serviço entre quadros)."""
    label: str
    reference_embedding: np.ndarray
    threshold: float
    last_bbox: Optional[BBox] = None
    last_frame_index: Optional[int] = None
    velocity: Tuple[float, float] = (0.0, 0.0)   # delta de centro por quadro
    anchor_embeddings: Optional[List[np.ndarray]] = None   # usado quando match_mode=ANY
    match_mode: str = "CENTROID"                            # CENTROID | ANY


@dataclass
class Assignment:
    target_label: str
    det_index: int
    similarity: float
    margin_to_second_best: Optional[float]
    assigned_by: str            # embedding | tiebreak_track | track_recovery
    spatial_iou: Optional[float]


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("-inf")
    return float(np.dot(_l2(a), _l2(b)))


def score(t: "TargetState", embedding: np.ndarray) -> float:
    """Similaridade alvo×embedding conforme o modo de combinação das âncoras:
    CENTROID = cosseno com a referência média; ANY = melhor cosseno entre as
    âncoras (casa se bater com QUALQUER uma)."""
    if getattr(t, "match_mode", "CENTROID") == "ANY" and t.anchor_embeddings:
        return max((cosine(a, embedding) for a in t.anchor_embeddings),
                   default=float("-inf"))
    return cosine(t.reference_embedding, embedding)


def _center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def predict_bbox(t: TargetState, frame_index: int) -> Optional[BBox]:
    """bbox esperada do alvo neste quadro, extrapolando pela velocidade."""
    if t.last_bbox is None or t.last_frame_index is None:
        return None
    dt = frame_index - t.last_frame_index
    dx, dy = t.velocity[0] * dt, t.velocity[1] * dt
    x1, y1, x2, y2 = t.last_bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _update_track(t: TargetState, bbox: BBox, frame_index: int) -> None:
    if t.last_bbox is not None and t.last_frame_index is not None:
        dt = max(1, frame_index - t.last_frame_index)
        oc, nc = _center(t.last_bbox), _center(bbox)
        t.velocity = ((nc[0] - oc[0]) / dt, (nc[1] - oc[1]) / dt)
    t.last_bbox = bbox
    t.last_frame_index = frame_index


def assign_frame(
    targets: Sequence[TargetState],
    candidates: Sequence[Candidate],
    frame_index: int,
    *,
    spatial_weight: float = 0.0,
    gap_tolerance: int = 5,
) -> List[Assignment]:
    """Atribui detecções a alvos neste quadro. Muta o estado de track dos
    alvos atribuídos. `spatial_weight=0` → identidade pura (compat Fase 1);
    >0 habilita o desempate espacial entre candidatos aprovados."""
    # aprovados (acima do limiar) por alvo, ordenados por similaridade
    approved: List[List[Tuple[int, float]]] = []
    for t in targets:
        lst = []
        for ci, c in enumerate(candidates):
            s = score(t, c.embedding)
            if s >= t.threshold:
                lst.append((ci, s))
        lst.sort(key=lambda p: p[1], reverse=True)
        approved.append(lst)

    # pares (score combinado, ti, ci, sim, iou) para o greedy
    pairs: List[Tuple[float, int, int, float, Optional[float]]] = []
    for ti, t in enumerate(targets):
        pred = predict_bbox(t, frame_index)
        for ci, s in approved[ti]:
            i = iou(pred, candidates[ci].bbox) if pred is not None else None
            combined = s + (spatial_weight * i if i is not None else 0.0)
            pairs.append((combined, ti, ci, s, i))
    pairs.sort(key=lambda p: p[0], reverse=True)

    used_t, used_c = set(), set()
    assignments: List[Assignment] = []
    for combined, ti, ci, s, i in pairs:
        if ti in used_t or ci in used_c:
            continue
        used_t.add(ti)
        used_c.add(ci)
        t = targets[ti]
        lst = approved[ti]
        margin = (lst[0][1] - lst[1][1]) if len(lst) >= 2 else None
        gap = (frame_index - t.last_frame_index) if t.last_frame_index is not None else None
        top_sim_ci = lst[0][0]
        if t.last_frame_index is None:
            assigned_by = "embedding"
        elif gap is not None and gap > gap_tolerance:
            assigned_by = "track_recovery"
        elif ci != top_sim_ci:
            assigned_by = "tiebreak_track"
        else:
            assigned_by = "embedding"
        assignments.append(Assignment(
            target_label=t.label, det_index=candidates[ci].det_index,
            similarity=s, margin_to_second_best=margin, assigned_by=assigned_by, spatial_iou=i,
        ))
        _update_track(t, candidates[ci].bbox, frame_index)
    return assignments
