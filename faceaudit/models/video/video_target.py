# faceaudit/models/video/video_target.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VideoTarget:
    """Um alvo de re-identificação no vídeo (objeto de RUNTIME).

    Modelado como objeto de 1ª classe numa LISTA desde a v1 (que expõe
    1 alvo) para que multi-alvo seja só destravar a UI + o assignment
    N×M. Carrega os embeddings de âncora (não serializados em JSON; os
    vetores vão para `_embeddings_<id>.parquet`, e só o
    `anchor_embedding_fingerprint` vai ao manifesto/sidecar).
    """

    label: str
    anchor_embeddings: List[np.ndarray] = field(default_factory=list)
    anchor_policy: str = "SINGLE"                 # SINGLE | AGGREGATED
    # Proveniência por âncora, paralela a `anchor_embeddings`. Cada item:
    # {type: "video"|"image", path, filename, frame_index, bbox, sha256, det_index}.
    # Permite ancorar o mesmo alvo em vários quadros/vídeos E em fotos estáticas.
    anchor_sources: List[dict] = field(default_factory=list)
    # Como as âncoras são combinadas no scoring:
    #   CENTROID = cosseno com a referência média (robusto p/ aparências da
    #              MESMA pessoa; conservador);
    #   ANY      = melhor cosseno entre as âncoras (casa se bater com QUALQUER
    #              uma; mais permissivo, mais recall e mais risco).
    anchor_match_mode: str = "CENTROID"           # CENTROID | ANY
    similarity_threshold: float = 0.30
    similarity_threshold_source: str = "MANUAL"   # MANUAL | SUGGESTED
    anchor_frames: List[int] = field(default_factory=list)  # quadros-âncora secundários
    recognition_model: Optional[str] = None       # modelo que produziu a âncora
    anchor_frame_index: Optional[int] = None       # quadro do clique (p/ quadro anotado)
    anchor_bbox: Optional[List[float]] = None      # bbox da face clicada (p/ quadro anotado)

    # Estado de track (Fase 4) — opcional na v1.
    last_bbox: Optional[List[float]] = None
    last_frame_index: Optional[int] = None
    velocity: Optional[List[float]] = None

    @property
    def anchor_embedding_count(self) -> int:
        return len(self.anchor_embeddings)

    def add_anchor(self, embedding: np.ndarray, source: dict) -> None:
        """Acrescenta uma âncora (embedding + proveniência da fonte). A política
        vira AGGREGATED a partir da 2ª âncora. A 1ª âncora de VÍDEO define o
        quadro/bbox usados no quadro-âncora anotado (fotos não têm quadro)."""
        self.anchor_embeddings.append(np.asarray(embedding))
        self.anchor_sources.append(dict(source))
        self.anchor_policy = "AGGREGATED" if len(self.anchor_embeddings) > 1 else "SINGLE"
        if (self.anchor_frame_index is None and source.get("type") == "video"
                and source.get("frame_index") is not None):
            self.anchor_frame_index = source.get("frame_index")
            self.anchor_bbox = source.get("bbox")

    def has_near_duplicate_anchor(self, embedding: np.ndarray, *, tol: float = 0.9999) -> bool:
        """True se `embedding` é praticamente idêntico (cosseno ≥ tol) a uma
        âncora já existente — usado para NÃO duplicar a mesma face (clicar duas
        vezes na mesma detecção)."""
        if embedding is None or not self.anchor_embeddings:
            return False
        e = self._l2(np.asarray(embedding, dtype=np.float64))
        return any(float(np.dot(self._l2(a), e)) >= tol for a in self.anchor_embeddings)

    def anchor_source_counts(self) -> Tuple[int, int]:
        """(n_video, n_image) entre as fontes de âncora registradas."""
        n_video = sum(1 for s in self.anchor_sources if s.get("type") == "video")
        n_image = sum(1 for s in self.anchor_sources if s.get("type") == "image")
        return n_video, n_image

    def remove_anchor(self, index: int) -> None:
        """Remove a âncora `index` (embedding + fonte). Recalcula a política e
        reassocia o quadro/bbox anotado à 1ª âncora de vídeo remanescente."""
        if not (0 <= index < len(self.anchor_embeddings)):
            return
        self.anchor_embeddings.pop(index)
        if index < len(self.anchor_sources):
            self.anchor_sources.pop(index)
        self.anchor_policy = "AGGREGATED" if len(self.anchor_embeddings) > 1 else "SINGLE"
        self.anchor_frame_index = None
        self.anchor_bbox = None
        for s in self.anchor_sources:
            if s.get("type") == "video" and s.get("frame_index") is not None:
                self.anchor_frame_index = s.get("frame_index")
                self.anchor_bbox = s.get("bbox")
                break

    def reference_embedding(self) -> Optional[np.ndarray]:
        """Embedding de referência (média L2-normalizada das âncoras),
        normalizado. None se não há âncora."""
        if not self.anchor_embeddings:
            return None
        stacked = np.stack([self._l2(e) for e in self.anchor_embeddings], axis=0)
        mean = stacked.mean(axis=0)
        return self._l2(mean)

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Similaridade do alvo com `embedding`, conforme `anchor_match_mode`:
        CENTROID = cosseno com a referência média; ANY = melhor cosseno entre
        as âncoras (casa se bater com QUALQUER uma)."""
        if embedding is None:
            return float("-inf")
        e = self._l2(np.asarray(embedding, dtype=np.float64))
        if self.anchor_match_mode == "ANY" and self.anchor_embeddings:
            return max((float(np.dot(self._l2(a), e)) for a in self.anchor_embeddings),
                       default=float("-inf"))
        ref = self.reference_embedding()
        if ref is None:
            return float("-inf")
        return float(np.dot(ref, e))

    def anchor_embedding_fingerprint(self) -> Optional[str]:
        """SHA-256 da referência (float32 canônico) — identifica a âncora
        no manifesto/sidecar sem expor o vetor."""
        ref = self.reference_embedding()
        if ref is None:
            return None
        return "sha256:" + hashlib.sha256(
            np.ascontiguousarray(ref, dtype=np.float32).tobytes()
        ).hexdigest()

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
