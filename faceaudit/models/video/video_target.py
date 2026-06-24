# faceaudit/models/video/video_target.py

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

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

    def reference_embedding(self) -> Optional[np.ndarray]:
        """Embedding de referência (média L2-normalizada das âncoras),
        normalizado. None se não há âncora."""
        if not self.anchor_embeddings:
            return None
        stacked = np.stack([self._l2(e) for e in self.anchor_embeddings], axis=0)
        mean = stacked.mean(axis=0)
        return self._l2(mean)

    def similarity_to(self, embedding: np.ndarray) -> float:
        """Similaridade de cosseno entre `embedding` e a referência."""
        ref = self.reference_embedding()
        if ref is None or embedding is None:
            return float("-inf")
        e = self._l2(np.asarray(embedding, dtype=np.float64))
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
