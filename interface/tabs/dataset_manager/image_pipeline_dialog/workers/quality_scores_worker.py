"""QualityScoresWorker — calcula scores de qualidade em thread (PR I3).

Pipeline interno por imagem (original e processada):
  1. Se `input_source == ALIGNED_FACE`: imagem JÁ é a face alinhada,
     pula passo 2.
  2. Senão: chama `ff_service.align_face_from_array` pra detectar +
     alinhar (~80-150 ms via RetinaFace).
  3. Chama `ff_service.extract_embeddings_from_aligned_face` pra
     extrair FIQA + embedding (~40-150 ms via AdaFace).

Após as duas extrações, calcula cosine similarity entre embeddings
(2 dot products + 2 normas — desprezível).

Custo total típico (CPU):
  - ALIGNED_FACE: ~80-300 ms (só extract × 2)
  - ORIGINAL: ~240-600 ms (align + extract × 2)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt6 import QtCore

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.models.image_processing.quality_scores_result import (
    QualityScoresResult,
)


class QualityScoresWorker(QtCore.QObject):
    """Calcula scores em thread separada e emite via signal."""

    finished = QtCore.pyqtSignal(object)  # QualityScoresResult
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        ff_service,
        original_bgr: np.ndarray,
        processed_bgr: np.ndarray,
        input_source: ImageInputSource,
    ):
        super().__init__()
        self._ff_service = ff_service
        self._original_bgr = original_bgr
        self._processed_bgr = processed_bgr
        self._input_source = input_source

    def run(self):
        """Roda na thread. Sempre emite `finished` com um
        `QualityScoresResult` (mesmo quando faces não são detectadas
        — campos viram None nesse caso)."""
        try:
            orig_aligned, orig_kps = self._get_aligned(self._original_bgr)
            proc_aligned, proc_kps = self._get_aligned(self._processed_bgr)

            orig_metrics = self._extract_metrics(orig_aligned, orig_kps)
            proc_metrics = self._extract_metrics(proc_aligned, proc_kps)

            cosine = self._cosine(
                orig_metrics["embedding"],
                proc_metrics["embedding"],
            )

            result = QualityScoresResult(
                original_fiqa=orig_metrics["fiqa"],
                original_emb_norm=orig_metrics["norm"],
                processed_fiqa=proc_metrics["fiqa"],
                processed_emb_norm=proc_metrics["norm"],
                cosine_similarity=cosine,
                original_face_detected=orig_aligned is not None,
                processed_face_detected=proc_aligned is not None,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_aligned(
        self, bgr: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Devolve `(face_alinhada, aligned_keypoints)` conforme `input_source`.

        ALIGNED_FACE: input já é a face — devolve direto sem keypoints
            (não temos como reconstruir keypoints alinhados a partir de
            uma face já recortada). Modelos KPRPE (sepaelv6) não funcionarão
            nesse caminho — mas o cenário típico desse worker é qualidade
            via v2/v4 onde keypoints são ignorados.
        ORIGINAL: chama `align_face_from_array` (detect + crop + keypoints).
        """
        if bgr is None:
            return None, None
        if self._input_source == ImageInputSource.ALIGNED_FACE:
            return bgr, None
        if self._ff_service is None:
            return None, None
        return self._ff_service.align_face_from_array(bgr)

    def _extract_metrics(
        self,
        aligned: Optional[np.ndarray],
        aligned_keypoints: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Extrai FIQA + embedding + ‖emb‖ a partir da face alinhada.

        Quando aligned é None, retorna tudo None — caso típico:
        oclusão massiva destruiu a face.
        """
        empty = {"fiqa": None, "norm": None, "embedding": None}
        if aligned is None or self._ff_service is None:
            return empty
        ff_info = self._ff_service.extract_embeddings_from_aligned_face(
            aligned, aligned_keypoints=aligned_keypoints,
        )
        if ff_info is None:
            return empty
        # Procura embedding no slot disponível (sepaelv2 default,
        # sepaelv4 fallback). FFInfo dos PRs L3+ aceita ambos.
        # `or` em numpy ndarray dispara truthy ambíguo — use `is not None`.
        emb = getattr(ff_info, "embedding_sepaelv2", None)
        if emb is None:
            emb = getattr(ff_info, "embedding_sepaelv4", None)
        norm = (
            float(np.linalg.norm(emb))
            if emb is not None
            else None
        )
        return {
            "fiqa": ff_info.fiqa_score,
            "norm": norm,
            "embedding": emb,
        }

    @staticmethod
    def _cosine(
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
    ) -> Optional[float]:
        """Cosine similarity. None quando algum vetor está ausente
        ou tem norma zero (caso patológico)."""
        if a is None or b is None:
            return None
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return None
        return float(np.dot(a, b) / (norm_a * norm_b))
