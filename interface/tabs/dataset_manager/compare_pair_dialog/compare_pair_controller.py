"""Controller do diálogo de comparação de par de faces.

Recebe duas `FaceInfo` em construção e exibe lado a lado. Quando
ambas têm embedding sepaelv2, calcula a similaridade cosseno via
`ComparisonService` injetado e mostra na barra inferior. Sem embedding
em uma das faces → label "indisponível" sem quebrar.

Uso típico (chamado pelo `dataset_manager_controller`):
    from .compare_pair_dialog.compare_pair_controller import ComparePairController
    cmp = ComparePairController(
        face_a, face_b,
        comparison_service=comp_svc,
        parent_view=self.view,
    )
    cmp.exec()
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets

from .compare_pair_view import ComparePairView


class ComparePairController(QtCore.QObject):
    """Coordena o ComparePairView com cálculo opcional de similaridade."""

    def __init__(
        self,
        face_a: Any,
        face_b: Any,
        comparison_service: Optional[Any] = None,
        parent_view: Optional[QtWidgets.QWidget] = None,
        dataset: Optional[Any] = None,
    ):
        super().__init__(parent_view)
        self._face_a = face_a
        self._face_b = face_b
        self._comparison = comparison_service

        self.view = ComparePairView(parent=parent_view)
        # Propaga dataset pra view resolver paths em modo RELATIVE
        # antes de carregar as thumbnails (Image Pipeline output).
        if dataset is not None and hasattr(self.view, "set_dataset"):
            self.view.set_dataset(dataset)

        # Identifica a face mais à esquerda como "a" — pra ordem ser
        # determinística no preview (embora o usuário não veja diferença).
        sim_label = self._compute_similarity_label()
        self.view.set_pair(face_a, face_b, similarity_label=sim_label)

    def exec(self):
        """Exibe o diálogo modalmente."""
        return self.view.exec()

    # ── Cálculo de similaridade ────────────────────────────────────────

    def _compute_similarity_label(self) -> Optional[str]:
        """Computa cosine similarity entre embeddings primários (sepaelv2).

        Retorna string formatada "Cosine = 0.823 ..." quando ambos têm
        embedding; None quando indisponível (controller passa pra view
        que mostra "indisponível").
        """
        emb_a = _extract_primary_embedding(self._face_a)
        emb_b = _extract_primary_embedding(self._face_b)
        if emb_a is None or emb_b is None:
            return None

        # Cosine similarity manual — não exigimos comparison_service aqui
        # pra simplificar a injeção (se quiser amanhã refatorar pra usar
        # o service, basta passar comparison_service.compare_pair).
        try:
            cos = float(_cosine_similarity(emb_a, emb_b))
        except Exception:
            return None

        # Hint qualitativa — mesma identidade declarada?
        same_id = (
            getattr(self._face_a, "identity", None) is not None
            and getattr(self._face_a, "identity", None)
            == getattr(self._face_b, "identity", None)
        )

        # Heurística pra hint visual; thresholds compatíveis com o
        # `audit_duplicate_identities` (0.85 pra centróide entre IDs).
        if cos >= 0.85:
            hint = "altíssima — forte indício de mesma pessoa"
        elif cos >= 0.65:
            hint = "alta — provável mesma pessoa"
        elif cos >= 0.40:
            hint = "moderada — sósia ou pose diferente"
        else:
            hint = "baixa — provavelmente pessoas diferentes"

        id_note = (
            " ✅ mesma identidade declarada"
            if same_id
            else " ⚠️ identidades diferentes"
        )
        return (
            f"Similaridade (cosine): {cos:.4f} — {hint}.{id_note}"
        )


# ── Helpers livres ────────────────────────────────────────────────────

def _extract_primary_embedding(face: Any) -> Optional[np.ndarray]:
    """Lê o embedding primário (sepaelv2) de uma FaceInfo, retornando
    None quando ausente. Aceita também faces que tenham `.embedding`
    direto (compat com EmbeddingInfo, embora o caso típico seja
    FaceInfo)."""
    ff = getattr(face, "ff_info", None)
    if ff is not None:
        emb = getattr(ff, "embedding_sepaelv2", None)
        if emb is not None:
            return np.asarray(emb, dtype=np.float64)
    # Fallback compat
    emb = getattr(face, "embedding", None)
    if emb is not None:
        return np.asarray(emb, dtype=np.float64)
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
