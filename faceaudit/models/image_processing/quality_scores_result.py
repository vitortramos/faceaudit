"""DTO de scores de qualidade da imagem original × processada (PR I3).

Resultado retornado pelo `QualityScoresWorker` ao terminar de
calcular FIQA + ‖embedding‖ + cosine similarity. Consumido pelo
`QualityScoresPanel` na janela do editor.

3 métricas por agora:
  - FIQA (0–100): qualidade da face do FIQA network do ForensicFace
  - ‖embedding‖: norma do embedding do modelo (proxy de "confiança")
  - Cosine similarity: cosseno entre embeddings original × processada

Métricas externas pesadas (OFIQ, SERFIQ) ficam fora do I3 — entram
em PR futuro com checkbox opt-in se necessário.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QualityScoresResult:
    """Resultado consolidado de scores de qualidade.

    Campos `Optional[float]` viram `None` quando a face não foi
    detectada na respectiva imagem (oclusão massiva, blur extremo,
    crop sem face — caso típico).

    Attributes:
        original_fiqa: FIQA da imagem original (após alinhamento).
        original_emb_norm: norma do embedding da original.
        processed_fiqa: FIQA da imagem processada.
        processed_emb_norm: norma do embedding da processada.
        cosine_similarity: cosseno entre os 2 embeddings (1 = idênticos,
            0 = ortogonais, -1 = opostos). None quando algum
            embedding não foi extraído.
        original_face_detected: True se a detecção achou face na original.
        processed_face_detected: True idem na processada.
    """

    original_fiqa: Optional[float] = None
    original_emb_norm: Optional[float] = None
    processed_fiqa: Optional[float] = None
    processed_emb_norm: Optional[float] = None
    cosine_similarity: Optional[float] = None
    original_face_detected: bool = False
    processed_face_detected: bool = False

    @property
    def has_any_result(self) -> bool:
        """True quando ao menos uma métrica foi computada."""
        return any(
            v is not None
            for v in (
                self.original_fiqa,
                self.original_emb_norm,
                self.processed_fiqa,
                self.processed_emb_norm,
                self.cosine_similarity,
            )
        )
