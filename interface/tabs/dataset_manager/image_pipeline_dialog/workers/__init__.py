"""Workers do editor de pipeline de imagens (PR I3+).

Threads que rodam computação pesada sem travar a UI:
  - `QualityScoresWorker` (PR I3) — calcula FIQA + ‖emb‖ + cosine
    pra original × processada após cada Apply.
"""

from .quality_scores_worker import QualityScoresWorker

__all__ = ["QualityScoresWorker"]
