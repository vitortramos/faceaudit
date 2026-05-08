"""DTO de configuração da execução do UMAP sobre embeddings.

Consumido pelo `UmapEmbeddingService.compute(...)` pra parametrizar o
fit do UMAP. Defaults escolhidos pra serem robustos pro caso de uso
de visualização exploratória de embeddings faciais.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


@dataclass(frozen=True)
class UmapConfig(JSONSerializableMixin):
    """Configuração de execução do UMAP.

    Defaults são empíricos:
      - `n_neighbors=15`: sweet spot pra preservar estrutura local
        sem fragmentar clusters; UMAP é robusto na faixa 10–50.
      - `min_dist=0.1`: clusters compactos mas sem colapsar pontos
        em um único ponto (default oficial do UMAP).
      - `metric="cosine"`: embeddings de FR (AdaFace, sepaelv2) são
        comparados via cosseno na pipeline canônica; usar a mesma
        métrica preserva a noção de similaridade vista pelo modelo.
      - `random_state=42`: garante reproducibilidade. UMAP é
        determinístico só com `random_state` fixo (e `n_jobs=1`,
        que ele força automaticamente quando seed é dada).
      - `max_samples=100_000`: UMAP aguenta ~100k em poucos minutos
        em CPU moderna. Acima disso, o subsample preserva a estrutura
        empírica e mantém o tempo gerenciável.
    """
    # Parâmetros UMAP
    n_components: int = 2  # 2 (default) ou 3
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int = 42

    # Subsampling
    max_samples: int = 100_000
    subsample_seed: int = 42

    # Atributos pra coloração — None significa auto-detect (recomendado).
    # Quando explícito, o service só popula esses atributos no resultado;
    # útil pra forçar um conjunto específico em uso programático.
    color_attributes: Optional[Tuple[str, ...]] = None
