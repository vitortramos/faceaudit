"""DTOs de saída do `UmapEmbeddingService.compute(...)`.

Estrutura:
- `UmapSampleInfo` (frozen): coords + atributos por face
- `UmapEmbeddingResult`: agregado — config usada + samples + metadados
  da execução

Persistido como JSON via `JSONSerializableMixin` — não há ndarrays,
todos os campos são primitivos ou listas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.umap_config import UmapConfig


T = TypeVar("T", bound="UmapEmbeddingResult")


@dataclass(frozen=True)
class UmapSampleInfo(JSONSerializableMixin):
    """Estado por sample no resultado do UMAP — uma linha do scatter
    plot. `index` é o índice no dataset ORIGINAL (em modo `faces`) ou
    o índice da identidade na matriz de centroides (em modo
    `centroids`); útil pra correlacionar com outras auditorias e pra
    o widget achar a imagem ao clicar num ponto."""
    index: int
    x: float
    y: float
    z: Optional[float] = None  # populado só quando n_components=3

    # Atributos — None quando ausente no dataset
    identity: Optional[str] = None
    image_path: Optional[str] = None
    file_name: Optional[str] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    origin: Optional[str] = None
    source: Optional[str] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    fiqa: Optional[float] = None
    norm: Optional[float] = None

    # Em modo `centroids`, quantas faces da identidade contribuíram
    # para esse centroide. Em modo `faces`, sempre None.
    n_faces_in_identity: Optional[int] = None


@dataclass
class UmapEmbeddingResult(JSONSerializableMixin):
    """Resultado da projeção UMAP de embeddings.

    Estrutura "leve" — sem estudos derivados. Apenas a config usada,
    a lista de samples (com coord + atributos), e metadados da
    execução pra reprodutibilidade/auditoria.
    """
    config: UmapConfig
    samples: List[UmapSampleInfo] = field(default_factory=list)

    # Metadata da execução
    n_total_in_dataset: int = 0    # pre-subsample (tamanho do dataset)
    n_samples_used: int = 0        # post-subsample (tamanho efetivo)
    n_components: int = 2          # cópia conveniente de config.n_components
    fit_time_seconds: float = 0.0
    dataset_name: str = ""

    # Lista de atributos efetivamente populados nas samples (≥50%
    # cobertura). Útil pra UI popular dropdown "Colorir por" só com
    # atributos válidos pra esse dataset.
    available_attributes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Atributo runtime — NÃO faz parte do dataclass (asdict ignora,
        # to_dict do mixin não emite). Em modo `centroids` carrega a
        # matriz K×D dos centroides L2-normalizados (np.ndarray) pro
        # widget computar cosseno on-click. Em modo `faces` fica None.
        self._embeddings_matrix = None  # type: ignore[attr-defined]

    @property
    def embeddings_matrix(self):
        """Matriz K×D (np.ndarray) dos centroides L2-norm — só
        populada em modo `centroids`. Atributo runtime, não persistido.
        """
        return getattr(self, "_embeddings_matrix", None)

    def attach_embeddings_matrix(self, matrix) -> None:
        """Anexa a matriz runtime após o fit. Chamada pelo
        `UmapEmbeddingService` em modo centroids."""
        self._embeddings_matrix = matrix  # type: ignore[attr-defined]

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Reconstrói o resultado a partir de um dict — necessário pra
        instanciar `UmapConfig` e `UmapSampleInfo` aninhados."""
        d = dict(data)
        cfg = d.get("config", {})
        if isinstance(cfg, dict):
            d["config"] = UmapConfig.from_dict(cfg)

        samples_raw = d.get("samples", []) or []
        d["samples"] = [
            UmapSampleInfo.from_dict(s) if isinstance(s, dict) else s
            for s in samples_raw
        ]
        return cls(**d)
