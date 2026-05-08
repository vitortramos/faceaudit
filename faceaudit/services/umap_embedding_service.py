"""Serviço de projeção UMAP de embeddings faciais para visualização.

Espelha o pattern do `HubnessAnalysisService`: lógica pura sobre
`FaceDataset`/`EmbeddingDataset`, com `logger_callback` opcional pra
progresso. A camada de UI (`UmapAuditWorker`) move a chamada pra um
QThread e empacota o resultado num `AuditFinding`.
"""

from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

import numpy as np

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.umap_config import UmapConfig
from faceaudit.models.umap_embedding_result import (
    UmapEmbeddingResult, UmapSampleInfo,
)


# Atributos candidatos a coloração — populados se ≥ THRESHOLD dos
# samples têm valor não-nulo. Identidade sempre vai (mesmo com 1 ID,
# o usuário pode querer ver). Origin/source também sempre.
_ATTRIBUTE_CANDIDATES = (
    "identity", "gender", "ethnicity", "origin", "source",
    "yaw", "pitch", "fiqa", "norm",
)
_AVAILABILITY_THRESHOLD = 0.5


class UmapEmbeddingService:
    """Projeta embeddings para 2D/3D via UMAP.

    Aceita `FaceDataset` (não `EmbeddingDataset`) porque precisa dos
    atributos demográficos (gender/ethnicity) que o EmbeddingDataset
    descarta na conversão.
    """

    def __init__(self, comparison_service):
        # Mantido por simetria com `HubnessAnalysisService` —
        # `comparison_service._l2_normalize_rows` poderia ser usado se
        # quisermos pré-normalizar. Atualmente não, porque metric=cosine
        # do UMAP normaliza implicitamente; mas a injeção mantém a
        # porta aberta caso futuras métricas precisem de pré-processo.
        self._comparison_service = comparison_service

    def compute(
        self,
        dataset: FaceDataset,
        config: UmapConfig = UmapConfig(),
        logger_callback: Optional[Callable[[str], None]] = None,
    ) -> UmapEmbeddingResult:
        """Roda UMAP no dataset e retorna `UmapEmbeddingResult`.

        Subsample reproduzível via `numpy.random.default_rng(seed)` quando
        `len(dataset) > config.max_samples`. UMAP é chamado com
        `random_state` pra determinismo.
        """
        log = logger_callback if logger_callback is not None else (lambda _msg: None)
        t_start = time.time()

        n_total = len(dataset.faces)
        if n_total == 0:
            log("Dataset vazio — nada a projetar.")
            return UmapEmbeddingResult(
                config=config,
                samples=[],
                n_total_in_dataset=0,
                n_samples_used=0,
                n_components=config.n_components,
                fit_time_seconds=time.time() - t_start,
                dataset_name=getattr(dataset, "name", "") or "",
            )

        # 1. Subsample (índices na faixa do dataset original)
        log(f"Convertendo {n_total:,} faces para EmbeddingDataset…")
        emb_ds = dataset.to_embedding_dataset()
        items = emb_ds.items
        if not items:
            log("Sem embeddings — projeção UMAP inviável.")
            return UmapEmbeddingResult(
                config=config,
                samples=[],
                n_total_in_dataset=n_total,
                n_samples_used=0,
                n_components=config.n_components,
                fit_time_seconds=time.time() - t_start,
                dataset_name=getattr(dataset, "name", "") or "",
            )

        n_emb = len(items)
        if n_emb > config.max_samples:
            log(
                f"Subsample reproduzível: {n_emb:,} → {config.max_samples:,} "
                f"(seed={config.subsample_seed})"
            )
            rng = np.random.default_rng(config.subsample_seed)
            indices = np.sort(
                rng.choice(n_emb, size=config.max_samples, replace=False)
            )
        else:
            indices = np.arange(n_emb)

        # 2. Monta a matriz (subset)
        log(f"Montando matriz de embeddings ({len(indices):,} × D)…")
        matrix = emb_ds.matrix[indices].astype(np.float32, copy=False)

        # 3. Roda UMAP
        log(
            f"Rodando UMAP (n_components={config.n_components}, "
            f"n_neighbors={config.n_neighbors}, min_dist={config.min_dist}, "
            f"metric={config.metric})… primeira chamada compila numba "
            f"(~10s extras)."
        )
        coords = self._fit_umap(matrix, config)

        # 4. Constrói samples — atributos vêm das FaceInfo correspondentes
        log("Empacotando samples…")
        samples = self._build_samples(
            dataset=dataset,
            embedding_items=items,
            indices=indices,
            coords=coords,
            n_components=config.n_components,
        )

        # 5. Auto-detecta atributos disponíveis (cobertura ≥ 50%)
        available = self._available_attributes(samples)

        elapsed = time.time() - t_start
        log(f"UMAP concluído em {elapsed:.1f}s. {len(samples):,} pontos.")

        return UmapEmbeddingResult(
            config=config,
            samples=samples,
            n_total_in_dataset=n_total,
            n_samples_used=len(samples),
            n_components=config.n_components,
            fit_time_seconds=elapsed,
            dataset_name=getattr(dataset, "name", "") or "",
            available_attributes=available,
        )

    # ── Helpers internos ─────────────────────────────────────────────

    @staticmethod
    def _fit_umap(matrix: np.ndarray, config: UmapConfig) -> np.ndarray:
        """Wrapper isolado pro fit do UMAP — facilita teste/mock e
        documenta exatamente quais kwargs são passados pro
        `umap.UMAP`."""
        # Import local: evita custo de carregamento (numba JIT) quando
        # o serviço é apenas instanciado mas nunca chamado.
        import umap

        reducer = umap.UMAP(
            n_components=int(config.n_components),
            n_neighbors=int(config.n_neighbors),
            min_dist=float(config.min_dist),
            metric=str(config.metric),
            random_state=int(config.random_state),
        )
        return reducer.fit_transform(matrix)

    @staticmethod
    def _build_samples(
        dataset: FaceDataset,
        embedding_items: List[Any],
        indices: np.ndarray,
        coords: np.ndarray,
        n_components: int,
    ) -> List[UmapSampleInfo]:
        """Pra cada índice subselecionado, monta UmapSampleInfo lendo
        atributos da FaceInfo correspondente. EmbeddingInfo só tem
        identity/origin/source/pose/quality_scores; gender/ethnicity
        precisam vir da FaceInfo do `dataset.faces` (parallel order
        com `embedding_items`)."""
        samples: List[UmapSampleInfo] = []
        n_faces = len(dataset.faces)
        for i, dataset_idx in enumerate(indices):
            dataset_idx = int(dataset_idx)
            emb = embedding_items[dataset_idx]
            face = (
                dataset.faces[dataset_idx]
                if 0 <= dataset_idx < n_faces else None
            )

            # Norma do embedding — proxy de confidence
            try:
                norm_val: Optional[float] = float(
                    np.linalg.norm(emb.embedding)
                )
            except Exception:
                norm_val = None

            quality = emb.quality_scores or {}
            fiqa_val = quality.get("fiqa")

            samples.append(UmapSampleInfo(
                index=dataset_idx,
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if n_components == 3 else None,
                identity=getattr(emb, "identity", None) or None,
                image_path=getattr(emb, "image_path", None) or None,
                file_name=getattr(emb, "file_name", None) or None,
                gender=(
                    str(getattr(face, "gender", "") or "") or None
                    if face is not None else None
                ),
                ethnicity=(
                    str(getattr(face, "ethnicity", "") or "") or None
                    if face is not None else None
                ),
                origin=getattr(emb, "origin", None) or None,
                source=getattr(emb, "source", None) or None,
                yaw=emb.pose.get("yaw") if emb.pose else None,
                pitch=emb.pose.get("pitch") if emb.pose else None,
                fiqa=float(fiqa_val) if fiqa_val is not None else None,
                norm=norm_val,
            ))
        return samples

    @staticmethod
    def _available_attributes(samples: List[UmapSampleInfo]) -> List[str]:
        """Lista atributos com cobertura ≥ 50% — alimenta o dropdown
        de coloração no widget. Identity sempre vai (mesmo com pouca
        cobertura, é o atributo de referência forense)."""
        if not samples:
            return []
        n = len(samples)
        out: List[str] = []
        for attr in _ATTRIBUTE_CANDIDATES:
            non_null = sum(
                1 for s in samples if getattr(s, attr, None) is not None
            )
            if attr == "identity" or (non_null / n >= _AVAILABILITY_THRESHOLD):
                out.append(attr)
        return out
