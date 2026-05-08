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

        Dispatch por `config.mode`:
          - "faces" (default): cada face = um ponto. Subsample
            reproduzível quando `len(dataset) > config.max_samples`.
          - "centroids": agrega faces por identidade num centroide
            L2-normalizado. Cada identidade = um ponto. Habilita
            click-to-lookalikes no widget (cosseno em 512D entre o
            centroide clicado e os outros — matriz é anexada no
            resultado via `attach_embeddings_matrix`).
        """
        if config.mode == UmapConfig.MODE_CENTROIDS:
            return self._compute_centroids(dataset, config, logger_callback)
        return self._compute_faces(dataset, config, logger_callback)

    # ── Modo "faces" ─────────────────────────────────────────────────

    def _compute_faces(
        self,
        dataset: FaceDataset,
        config: UmapConfig,
        logger_callback: Optional[Callable[[str], None]],
    ) -> UmapEmbeddingResult:
        log = logger_callback if logger_callback is not None else (lambda _msg: None)
        t_start = time.time()

        n_total = len(dataset.faces)
        if n_total == 0:
            log("Dataset vazio — nada a projetar.")
            return self._empty_result(config, n_total, t_start, dataset)

        log(f"Convertendo {n_total:,} faces para EmbeddingDataset…")
        emb_ds = dataset.to_embedding_dataset()
        items = emb_ds.items
        if not items:
            log("Sem embeddings — projeção UMAP inviável.")
            return self._empty_result(config, n_total, t_start, dataset)

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

        log(f"Montando matriz de embeddings ({len(indices):,} × D)…")
        matrix = emb_ds.matrix[indices].astype(np.float32, copy=False)

        log(
            f"Rodando UMAP (n_components={config.n_components}, "
            f"n_neighbors={config.n_neighbors}, min_dist={config.min_dist}, "
            f"metric={config.metric})… primeira chamada compila numba "
            f"(~10s extras)."
        )
        coords = self._fit_umap(matrix, config)

        log("Empacotando samples…")
        samples = self._build_samples(
            dataset=dataset,
            embedding_items=items,
            indices=indices,
            coords=coords,
            n_components=config.n_components,
        )
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

    # ── Modo "centroids" ─────────────────────────────────────────────

    def _compute_centroids(
        self,
        dataset: FaceDataset,
        config: UmapConfig,
        logger_callback: Optional[Callable[[str], None]],
    ) -> UmapEmbeddingResult:
        """Agrega faces por identidade em centroides L2-normalizados,
        roda UMAP nos centroides. Cada sample no resultado representa
        UMA identidade. Anexa a matriz K×D dos centroides ao resultado
        pra o widget computar cosseno on-click."""
        log = logger_callback if logger_callback is not None else (lambda _msg: None)
        t_start = time.time()

        n_total = len(dataset.faces)
        if n_total == 0:
            log("Dataset vazio — nada a projetar.")
            return self._empty_result(config, n_total, t_start, dataset)

        log(f"Convertendo {n_total:,} faces para EmbeddingDataset…")
        emb_ds = dataset.to_embedding_dataset()
        items = emb_ds.items
        if not items:
            log("Sem embeddings — projeção UMAP inviável.")
            return self._empty_result(config, n_total, t_start, dataset)

        # Agrupa índices por identidade — preserva ordem de aparição
        log("Agregando embeddings por identidade…")
        from collections import OrderedDict
        identity_to_indices: "OrderedDict[str, list]" = OrderedDict()
        for idx, item in enumerate(items):
            ident = getattr(item, "identity", None)
            if not ident:
                continue
            ident = str(ident)
            identity_to_indices.setdefault(ident, []).append(idx)

        n_identities = len(identity_to_indices)
        if n_identities == 0:
            log("Nenhuma identidade válida — projeção UMAP inviável.")
            return self._empty_result(config, n_total, t_start, dataset)

        # Centroide L2-normalizado por identidade. Aplica subsample
        # quando há > max_samples identidades (raro, mas possível).
        identities_in_use = list(identity_to_indices.keys())
        if n_identities > config.max_samples:
            log(
                f"Subsample reproduzível de identidades: "
                f"{n_identities:,} → {config.max_samples:,} "
                f"(seed={config.subsample_seed})"
            )
            rng = np.random.default_rng(config.subsample_seed)
            chosen = rng.choice(
                n_identities, size=config.max_samples, replace=False,
            )
            chosen.sort()
            identities_in_use = [identities_in_use[i] for i in chosen]

        K = len(identities_in_use)
        D = emb_ds.matrix.shape[1]
        log(f"Computando {K:,} centroides L2-normalizados ({D}D)…")
        centroids = np.empty((K, D), dtype=np.float32)
        for i, ident in enumerate(identities_in_use):
            idxs = identity_to_indices[ident]
            mean = emb_ds.matrix[idxs].mean(axis=0)
            n = float(np.linalg.norm(mean))
            centroids[i] = (mean / n).astype(np.float32) if n > 0 else mean.astype(np.float32)

        log(
            f"Rodando UMAP em {K:,} centroides (n_components="
            f"{config.n_components}, n_neighbors={config.n_neighbors}, "
            f"min_dist={config.min_dist}, metric={config.metric})…"
        )
        coords = self._fit_umap(centroids, config)

        log("Empacotando samples (uma identidade por sample)…")
        samples = self._build_centroid_samples(
            dataset=dataset,
            embedding_items=items,
            identity_to_indices=identity_to_indices,
            identities_in_use=identities_in_use,
            coords=coords,
            n_components=config.n_components,
        )
        available = self._available_attributes(samples)

        elapsed = time.time() - t_start
        log(f"UMAP concluído em {elapsed:.1f}s. {K:,} identidades.")

        result = UmapEmbeddingResult(
            config=config,
            samples=samples,
            n_total_in_dataset=n_total,
            n_samples_used=K,
            n_components=config.n_components,
            fit_time_seconds=elapsed,
            dataset_name=getattr(dataset, "name", "") or "",
            available_attributes=available,
        )
        # Anexa a matriz 512D pra click-to-lookalikes no widget
        result.attach_embeddings_matrix(centroids)
        return result

    @staticmethod
    def _empty_result(
        config: UmapConfig,
        n_total: int,
        t_start: float,
        dataset,
    ) -> UmapEmbeddingResult:
        """Resultado vazio — usado em early returns."""
        return UmapEmbeddingResult(
            config=config,
            samples=[],
            n_total_in_dataset=n_total,
            n_samples_used=0,
            n_components=config.n_components,
            fit_time_seconds=time.time() - t_start,
            dataset_name=getattr(dataset, "name", "") or "",
        )

    # ── Helpers internos ─────────────────────────────────────────────

    @staticmethod
    def _fit_umap(matrix: np.ndarray, config: UmapConfig) -> np.ndarray:
        """Wrapper isolado pro fit do UMAP — facilita teste/mock e
        documenta exatamente quais kwargs são passados pro
        `umap.UMAP`."""
        # Import local: evita custo de carregamento (numba JIT) quando
        # o serviço é apenas instanciado mas nunca chamado.
        import warnings
        import umap

        reducer = umap.UMAP(
            n_components=int(config.n_components),
            n_neighbors=int(config.n_neighbors),
            min_dist=float(config.min_dist),
            metric=str(config.metric),
            random_state=int(config.random_state),
        )
        # UMAP emite UserWarning informacional sempre que random_state
        # é setado, avisando que `n_jobs` foi forçado a 1 pra garantir
        # determinismo. Reprodutibilidade é desejável aqui — auditoria
        # forense precisa dos mesmos coords ao reexecutar — então o
        # comportamento é correto e o warning é só ruído. Suprimimos
        # localmente em vez de globalmente pra não esconder warnings
        # legítimos do UMAP em outras chamadas.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*n_jobs value .* overridden to 1 by setting random_state.*",
                category=UserWarning,
                module="umap.*",
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
    def _build_centroid_samples(
        dataset: FaceDataset,
        embedding_items: List[Any],
        identity_to_indices: dict,
        identities_in_use: List[str],
        coords: np.ndarray,
        n_components: int,
    ) -> List[UmapSampleInfo]:
        """Pra cada identidade, monta UmapSampleInfo com:
          - x/y/z = coord da identidade no espaço UMAP
          - identity = a identidade
          - image_path = primeira foto da identidade (representante)
          - gender/ethnicity/origin/source = valor da primeira face
            (todas faces da mesma pessoa devem compartilhar esses)
          - yaw/pitch/fiqa = MÉDIA sobre as faces da identidade
          - norm = norma média do centroide L2-norm = 1.0 (info pouco
            útil em centroide; deixamos None)
          - n_faces_in_identity = contagem de faces que entraram no
            centroide (sinaliza confiabilidade do ponto)
        """
        samples: List[UmapSampleInfo] = []
        n_faces = len(dataset.faces)

        for i, ident in enumerate(identities_in_use):
            face_indices = identity_to_indices[ident]
            n_faces_id = len(face_indices)

            # Primeira face — representante visual + atributos demográficos
            first_idx = face_indices[0]
            first_emb = embedding_items[first_idx]
            first_face = (
                dataset.faces[first_idx]
                if 0 <= first_idx < n_faces else None
            )

            # Atributos contínuos: média sobre as faces da identidade
            yaws: List[float] = []
            pitches: List[float] = []
            fiqas: List[float] = []
            for fi in face_indices:
                emb = embedding_items[fi]
                if emb.pose:
                    y_v = emb.pose.get("yaw")
                    p_v = emb.pose.get("pitch")
                    if y_v is not None:
                        yaws.append(float(y_v))
                    if p_v is not None:
                        pitches.append(float(p_v))
                q = emb.quality_scores or {}
                f_v = q.get("fiqa")
                if f_v is not None:
                    fiqas.append(float(f_v))

            yaw_mean = float(np.mean(yaws)) if yaws else None
            pitch_mean = float(np.mean(pitches)) if pitches else None
            fiqa_mean = float(np.mean(fiqas)) if fiqas else None

            samples.append(UmapSampleInfo(
                index=i,  # índice na matriz K×D dos centroides
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if n_components == 3 else None,
                identity=ident,
                image_path=getattr(first_emb, "image_path", None) or None,
                file_name=getattr(first_emb, "file_name", None) or None,
                gender=(
                    str(getattr(first_face, "gender", "") or "") or None
                    if first_face is not None else None
                ),
                ethnicity=(
                    str(getattr(first_face, "ethnicity", "") or "") or None
                    if first_face is not None else None
                ),
                origin=getattr(first_emb, "origin", None) or None,
                source=getattr(first_emb, "source", None) or None,
                yaw=yaw_mean,
                pitch=pitch_mean,
                fiqa=fiqa_mean,
                norm=None,  # centroide L2-norm sempre = 1.0; pouco útil
                n_faces_in_identity=int(n_faces_id),
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
