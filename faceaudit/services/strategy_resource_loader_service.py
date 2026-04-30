# faceaudit/services/strategy_resource_loader_service.py

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol


class StrategyResourceLoaderService:
    """
    Carrega (hidrata) o payload de um StrategyResource a partir de seu artifact_ref.

    Usado pelo StrategyExecutor para carregar recursos declarados na estratégia
    antes da execução — caso a definição tenha sido carregada de JSON
    (payload=None, artifact_ref preenchido).

    A hidratação é transparente:
    - Se payload já existe em memória → retorna sem modificação.
    - Se artifact_ref é None → retorna sem modificação (executor detectará na validação).
    - Se o arquivo não for encontrado → retorna sem modificação (executor detectará na validação).
    - Se o tipo não tiver loader registrado → retorna sem modificação.

    Tipos suportados:
        DATASET                     → EmbeddingDataset.from_parquet() ou FaceDataset.from_parquet()
        INTERMEDIATE_DATASET        → mesmo loader de DATASET
        PIPELINE_CONFIG             → TransformationPipelineConfig.from_dict()
        TRAINED_PIPELINE            → pickle.load()
        TRAINED_SIMILARITY_MODEL    → pickle.load()
        TRAINED_CALIBRATION_MODEL   → pickle.load()
        TRAJECTORY_DATA             → pickle.load()
        SIMILARITY_SCORES           → parquet (via SimilarityScoreDataset)
    """

    def __init__(self, artifact_storage: ArtifactStorageProtocol) -> None:
        self._storage = artifact_storage
        # Última falha de hidratação — preenchida por hydrate() quando o load
        # falha. Consultada pelo StrategyExecutionContext.ensure_payload para
        # tornar a falha visível no LogProgressDialog do usuário.
        self._last_error: Optional[str] = None

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def hydrate(
        self,
        resource: StrategyResource,
        embedding_column: Optional[str] = None,
    ) -> StrategyResource:
        """
        Carrega o payload do recurso a partir do artifact_ref, se necessário.

        A operação é in-place: modifica resource.payload diretamente.
        Retorna sempre o mesmo objeto para permitir encadeamento.

        Args:
            resource: Recurso a ser hidratado.
            embedding_column: Coluna de embedding a usar (override da estratégia).
                              None = auto-detect por prioridade.

        Returns:
            O mesmo objeto StrategyResource, com payload preenchido se possível.
        """
        self._last_error = None

        if resource.payload is not None:
            return resource

        if resource.artifact_ref is None:
            return resource

        abs_path = self._storage.resolve_artifact(resource.artifact_ref)
        if abs_path is None:
            self._last_error = (
                f"arquivo ausente em '{resource.artifact_ref.path}'"
            )
            return resource

        payload = self._load_by_type(
            resource.resource_type,
            abs_path,
            embedding_column=embedding_column,
        )
        if payload is not None:
            resource.payload = payload
        elif self._last_error is None:
            self._last_error = (
                f"loader retornou None para tipo {resource.resource_type} "
                f"(arquivo: {abs_path})"
            )

        return resource

    def hydrate_all(
        self,
        resources: list[StrategyResource],
        embedding_column: Optional[str] = None,
    ) -> None:
        """
        Hidrata uma lista de recursos in-place.

        Args:
            resources: Lista de StrategyResource a ser hidratada.
            embedding_column: Coluna de embedding a usar (override da estratégia).
        """
        for resource in resources:
            self.hydrate(resource, embedding_column=embedding_column)

    # =========================================================================
    # DESPACHO POR TIPO
    # =========================================================================

    def _load_by_type(
        self,
        resource_type: StrategyResourceType,
        path: Path,
        embedding_column: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Despacha o carregamento com base no tipo do recurso.

        Retorna None se o tipo não tiver loader registrado — o recurso
        permanecerá sem payload e a validação do executor reportará o erro.
        """
        if resource_type == StrategyResourceType.DATASET:
            return self._load_dataset(path, embedding_column=embedding_column)

        if resource_type == StrategyResourceType.INTERMEDIATE_DATASET:
            # Datasets intermediários produzidos por instruções anteriores —
            # mesmo formato parquet do DATASET.
            return self._load_dataset(path, embedding_column=embedding_column)

        if resource_type == StrategyResourceType.PIPELINE_CONFIG:
            return self._load_pipeline_config(path)

        if resource_type == StrategyResourceType.MINING_CONFIG:
            return self._load_mining_config(path)

        # Modelos treinados, dados de trajetória e resultados de validação
        # são persistidos como pickle (preserva ndarray e estrutura Python).
        if resource_type in (
            StrategyResourceType.TRAINED_PIPELINE,
            StrategyResourceType.TRAINED_SIMILARITY_MODEL,
            StrategyResourceType.TRAINED_CALIBRATION_MODEL,
            StrategyResourceType.TRAJECTORY_DATA,
            StrategyResourceType.VALIDATION_RESULT,
        ):
            return self._load_pickle(path)

        # SIMILARITY_SCORES e CALIBRATED_SCORES são persistidos como parquet
        # (rápido, libera GIL via pyarrow). Mesmo formato — apenas o tipo
        # semântico difere; CALIBRATED_SCORES inclui coluna `log_lr` que o
        # loader popula em sample.metadata["log_lr"].
        if resource_type in (
            StrategyResourceType.SIMILARITY_SCORES,
            StrategyResourceType.CALIBRATED_SCORES,
        ):
            return self._load_similarity_scores_from_parquet(path)

        return None

    # =========================================================================
    # LOADERS CONCRETOS
    # =========================================================================

    def _load_dataset(self, path: Path,
                      embedding_column: Optional[str] = None) -> Optional[Any]:
        """
        Carrega um dataset a partir de um arquivo .parquet.

        Tenta EmbeddingDataset primeiro (caso mais comum).
        Se falhar, tenta FaceDataset.
        Em ambos os casos passa embedding_column para que a coluna correta
        seja usada conforme configurado na estratégia.
        """
        first_exc: Optional[Exception] = None
        try:
            return EmbeddingDataset.from_parquet(path, embedding_column=embedding_column)
        except Exception as exc:
            first_exc = exc

        try:
            return FaceDataset.from_parquet(path, embedding_column=embedding_column)
        except Exception as exc:
            msg = (
                f"falha ao carregar dataset '{path}'. "
                f"EmbeddingDataset: {first_exc!r}. FaceDataset: {exc!r}"
            )
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

    def _load_pipeline_config(self, path: Path) -> Optional[TransformationPipelineConfig]:
        """
        Carrega uma TransformationPipelineConfig a partir de um arquivo .json.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TransformationPipelineConfig.from_dict(data)
        except Exception as exc:
            msg = f"falha ao carregar pipeline config '{path}': {exc!r}"
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

    def _load_mining_config(self, path: Path):
        """
        Carrega uma ComparisonMiningConfig a partir de um arquivo .json.
        """
        from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return ComparisonMiningConfig.from_dict(data)
        except Exception as exc:
            msg = f"falha ao carregar mining config '{path}': {exc!r}"
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

    def _load_pickle(self, path: Path) -> Optional[Any]:
        """
        Carrega um objeto Python qualquer a partir de um arquivo pickle.

        Usado para modelos treinados (pipeline, similaridade, calibração)
        persistidos via pickle.HIGHEST_PROTOCOL pelo
        LocalProjectArtifactStorage.

        Em caso de falha, registra a exceção em `self._last_error` para que
        o contexto exiba a mensagem real no LogProgressDialog do usuário,
        e também emite warnings.warn para o stderr (útil em terminais).
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as exc:
            msg = (
                f"falha ao carregar pickle '{path}': "
                f"{type(exc).__name__}: {exc}"
            )
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

    def _load_similarity_scores_from_parquet(self, path: Path) -> Optional[Any]:
        """
        Reconstrói um SimilarityScoreDataset a partir do parquet escrito por
        SimilarityScoreDataset.to_parquet(). EmbeddingInfo é reconstruído com
        embedding vazio (não persistido na coluna) — suficiente para os
        consumidores downstream que só precisam de raw_score/label/identity.
        """
        try:
            import numpy as np
            import pandas as pd
            from faceaudit.models.embedding_info import EmbeddingInfo
            from faceaudit.models.similarity.similarity_score_dataset import (
                SimilarityScoreDataset,
            )
            from faceaudit.models.similarity.similarity_score_sample import (
                SimilarityScoreSample,
            )
        except Exception as exc:
            msg = f"imports falharam para SimilarityScoreDataset: {exc!r}"
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            msg = f"falha ao ler parquet de scores '{path}': {exc!r}"
            self._last_error = msg
            warnings.warn(f"[ResourceLoader] {msg}", RuntimeWarning, stacklevel=2)
            return None

        name = ""
        description = None
        if "similarity_score_dataset_name" in df.columns and len(df) > 0:
            v = df["similarity_score_dataset_name"].iloc[0]
            if v is not None:
                name = str(v)
        if "similarity_score_dataset_description" in df.columns and len(df) > 0:
            v = df["similarity_score_dataset_description"].iloc[0]
            if v is not None:
                description = str(v)

        # Reconstrói samples a partir das colunas mínimas persistidas
        # (raw_score, label, same_person, probe_identity, reference_identity).
        # file_name/image_path/face_hash não são persistidos no novo formato —
        # EmbeddingInfo é reconstruído com strings vazias e embedding vazio.
        empty_emb = np.array([], dtype=np.float64)
        items: list = []

        for row in df.itertuples(index=False):
            row_dict = row._asdict() if hasattr(row, "_asdict") else {}

            probe = EmbeddingInfo(
                embedding=empty_emb,
                face_hash="",
                image_path="",
                identity=row_dict.get("probe_identity"),
                file_name="",
            )
            reference = EmbeddingInfo(
                embedding=empty_emb,
                face_hash="",
                image_path="",
                identity=row_dict.get("reference_identity"),
                file_name="",
            )

            same_person_raw = row_dict.get("same_person")
            same_person = (
                bool(same_person_raw)
                if same_person_raw is not None and not (
                    isinstance(same_person_raw, float) and np.isnan(same_person_raw)
                )
                else None
            )

            # log_lr é coluna opcional — presente em parquets de
            # CALIBRATED_SCORES, ausente em SIMILARITY_SCORES legacy.
            sample_metadata: dict = {}
            log_lr_raw = row_dict.get("log_lr")
            if log_lr_raw is not None and not (
                isinstance(log_lr_raw, float) and np.isnan(log_lr_raw)
            ):
                sample_metadata["log_lr"] = float(log_lr_raw)

            try:
                sample = SimilarityScoreSample(
                    probe=probe,
                    reference=reference,
                    raw_score=float(row_dict.get("raw_score", 0.0)),
                    same_person=same_person,
                    metadata=sample_metadata,
                )
            except Exception:
                continue
            items.append(sample)

        return SimilarityScoreDataset(
            name=name,
            description=description,
            items=items,
        )
