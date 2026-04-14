# faceaudit/services/strategy_inference_service.py

from __future__ import annotations

from typing import Optional

import numpy as np

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.services.strategy_resource_loader_service import StrategyResourceLoaderService


class StrategyInferenceService:
    """
    Serviço de inferência da nova arquitetura.

    Executa o fluxo completo de comparação forense a partir de um
    LRStrategyTrainedModel já carregado:

        probe dataset  ──► pipeline.transform_source()  ──┐
                                                           ├─► score_matrix() ──► calibrate() ──► SimilarityScoreDataset
        reference dataset ──► pipeline.transform_target() ─┘

    Responsabilidades:
    - Hidratar probe/reference a partir de disco (via StrategyResourceLoaderService)
      se seus payloads estiverem None.
    - Detectar o modo do pipeline (LINEAR/DUAL) e usar o método correto:
        • LINEAR: pipeline.transform() nos dois datasets
        • DUAL  : pipeline.transform_source() no probe,
                  pipeline.transform_target() no reference
    - Computar a matriz de scores para todos os pares (probe × reference).
    - Se o modelo treinado contiver TRAINED_CALIBRATION_MODEL, calibrar os
      scores e salvar o log-LR em SimilarityScoreSample.metadata["log_lr"].

    Convenção de modos DUAL:
        probe     → ramo Source (imagem questionada / casework)
        reference → ramo Target (imagem padrão / referência conhecida)

    Uso típico:
        svc = StrategyInferenceService(project.storage)
        result = svc.run(trained_model, probe_resource, reference_resource)

        # Para validação:
        genuine_log_lrs   = [s.metadata["log_lr"] for s in result if s.same_person]
        impostor_log_lrs  = [s.metadata["log_lr"] for s in result if not s.same_person]

        # Para perícia (LR consolidado):
        all_log_lrs = np.array([s.metadata.get("log_lr", s.raw_score) for s in result])
    """

    def __init__(
        self,
        artifact_storage: Optional[ArtifactStorageProtocol] = None,
    ) -> None:
        self._resource_loader = (
            StrategyResourceLoaderService(artifact_storage)
            if artifact_storage is not None
            else None
        )

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def run(
        self,
        trained_model: LRStrategyTrainedModel,
        probe_resource: StrategyResource,
        reference_resource: StrategyResource,
    ) -> SimilarityScoreDataset:
        """
        Executa a inferência completa e retorna o dataset de scores.

        Args:
            trained_model:      Modelo treinado da estratégia (já hidratado via pickle).
            probe_resource:     Recurso do dataset probe (imagem questionada).
                                Pode ter payload=None — será hidratado via artifact_ref
                                caso artifact_storage tenha sido fornecido no construtor.
            reference_resource: Recurso do dataset reference (imagem padrão / referência).
                                Mesmas regras de hidratação de probe_resource.

        Returns:
            SimilarityScoreDataset com um SimilarityScoreSample por par (probe × reference).
            Cada sample contém:
                - raw_score: score bruto do modelo de similaridade
                - same_person: inferido das identidades (True/False/None)
                - metadata["log_lr"]: log-LR calibrado, quando disponível

        Raises:
            ValueError: Se TRAINED_SIMILARITY_MODEL não for encontrado no modelo treinado,
                        ou se os datasets não puderem ser carregados.
        """
        self._hydrate_if_needed(probe_resource)
        self._hydrate_if_needed(reference_resource)

        probe_dataset = self._require_dataset(probe_resource, "probe")
        reference_dataset = self._require_dataset(reference_resource, "reference")

        pipeline = self._find_payload_by_type(trained_model, StrategyResourceType.TRAINED_PIPELINE)
        similarity_model = self._find_payload_by_type(
            trained_model, StrategyResourceType.TRAINED_SIMILARITY_MODEL
        )
        calibration_model = self._find_payload_by_type(
            trained_model, StrategyResourceType.TRAINED_CALIBRATION_MODEL
        )

        if similarity_model is None:
            raise ValueError(
                f"Modelo treinado '{trained_model.strategy_id}' não contém "
                "TRAINED_SIMILARITY_MODEL. Execute a estratégia antes de inferir."
            )

        transformed_probe, transformed_reference = self._apply_pipeline(
            pipeline, probe_dataset, reference_dataset
        )

        raw_matrix = similarity_model.score_matrix(
            probe_matrix=transformed_probe.matrix,
            ref_matrix=transformed_reference.matrix,
        )

        log_lr_matrix: Optional[np.ndarray] = None
        if calibration_model is not None:
            flat_scores = raw_matrix.ravel()
            flat_log_lrs = calibration_model.calibrate_scores(flat_scores)
            log_lr_matrix = flat_log_lrs.reshape(raw_matrix.shape)

        return self._build_score_dataset(
            strategy_id=trained_model.strategy_id,
            probe_dataset=transformed_probe,
            reference_dataset=transformed_reference,
            raw_matrix=raw_matrix,
            log_lr_matrix=log_lr_matrix,
        )

    # =========================================================================
    # HIDRATAÇÃO
    # =========================================================================

    def _hydrate_if_needed(self, resource: StrategyResource) -> None:
        if resource.payload is None and self._resource_loader is not None:
            self._resource_loader.hydrate(resource)

    def _require_dataset(
        self,
        resource: StrategyResource,
        label: str,
    ) -> EmbeddingDataset:
        if resource.payload is None:
            raise ValueError(
                f"Não foi possível carregar o dataset '{label}' "
                f"(resource_id='{resource.resource_id}'). "
                "Verifique se artifact_storage foi fornecido e o arquivo existe."
            )
        if not isinstance(resource.payload, EmbeddingDataset):
            raise ValueError(
                f"O recurso '{label}' (resource_id='{resource.resource_id}') "
                f"não é um EmbeddingDataset — tipo encontrado: {type(resource.payload).__name__}"
            )
        return resource.payload

    # =========================================================================
    # TRANSFORMAÇÃO
    # =========================================================================

    def _apply_pipeline(
        self,
        pipeline,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
    ):
        """
        Transforma probe e reference usando o pipeline treinado.

        Se pipeline for None, retorna os datasets originais sem transformação.
        """
        if pipeline is None:
            return probe_dataset, reference_dataset

        if pipeline.mode == PipelineMode.LINEAR:
            transformed_probe = pipeline.transform(probe_dataset)
            transformed_reference = pipeline.transform(reference_dataset)
        else:
            # DUAL: probe → source branch, reference → target branch
            transformed_probe = pipeline.transform_source(probe_dataset)
            transformed_reference = pipeline.transform_target(reference_dataset)

        return transformed_probe, transformed_reference

    # =========================================================================
    # CONSTRUÇÃO DO DATASET DE SCORES
    # =========================================================================

    def _build_score_dataset(
        self,
        strategy_id: str,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
        raw_matrix: np.ndarray,
        log_lr_matrix: Optional[np.ndarray],
    ) -> SimilarityScoreDataset:
        """
        Constrói SimilarityScoreDataset a partir da matriz de scores (probe × reference).
        """
        result = SimilarityScoreDataset(
            name=f"{strategy_id}_inference",
            description=(
                f"Inferência da estratégia '{strategy_id}': "
                f"{len(probe_dataset.items)} probe(s) × "
                f"{len(reference_dataset.items)} reference(s)"
            ),
        )

        for i, probe_info in enumerate(probe_dataset.items):
            for j, reference_info in enumerate(reference_dataset.items):
                raw_score = float(raw_matrix[i, j])

                sample_metadata = {}
                if log_lr_matrix is not None:
                    sample_metadata["log_lr"] = float(log_lr_matrix[i, j])

                sample = SimilarityScoreSample(
                    probe=probe_info,
                    reference=reference_info,
                    raw_score=raw_score,
                    metadata=sample_metadata,
                )
                result.add(sample)

        return result

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _find_payload_by_type(
        self,
        trained_model: LRStrategyTrainedModel,
        resource_type: StrategyResourceType,
    ):
        """
        Retorna o payload do primeiro recurso do tipo informado no modelo treinado.
        Retorna None se não encontrado.
        """
        for resource in trained_model.trained_resources:
            if resource.resource_type == resource_type and resource.payload is not None:
                return resource.payload
        return None
