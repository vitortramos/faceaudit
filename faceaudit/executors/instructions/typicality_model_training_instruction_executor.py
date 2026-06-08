# faceaudit/executors/instructions/typicality_model_training_instruction_executor.py

"""Executor da `TYPICALITY_MODEL_TRAINING` unificada (PR 2).

Despacha pelo `typicality_model_type` via
`TypicalityModelBackendRegistry`. A calibração da CDF/threshold OOD
delegada ao `ReferenceScoreCalibrationService` — em PR 2 só
`IN_SAMPLE_LEGACY` está implementado; `CROSS_FIT_IDENTITY` levanta
`NotImplementedError` (entra em PR 4).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.instructions.typicality_model_training_instruction import (
    TypicalityModelTrainingInstruction,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.typicality.knn_typicality_model_config import (
    KNNTypicalityModelConfig,
)
from faceaudit.models.typicality.movmf_typicality_model_config import (
    MoVMFTypicalityModelConfig,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)


class TypicalityModelTrainingInstructionExecutor(StrategyInstructionExecutorBase):
    """Executor unificado da instrução `TYPICALITY_MODEL_TRAINING`."""

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, TypicalityModelTrainingInstruction):
            return self._failure_result(
                "Executor recebeu instrução incompatível."
            )

        cfg = instruction.typicality_model_config
        if cfg is None:
            return self._failure_result(
                "TYPICALITY_MODEL_TRAINING sem typicality_model_config."
            )

        self._log(
            context,
            f"▶ Iniciando treino de modelo de tipicidade "
            f"({instruction.typicality_model_type.label_pt}): "
            f"{instruction.name} ({instruction.instruction_id})",
        )

        try:
            embedding_column = self._resolve_embedding_column(context, cfg)
            input_resource = context.get_resource(instruction.input_dataset_resource_id)
            payload = input_resource.payload

            need_identities = self._backend_needs_identities(cfg)
            embeddings, identities = self._extract_embeddings_and_identities(
                payload, embedding_column, need_identities=need_identities,
            )

            self._log(
                context,
                f"📦 Input: N_faces={embeddings.shape[0]}, "
                f"D={embeddings.shape[1]}, "
                f"backend={instruction.typicality_model_type.to_str()}, "
                f"embedding_column='{embedding_column}'.",
            )

            # Dispatch via registry → trainer_factory por backend.
            trainer_factory = self._trainer_factory_for(cfg)
            service = ReferenceScoreCalibrationService()
            calibration = service.calibrate(
                backend=instruction.typicality_model_type,
                mode=cfg.reference_score_calibration_mode,
                trainer_factory=trainer_factory,
                embeddings=embeddings,
                identity_ids=identities,
                n_folds=cfg.cv_n_folds,
                random_state=getattr(cfg, "random_state", 0),
            )
            model = calibration.refit_model

            self._log(
                context,
                f"✅ Modelo treinado. "
                f"enabled_score_types="
                f"{sorted(t.to_str() for t in model.enabled_score_types)}, "
                f"contract_fp={model.model_contract_fingerprint[:12]}..., "
                f"state_fp={model.model_state_fingerprint[:12]}...",
            )

            output_resource_id = instruction.trained_typicality_model_resource_id
            trained_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.TRAINED_TYPICALITY_MODEL,
                payload=model,
                artifact_ref=None,
            )
            context.set_resource(trained_resource)

            metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "input_dataset_resource_id": instruction.input_dataset_resource_id,
                "embedding_column": embedding_column,
                "typicality_model_type": instruction.typicality_model_type.to_str(),
                "trained_typicality_model_resource_id": output_resource_id,
                "n_input_faces": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "reference_score_calibration_mode":
                    cfg.reference_score_calibration_mode.to_str(),
                "model_contract_fingerprint": model.model_contract_fingerprint,
                "model_state_fingerprint": model.model_state_fingerprint,
                "enabled_score_types": [
                    t.to_str() for t in sorted(
                        model.enabled_score_types, key=lambda x: x.value
                    )
                ],
            }
            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata=metadata,
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao treinar modelo de tipicidade "
                f"{instruction.instruction_id}: {e}",
            )
            return self._failure_result(str(e))

    # ─────────────────────────── validação ────────────────────────────

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, TypicalityModelTrainingInstruction):
            result.add_error(
                "A instrução informada não é do tipo "
                "TypicalityModelTrainingInstruction."
            )
            return result

        if instruction.typicality_model_config is None:
            result.add_error(
                "TYPICALITY_MODEL_TRAINING sem typicality_model_config "
                "embutido. Reabra a instrução na UI para auto-instanciar "
                "o config do tipo selecionado."
            )
            return result

        if not instruction.input_dataset_resource_id:
            result.add_error("A instrução deve definir 'input_dataset_resource_id'.")
        elif not context.has_resource(instruction.input_dataset_resource_id):
            result.add_error(
                f"Recurso de dataset '{instruction.input_dataset_resource_id}' "
                f"não encontrado no contexto."
            )
        else:
            ds_resource = context.get_resource(instruction.input_dataset_resource_id)
            if ds_resource is None or ds_resource.payload is None:
                result.add_error(
                    "Falha ao recuperar o dataset (payload nulo). "
                    "Verifique se o artefato existe em disco e foi hidratado."
                )
            else:
                try:
                    embedding_column = self._resolve_embedding_column(
                        context, instruction.typicality_model_config,
                    )
                    self._extract_embeddings_and_identities(
                        ds_resource.payload, embedding_column,
                        need_identities=self._backend_needs_identities(
                            instruction.typicality_model_config
                        ),
                    )
                except Exception as e:
                    result.add_error(
                        f"Falha ao extrair embeddings do dataset: {e}"
                    )

        if not instruction.trained_typicality_model_resource_id:
            result.add_error(
                "A instrução deve definir "
                "'trained_typicality_model_resource_id'."
            )

        if (
            instruction.trained_typicality_model_resource_id
            and instruction.trained_typicality_model_resource_id
            == instruction.input_dataset_resource_id
        ):
            result.add_warning(
                "trained_typicality_model_resource_id coincide com o recurso "
                "de entrada — o contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ─────────────────────────── helpers ──────────────────────────────

    @staticmethod
    def _backend_needs_identities(cfg) -> bool:
        """Identidades necessárias quando o modo populacional baseia-se
        em identidade. `FACE_LEVEL_RAW` não exige.
        """
        from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
        return cfg.population_weighting_mode != PopulationWeightingMode.FACE_LEVEL_RAW

    def _trainer_factory_for(self, cfg):
        """Devolve um callable sem args que cria uma instância Fittable
        nova do backend escolhido com os hyperparams do config.

        Em PR 2 o trainer é direto: instancia o modelo legado com
        hyperparams compatíveis. PR 4 substitui o trainer do MoVMF por
        uma versão que passa pelo `ReferenceScoreCalibrationService` com
        cross-fit e refit final.
        """
        from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
        from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode

        if isinstance(cfg, KNNTypicalityModelConfig):
            # Mapeia PopulationWeightingMode → TypicalityReferenceMode (legado)
            #   IDENTITY_CENTROIDS → PER_IDENTITY
            #   FACE_LEVEL_RAW → PER_FACE
            #   INVERSE_IDENTITY_FREQUENCY → PR 4 (em PR 2 fail-fast)
            mode = cfg.population_weighting_mode
            if mode == PopulationWeightingMode.IDENTITY_CENTROIDS:
                legacy_mode = TypicalityReferenceMode.PER_IDENTITY
            elif mode == PopulationWeightingMode.FACE_LEVEL_RAW:
                legacy_mode = TypicalityReferenceMode.PER_FACE
            else:
                raise NotImplementedError(
                    "PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY "
                    "no k-NN entra em PR 4 (documentação matemática + "
                    "implementação). Use IDENTITY_CENTROIDS ou "
                    "FACE_LEVEL_RAW em PR 2."
                )

            def factory():
                return KNNTypicalityModel(
                    k=cfg.k,
                    ood_quantile=cfg.ood_quantile,
                    reference_mode=legacy_mode,
                )
            return factory

        if isinstance(cfg, MoVMFTypicalityModelConfig):
            if cfg.population_weighting_mode != PopulationWeightingMode.FACE_LEVEL_RAW:
                # MoVMF + IDENTITY_CENTROIDS/INVERSE_IDENTITY_FREQUENCY
                # entra em PR 4.
                raise NotImplementedError(
                    "PopulationWeightingMode != FACE_LEVEL_RAW no MoVMF "
                    "exige PR 4 (endurecimento experimental). Use "
                    "FACE_LEVEL_RAW em PR 2 ou troque para o backend "
                    "k-NN."
                )

            def factory():
                return MoVMFTypicalityModel(
                    population_label=cfg.population_label,
                    centering=cfg.centering,
                    n_components=cfg.n_components,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    convergence_patience=cfg.convergence_patience,
                    random_state=cfg.random_state,
                    n_init=cfg.n_init,
                    min_component_weight=cfg.min_component_weight,
                    min_kappa=cfg.min_kappa,
                    max_kappa=cfg.max_kappa,
                    ood_quantile=cfg.ood_quantile,
                )
            return factory

        raise TypeError(
            f"Config type desconhecido: {type(cfg).__name__}."
        )

    @staticmethod
    def _resolve_embedding_column(
        context: StrategyExecutionContextProtocol,
        cfg,
    ) -> str:
        """Override do config tem prioridade; senão usa o da estratégia."""
        override = getattr(cfg, "embedding_column", None)
        if override:
            return str(override)
        col = getattr(context, "embedding_column", None)
        if not col:
            raise ValueError(
                "A estratégia não definiu 'embedding_column' e o config "
                "não fornece override. Configure a coluna no nível da "
                "estratégia ou no config do backend."
            )
        return str(col)

    @staticmethod
    def _extract_embeddings_and_identities(
        payload,
        embedding_column: str,
        *,
        need_identities: bool,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Extrai (N, D) embeddings + lista de N identidades (se exigido).

        Reaproveita a lógica do executor legado V2.14.
        """
        if payload is None:
            raise ValueError("Payload do dataset é None.")

        if isinstance(payload, EmbeddingDataset):
            m = payload.matrix
            if m.size == 0:
                raise ValueError("EmbeddingDataset vazio.")
            if m.ndim != 2:
                raise ValueError(f"matrix shape inesperada: {m.shape}")
            identities = None
            if need_identities:
                identities = [str(item.identity) for item in payload.items]
                if any(ident is None or ident == "None" for ident in identities):
                    raise ValueError(
                        "Modo populacional baseado em identidade exige "
                        "todas as faces com identidade definida — "
                        "encontradas faces com identidade None."
                    )
            return m, identities

        items = getattr(payload, "items", None)
        if items is None:
            raise ValueError(
                f"Payload {type(payload).__name__} não expõe 'items' nem 'matrix'."
            )

        vectors = []
        identities_list: List[str] = [] if need_identities else None  # type: ignore[assignment]
        for it in items:
            v = getattr(it, embedding_column, None)
            if v is None:
                ff = getattr(it, "ff", None)
                if ff is not None:
                    v = getattr(ff, embedding_column, None)
            if v is None:
                raise ValueError(
                    f"Item sem atributo/coluna '{embedding_column}'."
                )
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.size == 0:
                raise ValueError("Embedding vazio em item.")
            vectors.append(arr)
            if need_identities:
                ident = getattr(it, "identity", None)
                if ident is None:
                    raise ValueError(
                        "Modo populacional baseado em identidade exige "
                        "todas as faces com identidade definida."
                    )
                identities_list.append(str(ident))

        if not vectors:
            raise ValueError("Dataset não tem items.")

        m = np.vstack(vectors)
        return m, (identities_list if need_identities else None)
