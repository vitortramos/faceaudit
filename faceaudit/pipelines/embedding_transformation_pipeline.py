# faceaudit/pipelines/embedding_transformation_pipeline.py

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy
from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.protocols.embedding_transformation_pipeline_protocol import (
    EmbeddingTransformationPipelineProtocol,
)
from faceaudit.protocols.embedding_transformation_step_protocol import (
    EmbeddingTransformationStepProtocol,
)
from faceaudit.services.embedding_math_service import EmbeddingMathService

# Reaproveitamento da arquitetura atual
from faceaudit.transformators.centering_transformation import CenteringTransformation
from faceaudit.transformators.coral_transformation import CoralTransformation
from faceaudit.transformators.identity_aggregation import IdentityAggregation
from faceaudit.transformators.l2_normalization import L2Normalization
from faceaudit.transformators.lda_transformation import LdaTransformation
from faceaudit.transformators.whitening_transformation import WhiteningTransformation


class EmbeddingTransformationPipeline(EmbeddingTransformationPipelineProtocol):
    """
    Novo pipeline de transformação de embeddings da arquitetura paralela.

    Suporta dois modos:

    - LINEAR:
        Um único fluxo de transformação.
    - DUAL:
        Dois fluxos (Source/Target), com treino sincronizado e possibilidade
        de assimetria estrutural entre os ramos.

    Esta implementação reaproveita os transformadores existentes no projeto,
    sem alterar o legado.

    Neste primeiro corte, eu ignorei por enquanto a PipelineApplicationPolicy 
    dentro de transform().
    Ela está na assinatura, mas ainda não altera comportamento real.

    Achei melhor assim por agora, para não inventar persistência/checkpoints 
    antes da hora.

    """

    _STEP_CLASS_MAP: Dict[TransformationType, Type[EmbeddingTransformationStepProtocol]] = {
        TransformationType.L2_NORMALIZATION: L2Normalization,
        TransformationType.WHITENING_TRANSFORMATION: WhiteningTransformation,
        TransformationType.CORAL_TRANSFORMATION: CoralTransformation,
        TransformationType.IDENTITY_AGGREGATION: IdentityAggregation,
        TransformationType.LDA_TRANSFORMATION: LdaTransformation,
        TransformationType.CENTERING_TRANSFORMATION: CenteringTransformation,
    }

    def __init__(
        self,
        mode: PipelineMode,
        config: Optional[TransformationPipelineConfig] = None,
        source_config: Optional[TransformationPipelineConfig] = None,
        target_config: Optional[TransformationPipelineConfig] = None,
        math_service: Optional[EmbeddingMathService] = None,
    ):
        self._mode = mode
        self._config = config
        self._source_config = source_config
        self._target_config = target_config
        self.math_service = math_service or EmbeddingMathService()

        self._steps: List[EmbeddingTransformationStepProtocol] = []
        self._source_steps: List[EmbeddingTransformationStepProtocol] = []
        self._target_steps: List[EmbeddingTransformationStepProtocol] = []

        self._fitted: bool = False
        self._training_metrics: Dict[str, Any] = {}
        self._execution_metrics: Dict[str, Any] = {}

        self._validate_init()
        self._build()

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def mode(self) -> PipelineMode:
        return self._mode

    @property
    def config(self) -> Optional[TransformationPipelineConfig]:
        return self._config

    @property
    def source_config(self) -> Optional[TransformationPipelineConfig]:
        return self._source_config

    @property
    def target_config(self) -> Optional[TransformationPipelineConfig]:
        return self._target_config

    @property
    def steps(self) -> List[EmbeddingTransformationStepProtocol]:
        return self._steps

    @property
    def source_steps(self) -> List[EmbeddingTransformationStepProtocol]:
        return self._source_steps

    @property
    def target_steps(self) -> List[EmbeddingTransformationStepProtocol]:
        return self._target_steps

    # =========================================================================
    # CONSTRUÇÃO
    # =========================================================================

    def _validate_init(self) -> None:
        if self.mode == PipelineMode.LINEAR:
            if self.config is None:
                raise ValueError("PipelineMode.LINEAR exige 'config'.")
            if self.source_config is not None or self.target_config is not None:
                raise ValueError(
                    "PipelineMode.LINEAR não deve receber source_config/target_config."
                )

        elif self.mode == PipelineMode.DUAL:
            if self.source_config is None or self.target_config is None:
                raise ValueError(
                    "PipelineMode.DUAL exige 'source_config' e 'target_config'."
                )
            if self.config is not None:
                raise ValueError("PipelineMode.DUAL não deve receber 'config'.")

        else:
            raise ValueError(f"Modo de pipeline não suportado: {self.mode}")

    def _build(self) -> None:
        if self.mode == PipelineMode.LINEAR:
            self._steps = self._build_steps_from_config(self.config)
            self._source_steps = []
            self._target_steps = []
        else:
            self._steps = []
            self._source_steps = self._build_steps_from_config(self.source_config)
            self._target_steps = self._build_steps_from_config(self.target_config)

    def _build_steps_from_config(
        self,
        pipeline_config: Optional[TransformationPipelineConfig]
    ) -> List[EmbeddingTransformationStepProtocol]:
        if pipeline_config is None:
            return []

        built_steps: List[EmbeddingTransformationStepProtocol] = []

        for t_config in pipeline_config.transformators:
            step_class = self._STEP_CLASS_MAP.get(t_config.type)
            if step_class is None:
                raise ValueError(f"Tipo de transformador não suportado: {t_config.type}")

            step = step_class(
                math_service=self.math_service,
                name=t_config.name,
                config=t_config.config,
            )
            built_steps.append(step)

        return built_steps

    # =========================================================================
    # FIT - LINEAR
    # =========================================================================

    def fit(
        self,
        dataset: EmbeddingDataset,
        training_policy: Optional[PipelineTrainingPolicy] = None,
    ) -> None:
        if self.mode != PipelineMode.LINEAR:
            raise RuntimeError("Use fit_dual() quando o pipeline estiver em modo DUAL.")

        if dataset is None or not dataset.items:
            raise ValueError("O dataset de treino do pipeline linear não pode ser vazio.")

        policy = training_policy or PipelineTrainingPolicy.default_linear()
        current_data = dataset

        per_step_metrics: List[Dict[str, Any]] = []

        for idx, step in enumerate(self.steps):
            step.fit(source_dataset=current_data, target_dataset=None)
            current_data = step.transform(current_data)

            per_step_metrics.append(
                {
                    "index": idx,
                    "step_name": getattr(step, "name", type(step).__name__),
                    "transformation_type": step.transformation_type.to_str()
                    if hasattr(step.transformation_type, "to_str")
                    else str(step.transformation_type),
                    "training_metrics": self._get_step_training_metrics(step),
                }
            )

        self._fitted = True
        self._training_metrics = {
            "mode": self.mode.to_str(),
            "n_steps": len(self.steps),
            "policy": policy.to_dict(),
            "step_metrics": per_step_metrics,
        }

    # =========================================================================
    # FIT - DUAL
    # =========================================================================

    def fit_dual(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: EmbeddingDataset,
        training_policy: Optional[PipelineTrainingPolicy] = None,
    ) -> None:
        if self.mode != PipelineMode.DUAL:
            raise RuntimeError("Use fit() quando o pipeline estiver em modo LINEAR.")

        if source_dataset is None or not source_dataset.items:
            raise ValueError("O source_dataset não pode ser vazio.")
        if target_dataset is None or not target_dataset.items:
            raise ValueError("O target_dataset não pode ser vazio.")

        policy = training_policy or PipelineTrainingPolicy.default_dual()
        mapping = self._build_dual_mapping()

        current_source = source_dataset
        current_target = target_dataset

        per_step_metrics: List[Dict[str, Any]] = []

        for src_index, src_step in enumerate(self.source_steps):
            fit_source, fit_target = self._choose_dual_fit_datasets(
                step=src_step,
                current_source=current_source,
                current_target=current_target,
                training_policy=policy,
            )

            src_step.fit(source_dataset=fit_source, target_dataset=fit_target)

            tgt_index = mapping.get(src_index)
            tgt_step = self.target_steps[tgt_index] if tgt_index is not None else None

            if tgt_step is not None and policy.copy_shared_state_to_target:
                self._copy_step_state(src=src_step, dst=tgt_step)

            current_source = src_step.transform(current_source)

            if tgt_step is not None:
                current_target = tgt_step.transform(current_target)

            per_step_metrics.append(
                {
                    "source_index": src_index,
                    "target_index": tgt_index,
                    "step_name": getattr(src_step, "name", type(src_step).__name__),
                    "transformation_type": src_step.transformation_type.to_str()
                    if hasattr(src_step.transformation_type, "to_str")
                    else str(src_step.transformation_type),
                    "training_metrics": self._get_step_training_metrics(src_step),
                }
            )

        self._fitted = True
        self._training_metrics = {
            "mode": self.mode.to_str(),
            "n_source_steps": len(self.source_steps),
            "n_target_steps": len(self.target_steps),
            "policy": policy.to_dict(),
            "mapping": mapping,
            "step_metrics": per_step_metrics,
        }

    def _build_dual_mapping(self) -> Dict[int, Optional[int]]:
        """
        Constrói o mapeamento Source -> Target por subsequência de tipos.

        Regra:
        - o Target deve ser uma subsequência compatível do Source
        - a ordem deve ser preservada
        """
        mapping: Dict[int, Optional[int]] = {}
        target_pointer = 0

        source_types = [step.transformation_type for step in self.source_steps]
        target_types = [step.transformation_type for step in self.target_steps]

        for src_index, src_type in enumerate(source_types):
            if target_pointer < len(target_types):
                tgt_type = target_types[target_pointer]

                src_value = src_type.value if hasattr(src_type, "value") else str(src_type)
                tgt_value = tgt_type.value if hasattr(tgt_type, "value") else str(tgt_type)

                if src_value == tgt_value:
                    mapping[src_index] = target_pointer
                    target_pointer += 1
                else:
                    mapping[src_index] = None
            else:
                mapping[src_index] = None

        if target_pointer < len(target_types):
            remaining = target_types[target_pointer:]
            remaining_names = [
                item.name if hasattr(item, "name") else str(item) for item in remaining
            ]
            raise ValueError(
                "Incompatibilidade estrutural entre source_config e target_config: "
                "o Target não é subsequência compatível do Source. "
                f"Tipos restantes no Target: {remaining_names}"
            )

        return mapping

    def _choose_dual_fit_datasets(
        self,
        step: EmbeddingTransformationStepProtocol,
        current_source: EmbeddingDataset,
        current_target: EmbeddingDataset,
        training_policy: PipelineTrainingPolicy,
    ) -> tuple[EmbeddingDataset, Optional[EmbeddingDataset]]:
        """
        Decide quais datasets entram no fit do step do ramo Source.
        """
        fit_source = current_source
        fit_target: Optional[EmbeddingDataset] = None

        if getattr(step, "requires_target_dataset", False):
            fit_target = current_target

        t_type = step.transformation_type
        t_value = t_type.value if hasattr(t_type, "value") else str(t_type)

        if t_value == TransformationType.CORAL_TRANSFORMATION.value:
            if training_policy.use_target_for_coral:
                fit_target = current_target

        elif t_value == TransformationType.WHITENING_TRANSFORMATION.value:
            if training_policy.use_target_for_whitening:
                fit_source = current_source + current_target

        elif t_value == TransformationType.LDA_TRANSFORMATION.value:
            if training_policy.use_target_for_lda:
                fit_source = current_source + current_target

        return fit_source, fit_target

    def _copy_step_state(
        self,
        src: EmbeddingTransformationStepProtocol,
        dst: EmbeddingTransformationStepProtocol,
    ) -> None:
        if hasattr(src, "get_state") and hasattr(dst, "set_state"):
            state = copy.deepcopy(src.get_state())
            dst.set_state(state)
            return

        # fallback defensivo
        state = copy.deepcopy(getattr(src, "__dict__", {}))
        for key in ("math_service", "logger", "name", "_name"):
            state.pop(key, None)
        for key, value in state.items():
            setattr(dst, key, value)

    # =========================================================================
    # TRANSFORM
    # =========================================================================

    def transform(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        if self.mode != PipelineMode.LINEAR:
            raise RuntimeError(
                "Use transform_source()/transform_target() quando o pipeline estiver em modo DUAL."
            )
        self._ensure_fitted()

        _ = application_policy or PipelineApplicationPolicy.default()
        current_data = dataset

        step_metrics: List[Dict[str, Any]] = []

        for idx, step in enumerate(self.steps):
            current_data = step.transform(current_data)
            step_metrics.append(
                {
                    "index": idx,
                    "step_name": getattr(step, "name", type(step).__name__),
                    "execution_metrics": self._get_step_execution_metrics(step),
                }
            )

        self._execution_metrics = {
            "mode": self.mode.to_str(),
            "n_steps": len(self.steps),
            "step_metrics": step_metrics,
        }

        return current_data

    def transform_source(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        if self.mode != PipelineMode.DUAL:
            raise RuntimeError("Use transform() quando o pipeline estiver em modo LINEAR.")
        self._ensure_fitted()

        _ = application_policy or PipelineApplicationPolicy.default()
        current_data = dataset
        step_metrics: List[Dict[str, Any]] = []

        for idx, step in enumerate(self.source_steps):
            current_data = step.transform(current_data)
            step_metrics.append(
                {
                    "index": idx,
                    "step_name": getattr(step, "name", type(step).__name__),
                    "execution_metrics": self._get_step_execution_metrics(step),
                }
            )

        self._execution_metrics = {
            "mode": self.mode.to_str(),
            "branch": "source",
            "n_steps": len(self.source_steps),
            "step_metrics": step_metrics,
        }

        return current_data

    def transform_target(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        if self.mode != PipelineMode.DUAL:
            raise RuntimeError("Use transform() quando o pipeline estiver em modo LINEAR.")
        self._ensure_fitted()

        _ = application_policy or PipelineApplicationPolicy.default()
        current_data = dataset
        step_metrics: List[Dict[str, Any]] = []

        for idx, step in enumerate(self.target_steps):
            current_data = step.transform(current_data)
            step_metrics.append(
                {
                    "index": idx,
                    "step_name": getattr(step, "name", type(step).__name__),
                    "execution_metrics": self._get_step_execution_metrics(step),
                }
            )

        self._execution_metrics = {
            "mode": self.mode.to_str(),
            "branch": "target",
            "n_steps": len(self.target_steps),
            "step_metrics": step_metrics,
        }

        return current_data

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if self.mode != PipelineMode.LINEAR:
            raise RuntimeError(
                "Use transform_single_source()/transform_single_target() em modo DUAL."
            )
        self._ensure_fitted()

        current_item = embedding_info
        for step in self.steps:
            current_item = step.transform_single(current_item)
        return current_item

    def transform_single_source(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if self.mode != PipelineMode.DUAL:
            raise RuntimeError("Use transform_single() quando o pipeline estiver em modo LINEAR.")
        self._ensure_fitted()

        current_item = embedding_info
        for step in self.source_steps:
            current_item = step.transform_single(current_item)
        return current_item

    def transform_single_target(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if self.mode != PipelineMode.DUAL:
            raise RuntimeError("Use transform_single() quando o pipeline estiver em modo LINEAR.")
        self._ensure_fitted()

        current_item = embedding_info
        for step in self.target_steps:
            current_item = step.transform_single(current_item)
        return current_item

    # =========================================================================
    # MÉTRICAS
    # =========================================================================

    def get_training_metrics(self) -> Dict[str, Any]:
        return self._training_metrics

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._execution_metrics

    def _get_step_training_metrics(
        self,
        step: EmbeddingTransformationStepProtocol,
    ) -> Dict[str, Any]:
        if hasattr(step, "get_training_metrics"):
            try:
                return step.get_training_metrics()
            except Exception:
                pass

        if hasattr(step, "get_execution_metrics"):
            try:
                return step.get_execution_metrics()
            except Exception:
                pass

        return {}

    def _get_step_execution_metrics(
        self,
        step: EmbeddingTransformationStepProtocol,
    ) -> Dict[str, Any]:
        if hasattr(step, "get_execution_metrics"):
            try:
                return step.get_execution_metrics()
            except Exception:
                return {}
        return {}

    # =========================================================================
    # ESTADO / PERSISTÊNCIA
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "mode": self.mode.to_str(),
            "fitted": self._fitted,
            "training_metrics": self._training_metrics,
            "execution_metrics": self._execution_metrics,
        }

        if self.mode == PipelineMode.LINEAR:
            state["config"] = self.config.to_dict() if self.config is not None else None
            state["steps_state"] = [step.get_state() for step in self.steps]

        else:
            state["source_config"] = (
                self.source_config.to_dict() if self.source_config is not None else None
            )
            state["target_config"] = (
                self.target_config.to_dict() if self.target_config is not None else None
            )
            state["source_steps_state"] = [step.get_state() for step in self.source_steps]
            state["target_steps_state"] = [step.get_state() for step in self.target_steps]

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        if not state:
            return

        self._mode = PipelineMode.from_str(state.get("mode"))

        config_dict = state.get("config")
        source_config_dict = state.get("source_config")
        target_config_dict = state.get("target_config")

        self._config = (
            TransformationPipelineConfig.from_dict(config_dict)
            if config_dict is not None
            else None
        )
        self._source_config = (
            TransformationPipelineConfig.from_dict(source_config_dict)
            if source_config_dict is not None
            else None
        )
        self._target_config = (
            TransformationPipelineConfig.from_dict(target_config_dict)
            if target_config_dict is not None
            else None
        )

        self._validate_init()
        self._build()

        if self.mode == PipelineMode.LINEAR:
            steps_state = state.get("steps_state", [])
            for step, step_state in zip(self.steps, steps_state):
                step.set_state(step_state)

        else:
            source_steps_state = state.get("source_steps_state", [])
            target_steps_state = state.get("target_steps_state", [])

            for step, step_state in zip(self.source_steps, source_steps_state):
                step.set_state(step_state)

            for step, step_state in zip(self.target_steps, target_steps_state):
                step.set_state(step_state)

        self._fitted = bool(state.get("fitted", False))
        self._training_metrics = state.get("training_metrics", {})
        self._execution_metrics = state.get("execution_metrics", {})

    def save(self, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self.get_state(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pipeline não encontrado: {file_path}")

        with open(file_path, "rb") as f:
            state = pickle.load(f)

        self.set_state(state)

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "O pipeline ainda não foi treinado. Execute fit() ou fit_dual() antes."
            )

    @classmethod
    def load_from_file(
        cls,
        path: str,
        math_service: Optional[EmbeddingMathService] = None,
    ) -> "EmbeddingTransformationPipeline":
        """
        Helper opcional para carregar o pipeline diretamente do disco.
        """
        dummy = cls(
            mode=PipelineMode.LINEAR,
            config=TransformationPipelineConfig(name="temp"),
            math_service=math_service,
        )
        dummy.load(path)
        return dummy