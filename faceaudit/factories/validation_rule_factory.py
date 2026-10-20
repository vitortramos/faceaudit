# faceaudit/factories/validation_rule_factory.py

from __future__ import annotations

from typing import Optional

from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol


class ValidationRuleFactory:
    """
    Instancia regras concretas de `faceaudit/validation_rules/` a partir de
    um `ValidationRuleConfig` serializável.

    Estratégia de `rule_id`:
        Derivado do `instruction_id` + `rule_type` quando informado:
            f"{instruction_id}_{rule_type.value.lower()}"
        Senão usa só o tipo (ex: regras a nível de estratégia).

        Garante unicidade por instrução e legibilidade no log.

    Tipos suportados: todos os valores de `ValidationRuleType`. Adicionar
    novos tipos exige duas mudanças aqui: import da classe concreta e
    branch no dispatch.

    Tratamento de params ausentes/inválidos: cada branch é defensivo —
    se um param obrigatório falta (ex: `dataset_resource_id`), levanta
    `ValueError` com mensagem específica. O pipeline de validação trata
    a exceção e reporta como erro CRITICAL na instrução.
    """

    @staticmethod
    def create(
        config: ValidationRuleConfig,
        instruction_id: Optional[str] = None,
    ) -> ValidationRuleProtocol:
        if config is None:
            raise ValueError("ValidationRuleConfig não pode ser nulo.")

        params = config.params or {}
        rule_type = config.rule_type
        rule_id = ValidationRuleFactory._make_rule_id(rule_type, instruction_id)

        if rule_type == ValidationRuleType.FISHER:
            from faceaudit.validation_rules.fisher_rule import FisherRule
            return FisherRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                safety_margin_ratio=float(params.get("safety_margin_ratio", 0.2)),
            )

        if rule_type == ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES:
            from faceaudit.validation_rules.min_identities_with_multiple_samples_rule import (
                MinIdentitiesWithMultipleSamplesRule,
            )
            return MinIdentitiesWithMultipleSamplesRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                min_required=int(params.get("min_required", 2)),
                safety_threshold=int(params.get("safety_threshold", 10)),
            )

        if rule_type == ValidationRuleType.MIN_CALIBRATION_PAIRS:
            from faceaudit.validation_rules.min_calibration_pairs_rule import (
                MinCalibrationPairsRule,
            )
            return MinCalibrationPairsRule(
                rule_id=rule_id,
                scores_resource_id=ValidationRuleFactory._req_str(
                    params, "scores_resource_id", rule_type
                ),
                min_genuines=int(params.get("min_genuines", 100)),
                min_impostors=int(params.get("min_impostors", 500)),
                warn_ratio_threshold=float(params.get("warn_ratio_threshold", 0.02)),
            )

        if rule_type == ValidationRuleType.DATASET_ISOLATION:
            from faceaudit.validation_rules.dataset_isolation_rule import (
                DatasetIsolationRule,
            )
            return DatasetIsolationRule(
                rule_id=rule_id,
                dataset_a_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_a_resource_id", rule_type
                ),
                dataset_b_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_b_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.DATASET_OVERLAP_REQUIRED:
            from faceaudit.validation_rules.dataset_overlap_required_rule import (
                DatasetOverlapRequiredRule,
            )
            return DatasetOverlapRequiredRule(
                rule_id=rule_id,
                dataset_a_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_a_resource_id", rule_type
                ),
                dataset_b_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_b_resource_id", rule_type
                ),
                min_overlap=int(params.get("min_overlap", 1)),
                warn_below_ratio=float(params.get("warn_below_ratio", 0.1)),
            )

        if rule_type == ValidationRuleType.HOLDOUT_LEAKAGE:
            from faceaudit.validation_rules.holdout_leakage_rule import (
                HoldoutLeakageRule,
            )
            return HoldoutLeakageRule(
                rule_id=rule_id,
                holdout_resource_ids=ValidationRuleFactory._req_list_str(
                    params, "holdout_resource_ids", rule_type
                ),
                train_resource_ids=ValidationRuleFactory._req_list_str(
                    params, "train_resource_ids", rule_type
                ),
            )

        if rule_type == ValidationRuleType.DUAL_PIPELINE_DIM_COMPAT:
            from faceaudit.validation_rules.dual_pipeline_dimension_compatibility_rule import (
                DualPipelineDimensionCompatibilityRule,
            )
            return DualPipelineDimensionCompatibilityRule(
                rule_id=rule_id,
                source_config_resource_id=ValidationRuleFactory._req_str(
                    params, "source_config_resource_id", rule_type
                ),
                target_config_resource_id=ValidationRuleFactory._req_str(
                    params, "target_config_resource_id", rule_type
                ),
                embedding_dimensions=params.get("embedding_dimensions"),
            )

        if rule_type == ValidationRuleType.DUAL_PIPELINE_SUBSEQUENCE:
            from faceaudit.validation_rules.dual_pipeline_subsequence_rule import (
                DualPipelineSubsequenceRule,
            )
            return DualPipelineSubsequenceRule(
                rule_id=rule_id,
                source_config_resource_id=ValidationRuleFactory._req_str(
                    params, "source_config_resource_id", rule_type
                ),
                target_config_resource_id=ValidationRuleFactory._req_str(
                    params, "target_config_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.MIN_IDENTITIES:
            from faceaudit.validation_rules.min_identities_rule import (
                MinIdentitiesRule,
            )
            return MinIdentitiesRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                min_required=int(params.get("min_required", 2)),
                safety_threshold=int(params.get("safety_threshold", 5)),
            )

        if rule_type == ValidationRuleType.MIN_SAMPLES:
            from faceaudit.validation_rules.min_samples_rule import (
                MinSamplesRule,
            )
            return MinSamplesRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                min_required=int(params.get("min_required", 2)),
                safety_threshold=int(params.get("safety_threshold", 100)),
            )

        if rule_type == ValidationRuleType.MODEL_FITTED:
            from faceaudit.validation_rules.model_fitted_rule import (
                ModelFittedRule,
            )
            return ModelFittedRule(
                rule_id=rule_id,
                model_resource_id=ValidationRuleFactory._req_str(
                    params, "model_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.EMBEDDINGS_COMPLETE:
            from faceaudit.validation_rules.embeddings_complete_rule import (
                EmbeddingsCompleteRule,
            )
            return EmbeddingsCompleteRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.EMBEDDING_DIM_MATCHES_MODEL:
            from faceaudit.validation_rules.embedding_dim_matches_model_rule import (
                EmbeddingDimMatchesModelRule,
            )
            return EmbeddingDimMatchesModelRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                model_resource_id=ValidationRuleFactory._req_str(
                    params, "model_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.SCORE_CALIBRATION_STATE:
            from faceaudit.validation_rules.score_calibration_state_rule import (
                ScoreCalibrationStateRule,
            )
            return ScoreCalibrationStateRule(
                rule_id=rule_id,
                scores_resource_id=ValidationRuleFactory._req_str(
                    params, "scores_resource_id", rule_type
                ),
                expected=str(params.get("expected", "RAW")),
            )

        if rule_type == ValidationRuleType.CASEWORK_STATE:
            from faceaudit.validation_rules.casework_state_rule import (
                CaseworkStateRule,
            )
            expected_raw = params.get("expected", False)
            return CaseworkStateRule(
                rule_id=rule_id,
                scores_resource_id=ValidationRuleFactory._req_str(
                    params, "scores_resource_id", rule_type
                ),
                expected=bool(expected_raw),
            )

        if rule_type == ValidationRuleType.SCORE_TYPE_MATCH:
            from faceaudit.validation_rules.score_type_match_rule import (
                ScoreTypeMatchRule,
            )
            return ScoreTypeMatchRule(
                rule_id=rule_id,
                reference_resource_id=ValidationRuleFactory._req_str(
                    params, "reference_resource_id", rule_type
                ),
                case_resource_id=ValidationRuleFactory._req_str(
                    params, "case_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.DATASETS_EMBEDDING_COMPAT:
            from faceaudit.validation_rules.datasets_embedding_compat_rule import (
                DatasetsEmbeddingCompatRule,
            )
            return DatasetsEmbeddingCompatRule(
                rule_id=rule_id,
                dataset_resource_ids=ValidationRuleFactory._req_list_str(
                    params, "dataset_resource_ids", rule_type
                ),
                embedding_column=params.get("embedding_column"),
            )

        if rule_type == ValidationRuleType.QUALITY_METRIC_AVAILABLE:
            from faceaudit.validation_rules.quality_metric_available_rule import (
                QualityMetricAvailableRule,
            )
            return QualityMetricAvailableRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                quality_metric=ValidationRuleFactory._req_str(
                    params, "quality_metric", rule_type
                ),
                min_coverage=float(params.get("min_coverage", 0.5)),
            )

        if rule_type == ValidationRuleType.HAS_POSE_METADATA:
            from faceaudit.validation_rules.has_pose_metadata_rule import (
                HasPoseMetadataRule,
            )
            return HasPoseMetadataRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.MAX_SAMPLES_WARNING:
            from faceaudit.validation_rules.max_samples_warning_rule import (
                MaxSamplesWarningRule,
            )
            return MaxSamplesWarningRule(
                rule_id=rule_id,
                dataset_resource_id=ValidationRuleFactory._req_str(
                    params, "dataset_resource_id", rule_type
                ),
                max_samples=int(params.get("max_samples", 100_000)),
            )

        if rule_type == ValidationRuleType.SCORE_VECTOR_PAIR_COMPATIBILITY:
            from faceaudit.validation_rules.score_vector_pair_compatibility_rule import (
                ScoreVectorPairCompatibilityRule,
            )
            # Aceita o formato canônico (lista) ou o legado (par A/B) pra
            # compat com configs JSON salvos antes da generalização N-input.
            ids_param = params.get("input_resource_ids")
            if isinstance(ids_param, (list, tuple)) and ids_param:
                input_ids = [str(x) for x in ids_param]
            else:
                input_ids = [
                    ValidationRuleFactory._req_str(
                        params, "input_a_resource_id", rule_type
                    ),
                    ValidationRuleFactory._req_str(
                        params, "input_b_resource_id", rule_type
                    ),
                ]
                extras = params.get("extra_input_resource_ids")
                if isinstance(extras, (list, tuple)):
                    input_ids.extend(str(x) for x in extras)
            return ScoreVectorPairCompatibilityRule(
                rule_id=rule_id,
                input_resource_ids=input_ids,
                max_examples=int(params.get("max_examples", 5)),
            )

        if rule_type == ValidationRuleType.SCORE_VECTOR_ORDER_CONSISTENCY:
            from faceaudit.validation_rules.score_vector_order_consistency_rule import (
                ScoreVectorOrderConsistencyRule,
            )
            return ScoreVectorOrderConsistencyRule(
                rule_id=rule_id,
                scores_resource_id=ValidationRuleFactory._req_str(
                    params, "scores_resource_id", rule_type
                ),
                calibrator_resource_id=ValidationRuleFactory._req_str(
                    params, "calibrator_resource_id", rule_type
                ),
            )

        if rule_type == ValidationRuleType.CONDITIONAL_MONOTONICITY:
            from faceaudit.validation_rules.conditional_monotonicity_rule import (
                ConditionalMonotonicityRule,
            )
            return ConditionalMonotonicityRule(
                rule_id=rule_id,
                calibrator_resource_id=ValidationRuleFactory._req_str(
                    params, "calibrator_resource_id", rule_type
                ),
                numerical_epsilon=float(params.get("numerical_epsilon", 1e-12)),
                operational_min_slope=(
                    float(params["operational_min_slope"])
                    if params.get("operational_min_slope") is not None else None
                ),
                box_quantile_low=float(params.get("box_quantile_low", 0.05)),
                box_quantile_high=float(params.get("box_quantile_high", 0.95)),
                reportable_box=ValidationRuleFactory._parse_reportable_box(
                    params.get("reportable_box"), rule_type
                ),
                dataset_resource_id_override=params.get("dataset_resource_id_override"),
            )

        raise ValueError(
            f"ValidationRuleFactory: tipo de regra não suportado: {rule_type}."
        )

    @staticmethod
    def _make_rule_id(
        rule_type: ValidationRuleType, instruction_id: Optional[str]
    ) -> str:
        suffix = rule_type.value.lower()
        if instruction_id:
            return f"{instruction_id}_{suffix}"
        return suffix

    @staticmethod
    def _req_str(params: dict, key: str, rule_type: ValidationRuleType) -> str:
        val = params.get(key)
        if not val or not isinstance(val, str):
            raise ValueError(
                f"Regra {rule_type.value}: parâmetro obrigatório '{key}' "
                "ausente ou inválido (esperado string não-vazia)."
            )
        return val

    @staticmethod
    def _parse_reportable_box(
        raw: Optional[dict], rule_type: ValidationRuleType
    ) -> Optional[dict]:
        """Parser tolerante para reportable_box (V2).

        Aceita None ou dict com chaves entre {"r_probe", "r_reference",
        "r_single"} mapeadas para (lo, hi) numéricos. Levanta ValueError
        em estrutura inválida — gate operacional não pode silenciar
        erro de config.
        """
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(
                f"Regra {rule_type.value}: 'reportable_box' deve ser dict "
                f"(recebeu {type(raw).__name__})."
            )
        # Chaves de raridade (LR_ARCH_VEC) + score bruto (piecewise V1.5)
        allowed = {"r_probe", "r_reference", "r_single", "s", "base_score"}
        parsed: dict = {}
        for key, val in raw.items():
            if key not in allowed:
                raise ValueError(
                    f"Regra {rule_type.value}: chave inválida em "
                    f"reportable_box: '{key}'. Permitidas: {sorted(allowed)}."
                )
            if not isinstance(val, (list, tuple)) or len(val) != 2:
                raise ValueError(
                    f"Regra {rule_type.value}: reportable_box['{key}'] deve "
                    "ser par (lo, hi)."
                )
            try:
                lo = float(val[0])
                hi = float(val[1])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Regra {rule_type.value}: reportable_box['{key}'] "
                    f"não numérico: {e}"
                )
            if lo > hi:
                raise ValueError(
                    f"Regra {rule_type.value}: reportable_box['{key}'] "
                    f"inválido: lo={lo} > hi={hi}."
                )
            parsed[key] = (lo, hi)
        return parsed or None

    @staticmethod
    def _req_list_str(
        params: dict, key: str, rule_type: ValidationRuleType
    ) -> list:
        val = params.get(key)
        if not isinstance(val, (list, tuple)) or not all(isinstance(x, str) for x in val):
            raise ValueError(
                f"Regra {rule_type.value}: parâmetro obrigatório '{key}' "
                "ausente ou inválido (esperado lista de strings)."
            )
        return list(val)
