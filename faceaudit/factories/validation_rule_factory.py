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
