# faceaudit/validation_rules/conditional_monotonicity_rule.py

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.monotonicity_derivative_provider_protocol import (
    MonotonicityRegionResult,
)
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.lr_architecture_derivative_provider import (
    LRArchitectureDerivativeProvider,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase

RULE_VERSION = "1.0"


class ConditionalMonotonicityRule(ValidationRuleBase):
    """
    Verifica que `∂ log_LR / ∂ s > 0` em todos os pares observados de
    `D_cal` (`observed_rows`) e nos cantos do retângulo central
    `[P_low, P_high]` de raridades observadas (`central_box`).

    Gate forense: aumentar a similaridade nunca pode reduzir o log-LR
    condicionado às raridades observadas. Auditoria executada após o
    refit final do calibrador; resultados persistidos no metadata.

    V1: cobre apenas arquiteturas declarativas de
    `LR_ARCHITECTURE_VECTORIZATION` via
    `LRArchitectureDerivativeProvider`. V1.5 estende a piecewise legado.

    Parâmetros configuráveis:
        calibrator_resource_id: id do recurso do calibrador a auditar
            (proveniência do dataset auditado vem do metadata do calibrador)
        numerical_epsilon: tolerância para ruído de ponto flutuante
            (default 1e-12)
        operational_min_slope: margem substantiva opcional para WARNING
            de "curva quase plana" (default None — não verifica)
        box_quantile_low: limite inferior da `central_box` (default 0.05)
        box_quantile_high: limite superior (default 0.95)
        dataset_resource_id_override: override explícito do dataset auditado
            (usar apenas em testes/depuração — gera warning persistido)
    """

    def __init__(
        self,
        rule_id: str,
        calibrator_resource_id: str,
        numerical_epsilon: float = 1e-12,
        operational_min_slope: Optional[float] = None,
        box_quantile_low: float = 0.05,
        box_quantile_high: float = 0.95,
        dataset_resource_id_override: Optional[str] = None,
        name: str = "Monotonicidade condicional",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                "Verifica ∂log_LR/∂s > 0 sobre pares observados e cantos da "
                "caixa central de raridades; gate estrutural sobre o calibrador "
                f"'{calibrator_resource_id}'."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._calibrator_resource_id = calibrator_resource_id
        self._numerical_epsilon = numerical_epsilon
        self._operational_min_slope = operational_min_slope
        self._box_quantile_low = box_quantile_low
        self._box_quantile_high = box_quantile_high
        self._dataset_resource_id_override = dataset_resource_id_override

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        # 1. Resolver calibrador
        if not context.has_resource(self._calibrator_resource_id):
            result.add_error(
                f"[{self.rule_id}] Calibrador '{self._calibrator_resource_id}' "
                "não encontrado no contexto."
            )
            return result
        calibrator_resource = context.ensure_payload(self._calibrator_resource_id)
        calibrator = calibrator_resource.payload if calibrator_resource else None
        if calibrator is None or not hasattr(calibrator, "metadata"):
            result.add_error(
                f"[{self.rule_id}] Calibrador '{self._calibrator_resource_id}' "
                "sem metadata persistido — auditoria de monotonicidade requer "
                "modelo treinado com V2.14 (LR_ARCHITECTURE_VECTORIZATION)."
            )
            return result
        calibrator_metadata = dict(calibrator.metadata or {})

        # 2. Verificar presença de architecture_state (V1: só LR_ARCH_VEC)
        if "architecture_state" not in calibrator_metadata:
            result.add_error(
                f"[{self.rule_id}] Calibrador sem 'architecture_state' — V1 "
                "da regra cobre apenas arquiteturas de "
                "LR_ARCHITECTURE_VECTORIZATION. Piecewise legado entra em V1.5."
            )
            return result

        # 3. Resolver raw_input_resource_id (proveniência do calibrador)
        if self._dataset_resource_id_override is not None:
            raw_input_resource_id = self._dataset_resource_id_override
            result.add_warning(
                f"[{self.rule_id}] dataset_resource_id_override em uso — "
                f"auditoria sobre '{raw_input_resource_id}' em vez do "
                "raw_input do calibrador. Use APENAS em testes/depuração."
            )
        else:
            # V2.14 não persiste training_raw_input_resource_id no calibrador
            # diretamente — proveniência completa exige fingerprint matching
            # (V1.5). V1: requer override explícito quando metadata ausente.
            raw_input_resource_id = calibrator_metadata.get(
                "training_raw_input_resource_id"
            )
            if not raw_input_resource_id:
                result.add_error(
                    f"[{self.rule_id}] Calibrador sem "
                    "'training_raw_input_resource_id' persistido — V1 "
                    "requer 'dataset_resource_id_override' explícito nesse "
                    "caso. V1.5 implementará resolução completa via fingerprint."
                )
                return result

        # 4. Verificar fingerprint consistency (best-effort em V1)
        expected_fp = calibrator_metadata.get("vectorization_contract_fingerprint")
        if not context.has_resource(raw_input_resource_id):
            result.add_error(
                f"[{self.rule_id}] Raw_input '{raw_input_resource_id}' "
                "não encontrado no contexto."
            )
            return result
        raw_input_resource = context.ensure_payload(raw_input_resource_id)
        raw_input_payload = raw_input_resource.payload if raw_input_resource else None
        if raw_input_payload is None:
            result.add_error(
                f"[{self.rule_id}] Raw_input '{raw_input_resource_id}' "
                "payload vazio."
            )
            return result

        # Contract fingerprint pode estar em vários lugares — V1 best-effort:
        #   raw_input_payload.vectorization_contract_fingerprint (campo direto)
        #   raw_input_payload.schema.fingerprint() (V2.14: do schema)
        raw_fp = getattr(raw_input_payload, "vectorization_contract_fingerprint", None)
        if raw_fp is None and hasattr(raw_input_payload, "schema"):
            schema = raw_input_payload.schema
            if hasattr(schema, "fingerprint"):
                try:
                    raw_fp = schema.fingerprint()
                except Exception:
                    raw_fp = None
        if expected_fp and raw_fp and expected_fp != raw_fp:
            # Em V1, divergência de fingerprint é WARNING (não CRITICAL)
            # porque ainda há margem de inconsistência entre nomes de
            # fingerprint do calibrador vs raw_input em V2.14.
            result.add_warning(
                f"[{self.rule_id}] Fingerprint divergente (V1 best-effort): "
                f"calibrador={expected_fp[:12]}... raw_input={raw_fp[:12]}... "
                "V1.5 implementará validação estrita."
            )

        # 5. Reconstituir spec via state
        from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
        from faceaudit.models.lr.lr_architecture_state import LRArchitectureStateFactory
        try:
            state = LRArchitectureStateFactory.from_dict(
                calibrator_metadata["architecture_state"]
            )
        except Exception as e:
            result.add_error(
                f"[{self.rule_id}] Falha ao reconstruir architecture_state: "
                f"{type(e).__name__}: {e}"
            )
            return result
        spec = LR_ARCHITECTURE_SPECS.get(state.architecture)
        if spec is None:
            result.add_error(
                f"[{self.rule_id}] Arquitetura {state.architecture} sem spec "
                "registrado em LR_ARCHITECTURE_SPECS."
            )
            return result

        # 6. Provider + avaliação
        provider = LRArchitectureDerivativeProvider(spec)
        try:
            regions = provider.evaluate_regions(
                calibrator_metadata=calibrator_metadata,
                raw_input_payload=raw_input_payload,
                box_quantile_low=self._box_quantile_low,
                box_quantile_high=self._box_quantile_high,
            )
        except Exception as e:
            result.add_error(
                f"[{self.rule_id}] Falha na avaliação de derivadas: "
                f"{type(e).__name__}: {e}"
            )
            return result

        # 7. Aplicar política de status
        audit_metadata = self._build_audit_metadata(
            regions=regions, spec_id=state.architecture.value,
            raw_input_resource_id=raw_input_resource_id,
            expected_fp=expected_fp,
        )

        min_obs, worst_obs_region = self._global_min_observed(regions)
        min_box, worst_box_region = self._global_min_central_box(regions)

        if min_obs is not None and min_obs <= self._numerical_epsilon:
            audit_metadata["audit_status"] = "CRITICAL"
            result.add_error(
                f"[{self.rule_id}] Monotonicidade VIOLADA em observed_rows: "
                f"min_slope = {min_obs:.6g} (região: {worst_obs_region}). "
                "Aumentar o cosseno reduz log-LR em pares reais de D_cal."
            )
        elif min_box is not None and min_box <= self._numerical_epsilon:
            audit_metadata["audit_status"] = "WARNING"
            result.add_warning(
                f"[{self.rule_id}] Slope positivo em observed_rows mas "
                f"negativo em canto da central_box: min={min_box:.6g} "
                f"(região: {worst_box_region}). Investigar suporte do canto."
            )
        elif (self._operational_min_slope is not None
              and min_obs is not None
              and min_obs < self._operational_min_slope):
            audit_metadata["audit_status"] = "WARNING"
            result.add_warning(
                f"[{self.rule_id}] Slope mínimo observado {min_obs:.6g} < "
                f"operational_min_slope {self._operational_min_slope:.6g} "
                "(curva muito plana — pode indicar problema de identificação)."
            )
        else:
            audit_metadata["audit_status"] = "OK"

        # Diagnóstico adicional: features degeneradas
        all_zero = []
        for r in regions:
            all_zero.extend(r.zero_variance_terms or [])
        if all_zero:
            result.add_warning(
                f"[{self.rule_id}] Termos com variância zero em alguma região: "
                f"{all_zero}"
            )

        # 8. Persistir metadata no calibrador
        try:
            self._persist_audit_metadata(calibrator, audit_metadata)
        except Exception as e:
            result.add_warning(
                f"[{self.rule_id}] Falha ao persistir audit metadata: "
                f"{type(e).__name__}: {e}"
            )

        return result

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _global_min_observed(regions: List[MonotonicityRegionResult]) -> tuple:
        best: Optional[float] = None
        worst_region: Optional[str] = None
        for r in regions:
            m = r.min_slope_observed()
            if m is not None and (best is None or m < best):
                best = m
                worst_region = r.region_id
        return best, worst_region

    @staticmethod
    def _global_min_central_box(regions: List[MonotonicityRegionResult]) -> tuple:
        best: Optional[float] = None
        worst_region: Optional[str] = None
        for r in regions:
            m = r.min_slope_central_box()
            if m is not None and (best is None or m < best):
                best = m
                worst_region = r.region_id
        return best, worst_region

    def _build_audit_metadata(
        self,
        *,
        regions: List[MonotonicityRegionResult],
        spec_id: str,
        raw_input_resource_id: str,
        expected_fp: Optional[str],
    ) -> dict:
        regions_summary = []
        for r in regions:
            regions_summary.append({
                "region_id": r.region_id,
                "region_description": r.region_description,
                "n_observed": r.n_observed_rows_in_region,
                "min_slope_observed": r.min_slope_observed(),
                "max_slope_observed": r.max_slope_observed(),
                "min_slope_central_box": r.min_slope_central_box(),
                "slope_at_central_box_corners": r.slope_at_central_box_corners,
                "corner_labels": r.corner_labels,
                "worst_observed_row_index": r.worst_observed_row_index,
                "worst_observed_row_features": r.worst_observed_row_features,
                "worst_central_corner_label": r.worst_central_corner_label,
                "worst_central_corner_values": r.worst_central_corner_values,
                "zero_variance_terms": r.zero_variance_terms,
                "near_zero_variance_terms": r.near_zero_variance_terms,
            })

        return {
            "rule_version": RULE_VERSION,
            "architecture": spec_id,
            "training_raw_input_resource_id": raw_input_resource_id,
            "vectorization_contract_fingerprint": expected_fp,
            "audit_domain_modes": ["observed_rows", "central_box"],
            "box_quantile_low": self._box_quantile_low,
            "box_quantile_high": self._box_quantile_high,
            "numerical_epsilon": self._numerical_epsilon,
            "operational_min_slope": self._operational_min_slope,
            "regions": regions_summary,
        }

    def _persist_audit_metadata(self, calibrator: Any, audit_metadata: dict) -> None:
        """Persiste audit no metadata do calibrador (mutação in-place)."""
        if not hasattr(calibrator, "metadata") or calibrator.metadata is None:
            calibrator.metadata = {}
        calibrator.metadata["monotonicity_audit"] = audit_metadata
