# faceaudit/validation_rules/score_type_match_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class ScoreTypeMatchRule(ValidationRuleBase):
    """
    Verifica que dois score datasets têm o mesmo tipo (RAW ou CALIBRATED).

    Aplicável a CASEWORK_VISUALIZATION onde reference e case precisam usar
    a mesma escala (comparar distribuição RAW de referência com log-LR do
    caso, ou vice-versa, é absurdo).

    Detecção via presença de `log_lr` em qualquer item (amostragem dos
    primeiros 100 pra performance).

    Parâmetros:
        reference_resource_id: id do score dataset de referência
        case_resource_id: id do score dataset do caso
    """

    def __init__(
        self,
        rule_id: str,
        reference_resource_id: str,
        case_resource_id: str,
        name: str = "Mesmo tipo de score (referência × caso)",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{reference_resource_id}' e "
                f"'{case_resource_id}' são do mesmo tipo (RAW ou CALIBRATED)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._reference_id = reference_resource_id
        self._case_id = case_resource_id

    @staticmethod
    def _detect_state(payload: SimilarityScoreDataset) -> str:
        SAMPLE = 100
        for s in payload.items[:SAMPLE]:
            md = s.metadata if isinstance(s.metadata, dict) else None
            if md is not None and md.get("log_lr") is not None:
                return "CALIBRATED"
        return "RAW"

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        for rid in (self._reference_id, self._case_id):
            if not context.has_resource(rid):
                result.add_error(
                    f"[{self.rule_id}] Recurso '{rid}' não encontrado no contexto."
                )
                return result

        ref = context.ensure_payload(self._reference_id)
        case = context.ensure_payload(self._case_id)
        ref_payload = ref.payload if ref is not None else None
        case_payload = case.payload if case is not None else None

        if ref_payload is None or case_payload is None:
            result.add_warning(
                f"[{self.rule_id}] Payload indisponível em um dos recursos. "
                "Pulando checagem de tipo."
            )
            return result

        if not isinstance(ref_payload, SimilarityScoreDataset) or not isinstance(
            case_payload, SimilarityScoreDataset
        ):
            result.add_error(
                f"[{self.rule_id}] Pelo menos um dos recursos não é um "
                "SimilarityScoreDataset."
            )
            return result

        ref_state = self._detect_state(ref_payload)
        case_state = self._detect_state(case_payload)

        if ref_state != case_state:
            result.add_error(
                f"[{self.rule_id}] Tipos de score divergem: "
                f"'{self._reference_id}' é {ref_state} e "
                f"'{self._case_id}' é {case_state}. Ambos devem ser do "
                "mesmo tipo (RAW ou CALIBRATED) pra que a sobreposição "
                "faça sentido."
            )
            return result

        result.mark_ok()
        return result
