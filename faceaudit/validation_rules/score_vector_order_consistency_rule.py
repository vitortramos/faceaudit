# faceaudit/validation_rules/score_vector_order_consistency_rule.py

from __future__ import annotations

from typing import Any, List, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class ScoreVectorOrderConsistencyRule(ValidationRuleBase):
    """
    Em `SCORE_CALIBRATION` com calibrador multivariado, verifica que a
    procedência ordenada das colunas do score vector de entrada bate
    com a procedência usada no treino do calibrador.

    Motivação: a matriz N×D do score vector é construída na ordem que
    o usuário preenche A, B, extras na `SCORE_VECTORIZATION`. O
    calibrador (LogReg multivariado) aprende coeficientes posicionais.
    Se a ordem dos sistemas na validação for diferente da do treino,
    o calibrador aplica `β_cosseno × score_PLDA + β_PLDA × score_cosseno`
    — log-LRs sem sentido, EER/Cllr explodem, plots ficam ruidosos.

    Algoritmo: lê `score_vector_model_ids` do metadata do input
    (gravado pelo `ScoreVectorizationInstructionExecutor`) e do
    atributo `score_vector_model_ids` do modelo treinado (gravado pelo
    `CalibrationModelTrainingInstructionExecutor`). Compara como
    listas posicionais. Diferenças → CRITICAL.

    Comportamento defensivo: passa silenciosamente quando
    - input não é score vector;
    - modelo treinado não tem `score_vector_model_ids` (calibrador
      legado treinado antes da introdução do rastreio);
    - qualquer posição na lista é None (legacy mix).
    """

    def __init__(
        self,
        rule_id: str,
        scores_resource_id: str,
        calibrator_resource_id: str,
        name: str = "Ordem dos sistemas no vetor (SCORE_CALIBRATION)",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que a procedência ordenada das colunas em "
                f"'{scores_resource_id}' bate com a procedência usada no "
                f"treino de '{calibrator_resource_id}'."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._scores_id = scores_resource_id
        self._calibrator_id = calibrator_resource_id

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._scores_id):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._scores_id}' não "
                "encontrado no contexto."
            )
            return result
        if not context.has_resource(self._calibrator_id):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._calibrator_id}' não "
                "encontrado no contexto."
            )
            return result

        scores_payload = context.ensure_payload(self._scores_id)
        calibrator_payload = context.ensure_payload(self._calibrator_id)

        # Defesa silenciosa: regra só faz sentido pra calibrador
        # multivariado treinado em score vector.
        if not isinstance(scores_payload, SimilarityScoreDataset):
            return result.mark_ok()
        if not getattr(scores_payload, "is_score_vector", False):
            return result.mark_ok()

        scores_model_ids: Optional[List[Optional[str]]] = (
            (scores_payload.metadata or {}).get("score_vector_model_ids")
        )
        calibrator_model_ids: Optional[List[Optional[str]]] = getattr(
            calibrator_payload, "score_vector_model_ids", None
        )

        # Sem rastreio em algum dos lados (legacy) — passa silenciosamente.
        if not scores_model_ids or not calibrator_model_ids:
            return result.mark_ok()
        if any(m is None for m in scores_model_ids):
            return result.mark_ok()
        if any(m is None for m in calibrator_model_ids):
            return result.mark_ok()

        if len(scores_model_ids) != len(calibrator_model_ids):
            result.add_error(
                f"[{self.rule_id}] Número de colunas do vetor diverge: "
                f"calibrador foi treinado com {len(calibrator_model_ids)} "
                f"sistemas ({list(calibrator_model_ids)}), mas a entrada "
                f"tem {len(scores_model_ids)} sistemas "
                f"({list(scores_model_ids)})."
            )
            return result

        if list(scores_model_ids) != list(calibrator_model_ids):
            result.add_error(
                f"[{self.rule_id}] Ordem dos sistemas no vetor diverge "
                f"entre o treino do calibrador e a aplicação. "
                f"Treino: {list(calibrator_model_ids)}. "
                f"Entrada: {list(scores_model_ids)}. "
                f"Reordene os inputs A/B/extras da SCORE_VECTORIZATION "
                f"que alimenta '{self._scores_id}' pra casar com a do "
                f"treino — caso contrário os pesos do LogReg multivariado "
                f"ficam aplicados às colunas erradas."
            )
            return result

        return result.mark_ok()
