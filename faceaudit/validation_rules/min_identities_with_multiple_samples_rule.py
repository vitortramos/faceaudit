# faceaudit/validation_rules/min_identities_with_multiple_samples_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class MinIdentitiesWithMultipleSamplesRule(ValidationRuleBase):
    """
    Verifica que o dataset tem identidades suficientes com 2+ amostras
    para treinar PLDA.

    Motivação: o `PLDASimilarityModel.fit` filtra agressivamente:
    - identity=None é removida
    - identidades com apenas 1 amostra (singletons) são descartadas
      porque a variância intra-classe não é estimável a partir de uma
      única observação

    Após esses filtros, o treino exige pelo menos 2 classes válidas para
    estimar B (between-class). Caso contrário, o fit explode em runtime
    com mensagem genérica.

    Esta regra inspeciona `DatasetStats.samples_per_identity` (populado
    pelo `DatasetStatsService`) e conta quantas identidades têm `count >= 2`.
    Falha CRITICAL se < `min_required`.

    Parâmetros:
        dataset_resource_id: id do recurso EmbeddingDataset/FaceDataset
        min_required: número mínimo absoluto de identidades não-singleton
                      (default 2 — mínimo matemático para PLDA)
        safety_threshold: emite WARNING se `count >= min_required` mas
                          < safety_threshold (default 10 — operacionalmente
                          frágil abaixo disso)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        min_required: int = 2,
        safety_threshold: int = 10,
        name: str = "Mínimo de identidades com 2+ amostras",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_resource_id}' tem pelo menos "
                f"{min_required} identidades com ≥ 2 amostras (mínimo para "
                "treinar PLDA — singletons são descartados no fit)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._min_required = int(min_required)
        self._safety_threshold = int(safety_threshold)

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        stats = self._get_or_compute_stats(self._dataset_id, context)

        if stats is None:
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._dataset_id}' não encontrado no contexto."
            )
            return result

        spi = getattr(stats, "samples_per_identity", None) or {}
        if not spi:
            result.add_warning(
                f"[{self.rule_id}] Não foi possível inspecionar "
                f"'samples_per_identity' em '{self._dataset_id}'. Pulando "
                "checagem de singletons."
            )
            return result

        n_multi = sum(1 for count in spi.values() if count >= 2)
        n_singleton = sum(1 for count in spi.values() if count == 1)
        total_ids = len(spi)

        if n_multi < self._min_required:
            result.add_error(
                f"[{self.rule_id}] Dados insuficientes para treinar PLDA: "
                f"'{self._dataset_id}' tem {n_multi} identidade(s) com ≥ 2 "
                f"amostras (mínimo {self._min_required}). "
                f"Total: {total_ids} identidade(s); singletons: {n_singleton} "
                f"(serão descartadas no fit). Solução: adicionar mais "
                "amostras às identidades ou ampliar o dataset."
            )
            return result

        if n_multi < self._safety_threshold:
            result.add_warning(
                f"[{self.rule_id}] '{self._dataset_id}' tem apenas {n_multi} "
                f"identidade(s) com ≥ 2 amostras (recomendado: ≥ "
                f"{self._safety_threshold}). O treino é viável mas estatisticamente "
                f"frágil. Singletons descartados: {n_singleton}."
            )
        else:
            result.mark_ok()

        return result
