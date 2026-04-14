# faceaudit/validation_rules/fisher_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class FisherRule(ValidationRuleBase):
    """
    Verifica a Regra de Fisher para viabilidade do treinamento PLDA.

    O PLDA (e qualquer análise discriminante) exige que o número de identidades
    no dataset de treino seja estritamente maior que a dimensão dos embeddings.
    Caso contrário, a matriz de dispersão entre classes é singular e o treino
    falhará numericamente.

    Regra: n_identidades > embedding_dim
    Teto Fisher: max_dim_suportada = n_identidades - 1

    Também emite WARNING se n_identidades <= embedding_dim * 1.2 (margem estreita).

    Parâmetros configuráveis:
        dataset_resource_id: recurso EmbeddingDataset de treino do PLDA
        safety_margin_ratio: emite WARNING se n_ids < embedding_dim * (1 + margem)
                             (default: 0.2 = 20% acima do mínimo)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        safety_margin_ratio: float = 0.2,
        name: str = "Fisher Rule (PLDA Geometric Feasibility)",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_resource_id}' tem identidades suficientes "
                "para o espaço dimensional dos embeddings (Regra de Fisher para PLDA)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._safety_margin_ratio = safety_margin_ratio

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

        n_ids = stats.n_ids
        emb_dim = stats.embedding_dim

        if emb_dim <= 0:
            result.add_error(
                f"[{self.rule_id}] Não foi possível determinar a dimensão dos embeddings "
                f"em '{self._dataset_id}' (embedding_dim={emb_dim})."
            )
            return result

        if n_ids < 2:
            result.add_error(
                f"[{self.rule_id}] O dataset '{self._dataset_id}' possui apenas {n_ids} "
                "identidade(s). O PLDA exige no mínimo 2."
            )
            return result

        fisher_ceiling = n_ids - 1  # máximo de dimensões suportado

        if emb_dim >= n_ids:
            result.add_error(
                f"[{self.rule_id}] Singularidade de Fisher: '{self._dataset_id}' possui "
                f"{n_ids} identidade(s), suportando no máximo {fisher_ceiling} dimensões. "
                f"Os embeddings têm {emb_dim}D. "
                "Solução: aumente o número de identidades no dataset ou reduza a dimensão "
                "via LDA/PCA no pipeline de transformação."
            )
            return result

        # Warning: margem estreita
        safe_threshold = int(emb_dim * (1 + self._safety_margin_ratio))
        if n_ids <= safe_threshold:
            result.add_warning(
                f"[{self.rule_id}] Margem de Fisher estreita: {n_ids} identidade(s) para "
                f"{emb_dim}D (mínimo absoluto: {emb_dim + 1}, recomendado: >{safe_threshold}). "
                "O modelo pode ser instável com datasets pequenos."
            )
        else:
            result.mark_ok()

        return result
