# faceaudit/validation_rules/datasets_embedding_compat_rule.py

from __future__ import annotations

from typing import Any, List, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class DatasetsEmbeddingCompatRule(ValidationRuleBase):
    """
    Verifica que múltiplos datasets têm a mesma coluna de embedding E a
    mesma dimensão.

    Aplicável a DATASET_MERGE — concatenar datasets com embeddings de
    schemas diferentes (coluna nomeada `sepael` em um e `adaface` em
    outro, ou mesmo 256D vs 512D) produz dataset inutilizável.

    Quando `embedding_column` é informado explicitamente, checa que TODOS
    os datasets oferecem aquela coluna específica. Quando omitido, valida
    apenas que TODOS compartilham pelo menos UMA coluna de embedding em
    comum e que a dim é a mesma entre eles.

    Parâmetros:
        dataset_resource_ids: lista de ids de datasets a comparar (≥ 2)
        embedding_column: coluna de embedding a verificar (opcional)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_ids: List[str],
        embedding_column: Optional[str] = None,
        name: str = "Compatibilidade de embedding entre datasets",
        description: Optional[str] = None,
    ) -> None:
        if not dataset_resource_ids or len(dataset_resource_ids) < 2:
            raise ValueError(
                "DatasetsEmbeddingCompatRule exige ≥ 2 dataset_resource_ids."
            )
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que {len(dataset_resource_ids)} datasets compartilham "
                "a mesma coluna de embedding e a mesma dimensão."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_ids = list(dataset_resource_ids)
        self._embedding_column = embedding_column

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        per_dataset = {}
        for rid in self._dataset_ids:
            stats = self._get_or_compute_stats(rid, context)
            if stats is None:
                result.add_error(
                    f"[{self.rule_id}] Dataset '{rid}' não encontrado no contexto."
                )
                return result
            cols = set(getattr(stats, "embedding_columns", None) or set())
            dim = int(getattr(stats, "embedding_dim", 0) or 0)
            per_dataset[rid] = (cols, dim)

        if self._embedding_column:
            target_col = self._embedding_column
            missing = [rid for rid, (cols, _) in per_dataset.items() if target_col not in cols]
            if missing:
                result.add_error(
                    f"[{self.rule_id}] Coluna '{target_col}' ausente em: "
                    f"{', '.join(missing)}."
                )
                return result
        else:
            common = None
            for _rid, (cols, _dim) in per_dataset.items():
                common = cols if common is None else (common & cols)
            if not common:
                result.add_error(
                    f"[{self.rule_id}] Datasets não compartilham nenhuma coluna "
                    f"de embedding em comum: "
                    f"{ {rid: sorted(cols) for rid, (cols, _) in per_dataset.items()} }"
                )
                return result

        dims = {dim for _cols, dim in per_dataset.values()}
        if len(dims) > 1:
            result.add_error(
                f"[{self.rule_id}] Dimensões divergentes entre datasets: "
                f"{ {rid: dim for rid, (_cols, dim) in per_dataset.items()} }"
            )
            return result

        result.mark_ok()
        return result
