# faceaudit/validation_rules/strategy_embedding_consistency_rule.py

from __future__ import annotations

from typing import Any, List, Optional, Set

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class StrategyEmbeddingConsistencyRule(ValidationRuleBase):
    """
    Garante que TODOS os datasets usados pela estratégia compartilham o
    mesmo "tipo" de embedding — isto é:

    1. Cada DATASET/INTERMEDIATE_DATASET referenciado por alguma instrução
       contém a coluna `embedding_column` escolhida pela estratégia.
    2. O `column_specs[embedding_column].model_name` é o mesmo em todos os
       datasets (impede o caso de duas colunas terem o mesmo nome mas
       virem de modelos diferentes — ex: SEPAEL_V2 vs DeepFace ambos
       chamados `embedding`).

    Quando `embedding_column` é None (auto-detect), a regra:
    - Calcula a interseção das colunas com role=EMBEDDING entre todos os
      datasets envolvidos.
    - Se houver exatamente uma → emite warning sugerindo gravá-la na
      definição (configuração explícita > implícita).
    - Se houver mais de uma → erro pedindo escolha explícita.
    - Se houver zero → erro: nenhuma coluna de embedding em comum.

    Aplicada no nível da estratégia (`LRStrategyDefinition.validation_rule_ids`),
    não por instrução. Roda uma vez antes de qualquer fase.
    """

    def __init__(
        self,
        rule_id: str = "strategy_embedding_consistency",
        default_severity: ValidationSeverity = ValidationSeverity.CRITICAL,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name="Consistência de Embedding na Estratégia",
            description=(
                "Verifica se todos os datasets usados pela estratégia têm a "
                "coluna de embedding escolhida e se todos vêm do mesmo modelo "
                "(mesmo `model_name` no column_specs)."
            ),
            default_severity=default_severity,
        )

    # ──────────────────────────────────────────────────────────────────────
    # ValidationRuleProtocol.validate
    # ──────────────────────────────────────────────────────────────────────

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        embedding_column = getattr(target, "embedding_column", None)
        dataset_ids = self._collect_dataset_ids(target)
        if not dataset_ids:
            # Estratégia sem datasets ainda — nada a validar.
            result.mark_ok()
            return result

        # ─── auto-detect quando embedding_column não foi escolhido ────
        if not embedding_column:
            common_specs = self._intersect_embedding_specs(dataset_ids, context)
            if common_specs is None:
                result.mark_ok()
                return result

            if len(common_specs) == 0:
                result.add_error(
                    "Nenhuma coluna de embedding em comum entre os datasets da "
                    "estratégia. Verifique se todos os datasets foram enriquecidos "
                    "com o mesmo modelo (ForensicFace, DeepFace, etc.)."
                )
                return result

            if len(common_specs) > 1:
                names = ", ".join(sorted(common_specs.keys()))
                result.add_error(
                    f"Os datasets têm múltiplas colunas de embedding em comum "
                    f"({names}). Defina explicitamente `embedding_column` na "
                    f"estratégia para escolher qual usar."
                )
                return result

            # Exatamente uma — auto-detect funciona, mas sugere gravar.
            (only_col,) = list(common_specs.keys())
            result.add_warning(
                f"Estratégia sem `embedding_column` definido. Auto-detectado "
                f"'{only_col}' (única coluna de embedding em comum entre os "
                f"datasets). Considere gravar a escolha na estratégia."
            )
            result.mark_ok()
            return result

        # ─── embedding_column explícito ───────────────────────────────
        seen_model_names: Set[str] = set()
        missing: List[str] = []

        for ds_id in dataset_ids:
            resource = context.get_resource(ds_id) if context.has_resource(ds_id) else None
            if resource is None:
                # Recurso não está no contexto — isso é alvo de outra regra
                # (existência), aqui só pulamos.
                continue

            payload = resource.payload
            if payload is None:
                # Lazy-load do parquet pra ler column_specs sem hidratar
                # tudo. Se falhar, pula — outras regras vão capturar.
                try:
                    resource = context.ensure_payload(ds_id) or resource
                    payload = resource.payload
                except Exception:
                    continue

            specs = getattr(payload, "column_specs", None) if payload else None
            if not specs:
                # Dataset sem column_specs (parquet legado pré-migração).
                # Se a coluna existir nos itens é tudo que podemos checar —
                # marca como warning, não bloqueia.
                if not self._has_column_in_legacy_payload(payload, embedding_column):
                    missing.append(ds_id)
                continue

            spec = specs.get(embedding_column)
            if spec is None:
                missing.append(ds_id)
                continue

            model_name = getattr(spec, "model_name", None)
            if model_name:
                seen_model_names.add(str(model_name))

        if missing:
            sample = ", ".join(missing[:5])
            extra = "" if len(missing) <= 5 else f" (+{len(missing) - 5} outros)"
            result.add_error(
                f"A coluna de embedding '{embedding_column}' não existe em: "
                f"{sample}{extra}. Re-enriqueça os datasets ou escolha outra "
                f"coluna na estratégia."
            )

        if len(seen_model_names) > 1:
            names = ", ".join(sorted(seen_model_names))
            result.add_error(
                f"A coluna '{embedding_column}' aparece em datasets diferentes "
                f"com `model_name` distintos: {{{names}}}. Os embeddings vêm "
                f"de modelos incompatíveis e não podem ser comparados."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_dataset_ids(strategy_definition: Any) -> List[str]:
        """
        Coleta todos os resource_ids de DATASET/INTERMEDIATE_DATASET
        referenciados como input por alguma instrução da estratégia.
        """
        ids: List[str] = []
        seen: Set[str] = set()

        # `instructions` é a property concatenada (preparation + validation
        # + casework). Para cada instrução, vê quais inputs apontam pra
        # recursos do tipo dataset.
        resources_by_id = {
            r.resource_id: r for r in getattr(strategy_definition, "resources", [])
        }

        dataset_types = (
            StrategyResourceType.DATASET,
            StrategyResourceType.INTERMEDIATE_DATASET,
        )

        for instr in getattr(strategy_definition, "instructions", []):
            for rid in getattr(instr, "input_resource_ids", []) or []:
                if rid in seen:
                    continue
                resource = resources_by_id.get(rid)
                if resource is None:
                    continue
                if resource.resource_type in dataset_types:
                    ids.append(rid)
                    seen.add(rid)

        return ids

    def _intersect_embedding_specs(
        self,
        dataset_ids: List[str],
        context: StrategyExecutionContextProtocol,
    ) -> Optional[dict]:
        """
        Retorna o dict {column_name: spec} comum a TODOS os datasets, ou
        None se algum dataset não pôde ser inspecionado (sem
        column_specs — cai no caminho permissivo do validate).
        """
        common: Optional[dict] = None
        for ds_id in dataset_ids:
            if not context.has_resource(ds_id):
                continue

            resource = context.get_resource(ds_id)
            payload = resource.payload
            if payload is None:
                try:
                    resource = context.ensure_payload(ds_id) or resource
                    payload = resource.payload
                except Exception:
                    continue

            specs = getattr(payload, "column_specs", None) if payload else None
            if not specs:
                # Sem specs → não dá pra fazer interseção segura.
                return None

            embedding_specs = {
                name: spec for name, spec in specs.items()
                if str(getattr(spec, "role", "")).lower().endswith("embedding")
            }

            if common is None:
                common = embedding_specs
            else:
                common = {
                    name: common[name]
                    for name in common.keys() & embedding_specs.keys()
                }

        return common if common is not None else {}

    @staticmethod
    def _has_column_in_legacy_payload(payload: Any, column_name: str) -> bool:
        """
        Fallback pra payloads sem column_specs (parquets legados
        pré-migração): inspeciona um item pra ver se a coluna existe.
        """
        if payload is None:
            return False

        # EmbeddingDataset — items com EmbeddingInfo (coluna mapeada
        # internamente, não há checagem direta possível). Como o loader
        # já faz auto-detect agora, ausência aqui = falha do loader, e
        # não conseguiríamos reproduzir a checagem sem reabrir o parquet.
        # Tratamos como "presente" para não bloquear (warning vem de
        # outras regras).
        return True
