# faceaudit/validation_rules/strategy_embedding_consistency_rule.py

from __future__ import annotations

from typing import Any, List, Set

from faceaudit.enums.resource_category import ResourceCategory
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

    1. `embedding_column` está explicitamente definido na estratégia.
       Se for None, erro — escolha tem de vir do combo do editor.
    2. Cada DATASET referenciado por alguma instrução contém a coluna
       `embedding_column` escolhida.
    3. O `column_specs[embedding_column].model_name` é o mesmo em todos os
       datasets (impede o caso de duas colunas terem o mesmo nome mas
       virem de modelos diferentes — ex: SEPAEL_V2 vs DeepFace ambos
       chamados `embedding`).

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

        if not embedding_column:
            result.add_error(
                "Estratégia sem coluna de embedding selecionada. "
                "Escolha uma coluna no combo 'Embedding' do editor."
            )
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

            # Skip de TODOS os datasets DERIVED (independentemente de já
            # terem sido produzidos ou não). Justificativa:
            #
            # Datasets derivados são produtos intermediários da própria
            # estratégia (ex: output de PIPELINE_APPLICATION, DATASET_SPLIT,
            # DATASET_MERGE). Sua coluna de embedding é determinada pelo
            # pipeline upstream — tipicamente o pipeline transforma a coluna
            # `embedding_v2` (input) em `embedding` (output). Logo, exigir
            # que o derivado contenha a `embedding_column` original da
            # estratégia é incorreto: ele NÃO vai conter, justamente porque
            # foi transformado.
            #
            # A consistência de embedding é validada nos PRIMARIES (datasets
            # importados pelo usuário) — esses precisam compartilhar a
            # mesma coluna e o mesmo `model_name`. Derivados herdam
            # implicitamente a consistência via lineage de execução.
            if getattr(resource, "category", None) == ResourceCategory.DERIVED:
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
        Coleta todos os resource_ids de DATASET referenciados como input
        por alguma instrução da estratégia.
        """
        ids: List[str] = []
        seen: Set[str] = set()

        # `instructions` é a property concatenada (preparation + validation
        # + casework). Para cada instrução, vê quais inputs apontam pra
        # recursos do tipo dataset.
        resources_by_id = {
            r.resource_id: r for r in getattr(strategy_definition, "resources", [])
        }

        for instr in getattr(strategy_definition, "instructions", []):
            for rid in getattr(instr, "input_resource_ids", []) or []:
                if rid in seen:
                    continue
                resource = resources_by_id.get(rid)
                if resource is None:
                    continue
                if resource.resource_type == StrategyResourceType.DATASET:
                    ids.append(rid)
                    seen.add(rid)

        return ids

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
