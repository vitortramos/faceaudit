# faceaudit/validation_rules/fused_pair_compatibility_rule.py

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set, Tuple

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class FusedPairCompatibilityRule(ValidationRuleBase):
    """
    Verifica que **N (≥2) SimilarityScoreDataset** compartilham
    EXATAMENTE o mesmo conjunto de pares
    `(probe.face_hash, reference.face_hash)`.

    Catch antecipado pra evitar erro tardio do
    `ScoreFusionInstructionExecutor._build_fused_dataset` (que levanta
    ValueError no meio da execução). Quando todos os sistemas processaram
    o mesmo conjunto de pares, fusão é segura; qualquer divergência
    pode indicar que o usuário cruzou splits diferentes ou usou
    PAIR_MINING em apenas um dos lados.

    Algoritmo: o primeiro input é o **anchor**; cada input subsequente
    é comparado contra o anchor. Diferenças mostradas com hashes truncados
    pra primeira ocorrência (até `max_examples` por divergência).

    Parâmetros:
        input_resource_ids: lista de IDs dos SimilarityScoreDataset
            (mínimo 2). O primeiro vira anchor das comparações.
        max_examples: máximo de hashes mostrados por par divergente
            (default 5).
    """

    def __init__(
        self,
        rule_id: str,
        input_resource_ids: Sequence[str],
        max_examples: int = 5,
        name: str = "Compatibilidade de pares (SCORE_FUSION)",
        description: Optional[str] = None,
    ) -> None:
        ids = list(input_resource_ids)
        if len(ids) < 2:
            raise ValueError(
                "FusedPairCompatibilityRule exige pelo menos 2 input_resource_ids."
            )
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que {len(ids)} datasets ({', '.join(ids)}) "
                "compartilham o mesmo conjunto de pares "
                "(probe.face_hash + reference.face_hash)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._ids: List[str] = ids
        self._max_examples = int(max_examples)

    @staticmethod
    def _pair_key(sample: SimilarityScoreSample) -> Tuple[str, str]:
        probe_hash = getattr(sample.probe, "face_hash", None) or ""
        ref_hash = getattr(sample.reference, "face_hash", None) or ""
        return (str(probe_hash), str(ref_hash))

    def _collect_keys(
        self, payload: SimilarityScoreDataset
    ) -> Set[Tuple[str, str]]:
        return {self._pair_key(s) for s in payload.items}

    def _fmt_missing(self, missing: Set[Tuple[str, str]]) -> str:
        head = list(missing)[:self._max_examples]
        rendered = ", ".join(f"{p[:8]}..→{r[:8]}.." for p, r in head)
        more = "" if len(missing) <= self._max_examples else (
            f" (+{len(missing) - self._max_examples} outros)"
        )
        return rendered + more

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        # Existência no contexto
        for rid in self._ids:
            if not context.has_resource(rid):
                result.add_error(
                    f"[{self.rule_id}] Recurso '{rid}' não encontrado no contexto."
                )
                return result

        # Hidratação + validação de tipo
        payloads: List[SimilarityScoreDataset] = []
        for rid in self._ids:
            res = context.ensure_payload(rid)
            payload = res.payload if res is not None else None
            if payload is None:
                result.add_warning(
                    f"[{self.rule_id}] Payload indisponível em '{rid}' — "
                    "pulando checagem de compatibilidade."
                )
                return result
            if not isinstance(payload, SimilarityScoreDataset):
                result.add_error(
                    f"[{self.rule_id}] '{rid}' não é SimilarityScoreDataset "
                    f"(got {type(payload).__name__}) — fusão exige datasets "
                    "de score."
                )
                return result
            payloads.append(payload)

        # Coleta keys + checa não-vazio
        keys_per_input: List[Set[Tuple[str, str]]] = []
        for rid, payload in zip(self._ids, payloads):
            keys = self._collect_keys(payload)
            if not keys:
                result.add_error(
                    f"[{self.rule_id}] Dataset '{rid}' está vazio — score "
                    "fusion exige todos os inputs não-vazios."
                )
                return result
            keys_per_input.append(keys)

        # Anchor = primeiro input. Cada subsequente é comparado contra ele.
        anchor_id = self._ids[0]
        anchor_keys = keys_per_input[0]

        divergences: List[str] = []
        for rid, keys in zip(self._ids[1:], keys_per_input[1:]):
            only_anchor = anchor_keys - keys
            only_other = keys - anchor_keys
            if not (only_anchor or only_other):
                continue
            parts = []
            if only_anchor:
                parts.append(
                    f"{len(only_anchor)} só em '{anchor_id}': "
                    f"{self._fmt_missing(only_anchor)}"
                )
            if only_other:
                parts.append(
                    f"{len(only_other)} só em '{rid}': "
                    f"{self._fmt_missing(only_other)}"
                )
            divergences.append(" | ".join(parts))

        if divergences:
            joined = "; ".join(divergences)
            result.add_error(
                f"[{self.rule_id}] Os {len(self._ids)} inputs não "
                f"compartilham o mesmo conjunto de pares (anchor: "
                f"'{anchor_id}'). {joined}. Score fusion exige que TODOS "
                "os sistemas tenham processado exatamente os mesmos pares "
                "(mesmo SCORING input, sem PAIR_MINING em apenas um lado)."
            )
            return result

        result.mark_ok()
        return result
