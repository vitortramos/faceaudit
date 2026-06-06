# faceaudit/services/architecture_state_compat_service.py
"""
Validação de compatibilidade entre um `LRArchitectureState` herdado e o
contexto de execução atual (INHERIT).

20+ checagens estruturais e semânticas — falha qualquer campo crítico
divergente. Retorna `list[CompatibilityIssue]` em vez de levantar exceção
para permitir que o caller agregue múltiplas issues numa única mensagem.

Esta camada é PURA: recebe state + spec + dicts de metadata dos inputs.
Nenhum I/O de contexto FaceAudit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Mapping, Optional

from faceaudit.models.lr.lr_architecture_spec import LRArchitectureSpec
from faceaudit.models.lr.lr_architecture_state import LRArchitectureState


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class CompatibilityIssue:
    """Uma divergência detectada entre state e contexto atual."""

    field: str
    expected: Any
    actual: Any
    severity: Severity
    message: str


def validate_state_compatibility(
    state: LRArchitectureState,
    spec: LRArchitectureSpec,
    incoming_base_meta: Optional[Mapping[str, Any]] = None,
    incoming_rarity_probe_meta: Optional[Mapping[str, Any]] = None,
    incoming_rarity_reference_meta: Optional[Mapping[str, Any]] = None,
    incoming_rarity_single_meta: Optional[Mapping[str, Any]] = None,
) -> List[CompatibilityIssue]:
    """
    Compara state herdado vs contexto atual em 20+ campos críticos.

    `incoming_*_meta` são dicts de metadata dos recursos sendo consumidos
    pelo executor — None quando o recurso não está presente (e.g.,
    `incoming_rarity_*_meta=None` em arquitetura sem tipicidade).

    Convenção de campos esperados no metadata:
      base: `score_kind`, `pair_role_orientation`
      rarity_*: `score_type`, `transform_id`, `transform_formula_version`,
                `transform_fingerprint`, `combiner`, `model_fingerprint`,
                `model_resource_id`

    Lista vazia = compatível. Caller decide se warnings são bloqueantes.
    """
    issues: List[CompatibilityIssue] = []

    # ── 1-5: identidade da arquitetura ──
    if state.architecture != spec.architecture:
        issues.append(CompatibilityIssue(
            field="architecture",
            expected=spec.architecture.value, actual=state.architecture.value,
            severity="error",
            message=f"State foi treinado para {state.architecture.value}, "
                    f"mas a instrução declara {spec.architecture.value}",
        ))

    if state.formula_version != spec.formula_version:
        issues.append(CompatibilityIssue(
            field="formula_version",
            expected=spec.formula_version, actual=state.formula_version,
            severity="error",
            message=f"Versão da fórmula divergente",
        ))

    if state.architecture_spec_fingerprint != spec.fingerprint():
        issues.append(CompatibilityIssue(
            field="architecture_spec_fingerprint",
            expected=spec.fingerprint(), actual=state.architecture_spec_fingerprint,
            severity="error",
            message="Fingerprint do spec divergente — implementação dos "
                    "termos ou composição da arquitetura mudou desde o ajuste",
        ))

    if tuple(state.feature_terms) != tuple(spec.feature_terms):
        issues.append(CompatibilityIssue(
            field="feature_terms",
            expected=[t.value for t in spec.feature_terms],
            actual=[t.value for t in state.feature_terms],
            severity="error",
            message="Composição de termos divergente",
        ))

    if state.output_dim != spec.output_dim:
        issues.append(CompatibilityIssue(
            field="output_dim",
            expected=spec.output_dim, actual=state.output_dim,
            severity="error",
            message="Dimensão de saída divergente",
        ))

    # ── 6-7: proveniência da base ──
    if incoming_base_meta is not None:
        incoming_base_kind = incoming_base_meta.get("score_kind")
        if incoming_base_kind is not None and incoming_base_kind != state.base_score_kind:
            issues.append(CompatibilityIssue(
                field="base_score_kind",
                expected=state.base_score_kind, actual=incoming_base_kind,
                severity="error",
                message=f"Tipo de score base divergente: state foi treinado "
                        f"com {state.base_score_kind}, recebendo {incoming_base_kind}",
            ))

        incoming_orientation = incoming_base_meta.get("pair_role_orientation")
        if (
            incoming_orientation is not None
            and incoming_orientation != state.pair_role_orientation.value
            and spec.requires_ordered_pair_roles
        ):
            issues.append(CompatibilityIssue(
                field="pair_role_orientation",
                expected=state.pair_role_orientation.value,
                actual=incoming_orientation,
                severity="error",
                message="Orientação dos papéis (probe/reference) divergente em "
                        "arquitetura que distingue lados",
            ))

    # ── 8-15: proveniência da tipicidade (lado probe) ──
    if spec.requires_typicality_probe:
        if incoming_rarity_probe_meta is None:
            issues.append(CompatibilityIssue(
                field="rarity_probe_meta",
                expected="metadata presente",
                actual=None,
                severity="error",
                message="Arquitetura exige rarity_probe mas o metadata do "
                        "recurso de entrada está ausente",
            ))
        else:
            _check_rarity_side(
                issues, "probe",
                state.rarity_probe_score_type,
                state.rarity_probe_transform_id,
                state.rarity_probe_transform_formula_version,
                state.rarity_probe_transform_fingerprint,
                state.rarity_probe_combiner,
                state.rarity_probe_model_fingerprint,
                incoming_rarity_probe_meta,
            )

    # ── 16-23: proveniência da tipicidade (lado reference) ──
    if spec.requires_typicality_reference:
        if incoming_rarity_reference_meta is None:
            issues.append(CompatibilityIssue(
                field="rarity_reference_meta",
                expected="metadata presente",
                actual=None,
                severity="error",
                message="Arquitetura exige rarity_reference mas o metadata do "
                        "recurso de entrada está ausente",
            ))
        else:
            _check_rarity_side(
                issues, "reference",
                state.rarity_reference_score_type,
                state.rarity_reference_transform_id,
                state.rarity_reference_transform_formula_version,
                state.rarity_reference_transform_fingerprint,
                state.rarity_reference_combiner,
                state.rarity_reference_model_fingerprint,
                incoming_rarity_reference_meta,
            )

    # ── ONE_FACE (V2.14.1): proveniência da tipicidade (face única) ──
    if getattr(spec, "requires_typicality_single", False):
        if incoming_rarity_single_meta is None:
            issues.append(CompatibilityIssue(
                field="rarity_single_meta",
                expected="metadata presente",
                actual=None,
                severity="error",
                message="Arquitetura ONE_FACE exige rarity_single mas o metadata "
                        "do recurso de entrada está ausente",
            ))
        else:
            _check_rarity_side(
                issues, "single",
                state.rarity_single_score_type,
                state.rarity_single_transform_id,
                state.rarity_single_transform_formula_version,
                state.rarity_single_transform_fingerprint,
                state.rarity_single_combiner,
                state.rarity_single_model_fingerprint,
                incoming_rarity_single_meta,
            )

    # ── 24: centering coerente ──
    if state.centering_mode != spec.centering_mode:
        issues.append(CompatibilityIssue(
            field="centering_mode",
            expected=spec.centering_mode.value, actual=state.centering_mode.value,
            severity="error",
            message="Modo de centragem divergente",
        ))

    return issues


def _check_rarity_side(
    issues: List[CompatibilityIssue],
    side_label: str,
    state_score_type: Optional[str],
    state_transform_id: Optional[str],
    state_transform_formula_version: Optional[str],
    state_transform_fingerprint: Optional[str],
    state_combiner: Optional[str],
    state_model_fingerprint: Optional[str],
    incoming: Mapping[str, Any],
) -> None:
    """Aplica 6 checagens de proveniência para um lado (probe ou reference)."""

    incoming_score_type = incoming.get("score_type")
    if (
        state_score_type is not None
        and incoming_score_type is not None
        and incoming_score_type != state_score_type
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_score_type",
            expected=state_score_type, actual=incoming_score_type,
            severity="error",
            message=f"Score type de tipicidade divergente no lado {side_label}",
        ))

    incoming_transform_id = incoming.get("transform_id")
    if (
        state_transform_id is not None
        and incoming_transform_id is not None
        and incoming_transform_id != state_transform_id
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_transform_id",
            expected=state_transform_id, actual=incoming_transform_id,
            severity="error",
            message=f"transform_id de raridade divergente no lado {side_label}",
        ))

    incoming_transform_formula_version = incoming.get("transform_formula_version")
    if (
        state_transform_formula_version is not None
        and incoming_transform_formula_version is not None
        and incoming_transform_formula_version != state_transform_formula_version
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_transform_formula_version",
            expected=state_transform_formula_version,
            actual=incoming_transform_formula_version,
            severity="error",
            message=f"Versão da fórmula do transform de raridade divergente no lado {side_label}",
        ))

    incoming_transform_fingerprint = incoming.get("transform_fingerprint")
    if (
        state_transform_fingerprint is not None
        and incoming_transform_fingerprint is not None
        and incoming_transform_fingerprint != state_transform_fingerprint
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_transform_fingerprint",
            expected=state_transform_fingerprint,
            actual=incoming_transform_fingerprint,
            severity="error",
            message=f"Fingerprint do transform de raridade divergente no lado {side_label}",
        ))

    incoming_combiner = incoming.get("combiner")
    if (
        state_combiner is not None
        and incoming_combiner is not None
        and incoming_combiner != state_combiner
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_combiner",
            expected=state_combiner, actual=incoming_combiner,
            severity="error",
            message=f"Combiner de tipicidade divergente no lado {side_label}",
        ))

    incoming_model_fingerprint = incoming.get("model_fingerprint")
    if (
        state_model_fingerprint is not None
        and incoming_model_fingerprint is not None
        and incoming_model_fingerprint != state_model_fingerprint
    ):
        issues.append(CompatibilityIssue(
            field=f"rarity_{side_label}_model_fingerprint",
            expected=state_model_fingerprint,
            actual=incoming_model_fingerprint,
            severity="error",
            message=f"Fingerprint do modelo de tipicidade divergente no lado "
                    f"{side_label} — state foi ajustado contra um modelo de "
                    "referência distinto",
        ))
