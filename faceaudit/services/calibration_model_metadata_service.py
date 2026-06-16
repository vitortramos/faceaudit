# faceaudit/services/calibration_model_metadata_service.py
"""
Service central para o vínculo entre vetorização e calibrador via
`vectorization_contract_fingerprint`.

Duas operações:

  - `bind_architecture_state(model_metadata, dataset_metadata)`:
    chamada pelo executor de `CALIBRATION_MODEL_TRAINING` no fim do
    treino. Copia para o metadata do modelo todos os campos contratuais
    necessários para futura aplicação:
      * `vectorization_contract_fingerprint` (comparado na aplicação)
      * `training_score_vector_contract_fingerprint` (auditoria)
      * `training_score_vector_numeric_fingerprint` (auditoria)
      * `training_raw_input_matrix_fingerprint` (auditoria)
      * `training_pair_key_order_hash` (auditoria)
      * `lr_architecture`, `lr_architecture_formula_version`
      * `architecture_spec_fingerprint`
      * `architecture_state` (state completo serializado, para INHERIT
        em VALIDATION/CASEWORK)

  - `validate_compatibility(model_metadata, dataset_metadata, mode)`:
    chamada pelo executor de `SCORE_CALIBRATION` antes de aplicar o
    modelo. Implementa a tabela assimétrica:

      | Dataset           | Calibrador        | STRICT       | LEGACY_PERMISSIVE |
      | ----------------- | ----------------- | ------------ | ----------------- |
      | sem fp            | sem fp            | OK + warn    | OK + warn         |
      | com fp            | com fp igual      | OK           | OK                |
      | com fp            | com fp diferente  | erro         | erro              |
      | com fp            | sem fp            | erro         | warn forte        |
      | sem fp            | com fp            | erro sempre  | erro sempre       |

    Retorna `list[CompatibilityIssue]`. Lista vazia = compatível. Caller
    decide se warnings bloqueiam.

Calibradores antigos (sem `vectorization_contract_fingerprint`) executam
sem mudança em `compatibility_mode=LEGACY_PERMISSIVE` (opt-in). Default
`STRICT` rejeita a combinação "dataset com fp + calibrador sem fp"
porque é insegura — não há como verificar que as features atuais
correspondem ao calibrador.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional

from faceaudit.enums.calibration_compatibility_mode import (
    CalibrationCompatibilityMode,
)


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class CompatibilityIssue:
    """Divergência detectada na validação do vínculo dataset↔calibrador."""

    field: str
    expected: Any
    actual: Any
    severity: Severity
    message: str


# Campos contratuais copiados do dataset para o modelo no bind.
# `vectorization_contract_fingerprint` é o único campo comparado na
# aplicação; os demais são auditoria.
_CONTRACTUAL_FIELDS = (
    "vectorization_contract_fingerprint",
    "lr_architecture",
    "lr_architecture_formula_version",
    "architecture_spec_fingerprint",
)

_TRAINING_PROVENANCE_FIELDS = (
    ("score_vector_contract_fingerprint", "training_score_vector_contract_fingerprint"),
    ("score_vector_numeric_fingerprint", "training_score_vector_numeric_fingerprint"),
    ("raw_input_matrix_fingerprint", "training_raw_input_matrix_fingerprint"),
    ("pair_key_order_hash", "training_pair_key_order_hash"),
)

# Snapshot completo do state (dict serializado) — necessário para INHERIT
# em VALIDATION/CASEWORK reaplicar parâmetros do calibrador.
_STATE_FIELD = "architecture_state"


class CalibrationModelMetadataService:
    """Service stateless para bind + validate."""

    @staticmethod
    def bind_architecture_state(
        model_metadata: Dict[str, Any],
        dataset_metadata: Mapping[str, Any],
    ) -> None:
        """
        Copia os campos contratuais + de auditoria do dataset vetorizado
        para o metadata do modelo. Idempotente — segura para chamar
        múltiplas vezes (caller passa pelo mesmo dataset).

        Não levanta exceção quando o dataset não tem campos LR (calibrador
        treinado em vetor não-arquitetural pula o bind sem erro).
        """
        contract_fp = dataset_metadata.get("vectorization_contract_fingerprint")
        if contract_fp is None or contract_fp == "":
            # Dataset legado — nada a vincular. Bind é no-op.
            return

        for field in _CONTRACTUAL_FIELDS:
            value = dataset_metadata.get(field)
            if value is not None:
                model_metadata[field] = value

        for src_field, dst_field in _TRAINING_PROVENANCE_FIELDS:
            value = dataset_metadata.get(src_field)
            if value is not None:
                model_metadata[dst_field] = value

        state_dict = dataset_metadata.get(_STATE_FIELD)
        if state_dict is not None:
            model_metadata[_STATE_FIELD] = state_dict

    @staticmethod
    def validate_compatibility(
        model_metadata: Mapping[str, Any],
        dataset_metadata: Mapping[str, Any],
        *,
        compatibility_mode: CalibrationCompatibilityMode = CalibrationCompatibilityMode.STRICT,
    ) -> List[CompatibilityIssue]:
        """
        Aplica a tabela assimétrica. Lista vazia = compatível.

        O caller (executor de SCORE_CALIBRATION) decide se warnings são
        bloqueantes. Erros sempre devem bloquear.
        """
        issues: List[CompatibilityIssue] = []

        ds_fp = dataset_metadata.get("vectorization_contract_fingerprint")
        model_fp = model_metadata.get("vectorization_contract_fingerprint")

        ds_has = bool(ds_fp)
        model_has = bool(model_fp)

        if not ds_has and not model_has:
            # Ambos legados — OK com warning informativo.
            issues.append(CompatibilityIssue(
                field="vectorization_contract_fingerprint",
                expected="ambos legados (sem fingerprint)",
                actual="ambos legados",
                severity="warning",
                message="Dataset e calibrador são legados (sem fingerprint). "
                        "Vínculo não pode ser verificado — assumindo compatível "
                        "por contexto histórico.",
            ))
            return issues

        if ds_has and model_has:
            if ds_fp == model_fp:
                return issues  # OK, sem issues
            issues.append(CompatibilityIssue(
                field="vectorization_contract_fingerprint",
                expected=model_fp, actual=ds_fp,
                severity="error",
                message=(
                    "Fingerprint do contrato de vetorização divergente entre "
                    "dataset e calibrador. As features atuais NÃO correspondem "
                    "às features sobre as quais o calibrador foi treinado."
                ),
            ))
            return issues

        if ds_has and not model_has:
            # Dataset com fp + calibrador sem fp.
            if compatibility_mode == CalibrationCompatibilityMode.LEGACY_PERMISSIVE:
                issues.append(CompatibilityIssue(
                    field="vectorization_contract_fingerprint",
                    expected="presente em ambos", actual="ausente no calibrador",
                    severity="warning",
                    message=(
                        "Dataset vetorizado tem fingerprint, mas o calibrador é "
                        "legado (sem fingerprint). Compatibilidade não pode ser "
                        "garantida — opt-in via LEGACY_PERMISSIVE. Use por sua "
                        "conta e risco."
                    ),
                ))
                # Warning forte também loga via warnings module
                warnings.warn(
                    "CalibrationModelMetadataService: aplicando calibrador "
                    "legado a dataset vetorizado novo em LEGACY_PERMISSIVE",
                    stacklevel=2,
                )
                return issues
            issues.append(CompatibilityIssue(
                field="vectorization_contract_fingerprint",
                expected="presente em ambos", actual="ausente no calibrador",
                severity="error",
                message=(
                    "Calibrador legado (sem fingerprint) recebendo dataset com "
                    "fingerprint — modo STRICT rejeita. Use "
                    "compatibility_mode=LEGACY_PERMISSIVE para forçar."
                ),
            ))
            return issues

        # Caso restante: dataset sem fp + calibrador com fp.
        # SEMPRE erro — independente de modo. Não é possível saber se as
        # features atuais correspondem ao que o calibrador espera.
        issues.append(CompatibilityIssue(
            field="vectorization_contract_fingerprint",
            expected=model_fp, actual="ausente no dataset",
            severity="error",
            message=(
                "Calibrador novo (com fingerprint) recebendo dataset legado "
                "(sem fingerprint) — combinação SEMPRE proibida, mesmo em "
                "LEGACY_PERMISSIVE, porque não há como verificar que as "
                "features atuais correspondem ao calibrador."
            ),
        ))
        return issues
