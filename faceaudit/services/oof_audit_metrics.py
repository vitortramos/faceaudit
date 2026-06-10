"""Helpers compartilhados (P0-R5-2/3/6) para validação + fingerprint
auditável das métricas OOF persistidas pelos backends de tipicidade
após `install_reference_score_calibration`.

Por que módulo separado:
- Mesma validação aplica-se ao MoVMF e ao k-NN; sem helper, qualquer
  divergência entre install signatures viraria assimetria semântica.
- O fingerprint é uma propriedade do contrato R5 — recalculado e
  comparado pelo `from_dict` (defesa contra adulteração de pickle —
  o revisor demonstrou que `payload["state"]["oof_predictive_avg_score"]
  += 1234` passava sem ser detectado em R4).
"""
from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np


class OofAuditMetricMismatchError(ValueError):
    """Adulteração ou corrupção das métricas OOF detectada pelo fingerprint
    auditável.

    P0-R5-3: garantia de que valores carregados de pickle correspondem
    aos que foram instalados — bloqueia round-trip que muta
    `oof_predictive_avg_score` por fora do `install_reference_score_calibration`.
    """

    def __init__(self, *, expected: str, computed: str) -> None:
        super().__init__(
            f"OofAuditMetricMismatch: fingerprint persistido={expected!r} "
            f"não bate com recomputado={computed!r}. O pickle foi alterado "
            "fora do `install_reference_score_calibration` ou está corrompido."
        )
        self.expected = expected
        self.computed = computed


class MissingOofAuditMetricFingerprintError(ValueError):
    """Payload OOF presente sem fingerprint auditável correspondente.

    P0-R6-2 (revisão externa 2026-06-09 round 6): fechar o bypass
    demonstrado pelo revisor — `payload["state"]["audit_metrics_fingerprint"]
    = None` permitia que QUALQUER alteração das métricas OOF passasse
    despercebida. A nova regra é: se há qualquer campo OOF persistido,
    o fingerprint é obrigatório.
    """

    def __init__(self, *, fields_present: tuple) -> None:
        super().__init__(
            "MissingOofAuditMetricFingerprint: payload tem campos OOF "
            f"persistidos {fields_present!r} mas `audit_metrics_fingerprint` "
            "é None. Em modelos com payload OOF, o fingerprint auditável "
            "é obrigatório — removê-lo desativaria a proteção contra "
            "adulteração das métricas."
        )
        self.fields_present = fields_present


# ─── Validadores de finitude e comprimento ──────────────────────────────


def _normalize_oof_avg_score(value, *, field: str) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(
            f"install_reference_score_calibration: {field} não-finito "
            f"({v!r})."
        )
    return v


def _normalize_oof_per_fold_floats(value, *, field: str) -> Optional[tuple]:
    if value is None:
        return None
    out = tuple(float(x) for x in value)
    for i, v in enumerate(out):
        if not np.isfinite(v):
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"não-finito ({v!r})."
            )
    return out


def _normalize_oof_per_fold_floats_optional(
    value, *, field: str
) -> Optional[tuple]:
    """Versão de `_normalize_oof_per_fold_floats` que aceita `None` em
    elementos (caso do `ood_threshold_by_fold` quando `ood_quantile=0`)."""
    if value is None:
        return None
    out = []
    for i, x in enumerate(value):
        if x is None:
            out.append(None)
            continue
        v = float(x)
        if not np.isfinite(v):
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"não-finito ({v!r})."
            )
        out.append(v)
    return tuple(out)


def _normalize_oof_per_fold_ints(value, *, field: str) -> Optional[tuple]:
    """P1-R6-6 (revisão externa round 6): aceita SOMENTE inteiros
    estritos (int/np.integer ou float com `.is_integer()=True`).
    Floats fracionários como 1.9 ou 2.2 são rejeitados — antes,
    `int(1.9)` truncava silenciosamente para 1.

    `bool` é explicitamente rejeitado (Python `bool` é subclasse de
    `int` mas semanticamente seria ambíguo aqui).
    """
    if value is None:
        return None
    out = []
    for i, x in enumerate(value):
        if isinstance(x, bool):
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"bool não é aceito ({x!r})."
            )
        if isinstance(x, (int, np.integer)):
            v = int(x)
        elif isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                raise ValueError(
                    f"install_reference_score_calibration: {field}[{i}] "
                    f"não-finito ({x!r})."
                )
            if not float(x).is_integer():
                raise ValueError(
                    f"install_reference_score_calibration: {field}[{i}] "
                    f"= {x!r} não é inteiro (fração não nula)."
                )
            v = int(x)
        else:
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"= {x!r} não é numérico inteiro."
            )
        if v < 0:
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"negativo ({v})."
            )
        out.append(v)
    return tuple(out)


def _normalize_n_folds_used(value) -> Optional[int]:
    """P1-R6-6: `n_folds_used` aceita None, int não-negativo ou float
    com parte fracionária zero. Floats fracionários levantam.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"install_reference_score_calibration: n_folds_used bool "
            f"não é aceito ({value!r})."
        )
    if isinstance(value, (int, np.integer)):
        v = int(value)
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError(
                f"install_reference_score_calibration: n_folds_used "
                f"não-finito ({value!r})."
            )
        if not float(value).is_integer():
            raise ValueError(
                f"install_reference_score_calibration: n_folds_used "
                f"= {value!r} não é inteiro."
            )
        v = int(value)
    else:
        raise ValueError(
            f"install_reference_score_calibration: n_folds_used "
            f"= {value!r} não é numérico inteiro."
        )
    if v < 0:
        raise ValueError(
            f"install_reference_score_calibration: n_folds_used "
            f"negativo ({v})."
        )
    return v


def _validate_oof_per_fold_lengths(
    *,
    n_folds_used: Optional[int],
    oof_by_fold: Optional[tuple],
    n_identities_by_fold: Optional[tuple],
    n_scores_by_fold: Optional[tuple],
    ood_threshold_by_fold: Optional[tuple],
    fold_seeds: Optional[tuple],
) -> None:
    """Se `n_folds_used` está definido, todos os arrays por-fold passados
    devem ter exatamente esse comprimento. Arrays None são tolerados (o
    caller pode não estar passando aquele diagnóstico).
    """
    if n_folds_used is None:
        # Sem n_folds_used, exigimos que os arrays por-fold concordem
        # entre si (qualquer um define o tamanho).
        present = [
            arr for arr in (
                oof_by_fold, n_identities_by_fold, n_scores_by_fold,
                ood_threshold_by_fold, fold_seeds,
            ) if arr is not None
        ]
        if not present:
            return
        target = len(present[0])
        for arr in present[1:]:
            if len(arr) != target:
                raise ValueError(
                    f"install_reference_score_calibration: arrays por-fold "
                    f"com comprimentos diferentes ({target} vs {len(arr)})."
                )
        return

    for field_name, arr in (
        ("oof_predictive_avg_score_by_fold", oof_by_fold),
        ("n_identities_by_fold", n_identities_by_fold),
        ("n_scores_by_fold", n_scores_by_fold),
        ("ood_threshold_by_fold", ood_threshold_by_fold),
        ("fold_seeds", fold_seeds),
    ):
        if arr is None:
            continue
        if len(arr) != n_folds_used:
            raise ValueError(
                f"install_reference_score_calibration: len({field_name}) "
                f"= {len(arr)} ≠ n_folds_used = {n_folds_used}."
            )


# ─── Fingerprint auditável ──────────────────────────────────────────────


def _compute_audit_metrics_fingerprint(
    *,
    oof_predictive_avg_score: Optional[float],
    oof_predictive_avg_score_by_fold: Optional[tuple],
    oof_predictive_score_type: Optional[str],
    n_identities_by_fold: Optional[tuple],
    n_scores_by_fold: Optional[tuple],
    ood_threshold_by_fold: Optional[tuple],
    split_fingerprint: Optional[str],
    fold_seeds: Optional[tuple],
    n_folds_used: Optional[int],
) -> Optional[str]:
    """SHA-256 sobre representação textual canônica das métricas OOF.

    Cobre os campos que o revisor R5 enumerou explicitamente. Retorna
    None quando NENHUMA métrica OOF foi instalada (todos os campos None
    — caso IN_SAMPLE_LEGACY); senão retorna hex digest.

    Determinismo: floats serializados com `repr` (preserva precisão);
    tuplas serializadas como JSON-like. Mudar a ordem dos campos quebra
    fingerprint — propositadamente; alterar este código exige migração.
    """
    fields = [
        oof_predictive_avg_score,
        oof_predictive_avg_score_by_fold,
        oof_predictive_score_type,
        n_identities_by_fold,
        n_scores_by_fold,
        ood_threshold_by_fold,
        split_fingerprint,
        fold_seeds,
        n_folds_used,
    ]
    if all(f is None for f in fields):
        return None

    h = hashlib.sha256()

    def _feed(label: str, value) -> None:
        h.update(label.encode("utf-8"))
        h.update(b"=")
        if value is None:
            h.update(b"None")
        elif isinstance(value, float):
            h.update(repr(value).encode("utf-8"))
        elif isinstance(value, int):
            h.update(str(value).encode("utf-8"))
        elif isinstance(value, str):
            h.update(value.encode("utf-8"))
        elif isinstance(value, (tuple, list)):
            h.update(b"(")
            for i, x in enumerate(value):
                if i > 0:
                    h.update(b",")
                if x is None:
                    h.update(b"None")
                elif isinstance(x, float):
                    h.update(repr(x).encode("utf-8"))
                elif isinstance(x, int):
                    h.update(str(x).encode("utf-8"))
                else:
                    h.update(str(x).encode("utf-8"))
            h.update(b")")
        else:
            h.update(repr(value).encode("utf-8"))
        h.update(b";")

    _feed("oof_predictive_avg_score", oof_predictive_avg_score)
    _feed("oof_predictive_avg_score_by_fold", oof_predictive_avg_score_by_fold)
    _feed("oof_predictive_score_type", oof_predictive_score_type)
    _feed("n_identities_by_fold", n_identities_by_fold)
    _feed("n_scores_by_fold", n_scores_by_fold)
    _feed("ood_threshold_by_fold", ood_threshold_by_fold)
    _feed("split_fingerprint", split_fingerprint)
    _feed("fold_seeds", fold_seeds)
    _feed("n_folds_used", n_folds_used)
    return h.hexdigest()


def _revalidate_oof_payload_or_raise(
    *,
    oof_predictive_avg_score,
    oof_predictive_avg_score_by_fold,
    n_identities_by_fold,
    n_scores_by_fold,
    ood_threshold_by_fold,
    fold_seeds,
    n_folds_used,
) -> None:
    """P0-R6-3 (revisão externa round 6): re-aplica as validações
    estruturais sobre o payload OOF carregado de pickle.

    SHA-256 não substitui validação semântica — alguém que tenha
    capacidade de adulterar o payload pode recalcular o hash. Aqui
    re-rodamos os normalizadores: NaN/inf, contagens negativas,
    comprimentos incoerentes. Levanta `ValueError` em qualquer
    violação.

    Tolera campos None (pickle pré-R4 sem cross-fit). Mas quando há
    payload, os comprimentos têm de ser coerentes.
    """
    _normalize_oof_avg_score(
        oof_predictive_avg_score, field="oof_predictive_avg_score",
    )
    _normalize_oof_per_fold_floats(
        oof_predictive_avg_score_by_fold,
        field="oof_predictive_avg_score_by_fold",
    )
    _normalize_oof_per_fold_ints(
        n_identities_by_fold, field="n_identities_by_fold",
    )
    _normalize_oof_per_fold_ints(
        n_scores_by_fold, field="n_scores_by_fold",
    )
    _normalize_oof_per_fold_floats_optional(
        ood_threshold_by_fold, field="ood_threshold_by_fold",
    )
    _normalize_oof_per_fold_ints(fold_seeds, field="fold_seeds")
    _validate_oof_per_fold_lengths(
        n_folds_used=(int(n_folds_used) if n_folds_used is not None else None),
        oof_by_fold=oof_predictive_avg_score_by_fold,
        n_identities_by_fold=n_identities_by_fold,
        n_scores_by_fold=n_scores_by_fold,
        ood_threshold_by_fold=ood_threshold_by_fold,
        fold_seeds=fold_seeds,
    )


def _verify_audit_metrics_fingerprint_or_raise(
    *,
    persisted_fingerprint: Optional[str],
    oof_predictive_avg_score: Optional[float],
    oof_predictive_avg_score_by_fold: Optional[tuple],
    oof_predictive_score_type: Optional[str],
    n_identities_by_fold: Optional[tuple],
    n_scores_by_fold: Optional[tuple],
    ood_threshold_by_fold: Optional[tuple],
    split_fingerprint: Optional[str],
    fold_seeds: Optional[tuple],
    n_folds_used: Optional[int],
) -> None:
    """Recalcula o fingerprint e compara ao persistido. Para uso no
    `from_dict` dos backends.

    P0-R6-2 (revisão externa round 6): regra fail-closed —
    - SEM nenhum campo OOF persistido (todos None): pickle pré-R4 sem
      cross-fit. Permite migração (fingerprint pode ser None).
    - COM qualquer campo OOF persistido + fingerprint None: rejeita
      com `MissingOofAuditMetricFingerprintError` (fechar bypass
      demonstrado pelo revisor: remover o fingerprint desativava a
      proteção).
    - COM payload OOF + fingerprint presente: recalcula e compara.
    """
    fields_by_name = {
        "oof_predictive_avg_score": oof_predictive_avg_score,
        "oof_predictive_avg_score_by_fold": oof_predictive_avg_score_by_fold,
        "oof_predictive_score_type": oof_predictive_score_type,
        "n_identities_by_fold": n_identities_by_fold,
        "n_scores_by_fold": n_scores_by_fold,
        "ood_threshold_by_fold": ood_threshold_by_fold,
        "split_fingerprint": split_fingerprint,
        "fold_seeds": fold_seeds,
        "n_folds_used": n_folds_used,
    }
    fields_present = tuple(
        name for name, value in fields_by_name.items()
        if value is not None
    )

    if persisted_fingerprint is None:
        if fields_present:
            raise MissingOofAuditMetricFingerprintError(
                fields_present=fields_present,
            )
        return

    computed = _compute_audit_metrics_fingerprint(
        oof_predictive_avg_score=oof_predictive_avg_score,
        oof_predictive_avg_score_by_fold=oof_predictive_avg_score_by_fold,
        oof_predictive_score_type=oof_predictive_score_type,
        n_identities_by_fold=n_identities_by_fold,
        n_scores_by_fold=n_scores_by_fold,
        ood_threshold_by_fold=ood_threshold_by_fold,
        split_fingerprint=split_fingerprint,
        fold_seeds=fold_seeds,
        n_folds_used=n_folds_used,
    )
    if computed != persisted_fingerprint:
        raise OofAuditMetricMismatchError(
            expected=str(persisted_fingerprint),
            computed=str(computed),
        )
