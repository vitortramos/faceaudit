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
    if value is None:
        return None
    out = []
    for i, x in enumerate(value):
        try:
            v = int(x)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"não inteiro ({x!r})."
            ) from exc
        if v < 0:
            raise ValueError(
                f"install_reference_score_calibration: {field}[{i}] "
                f"negativo ({v})."
            )
        out.append(v)
    return tuple(out)


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

    `persisted_fingerprint` None significa pickle antigo (anterior a
    R5) — sem verificação possível. Note que o revisor R5 também aceita
    isso porque a verificação só protege estados POSTERIORES ao install
    R5; nada a defender em pickles pré-R5.
    """
    if persisted_fingerprint is None:
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
