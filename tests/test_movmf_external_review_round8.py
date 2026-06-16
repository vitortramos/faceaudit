"""Regression tests para os 3 P0 + 3 P1 + 2 P2 da oitava rodada da
revisão externa (2026-06-09 round 8).

P0:
- R8-1: k-NN com `ood_quantile=0.0` sobrevive ao round-trip
        (`ood_threshold=None` no lugar de `-inf`).
- R8-2: validações semânticas do bloco OOF — `split_fingerprint` em
        SHA-256 hex; `sum(n_scores_by_fold) == len(reference_scores)`;
        IDENTITY_CENTROIDS exige `n_scores == n_identities`; média
        OOF global ≈ média ponderada dos scores persistidos;
        `ood_threshold_by_fold` coerente com `ood_quantile`; bloco
        OOF ausente em modos não-CROSS_FIT.
- R8-3: envelope da factory persiste + valida `audit_metrics_fingerprint`.

P1:
- R8-6: identity_ids rejeitam None/""/"  "/NaN antes de str(...).

P2:
- R8-7 (doc): doc cita estimador de quantil correto — só validado por
  inspeção manual.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.oof_audit_metrics import OofAuditSemanticError
from faceaudit.services.reference_score_calibration_service import (
    InvalidIdentityIdError,
    ReferenceScoreCalibrationService,
)
from faceaudit.services.typicality_model_factory import (
    EnvelopeFingerprintMismatch,
    TypicalityModelFactory,
)


def _make_data(n_id=12, per=20, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d); c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(f"id_{i}")
    return np.asarray(X), tuple(ids)


# ─── P0-R8-1 ───────────────────────────────────────────────────────────


def test_knn_q0_in_sample_roundtrip():
    """k-NN com `ood_quantile=0.0` deve sobreviver ao round-trip
    via IN_SAMPLE_LEGACY."""
    X, ids = _make_data()
    m = KNNTypicalityModel(
        k=3, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X, identities=ids)
    payload = m.to_dict()
    restored = KNNTypicalityModel.from_dict(payload)
    assert restored._state.ood_threshold is None


def test_knn_q0_cross_fit_roundtrip():
    """k-NN com `ood_quantile=0.0` deve sobreviver ao round-trip via
    CROSS_FIT_IDENTITY (`ood_threshold_by_fold == (None, None, None)`)."""
    X, ids = _make_data()
    factory = lambda **kw: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    payload = result.refit_model.to_dict()
    restored = KNNTypicalityModel.from_dict(payload)
    assert restored._state.ood_threshold is None
    assert all(t is None for t in restored._ood_threshold_by_fold)


# ─── P0-R8-2 ───────────────────────────────────────────────────────────


def _good_movmf_factory(seed=None):
    return MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.05,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=seed if seed is not None else 0,
    )


def _train_movmf_payload(X, ids, n_folds=3, random_state=0):
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=_good_movmf_factory,
        embeddings=X, identity_ids=ids,
        n_folds=n_folds, random_state=random_state,
    )
    return result.refit_model, result.refit_model.to_dict()


def _refresh_fingerprint(state):
    """Recalcula `audit_metrics_fingerprint` para esconder adulteração
    estrutural e forçar a validação semântica a disparar."""
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    state["audit_metrics_fingerprint"] = _compute_audit_metrics_fingerprint(
        oof_predictive_avg_score=state["oof_predictive_avg_score"],
        oof_predictive_avg_score_by_fold=tuple(
            state["oof_predictive_avg_score_by_fold"]
        ),
        oof_predictive_score_type=state["oof_predictive_score_type"],
        n_identities_by_fold=tuple(state["n_identities_by_fold"]),
        n_scores_by_fold=tuple(state["n_scores_by_fold"]),
        ood_threshold_by_fold=tuple(state["ood_threshold_by_fold"]),
        split_fingerprint=state["split_fingerprint"],
        fold_seeds=tuple(state["fold_seeds"]),
        n_folds_used=state["n_folds_used"],
    )


def test_cross_fit_rejects_malformed_split_fingerprint():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    # split_fingerprint vazio
    payload["state"]["split_fingerprint"] = ""
    _refresh_fingerprint(payload["state"])
    with pytest.raises(OofAuditSemanticError, match="split_fingerprint"):
        MoVMFTypicalityModel.from_dict(payload)


def test_cross_fit_rejects_non_sha256_split_fingerprint():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    payload["state"]["split_fingerprint"] = "not-a-sha256"
    _refresh_fingerprint(payload["state"])
    with pytest.raises(OofAuditSemanticError, match="split_fingerprint"):
        MoVMFTypicalityModel.from_dict(payload)


def test_loader_rejects_counts_not_matching_persisted_reference():
    """sum(n_scores_by_fold) deve == len(reference_scores)."""
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    # Substitui contagens por valores artificiais grandes.
    n = len(payload["state"]["n_scores_by_fold"])
    payload["state"]["n_scores_by_fold"] = [999] * n
    payload["state"]["n_identities_by_fold"] = [999] * n
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError, match="sum_n_scores_equals_reference_length",
    ):
        MoVMFTypicalityModel.from_dict(payload)


def test_loader_rejects_identity_centroids_mismatch():
    """IDENTITY_CENTROIDS: n_scores_by_fold == n_identities_by_fold."""
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    n_scores = list(payload["state"]["n_scores_by_fold"])
    payload["state"]["n_identities_by_fold"] = [
        max(1, ns // 2) for ns in n_scores
    ]
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError,
        match="identity_centroids_one_score_per_identity",
    ):
        MoVMFTypicalityModel.from_dict(payload)


def test_loader_rejects_global_oof_average_not_matching_reference():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    payload["state"]["oof_predictive_avg_score"] = 1234.5
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError, match="avg_score_matches_weighted_reference",
    ):
        MoVMFTypicalityModel.from_dict(payload)


def test_loader_rejects_oof_threshold_none_when_quantile_positive():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    n = len(payload["state"]["ood_threshold_by_fold"])
    payload["state"]["ood_threshold_by_fold"] = [None] * n
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError,
        match="ood_threshold_required_when_quantile_positive",
    ):
        MoVMFTypicalityModel.from_dict(payload)


def test_loader_rejects_oof_payload_when_mode_is_in_sample_legacy():
    """Bloco OOF anexado a `IN_SAMPLE_LEGACY` é semanticamente inválido."""
    X, ids = _make_data()
    m, payload = _train_movmf_payload(X, ids)
    # Muda o modo para IN_SAMPLE_LEGACY mantendo o bloco OOF.
    payload["state"]["reference_score_calibration_mode"] = "IN_SAMPLE_LEGACY"
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError, match="oof_block_requires_cross_fit_mode",
    ):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R8-3 ───────────────────────────────────────────────────────────


def test_envelope_persists_audit_metrics_fingerprint():
    X, ids = _make_data()
    m, _ = _train_movmf_payload(X, ids)
    envelope = TypicalityModelFactory.to_dict(m)
    assert "audit_metrics_fingerprint" in envelope["fingerprints"]
    assert envelope["fingerprints"]["audit_metrics_fingerprint"] is not None


def test_envelope_binds_audit_metrics_fingerprint():
    """Adulterar o `audit_metrics_fingerprint` do envelope sem mudar
    o do modelo (ou vice-versa) → erro."""
    X, ids = _make_data()
    m, _ = _train_movmf_payload(X, ids)
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["fingerprints"]["audit_metrics_fingerprint"] = "a" * 64
    with pytest.raises(EnvelopeFingerprintMismatch):
        TypicalityModelFactory.from_dict(envelope)


def test_envelope_rejects_audit_fp_removed_when_model_has_one():
    X, ids = _make_data()
    m, _ = _train_movmf_payload(X, ids)
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["fingerprints"]["audit_metrics_fingerprint"] = None
    with pytest.raises(EnvelopeFingerprintMismatch):
        TypicalityModelFactory.from_dict(envelope)


# ─── P1-R8-6 ───────────────────────────────────────────────────────────


def test_calibration_rejects_none_identity():
    X, ids = _make_data()
    bad_ids = (None,) + ids[1:]
    factory = _good_movmf_factory
    with pytest.raises(InvalidIdentityIdError, match="None"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_calibration_rejects_empty_string_identity():
    X, ids = _make_data()
    bad_ids = ("",) + ids[1:]
    factory = _good_movmf_factory
    with pytest.raises(InvalidIdentityIdError, match="vazia"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_calibration_rejects_whitespace_only_identity():
    X, ids = _make_data()
    bad_ids = ("   ",) + ids[1:]
    factory = _good_movmf_factory
    with pytest.raises(InvalidIdentityIdError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_calibration_rejects_nan_identity():
    X, ids = _make_data()
    bad_ids = (float("nan"),) + ids[1:]
    factory = _good_movmf_factory
    with pytest.raises(InvalidIdentityIdError, match="não-finito"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )
