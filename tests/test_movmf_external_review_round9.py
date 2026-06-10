"""Regression tests para os 3 P0 + 3 P1 + 2 P2 da nona rodada da
revisão externa (2026-06-10 round 9).

P0:
- R9-1: CROSS_FIT_IDENTITY exige `reference_weights` e `reference_cdf`
        explícitos (não-None).
- R9-2: `ood_threshold` global vinculado a `ood_quantile`:
        q==0 ⇒ threshold None; q>0 ⇒ threshold == score em CDF>=q.
- R9-3: install usa a MESMA validação semântica do loader (helper único).

P1:
- R9-4: `oof_predictive_avg_score_by_fold` ponderado por
        `fold_mass` agrega a `oof_predictive_avg_score`.
- R9-5: seeds vinculadas a estratégia declarada
        (`fold_seeds == base + range(n_folds_used)`).
- R9-6: identity_ids rejeitam np.float32(nan), bool, bytes, complex.

P2:
- R9-8: `ENVELOPE_SCHEMA_VERSION = 2`.
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
    ENVELOPE_SCHEMA_VERSION,
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
        base_random_state=state.get("base_random_state"),
        fold_seed_strategy=state.get("fold_seed_strategy"),
    )


# ─── P0-R9-1 ───────────────────────────────────────────────────────────


def test_cross_fit_requires_explicit_reference_weights_and_cdf():
    """Remover `train_log_likelihoods_weights` ou `_cdf` → erro."""
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    payload["state"]["train_log_likelihoods_weights"] = None
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError,
        match="cross_fit_requires_explicit_weights",
    ):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R9-2 ───────────────────────────────────────────────────────────


def test_q_positive_rejects_global_threshold_not_matching_quantile():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    payload["state"]["ood_log_likelihood_threshold"] = 1e9
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError,
        match="ood_global_threshold_matches_weighted_quantile",
    ):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R9-3 ───────────────────────────────────────────────────────────


def test_install_rejects_incomplete_cross_fit_block_atomically():
    """install em CROSS_FIT_IDENTITY sem bloco completo → erro
    *antes* de mutar o estado (mesmo helper que o loader)."""
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    from faceaudit.services.oof_audit_metrics import (
        IncompleteCrossFitAuditPayloadError,
    )
    with pytest.raises(IncompleteCrossFitAuditPayloadError):
        m.install_reference_score_calibration(
            reference_scores=np.array([1.0, 2.0, 3.0]),
            reference_weights=np.array([1.0, 1.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        )


# ─── P1-R9-4 ───────────────────────────────────────────────────────────


def test_by_fold_average_must_aggregate_to_global_average():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    n = len(payload["state"]["oof_predictive_avg_score_by_fold"])
    # Substitui por valores arbitrários que não agregam corretamente.
    payload["state"]["oof_predictive_avg_score_by_fold"] = (
        [-999.0, 0.0, 999.0][:n]
    )
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError, match="by_fold_aggregates_to_global_average",
    ):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P1-R9-5 ───────────────────────────────────────────────────────────


def test_fold_seeds_must_match_declared_strategy():
    X, ids = _make_data()
    _, payload = _train_movmf_payload(X, ids)
    # Substitui fold_seeds por sequência incoerente com base+offsets.
    payload["state"]["fold_seeds"] = [10, 999, 2]
    _refresh_fingerprint(payload["state"])
    with pytest.raises(
        OofAuditSemanticError,
        match="fold_seeds_match_declared_strategy",
    ):
        MoVMFTypicalityModel.from_dict(payload)


def test_cross_fit_persists_base_random_state_and_strategy():
    X, ids = _make_data()
    m, _ = _train_movmf_payload(X, ids, random_state=42)
    assert m._base_random_state == 42
    assert m._fold_seed_strategy == "BASE_PLUS_FOLD_INDEX"


# ─── P1-R9-6 ───────────────────────────────────────────────────────────


def test_identity_rejects_numpy_float32_nan():
    X, ids = _make_data()
    bad_ids = (np.float32(np.nan),) + ids[1:]
    with pytest.raises(InvalidIdentityIdError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=_good_movmf_factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_identity_rejects_numpy_bool():
    X, ids = _make_data()
    bad_ids = (np.bool_(True),) + ids[1:]
    with pytest.raises(InvalidIdentityIdError, match="bool"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=_good_movmf_factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_identity_rejects_bytes():
    X, ids = _make_data()
    bad_ids = (b"id_x",) + ids[1:]
    with pytest.raises(InvalidIdentityIdError, match="bytes"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=_good_movmf_factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


def test_identity_rejects_complex():
    X, ids = _make_data()
    bad_ids = (complex(1.0, 0.0),) + ids[1:]
    with pytest.raises(InvalidIdentityIdError, match="complex"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=_good_movmf_factory,
            embeddings=X, identity_ids=bad_ids,
            n_folds=3, random_state=0,
        )


# ─── P2-R9-8 ───────────────────────────────────────────────────────────


def test_envelope_schema_version_is_2():
    assert ENVELOPE_SCHEMA_VERSION == 2

    X, ids = _make_data()
    m, _ = _train_movmf_payload(X, ids)
    envelope = TypicalityModelFactory.to_dict(m)
    assert envelope["schema_version"] == 2
