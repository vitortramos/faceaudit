"""R4 (revisão externa 2026-06-09 round 4): tests cobrindo a
log-likelihood preditiva OOF e os diagnósticos por fold.

A pedido do revisor:
> "Hoje você possui Cllr_min, AIC/BIC, convergência, mas ainda não
> possui formalmente a log-likelihood preditiva OOF agregada. Acrescentaria
> isso antes de criar um procedimento automático para selecionar K."

Esta suíte garante:
- O serviço expõe `oof_predictive_avg_score` + `*_by_fold` em
  CROSS_FIT_IDENTITY (MoVMF e k-NN).
- IN_SAMPLE_LEGACY mantém esses campos como None (não há OOF agregada).
- `oof_predictive_score_type` reflete a base canônica do backend.
- Contagens por fold somam ao N esperado.
- `ood_threshold_by_fold` reflete os thresholds dos modelos transitórios.
- O `summary()` do MoVMF expõe `oof_predictive_avg_log_likelihood` como
  alias quando score_type == LOG_DENSITY_ESTIMATE.
- O `summary()` do k-NN expõe os campos sem o alias log-likelihood
  (basis = MEAN_KNN_SIMILARITY, não é log-densidade).
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
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)


def _make_data(n_id=15, per=20, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d); c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.18 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(f"id_{i}")
    return np.asarray(X), tuple(ids)


# ─── Result fields ─────────────────────────────────────────────────────


def test_movmf_cross_fit_exposes_oof_predictive_avg_log_likelihood():
    """`oof_predictive_avg_score` finito e tipo LOG_DENSITY_ESTIMATE.

    Configura IDENTITY_CENTROIDS — o cenário forense relevante: cada
    fold contribui com 1 score por identidade de teste. Em
    FACE_LEVEL_RAW (default histórico), cada fold contribui com N_faces.
    """
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1,
        ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    assert result.oof_predictive_avg_score is not None
    assert np.isfinite(result.oof_predictive_avg_score)
    assert result.oof_predictive_avg_score_by_fold is not None
    assert len(result.oof_predictive_avg_score_by_fold) == 3
    # MoVMF expõe LOG_DENSITY_ESTIMATE como base.
    assert result.oof_predictive_score_type == "LOG_DENSITY_ESTIMATE"
    # Counts por fold somam ao total de identidades únicas
    # (IDENTITY_CENTROIDS: 1 score por identidade).
    n_unique_ids = len(set(ids))
    assert sum(result.n_identities_by_fold) == n_unique_ids
    assert sum(result.n_scores_by_fold) == n_unique_ids


def test_movmf_face_level_raw_n_scores_match_total_faces():
    """Em FACE_LEVEL_RAW (default MoVMF), n_scores_by_fold soma ao N
    total de faces (não ao n_unique_ids)."""
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    assert sum(result.n_scores_by_fold) == X.shape[0]


def test_knn_cross_fit_exposes_oof_predictive_avg_score():
    """No k-NN o tipo é MEAN_KNN_SIMILARITY — NÃO é log-likelihood."""
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
    assert result.oof_predictive_avg_score is not None
    assert np.isfinite(result.oof_predictive_avg_score)
    # k-NN basis é MEAN_KNN_SIMILARITY — T_k ∈ [-1, +1].
    assert result.oof_predictive_score_type == "MEAN_KNN_SIMILARITY"
    assert -1.0 <= result.oof_predictive_avg_score <= 1.0


def test_in_sample_legacy_has_no_oof_predictive_metrics():
    X, ids = _make_data()
    factory = lambda **kw: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=0, random_state=0,
    )
    assert result.oof_predictive_avg_score is None
    assert result.oof_predictive_avg_score_by_fold is None
    assert result.oof_predictive_score_type is None
    assert result.n_identities_by_fold is None


# ─── Summary do MoVMF / k-NN ───────────────────────────────────────────


def test_movmf_summary_exposes_log_likelihood_alias():
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    s = result.refit_model.summary()
    bd = s.backend_details
    # Alias log-likelihood populado (basis == LOG_DENSITY_ESTIMATE).
    assert bd["oof_predictive_avg_log_likelihood"] is not None
    assert bd["oof_predictive_avg_log_likelihood"] == bd["oof_predictive_avg_score"]
    assert bd["oof_predictive_log_likelihood_by_fold"] is not None
    assert len(bd["oof_predictive_log_likelihood_by_fold"]) == 3
    assert bd["oof_predictive_score_type"] == "LOG_DENSITY_ESTIMATE"


def test_knn_summary_no_log_likelihood_alias():
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
    s = result.refit_model.summary()
    bd = s.backend_details
    # k-NN expõe o score genérico, não o alias log-likelihood.
    assert bd["oof_predictive_avg_score"] is not None
    assert "oof_predictive_avg_log_likelihood" not in bd
    assert bd["oof_predictive_score_type"] == "MEAN_KNN_SIMILARITY"


# ─── Coerência por-fold ────────────────────────────────────────────────


def test_movmf_per_fold_avg_consistency_with_global():
    """A média ponderada das médias por-fold (pelo n_scores) deve
    aproximar a média global em FACE_LEVEL_RAW / IDENTITY_CENTROIDS."""
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    fold_avgs = np.asarray(result.oof_predictive_avg_score_by_fold)
    n_per_fold = np.asarray(result.n_scores_by_fold)
    weighted = float(np.sum(fold_avgs * n_per_fold) / n_per_fold.sum())
    np.testing.assert_allclose(
        weighted, result.oof_predictive_avg_score, atol=1e-9,
    )


def test_movmf_oof_predictive_metrics_survive_dict_roundtrip():
    """`to_dict` + `from_dict` preserva os campos R4."""
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    avg_before = result.refit_model._oof_predictive_avg_score
    by_fold_before = result.refit_model._oof_predictive_avg_score_by_fold
    type_before = result.refit_model._oof_predictive_score_type

    payload = result.refit_model.to_dict()
    restored = MoVMFTypicalityModel.from_dict(payload)
    assert restored._oof_predictive_avg_score == avg_before
    assert restored._oof_predictive_avg_score_by_fold == by_fold_before
    assert restored._oof_predictive_score_type == type_before


def test_movmf_ood_threshold_by_fold_finite_when_quantile_active():
    """Quando ood_quantile > 0, cada fold_model produz um threshold
    finito propagado ao result."""
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.05,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    assert result.ood_threshold_by_fold is not None
    assert len(result.ood_threshold_by_fold) == 3
    for t in result.ood_threshold_by_fold:
        assert t is not None
        assert np.isfinite(t)
