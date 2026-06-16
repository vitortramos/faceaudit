"""Regression tests para P1 da segunda rodada da revisão externa
(2026-06-09 round 2).

- P1-R2-6: AIC/BIC retorna None em INVERSE_IDENTITY_FREQUENCY.
- P1-R2-7: from_dict valida integridade estrutural.
- P1-R2-8: KNN inference externa NÃO descarta self-match por cosseno.
- P1-R2-9: TypicalityScoreBatch invariantes estritos.
- P1-R2-10: inspect.signature em vez de try/except TypeError.
- P1-R2-11: LOG_DENSITY_ESTIMATE oculto em INV_ID_FREQ.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
    _factory_accepts_seed,
)
from faceaudit.services.reference_score_distribution_validator import (
    InvalidReferenceDistributionError,
    validate_reference_score_distribution,
)


def _make_data(n_id=10, per=10, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d)
        c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.18 * rng.normal(size=d)
            v /= np.linalg.norm(v)
            X.append(v)
            ids.append(f"id_{i}")
    return np.asarray(X), tuple(ids)


# ─── P1-R2-6 ─────────────────────────────────────────────────────────────


def test_movmf_aic_bic_none_in_inv_id_freq():
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    assert m.aic_ is None
    assert m.bic_ is None


def test_movmf_aic_bic_defined_in_face_level_raw():
    X, _ = _make_data()
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m.fit(X)
    assert m.aic_ is not None
    assert m.bic_ is not None
    assert np.isfinite(m.aic_)
    assert np.isfinite(m.bic_)


# ─── P1-R2-7 ─────────────────────────────────────────────────────────────


def test_validator_rejects_unsorted_scores():
    with pytest.raises(InvalidReferenceDistributionError, match="ordenado"):
        validate_reference_score_distribution(
            scores=np.array([3.0, 1.0, 2.0]),
        )


def test_validator_rejects_nan_scores():
    with pytest.raises(InvalidReferenceDistributionError, match="finitos"):
        validate_reference_score_distribution(
            scores=np.array([1.0, np.nan, 3.0]),
        )


def test_validator_rejects_negative_weights():
    with pytest.raises(InvalidReferenceDistributionError, match="não-negativo"):
        validate_reference_score_distribution(
            scores=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, -0.5, 1.0]),
        )


def test_validator_rejects_non_monotonic_cdf():
    with pytest.raises(InvalidReferenceDistributionError, match="monotônica"):
        validate_reference_score_distribution(
            scores=np.array([1.0, 2.0, 3.0]),
            cdf=np.array([0.3, 0.2, 1.0]),
        )


def test_validator_rejects_cdf_not_reaching_one():
    with pytest.raises(InvalidReferenceDistributionError, match="último valor"):
        validate_reference_score_distribution(
            scores=np.array([1.0, 2.0, 3.0]),
            cdf=np.array([0.3, 0.6, 0.85]),
        )


def test_movmf_from_dict_rejects_corrupted_state():
    """from_dict valida integridade quando o pickle vem corrompido."""
    X, ids = _make_data(n_id=8, per=6)
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    d = m.to_dict()
    # Corrompe scores: shuffle (perde ordenação)
    d["state"]["train_log_likelihoods"] = [3.0, 1.0, 2.0, 5.0, 4.0]
    with pytest.raises(InvalidReferenceDistributionError):
        MoVMFTypicalityModel.from_dict(d)


# ─── P1-R2-8 ─────────────────────────────────────────────────────────────


def test_knn_inference_does_not_drop_legitimate_close_neighbor():
    """Consulta com cos ≈ 1 mantém o vizinho (não descarta como
    self-match heuristicamente)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 16))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    m = KNNTypicalityModel(
        k=10, ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    ).fit(X)
    # Query = primeira face do dataset (re-scoring legítimo)
    info = m.get_typicality_info(X[0])
    # Vizinho mais próximo deve ser ela mesma (cos = 1.0)
    assert info.knn_similarities[0] > 0.999


# ─── P1-R2-9 ─────────────────────────────────────────────────────────────


def test_batch_rejects_nonfinite_scores():
    with pytest.raises(ValueError, match="não finitos"):
        TypicalityScoreBatch(
            scores=np.array([1.0, np.nan]),
            n=2, embedding_dim=8,
            score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            score_semantics_id="X",
            model_type=TypicalityModelType.KNN,
            model_state_fingerprint="aa",
            model_contract_fingerprint="bb",
        )


def test_batch_rejects_zero_embedding_dim():
    with pytest.raises(ValueError, match="embedding_dim"):
        TypicalityScoreBatch(
            scores=np.array([1.0]),
            n=1, embedding_dim=0,
            score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            score_semantics_id="X",
            model_type=TypicalityModelType.KNN,
            model_state_fingerprint="aa",
            model_contract_fingerprint="bb",
        )


def test_batch_rejects_missing_fingerprints():
    with pytest.raises(ValueError, match="model_state_fingerprint"):
        TypicalityScoreBatch(
            scores=np.array([1.0]),
            n=1, embedding_dim=8,
            score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            score_semantics_id="X",
            model_type=TypicalityModelType.KNN,
            model_state_fingerprint="",
            model_contract_fingerprint="bb",
        )


def test_batch_percentile_requires_reference_distribution_fingerprint():
    with pytest.raises(ValueError, match="reference_score_distribution_fingerprint"):
        TypicalityScoreBatch(
            scores=np.array([50.0]),
            n=1, embedding_dim=8,
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            score_semantics_id="X",
            percentile_basis_score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            reference_score_distribution_fingerprint="",
            model_type=TypicalityModelType.KNN,
            model_state_fingerprint="aa",
            model_contract_fingerprint="bb",
        )


def test_batch_percentile_rejects_out_of_range():
    with pytest.raises(ValueError, match="percentile fora"):
        TypicalityScoreBatch(
            scores=np.array([105.0]),
            n=1, embedding_dim=8,
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            score_semantics_id="X",
            percentile_basis_score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            reference_score_distribution_fingerprint="dd",
            model_type=TypicalityModelType.KNN,
            model_state_fingerprint="aa",
            model_contract_fingerprint="bb",
        )


# ─── P1-R2-10 ────────────────────────────────────────────────────────────


def test_factory_accepts_seed_inspects_signature():
    def factory_with(seed=None):
        return None

    def factory_without():
        return None

    def factory_kwargs(**kw):
        return None

    assert _factory_accepts_seed(factory_with)
    assert not _factory_accepts_seed(factory_without)
    assert _factory_accepts_seed(factory_kwargs)


def test_factory_typeerror_is_not_silently_swallowed():
    """Se a factory levantar TypeError DENTRO (erro de tipo legítimo),
    o serviço deve propagar — não cair no fallback silencioso."""
    X, ids = _make_data(n_id=8, per=6)

    def factory_buggy(seed=None):
        raise TypeError("erro interno legítimo")

    svc = ReferenceScoreCalibrationService()
    with pytest.raises(TypeError, match="erro interno legítimo"):
        svc.calibrate(
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory_buggy,
            embeddings=X, identity_ids=ids,
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            n_folds=3, random_state=0,
        )


# ─── P1-R2-11 ────────────────────────────────────────────────────────────


def test_knn_inv_id_freq_hides_log_density():
    X, ids = _make_data()
    m = KNNTypicalityModel(
        k=5,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    assert TypicalityScoreType.LOG_DENSITY_ESTIMATE not in m.enabled_score_types
    assert TypicalityScoreType.MEAN_KNN_SIMILARITY in m.enabled_score_types
    assert TypicalityScoreType.TYPICALITY_PERCENTILE in m.enabled_score_types


def test_knn_face_level_raw_exposes_log_density():
    X, ids = _make_data()
    m = KNNTypicalityModel(
        k=5,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m.fit(X, identity_ids=ids)
    assert TypicalityScoreType.LOG_DENSITY_ESTIMATE in m.enabled_score_types
