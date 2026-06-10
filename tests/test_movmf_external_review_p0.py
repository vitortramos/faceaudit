"""Regression tests for P0 fixes from external review (2026-06-09).

Cobertura:
- P0-1: install_reference_score_calibration substitui referência atomicamente
        e ReferenceScoreCalibrationService chama-o no CROSS_FIT_IDENTITY.
- P0-2: get_percentiles consulta CDF ponderada quando presente.
- P0-3: serialização preserva pesos + CDF + proveniência.
- P0-4: OOF da CDF respeita population_weighting_mode (INV_ID_FREQ pondera
        por 1/n_i identidade, IDENTITY_CENTROIDS agrega por identidade).
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)


def _make_data(n_id=10, per=20, d=16, seed=0):
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


# ─── P0-1 ────────────────────────────────────────────────────────────────


def test_movmf_install_replaces_in_sample_reference_atomically():
    """Após install_reference_score_calibration o estado in-sample do fit
    é substituído integralmente — get_percentiles e is_in_distribution
    consultam a referência nova."""
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none", ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    # Snapshot do estado in-sample
    in_sample_ref = m.reference_log_likelihoods.copy()
    in_sample_th = m.ood_threshold

    # Instala referência arbitrária
    oof_scores = np.linspace(-5, 5, 100)
    oof_weights = np.ones(100)
    oof_threshold = -3.5
    m.install_reference_score_calibration(
        reference_scores=oof_scores,
        reference_weights=oof_weights,
        ood_threshold=oof_threshold,
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        split_fingerprint="abc123",
    )

    # Estado in-sample foi substituído
    assert not np.array_equal(m.reference_log_likelihoods, in_sample_ref)
    assert m.ood_threshold != in_sample_th
    assert m.ood_threshold == pytest.approx(oof_threshold)
    assert (
        m._reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )
    assert m._split_fingerprint == "abc123"


def test_service_cross_fit_installs_oof_into_refit_model():
    """Bug #1 do revisor: result.ood_threshold == refit_model.ood_threshold
    e result.oof_cdf == refit_model.reference_log_likelihoods."""
    X, ids = _make_data(n_id=12, per=15)

    def trainer():
        return MoVMFTypicalityModel(
            n_components=2, n_init=2, centering="none", ood_quantile=0.1,
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=4, random_state=42,
    )
    assert result.ood_threshold == result.refit_model.ood_threshold
    assert np.array_equal(
        result.oof_cdf, result.refit_model.reference_log_likelihoods
    )
    assert (
        result.refit_model._reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )
    assert result.refit_model._split_fingerprint == result.split_fingerprint


def test_knn_install_replaces_reference_atomically():
    """KNN: análogo do MoVMF — install substitui o `_state` frozen."""
    X, ids = _make_data(n_id=12, per=15)
    m = KNNTypicalityModel(
        k=10, ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    in_sample_sorted = m._state.loo_mean_knn_sorted.copy()
    in_sample_th = m._state.ood_threshold

    oof_scores = np.linspace(0, 1, 50)
    oof_weights = np.ones(50)
    m.install_reference_score_calibration(
        reference_scores=oof_scores,
        reference_weights=oof_weights,
        ood_threshold=0.3,
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        split_fingerprint="def456",
    )
    assert not np.array_equal(m._state.loo_mean_knn_sorted, in_sample_sorted)
    assert m._state.ood_threshold != in_sample_th
    assert m._state.ood_threshold == pytest.approx(0.3)
    assert (
        m._state.reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )
    assert m._state.split_fingerprint == "def456"


# ─── P0-2 ────────────────────────────────────────────────────────────────


def test_movmf_get_percentiles_uses_weighted_cdf():
    """Bug #2 do revisor: com INV_ID_FREQ ativo, get_percentiles deve
    produzir percentil diferente do rank uniforme."""
    rng = np.random.default_rng(0)
    # Identidade altamente desbalanceada: id_0 = 80 imgs, demais = 5
    X, ids = [], []
    d = 16
    for i in range(10):
        c = rng.normal(size=d)
        c /= np.linalg.norm(c)
        n = 80 if i == 0 else 5
        for _ in range(n):
            v = c + 0.2 * rng.normal(size=d)
            v /= np.linalg.norm(v)
            X.append(v)
            ids.append(f"id_{i}")
    X = np.asarray(X)
    ids = tuple(ids)

    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none", ood_quantile=0.05,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    assert m._train_log_likelihoods_cdf is not None

    test = X[:20]
    pcts_weighted = m.get_percentiles(test)

    # Computa o que get_percentiles devolveria com rank uniforme
    ll = m.score_samples(test)
    ref = m._train_log_likelihoods
    left = np.searchsorted(ref, ll, side="left")
    right = np.searchsorted(ref, ll, side="right")
    mid_rank = (left + right) / 2.0
    pcts_uniform = 100.0 * mid_rank / len(ref)

    # Espera-se diferença mensurável (>1pp em pelo menos uma célula)
    assert np.max(np.abs(pcts_weighted - pcts_uniform)) > 1.0


# ─── P0-3 ────────────────────────────────────────────────────────────────


def test_movmf_serialization_preserves_weighted_state():
    """Round-trip preserva pesos + CDF + proveniência da referência."""
    X, ids = _make_data(n_id=8, per=10)
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none", ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    # Instala referência com pesos não-uniformes.
    # R7: CROSS_FIT_IDENTITY exige bloco OOF completo + score_type
    # esperado por backend (LOG_DENSITY_ESTIMATE no MoVMF).
    rng = np.random.default_rng(0)
    oof = rng.normal(size=50)
    w = rng.uniform(0.5, 2.0, size=50)
    m.install_reference_score_calibration(
        reference_scores=oof,
        reference_weights=w,
        ood_threshold=float(np.quantile(oof, 0.1)),
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        # R8 exige split_fingerprint no formato SHA-256 hex (64 chars).
        split_fingerprint="a" * 64,
        # R8: avg deve bater com média ponderada dos scores
        # persistidos (oof + w); n_scores_by_fold == n_identities_by_fold
        # em IDENTITY_CENTROIDS; soma de n_scores_by_fold == len(oof).
        oof_predictive_avg_score=float((w * oof).sum() / w.sum()),
        oof_predictive_avg_score_by_fold=(-1.0, -0.5, -0.8, -0.4),
        oof_predictive_score_type="LOG_DENSITY_ESTIMATE",
        n_identities_by_fold=(13, 13, 12, 12),
        n_scores_by_fold=(13, 13, 12, 12),
        ood_threshold_by_fold=(-1.5, -1.4, -1.3, -1.2),
        fold_seeds=(0, 1, 2, 3),
        n_folds_used=4,
    )

    test = X[:5]
    pcts_before = m.get_percentiles(test)
    sf_before = m.model_state_fingerprint
    cf_before = m.model_contract_fingerprint

    d = m.to_dict()
    assert d["format_version"] == 4
    assert d["state"]["train_log_likelihoods_weights"] is not None
    assert d["state"]["train_log_likelihoods_cdf"] is not None
    assert d["state"]["reference_score_calibration_mode"] == "CROSS_FIT_IDENTITY"
    assert d["state"]["split_fingerprint"] == "a" * 64

    m2 = MoVMFTypicalityModel.from_dict(d)
    pcts_after = m2.get_percentiles(test)
    assert np.allclose(pcts_before, pcts_after)
    assert m2.model_state_fingerprint == sf_before
    assert m2.model_contract_fingerprint == cf_before
    assert (
        m2._reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )


def test_movmf_serialization_back_compat_v3():
    """Pickles v3 (sem campos novos) carregam com defaults seguros."""
    X, ids = _make_data(n_id=6, per=8)
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none", ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
    )
    m.fit(X, identity_ids=ids)

    d = m.to_dict()
    # Simula pickle v3 (PR 4/5): bumpa format_version pra trás + remove
    # campos novos
    d["format_version"] = 3
    for k in (
        "train_log_likelihoods_weights",
        "train_log_likelihoods_cdf",
        "reference_score_calibration_mode",
        "split_fingerprint",
        "n_input_faces",
        "n_identities_inv_idfreq",
    ):
        d["state"].pop(k, None)

    m2 = MoVMFTypicalityModel.from_dict(d)
    # Carrega sem erro; campos novos são None
    assert m2._reference_score_calibration_mode is None
    assert m2._split_fingerprint is None
    assert m2._train_log_likelihoods_weights is None
    assert m2._train_log_likelihoods_cdf is None


# ─── P0-4 ────────────────────────────────────────────────────────────────


def test_cross_fit_inv_id_freq_weights_by_identity():
    """INVERSE_IDENTITY_FREQUENCY: pesos OOF derivados de 1/n_i por face."""
    # Dataset desbalanceado: id_0 com 30 imgs, demais com 5
    rng = np.random.default_rng(0)
    X, ids = [], []
    d = 16
    for i in range(8):
        c = rng.normal(size=d)
        c /= np.linalg.norm(c)
        n = 30 if i == 0 else 5
        for _ in range(n):
            v = c + 0.2 * rng.normal(size=d)
            v /= np.linalg.norm(v)
            X.append(v)
            ids.append(f"id_{i}")
    X = np.asarray(X)
    ids = tuple(ids)

    def trainer():
        return MoVMFTypicalityModel(
            n_components=2, n_init=2, centering="none", ood_quantile=0.1,
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=4, random_state=0,
    )
    weights = result.refit_model._train_log_likelihoods_weights
    assert weights is not None
    # Pesos não-uniformes (assertion forte: std não-zero)
    assert weights.std() > 1e-6


# ─── P0-5 ────────────────────────────────────────────────────────────────


def test_summary_canonical_returns_dto():
    """`summary()` agora retorna TypicalityModelSummary (cumpre o protocolo)."""
    from faceaudit.models.typicality.typicality_model_summary import (
        TypicalityModelSummary,
    )
    X, ids = _make_data(n_id=6, per=8)
    knn = KNNTypicalityModel(
        k=5, ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    knn.fit(X, identity_ids=ids)
    assert isinstance(knn.summary(), TypicalityModelSummary)
    # Back-compat: summary_legacy() ainda devolve dict
    assert isinstance(knn.summary_legacy(), dict)

    mv = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none", ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    mv.fit(X, identity_ids=ids)
    assert isinstance(mv.summary(), TypicalityModelSummary)
    assert isinstance(mv.summary_legacy(), dict)


# ─── P0-6 ────────────────────────────────────────────────────────────────


def test_batch_fingerprint_invariant_to_superfluous_metadata():
    """Batches semanticamente equivalentes têm o mesmo fingerprint, mesmo
    quando o caller passa campos supérfluos (percentile_basis em score
    não-percentil) — esses são limpos ANTES do hash."""
    from faceaudit.models.typicality.typicality_score_batch import (
        TypicalityScoreBatch,
    )
    from faceaudit.enums.typicality_score_type import TypicalityScoreType

    # P1-R2-9 (round 2): fingerprints obrigatórios.
    sfp = "aabb"
    cfp = "ccdd"
    arr = np.array([0.1, 0.2, 0.3])
    b_clean = TypicalityScoreBatch(
        scores=arr, n=3, embedding_dim=64,
        score_type=TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        score_semantics_id="LOG_LIKELIHOOD_MOVMF",
        model_type=TypicalityModelType.MOVMF,
        model_state_fingerprint=sfp, model_contract_fingerprint=cfp,
    )
    b_superfluous = TypicalityScoreBatch(
        scores=arr, n=3, embedding_dim=64,
        score_type=TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        score_semantics_id="LOG_LIKELIHOOD_MOVMF",
        percentile_basis_score_type=TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        reference_score_distribution_fingerprint="ignored",
        model_type=TypicalityModelType.MOVMF,
        model_state_fingerprint=sfp, model_contract_fingerprint=cfp,
    )
    assert b_clean.batch_fingerprint == b_superfluous.batch_fingerprint


def test_batch_fingerprint_rejects_external_mismatch():
    """Caller passando batch_fingerprint divergente do canônico — rejeita."""
    from faceaudit.models.typicality.typicality_score_batch import (
        TypicalityScoreBatch,
    )
    from faceaudit.enums.typicality_score_type import TypicalityScoreType

    arr = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="diverge do canônico"):
        TypicalityScoreBatch(
            scores=arr, n=3, embedding_dim=64,
            score_type=TypicalityScoreType.LOG_DENSITY_ESTIMATE,
            score_semantics_id="LOG_LIKELIHOOD_MOVMF",
            model_type=TypicalityModelType.MOVMF,
            model_state_fingerprint="aa", model_contract_fingerprint="bb",
            batch_fingerprint="hash-falso",
        )


def test_cross_fit_identity_centroids_aggregates_per_identity():
    """IDENTITY_CENTROIDS: OOF agrega para 1 score por identidade."""
    X, ids = _make_data(n_id=12, per=10)

    def trainer():
        return MoVMFTypicalityModel(
            n_components=2, n_init=2, centering="none", ood_quantile=0.1,
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=4, random_state=0,
    )
    # 12 identidades únicas → CDF OOF tem 12 entradas
    assert len(result.oof_cdf) == 12
    weights = result.refit_model._train_log_likelihoods_weights
    assert weights is not None
    # IDENTITY_CENTROIDS: pesos uniformes (1 por identidade)
    assert np.all(weights == 1.0)
