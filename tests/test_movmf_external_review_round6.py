"""Regression tests para os 5 P0 + 5 P1 da sexta rodada da revisão
externa (2026-06-09 round 6).

P0:
- R6-1: contrato cross-fit cobre hiperparâmetros de treino (k,
        n_components, centering, ood_quantile, unit_norm_policy, ...).
        Factory alternando `k`/`n_components` entre folds → erro.
- R6-2: `audit_metrics_fingerprint=None` em payload OOF presente →
        `MissingOofAuditMetricFingerprintError`.
- R6-3: `from_dict` re-aplica validadores de NaN + comprimento + fingerprint.
- R6-4: envelope da factory exige model_state e model_contract
        fingerprints (não pode ser bypassed removendo o bloco).
- R6-5: colisão de identidades após coerção `str(i)` →
        `IdentityCanonicalizationCollisionError`.

P1:
- R6-6: floats fracionários em contagens/seeds rejeitados.
- R6-7: CDF sem weights rejeitada (formato canônico).
- R6-8: `reference_point` do MoVMF validado no `from_dict`.
- R6-9: contrato re-validado pós-fit (cobre backends futuros).
- R6-10: comentários editoriais (auditado por inspeção).
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
from faceaudit.services.oof_audit_metrics import (
    MissingOofAuditMetricFingerprintError,
)
from faceaudit.services.reference_score_calibration_service import (
    CrossFitContractMismatchError,
    IdentityCanonicalizationCollisionError,
    ReferenceScoreCalibrationService,
)
from faceaudit.services.reference_score_distribution_validator import (
    InvalidReferenceDistributionError,
    validate_reference_score_distribution,
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


# ─── P0-R6-1 ───────────────────────────────────────────────────────────


def test_cross_fit_rejects_alternating_k_in_knn_factory():
    """k-NN factory alternando k=2 e k=5 entre folds deve falhar."""
    X, ids = _make_data()
    call_count = [0]

    def alternating_factory(*, seed=None):
        i = call_count[0]
        call_count[0] += 1
        k_value = 2 if i % 2 == 0 else 5
        return KNNTypicalityModel(
            k=k_value, ood_quantile=0.0,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        )

    with pytest.raises(
        CrossFitContractMismatchError, match=r"training_contract\.k",
    ):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.KNN,
            trainer_factory=alternating_factory,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=0,
        )


def test_cross_fit_rejects_alternating_n_components_in_movmf_factory():
    """MoVMF alternando n_components=1 e n_components=2 deve falhar."""
    X, ids = _make_data()
    call_count = [0]

    def alternating_factory(*, seed=None):
        i = call_count[0]
        call_count[0] += 1
        K = 1 if i % 2 == 0 else 2
        return MoVMFTypicalityModel(
            n_components=K, ood_quantile=0.0,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            random_state=seed if seed is not None else 0,
        )

    with pytest.raises(
        CrossFitContractMismatchError,
        match=r"training_contract\.n_components",
    ):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=alternating_factory,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=0,
        )


def test_cross_fit_rejects_alternating_ood_quantile_in_factory():
    """Mudança de `ood_quantile` entre folds também é proibida."""
    X, ids = _make_data()
    call_count = [0]

    def alternating_factory(*, seed=None):
        i = call_count[0]
        call_count[0] += 1
        q = 0.0 if i % 2 == 0 else 0.05
        return MoVMFTypicalityModel(
            n_components=1, ood_quantile=q,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            random_state=seed if seed is not None else 0,
        )

    with pytest.raises(
        CrossFitContractMismatchError,
        match=r"training_contract\.ood_quantile",
    ):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=alternating_factory,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=0,
        )


# ─── P0-R6-2 ───────────────────────────────────────────────────────────


def test_audit_fingerprint_removal_bypass_blocked():
    """`audit_metrics_fingerprint=None` em payload com métricas OOF
    presente → MissingOofAuditMetricFingerprintError (não silencioso)."""
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
    payload = result.refit_model.to_dict()
    # Demonstração canônica do revisor:
    payload["state"]["oof_predictive_avg_score"] += 999.0
    payload["state"]["audit_metrics_fingerprint"] = None
    with pytest.raises(MissingOofAuditMetricFingerprintError):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R6-3 ───────────────────────────────────────────────────────────


def test_from_dict_rejects_nan_with_recomputed_fingerprint():
    """Adulteração com fingerprint recalculado para esconder NaN deve
    falhar pela revalidação estrutural em from_dict."""
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
    payload = result.refit_model.to_dict()
    payload["state"]["oof_predictive_avg_score"] = float("nan")
    # Recalcula fingerprint malicioso para esconder o NaN:
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    payload["state"]["audit_metrics_fingerprint"] = (
        _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=float("nan"),
            oof_predictive_avg_score_by_fold=tuple(
                payload["state"]["oof_predictive_avg_score_by_fold"]
            ),
            oof_predictive_score_type=payload["state"]["oof_predictive_score_type"],
            n_identities_by_fold=tuple(
                payload["state"]["n_identities_by_fold"]
            ),
            n_scores_by_fold=tuple(
                payload["state"]["n_scores_by_fold"]
            ),
            ood_threshold_by_fold=tuple(
                payload["state"]["ood_threshold_by_fold"]
            ),
            split_fingerprint=payload["state"]["split_fingerprint"],
            fold_seeds=tuple(payload["state"]["fold_seeds"]),
            n_folds_used=payload["state"]["n_folds_used"],
        )
    )
    with pytest.raises(ValueError, match="oof_predictive_avg_score"):
        MoVMFTypicalityModel.from_dict(payload)


def test_from_dict_rejects_length_mismatch_with_recomputed_fingerprint():
    """`n_folds_used=99` com 3 folds reais — re-validação detecta."""
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
    payload = result.refit_model.to_dict()
    payload["state"]["n_folds_used"] = 99
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    payload["state"]["audit_metrics_fingerprint"] = (
        _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=payload["state"]["oof_predictive_avg_score"],
            oof_predictive_avg_score_by_fold=tuple(
                payload["state"]["oof_predictive_avg_score_by_fold"]
            ),
            oof_predictive_score_type=payload["state"]["oof_predictive_score_type"],
            n_identities_by_fold=tuple(
                payload["state"]["n_identities_by_fold"]
            ),
            n_scores_by_fold=tuple(
                payload["state"]["n_scores_by_fold"]
            ),
            ood_threshold_by_fold=tuple(
                payload["state"]["ood_threshold_by_fold"]
            ),
            split_fingerprint=payload["state"]["split_fingerprint"],
            fold_seeds=tuple(payload["state"]["fold_seeds"]),
            n_folds_used=99,
        )
    )
    with pytest.raises(ValueError, match="n_folds_used"):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R6-4 ───────────────────────────────────────────────────────────


def test_factory_envelope_requires_state_fingerprint():
    """`TypicalityModelFactory.from_dict` rejeita envelope sem
    `model_state_fingerprint`."""
    from faceaudit.services.typicality_model_factory import (
        TypicalityModelFactory,
    )

    X, _ = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    envelope = TypicalityModelFactory.to_dict(m)
    # Bypass: remove o fingerprint de state.
    envelope["fingerprints"].pop("model_state_fingerprint", None)
    with pytest.raises(ValueError, match="model_state_fingerprint"):
        TypicalityModelFactory.from_dict(envelope)


def test_factory_envelope_requires_contract_fingerprint():
    from faceaudit.services.typicality_model_factory import (
        TypicalityModelFactory,
    )

    X, _ = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["fingerprints"].pop("model_contract_fingerprint", None)
    with pytest.raises(ValueError, match="model_contract_fingerprint"):
        TypicalityModelFactory.from_dict(envelope)


# ─── P0-R6-5 ───────────────────────────────────────────────────────────


def test_cross_fit_rejects_identity_canonicalization_collision():
    """ids como int(1) e str('1') colidem após coerção — erro."""
    rng = np.random.default_rng(0)
    n_per_id, d = 8, 16
    X = []
    ids = []
    centers = [rng.normal(size=d) for _ in range(4)]
    for c in centers:
        c /= np.linalg.norm(c)
    # Identidades: int 1, str "1", int 2, str "2"
    raw_ids = [1, "1", 2, "2"]
    for orig_id, center in zip(raw_ids, centers):
        for _ in range(n_per_id):
            v = center + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(orig_id)
    X = np.asarray(X)
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    with pytest.raises(IdentityCanonicalizationCollisionError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory,
            embeddings=X, identity_ids=tuple(ids),
            n_folds=2, random_state=0,
        )


# ─── P1-R6-6 ───────────────────────────────────────────────────────────


def test_install_rejects_fractional_floats_in_by_fold():
    X, _ = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    with pytest.raises(ValueError, match="n_identities_by_fold"):
        m.install_reference_score_calibration(
            reference_scores=np.array([-1.0, 0.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_predictive_avg_score=-2.0,
            oof_predictive_avg_score_by_fold=(-2.0, -1.0, -3.0),
            n_identities_by_fold=(1.9, 2, 3),  # 1.9 NÃO é inteiro
            n_folds_used=3,
        )


def test_install_rejects_fractional_n_folds_used():
    X, _ = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    with pytest.raises(ValueError, match="n_folds_used"):
        m.install_reference_score_calibration(
            reference_scores=np.array([-1.0, 0.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            n_folds_used=2.5,
        )


# ─── P1-R6-7 ───────────────────────────────────────────────────────────


def test_validator_rejects_cdf_without_weights():
    scores = np.array([0.1, 0.5, 0.9])
    cdf = np.array([1.0 / 3, 2.0 / 3, 1.0])
    with pytest.raises(InvalidReferenceDistributionError, match="weights"):
        validate_reference_score_distribution(
            scores=scores, weights=None, cdf=cdf,
        )


# ─── P1-R6-8 ───────────────────────────────────────────────────────────


def test_from_dict_rejects_reference_point_with_wrong_dim():
    X, _ = _make_data(d=16)
    m = MoVMFTypicalityModel(
        n_components=1, centering="euclidean_legacy",
        ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    payload = m.to_dict()
    # Substitui por vetor de dimensão errada.
    payload["state"]["reference_point"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="reference_point"):
        MoVMFTypicalityModel.from_dict(payload)


def test_from_dict_rejects_reference_point_with_nan():
    X, _ = _make_data(d=16)
    m = MoVMFTypicalityModel(
        n_components=1, centering="euclidean_legacy",
        ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    payload = m.to_dict()
    bad = [float("nan")] * 16
    payload["state"]["reference_point"] = bad
    with pytest.raises(ValueError, match="reference_point"):
        MoVMFTypicalityModel.from_dict(payload)
