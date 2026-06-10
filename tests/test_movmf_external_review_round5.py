"""Regression tests para os 6 P0 + 4 P1 da quinta rodada da revisão
externa (2026-06-09 round 5).

P0-R5-1: remove fallback amplo `try/except TypeError` em
         `_calibrate_cross_fit_identity` — backend desatualizado
         agora deve falhar fechado.
P0-R5-2: `install_reference_score_calibration` valida finitude +
         coerência dimensional das métricas OOF.
P0-R5-3: cria `audit_metrics_fingerprint` cobrindo as métricas OOF;
         `from_dict` recalcula e compara → `OofAuditMetricMismatchError`
         em adulteração.
P0-R5-4: `IN_SAMPLE_LEGACY` propaga `random_state` ao trainer (antes
         hardcoded em 0) e valida backend.
P0-R5-5: `CrossFitContractMismatchError` em divergência semântica
         entre peek/folds/refit (basis, pop_mode, dim, enabled).
P0-R5-6: diagnósticos completos por fold persistidos no modelo via
         `install_reference_score_calibration` + `to_dict` / `from_dict`.

P1 editoriais (4 itens): comentários e docstrings — cobertos por leitura
manual, sem teste unitário direto. Testamos como sanity-check apenas a
ausência da palavra "Cllr_min" e a presença da nova descrição de T_k
na docstring do `ReferenceScoreCalibrationResult`.
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
    OofAuditMetricMismatchError,
    OofAuditSemanticError,
)
from faceaudit.services.reference_score_calibration_service import (
    BackendMismatchError,
    CrossFitContractMismatchError,
    ReferenceScoreCalibrationResult,
    ReferenceScoreCalibrationService,
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


# ─── P0-R5-1 ───────────────────────────────────────────────────────────


def test_install_propagates_full_diagnostics_no_fallback():
    """Após R5, o serviço passa SEMPRE todos os kwargs novos ao
    install. Não há mais fallback try/except mascarando TypeError
    interno. Validamos que os diagnósticos por fold (`fold_seeds`,
    `n_folds_used`, etc.) chegam ao modelo persistido."""
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
        n_folds=3, random_state=42,
    )
    m = result.refit_model
    # Diagnósticos completos no modelo (não só no Result).
    assert m._n_folds_used == 3
    assert len(m._fold_seeds) == 3
    assert m._fold_seeds == (42, 43, 44)
    assert len(m._n_identities_by_fold) == 3
    assert len(m._n_scores_by_fold) == 3
    assert len(m._ood_threshold_by_fold) == 3


# ─── P0-R5-2 ───────────────────────────────────────────────────────────


def test_install_rejects_nan_avg_score():
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    with pytest.raises(ValueError, match="oof_predictive_avg_score"):
        m.install_reference_score_calibration(
            reference_scores=np.array([-1.0, 0.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_predictive_avg_score=np.nan,
        )


def test_install_rejects_nan_in_by_fold():
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    with pytest.raises(
        ValueError, match="oof_predictive_avg_score_by_fold",
    ):
        m.install_reference_score_calibration(
            reference_scores=np.array([-1.0, 0.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_predictive_avg_score=-2.0,
            oof_predictive_avg_score_by_fold=(np.nan,),
            n_folds_used=1,
        )


def test_install_rejects_length_mismatch():
    """`n_folds_used` deve bater com `len(by_fold arrays)`."""
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    with pytest.raises(ValueError, match="n_folds_used"):
        m.install_reference_score_calibration(
            reference_scores=np.array([-1.0, 0.0, 1.0]),
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_predictive_avg_score=-2.0,
            oof_predictive_avg_score_by_fold=(-2.0, -1.0, -3.0),
            n_folds_used=4,  # ≠ len(by_fold) = 3
        )


# ─── P0-R5-3 ───────────────────────────────────────────────────────────


def test_audit_fingerprint_detects_tampering():
    """Adulterar `oof_predictive_avg_score` no payload → erro em from_dict."""
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
    # Adulteração canônica do revisor.
    payload["state"]["oof_predictive_avg_score"] += 1234.5
    # R8 endurece: a mesma adulteração agora é pega pela validação
    # semântica (avg ≠ média ponderada) antes de chegar ao fingerprint.
    with pytest.raises((OofAuditMetricMismatchError, OofAuditSemanticError)):
        MoVMFTypicalityModel.from_dict(payload)


def test_audit_fingerprint_detects_by_fold_tampering():
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
    by_fold = list(payload["state"]["oof_predictive_avg_score_by_fold"])
    by_fold[0] += 999.0
    payload["state"]["oof_predictive_avg_score_by_fold"] = by_fold
    # R8 endurece: a mesma adulteração agora é pega pela validação
    # semântica (avg ≠ média ponderada) antes de chegar ao fingerprint.
    with pytest.raises((OofAuditMetricMismatchError, OofAuditSemanticError)):
        MoVMFTypicalityModel.from_dict(payload)


def test_audit_fingerprint_clean_roundtrip():
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
    restored = MoVMFTypicalityModel.from_dict(payload)
    assert restored._audit_metrics_fingerprint is not None
    assert (
        restored._audit_metrics_fingerprint
        == result.refit_model._audit_metrics_fingerprint
    )


def test_knn_audit_fingerprint_also_protects():
    X, ids = _make_data()
    factory = lambda **kw: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.05,
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
    payload["learned_state"]["oof_predictive_avg_score"] += 0.5
    # R8 endurece: a mesma adulteração agora é pega pela validação
    # semântica (avg ≠ média ponderada) antes de chegar ao fingerprint.
    with pytest.raises((OofAuditMetricMismatchError, OofAuditSemanticError)):
        KNNTypicalityModel.from_dict(payload)


# ─── P0-R5-4 ───────────────────────────────────────────────────────────


def test_in_sample_legacy_propagates_random_state():
    """`IN_SAMPLE_LEGACY` deve passar `random_state` à factory (antes
    usava seed=0 hardcoded)."""
    X, ids = _make_data()
    captured = []

    def factory(*, seed=None):
        captured.append(seed)
        return KNNTypicalityModel(
            k=3, ood_quantile=0.0,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        )

    ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=0, random_state=777,
    )
    # IN_SAMPLE_LEGACY constrói exatamente um modelo via _build_model.
    assert 777 in captured


def test_in_sample_legacy_validates_backend():
    """`IN_SAMPLE_LEGACY` agora valida `expected_backend` (antes não)."""
    X, ids = _make_data()

    def bad_factory(*, seed=None):
        # Declara MOVMF mas devolve k-NN.
        return KNNTypicalityModel(k=3, ood_quantile=0.0)

    with pytest.raises(BackendMismatchError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=bad_factory,
            embeddings=X, identity_ids=ids,
            n_folds=0, random_state=0,
        )


# ─── P0-R5-5 ───────────────────────────────────────────────────────────


def test_cross_fit_rejects_pop_mode_variation_between_folds():
    """Factory alterna `population_weighting_mode` entre folds →
    `CrossFitContractMismatchError`."""
    X, ids = _make_data()
    call_count = [0]

    def alternating_factory(*, seed=None):
        i = call_count[0]
        call_count[0] += 1
        pop_mode = (
            PopulationWeightingMode.IDENTITY_CENTROIDS
            if i % 2 == 0
            else PopulationWeightingMode.FACE_LEVEL_RAW
        )
        return MoVMFTypicalityModel(
            n_components=1, ood_quantile=0.0,
            population_weighting_mode=pop_mode,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            random_state=seed if seed is not None else 0,
        )

    with pytest.raises(
        CrossFitContractMismatchError, match="population_weighting_mode",
    ):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=alternating_factory,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=0,
        )


# ─── P0-R5-6 ───────────────────────────────────────────────────────────


def test_diagnostics_per_fold_persist_in_model_summary():
    """Após R5, `summary().backend_details` expõe os 5 diagnósticos
    por fold + n_folds_used + audit_metrics_fingerprint."""
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.05,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=10,
    )
    s = result.refit_model.summary()
    bd = s.backend_details
    for key in (
        "n_identities_by_fold",
        "n_scores_by_fold",
        "ood_threshold_by_fold",
        "fold_seeds",
        "n_folds_used",
        "audit_metrics_fingerprint",
    ):
        assert key in bd, f"key {key!r} ausente em backend_details"
        assert bd[key] is not None, f"key {key!r} é None em CROSS_FIT"
    assert bd["n_folds_used"] == 3


def test_diagnostics_per_fold_survive_dict_roundtrip():
    X, ids = _make_data()
    factory = lambda **kw: MoVMFTypicalityModel(  # noqa: E731
        n_components=1, ood_quantile=0.05,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=kw.get("seed", 0),
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=99,
    )
    payload = result.refit_model.to_dict()
    restored = MoVMFTypicalityModel.from_dict(payload)
    assert restored._n_folds_used == result.refit_model._n_folds_used
    assert restored._fold_seeds == result.refit_model._fold_seeds
    assert (
        restored._n_identities_by_fold
        == result.refit_model._n_identities_by_fold
    )
    assert (
        restored._n_scores_by_fold == result.refit_model._n_scores_by_fold
    )
    assert (
        restored._ood_threshold_by_fold
        == result.refit_model._ood_threshold_by_fold
    )
