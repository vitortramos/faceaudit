"""V2.13 — STRICT_IDENTITY em multivariada + standardização per-fold.

Cobre:
- `STRICT_IDENTITY` aplicado a X multivariado não cai mais em fallback;
- `IdentityDisjointPairKFold` é usado como CV interno do GridSearchCV;
- `mode_applied` permanece `STRICT_IDENTITY`, `disjointness_level` = `full`;
- Standardização vira per-fold via `sklearn.Pipeline` (StandardScaler +
  LogReg) quando `standardize_features=True` e D > 1;
- Após refit, `self._feature_mean` / `self._feature_std` carregam μ/σ
  extraídos da scaler treinada (sobre D_cal completo) — match com
  `X.mean(axis=0)` / `X.std(axis=0, ddof=0)`;
- `self.logreg` permanece como `LogisticRegression` (não Pipeline) —
  backward-compat para downstream que acessa `.coef_`/`.intercept_`;
- Em D=1, ou com `standardize_features=False`, μ/σ continuam None;
- Round-trip save/load preserva o estado (mean/std + logreg).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from faceaudit.enums.logreg_gridsearch_cv_mode import LogRegGridSearchCVMode
from faceaudit.models.calibration.logistic_regression_calibration_model import (
    LogisticRegressionCalibrationModel,
)
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)


def _synth_multivariate(
    n_ids: int = 30,
    n_pairs: int = 200,
    n_features: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Gera X multivariado + labels balanceadas + listas de identidades.

    Mantém Hp e Hd em proporções equilibradas e injeta sinal só na
    feature 0 para evitar degeneração.
    """
    rng = np.random.default_rng(seed)
    ids = [f"id_{i:03d}" for i in range(n_ids)]
    probe_ids = rng.choice(ids, size=n_pairs).tolist()
    labels = rng.integers(0, 2, size=n_pairs)
    ref_ids: List[str] = []
    for p, lab in zip(probe_ids, labels):
        if lab == 1:
            ref_ids.append(p)
        else:
            cand = [i for i in ids if i != p]
            ref_ids.append(rng.choice(cand))
    X = rng.normal(size=(n_pairs, n_features))
    X[:, 0] += np.where(labels == 1, 1.0, -1.0)
    return X, labels, probe_ids, ref_ids


def _silent(_msg: str) -> None:
    return None


class TestStrictIdentityMultivariate:
    def test_strict_identity_no_fallback_on_multivariate(self) -> None:
        X, y, p_ids, r_ids = _synth_multivariate(n_features=4, seed=1)
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[0.1, 1.0, 10.0],
            cv_folds=3,
            gridsearch_cv_mode=LogRegGridSearchCVMode.STRICT_IDENTITY,
            standardize_features=False,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.metadata["train_probe_identities"] = p_ids
        model.metadata["train_reference_identities"] = r_ids
        model.fit(X, y, logger_callback=_silent)

        assert model.metadata["cv_split_mode_requested"] == "STRICT_IDENTITY"
        assert model.metadata["cv_split_mode_applied"] == "STRICT_IDENTITY"
        assert model.metadata["cv_split_mode_fallback_reason"] is None
        assert model.metadata["cv_identity_disjointness_level"] == "full"
        assert model.metadata["cv_identity_disjointness_verified"] is True
        assert model.metadata["cv_splitter_type"] == "IdentityDisjointPairKFold"
        # cv_strict_* metadata populated only no STRICT
        assert "cv_strict_id_to_group_mapping" in model.metadata
        assert "cv_strict_fold_health" in model.metadata

    def test_strict_identity_groups_disjoint_per_fold(self) -> None:
        """Reproduz folds dinâmicos e confirma disjunção total."""
        X, y, p_ids, r_ids = _synth_multivariate(n_features=3, seed=2)
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0],
            cv_folds=3,
            gridsearch_cv_mode=LogRegGridSearchCVMode.STRICT_IDENTITY,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.metadata["train_probe_identities"] = p_ids
        model.metadata["train_reference_identities"] = r_ids
        model.fit(X, y, logger_callback=_silent)

        mapping = model.metadata["cv_strict_id_to_group_mapping"]
        p_arr = np.asarray(p_ids)
        r_arr = np.asarray(r_ids)
        from faceaudit.services.identity_disjoint_pair_kfold import (
            IdentityDisjointPairKFold,
        )
        kf = IdentityDisjointPairKFold(
            probe_identities=p_ids,
            reference_identities=r_ids,
            n_splits=3,
            random_state=42,
            assignment_strategy="balanced",
        )
        for train_idx, test_idx in kf.split(X, y):
            train_ids = set(p_arr[train_idx]) | set(r_arr[train_idx])
            test_ids = set(p_arr[test_idx]) | set(r_arr[test_idx])
            assert train_ids.isdisjoint(test_ids), (
                "STRICT_IDENTITY exige train ∩ test = ∅"
            )

    def test_strict_identity_works_with_standardization(self) -> None:
        """STRICT_IDENTITY + standardize_features=True coexistem em D>1."""
        X, y, p_ids, r_ids = _synth_multivariate(n_features=4, seed=3)
        # Features de escalas dispares — caso típico para standardize.
        X[:, 0] *= 1000.0
        X[:, 1] *= 0.001
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0, 10.0],
            cv_folds=3,
            gridsearch_cv_mode=LogRegGridSearchCVMode.STRICT_IDENTITY,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.metadata["train_probe_identities"] = p_ids
        model.metadata["train_reference_identities"] = r_ids
        model.fit(X, y, logger_callback=_silent)

        assert model.metadata["cv_split_mode_applied"] == "STRICT_IDENTITY"
        assert model._feature_mean is not None
        assert model._feature_std is not None
        assert model._feature_mean.shape == (4,)
        assert isinstance(model.logreg, LogisticRegression)


class TestPerFoldStandardization:
    def test_scaler_state_matches_full_d_cal_after_refit(self) -> None:
        X, y, _, _ = _synth_multivariate(n_features=3, seed=4)
        X[:, 0] *= 100.0
        X[:, 1] *= 0.01
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0, 10.0],
            cv_folds=3,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)

        assert model._feature_mean is not None
        assert model._feature_std is not None
        np.testing.assert_allclose(
            model._feature_mean, X.mean(axis=0), atol=1e-9
        )
        np.testing.assert_allclose(
            model._feature_std, X.std(axis=0, ddof=0), atol=1e-9
        )

    def test_logreg_attribute_points_to_inner_logreg(self) -> None:
        """Backward-compat: downstream lê `model.logreg.coef_`."""
        X, y, _, _ = _synth_multivariate(n_features=3, seed=5)
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0],
            cv_folds=3,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)
        assert isinstance(model.logreg, LogisticRegression)
        assert not isinstance(model.logreg, Pipeline)
        assert hasattr(model.logreg, "coef_")
        assert hasattr(model.logreg, "intercept_")
        assert model.logreg.coef_.shape == (1, 3)

    def test_no_standardization_when_disabled(self) -> None:
        X, y, _, _ = _synth_multivariate(n_features=3, seed=6)
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0],
            cv_folds=3,
            standardize_features=False,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)
        assert model._feature_mean is None
        assert model._feature_std is None
        assert isinstance(model.logreg, LogisticRegression)

    def test_no_standardization_when_1d(self) -> None:
        """D=1: standardize_features=True não deve ativar scaler."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 1))
        y = rng.integers(0, 2, size=200)
        X[:, 0] += np.where(y == 1, 1.0, -1.0)
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0],
            cv_folds=3,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)
        assert model._feature_mean is None
        assert model._feature_std is None

    def test_no_gridsearch_branch_still_standardizes(self) -> None:
        """use_gridsearch=False também roda via Pipeline e popula μ/σ."""
        X, y, _, _ = _synth_multivariate(n_features=4, seed=8)
        X[:, 0] *= 1000.0
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=False,
            standardize_features=True,
            C=1.0,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)
        assert model._feature_mean is not None
        assert model._feature_std is not None
        np.testing.assert_allclose(
            model._feature_mean, X.mean(axis=0), atol=1e-9
        )


class TestInferenceConsistency:
    def test_calibrate_scores_applies_standardization(self) -> None:
        """`_features_to_X` standardiza usando μ/σ do refit."""
        X, y, _, _ = _synth_multivariate(n_features=3, seed=9)
        X[:, 0] *= 100.0
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0],
            cv_folds=3,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)

        # Manual standardization + decision_function deve bater com
        # `calibrate_scores`.
        X_z = (X[:10] - model._feature_mean) / model._feature_std
        manual_llr = model.logreg.decision_function(X_z)
        auto_llr = model.calibrate_scores(X[:10])
        np.testing.assert_allclose(auto_llr, manual_llr, atol=1e-10)


class TestPersistenceRoundTrip:
    def test_save_load_preserves_scaler_state(self) -> None:
        X, y, _, _ = _synth_multivariate(n_features=4, seed=10)
        X[:, 0] *= 50.0
        config = LogisticRegressionCalibrationModelConfig(
            use_gridsearch=True,
            c_grid=[1.0, 10.0],
            cv_folds=3,
            standardize_features=True,
        )
        model = LogisticRegressionCalibrationModel(name="t", config=config)
        model.fit(X, y, logger_callback=_silent)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "calib.pkl"
            model.save(str(path))
            restored = LogisticRegressionCalibrationModel(name="r")
            restored.load(str(path))

            assert restored._feature_mean is not None
            assert restored._feature_std is not None
            np.testing.assert_allclose(
                restored._feature_mean, model._feature_mean, atol=1e-12
            )
            np.testing.assert_allclose(
                restored._feature_std, model._feature_std, atol=1e-12
            )

            llr_orig = model.calibrate_scores(X[:10])
            llr_restored = restored.calibrate_scores(X[:10])
            np.testing.assert_allclose(llr_orig, llr_restored, atol=1e-12)
