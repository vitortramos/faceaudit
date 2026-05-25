# tests/test_shrinkage_two_covariance_plda.py

"""
Testes da variante PLDA SHRINKAGE_TWO_COVARIANCE.

Cobertura:
- fit produz state com metadados de auditoria populados
- modo NONE reproduz TWO_COVARIANCE puro (W e B numericamente idênticas)
- α=0 manual ↔ W empírica (sem mudança)
- α=1 manual ↔ W colapsa para target scaled-identity
- score simétrico (LR(x1,x2) == LR(x2,x1))
- save/load preserva metadados de shrinkage
- Ledoit-Wolf retorna sempre α ∈ [0, 1]
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from faceaudit.enums.covariance_shrinkage_mode import CovarianceShrinkageMode
from faceaudit.enums.plda_variant import PLDAVariant
from faceaudit.models.similarity.plda_similarity_model import PLDASimilarityModel
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.services.shrinkage_two_covariance_plda_service import (
    apply_shrinkage_to_w,
    fit_shrinkage_two_covariance,
    ledoit_wolf_alpha,
)
from faceaudit.services.two_covariance_plda_service import (
    fit_two_covariance,
    score_two_covariance_pair,
)


def _make_synthetic(n_ids: int = 30, n_per_id: int = 8, D: int = 32, seed: int = 0):
    """
    Gera embeddings sintéticos com `n_ids` identidades e `n_per_id` amostras
    cada. Cada identidade tem centro próprio sorteado de N(0, I); amostras
    são desvios pequenos N(centro, 0.3·I). Suficiente para PLDA fitar B e W
    sem degeneração.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(n_ids, D))
    X_list = []
    y_list = []
    for i, c in enumerate(centers):
        samples = c + rng.normal(0.0, 0.3, size=(n_per_id, D))
        X_list.append(samples)
        y_list.extend([f"id_{i:03d}"] * n_per_id)
    X = np.vstack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=object)
    return X, y


def test_fit_shrinkage_two_covariance_basic():
    X, y = _make_synthetic(seed=1)
    state = fit_shrinkage_two_covariance(
        X=X, y=y,
        shrinkage_mode=CovarianceShrinkageMode.LEDOIT_WOLF,
    )
    assert state.shrinkage_mode == "LEDOIT_WOLF"
    assert state.shrinkage_target == "SCALED_IDENTITY"
    assert state.shrinkage_alpha_w is not None
    assert 0.0 <= state.shrinkage_alpha_w <= 1.0
    assert state.shrinkage_alpha_b is None
    assert state.W.shape == (X.shape[1], X.shape[1])
    assert state.B.shape == (X.shape[1], X.shape[1])


def test_shrinkage_none_matches_two_covariance():
    X, y = _make_synthetic(seed=2)
    state_none = fit_shrinkage_two_covariance(
        X=X, y=y,
        shrinkage_mode=CovarianceShrinkageMode.NONE,
        regularization=0.0,
    )
    state_pure = fit_two_covariance(X=X, y=y, regularization=0.0)

    # Mesma estimativa empírica de W e B.
    assert np.allclose(state_none.W, state_pure.W)
    assert np.allclose(state_none.B, state_pure.B)
    assert np.allclose(state_none.mu, state_pure.mu)
    # α aplicado é 0.
    assert state_none.shrinkage_alpha_w == 0.0


def test_manual_alpha_zero_matches_empirical():
    X, y = _make_synthetic(seed=3)
    state = fit_shrinkage_two_covariance(
        X=X, y=y,
        shrinkage_mode=CovarianceShrinkageMode.MANUAL,
        shrinkage_intensity=0.0,
        regularization=0.0,
    )
    state_pure = fit_two_covariance(X=X, y=y, regularization=0.0)
    assert np.allclose(state.W, state_pure.W)
    assert state.shrinkage_alpha_w == 0.0


def test_manual_alpha_one_matches_target():
    X, y = _make_synthetic(seed=4)
    state = fit_shrinkage_two_covariance(
        X=X, y=y,
        shrinkage_mode=CovarianceShrinkageMode.MANUAL,
        shrinkage_intensity=1.0,
        regularization=0.0,
    )
    D = state.D
    # Computa W_emp pra obter tr(W_emp)/D — o target depende disso, então
    # precisamos do W empírico (não-shrunkado) para comparar.
    state_pure = fit_two_covariance(X=X, y=y, regularization=0.0)
    mu_trace = np.trace(state_pure.W) / D
    expected = mu_trace * np.eye(D)
    assert np.allclose(state.W, expected, atol=1e-8)
    assert state.shrinkage_alpha_w == 1.0


def test_score_symmetry():
    X, y = _make_synthetic(seed=5)
    state = fit_shrinkage_two_covariance(
        X=X, y=y,
        shrinkage_mode=CovarianceShrinkageMode.LEDOIT_WOLF,
    )
    rng = np.random.default_rng(99)
    v1 = rng.normal(0.0, 1.0, size=state.D)
    v2 = rng.normal(0.0, 1.0, size=state.D)
    lr12 = score_two_covariance_pair(state, v1, v2)
    lr21 = score_two_covariance_pair(state, v2, v1)
    assert np.isclose(lr12, lr21, atol=1e-10)


def test_persistence_roundtrip():
    X, y = _make_synthetic(seed=6)
    # Constrói o modelo via PLDASimilarityModel.fit (caminho real do código).
    config = PLDASimilarityModelConfig(
        plda_variant=PLDAVariant.SHRINKAGE_TWO_COVARIANCE,
        shrinkage_mode=CovarianceShrinkageMode.LEDOIT_WOLF,
    )
    model = PLDASimilarityModel(config=config, name="test_shrinkage")

    # Constrói um EmbeddingDataset mínimo via API pública.
    from faceaudit.models.embedding_dataset import EmbeddingDataset
    from faceaudit.models.embedding_info import EmbeddingInfo

    items = [
        EmbeddingInfo(
            embedding=X[i],
            face_hash=f"hash_{i:04d}",
            image_path=f"img_{i}.jpg",
            identity=str(y[i]),
            file_name=f"img_{i}.jpg",
        )
        for i in range(X.shape[0])
    ]
    dataset = EmbeddingDataset(items=items, name="test_dataset")

    model.fit(dataset, logger_callback=lambda msg: None)
    assert model.is_fitted
    alpha_before = model.two_cov_state.shrinkage_alpha_w

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pkl")
        model.save(path)

        loaded = PLDASimilarityModel()
        loaded.load(path)

    assert loaded.is_fitted
    assert loaded.two_cov_state.shrinkage_mode == "LEDOIT_WOLF"
    assert loaded.two_cov_state.shrinkage_target == "SCALED_IDENTITY"
    assert loaded.two_cov_state.shrinkage_alpha_w == pytest.approx(alpha_before)
    assert loaded.two_cov_state.shrinkage_alpha_b is None
    assert np.allclose(loaded.two_cov_state.W, model.two_cov_state.W)
    assert np.allclose(loaded.two_cov_state.B, model.two_cov_state.B)


def test_ledoit_wolf_alpha_bounded():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        D = 16
        N = 200
        # Residuais sintéticos (média ~0).
        R = rng.normal(0.0, 1.0, size=(N, D))
        alpha = ledoit_wolf_alpha(R)
        assert 0.0 <= alpha <= 1.0, f"seed={seed}: α={alpha} fora de [0, 1]"


def test_apply_shrinkage_to_w_none_returns_input():
    rng = np.random.default_rng(7)
    D = 8
    W = rng.normal(0.0, 1.0, size=(D, D))
    W = W @ W.T  # positiva definida
    R = rng.normal(0.0, 1.0, size=(20, D))
    W_out, alpha = apply_shrinkage_to_w(
        W, R, mode=CovarianceShrinkageMode.NONE, manual_intensity=0.5
    )
    assert alpha == 0.0
    assert np.allclose(W_out, W)
