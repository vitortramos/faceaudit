"""Testes do KNNTypicalityModel.

Cobre:
- API básica (fit, get_typicality_info, batch, percentile, OOD).
- Comportamento estatístico: queries do "modo" são típicas (percentile alto);
  queries OOD são atípicas (percentile baixo, is_in_distribution=False).
- Self-match handling: query igual a um ponto da referência.
- Serialização: roundtrip to_dict/from_dict preserva todas as métricas.
- Validações: N < k+1, dimensão errada, embeddings degenerados, parâmetros
  fora de range.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.models.knn_typicality_info import KNNTypicalityInfo
from faceaudit.services.knn_typicality_model import KNNTypicalityModel


# ─────────────────────────── helpers ────────────────────────────


def _sample_vmf(n: int, mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """
    Amostragem aproximada de vMF via projeção. Suficiente pra gerar
    population de teste com concentração controlada — não precisa ser
    Wood (1994) exato pra esses testes.
    """
    d = mu.shape[0]
    # Amostra gaussiana isotrópica e empurra na direção de mu segundo kappa
    samples = rng.normal(size=(n, d))
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    # Combina linearmente: queremos média de cos(x, mu) ≈ kappa / (d + kappa)
    target_cos = kappa / (d - 1 + kappa)  # forma fechada da PS, suficiente como proxy
    coef = np.sqrt(max(0.0, 1.0 - target_cos ** 2))
    out = target_cos * mu[None, :] + coef * samples
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


# ─────────────────────────── construtor / validações ────────────


def test_construct_with_defaults():
    m = KNNTypicalityModel()
    assert m.k == 50
    assert m.ood_quantile == 0.01
    assert not m.is_fitted()


def test_construct_rejects_invalid_k():
    with pytest.raises(ValueError, match="k deve ser >= 1"):
        KNNTypicalityModel(k=0)


def test_construct_rejects_invalid_ood_quantile():
    with pytest.raises(ValueError, match="ood_quantile"):
        KNNTypicalityModel(ood_quantile=0.5)
    with pytest.raises(ValueError, match="ood_quantile"):
        KNNTypicalityModel(ood_quantile=-0.1)


# ─────────────────────────── fit / shape ────────────────────────


def test_fit_requires_enough_samples():
    m = KNNTypicalityModel(k=10)
    X = np.random.default_rng(0).normal(size=(5, 8))
    with pytest.raises(ValueError, match="k\\+1"):
        m.fit(X)


def test_fit_sets_internal_state():
    rng = np.random.default_rng(42)
    X = _normalize(rng.normal(size=(200, 32)))
    m = KNNTypicalityModel(k=20).fit(X, population_label="test")
    assert m.is_fitted()
    assert m.sample_count == 200
    assert m.embedding_dim == 32
    assert m.population_label == "test"
    assert m.ood_threshold is not None
    assert m.loo_distribution is not None
    assert m.loo_distribution.shape == (200,)


def test_fit_normalizes_unnormalized_input():
    """Input não-normalizado deve ser normalizado internamente."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 16)) * 5.0  # norma variável
    m = KNNTypicalityModel(k=10).fit(X)
    # O state.reference deve estar normalizado
    norms = np.linalg.norm(m._state.reference, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


# ─────────────────────────── inferência: shape e tipos ─────────


def test_get_typicality_info_single():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(200, 16)))
    m = KNNTypicalityModel(k=20).fit(X)
    info = m.get_typicality_info(X[0])
    assert isinstance(info, KNNTypicalityInfo)
    assert info.embedding_dim == 16
    assert info.k == 20
    assert info.sample_count == 200
    assert 0.0 <= info.typicality_percentile <= 100.0
    assert 0.0 <= info.angular_radius_to_kth_nn <= np.pi
    assert info.knn_similarities is not None
    assert len(info.knn_similarities) == 20


def test_batch_returns_correct_length():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(150, 16)))
    m = KNNTypicalityModel(k=10).fit(X)
    queries = _normalize(rng.normal(size=(7, 16)))
    infos = m.get_typicality_infos(queries)
    assert len(infos) == 7
    # default include_top_k=False — economiza memória
    assert all(i.knn_similarities is None for i in infos)


def test_score_samples_and_percentiles_shapes():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(150, 16)))
    m = KNNTypicalityModel(k=10).fit(X)
    queries = _normalize(rng.normal(size=(5, 16)))
    lds = m.score_samples(queries)
    pcts = m.get_percentiles(queries)
    ood = m.is_in_distribution(queries)
    assert lds.shape == (5,)
    assert pcts.shape == (5,)
    assert ood.shape == (5,) and ood.dtype == bool


# ─────────────────────────── comportamento estatístico ─────────


def test_typical_query_has_high_percentile():
    """
    Query no centro da população treinada (média geodésica aproximada)
    deve ter percentile alto.
    """
    rng = np.random.default_rng(7)
    d = 32
    mu = np.zeros(d)
    mu[0] = 1.0
    # Population concentrada em torno de e_1
    X = _sample_vmf(500, mu, kappa=20.0, rng=rng)
    m = KNNTypicalityModel(k=30).fit(X)

    # Query exatamente em mu (no centro)
    info_typical = m.get_typicality_info(mu)
    # Query antipodal (totalmente OOD)
    info_atypical = m.get_typicality_info(-mu)

    assert info_typical.typicality_percentile > 75.0, (
        f"Query no centro deveria ter percentile alto, got {info_typical.typicality_percentile}"
    )
    assert info_atypical.typicality_percentile < 5.0, (
        f"Query antipodal deveria ter percentile baixíssimo, got {info_atypical.typicality_percentile}"
    )
    # OOD detection
    assert info_typical.is_in_distribution
    assert not info_atypical.is_in_distribution


def test_log_density_is_higher_for_typical():
    rng = np.random.default_rng(8)
    d = 32
    mu = np.zeros(d)
    mu[0] = 1.0
    X = _sample_vmf(500, mu, kappa=20.0, rng=rng)
    m = KNNTypicalityModel(k=30).fit(X)

    ld_typical = m.get_typicality_info(mu).log_density_knn
    ld_atypical = m.get_typicality_info(-mu).log_density_knn
    assert ld_typical > ld_atypical, (
        f"log_density deveria ser maior pra query típica: typical={ld_typical}, atypical={ld_atypical}"
    )


def test_self_match_is_handled():
    """
    Query exatamente igual a um ponto da referência não deve "trapacear"
    inflando o T_k via auto-similaridade. Deve produzir percentile coerente
    com a CDF LOO (que também exclui self).
    """
    rng = np.random.default_rng(9)
    X = _normalize(rng.normal(size=(300, 32)))
    m = KNNTypicalityModel(k=30).fit(X)

    # Query é EXATAMENTE X[42]
    info = m.get_typicality_info(X[42])
    # O top similarity deve ser próximo do segundo maior na linha 42 da
    # matriz NxN (não 1.0 do auto)
    assert info.knn_similarities[0] < 0.999, (
        f"Self-match não foi excluído: top sim = {info.knn_similarities[0]}"
    )


def test_dim_mismatch_raises():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 32)))
    m = KNNTypicalityModel(k=10).fit(X)
    with pytest.raises(ValueError, match="Dimensão"):
        m.get_typicality_info(np.ones(16))


# ─────────────────────────── serialização ──────────────────────


def test_to_dict_from_dict_roundtrip():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m1 = KNNTypicalityModel(k=10, ood_quantile=0.02).fit(X, population_label="pop_A")

    data = m1.to_dict()
    m2 = KNNTypicalityModel.from_dict(data)

    assert m2.is_fitted()
    assert m2.k == m1.k
    assert m2.ood_quantile == m1.ood_quantile
    assert m2.population_label == m1.population_label
    assert m2.sample_count == m1.sample_count
    assert m2.embedding_dim == m1.embedding_dim

    # Resultados de inferência devem ser numericamente idênticos
    query = _normalize(rng.normal(size=(5, 16)))
    pcts1 = m1.get_percentiles(query)
    pcts2 = m2.get_percentiles(query)
    np.testing.assert_allclose(pcts1, pcts2)

    lds1 = m1.score_samples(query)
    lds2 = m2.score_samples(query)
    np.testing.assert_allclose(lds1, lds2)


def test_from_dict_rejects_unknown_version():
    bogus = {"format_version": 999, "hyperparameters": {}, "learned_state": {}}
    with pytest.raises(ValueError, match="Versão de formato"):
        KNNTypicalityModel.from_dict(bogus)


def test_from_dict_rejects_unnormalized_reference():
    """Defesa contra JSON corrompido: reference precisa estar L2-normalizada."""
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m = KNNTypicalityModel(k=10).fit(X)
    data = m.to_dict()
    # Corrompe a referência multiplicando por 3
    data["learned_state"]["reference"] = (
        np.asarray(data["learned_state"]["reference"]) * 3.0
    ).tolist()
    with pytest.raises(ValueError, match="L2-normalizada"):
        KNNTypicalityModel.from_dict(data)


def test_from_dict_rejects_inconsistent_shapes():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m = KNNTypicalityModel(k=10).fit(X)
    data = m.to_dict()
    # Trunca a CDF — fica inconsistente com a referência
    data["learned_state"]["loo_mean_knn_sorted"] = data["learned_state"][
        "loo_mean_knn_sorted"
    ][:50]
    with pytest.raises(ValueError, match="Inconsistência"):
        KNNTypicalityModel.from_dict(data)


# ─────────────────────────── summary / diagnóstico ─────────────


def test_summary_unfitted_returns_status():
    s = KNNTypicalityModel().summary()
    assert s == {"status": "unfitted"}


def test_summary_fitted_contains_loo_stats():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m = KNNTypicalityModel(k=10).fit(X, population_label="x")
    s = m.summary()
    assert s["status"] == "fitted"
    assert s["k"] == 10
    assert s["sample_count"] == 100
    assert s["embedding_dim"] == 16
    assert s["population_label"] == "x"
    assert "loo_T_k_median" in s
    assert "loo_T_k_min" in s
    assert "ood_threshold" in s
