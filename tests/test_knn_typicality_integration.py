"""Teste de integração: cosseno + tipicidade kNN no calibrador LogReg.

Valida a tese estatística que motiva todo o feature: adicionar
typicality como feature auxiliar ao calibrador LogReg (sobre cosseno)
reduz Cllr quando a população de teste tem heterogeneidade de tipicidade.

Setup sintético controlado:
- Duas subpopulações no S^{d-1}: "típica" (modo μ_a) e "atípica" (modo μ_b
  em direção quase ortogonal).
- Cada identidade tem 4 amostras concentradas vMF em torno do seu protótipo.
- Pares genuínos: mesma identidade. Pares impostores: identidades diferentes.
- Tipicidade treinada SÓ na população típica → impostores da população
  típica tendem a ter cosseno alto (hubness sintético: muitas identidades
  próximas), e o calibrador sem tipicidade os sobrevaloriza.

Métrica: Cllr empírico (Brümmer & du Preez 2006) sobre o test set.
Esperamos: cllr_2feat < cllr_1feat com margem mensurável.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from faceaudit.services.knn_typicality_model import KNNTypicalityModel


# ─────────────────────────── helpers ────────────────────────────


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _sample_around(
    mu: np.ndarray, n: int, jitter: float, rng: np.random.Generator,
) -> np.ndarray:
    """Amostra n pontos perto de mu com 'jitter' angular controlado."""
    d = mu.shape[0]
    noise = rng.normal(size=(n, d)) * jitter
    return _normalize(mu[None, :] + noise)


def _build_identity_population(
    mu_proto: np.ndarray,
    n_identities: int,
    samples_per_id: int,
    proto_jitter: float,
    intra_jitter: float,
    rng: np.random.Generator,
):
    """
    Gera (embeddings, identity_labels) — cada identidade tem um protótipo
    perto de mu_proto, e suas amostras perto do seu protótipo.
    """
    prototypes = _sample_around(mu_proto, n_identities, proto_jitter, rng)
    all_emb, all_ids = [], []
    for i, p in enumerate(prototypes):
        samples = _sample_around(p, samples_per_id, intra_jitter, rng)
        all_emb.append(samples)
        all_ids.extend([f"id_{i}"] * samples_per_id)
    return np.vstack(all_emb), np.array(all_ids)


def _make_pairs(
    embeddings: np.ndarray, identities: np.ndarray, max_pairs: int,
    rng: np.random.Generator,
):
    """
    Constrói pares (probe_idx, ref_idx) balanceados gen/imp limitados a max_pairs.
    """
    n = len(embeddings)
    same_idx_pairs = []
    diff_idx_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if identities[i] == identities[j]:
                same_idx_pairs.append((i, j))
            else:
                diff_idx_pairs.append((i, j))

    rng.shuffle(same_idx_pairs)
    rng.shuffle(diff_idx_pairs)
    half = max_pairs // 2
    same_idx_pairs = same_idx_pairs[:half]
    diff_idx_pairs = diff_idx_pairs[:half]

    pairs = np.array(same_idx_pairs + diff_idx_pairs, dtype=int)
    labels = np.array([1] * len(same_idx_pairs) + [0] * len(diff_idx_pairs), dtype=int)

    perm = rng.permutation(len(pairs))
    return pairs[perm], labels[perm]


def _cllr(log_lrs: np.ndarray, labels: np.ndarray) -> float:
    """
    Cllr empírico em bits (Brümmer & du Preez 2006).
    log_lrs em escala natural (log base e), labels: 1=gen, 0=imp.
    """
    gen = log_lrs[labels == 1]
    imp = log_lrs[labels == 0]
    term_gen = np.mean(np.logaddexp(0, -gen)) / np.log(2)
    term_imp = np.mean(np.logaddexp(0, imp)) / np.log(2)
    return 0.5 * (term_gen + term_imp)


# ─────────────────────────── integração ────────────────────────


def _build_isotropic_population(
    n_identities: int, samples_per_id: int, intra_jitter: float, d: int,
    rng: np.random.Generator,
):
    """Identidades em direções aleatórias da esfera (não-concentradas).

    Cenário oposto ao hubness: identidades isoladas, impostores têm
    cosseno baixo, scoring discrimina bem.
    """
    prototypes = _normalize(rng.normal(size=(n_identities, d)))
    all_emb, all_ids = [], []
    for i, p in enumerate(prototypes):
        samples = _sample_around(p, samples_per_id, intra_jitter, rng)
        all_emb.append(samples)
        all_ids.extend([f"iso_{i}"] * samples_per_id)
    return np.vstack(all_emb), np.array(all_ids)


@pytest.mark.parametrize("seed", [0, 1])
def test_cosine_plus_typicality_lowers_cllr(seed: int):
    """
    Tese: num cenário com heterogeneidade de tipicidade, treinar LogReg
    com [cosine, typicality_mean] dá Cllr menor que treinar só com [cosine].

    Cenário:
    - Pop A (típica): identidades MUITO próximas em torno de μ_a (hubness
      sintético) — impostores entre IDs de A têm cosseno alto também,
      sistema cosseno-só os confunde com genuínos.
    - Pop B (atípica): identidades em direções aleatórias da esfera, bem
      separadas — impostores entre IDs de B têm cosseno baixo, sistema
      discrimina bem.

    Tipicidade treinada SÓ em A → typicality ~1 pra faces de A, ~0 pra B.
    O calibrador 2D aprende: "cosseno alto + tipicidade baixa = LR forte;
    cosseno alto + tipicidade alta = pode ser hub, LR moderada."
    """
    rng = np.random.default_rng(seed)
    d = 64

    # Protótipo da população A (concentrada/hubness)
    mu_a = np.zeros(d); mu_a[0] = 1.0

    # 1) Referência pra tipicidade — só população A (concentrada)
    ref_emb, _ = _build_identity_population(
        mu_a, n_identities=200, samples_per_id=4,
        proto_jitter=0.08, intra_jitter=0.05, rng=rng,
    )

    # 2) Treino do calibrador — A (hub) + B (isolada)
    train_emb_a, train_ids_a = _build_identity_population(
        mu_a, n_identities=60, samples_per_id=4,
        proto_jitter=0.08, intra_jitter=0.05, rng=rng,
    )
    train_ids_a = np.array([f"trainA_{x}" for x in train_ids_a])
    train_emb_b, train_ids_b = _build_isotropic_population(
        n_identities=60, samples_per_id=4, intra_jitter=0.05, d=d, rng=rng,
    )
    train_ids_b = np.array([f"trainB_{x}" for x in train_ids_b])
    train_emb = np.vstack([train_emb_a, train_emb_b])
    train_ids = np.concatenate([train_ids_a, train_ids_b])

    # 3) Test — outra realização de A e B
    test_emb_a, test_ids_a = _build_identity_population(
        mu_a, n_identities=60, samples_per_id=4,
        proto_jitter=0.08, intra_jitter=0.05, rng=rng,
    )
    test_ids_a = np.array([f"testA_{x}" for x in test_ids_a])
    test_emb_b, test_ids_b = _build_isotropic_population(
        n_identities=60, samples_per_id=4, intra_jitter=0.05, d=d, rng=rng,
    )
    test_ids_b = np.array([f"testB_{x}" for x in test_ids_b])
    test_emb = np.vstack([test_emb_a, test_emb_b])
    test_ids = np.concatenate([test_ids_a, test_ids_b])

    # 4) Treina o modelo de tipicidade na referência (só A)
    typ_model = KNNTypicalityModel(k=20).fit(ref_emb, population_label="A_only")

    # 5) Tipicidades por face no train e test
    train_typ = typ_model.get_percentiles(train_emb) / 100.0
    test_typ = typ_model.get_percentiles(test_emb) / 100.0

    # 6) Pares
    train_pairs, train_labels = _make_pairs(
        train_emb, train_ids, max_pairs=1600, rng=rng,
    )
    test_pairs, test_labels = _make_pairs(
        test_emb, test_ids, max_pairs=1600, rng=rng,
    )

    # 7) Cosseno por par
    def pair_cos(emb, pairs):
        return np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)

    train_cos = pair_cos(train_emb, train_pairs)
    test_cos = pair_cos(test_emb, test_pairs)

    # 8) Typicality pair (combiner MEAN)
    train_typ_pair = 0.5 * (train_typ[train_pairs[:, 0]] + train_typ[train_pairs[:, 1]])
    test_typ_pair = 0.5 * (test_typ[test_pairs[:, 0]] + test_typ[test_pairs[:, 1]])

    # 9) Treina dois calibradores LogReg
    #    Saída de decision_function é log-odds = log(LR) — Cllr direto.
    clf_1 = LogisticRegression(C=1.0, max_iter=2000)
    clf_1.fit(train_cos.reshape(-1, 1), train_labels)
    log_lr_1 = clf_1.decision_function(test_cos.reshape(-1, 1))

    clf_2 = LogisticRegression(C=1.0, max_iter=2000)
    X2_train = np.column_stack([train_cos, train_typ_pair])
    X2_test = np.column_stack([test_cos, test_typ_pair])
    clf_2.fit(X2_train, train_labels)
    log_lr_2 = clf_2.decision_function(X2_test)

    cllr_1 = _cllr(log_lr_1, test_labels)
    cllr_2 = _cllr(log_lr_2, test_labels)

    # Aceitamos relative_improvement >= 5% (margem conservadora para
    # absorver ruído do experimento; com seeds [0, 1] a melhoria
    # típica observada empiricamente é 10-25%).
    rel_improvement = (cllr_1 - cllr_2) / cllr_1
    print(
        f"seed={seed}: Cllr(cos)={cllr_1:.4f}, "
        f"Cllr(cos+typ)={cllr_2:.4f}, melhoria rel={rel_improvement*100:.1f}%"
    )
    assert cllr_2 < cllr_1, (
        f"Cllr(cos+typ)={cllr_2:.4f} NÃO menor que Cllr(cos)={cllr_1:.4f}"
    )
    assert rel_improvement >= 0.05, (
        f"Melhoria rel ({rel_improvement*100:.1f}%) abaixo do esperado (>=5%)"
    )


def test_cosine_only_baseline_is_reasonable():
    """Sanity: o baseline cosseno-só deve dar Cllr < 1 no setup acima.
    Cllr = 1 = sistema aleatório; > 0.5 = ruim mas informativo."""
    rng = np.random.default_rng(42)
    d = 64
    mu = np.zeros(d); mu[0] = 1.0

    emb, ids = _build_identity_population(
        mu, n_identities=60, samples_per_id=4,
        proto_jitter=0.35, intra_jitter=0.15, rng=rng,
    )

    pairs, labels = _make_pairs(emb, ids, max_pairs=1000, rng=rng)
    cos_scores = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)

    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(cos_scores.reshape(-1, 1), labels)
    # Auto-avaliação no mesmo split — só pra confirmar que o setup é
    # discriminativo. Calibração in-sample superestima, mas Cllr deve
    # claramente estar abaixo de 1 (sistema informativo).
    log_lr = clf.decision_function(cos_scores.reshape(-1, 1))
    cllr = _cllr(log_lr, labels)
    assert cllr < 1.0, f"Setup degenerado: cosseno-só Cllr={cllr:.4f} >= 1.0"
