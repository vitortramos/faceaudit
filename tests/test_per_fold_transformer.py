# tests/test_per_fold_transformer.py
"""
PR 3 — Correção metodológica V2.

Cobre quatro propriedades do `ArchitectureVectorizerTransformer`
+ pipeline `[arch_vec → scaler → logreg]` dentro do
`GridSearchCV(cv=IdentityDisjointPairKFold)`:

1. **Sentinela de leakage**: perturbar APENAS linhas que cairão no
   test fold do CV não altera o `state_` que o transformer aprende
   no train fold daquela mesma chamada (`fit(X_train, y_train)`).
   Garante que s₀ e r̄ são funções exclusivas do train fold.

2. **Consistência do refit final**: após `GridSearchCV(refit=True)`,
   o `state_` do `best_estimator_` é idêntico ao state que se
   obtém fitando o transformer diretamente sobre todo D_cal.

3. **Equivalência arquitetural**: transformer fitado em D_cal
   completo produz a mesma matriz vetorizada que o caminho legado
   `FIT_FULL_CALIBRATION_LEGACY` (mesmo s₀, mesmo r̄, mesma
   builder).

4. **CV identidade-disjunta real**: para cada fold, nenhuma
   identidade (probe OU reference) aparece simultaneamente em
   pares de train E test.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    schema_for_base_and_rarities,
    schema_for_base_only,
)
from faceaudit.services.architecture_vectorizer_transformer import (
    ArchitectureVectorizerTransformer,
)
from faceaudit.services.identity_disjoint_pair_kfold import (
    IdentityDisjointPairKFold,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _build_synthetic_dataset(
    *,
    n_identities: int = 30,
    seed: int = 7,
) -> dict:
    """
    Gera dataset de pares cosseno × raridade onde:
      - genuíno → score base alto + raridade tipicamente baixa
      - impostor → score base baixo + raridade ~ unif

    Pair role: probe (lateral) → reference (frontal).
    """
    rng = np.random.default_rng(seed)
    probe_ids: list[str] = []
    reference_ids: list[str] = []
    pair_keys: list[tuple[str, str]] = []
    matrix_rows: list[list[float]] = []
    same_person: list[bool] = []

    # Cada identidade tem 1 probe e 1 reference
    for i in range(n_identities):
        for j in range(n_identities):
            # par (probe_i, reference_j); genuíno sse i == j
            label = i == j
            base = (
                float(rng.normal(0.72, 0.05))
                if label
                else float(rng.normal(0.35, 0.08))
            )
            r_F = float(rng.uniform(0.05, 0.5))
            r_P = float(rng.uniform(0.05, 0.5))
            probe_ids.append(f"P{i:03d}")
            reference_ids.append(f"R{j:03d}")
            pair_keys.append((f"hashP{i:03d}", f"hashR{j:03d}"))
            matrix_rows.append([base, r_F, r_P])
            same_person.append(label)

    return {
        "matrix": np.asarray(matrix_rows, dtype="<f8"),
        "y": np.asarray(same_person, dtype=int),
        "probe_ids": tuple(probe_ids),
        "reference_ids": tuple(reference_ids),
        "pair_keys": tuple(pair_keys),
    }


# ── 1. Sentinela de leakage ─────────────────────────────────────────


def test_perturbing_test_fold_does_not_affect_state_from_train_fold():
    """
    Estrutura: transformer.fit é chamado APENAS sobre X_train.
    Perturbar arbitrariamente as linhas correspondentes ao test fold
    NÃO pode alterar `state_` aprendido — porque o transformer não
    as enxerga durante o `fit`. Esta é uma garantia ESTRUTURAL da
    integração via Pipeline; se este teste falhar, há regressão
    semântica grave.
    """
    data = _build_synthetic_dataset(n_identities=20, seed=11)
    schema = schema_for_base_and_rarities()
    splitter = IdentityDisjointPairKFold(
        probe_identities=list(data["probe_ids"]),
        reference_identities=list(data["reference_ids"]),
        n_splits=4,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(data["matrix"], data["y"]))

    X_train = data["matrix"][train_idx]
    y_train = data["y"][train_idx]

    transformer = ArchitectureVectorizerTransformer(
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        schema=schema,
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
    )
    transformer.fit(X_train, y_train)
    s0_baseline = transformer.state_.s0
    r0_F_baseline = transformer.state_.r0_probe
    r0_P_baseline = transformer.state_.r0_reference

    # Perturba TODAS as linhas que NÃO foram vistas pelo fit.
    # Em particular, mexe brutalmente nas linhas do test fold +
    # nas linhas-boundary (excluídas deste fold).
    rng = np.random.default_rng(99)
    perturbed = data["matrix"].copy()
    not_seen = np.setdiff1d(
        np.arange(len(data["matrix"]), dtype=np.int64),
        train_idx,
    )
    perturbed[not_seen, :] = rng.normal(0, 10, size=(len(not_seen), 3))

    X_train_again = perturbed[train_idx]
    y_train_again = data["y"][train_idx]
    transformer2 = ArchitectureVectorizerTransformer(
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        schema=schema,
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
    )
    transformer2.fit(X_train_again, y_train_again)

    assert transformer2.state_.s0 == pytest.approx(s0_baseline, abs=1e-12)
    assert transformer2.state_.r0_probe == pytest.approx(r0_F_baseline, abs=1e-12)
    assert transformer2.state_.r0_reference == pytest.approx(r0_P_baseline, abs=1e-12)


# ── 2. Consistência do refit final ──────────────────────────────────


def test_final_refit_state_equals_independent_fit_on_full_d_cal():
    """
    Após GridSearchCV(refit=True), `best_estimator_.named_steps["arch_vec"]
    .state_` deve ser idêntico ao state que se obtém fitando o transformer
    sobre todo D_cal numa chamada única.
    """
    # n_identities maior + balanceamento por subamostragem dos impostores
    # garante folds com ambas as classes (evita NaN no neg_log_loss CV).
    data = _build_synthetic_dataset(n_identities=40, seed=23)
    rng = np.random.default_rng(123)
    y = data["y"]
    gen_idx = np.flatnonzero(y == 1)
    imp_idx = np.flatnonzero(y == 0)
    imp_keep = rng.choice(imp_idx, size=4 * len(gen_idx), replace=False)
    keep = np.sort(np.concatenate([gen_idx, imp_keep]))
    data = {
        "matrix": data["matrix"][keep],
        "y": data["y"][keep],
        "probe_ids": tuple(np.asarray(data["probe_ids"])[keep].tolist()),
        "reference_ids": tuple(np.asarray(data["reference_ids"])[keep].tolist()),
        "pair_keys": tuple(np.asarray(data["pair_keys"], dtype=object)[keep].tolist()),
    }
    schema = schema_for_base_and_rarities()
    splitter = IdentityDisjointPairKFold(
        probe_identities=list(data["probe_ids"]),
        reference_identities=list(data["reference_ids"]),
        n_splits=4,
        random_state=42,
    )

    pipe = Pipeline([
        ("arch_vec", ArchitectureVectorizerTransformer(
            architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
            schema=schema,
            pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
        )),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=1000,
        )),
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid={"logreg__C": [0.1, 1.0]},
        cv=splitter,
        scoring="neg_log_loss",
        refit=True,
        n_jobs=1,
    )
    grid.fit(data["matrix"], data["y"])
    state_from_pipeline = grid.best_estimator_.named_steps["arch_vec"].state_

    # Ajuste independente sobre todo D_cal
    independent = ArchitectureVectorizerTransformer(
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        schema=schema,
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
    )
    independent.fit(data["matrix"], data["y"])
    state_independent = independent.state_

    assert state_from_pipeline.s0 == pytest.approx(state_independent.s0, abs=1e-12)
    assert state_from_pipeline.r0_probe == pytest.approx(
        state_independent.r0_probe, abs=1e-12,
    )
    assert state_from_pipeline.r0_reference == pytest.approx(
        state_independent.r0_reference, abs=1e-12,
    )
    assert (
        state_from_pipeline.fit_scope
        == ArchitectureFitScope.PER_FOLD_REFIT_FULL_CALIBRATION
    )


# ── 3. Equivalência arquitetural ────────────────────────────────────


def test_transformer_full_fit_matches_legacy_builder_matrix():
    """
    A matriz vetorizada produzida pelo transformer sobre D_cal completo
    deve coincidir bit-a-bit com a matriz que o caminho legado
    `FIT_FULL_CALIBRATION_LEGACY` produziria — porque ambos resolvem
    s₀/r̄ por estimadores idênticos e delegam ao mesmo `_builder`.
    """
    data = _build_synthetic_dataset(n_identities=15, seed=37)
    schema = schema_for_base_and_rarities()

    transformer = ArchitectureVectorizerTransformer(
        architecture=LRCalibrationArchitecture.LINEAR_CENTERED,
        schema=schema,
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
    )
    transformer.fit(data["matrix"], data["y"])
    X_via_transformer = transformer.transform(data["matrix"])

    # Caminho legado direto via _builder + state_factory
    from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
    from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
    from faceaudit.models.lr.lr_architecture_state import (
        LRArchitectureStateFactory,
    )
    from faceaudit.services import architecture_feature_builder as _builder
    from faceaudit.services.s0_resolution_service import resolve_s0_auto_logreg

    spec = LR_ARCHITECTURE_SPECS[LRCalibrationArchitecture.LINEAR_CENTERED]
    base = data["matrix"][:, schema.base_score_column]
    r_F = data["matrix"][:, schema.rarity_probe_column]
    r_P = data["matrix"][:, schema.rarity_reference_column]

    s0_summary = resolve_s0_auto_logreg(base, data["y"])
    legacy_state = LRArchitectureStateFactory.create(
        spec=spec,
        s0=s0_summary.s0_value,
        r0_probe=float(r_F.mean()),
        r0_reference=float(r_P.mean()),
        base_score_kind="cosine",
        pair_role_orientation=PairRoleOrientation.PROBE_REFERENCE,
        fit_scope=ArchitectureFitScope.FULL_CALIBRATION_LEGACY,
        s0_resolution_method=S0ResolutionMethod.AUTO_LOGREG,
        r0_probe_estimator="MEAN",
        r0_reference_estimator="MEAN",
        n_fitted_pairs=len(base),
        n_fitted_genuine_pairs=int(data["y"].sum()),
        n_fitted_impostor_pairs=int((1 - data["y"]).sum()),
        s0_logreg_config=None,
        s0_fit_summary=s0_summary.to_dict_hex(),
    )
    X_via_legacy = _builder.build(spec, legacy_state, base, r_F, r_P)

    np.testing.assert_allclose(
        X_via_transformer, X_via_legacy, atol=1e-12, rtol=0,
    )


# ── 4. CV identidade-disjunta real ──────────────────────────────────


def test_cv_folds_are_identity_disjoint_on_both_sides():
    """
    Para cada fold (train_idx, test_idx) emitido pelo splitter:
      - Identidades dos probes em train ∩ identidades dos probes em
        test = ∅
      - Identidades dos references em train ∩ identidades dos
        references em test = ∅
      - Identidades dos probes em train ∩ identidades dos references
        em test = ∅ (e vice-versa) — porque o splitter parte do conjunto
        UNIÃO `unique(probe ∪ reference)` e atribui cada id a UM grupo.
    """
    data = _build_synthetic_dataset(n_identities=24, seed=51)
    splitter = IdentityDisjointPairKFold(
        probe_identities=list(data["probe_ids"]),
        reference_identities=list(data["reference_ids"]),
        n_splits=5,
        random_state=42,
    )
    probe_arr = np.asarray(data["probe_ids"])
    reference_arr = np.asarray(data["reference_ids"])

    for train_idx, test_idx in splitter.split(data["matrix"], data["y"]):
        probe_train = set(probe_arr[train_idx])
        probe_test = set(probe_arr[test_idx])
        ref_train = set(reference_arr[train_idx])
        ref_test = set(reference_arr[test_idx])

        assert probe_train.isdisjoint(probe_test)
        assert ref_train.isdisjoint(ref_test)
        # União das identidades em train não intersecta união em test:
        union_train = probe_train | ref_train
        union_test = probe_test | ref_test
        assert union_train.isdisjoint(union_test), (
            "Identidade aparece em train (probe ou ref) E em test "
            "(probe ou ref) — splitter quebrou a propriedade de "
            "disjunção dos dois lados."
        )
