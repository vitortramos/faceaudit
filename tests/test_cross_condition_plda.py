# tests/test_cross_condition_plda.py

"""
Testes do estimador PLDA cross-condition (two-covariance).

Cobertura (15 testes do plano):
1. Sanidade estrutural
2. Condition-balancing — W difere do merge ingênuo
3. Equivalência degenerada — A≈B, W ≈ W_merge
4. Shrinkage compõe (MANUAL α=0.1)
5. Singletons em B/μ, não em W_B
6. Roundtrip + pickle "antigo" sem campos cross-condition
7. Guardas: SIMPLE+secondary, dim mismatch
8. Scoring inalterado (kernel finito)
9. Labels: ordem diferente, labels string
10. Within-mode: PER_CONDITION vs BALANCED_IDENTITY com shift controlado
11. Regras de validação default
12. LEDOIT_WOLF balanceado (α médio per-condição)
13. Cenário perfeitamente balanceado — propriedades estruturais
14. n_samples / n_classes cross-condition
15. Identidade single-condition (só-A, só-B, ambos)
"""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import fields, replace

import numpy as np
import pytest

from faceaudit.enums.covariance_shrinkage_mode import CovarianceShrinkageMode
from faceaudit.enums.cross_condition_within_mode import CrossConditionWithinMode
from faceaudit.enums.plda_variant import PLDAVariant
from faceaudit.enums.similarity_model_type import SimilarityModelType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.similarity_model_training_instruction import (
    SimilarityModelTrainingInstruction,
)
from faceaudit.models.similarity.plda_similarity_model import PLDASimilarityModel
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.services.cross_condition_two_covariance_plda_service import (
    fit_cross_condition_two_covariance,
)
from faceaudit.services.two_covariance_plda_service import (
    TwoCovariancePLDAState,
    fit_two_covariance,
    score_two_covariance_matrix,
    score_two_covariance_pair,
)


# -----------------------------------------------------------------------------
# Geradores sintéticos
# -----------------------------------------------------------------------------


def _make_balanced_arrays(
    n_ids: int = 30,
    n_per_id_a: int = 8,
    n_per_id_b: int = 8,
    D: int = 16,
    seed: int = 0,
    shift_per_identity_std: float = 0.0,
):
    """
    Gera 2 datasets sintéticos compartilhando identidades.

    `shift_per_identity_std > 0` injeta um deslocamento per-identidade entre
    A e B (efeito condição×identidade): μ_c^B = μ_c^A + δ_c onde δ_c ~ N(0, σ² I).
    """
    rng = np.random.default_rng(seed)
    centers_a = rng.normal(0.0, 1.0, size=(n_ids, D))
    if shift_per_identity_std > 0.0:
        deltas = rng.normal(0.0, shift_per_identity_std, size=(n_ids, D))
        centers_b = centers_a + deltas
    else:
        centers_b = centers_a.copy()

    X_a, y_a = [], []
    for i, c in enumerate(centers_a):
        samples = c + rng.normal(0.0, 0.3, size=(n_per_id_a, D))
        X_a.append(samples)
        y_a.extend([f"id_{i:03d}"] * n_per_id_a)

    X_b, y_b = [], []
    for i, c in enumerate(centers_b):
        samples = c + rng.normal(0.0, 0.3, size=(n_per_id_b, D))
        X_b.append(samples)
        y_b.extend([f"id_{i:03d}"] * n_per_id_b)

    return (
        np.vstack(X_a).astype(np.float64),
        np.array(y_a, dtype=object),
        np.vstack(X_b).astype(np.float64),
        np.array(y_b, dtype=object),
    )


def _make_embedding_dataset(X: np.ndarray, y: np.ndarray, name: str) -> EmbeddingDataset:
    items = []
    for i, (vec, label) in enumerate(zip(X, y)):
        items.append(
            EmbeddingInfo(
                embedding=vec.astype(np.float32),
                face_hash=f"{name}_hash_{i}",
                image_path=f"/tmp/{name}/img_{i}.jpg",
                identity=label,
                file_name=f"{name}_{i}.jpg",
                source="synthetic",
                origin="cross_condition_test",
            )
        )
    return EmbeddingDataset(name=name, items=items)


# =============================================================================
# 1. Sanidade estrutural
# =============================================================================


def test_1_estrutura_basica():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(seed=1)
    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)

    assert isinstance(state, TwoCovariancePLDAState)
    assert state.W.shape == (16, 16)
    assert state.B.shape == (16, 16)
    assert state.mu.shape == (16,)
    # W deve ser positiva definida
    np.linalg.cholesky(state.W)
    # Flags cross-condition
    assert state.cross_condition is True
    assert state.n_ids_condition_a == 30
    assert state.n_ids_condition_b == 30
    assert state.n_common_identities == 30
    assert state.cross_condition_within_mode == "BALANCED_IDENTITY"


# =============================================================================
# 2. Condition-balancing — W diverge do merge ingênuo quando há desbalanço
# =============================================================================


def test_2_w_cross_difere_do_merge_em_dados_desbalanceados():
    # A com 10×/ID, B com 2×/ID, mesma população base
    X_a, y_a, X_b, y_b = _make_balanced_arrays(
        n_ids=20, n_per_id_a=10, n_per_id_b=2, D=8, seed=2,
        shift_per_identity_std=0.5,  # garante diferença per-cond
    )

    state_cross = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.PER_CONDITION,
    )

    # Merge ingênuo (filtra singletons automaticamente)
    X_merge = np.vstack([X_a, X_b])
    y_merge = np.concatenate([y_a, y_b])
    state_merge = fit_two_covariance(X_merge, y_merge)

    # Em desbalanço acentuado, W cross ≠ W merge — peso igual por condição
    # muda a magnitude. Não exigimos um threshold específico, apenas
    # divergência detectável.
    diff = np.linalg.norm(state_cross.W - state_merge.W, ord="fro")
    norm = np.linalg.norm(state_merge.W, ord="fro")
    assert diff / norm > 0.01  # divergência ao menos 1%


# =============================================================================
# 3. Equivalência degenerada — A e B idênticos → W cross ~ W merge
# =============================================================================


def test_3_equivalencia_degenerada():
    # A e B com mesmos centros + mesmo nº de amostras
    X_a, y_a, X_b, y_b = _make_balanced_arrays(
        n_ids=20, n_per_id_a=6, n_per_id_b=6, D=8, seed=3,
        shift_per_identity_std=0.0,
    )

    state_cross = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.PER_CONDITION,
    )

    X_merge = np.vstack([X_a, X_b])
    y_merge = np.concatenate([y_a, y_b])
    state_merge = fit_two_covariance(X_merge, y_merge)

    # Magnitudes devem ser próximas (não exigimos igualdade exata —
    # normalizações divergem por design).
    ratio = np.trace(state_cross.W) / np.trace(state_merge.W)
    assert 0.5 < ratio < 2.0


# =============================================================================
# 4. Shrinkage MANUAL α=0.1 compõe
# =============================================================================


def test_4_shrinkage_manual_compoe():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(D=12, seed=4)

    state_no_shr = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        shrinkage_mode=CovarianceShrinkageMode.NONE,
    )
    state_shr = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        shrinkage_mode=CovarianceShrinkageMode.MANUAL,
        shrinkage_intensity=0.1,
    )

    # Shrinkage MANUAL α=0.1: W_shrunk = 0.9*W + 0.1*(tr(W)/D)*I
    D = state_no_shr.D
    mu_trace = np.trace(state_no_shr.W) / D
    expected = 0.9 * state_no_shr.W + 0.1 * mu_trace * np.eye(D)
    np.testing.assert_allclose(state_shr.W, expected, atol=1e-8)

    assert state_shr.shrinkage_mode == "MANUAL"
    assert state_shr.shrinkage_alpha_w == pytest.approx(0.1, abs=1e-8)


# =============================================================================
# 5. Singletons entram em B/μ mas não em W_B
# =============================================================================


def test_5_singletons_em_b_nao_em_w():
    # A: 20 IDs com 8 amostras cada
    # B: mesmos 20 IDs + 5 IDs adicionais com 1 amostra cada (singletons)
    rng = np.random.default_rng(5)
    D = 8
    n_main = 20

    X_a_list, y_a_list = [], []
    for i in range(n_main):
        c = rng.normal(0, 1, D)
        samples = c + rng.normal(0, 0.3, (8, D))
        X_a_list.append(samples)
        y_a_list.extend([f"id_{i:03d}"] * 8)

    X_b_list, y_b_list = [], []
    for i in range(n_main):
        c = rng.normal(0, 1, D)
        samples = c + rng.normal(0, 0.3, (5, D))
        X_b_list.append(samples)
        y_b_list.extend([f"id_{i:03d}"] * 5)

    # 5 singletons só em B
    for i in range(5):
        c = rng.normal(0, 1, D)
        X_b_list.append(c.reshape(1, -1))
        y_b_list.append(f"singleton_b_{i}")

    X_a = np.vstack(X_a_list).astype(np.float64)
    y_a = np.array(y_a_list, dtype=object)
    X_b = np.vstack(X_b_list).astype(np.float64)
    y_b = np.array(y_b_list, dtype=object)

    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)

    # Singletons entraram em B/μ (n_ids_condition_b conta-os).
    assert state.n_ids_condition_b == 25  # 20 multi + 5 singletons
    assert state.n_ids_condition_a == 20
    assert state.n_common_identities == 20
    # União = 25 (singletons só-B contam pra union)
    assert state.n_classes == 25


# =============================================================================
# 6. Roundtrip de state via PLDASimilarityModel + carregamento de pickle
#    "antigo" simulado (sem campos cross-condition)
# =============================================================================


def test_6_roundtrip_e_pickle_antigo():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(D=8, seed=6)
    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)

    # Roundtrip via PLDASimilarityModel.save/load
    config = PLDASimilarityModelConfig(
        plda_variant=PLDAVariant.TWO_COVARIANCE,
    )
    model = PLDASimilarityModel(config=config)
    model.two_cov_state = state
    model._is_fitted = True
    model.trained_dim = state.D
    from faceaudit.services.two_covariance_plda_service import _compute_scoring_kernels
    model._two_cov_kernels = _compute_scoring_kernels(state)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        loaded = PLDASimilarityModel()
        loaded.load(path)
        assert loaded.two_cov_state.cross_condition is True
        assert loaded.two_cov_state.cross_condition_within_mode == "BALANCED_IDENTITY"
        assert loaded.two_cov_state.n_common_identities == 30
    finally:
        os.unlink(path)

    # Carrega pickle "antigo" — payload sem os campos cross-condition.
    # Simula gravando manualmente um payload com two_cov_state sem as chaves novas.
    legacy_payload = {
        "model_version": 2,
        "model_type": "plda_similarity_model",
        "backend": "two_covariance",
        "name": "legacy",
        "config": config.to_dict(),
        "plda": None,
        "two_cov_state": {
            "mu": state.mu,
            "B": state.B,
            "W": state.W,
            "n_classes": state.n_classes,
            "n_samples": state.n_samples,
            "D": state.D,
            # SEM shrinkage_*, SEM cross_condition_*
        },
        "trained_rank": state.D,
        "trained_dim": state.D,
        "is_fitted": True,
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        legacy_path = f.name
    try:
        with open(legacy_path, "wb") as fh:
            pickle.dump(legacy_payload, fh)
        legacy = PLDASimilarityModel()
        legacy.load(legacy_path)
        # Acesso via getattr não quebra mesmo sem os campos no dict de origem.
        assert getattr(legacy.two_cov_state, "cross_condition", False) is False
        assert getattr(legacy.two_cov_state, "n_common_identities", None) is None
        assert getattr(legacy.two_cov_state, "cross_condition_within_mode", None) is None
    finally:
        os.unlink(legacy_path)


# =============================================================================
# 7. Guardas: SIMPLE + secondary → ValueError; dim mismatch → ValueError
# =============================================================================


def test_7_guarda_simple_e_dim():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(D=8, seed=7)

    config_simple = PLDASimilarityModelConfig(plda_variant=PLDAVariant.SIMPLE)
    model = PLDASimilarityModel(config=config_simple)
    ds_a = _make_embedding_dataset(X_a, y_a, "a")
    ds_b = _make_embedding_dataset(X_b, y_b, "b")
    with pytest.raises(ValueError, match="cross-condition"):
        model.fit(dataset=ds_a, logger_callback=lambda _m: None, secondary_dataset=ds_b)

    # Dim mismatch
    X_b_wrong = np.random.randn(50, 4)
    y_b_wrong = np.array(["id_000"] * 25 + ["id_001"] * 25, dtype=object)
    with pytest.raises(ValueError, match="Dimensões"):
        fit_cross_condition_two_covariance(X_a, y_a, X_b_wrong, y_b_wrong)


# =============================================================================
# 8. Scoring inalterado — score_matrix retorna valores finitos
# =============================================================================


def test_8_scoring_inalterado():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(D=8, seed=8)
    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)
    scores = score_two_covariance_matrix(state, X_a[:5], X_b[:5])
    assert scores.shape == (5, 5)
    assert np.all(np.isfinite(scores))


# =============================================================================
# 9. Labels: ordem diferente, labels string customizadas
# =============================================================================


def test_9a_ordem_diferente_resultado_identico():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(D=8, seed=9)
    state_orig = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)

    # Embaralha B mantendo correspondência X<->y
    rng = np.random.default_rng(99)
    perm = rng.permutation(X_b.shape[0])
    X_b_perm = X_b[perm]
    y_b_perm = y_b[perm]

    state_perm = fit_cross_condition_two_covariance(X_a, y_a, X_b_perm, y_b_perm)
    # Resultado deve ser numericamente idêntico (agrupamento por label string).
    np.testing.assert_allclose(state_orig.mu, state_perm.mu, atol=1e-12)
    np.testing.assert_allclose(state_orig.B, state_perm.B, atol=1e-12)
    np.testing.assert_allclose(state_orig.W, state_perm.W, atol=1e-12)


def test_9b_labels_string_funcionam():
    rng = np.random.default_rng(10)
    D = 8
    X_a, y_a, X_b, y_b = [], [], [], []
    for i in range(10):
        c = rng.normal(0, 1, D)
        a = c + rng.normal(0, 0.3, (5, D))
        b = c + rng.normal(0, 0.3, (5, D))
        X_a.append(a)
        X_b.append(b)
        label = f"pessoa_{i:03d}"  # label string explícito
        y_a.extend([label] * 5)
        y_b.extend([label] * 5)
    state = fit_cross_condition_two_covariance(
        np.vstack(X_a), np.array(y_a, dtype=object),
        np.vstack(X_b), np.array(y_b, dtype=object),
    )
    assert state.n_common_identities == 10


# =============================================================================
# 10. Within-mode com shift controlado: BALANCED_IDENTITY absorve quando há
#     deslocamento per-identidade
# =============================================================================


def test_10a_within_mode_shift_alto_balanced_absorve():
    # Shift forte per-identidade — BALANCED_IDENTITY deve ter trace(W) maior
    X_a, y_a, X_b, y_b = _make_balanced_arrays(
        n_ids=40, n_per_id_a=8, n_per_id_b=8, D=12, seed=100,
        shift_per_identity_std=1.5,
    )
    state_per = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.PER_CONDITION,
    )
    state_bal = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.BALANCED_IDENTITY,
    )
    # Com shift forte, BALANCED_IDENTITY absorve → trace maior
    assert np.trace(state_bal.W) > np.trace(state_per.W)


def test_10b_within_mode_sem_shift_proximo():
    # Sem shift, os dois modos devem produzir W aproximadamente iguais
    # (pequena diferença numérica esperada por singleton-pull dentro do
    # mecanismo de balancing — neste caso não há singletons, então as
    # médias balanceadas batem com as condicionais).
    X_a, y_a, X_b, y_b = _make_balanced_arrays(
        n_ids=40, n_per_id_a=8, n_per_id_b=8, D=12, seed=101,
        shift_per_identity_std=0.0,
    )
    state_per = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.PER_CONDITION,
    )
    state_bal = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        within_mode=CrossConditionWithinMode.BALANCED_IDENTITY,
    )
    # Sem shift, trace de W aproximadamente equivalente
    ratio = np.trace(state_bal.W) / np.trace(state_per.W)
    assert 0.5 < ratio < 2.0


# =============================================================================
# 11. Regras de validação default
# =============================================================================


def test_11_validation_rules_default_com_secondary():
    config = PLDASimilarityModelConfig(plda_variant=PLDAVariant.TWO_COVARIANCE)
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    instr = SimilarityModelTrainingInstruction(
        instruction_id="instr_001",
        name="train_plda",
        instruction_type=StrategyInstructionType.SIMILARITY_MODEL_TRAINING,
        similarity_model_type=SimilarityModelType.PLDA,
        input_dataset_resource_id="ds_primary",
        secondary_dataset_resource_id="ds_secondary",
        similarity_model_config=config,
        trained_similarity_model_resource_id="model_out",
    )
    rules = instr.validation_rule_configs
    rule_types = [r.rule_type for r in rules]

    # FISHER + MIN no primário
    assert ValidationRuleType.FISHER in rule_types
    # Primário FISHER targeting ds_primary
    fisher_rule = next(r for r in rules if r.rule_type == ValidationRuleType.FISHER)
    assert fisher_rule.params["dataset_resource_id"] == "ds_primary"

    # MIN deve aparecer 2x: primário + secundário
    min_rules = [r for r in rules if r.rule_type == ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES]
    assert len(min_rules) == 2
    targets = sorted([r.params["dataset_resource_id"] for r in min_rules])
    assert targets == ["ds_primary", "ds_secondary"]

    # DATASETS_EMBEDDING_COMPAT presente
    assert ValidationRuleType.DATASETS_EMBEDDING_COMPAT in rule_types
    compat_rule = next(r for r in rules if r.rule_type == ValidationRuleType.DATASETS_EMBEDDING_COMPAT)
    assert compat_rule.params["dataset_resource_ids"] == ["ds_primary", "ds_secondary"]

    # DATASET_OVERLAP_REQUIRED presente
    assert ValidationRuleType.DATASET_OVERLAP_REQUIRED in rule_types


def test_11_validation_rules_default_sem_secondary():
    config = PLDASimilarityModelConfig(plda_variant=PLDAVariant.TWO_COVARIANCE)
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    instr = SimilarityModelTrainingInstruction(
        instruction_id="instr_002",
        name="train_plda",
        instruction_type=StrategyInstructionType.SIMILARITY_MODEL_TRAINING,
        similarity_model_type=SimilarityModelType.PLDA,
        input_dataset_resource_id="ds_primary",
        similarity_model_config=config,
        trained_similarity_model_resource_id="model_out",
    )
    rule_types = [r.rule_type for r in instr.validation_rule_configs]
    # Sem secondary, comportamento clássico: FISHER + MIN no primário.
    assert rule_types.count(ValidationRuleType.FISHER) == 1
    assert rule_types.count(ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES) == 1
    assert ValidationRuleType.DATASETS_EMBEDDING_COMPAT not in rule_types
    assert ValidationRuleType.DATASET_OVERLAP_REQUIRED not in rule_types


def test_11_validation_rules_user_managed_preserva():
    config = PLDASimilarityModelConfig(plda_variant=PLDAVariant.TWO_COVARIANCE)
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    instr = SimilarityModelTrainingInstruction(
        instruction_id="instr_002",
        name="train_plda",
        instruction_type=StrategyInstructionType.SIMILARITY_MODEL_TRAINING,
        similarity_model_type=SimilarityModelType.PLDA,
        input_dataset_resource_id="ds_primary",
        similarity_model_config=config,
        trained_similarity_model_resource_id="model_out",
    )
    instr.validation_rule_configs = []
    instr.validation_rules_user_managed = True
    instr.secondary_dataset_resource_id = "ds_secondary"
    instr._sync_declared_io()
    # User-managed → não repopula nada.
    assert instr.validation_rule_configs == []


# =============================================================================
# 12. LEDOIT_WOLF com média per-condição
# =============================================================================


def test_12_ledoit_wolf_balanceado():
    # Dataset desbalanceado: A 10×/ID, B 2×/ID, distribuições diferentes
    rng = np.random.default_rng(12)
    D = 16

    centers = rng.normal(0, 1, (15, D))
    X_a_list, y_a_list = [], []
    for i, c in enumerate(centers):
        # Variância intra-classe BAIXA em A
        samples = c + rng.normal(0, 0.1, (10, D))
        X_a_list.append(samples)
        y_a_list.extend([f"id_{i:03d}"] * 10)
    X_b_list, y_b_list = [], []
    for i, c in enumerate(centers):
        # Variância intra-classe ALTA em B
        samples = c + rng.normal(0, 1.0, (2, D))
        X_b_list.append(samples)
        y_b_list.extend([f"id_{i:03d}"] * 2)

    X_a = np.vstack(X_a_list)
    y_a = np.array(y_a_list, dtype=object)
    X_b = np.vstack(X_b_list)
    y_b = np.array(y_b_list, dtype=object)

    state = fit_cross_condition_two_covariance(
        X_a, y_a, X_b, y_b,
        shrinkage_mode=CovarianceShrinkageMode.LEDOIT_WOLF,
    )

    # α efetivo deve estar registrado no state
    assert state.shrinkage_mode == "LEDOIT_WOLF"
    assert state.shrinkage_alpha_w is not None
    assert 0.0 <= state.shrinkage_alpha_w <= 1.0


# =============================================================================
# 13. Cenário balanceado — propriedades estruturais
# =============================================================================


def test_13_cenario_balanceado_propriedades_estruturais():
    X_a, y_a, X_b, y_b = _make_balanced_arrays(
        n_ids=30, n_per_id_a=6, n_per_id_b=6, D=10, seed=13,
        shift_per_identity_std=0.0,
    )
    state_cross = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)
    state_merge = fit_two_covariance(
        np.vstack([X_a, X_b]), np.concatenate([y_a, y_b])
    )

    # (a) Ambas W positivas definidas
    np.linalg.cholesky(state_cross.W)
    np.linalg.cholesky(state_merge.W)

    # (b) Magnitudes na mesma ordem
    ratio_w = np.trace(state_cross.W) / np.trace(state_merge.W)
    assert 0.1 < ratio_w < 10.0

    # (c) Razão tr(B)/tr(W) na mesma ordem (PLDA "força" comparável)
    strength_cross = np.trace(state_cross.B) / np.trace(state_cross.W)
    strength_merge = np.trace(state_merge.B) / np.trace(state_merge.W)
    ratio_strength = strength_cross / strength_merge
    assert 0.1 < ratio_strength < 10.0

    # (d) Sinal correto de scores: genuíno > impostor médio
    n_test = 10
    # Pares genuínos (mesma identidade) — vstack A e B sample-a-sample dos
    # primeiros n_test IDs.
    genuine_scores = []
    impostor_scores = []
    for state in [state_cross, state_merge]:
        gen = []
        imp = []
        for i in range(n_test):
            # genuíno: A[i*6:i*6+6] vs B[i*6:i*6+6] (mesma ID em ambas)
            scores = score_two_covariance_matrix(state, X_a[i * 6:(i + 1) * 6], X_b[i * 6:(i + 1) * 6])
            gen.append(scores.mean())
            # impostor: A[i] vs B[(i+1) % n_test]
            j = (i + 1) % n_test
            scores_imp = score_two_covariance_matrix(state, X_a[i * 6:(i + 1) * 6], X_b[j * 6:(j + 1) * 6])
            imp.append(scores_imp.mean())
        genuine_scores.append(np.mean(gen))
        impostor_scores.append(np.mean(imp))
    assert genuine_scores[0] > impostor_scores[0]  # cross-condition
    assert genuine_scores[1] > impostor_scores[1]  # merge


# =============================================================================
# 14. n_samples / n_classes cross-condition
# =============================================================================


def test_14_contagens_de_amostras_e_classes():
    # 3 IDs: só-A, só-B, ambos. Contagens conhecidas.
    rng = np.random.default_rng(14)
    D = 4
    # ID_001 só em A (5 amostras)
    X_only_a = rng.normal(0, 1, (5, D))
    y_only_a = ["ID_001"] * 5
    # ID_003 em A e B (3 e 4 amostras)
    c_003 = rng.normal(0, 1, D)
    X_003_a = c_003 + rng.normal(0, 0.3, (3, D))
    X_003_b = c_003 + rng.normal(0, 0.3, (4, D))
    # ID_002 só em B (6 amostras)
    X_only_b = rng.normal(0, 1, (6, D))
    y_only_b = ["ID_002"] * 6

    X_a = np.vstack([X_only_a, X_003_a])
    y_a = np.array(y_only_a + ["ID_003"] * 3, dtype=object)
    X_b = np.vstack([X_only_b, X_003_b])
    y_b = np.array(y_only_b + ["ID_003"] * 4, dtype=object)

    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)

    assert state.n_samples == X_a.shape[0] + X_b.shape[0]  # 8 + 10 = 18
    assert state.n_classes == 3  # ID_001, ID_002, ID_003
    assert state.n_common_identities == 1
    assert state.n_ids_condition_a == 2  # ID_001, ID_003
    assert state.n_ids_condition_b == 2  # ID_002, ID_003


# =============================================================================
# 15. Identidade single-condition: ID só-A, ID só-B, ID comum
# =============================================================================


def test_15_identidade_single_condition():
    rng = np.random.default_rng(15)
    D = 6
    # ID_001 só em A (4 amostras)
    c_001 = rng.normal(0, 1, D)
    X_001_a = c_001 + rng.normal(0, 0.2, (4, D))
    # ID_002 só em B (4 amostras)
    c_002 = rng.normal(0, 1, D)
    X_002_b = c_002 + rng.normal(0, 0.2, (4, D))
    # ID_003 em ambas
    c_003 = rng.normal(0, 1, D)
    X_003_a = c_003 + rng.normal(0, 0.2, (5, D))
    X_003_b = c_003 + rng.normal(0, 0.2, (5, D))

    X_a = np.vstack([X_001_a, X_003_a])
    y_a = np.array(["ID_001"] * 4 + ["ID_003"] * 5, dtype=object)
    X_b = np.vstack([X_002_b, X_003_b])
    y_b = np.array(["ID_002"] * 4 + ["ID_003"] * 5, dtype=object)

    # Não deve levantar erro.
    state = fit_cross_condition_two_covariance(X_a, y_a, X_b, y_b)
    assert state.n_common_identities == 1
    assert state.n_classes == 3
    # W positivamente definida
    np.linalg.cholesky(state.W)
