"""
Testes do transformator LDA (Linear Discriminant Analysis).

Cobre fit/transform, estimadores (STANDARD/LEDOIT_WOLF/OAS), shrinkage,
grid search (FISHER_RATIO e COSINE_AUC), backward compat (default config
= comportamento original com STANDARD) e save/load.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from faceaudit.enums.lda_estimator_type import LdaEstimatorType
from faceaudit.enums.lda_grid_search_metric import LdaGridSearchMetric
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.lda_config import LDAConfig
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.transformators.lda_transformation import LdaTransformation


# =============================================================================
# Helpers
# =============================================================================

def _build_labeled_dataset(
    n_classes: int = 20,
    samples_per_class: int = 5,
    dim: int = 32,
    intra_std: float = 1.0,
    inter_scale: float = 5.0,
    seed: int = 42,
    n_singletons: int = 0,
) -> EmbeddingDataset:
    """Dataset rotulado sintético com classes razoavelmente separadas."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=inter_scale, size=(n_classes, dim))
    items: List[EmbeddingInfo] = []
    for cls_i in range(n_classes):
        for s in range(samples_per_class):
            emb = means[cls_i] + rng.normal(scale=intra_std, size=dim)
            items.append(EmbeddingInfo(
                embedding=emb.astype(np.float32),
                face_hash=f"h_{cls_i}_{s}",
                image_path=f"/tmp/{cls_i}_{s}.jpg",
                identity=f"id_{cls_i}",
                file_name=f"f_{cls_i}_{s}.jpg",
            ))
    for k in range(n_singletons):
        emb = rng.normal(scale=intra_std, size=dim)
        items.append(EmbeddingInfo(
            embedding=emb.astype(np.float32),
            face_hash=f"h_solo_{k}",
            image_path=f"/tmp/solo_{k}.jpg",
            identity=f"solo_{k}",
            file_name=f"solo_{k}.jpg",
        ))
    return EmbeddingDataset(name="ds", items=items)


def _make_lda(config: Optional[LDAConfig] = None, seed: int = 42) -> LdaTransformation:
    return LdaTransformation(
        math_service=EmbeddingMathService(seed=seed),
        config=config or LDAConfig(),
    )


# =============================================================================
# 1. Sanidade: fisher ratio aumenta com qualquer estimator
# =============================================================================

@pytest.mark.parametrize("estimator", [
    LdaEstimatorType.STANDARD,
    LdaEstimatorType.LEDOIT_WOLF,
    LdaEstimatorType.OAS,
])
def test_lda_fisher_ratio_increases(estimator):
    ds = _build_labeled_dataset(n_classes=20, samples_per_class=5, dim=32)
    lda = _make_lda(LDAConfig(estimator=estimator, regularization=1e-4))
    lda.fit(ds)
    m = lda.get_execution_metrics()
    assert m["fisher_ratio_after"] > m["fisher_ratio_before"], (
        f"{estimator.label}: Fisher ratio não aumentou após LDA"
    )
    assert m["estimator_used"] == estimator.value


# =============================================================================
# 2. Output dimension respects n_components OR max (n_classes - 1)
# =============================================================================

def test_lda_n_components_auto_caps_at_n_classes_minus_one():
    ds = _build_labeled_dataset(n_classes=5, samples_per_class=4, dim=32)
    lda = _make_lda(LDAConfig(n_components=None))
    lda.fit(ds)
    transformed = lda.transform(ds)
    assert transformed.matrix.shape[1] == 4  # n_classes - 1


def test_lda_n_components_explicit_respected():
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=4, dim=32)
    lda = _make_lda(LDAConfig(n_components=3))
    lda.fit(ds)
    transformed = lda.transform(ds)
    assert transformed.matrix.shape[1] == 3


# =============================================================================
# 3. Singletons descartados (≥ 2 amostras por classe obrigatório)
# =============================================================================

def test_lda_skips_singletons():
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=3, dim=32, n_singletons=5)
    lda = _make_lda()
    lda.fit(ds)
    m = lda.get_execution_metrics()
    assert m["n_classes_used"] == 10
    assert m["n_singletons_skipped"] == 5


# =============================================================================
# 4. Backward compat: LDAConfig() default mantém estimator STANDARD
# =============================================================================

def test_lda_default_config_is_standard_estimator():
    config = LDAConfig()
    assert config.estimator == LdaEstimatorType.STANDARD
    assert config.shrinkage == 0.0
    assert config.use_grid_search is False


# =============================================================================
# 5. Shrinkage explícito é aceito e refletido nas métricas
# =============================================================================

def test_lda_shrinkage_recorded_in_metrics():
    ds = _build_labeled_dataset(n_classes=15, samples_per_class=4, dim=32)
    lda = _make_lda(LDAConfig(shrinkage=0.3))
    lda.fit(ds)
    m = lda.get_execution_metrics()
    assert m["final_shrinkage"] == 0.3


# =============================================================================
# 6. Grid search com FISHER_RATIO escolhe ε válido
# =============================================================================

def test_lda_grid_search_fisher_ratio():
    ds = _build_labeled_dataset(n_classes=40, samples_per_class=5, dim=32)
    config = LDAConfig(
        use_grid_search=True,
        reg_grid=[1e-6, 1e-4, 1e-2, 1.0],
        grid_search_metric=LdaGridSearchMetric.FISHER_RATIO,
        grid_search_holdout_ratio=0.2,
    )
    lda = _make_lda(config)
    lda.fit(ds)
    m = lda.get_execution_metrics()
    assert m["grid_search_used"] is True
    assert m["grid_search_metric"] == "fisher_ratio"
    assert m["grid_search_best_eps"] in config.reg_grid
    assert m["grid_search_best_score"] is not None


# =============================================================================
# 7. Grid search com COSINE_AUC funciona
# =============================================================================

def test_lda_grid_search_cosine_auc():
    ds = _build_labeled_dataset(n_classes=40, samples_per_class=5, dim=32)
    config = LDAConfig(
        use_grid_search=True,
        reg_grid=[1e-4, 1e-2],
        grid_search_metric=LdaGridSearchMetric.COSINE_AUC,
    )
    lda = _make_lda(config)
    lda.fit(ds)
    m = lda.get_execution_metrics()
    assert m["grid_search_metric"] == "cosine_auc"
    # AUC ∈ [0, 1] sob dataset separável
    assert 0.0 <= m["grid_search_best_score"] <= 1.0


# =============================================================================
# 8. Save/load round-trip preserva config e parâmetros
# =============================================================================

def test_lda_save_load_round_trip(tmp_path: Path):
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=4, dim=32)
    config = LDAConfig(
        estimator=LdaEstimatorType.LEDOIT_WOLF,
        shrinkage=0.1,
        n_components=3,
    )
    lda = _make_lda(config)
    lda.fit(ds)
    matrix_before = lda._lda_matrix.copy()
    mean_before = lda._mean_vector.copy()

    path = tmp_path / "lda.pkl"
    lda.save(str(path))

    lda2 = _make_lda()
    lda2.load(str(path))
    np.testing.assert_array_equal(lda2._lda_matrix, matrix_before)
    np.testing.assert_array_equal(lda2._mean_vector, mean_before)
    assert lda2.config.estimator == LdaEstimatorType.LEDOIT_WOLF
    assert lda2.config.shrinkage == 0.1
    assert lda2.config.n_components == 3


# =============================================================================
# 9. LDAConfig.from_dict reconstrói enums a partir de strings
# =============================================================================

def test_lda_config_from_dict_rehydrates_enums():
    data = {
        "n_components": 5,
        "estimator": "ledoit_wolf",
        "shrinkage": 0.2,
        "use_grid_search": True,
        "reg_grid": "1e-6,1e-4,1e-2",
        "grid_search_metric": "cosine_auc",
    }
    config = LDAConfig.from_dict(data)
    assert config.estimator == LdaEstimatorType.LEDOIT_WOLF
    assert config.grid_search_metric == LdaGridSearchMetric.COSINE_AUC
    assert config.reg_grid == [1e-6, 1e-4, 1e-2]
    assert config.use_grid_search is True


# =============================================================================
# 10. Validações: < 2 classes válidas → erro claro
# =============================================================================

def test_lda_requires_two_classes_with_two_samples():
    # Só 1 classe não-singleton: deve falhar (mensagem do transformator
    # OU do math service — qualquer um pega antes do eigh estourar).
    ds = _build_labeled_dataset(n_classes=1, samples_per_class=5, dim=16)
    lda = _make_lda()
    with pytest.raises(ValueError, match=r"(2 identidades|2 classes)"):
        lda.fit(ds)
