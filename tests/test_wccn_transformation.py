"""
Testes do transformator WCCN (Within-Class Covariance Normalization).

Cobre fit/transform, edge cases (singletons, sem labels, falhas de holdout),
estimadores (STANDARD/LEDOIT_WOLF/OAS), shrinkage, grid search (FISHER_RATIO
e COSINE_AUC), determinismo, persistência e state sync DUAL.
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.wccn_estimator_type import WccnEstimatorType
from faceaudit.enums.wccn_grid_search_metric import WccnGridSearchMetric
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.models.wccn_config import WCCNConfig
from faceaudit.pipelines.embedding_transformation_pipeline import EmbeddingTransformationPipeline
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.transformators import WCCNTransformation


# =============================================================================
# Helpers
# =============================================================================

def _build_labeled_dataset(
    name: str = "ds",
    n_classes: int = 30,
    samples_per_class: int = 5,
    dim: int = 64,
    intra_std: float = 1.0,
    inter_scale: float = 5.0,
    seed: int = 42,
    n_singletons: int = 0,
    drop_identity: bool = False,
) -> EmbeddingDataset:
    """Cria um dataset rotulado sintético com classes bem separadas."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=inter_scale, size=(n_classes, dim))
    items: List[EmbeddingInfo] = []
    for cls_i in range(n_classes):
        for s in range(samples_per_class):
            emb = means[cls_i] + rng.normal(scale=intra_std, size=dim)
            identity = None if drop_identity else f"id_{cls_i}"
            items.append(EmbeddingInfo(
                embedding=emb.astype(np.float32),
                face_hash=f"h_{cls_i}_{s}",
                image_path=f"/tmp/{cls_i}_{s}.jpg",
                identity=identity,
                file_name=f"f_{cls_i}_{s}.jpg",
            ))
    # Singletons após
    for k in range(n_singletons):
        emb = rng.normal(scale=intra_std, size=dim)
        items.append(EmbeddingInfo(
            embedding=emb.astype(np.float32),
            face_hash=f"h_solo_{k}",
            image_path=f"/tmp/solo_{k}.jpg",
            identity=f"solo_{k}",
            file_name=f"solo_{k}.jpg",
        ))
    return EmbeddingDataset(name=name, items=items)


def _make_wccn(config: Optional[WCCNConfig] = None, seed: int = 42) -> WCCNTransformation:
    return WCCNTransformation(
        math_service=EmbeddingMathService(seed=seed),
        config=config or WCCNConfig(),
    )


# =============================================================================
# 1. Sanidade: fisher ratio aumenta
# =============================================================================

def test_wccn_increases_fisher_ratio():
    ds = _build_labeled_dataset(n_classes=20, samples_per_class=5, dim=32)
    wccn = _make_wccn(WCCNConfig(regularization=1e-4))
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["fisher_ratio_after"] > m["fisher_ratio_before"]
    assert m["n_classes_used"] == 20
    assert m["n_singletons_skipped"] == 0
    assert m["output_dimension"] == 32


# =============================================================================
# 2. Singletons são descartados
# =============================================================================

def test_wccn_skips_singletons():
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=3, dim=32, n_singletons=5)
    wccn = _make_wccn()
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["n_classes_used"] == 10
    assert m["n_singletons_skipped"] == 5


# =============================================================================
# 3. Regularização salva W mal-condicionado (D >> K)
# =============================================================================

def test_wccn_regularization_handles_singular_w():
    # D=128, apenas 3 classes pequenas → W extremamente mal-condicionado
    ds = _build_labeled_dataset(n_classes=3, samples_per_class=2, dim=128)
    wccn = _make_wccn(WCCNConfig(regularization=1e-3))
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["condition_number"] is not None
    assert np.isfinite(m["condition_number"])


# =============================================================================
# 4. save/load preserva a transformação exata
# =============================================================================

def test_wccn_save_load_roundtrip(tmp_path: Path):
    ds = _build_labeled_dataset(n_classes=15, samples_per_class=4, dim=32)
    wccn = _make_wccn(WCCNConfig(regularization=1e-5, estimator=WccnEstimatorType.LEDOIT_WOLF))
    wccn.fit(ds)
    out_before = wccn.transform(ds).matrix

    path = tmp_path / "wccn.pkl"
    wccn.save(str(path))
    assert path.exists()

    wccn2 = _make_wccn()
    wccn2.load(str(path))
    out_after = wccn2.transform(ds).matrix

    np.testing.assert_array_equal(out_before, out_after)
    assert wccn2.config.estimator == WccnEstimatorType.LEDOIT_WOLF


# =============================================================================
# 5. State sync (get_state/set_state) — usado em pipeline DUAL
# =============================================================================

def test_wccn_state_sync():
    ds = _build_labeled_dataset(n_classes=15, samples_per_class=4, dim=32)
    src = _make_wccn(WCCNConfig(regularization=1e-4))
    src.fit(ds)
    out_src = src.transform(ds).matrix

    dst = _make_wccn()
    dst.set_state(src.get_state())
    out_dst = dst.transform(ds).matrix

    np.testing.assert_array_equal(out_src, out_dst)


# =============================================================================
# 5b. Pipeline DUAL com use_target_for_wccn=True
# =============================================================================

def test_pipeline_dual_with_target_for_wccn_true():
    src = _build_labeled_dataset(name="src", n_classes=20, samples_per_class=4, dim=32, seed=1)
    tgt = _build_labeled_dataset(name="tgt", n_classes=20, samples_per_class=4, dim=32, seed=2)

    cfg = TransformationPipelineConfig(
        name="dual_wccn",
        transformators=[
            TransformatorConfig(type=TransformationType.WCCN_TRANSFORMATION, config=WCCNConfig(regularization=1e-4)),
        ],
    )
    policy = PipelineTrainingPolicy(use_target_for_wccn=True, copy_shared_state_to_target=True)
    pipe = EmbeddingTransformationPipeline(
        mode=PipelineMode.DUAL,
        source_config=cfg,
        target_config=cfg,
    )
    pipe.fit_dual(source_dataset=src, target_dataset=tgt, training_policy=policy)

    src_step = pipe.source_steps[0]
    tgt_step = pipe.target_steps[0]
    # Mesma matriz nos dois branches
    np.testing.assert_array_equal(src_step._wccn_matrix, tgt_step._wccn_matrix)


# =============================================================================
# 5c. Pipeline DUAL com target sem labels deve falhar
# =============================================================================

def test_pipeline_dual_use_target_for_wccn_requires_target_labels():
    src = _build_labeled_dataset(name="src", n_classes=20, samples_per_class=4, dim=32, seed=1)
    tgt = _build_labeled_dataset(
        name="tgt", n_classes=20, samples_per_class=4, dim=32,
        seed=2, drop_identity=True,
    )

    cfg = TransformationPipelineConfig(
        name="dual_wccn_bad",
        transformators=[
            TransformatorConfig(type=TransformationType.WCCN_TRANSFORMATION, config=WCCNConfig(regularization=1e-4)),
        ],
    )
    policy = PipelineTrainingPolicy(use_target_for_wccn=True)
    pipe = EmbeddingTransformationPipeline(
        mode=PipelineMode.DUAL,
        source_config=cfg,
        target_config=cfg,
    )
    with pytest.raises(ValueError, match="use_target_for_wccn"):
        pipe.fit_dual(source_dataset=src, target_dataset=tgt, training_policy=policy)


# =============================================================================
# 6. Validação: dataset sem identity rejeitado
# =============================================================================

def test_wccn_rejects_dataset_without_identity():
    ds = _build_labeled_dataset(n_classes=5, samples_per_class=3, dim=16, drop_identity=True)
    wccn = _make_wccn()
    with pytest.raises(ValueError, match="identity"):
        wccn.fit(ds)


# =============================================================================
# 7. Cobertura dos 3 estimadores
# =============================================================================

@pytest.mark.parametrize("estimator", [
    WccnEstimatorType.STANDARD,
    WccnEstimatorType.LEDOIT_WOLF,
    WccnEstimatorType.OAS,
])
def test_wccn_estimator_variants(estimator: WccnEstimatorType):
    ds = _build_labeled_dataset(n_classes=20, samples_per_class=5, dim=32)
    wccn = _make_wccn(WCCNConfig(estimator=estimator, regularization=1e-5))
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["estimator_used"] == estimator.value
    assert np.isfinite(m["condition_number"])
    assert m["fisher_ratio_after"] > m["fisher_ratio_before"]


# =============================================================================
# 8. Shrinkage explícito
# =============================================================================

def test_wccn_shrinkage_one_yields_identity_transform():
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=3, dim=16)
    wccn = _make_wccn(WCCNConfig(shrinkage=1.0, regularization=1e-12))
    wccn.fit(ds)
    # Com shrinkage=1.0, W ≈ I → inv(W) ≈ I → chol(I) = I → B ≈ I
    B = wccn._wccn_matrix
    np.testing.assert_allclose(B, np.eye(16), atol=1e-3)


# =============================================================================
# 9. Grid Search — funcional + edge cases
# =============================================================================

def test_grid_search_basic():
    ds = _build_labeled_dataset(n_classes=50, samples_per_class=5, dim=32)
    cfg = WCCNConfig(use_grid_search=True, reg_grid=[1e-6, 1e-4, 1e-2, 1.0])
    wccn = _make_wccn(cfg, seed=42)
    wccn.fit(ds)
    m = wccn.get_execution_metrics()

    assert m["grid_search_used"] is True
    assert m["grid_search_metric"] == "fisher_ratio"
    assert m["grid_search_best_eps"] in {1e-6, 1e-4, 1e-2, 1.0}
    assert m["grid_search_best_score"] is not None
    assert len(m["grid_search_scores"]) == 4


def test_grid_search_holdout_insufficient_raises():
    # 10 classes × 0.2 = holdout de 2 → < 5 mínimo
    ds = _build_labeled_dataset(n_classes=10, samples_per_class=3, dim=16)
    cfg = WCCNConfig(use_grid_search=True, grid_search_holdout_ratio=0.2)
    wccn = _make_wccn(cfg)
    with pytest.raises(ValueError, match="Holdout insuficiente"):
        wccn.fit(ds)


def test_grid_search_fisher_ratio_metric_improves_fisher_after():
    ds = _build_labeled_dataset(n_classes=50, samples_per_class=5, dim=32)
    cfg = WCCNConfig(use_grid_search=True, grid_search_metric=WccnGridSearchMetric.FISHER_RATIO)
    wccn = _make_wccn(cfg, seed=42)
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["fisher_ratio_after"] > m["fisher_ratio_before"]


def test_grid_search_cosine_auc_metric_runs():
    ds = _build_labeled_dataset(n_classes=50, samples_per_class=5, dim=32)
    cfg = WCCNConfig(use_grid_search=True, grid_search_metric=WccnGridSearchMetric.COSINE_AUC)
    wccn = _make_wccn(cfg, seed=42)
    wccn.fit(ds)
    m = wccn.get_execution_metrics()
    assert m["grid_search_metric"] == "cosine_auc"
    # AUC ∈ [0, 1] para qualquer ε bem-sucedido
    for score in m["grid_search_scores"].values():
        if score != float("-inf"):
            assert 0.0 <= score <= 1.0


def test_grid_search_deterministic_with_same_seed():
    ds = _build_labeled_dataset(n_classes=50, samples_per_class=5, dim=32)
    cfg = WCCNConfig(use_grid_search=True)
    wccn_a = _make_wccn(cfg, seed=42)
    wccn_b = _make_wccn(cfg, seed=42)
    wccn_a.fit(ds)
    wccn_b.fit(ds)
    assert wccn_a.get_execution_metrics()["grid_search_best_eps"] == \
        wccn_b.get_execution_metrics()["grid_search_best_eps"]


# =============================================================================
# 10. Config: __post_init__ valida ranges
# =============================================================================

@pytest.mark.parametrize("kwargs,match", [
    ({"regularization": -1.0}, "regularization"),
    ({"shrinkage": -0.1}, "shrinkage"),
    ({"shrinkage": 1.1}, "shrinkage"),
    ({"use_grid_search": True, "reg_grid": []}, "reg_grid"),
    ({"use_grid_search": True, "reg_grid": [-1.0, 1.0]}, "reg_grid"),
    ({"use_grid_search": True, "grid_search_holdout_ratio": 0.01}, "holdout_ratio"),
    ({"use_grid_search": True, "grid_search_holdout_ratio": 0.9}, "holdout_ratio"),
])
def test_wccn_config_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WCCNConfig(**kwargs)


# =============================================================================
# 11. from_dict roundtrip (DTO serialization)
# =============================================================================

def test_wccn_config_from_dict_roundtrip():
    original = WCCNConfig(
        regularization=1e-3,
        shrinkage=0.1,
        estimator=WccnEstimatorType.OAS,
        use_grid_search=True,
        reg_grid=[1e-5, 1e-3, 1e-1],
        grid_search_holdout_ratio=0.3,
        grid_search_metric=WccnGridSearchMetric.COSINE_AUC,
    )
    serialized = original.to_dict()
    restored = WCCNConfig.from_dict(serialized)
    assert restored.regularization == original.regularization
    assert restored.shrinkage == original.shrinkage
    assert restored.estimator == original.estimator
    assert restored.use_grid_search == original.use_grid_search
    assert restored.reg_grid == original.reg_grid
    assert restored.grid_search_holdout_ratio == original.grid_search_holdout_ratio
    assert restored.grid_search_metric == original.grid_search_metric
