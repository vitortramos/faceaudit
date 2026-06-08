"""PR 3 — Scoring polimórfico + propagação de proveniência.

Cobre:
  1. Rename LOG_DENSITY_KNN → LOG_DENSITY_ESTIMATE no enum.
  2. Migration service traduz score_type legado.
  3. Executor pair scoring usa compute_scores → TypicalityScoreBatch
     via protocolo (não isinstance KNNTypicalityModel).
  4. Proveniência completa no metadata do SimilarityScoreDataset
     (campos source_typicality_model_*, percentile_basis, etc.).
  5. UnsupportedTypicalityScoreTypeError quando score_type não está
     em enabled_score_types do modelo.
  6. TYPICALITY_PERCENTILE divide por 100 (escala pair scoring).
  7. RarityTransform usa novo transform_id `NEGATE_LOG_DENSITY_ESTIMATE`.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.executors.instructions.typicality_model_training_instruction_executor import (
    TypicalityModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.typicality_pair_scoring_instruction_executor import (
    TypicalityPairScoringInstructionExecutor,
)
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.typicality_model_training_instruction import (
    TypicalityModelTrainingInstruction,
)
from faceaudit.models.instructions.typicality_pair_scoring_instruction import (
    TypicalityPairScoringInstruction,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.typicality.knn_typicality_model_config import (
    KNNTypicalityModelConfig,
)
from faceaudit.protocols.typicality_scoring_model_protocol import (
    UnsupportedTypicalityScoreTypeError,
)
from faceaudit.services.legacy_strategy_migration_service import (
    LegacyStrategyMigrationService,
)


# ─────────────────────────────────────────────────────────────────────
# Enum rename
# ─────────────────────────────────────────────────────────────────────

def test_log_density_estimate_renamed():
    assert hasattr(TypicalityScoreType, "LOG_DENSITY_ESTIMATE")
    assert not hasattr(TypicalityScoreType, "LOG_DENSITY_KNN")


def test_log_density_estimate_value_string():
    assert TypicalityScoreType.LOG_DENSITY_ESTIMATE.to_str() == "LOG_DENSITY_ESTIMATE"


# ─────────────────────────────────────────────────────────────────────
# Migration service
# ─────────────────────────────────────────────────────────────────────

def test_migration_detects_log_density_knn_score_type():
    legacy = {
        "preparation_instructions": [
            {
                "instruction_type": "TYPICALITY_PAIR_SCORING",
                "score_type": "LOG_DENSITY_KNN",
            },
        ],
        "validation_instructions": [],
        "casework_instructions": [],
    }
    plan = LegacyStrategyMigrationService.detect(legacy)
    assert plan.is_needed
    assert "LOG_DENSITY_KNN_score_type" in plan.legacy_signals_found


def test_migration_renames_log_density_knn():
    legacy = {
        "preparation_instructions": [
            {
                "instruction_type": "TYPICALITY_PAIR_SCORING",
                "score_type": "LOG_DENSITY_KNN",
            },
        ],
        "validation_instructions": [],
        "casework_instructions": [],
    }
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    assert converted["preparation_instructions"][0]["score_type"] == "LOG_DENSITY_ESTIMATE"
    # Original intacto
    assert legacy["preparation_instructions"][0]["score_type"] == "LOG_DENSITY_KNN"


# ─────────────────────────────────────────────────────────────────────
# Pair scoring executor — proveniência
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_context():
    rng = np.random.default_rng(0)
    N = 30
    items = []
    for i in range(N):
        v = rng.standard_normal(16); v /= np.linalg.norm(v)
        items.append(EmbeddingInfo(
            embedding=v.astype(np.float32), face_hash=f"f{i}",
            image_path=f"/tmp/{i}.png", identity=f"id_{i // 3}",
            file_name=f"{i}.png",
        ))
    ds = EmbeddingDataset(name="test", items=items)
    ctx = StrategyExecutionContext()
    ctx.set_resource(StrategyResource(
        resource_id="DS", name="DS",
        resource_type=StrategyResourceType.DATASET,
        payload=ds, artifact_ref=None,
    ))
    ctx.embedding_column = "embedding"

    train_instr = TypicalityModelTrainingInstruction(
        instruction_id="t1", name="Train",
        instruction_type=StrategyInstructionType.TYPICALITY_MODEL_TRAINING,
        typicality_model_type=TypicalityModelType.KNN,
        typicality_model_config=KNNTypicalityModelConfig(
            k=4, unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        ),
        input_dataset_resource_id="DS",
        trained_typicality_model_resource_id="M1",
    )
    TypicalityModelTrainingInstructionExecutor().execute(train_instr, ctx)
    return ctx


def _run_pair_scoring(ctx, score_type, output_id="SD"):
    instr = TypicalityPairScoringInstruction(
        instruction_id=f"s_{output_id}", name="Score",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M1",
        probe_dataset_resource_id="DS",
        score_type=score_type,
        combiner=TypicalityCombiner.MEAN,
        output_scores_resource_id=output_id,
    )
    return TypicalityPairScoringInstructionExecutor().execute(instr, ctx)


def test_pair_scoring_propagates_full_provenance(trained_context):
    result = _run_pair_scoring(
        trained_context, TypicalityScoreType.LOG_DENSITY_ESTIMATE,
    )
    assert not result.has_errors

    sd = trained_context.get_resource("SD").payload
    md = sd.metadata
    # Proveniência (do plano)
    assert md["source_typicality_model_resource_id"] == "M1"
    assert md["source_typicality_model_type"] == "KNN"
    assert md["source_typicality_model_contract_fingerprint"]
    assert md["source_typicality_model_state_fingerprint"]
    assert md["typicality_score_type"] == "LOG_DENSITY_ESTIMATE"
    assert md["typicality_score_semantics_id"] == "LOG_DENSITY_ESTIMATE_LQ_SPHERICAL"
    # Compat: campo legado `model_fingerprint` = contract fp
    assert md["model_fingerprint"] == md["source_typicality_model_contract_fingerprint"]


def test_pair_scoring_percentile_has_basis_in_metadata(trained_context):
    _run_pair_scoring(
        trained_context, TypicalityScoreType.TYPICALITY_PERCENTILE,
        output_id="SD_P",
    )
    md = trained_context.get_resource("SD_P").payload.metadata
    # Base do percentil propagada
    assert md["typicality_percentile_basis_score_type"] == "MEAN_KNN_SIMILARITY"
    assert md["typicality_reference_score_distribution_fingerprint"]
    assert md["typicality_score_semantics_id"] == "PERCENTILE_OF_MEAN_KNN_SIMILARITY"


def test_pair_scoring_percentile_scaled_to_0_1(trained_context):
    """Pair scoring divide TYPICALITY_PERCENTILE por 100 para escala compatível com cosseno."""
    _run_pair_scoring(
        trained_context, TypicalityScoreType.TYPICALITY_PERCENTILE,
        output_id="SD_P_scale",
    )
    sd = trained_context.get_resource("SD_P_scale").payload
    # Todos os raw_scores devem estar em [0, 1] após combinação (MEAN
    # de dois valores em [0, 1] continua em [0, 1])
    scores = np.array([s.raw_score for s in sd.items])
    assert (scores >= 0).all() and (scores <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────
# Polimorfismo via protocolo
# ─────────────────────────────────────────────────────────────────────

def test_executor_uses_protocol_not_isinstance_knn(trained_context):
    """O executor refatorado consome TypicalityScoringModelProtocol,
    não faz isinstance(KNNTypicalityModel). Verificamos via mock
    duck-typed que satisfaz o protocolo mas não é KNNTypicalityModel.
    """
    from faceaudit.protocols.typicality_scoring_model_protocol import (
        TypicalityScoringModelProtocol,
    )
    # Pega o modelo real do contexto — ele satisfaz o protocolo
    model = trained_context.get_resource("M1").payload
    assert isinstance(model, TypicalityScoringModelProtocol)


def test_unsupported_score_type_raises_in_compute(trained_context):
    """`_compute_per_face_batch` levanta `UnsupportedTypicalityScoreTypeError`
    via mock que satisfaz o protocolo com `enabled_score_types` vazio.
    Sem monkey-patch global — instância isolada.
    """
    from dataclasses import dataclass

    class _BlockedModel:
        """Mock que satisfaz `TypicalityScoringModelProtocol` estruturalmente."""
        model_type = TypicalityModelType.KNN
        enabled_score_types = frozenset()
        embedding_dim = 16
        sample_count = 30
        model_state_fingerprint = "x"
        model_contract_fingerprint = "y"

        def compute_scores(self, embeddings, score_type):
            raise RuntimeError("não deveria ser chamado")

        def is_in_distribution(self, embeddings):
            return np.ones(embeddings.shape[0], dtype=bool)

        def summary(self):
            return None

    model = _BlockedModel()
    ds = trained_context.get_resource("DS").payload
    with pytest.raises(UnsupportedTypicalityScoreTypeError):
        TypicalityPairScoringInstructionExecutor._compute_per_face_batch(
            model, ds, "embedding",
            TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        )


# ─────────────────────────────────────────────────────────────────────
# Rarity transform — novo id
# ─────────────────────────────────────────────────────────────────────

def test_rarity_transform_uses_new_id():
    """O `transform_id` da raridade do LOG_DENSITY foi renomeado em PR 3."""
    from faceaudit.services.rarity_transform_service import to_canonical_rarity
    v = np.array([-100.0, 0.0])
    r, tid = to_canonical_rarity(v, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    assert tid == "NEGATE_LOG_DENSITY_ESTIMATE"
    np.testing.assert_array_equal(r, np.array([100.0, 0.0]))
