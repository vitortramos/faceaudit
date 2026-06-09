"""PR 2 — Testes da instrução unificada + migração preservativa.

Cobre:
  1. Os 4 novos enums (from_str, defaults, choices).
  2. KNNTypicalityModelConfig + MoVMFTypicalityModelConfig (defaults,
     get_hidden_fields, validação, from_dict).
  3. TypicalityModelTrainingInstruction (auto-instancia config, troca
     de tipo reconcilia, dispatcher em from_dict, prune_unused_fields).
  4. LegacyStrategyMigrationService: detect / convert_in_memory pura,
     defaults históricos materializados literais, side-by-side writer
     com original intacto.
  5. LRStrategyDefinition.from_dict aciona migração quando há sinal
     legado; modo estrito rejeita.
  6. Factory registra novo executor; legado removido.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_percentile_basis import TypicalityPercentileBasis
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.models.instructions.typicality_model_training_instruction import (
    TypicalityModelTrainingInstruction,
)
from faceaudit.models.typicality.knn_typicality_model_config import (
    KNNTypicalityModelConfig,
)
from faceaudit.models.typicality.movmf_typicality_model_config import (
    MoVMFTypicalityModelConfig,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.services.legacy_strategy_migration_service import (
    LegacyStrategyMigrationService,
    LegacyStrategyVersionRejected,
    MIGRATION_VERSION,
)


# ─────────────────────────────────────────────────────────────────────
# Enums novos
# ─────────────────────────────────────────────────────────────────────

def test_population_weighting_mode_defaults_to_identity_centroids():
    assert PopulationWeightingMode.from_str("") == PopulationWeightingMode.IDENTITY_CENTROIDS
    assert PopulationWeightingMode.from_str("FACE_LEVEL_RAW") == PopulationWeightingMode.FACE_LEVEL_RAW
    assert PopulationWeightingMode.from_str("inverse_identity_frequency") == \
        PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY


def test_unit_norm_policy_defaults_to_error():
    assert UnitNormPolicy.from_str("") == UnitNormPolicy.ERROR
    assert UnitNormPolicy.from_str("NORMALIZE_SILENTLY") == UnitNormPolicy.NORMALIZE_SILENTLY


def test_reference_score_calibration_mode_defaults_to_cross_fit():
    assert ReferenceScoreCalibrationMode.from_str("") == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    assert ReferenceScoreCalibrationMode.from_str("IN_SAMPLE_LEGACY") == \
        ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY


def test_typicality_percentile_basis_choices():
    rows = TypicalityPercentileBasis.choices_with_description()
    assert len(rows) == 2


# ─────────────────────────────────────────────────────────────────────
# KNNTypicalityModelConfig
# ─────────────────────────────────────────────────────────────────────

def test_knn_config_defaults():
    c = KNNTypicalityModelConfig()
    assert c.k == 50
    assert c.population_weighting_mode == PopulationWeightingMode.IDENTITY_CENTROIDS
    assert c.unit_norm_policy == UnitNormPolicy.ERROR
    assert c.percentile_basis_score_type == TypicalityPercentileBasis.MEAN_KNN_SIMILARITY
    # P1-R3-11 (revisão externa 2026-06-09 round 3): default mudou para
    # CROSS_FIT_IDENTITY — novas estratégias nascem com calibração
    # honesta de OOF.
    assert (
        c.reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )


def test_knn_config_round_trip():
    c = KNNTypicalityModelConfig(
        k=25, ood_quantile=0.02,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    d = c.to_dict()
    c2 = KNNTypicalityModelConfig.from_dict(d)
    assert c2.k == 25
    assert c2.population_weighting_mode == PopulationWeightingMode.FACE_LEVEL_RAW


def test_knn_config_hidden_fields_in_sample():
    # P1-R3-11: novo default é CROSS_FIT_IDENTITY (cv_n_folds visível).
    # Forçamos IN_SAMPLE_LEGACY explicitamente para validar o hide.
    c = KNNTypicalityModelConfig(
        reference_score_calibration_mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
    )
    assert "cv_n_folds" in c.get_hidden_fields()


def test_knn_config_validations():
    with pytest.raises(ValueError, match="k deve ser >= 1"):
        KNNTypicalityModelConfig(k=0)
    with pytest.raises(ValueError, match="ood_quantile"):
        KNNTypicalityModelConfig(ood_quantile=0.6)


# ─────────────────────────────────────────────────────────────────────
# MoVMFTypicalityModelConfig
# ─────────────────────────────────────────────────────────────────────

def test_movmf_config_defaults():
    c = MoVMFTypicalityModelConfig()
    assert c.n_components == 1
    assert c.population_weighting_mode == PopulationWeightingMode.IDENTITY_CENTROIDS
    assert c.percentile_basis_score_type == TypicalityPercentileBasis.LOG_DENSITY_ESTIMATE
    assert c.reference_score_calibration_mode == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY


def test_movmf_config_hidden_fields_with_single_component():
    c = MoVMFTypicalityModelConfig(n_components=1)
    hidden = c.get_hidden_fields()
    # n_init só relevante para mistura > 1 componente
    assert "n_init" in hidden


def test_movmf_config_cross_fit_requires_min_2_folds():
    with pytest.raises(ValueError, match="cv_n_folds deve ser >= 2"):
        MoVMFTypicalityModelConfig(
            reference_score_calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            cv_n_folds=1,
        )


# ─────────────────────────────────────────────────────────────────────
# TypicalityModelTrainingInstruction
# ─────────────────────────────────────────────────────────────────────

def _make_instruction(model_type=TypicalityModelType.KNN, with_io=True):
    kwargs = {
        "instruction_id": "i1",
        "name": "Train",
        "instruction_type": StrategyInstructionType.TYPICALITY_MODEL_TRAINING,
        "typicality_model_type": model_type,
    }
    if with_io:
        kwargs["input_dataset_resource_id"] = "DS"
        kwargs["trained_typicality_model_resource_id"] = "M1"
    return TypicalityModelTrainingInstruction(**kwargs)


def test_instruction_auto_instantiates_knn_config():
    i = _make_instruction()
    assert isinstance(i.typicality_model_config, KNNTypicalityModelConfig)
    assert i.uses_knn


def test_instruction_auto_instantiates_movmf_config():
    i = _make_instruction(model_type=TypicalityModelType.MOVMF)
    assert isinstance(i.typicality_model_config, MoVMFTypicalityModelConfig)
    assert i.uses_movmf


def test_instruction_sync_io():
    i = _make_instruction()
    assert i.input_resource_ids == ["DS"]
    assert i.output_resource_ids == ["M1"]
    # 2 regras de validação default (MIN_SAMPLES + MAX_SAMPLES_WARNING)
    assert len(i.validation_rule_configs) == 2


def test_instruction_from_dict_dispatches_movmf():
    data = {
        "instruction_id": "i9",
        "name": "T",
        "instruction_type": "TYPICALITY_MODEL_TRAINING",
        "typicality_model_type": "MOVMF",
        "typicality_model_config": {"n_components": 3, "n_init": 4},
        "input_dataset_resource_id": "DS",
        "trained_typicality_model_resource_id": "M1",
    }
    instr = TypicalityModelTrainingInstruction.from_dict(data)
    assert isinstance(instr.typicality_model_config, MoVMFTypicalityModelConfig)
    assert instr.typicality_model_config.n_components == 3


def test_instruction_change_type_reconciles_config():
    i = _make_instruction(model_type=TypicalityModelType.KNN)
    assert isinstance(i.typicality_model_config, KNNTypicalityModelConfig)
    i.typicality_model_type = TypicalityModelType.MOVMF
    i._reconcile_config_type()
    assert isinstance(i.typicality_model_config, MoVMFTypicalityModelConfig)


# ─────────────────────────────────────────────────────────────────────
# LegacyStrategyMigrationService
# ─────────────────────────────────────────────────────────────────────

def _legacy_strategy_json():
    return {
        "strategy_id": "s1",
        "name": "legacy",
        "resources": [],
        "preparation_instructions": [
            {
                "instruction_id": "ikn",
                "name": "kNN training",
                "instruction_type": "KNN_TYPICALITY_MODEL_TRAINING",
                "k": 25,
                "ood_quantile": 0.05,
                "reference_mode": "PER_IDENTITY",
                "input_dataset_resource_id": "DS",
                "trained_typicality_model_resource_id": "M1",
            }
        ],
        "validation_instructions": [],
        "casework_instructions": [],
        "validation_rule_ids": [],
        "embedding_column": "embedding",
        "version": "2.14",
    }


def test_migration_detect_flags_legacy_signal():
    plan = LegacyStrategyMigrationService.detect(_legacy_strategy_json())
    assert plan.is_needed
    assert "KNN_TYPICALITY_MODEL_TRAINING" in plan.legacy_signals_found
    assert len(plan.steps) == 1


def test_migration_detect_pure_does_not_mutate():
    legacy = _legacy_strategy_json()
    original_type = legacy["preparation_instructions"][0]["instruction_type"]
    LegacyStrategyMigrationService.detect(legacy)
    assert legacy["preparation_instructions"][0]["instruction_type"] == original_type


def test_migration_convert_pure_does_not_mutate_original():
    legacy = _legacy_strategy_json()
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    assert converted is not legacy
    assert legacy["preparation_instructions"][0]["instruction_type"] == "KNN_TYPICALITY_MODEL_TRAINING"
    assert converted["preparation_instructions"][0]["instruction_type"] == "TYPICALITY_MODEL_TRAINING"


def test_migration_materializes_historical_defaults_literally():
    legacy = _legacy_strategy_json()
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    instr = converted["preparation_instructions"][0]
    cfg = instr["typicality_model_config"]
    # PER_IDENTITY → IDENTITY_CENTROIDS
    assert cfg["population_weighting_mode"] == "IDENTITY_CENTROIDS"
    # k-NN sempre normalizava silenciosamente; LOO de imagem (LEGACY); base = T_k
    assert cfg["unit_norm_policy"] == "NORMALIZE_SILENTLY"
    assert cfg["reference_score_calibration_mode"] == "IN_SAMPLE_LEGACY"
    assert cfg["percentile_basis_score_type"] == "MEAN_KNN_SIMILARITY"
    assert cfg["cv_n_folds"] == 0
    # Hyperparams preservados literalmente
    assert cfg["k"] == 25
    assert cfg["ood_quantile"] == 0.05


def test_migration_per_face_maps_to_face_level_raw():
    legacy = _legacy_strategy_json()
    legacy["preparation_instructions"][0]["reference_mode"] = "PER_FACE"
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    cfg = converted["preparation_instructions"][0]["typicality_model_config"]
    assert cfg["population_weighting_mode"] == "FACE_LEVEL_RAW"


def test_migration_build_metadata():
    legacy = _legacy_strategy_json()
    plan = LegacyStrategyMigrationService.detect(legacy)
    meta = LegacyStrategyMigrationService.build_migration_metadata(plan)
    assert meta["migration_version"] == MIGRATION_VERSION
    assert "KNN_TYPICALITY_MODEL_TRAINING" in meta["migration_signals"]


def test_migration_write_side_by_side_preserves_original(tmp_path: Path):
    src = tmp_path / "strat.fa"
    src.write_text(json.dumps(_legacy_strategy_json()))
    src_mtime_before = src.stat().st_mtime

    legacy = json.loads(src.read_text())
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    target = LegacyStrategyMigrationService.write_side_by_side_copy(
        converted, src, plan=plan,
    )

    # Cópia ao lado
    assert target.exists()
    assert target.name == f"strat.{MIGRATION_VERSION}.fa"
    # Original intacto (mtime preservado)
    assert src.exists()
    assert src.stat().st_mtime == pytest.approx(src_mtime_before, abs=1.0)
    assert json.loads(src.read_text())["preparation_instructions"][0]["instruction_type"] == \
        "KNN_TYPICALITY_MODEL_TRAINING"
    # Cópia tem o tipo novo + metadata
    target_payload = json.loads(target.read_text())
    assert target_payload["preparation_instructions"][0]["instruction_type"] == \
        "TYPICALITY_MODEL_TRAINING"
    assert "_migration_metadata" in target_payload


def test_migration_write_refuses_to_overwrite(tmp_path: Path):
    src = tmp_path / "strat.fa"
    src.write_text(json.dumps(_legacy_strategy_json()))
    target = src.with_name(f"{src.stem}.{MIGRATION_VERSION}{src.suffix}")
    target.write_text("{}")

    plan = LegacyStrategyMigrationService.detect(json.loads(src.read_text()))
    converted = LegacyStrategyMigrationService.convert_in_memory(
        json.loads(src.read_text()), plan,
    )
    with pytest.raises(FileExistsError):
        LegacyStrategyMigrationService.write_side_by_side_copy(converted, src)


# ─────────────────────────────────────────────────────────────────────
# LRStrategyDefinition.from_dict + migração
# ─────────────────────────────────────────────────────────────────────

def test_lr_strategy_from_dict_auto_migrates_legacy_strategy():
    legacy = _legacy_strategy_json()
    strat = LRStrategyDefinition.from_dict(legacy)
    assert strat is not None
    assert len(strat.preparation_instructions) == 1
    instr = strat.preparation_instructions[0]
    assert isinstance(instr, TypicalityModelTrainingInstruction)
    assert instr.typicality_model_type == TypicalityModelType.KNN
    assert instr.typicality_model_config.k == 25


def test_lr_strategy_strict_mode_rejects_legacy():
    legacy = _legacy_strategy_json()
    with pytest.raises(LegacyStrategyVersionRejected):
        LRStrategyDefinition.from_dict(legacy, allow_legacy_auto_migration=False)


def test_lr_strategy_ignores_migration_metadata_block():
    """JSONs gerados pelo writer carregam `_migration_metadata` lado-a-lado
    com o envelope — o loader não deve falhar por isso.
    """
    legacy = _legacy_strategy_json()
    plan = LegacyStrategyMigrationService.detect(legacy)
    converted = LegacyStrategyMigrationService.convert_in_memory(legacy, plan)
    converted["_migration_metadata"] = (
        LegacyStrategyMigrationService.build_migration_metadata(plan)
    )
    strat = LRStrategyDefinition.from_dict(converted)
    assert strat is not None


# ─────────────────────────────────────────────────────────────────────
# Factory + dispatch
# ─────────────────────────────────────────────────────────────────────

def test_factory_registers_new_executor():
    from faceaudit.factories.strategy_executor_factory import StrategyExecutorFactory
    from faceaudit.executors.instructions.typicality_model_training_instruction_executor import (
        TypicalityModelTrainingInstructionExecutor,
    )

    registry = StrategyExecutorFactory.build_instruction_executor_registry()
    executor = registry.resolve(StrategyInstructionType.TYPICALITY_MODEL_TRAINING)
    assert isinstance(executor, TypicalityModelTrainingInstructionExecutor)


def test_factory_old_enum_value_removed():
    """KNN_TYPICALITY_MODEL_TRAINING não deve existir mais como enum."""
    assert not hasattr(
        StrategyInstructionType, "KNN_TYPICALITY_MODEL_TRAINING"
    )
