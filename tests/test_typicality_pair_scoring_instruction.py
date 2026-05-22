"""Testes do TypicalityPairScoringInstruction (DTO).

Cobre os dois modos suportados:
  - Modelo único (típico/simétrico): mesmo modelo aplicado nos dois lados do par.
  - Dois modelos (cross-população, ex.: cross-pose): modelos distintos no
    probe e no reference.

E o backward compat do campo renomeado.
"""
from __future__ import annotations

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.instructions.typicality_pair_scoring_instruction import (
    TypicalityPairScoringInstruction,
)


# ─────────────────────────── construção e IO sync ────────────────


def test_default_construction():
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
    )
    assert inst.score_type is TypicalityScoreType.MEAN_KNN_SIMILARITY
    assert inst.combiner is TypicalityCombiner.MEAN
    assert inst.typicality_model_probe_resource_id is None
    assert inst.typicality_model_reference_resource_id is None
    assert inst.uses_dual_typicality_models is False
    assert inst.input_resource_ids == []
    assert inst.output_resource_ids == []


def test_single_model_self_comparison():
    """Modelo único, sem reference dataset — auto-comparação típica."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M_pop",
        probe_dataset_resource_id="D",
        output_scores_resource_id="scores",
    )
    assert inst.input_resource_ids == ["M_pop", "D"]
    assert inst.output_resource_ids == ["scores"]
    assert inst.uses_self_comparison
    assert not inst.uses_cross_comparison
    assert not inst.uses_dual_typicality_models


def test_single_model_cross_comparison():
    """Modelo único aplicado em probe E reference (caso simétrico)."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M_pop",
        probe_dataset_resource_id="D_probe",
        reference_dataset_resource_id="D_ref",
        output_scores_resource_id="scores",
    )
    assert inst.input_resource_ids == ["M_pop", "D_probe", "D_ref"]
    assert inst.uses_cross_comparison
    assert not inst.uses_dual_typicality_models


def test_dual_models_cross_comparison():
    """Modelos distintos pro probe (lateral) e reference (frontal)."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M_lateral",
        typicality_model_reference_resource_id="M_frontal",
        probe_dataset_resource_id="D_lat",
        reference_dataset_resource_id="D_front",
        output_scores_resource_id="scores",
    )
    assert inst.input_resource_ids == ["M_lateral", "M_frontal", "D_lat", "D_front"]
    assert inst.uses_cross_comparison
    assert inst.uses_dual_typicality_models


# ─────────────────────────── resource field specs ────────────────


def test_resource_specs_without_reference_dataset():
    """Sem reference_dataset definido, o spec do model_reference não aparece."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M",
        probe_dataset_resource_id="D",
        output_scores_resource_id="scores",
    )
    specs = inst.get_resource_field_specs()
    field_names = [s.field_name for s in specs]
    assert "typicality_model_probe_resource_id" in field_names
    assert "typicality_model_reference_resource_id" not in field_names
    assert len(specs) == 4  # model_probe + 2 datasets + output


def test_resource_specs_with_reference_dataset():
    """Com reference_dataset definido, o spec do model_reference aparece."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M",
        probe_dataset_resource_id="D1",
        reference_dataset_resource_id="D2",
        output_scores_resource_id="scores",
    )
    specs = inst.get_resource_field_specs()
    field_names = [s.field_name for s in specs]
    assert "typicality_model_reference_resource_id" in field_names
    assert len(specs) == 5  # +1 spec quando há reference dataset


# ─────────────────────────── validation rules ────────────────────


def test_validation_rules_self_comparison():
    """Self: só checa probe model fitted + dim do probe dataset."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M",
        probe_dataset_resource_id="D",
        output_scores_resource_id="scores",
    )
    rules = inst.build_validation_rules()
    rule_ids = [r.rule_id for r in rules]
    assert "tps1_probe_model_fitted" in rule_ids
    assert "tps1_dim_compat_probe" in rule_ids
    assert "tps1_ref_model_fitted" not in rule_ids
    assert "tps1_dim_compat_reference" not in rule_ids


def test_validation_rules_cross_single_model():
    """Cross sem ref model: dim do reference roda contra o probe model
    (fallback). NÃO há ref_model_fitted (não há modelo distinto)."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M",
        probe_dataset_resource_id="D1",
        reference_dataset_resource_id="D2",
        output_scores_resource_id="scores",
    )
    rules = inst.build_validation_rules()
    rule_ids = [r.rule_id for r in rules]
    assert "tps1_probe_model_fitted" in rule_ids
    assert "tps1_dim_compat_probe" in rule_ids
    assert "tps1_dim_compat_reference" in rule_ids
    assert "tps1_ref_model_fitted" not in rule_ids


def test_validation_rules_cross_dual_models():
    """Cross com 2 modelos: 4 regras (fitted x 2 + dim x 2)."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M_lat",
        typicality_model_reference_resource_id="M_front",
        probe_dataset_resource_id="D_lat",
        reference_dataset_resource_id="D_front",
        output_scores_resource_id="scores",
    )
    rules = inst.build_validation_rules()
    rule_ids = [r.rule_id for r in rules]
    assert len(rules) == 4
    assert "tps1_probe_model_fitted" in rule_ids
    assert "tps1_ref_model_fitted" in rule_ids
    assert "tps1_dim_compat_probe" in rule_ids
    assert "tps1_dim_compat_reference" in rule_ids


# ─────────────────────────── backward compat ─────────────────────


def test_legacy_field_name_migrated():
    """JSON com `trained_typicality_model_resource_id` (nome antigo)
    é migrado pra `typicality_model_probe_resource_id`."""
    data = {
        "instruction_id": "tps_old", "name": "old",
        "instruction_type": "TYPICALITY_PAIR_SCORING",
        "trained_typicality_model_resource_id": "M_old",
        "probe_dataset_resource_id": "D",
        "output_scores_resource_id": "scores",
    }
    inst = StrategyInstructionDefinition.from_dict(data)
    assert isinstance(inst, TypicalityPairScoringInstruction)
    assert inst.typicality_model_probe_resource_id == "M_old"
    assert inst.typicality_model_reference_resource_id is None
    # Atributo legado não deve sobreviver no DTO.
    assert not hasattr(inst, "trained_typicality_model_resource_id")


def test_legacy_embedding_column_removed():
    """Campo legado embedding_column é descartado silenciosamente."""
    data = {
        "instruction_id": "tps", "name": "t",
        "instruction_type": "TYPICALITY_PAIR_SCORING",
        "typicality_model_probe_resource_id": "M",
        "embedding_column": "embedding_v2",  # legado
        "probe_dataset_resource_id": "D",
        "output_scores_resource_id": "scores",
    }
    inst = StrategyInstructionDefinition.from_dict(data)
    assert isinstance(inst, TypicalityPairScoringInstruction)
    assert not hasattr(inst, "embedding_column")


def test_legacy_does_not_override_new_field_if_both_present():
    """Se JSON tem AMBOS (situação extrema), o novo nome prevalece."""
    data = {
        "instruction_id": "tps", "name": "t",
        "instruction_type": "TYPICALITY_PAIR_SCORING",
        "typicality_model_probe_resource_id": "M_new",
        "trained_typicality_model_resource_id": "M_old",
        "probe_dataset_resource_id": "D",
        "output_scores_resource_id": "scores",
    }
    inst = StrategyInstructionDefinition.from_dict(data)
    assert inst.typicality_model_probe_resource_id == "M_new"


def test_roundtrip_dual_models():
    """to_dict / from_dict preservam ambos modelos."""
    inst = TypicalityPairScoringInstruction(
        instruction_id="tps1", name="t",
        instruction_type=StrategyInstructionType.TYPICALITY_PAIR_SCORING,
        typicality_model_probe_resource_id="M_lat",
        typicality_model_reference_resource_id="M_front",
        probe_dataset_resource_id="D_lat",
        reference_dataset_resource_id="D_front",
        score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
        combiner=TypicalityCombiner.MEAN,
        output_scores_resource_id="scores",
    )
    data = inst.to_dict()
    inst2 = StrategyInstructionDefinition.from_dict(data)
    assert isinstance(inst2, TypicalityPairScoringInstruction)
    assert inst2.typicality_model_probe_resource_id == "M_lat"
    assert inst2.typicality_model_reference_resource_id == "M_front"
    assert inst2.score_type is TypicalityScoreType.TYPICALITY_PERCENTILE
    assert inst2.combiner is TypicalityCombiner.MEAN
