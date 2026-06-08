"""Confirma que `_ensure_default_validation_rules` reconstrói os params
quando o usuário muda os input fields da instrução.

Bug original: `if self.validation_rule_configs: return` no início do método
travava a regra com os IDs dos datasets antigos, mesmo após o usuário
trocar pra outros datasets no dropdown.

Cobertura:
  - PAIRED_DATASET_SPLIT (DATASET_OVERLAP_REQUIRED)
  - CALIBRATION_MODEL_TRAINING (MIN_CALIBRATION_PAIRS + SCORE_CALIBRATION_STATE)
  - DATASET_SPLIT (muda split_mode: identity_ratio → index_ratio → supervised)
  - Preservação quando user_managed=True
  - Clear quando input volta a None
"""
from __future__ import annotations

# Força sys.path a apontar pra raiz deste worktree ANTES de importar faceaudit.
# Sem isso, o editable-install do venv (apontado pro projeto pai) faria as
# imports resolverem pro código não-modificado, e o teste validaria o bug
# em vez do fix.
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))

from faceaudit.enums.dataset_split_mode import DatasetSplitMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.calibration_model_training_instruction import (
    CalibrationModelTrainingInstruction,
)
from faceaudit.models.instructions.dataset_split_instruction import (
    DatasetSplitInstruction,
)
from faceaudit.models.instructions.paired_dataset_split_instruction import (
    PairedDatasetSplitInstruction,
)


def test_paired_split_rebuilds_overlap_params_on_dataset_change():
    instr = PairedDatasetSplitInstruction(
        instruction_id="i1",
        name="paired",
        instruction_type=StrategyInstructionType.PAIRED_DATASET_SPLIT,
        primary_dataset_resource_id="frontais_v1",
        secondary_dataset_resource_id="laterais_v1",
        output_primary_a_resource_id="pa",
        output_primary_b_resource_id="pb",
        output_secondary_a_resource_id="sa",
        output_secondary_b_resource_id="sb",
    )
    # __post_init__ ja chamou _sync_declared_io → defaults populados.
    assert len(instr.validation_rule_configs) == 1
    cfg = instr.validation_rule_configs[0]
    assert cfg.rule_type == ValidationRuleType.DATASET_OVERLAP_REQUIRED
    assert cfg.params["dataset_a_resource_id"] == "frontais_v1"
    assert cfg.params["dataset_b_resource_id"] == "laterais_v1"

    # Usuario troca os datasets pra V2 e re-sincroniza.
    instr.primary_dataset_resource_id = "frontais_v2"
    instr.secondary_dataset_resource_id = "laterais_v2"
    instr._sync_declared_io()

    assert len(instr.validation_rule_configs) == 1
    cfg_new = instr.validation_rule_configs[0]  # re-fetch: instancia nova
    assert cfg_new.params["dataset_a_resource_id"] == "frontais_v2", \
        f"Regra deve apontar pro dataset novo (frontais_v2), nao pra " \
        f"{cfg_new.params['dataset_a_resource_id']}"
    assert cfg_new.params["dataset_b_resource_id"] == "laterais_v2", \
        f"Regra deve apontar pro dataset novo (laterais_v2), nao pra " \
        f"{cfg_new.params['dataset_b_resource_id']}"
    print("[OK] PAIRED_DATASET_SPLIT: params da regra DATASET_OVERLAP_REQUIRED "
          "acompanham mudanca dos datasets.")


def test_calibration_training_rebuilds_input_params_on_change():
    instr = CalibrationModelTrainingInstruction(
        instruction_id="i2",
        name="cal",
        instruction_type=StrategyInstructionType.CALIBRATION_MODEL_TRAINING,
        input_resource_id="scores_v1",
        trained_calibration_model_resource_id="cal_v1",
    )
    # 2 defaults pré-execução: MIN_CALIBRATION_PAIRS, SCORE_CALIBRATION_STATE.
    # CONDITIONAL_MONOTONICITY foi movido para auditoria embutida no executor
    # (V1.5) — não aparece mais como regra de pré-execução porque precisa do
    # calibrador refitado para avaliar.
    assert len(instr.validation_rule_configs) == 2
    for cfg in instr.validation_rule_configs:
        assert cfg.params["scores_resource_id"] == "scores_v1"

    instr.input_resource_id = "scores_v2"
    instr._sync_declared_io()

    assert len(instr.validation_rule_configs) == 2
    for cfg in instr.validation_rule_configs:
        assert cfg.params["scores_resource_id"] == "scores_v2", \
            f"Regra {cfg.rule_type} ainda aponta pro scores antigo"
    print("[OK] CALIBRATION_MODEL_TRAINING: params (MIN_CALIBRATION_PAIRS + "
          "SCORE_CALIBRATION_STATE) acompanham mudanca do input.")


def test_dataset_split_rebuilds_rules_on_mode_change():
    instr = DatasetSplitInstruction(
        instruction_id="i3",
        name="split",
        instruction_type=StrategyInstructionType.DATASET_SPLIT,
        dataset_resource_id="ds1",
        split_mode=DatasetSplitMode.BY_IDENTITY_RATIO,
        output_dataset_a_resource_id="a",
        output_dataset_b_resource_id="b",
    )
    # BY_IDENTITY_RATIO → MIN_IDENTITIES
    assert len(instr.validation_rule_configs) == 1
    assert instr.validation_rule_configs[0].rule_type == ValidationRuleType.MIN_IDENTITIES

    # Muda pra BY_INDEX_RATIO → deve virar MIN_SAMPLES (regra DIFERENTE).
    instr.split_mode = DatasetSplitMode.BY_INDEX_RATIO
    instr._sync_declared_io()
    assert len(instr.validation_rule_configs) == 1
    assert instr.validation_rule_configs[0].rule_type == ValidationRuleType.MIN_SAMPLES, \
        f"Esperado MIN_SAMPLES, veio {instr.validation_rule_configs[0].rule_type}"

    # Muda pra BY_SUPERVISED_ADAPTATION com target → 3 regras.
    instr.split_mode = DatasetSplitMode.BY_SUPERVISED_ADAPTATION
    instr.target_dataset_resource_id = "ds2"
    instr._sync_declared_io()
    types = [cfg.rule_type for cfg in instr.validation_rule_configs]
    assert ValidationRuleType.MIN_IDENTITIES in types
    assert ValidationRuleType.DATASET_OVERLAP_REQUIRED in types
    print("[OK] DATASET_SPLIT: regras refletem split_mode atual "
          "(MIN_IDENTITIES <-> MIN_SAMPLES <-> SDA com 3 regras).")


def test_user_managed_preserves_user_choice():
    instr = PairedDatasetSplitInstruction(
        instruction_id="i4",
        name="paired",
        instruction_type=StrategyInstructionType.PAIRED_DATASET_SPLIT,
        primary_dataset_resource_id="frontais",
        secondary_dataset_resource_id="laterais",
    )
    # Usuario assumiu controle e ESVAZIOU a lista (botao lixeira).
    instr.validation_rule_configs = []
    instr.validation_rules_user_managed = True

    instr._sync_declared_io()
    assert instr.validation_rule_configs == [], \
        "user_managed=True deve preservar lista vazia do usuario"

    # Tambem nao deve repopular ao mudar dataset.
    instr.primary_dataset_resource_id = "frontais_v2"
    instr._sync_declared_io()
    assert instr.validation_rule_configs == [], \
        "user_managed=True deve continuar preservando mesmo apos mudanca"
    print("[OK] user_managed=True preserva escolha do usuario (lista vazia).")


def test_clears_rules_when_input_becomes_none():
    instr = CalibrationModelTrainingInstruction(
        instruction_id="i5",
        name="cal",
        instruction_type=StrategyInstructionType.CALIBRATION_MODEL_TRAINING,
        input_resource_id="scores",
        trained_calibration_model_resource_id="cal",
    )
    assert len(instr.validation_rule_configs) > 0

    # Usuario apaga o input no dropdown.
    instr.input_resource_id = None
    instr._sync_declared_io()
    assert instr.validation_rule_configs == [], \
        "Lista deve ficar vazia quando o input field e desfeito"
    print("[OK] Limpa regras quando input volta a None.")


if __name__ == "__main__":
    test_paired_split_rebuilds_overlap_params_on_dataset_change()
    test_calibration_training_rebuilds_input_params_on_change()
    test_dataset_split_rebuilds_rules_on_mode_change()
    test_user_managed_preserves_user_choice()
    test_clears_rules_when_input_becomes_none()
    print("\nTodos os testes passaram.")
