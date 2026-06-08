"""Testes do SCORE_INTERACTION_FEATURE (DTO + executor).

Cobre:

- Construção do DTO + dispatch from_dict polimórfico.
- Resolução de s0 nos três modos (AUTO_LOGREG, MANUAL, INHERIT).
- Inversão+normalização tipicidade→raridade por score_type.
- Modos LINEAR / ONE_SIDED_POSITIVE / HINGE_BASE.
- Não-equivalência ESTRUTURAL no unilateral (rank da matriz de design).
- Combiner REFERENCE_ONLY / PROBE_ONLY no TYPICALITY_PAIR_SCORING.
- Alinhamento orientado vs unordered, duplicatas, mismatch same_person.
- Validações: anti-recursão, modifier não-tipicidade, casework + AUTO_LOGREG.
- Persistência do metadata em roundtrip.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.s0_mode import S0Mode
from faceaudit.enums.score_interaction_mode import ScoreInteractionMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.score_interaction_feature_instruction_executor import (
    ScoreInteractionFeatureInstructionExecutor,
)
from faceaudit.models.calibration.logreg_s0_config import LogRegS0Config
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.score_interaction_feature_instruction import (
    ScoreInteractionFeatureInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.strategy_resource import StrategyResource


# ────────────────────────── fixtures ───────────────────────────


def _emb(face_hash: str, identity: str | None) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=np.zeros(2, dtype=np.float64),
        face_hash=face_hash,
        image_path=f"/dev/null/{face_hash}.jpg",
        identity=identity,
        file_name=f"{face_hash}.jpg",
    )


def _sample(
    probe_hash: str,
    ref_hash: str,
    raw_score: float,
    same_person: bool | None,
    probe_id: str | None = None,
    ref_id: str | None = None,
) -> SimilarityScoreSample:
    return SimilarityScoreSample(
        probe=_emb(probe_hash, probe_id or probe_hash),
        reference=_emb(ref_hash, ref_id or ref_hash),
        raw_score=raw_score,
        same_person=same_person,
    )


def _build_base_dataset(
    n_gen: int = 200,
    n_imp: int = 800,
    mean_gen: float = 0.6,
    mean_imp: float = 0.0,
    std: float = 0.15,
    seed: int = 0,
    name: str = "base_cosine",
) -> SimilarityScoreDataset:
    """Constrói um SimilarityScoreDataset cosseno sintético separável."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_gen):
        items.append(
            _sample(
                probe_hash=f"P{i:05d}",
                ref_hash=f"R{i:05d}",
                raw_score=float(rng.normal(mean_gen, std)),
                same_person=True,
                probe_id=f"id_{i}",
                ref_id=f"id_{i}",
            )
        )
    for j in range(n_imp):
        items.append(
            _sample(
                probe_hash=f"P{n_gen + j:05d}",
                ref_hash=f"R{n_gen + j:05d}",
                raw_score=float(rng.normal(mean_imp, std)),
                same_person=False,
                probe_id=f"id_g{j}",
                ref_id=f"id_h{j}",
            )
        )
    return SimilarityScoreDataset(
        name=name,
        description="base cosine fixture",
        items=items,
        metadata={
            "similarity_model_id": "cosine_v1",
            "embedding_column": "embedding_adaface_v1",
        },
    )


def _build_modifier_dataset(
    base: SimilarityScoreDataset,
    score_type: TypicalityScoreType,
    combiner: TypicalityCombiner,
    name: str = "typicality_ref",
    shift: float = 0.0,
) -> SimilarityScoreDataset:
    """Constrói um modifier alinhado posicionalmente com `base`.

    Cada sample carrega a MESMA chave (probe_hash, ref_hash) que o base
    e a mesma `same_person` — o executor deve aceitar sem reordenar.
    O valor de tipicidade é determinístico por índice para que os
    testes possam afirmar igualdade exata.
    """
    items = []
    rng = np.random.default_rng(42)
    for idx, s in enumerate(base.items):
        if score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
            # Pipeline real (TYPICALITY_PAIR_SCORING) entrega percentil
            # já normalizado em [0,1] — replicamos essa escala aqui.
            val = float(((idx % 100) / 100.0) + shift)
        elif score_type is TypicalityScoreType.MEAN_KNN_SIMILARITY:
            val = float(rng.uniform(-0.5, 0.5))
        else:
            val = float(rng.normal(-100.0, 10.0))
        items.append(
            SimilarityScoreSample(
                probe=s.probe,
                reference=s.reference,
                raw_score=val,
                same_person=s.same_person,
            )
        )
    return SimilarityScoreDataset(
        name=name,
        description="typicality pair fixture",
        items=items,
        metadata={
            "score_kind": "typicality_pair",
            "score_type": score_type.to_str(),
            "combiner": combiner.to_str(),
        },
    )


def _ctx_with(*resources: StrategyResource) -> StrategyExecutionContext:
    ctx = StrategyExecutionContext()
    for r in resources:
        ctx.set_resource(r)
    return ctx


def _as_resource(
    rid: str,
    payload,
    rtype: StrategyResourceType = StrategyResourceType.SIMILARITY_SCORES,
) -> StrategyResource:
    return StrategyResource(
        resource_id=rid, name=rid, resource_type=rtype, payload=payload,
    )


# ─────────────────────── DTO básico ────────────────────────────


def test_dto_defaults():
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
    )
    assert inst.interaction_mode is ScoreInteractionMode.ONE_SIDED_POSITIVE
    assert inst.s0_mode is S0Mode.AUTO_LOGREG
    assert inst.invert_rarity is True
    assert inst.is_casework is False
    assert inst.input_resource_ids == []
    assert inst.output_resource_ids == []


def test_dto_sync_io_with_resources():
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter_out",
    )
    assert inst.input_resource_ids == ["cos", "typ"]
    assert inst.output_resource_ids == ["inter_out"]


def test_dto_hinge_base_drops_modifier_from_inputs():
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        interaction_mode=ScoreInteractionMode.HINGE_BASE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",  # ignorado em HINGE_BASE
        output_resource_id="hinge_out",
    )
    assert inst.input_resource_ids == ["cos"]


def test_dto_inherit_adds_inherit_resource_input():
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.INHERIT,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        inherit_s0_from_resource_id="inter_cal",
        output_resource_id="inter_val",
    )
    assert "inter_cal" in inst.input_resource_ids


def test_dto_from_dict_dispatch_polymorphic():
    data = {
        "instruction_id": "sif1",
        "name": "t",
        "instruction_type": "SCORE_INTERACTION_FEATURE",
        "base_score_resource_id": "cos",
        "modifier_score_resource_id": "typ",
        "interaction_mode": "ONE_SIDED_POSITIVE",
        "s0_mode": "MANUAL",
        "s0_manual_value": 0.42,
        "output_resource_id": "inter",
    }
    inst = StrategyInstructionDefinition.from_dict(data)
    assert isinstance(inst, ScoreInteractionFeatureInstruction)
    assert inst.s0_mode is S0Mode.MANUAL
    assert inst.s0_manual_value == 0.42


def test_dto_default_validation_rule_min_pairs():
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    rule_types = [cfg.rule_type.value for cfg in inst.validation_rule_configs]
    assert "MIN_CALIBRATION_PAIRS" in rule_types

    # MANUAL/INHERIT — sem fit, não cabe default de pares mínimos.
    inst2 = ScoreInteractionFeatureInstruction(
        instruction_id="sif2",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.5,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    assert all(
        cfg.rule_type.value != "MIN_CALIBRATION_PAIRS"
        for cfg in inst2.validation_rule_configs
    )


# ─────────────────────── s0 modes (executor) ──────────────────


def test_auto_logreg_s0_matches_intercept_over_coef():
    """s0 = −intercept/coef do LogReg auxiliar univariado."""
    base = _build_base_dataset(n_gen=300, n_imp=600, seed=1)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    res = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert res.metadata["interaction_s0_mode"] == "AUTO_LOGREG"

    out = ctx.get_resource("inter").payload
    meta = out.metadata
    coef = meta["interaction_s0_coef"]
    intercept = meta["interaction_s0_intercept"]
    s0 = meta["interaction_s0"]
    assert coef > 0
    assert abs(s0 - (-intercept / coef)) < 1e-9


def test_manual_s0_passthrough():
    base = _build_base_dataset(seed=2)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    res = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert res.metadata["interaction_s0"] == 0.3
    assert res.metadata["interaction_s0_mode"] == "MANUAL"
    out = ctx.get_resource("inter").payload
    assert out.metadata["interaction_s0_training_method"] == "manual"
    assert out.metadata["interaction_s0_C"] is None


def test_inherit_s0_reads_metadata():
    base = _build_base_dataset(seed=3)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    # Constrói um "interaction dataset" anterior já carregando s0.
    prior_items = list(base.items)
    prior = SimilarityScoreDataset(
        name="prior_inter",
        items=prior_items,
        metadata={
            "is_interaction_feature": True,
            "interaction_s0": 0.55,
            "interaction_s0_coef": 1.2,
            "interaction_s0_intercept": -0.66,
            "interaction_s0_C": 1.0,
            "interaction_s0_training_method": "auto_logreg",
            "base_score_type": "cosine",
            "base_similarity_model_resource_id": "cosine_v1",
            "base_embedding_column": "embedding_adaface_v1",
        },
    )
    ctx = _ctx_with(
        _as_resource("cos", base),
        _as_resource("typ", modifier),
        _as_resource("prior", prior),
    )
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.INHERIT,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        inherit_s0_from_resource_id="prior",
        output_resource_id="inter",
    )
    res = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert res.metadata["interaction_s0"] == 0.55
    assert res.metadata["interaction_s0_mode"] == "INHERIT"
    out = ctx.get_resource("inter").payload
    assert out.metadata["interaction_s0_source_resource_id"] == "prior"


def test_inherit_fails_when_metadata_missing():
    base = _build_base_dataset(seed=4)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    prior = SimilarityScoreDataset(name="prior", items=list(base.items), metadata={})
    ctx = _ctx_with(
        _as_resource("cos", base),
        _as_resource("typ", modifier),
        _as_resource("prior", prior),
    )
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.INHERIT,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        inherit_s0_from_resource_id="prior",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("interaction_s0" in e for e in result.errors)


def test_logreg_s0_config_override_records_C_in_metadata():
    base = _build_base_dataset(seed=5)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_logreg_config_override=LogRegS0Config(C=0.05, max_iter=500),
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    out = ctx.get_resource("inter").payload
    assert out.metadata["interaction_s0_C"] == 0.05


# ─────────────────────── rarity conversion ─────────────────────


def test_to_rarity_percentile_inverts_in_0_1():
    """TYPICALITY_PAIR_SCORING já entrega percentil em [0,1] (divisão
    por 100 ocorre upstream em `get_percentiles(matrix)/100.0`). O
    executor recebe valores em [0,1] e aplica `1 − v` quando
    `invert=True`. Dividir por 100 de novo aqui produziria raridade
    saturada em ~0.99 — bug histórico já corrigido no executor."""
    typicality = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    rarity, label = ScoreInteractionFeatureInstructionExecutor._to_rarity(
        typicality, TypicalityScoreType.TYPICALITY_PERCENTILE, invert=True,
    )
    np.testing.assert_allclose(rarity, [1.0, 0.5, 0.0])
    assert label == "1 - percentile_0_1"


def test_to_rarity_mean_knn_similarity_uses_minus_v():
    typicality = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    rarity, label = ScoreInteractionFeatureInstructionExecutor._to_rarity(
        typicality, TypicalityScoreType.MEAN_KNN_SIMILARITY, invert=True,
    )
    np.testing.assert_allclose(rarity, [0.5, 0.0, -0.5])
    assert label == "−mean_knn_similarity"


def test_to_rarity_log_density_uses_minus_v():
    typicality = np.array([-150.0, -100.0, -50.0], dtype=np.float64)
    rarity, label = ScoreInteractionFeatureInstructionExecutor._to_rarity(
        typicality, TypicalityScoreType.LOG_DENSITY_ESTIMATE, invert=True,
    )
    np.testing.assert_allclose(rarity, [150.0, 100.0, 50.0])
    assert label == "−log_density_knn"


def test_modifier_without_score_type_raises():
    base = _build_base_dataset(seed=6)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    modifier.metadata.pop("score_type", None)
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    # Modifier ainda tem score_kind=typicality_pair, então passa na
    # validação intrínseca; mas o cálculo da raridade exige score_type
    # e levanta no execute.
    assert not result.success


def test_modifier_non_typicality_family_fails_validation():
    """Modifier sem score_kind=typicality_pair e sem score_type válido →
    erro estrutural na validação intrínseca."""
    base = _build_base_dataset(seed=7)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    # Apaga as duas marcas — vira "outro tipo de score".
    modifier.metadata.pop("score_kind", None)
    modifier.metadata.pop("score_type", None)
    modifier.metadata["score_kind"] = "raw_cosine"
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("tipicidade" in e for e in result.errors)


# ─────────────────────── interaction modes ─────────────────────


def test_one_sided_positive_clamps_at_s0():
    base_scores = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64)
    rarity = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    feat = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.ONE_SIDED_POSITIVE, base_scores, rarity, s0=0.4,
    )
    # Abaixo de s0=0.4 → zero; acima → (base−s0)·rarity
    np.testing.assert_allclose(feat, [0.0, 0.0, 0.1 * 0.7, 0.3 * 0.6])


def test_linear_mode_is_signed():
    base_scores = np.array([0.0, 0.2, 0.4, 0.6], dtype=np.float64)
    rarity = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    feat = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.LINEAR, base_scores, rarity, s0=0.3,
    )
    np.testing.assert_allclose(feat, [-0.3, -0.1, 0.1, 0.3])


def test_hinge_base_ignores_modifier():
    base_scores = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    feat = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.HINGE_BASE, base_scores, rarity=None, s0=0.5,
    )
    np.testing.assert_allclose(feat, [0.0, 0.0, 0.5])


# ─────────────────────── V2.13 — variantes _NEGATIVE ─────────


def test_hinge_base_negative_clamps_above_s0():
    """h_-(s) = max(s0 − s, 0): >0 só abaixo de s0; ignora modifier."""
    base_scores = np.array([0.0, 0.3, 0.5, 0.7, 1.0], dtype=np.float64)
    feat = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.HINGE_BASE_NEGATIVE,
        base_scores,
        rarity=None,
        s0=0.5,
    )
    # max(0.5 - x, 0) → [0.5, 0.2, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(feat, [0.5, 0.2, 0.0, 0.0, 0.0])


def test_one_sided_negative_clamps_above_s0():
    """h_-(s)·rarity: ativo só abaixo de s0, multiplicado pela raridade."""
    base_scores = np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64)
    rarity = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    feat = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.ONE_SIDED_NEGATIVE, base_scores, rarity, s0=0.4,
    )
    # max(0.4 - x, 0) · rarity → [0.3*0.9, 0.1*0.8, 0.0, 0.0]
    np.testing.assert_allclose(feat, [0.27, 0.08, 0.0, 0.0])


def test_hinge_sum_equals_abs_deviation():
    """h_+(s) + h_-(s) = |s − s_0| em todo o domínio (identidade exata)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(-0.5, 1.0, size=200)
    s0 = 0.27
    h_pos = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.HINGE_BASE, base, rarity=None, s0=s0,
    )
    h_neg = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.HINGE_BASE_NEGATIVE, base, rarity=None, s0=s0,
    )
    np.testing.assert_allclose(h_pos + h_neg, np.abs(base - s0), atol=1e-12)


def test_one_sided_sum_equals_abs_deviation_times_rarity():
    """os_+(s) + os_-(s) = |s − s_0|·rarity (identidade exata)."""
    rng = np.random.default_rng(7)
    n = 200
    base = rng.uniform(-0.5, 1.0, size=n)
    rarity = rng.uniform(0.0, 1.0, size=n)
    s0 = 0.31
    f_pos = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.ONE_SIDED_POSITIVE, base, rarity, s0,
    )
    f_neg = ScoreInteractionFeatureInstructionExecutor._build_feature(
        ScoreInteractionMode.ONE_SIDED_NEGATIVE, base, rarity, s0,
    )
    np.testing.assert_allclose(f_pos + f_neg, np.abs(base - s0) * rarity, atol=1e-12)


def test_negative_one_sided_requires_modifier():
    """ONE_SIDED_NEGATIVE deve falhar com rarity=None."""
    base = np.array([0.1, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="requer modifier"):
        ScoreInteractionFeatureInstructionExecutor._build_feature(
            ScoreInteractionMode.ONE_SIDED_NEGATIVE, base, rarity=None, s0=0.3,
        )


def test_enum_properties_for_negative_modes():
    """Properties is_negative_side, is_hinge_only, requires_modifier."""
    pos = ScoreInteractionMode.ONE_SIDED_POSITIVE
    neg = ScoreInteractionMode.ONE_SIDED_NEGATIVE
    hpos = ScoreInteractionMode.HINGE_BASE
    hneg = ScoreInteractionMode.HINGE_BASE_NEGATIVE
    lin = ScoreInteractionMode.LINEAR

    # is_negative_side
    assert not pos.is_negative_side
    assert neg.is_negative_side
    assert not hpos.is_negative_side
    assert hneg.is_negative_side
    assert not lin.is_negative_side

    # is_hinge_only
    assert not pos.is_hinge_only
    assert not neg.is_hinge_only
    assert hpos.is_hinge_only
    assert hneg.is_hinge_only
    assert not lin.is_hinge_only

    # requires_modifier
    assert pos.requires_modifier
    assert neg.requires_modifier
    assert not hpos.requires_modifier
    assert not hneg.requires_modifier
    assert lin.requires_modifier


def test_enum_from_str_round_trip_for_negative():
    for label in ("ONE_SIDED_NEGATIVE", "one_sided_negative", "ONE-SIDED-NEGATIVE",
                  "HINGE_BASE_NEGATIVE", "hinge_base_negative"):
        m = ScoreInteractionMode.from_str(label)
        assert m in (
            ScoreInteractionMode.ONE_SIDED_NEGATIVE,
            ScoreInteractionMode.HINGE_BASE_NEGATIVE,
        ), f"from_str({label!r}) mapped to {m!r}"


# ─────────────────────── non-equivalence structural ─────────


def test_unilateral_non_equivalence_structural():
    """No unilateral, `max(base − s0, 0)·rarity_pct` NÃO está no span de
    `{1, base, max(base−s0,0)·typicality_pct/100}`. Prova matemática
    sem depender de Cllr/fit."""
    rng = np.random.default_rng(123)
    n = 50
    base = rng.uniform(0.0, 1.0, size=n)
    typ_pct = rng.uniform(0.0, 100.0, size=n)
    s0 = 0.5

    hinge = np.maximum(base - s0, 0.0)
    rarity_pct = (1.0 - typ_pct / 100.0)
    typicality_norm = typ_pct / 100.0

    # Design matrix com [1, base, hinge·typicality_norm]
    design = np.column_stack(
        [np.ones(n), base, hinge * typicality_norm]
    )
    target = hinge * rarity_pct

    # Resíduo ortogonal a design: ‖resid‖ > 0 ⇒ target ∉ span(design)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    assert np.linalg.norm(residual) > 1e-6, (
        "Esperado: rarity unilateral NÃO está no span de "
        "[1, base, typicality unilateral] — a inversão deixa de ser cosmética."
    )


def test_linear_equivalence_structural():
    """No bilateral LINEAR, a coluna invertida ESTÁ no span de
    `{1, base, (base − s0)·typicality}` (cosmética, absorvível)."""
    rng = np.random.default_rng(456)
    n = 50
    base = rng.uniform(0.0, 1.0, size=n)
    typ_pct = rng.uniform(0.0, 100.0, size=n)
    s0 = 0.5
    typicality_norm = typ_pct / 100.0
    rarity_norm = 1.0 - typicality_norm

    design = np.column_stack(
        [np.ones(n), base, (base - s0) * typicality_norm]
    )
    target = (base - s0) * rarity_norm
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    assert np.linalg.norm(residual) < 1e-9, (
        "No LINEAR a inversão é cosmética — esperado resíduo numericamente "
        "nulo no span de [1, base, coluna original]."
    )


# ─────────────────────── alignment ─────────────────────────────


def test_pair_alignment_unordered_reorders():
    """Modifier simétrico fora de ordem → executor reordena via frozenset."""
    base = _build_base_dataset(n_gen=10, n_imp=10, seed=10)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    # Embaralha o modifier
    modifier.items = list(reversed(modifier.items))
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert result.success
    out = ctx.get_resource("inter").payload
    assert len(out.items) == len(base.items)
    # Probes preservam ordem do base (executor reusa samples da base).
    for src, derived in zip(base.items, out.items):
        assert src.probe.face_hash == derived.probe.face_hash


def test_pair_alignment_ordered_rejects_swapped_roles():
    """Modifier com combiner unilateral força chave orientada — trocar
    probe↔ref em metade dos pares deve falhar."""
    base = _build_base_dataset(n_gen=5, n_imp=5, seed=20)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.REFERENCE_ONLY,
    )
    # Inverte probe/reference em metade — quebra orientação.
    half = len(modifier.items) // 2
    for i in range(half):
        s = modifier.items[i]
        modifier.items[i] = SimilarityScoreSample(
            probe=s.reference,
            reference=s.probe,
            raw_score=s.raw_score,
            same_person=s.same_person,
        )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    # A mensagem cita orientação.
    assert any("ordered" in e or "pares" in e.lower() for e in result.errors)


def test_duplicate_keys_in_base_raise():
    base = _build_base_dataset(n_gen=5, n_imp=5, seed=30)
    # Duplica explicitamente um par no base.
    base.items.append(base.items[0])
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("duplicados" in e for e in result.errors)


def test_same_person_mismatch_rejected_in_calibration_mode():
    base = _build_base_dataset(n_gen=20, n_imp=20, seed=40)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    # Adultera same_person em ALGUNS samples do modifier.
    modifier.items[0].same_person = not modifier.items[0].same_person
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("same_person" in e or "divergem" in e for e in result.errors)


# ─────────────────────── combiner dispatch ────────────────────


def test_typicality_combiner_reference_only_helper():
    """`_combine_typicalities` lê só a tipicidade do REFERENCE."""
    from faceaudit.executors.instructions.typicality_pair_scoring_instruction_executor import (
        TypicalityPairScoringInstructionExecutor,
    )
    raw = TypicalityPairScoringInstructionExecutor._combine_typicalities(
        TypicalityCombiner.REFERENCE_ONLY, t_probe=0.1, t_reference=0.9,
    )
    assert raw == 0.9


def test_typicality_combiner_probe_only_helper():
    from faceaudit.executors.instructions.typicality_pair_scoring_instruction_executor import (
        TypicalityPairScoringInstructionExecutor,
    )
    raw = TypicalityPairScoringInstructionExecutor._combine_typicalities(
        TypicalityCombiner.PROBE_ONLY, t_probe=0.7, t_reference=0.2,
    )
    assert raw == 0.7


def test_typicality_combiner_combine_raises_for_unilateral():
    with pytest.raises(ValueError):
        TypicalityCombiner.REFERENCE_ONLY.combine(0.1, 0.2)
    with pytest.raises(ValueError):
        TypicalityCombiner.PROBE_ONLY.combine(0.1, 0.2)


def test_typicality_combiner_is_symmetric():
    assert TypicalityCombiner.MEAN.is_symmetric
    assert TypicalityCombiner.PRODUCT.is_symmetric
    assert not TypicalityCombiner.REFERENCE_ONLY.is_symmetric
    assert not TypicalityCombiner.PROBE_ONLY.is_symmetric


# ─────────────────────── casework guard ───────────────────────


def test_casework_rejects_auto_logreg():
    base = _build_base_dataset(seed=50)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
        is_casework=True,
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("casework" in e.lower() for e in result.errors)


def test_casework_inherit_zeroes_same_person():
    """is_casework=True + INHERIT funciona; output zera same_person."""
    base = _build_base_dataset(seed=60)
    # Simula dataset case (sem ground truth) — zera same_person no base.
    case_items = []
    for s in base.items:
        case_items.append(
            SimilarityScoreSample(
                probe=s.probe, reference=s.reference,
                raw_score=s.raw_score, same_person=None,
            )
        )
    case_base = SimilarityScoreDataset(
        name="case_base", items=case_items, metadata=dict(base.metadata),
    )
    modifier = _build_modifier_dataset(
        case_base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    # Modifier também sem ground truth.
    for m in modifier.items:
        m.same_person = None
    prior = SimilarityScoreDataset(
        name="prior_inter",
        items=list(base.items),
        metadata={
            "is_interaction_feature": True,
            "interaction_s0": 0.5,
            "base_score_type": "cosine",
            "base_similarity_model_resource_id": "cosine_v1",
            "base_embedding_column": "embedding_adaface_v1",
        },
    )
    ctx = _ctx_with(
        _as_resource("cos", case_base),
        _as_resource("typ", modifier),
        _as_resource("prior", prior),
    )
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.INHERIT,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        inherit_s0_from_resource_id="prior",
        output_resource_id="inter",
        is_casework=True,
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert result.success, f"errors={result.errors}"
    out = ctx.get_resource("inter").payload
    assert out.metadata["is_casework"] is True
    assert all(s.same_person is None for s in out.items)


def test_recursive_base_rejected():
    """Base com `is_interaction_feature=True` → anti-recursão."""
    base = _build_base_dataset(seed=70)
    # Marca o base como já sendo feature derivada.
    base.metadata["is_interaction_feature"] = True
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=0.3,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    result = ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    assert not result.success
    assert any("recurs" in e.lower() for e in result.errors)


# ─────────────────────── diagnostics ──────────────────────────


def test_diagnostics_fraction_hinge_active():
    """`fraction_hinge_active` é a fração de scores acima de s0 em
    modos unilaterais."""
    base = _build_base_dataset(n_gen=100, n_imp=100, mean_gen=0.6, mean_imp=0.0,
                               std=0.05, seed=80)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        combiner=TypicalityCombiner.MEAN,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        s0_mode=S0Mode.MANUAL,
        s0_manual_value=10.0,  # FORA do range — feature toda zero
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    out = ctx.get_resource("inter").payload
    assert out.metadata["fraction_hinge_active"] == 0.0
    assert out.metadata["interaction_s0_inside_score_range"] is False


# ─────────────────────── metadata payload completeness ────────


def test_metadata_carries_all_audit_fields():
    base = _build_base_dataset(seed=90)
    modifier = _build_modifier_dataset(
        base,
        score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
        combiner=TypicalityCombiner.REFERENCE_ONLY,
    )
    ctx = _ctx_with(_as_resource("cos", base), _as_resource("typ", modifier))
    inst = ScoreInteractionFeatureInstruction(
        instruction_id="sif1",
        name="t",
        instruction_type=StrategyInstructionType.SCORE_INTERACTION_FEATURE,
        base_score_resource_id="cos",
        modifier_score_resource_id="typ",
        output_resource_id="inter",
    )
    ScoreInteractionFeatureInstructionExecutor().execute(inst, ctx)
    meta = ctx.get_resource("inter").payload.metadata
    required = {
        "is_interaction_feature",
        "interaction_mode",
        "interaction_s0",
        "interaction_s0_mode",
        "interaction_s0_training_method",
        "interaction_s0_coef_sign",
        "interaction_s0_inside_score_range",
        "fraction_scores_above_s0",
        "fraction_hinge_active",
        "invert_rarity",
        "rarity_transform",
        "base_source_resource_id",
        "base_score_name",
        "modifier_source_resource_id",
        "modifier_score_type",
        "modifier_combiner",
        "alignment_orientation",
        "is_casework",
        "score_kind",
    }
    missing = required - set(meta.keys())
    assert not missing, f"campos faltantes: {missing}"
    assert meta["alignment_orientation"] == "ordered"  # REFERENCE_ONLY
    assert meta["modifier_combiner"] == "REFERENCE_ONLY"
    assert meta["rarity_transform"] == "1 - percentile_0_1"
    assert meta["score_kind"] == "interaction_feature"
