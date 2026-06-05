#!/usr/bin/env python3
"""
Demo CLI da nova instrução LR_ARCHITECTURE_VECTORIZATION + integração
com calibrador e SCORE_CALIBRATION.

Exercita o fluxo completo:

  PREPARATION:
    1) LR_ARCHITECTURE_VECTORIZATION (FIT_FULL_CALIBRATION_LEGACY)
       → vetoriza cosseno + raridades em features BILATERAL cheia.
    2) CALIBRATION_MODEL_TRAINING (LogReg)
       → treina sobre o vetor; binda vectorization_contract_fingerprint.

  VALIDATION:
    3) LR_ARCHITECTURE_VECTORIZATION (INHERIT)
       → reaplica o state literal sobre o conjunto de validação.
    4) SCORE_CALIBRATION
       → valida fingerprint vetor↔calibrador (STRICT) e produz log-LRs.

Rodar:
    python3 demo_lr_architecture.py
"""

from __future__ import annotations

import sys
import textwrap
from typing import List

import numpy as np

from faceaudit.contexts.strategy_execution_context import StrategyExecutionContext
from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.calibration_model_type import CalibrationModelType
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.executors.instructions.calibration_model_training_instruction_executor import (
    CalibrationModelTrainingInstructionExecutor,
)
from faceaudit.executors.instructions.lr_architecture_vectorization_instruction_executor import (
    LRArchitectureVectorizationInstructionExecutor,
)
from faceaudit.executors.instructions.score_calibration_instruction_executor import (
    ScoreCalibrationInstructionExecutor,
)
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.calibration_model_training_instruction import (
    CalibrationModelTrainingInstruction,
)
from faceaudit.models.instructions.lr_architecture_vectorization_instruction import (
    LRArchitectureVectorizationInstruction,
)
from faceaudit.models.instructions.score_calibration_instruction import (
    ScoreCalibrationInstruction,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample
from faceaudit.models.strategy.instruction_execution_frame import (
    InstructionExecutionFrame,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource


# ──────────────────────────────────────────────────────────────────────
# Helpers de impressão
# ──────────────────────────────────────────────────────────────────────


def hdr(title: str) -> None:
    print()
    print("═" * 72)
    print(f"  {title}")
    print("═" * 72)


def step(label: str, detail: str = "") -> None:
    print(f"\n▶ {label}")
    if detail:
        for line in textwrap.dedent(detail).strip().splitlines():
            print(f"   {line}")


def ok(msg: str) -> None:
    print(f"  ✅ {msg}")


def info(msg: str) -> None:
    print(f"  • {msg}")


def err(msg: str) -> None:
    print(f"  ❌ {msg}")


# ──────────────────────────────────────────────────────────────────────
# Geração de dados sintéticos
# ──────────────────────────────────────────────────────────────────────


def mk_emb(idx: int, ident: str) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=np.zeros(8),
        face_hash=f"h_{idx:06d}",
        identity=ident,
        image_path=f"img{idx}",
        file_name=f"img{idx}.jpg",
    )


def make_split(
    n_gen: int,
    n_imp: int,
    seed: int,
    role_label: str,
) -> tuple[SimilarityScoreDataset, SimilarityScoreDataset, SimilarityScoreDataset]:
    """
    Constrói (base, rarity_probe, rarity_reference) — pair_keys alinhados.

    role_label entra no nome para evitar colisão de identidades entre
    PREPARATION e VALIDATION quando reusarmos o mesmo gerador (ids
    disjuntas garantem que validação é genuinamente holdout).
    """
    rng = np.random.RandomState(seed)
    cos_imp = rng.normal(-0.05, 0.08, n_imp)
    cos_gen = rng.normal(0.4, 0.05, n_gen)
    cos_all = np.concatenate([cos_imp, cos_gen])
    same = [False] * n_imp + [True] * n_gen
    n = len(cos_all)

    items: List[SimilarityScoreSample] = []
    for i in range(n):
        if same[i]:
            pid = rid = f"id_{role_label}_{i//2}"
        else:
            pid = f"id_{role_label}_p_{i}"
            rid = f"id_{role_label}_r_{i}"
        items.append(
            SimilarityScoreSample(
                probe=mk_emb(2 * i, pid),
                reference=mk_emb(2 * i + 1, rid),
                raw_score=float(cos_all[i]),
                same_person=same[i],
            )
        )

    base = SimilarityScoreDataset(
        name=f"base_{role_label}",
        items=items,
        metadata={
            "score_kind": "cosine",
            "pair_role_orientation": "PROBE_REFERENCE",
            "pair_key_mode": "ORDERED",
        },
    )

    rng_t = np.random.RandomState(seed + 99)
    typ_F = rng_t.uniform(0.0, 1.0, n)
    typ_P = rng_t.uniform(0.0, 1.0, n)

    def _mk_typ(scores: np.ndarray, combiner: str) -> SimilarityScoreDataset:
        ti = [
            SimilarityScoreSample(
                probe=items[i].probe,
                reference=items[i].reference,
                raw_score=float(s),
                same_person=same[i],
            )
            for i, s in enumerate(scores)
        ]
        return SimilarityScoreDataset(
            name=f"typ_{role_label}_{combiner}",
            items=ti,
            metadata={
                "score_type": TypicalityScoreType.TYPICALITY_PERCENTILE.value,
                "combiner": combiner,
                "pair_role_orientation": "PROBE_REFERENCE",
            },
        )

    return base, _mk_typ(typ_F, "PROBE_ONLY"), _mk_typ(typ_P, "REFERENCE_ONLY")


# ──────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    hdr("Demo LR_ARCHITECTURE_VECTORIZATION (BILATERAL_FULL_CENTERED, D=7)")
    print(
        "Arquitetura: BILATERAL cheia centrada com 7 features\n"
        "  [s, h⁺, h⁻, h⁺·r̃_F, h⁺·r̃_P, h⁻·r̃_F, h⁻·r̃_P]\n"
        "Tipicidade: TYPICALITY_PERCENTILE (raridade canônica r = 1 − v_01)"
    )

    # ── 1. Dados sintéticos: PREPARATION + VALIDATION ────────────────
    step(
        "Geração de dados sintéticos",
        """
        PREPARATION: 150 genuínos + 600 impostores (~750 pares)
        VALIDATION : 100 genuínos + 400 impostores (~500 pares, ids disjuntas)
        """,
    )
    base_p, rp_p, rr_p = make_split(150, 600, seed=42, role_label="prep")
    base_v, rp_v, rr_v = make_split(100, 400, seed=43, role_label="val")
    info(f"PREPARATION: {len(base_p.items)} pares")
    info(f"VALIDATION : {len(base_v.items)} pares")

    # ── 2. Contexto + recursos ───────────────────────────────────────
    step("Setup do contexto de execução")
    ctx = StrategyExecutionContext()
    for rid, ds in [
        ("BASE", base_p),
        ("RP", rp_p),
        ("RR", rr_p),
        ("BASE_VAL", base_v),
        ("RP_VAL", rp_v),
        ("RR_VAL", rr_v),
    ]:
        ctx.set_resource(
            StrategyResource(
                resource_id=rid,
                name=rid,
                resource_type=StrategyResourceType.SIMILARITY_SCORES,
                payload=ds,
            )
        )
    ok(f"{len(ctx.resources)} recursos no contexto")

    # ── 3. PREPARATION: vetorização ──────────────────────────────────
    step(
        "PREPARATION #1 — LR_ARCHITECTURE_VECTORIZATION (FIT_FULL_CALIBRATION_LEGACY)",
        """
        Carrega base + raridades, ajusta s₀ via AUTO_LOGREG (LogReg 1D auxiliar),
        ajusta r̄_F e r̄_P como média em D_cal, e constrói matriz BILATERAL D=7.
        Persiste output vetorizado + payload bruto + state com 2 fingerprints.
        """,
    )
    instr_prep_vec = LRArchitectureVectorizationInstruction(
        instruction_id="vec_prep",
        name="Vetorização Preparação",
        instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        parameter_mode=ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
        base_score_resource_id="BASE",
        rarity_probe_resource_id="RP",
        rarity_reference_resource_id="RR",
        output_resource_id="VEC_PREP",
        output_raw_input_resource_id="RAW_PREP",
    )
    frame_prep = InstructionExecutionFrame(
        phase=StrategyPhase.PREPARATION, instruction_index=0, strategy_id="demo"
    )
    ctx_prep = ctx.with_frame(frame_prep)

    ex_vec = LRArchitectureVectorizationInstructionExecutor()
    res = ex_vec.execute(instr_prep_vec, ctx_prep)
    if not res.success:
        err(f"Falha: {res.errors}")
        return 1

    vec_ds = ctx_prep.get_resource("VEC_PREP").payload
    state = vec_ds.metadata["architecture_state"]
    ok(f"Matriz vetorizada: {vec_ds.metadata['score_vector_matrix'].shape}")
    info(f"s₀ ajustado:       {state['s0']:+.4f}")
    info(f"r̄_F (probe):      {state['r0_probe']:+.4f}")
    info(f"r̄_P (reference):  {state['r0_reference']:+.4f}")
    info(f"AUTO_LOGREG converged + stable: {state['s0_fit_summary']['is_stable']}")
    info(f"vectorization_contract_fp: {vec_ds.metadata['vectorization_contract_fingerprint'][:16]}…")
    info(f"score_vector_numeric_fp:   {vec_ds.metadata['score_vector_numeric_fingerprint'][:16]}…")
    info(f"score_vector_contract_fp:  {vec_ds.metadata['score_vector_contract_fingerprint'][:16]}…")
    raw_ds = ctx_prep.get_resource("RAW_PREP").payload
    info(f"raw payload shape: {raw_ds.matrix.shape}, immutable: {not raw_ds.matrix.flags.writeable}")

    # ── 4. PREPARATION: treinamento do calibrador ────────────────────
    step(
        "PREPARATION #2 — CALIBRATION_MODEL_TRAINING (LogReg)",
        """
        Treina LogReg sobre o vetor 7D. O executor faz gate de capacidade
        (LogReg.supports_score_vector=True, max_input_dim=None) e chama
        CalibrationModelMetadataService.bind_architecture_state(…) para
        copiar vectorization_contract_fingerprint do dataset → modelo.
        """,
    )
    cal_config = LogisticRegressionCalibrationModelConfig(
        standardize_features=True,
        use_gridsearch=False,
        C=1.0,
    )
    instr_train = CalibrationModelTrainingInstruction(
        instruction_id="train_cal",
        name="Treinar Calibrador",
        instruction_type=StrategyInstructionType.CALIBRATION_MODEL_TRAINING,
        calibration_model_type=CalibrationModelType.LOGISTIC_REGRESSION,
        input_resource_id="VEC_PREP",
        trained_calibration_model_resource_id="CAL_MODEL",
        logreg_config=cal_config,
    )
    frame_prep2 = InstructionExecutionFrame(
        phase=StrategyPhase.PREPARATION, instruction_index=1, strategy_id="demo"
    )
    ctx_prep2 = ctx.with_frame(frame_prep2)

    ex_train = CalibrationModelTrainingInstructionExecutor()
    res = ex_train.execute(instr_train, ctx_prep2)
    if not res.success:
        err(f"Falha: {res.errors}")
        return 1

    cal_model = ctx_prep2.get_resource("CAL_MODEL").payload
    cal_meta = getattr(cal_model, "metadata", {}) or {}
    ok(f"Calibrador treinado: {type(cal_model).__name__}")
    info(f"coef shape: {cal_model.logreg.coef_.shape}")
    coefs = cal_model.logreg.coef_.ravel()
    feature_names = vec_ds.metadata["feature_names"]
    print()
    print("   Coeficientes Z-padronizados:")
    for n, c in zip(feature_names, coefs):
        print(f"     {n:50s}  {c:+8.4f}")
    print()
    info(f"calibrador.vectorization_contract_fp: {cal_meta.get('vectorization_contract_fingerprint', '')[:16]}…")
    bind_ok = cal_meta.get("vectorization_contract_fingerprint") == \
              vec_ds.metadata["vectorization_contract_fingerprint"]
    ok(f"Binding fingerprint vetor↔calibrador: {'MATCH ✓' if bind_ok else 'MISMATCH ✗'}")

    # ── 5. VALIDATION: vetorização com INHERIT ───────────────────────
    step(
        "VALIDATION #1 — LR_ARCHITECTURE_VECTORIZATION (INHERIT)",
        """
        Em VALIDATION, parameter_mode=INHERIT é obrigatório. O executor
        carrega state do recurso de PREPARATION e o reaplica literalmente:
        s₀ + r̄_F + r̄_P congelados. Sem fit → sem leakage.
        """,
    )
    instr_val_vec = LRArchitectureVectorizationInstruction(
        instruction_id="vec_val",
        name="Vetorização Validação",
        instruction_type=StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION,
        architecture=LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
        parameter_mode=ArchitectureParameterMode.INHERIT,
        base_score_resource_id="BASE_VAL",
        rarity_probe_resource_id="RP_VAL",
        rarity_reference_resource_id="RR_VAL",
        inherit_architecture_state_from_resource_id="VEC_PREP",
        output_resource_id="VEC_VAL",
        output_raw_input_resource_id="RAW_VAL",
    )
    frame_val = InstructionExecutionFrame(
        phase=StrategyPhase.VALIDATION, instruction_index=0, strategy_id="demo"
    )
    ctx_val = ctx.with_frame(frame_val)
    res = ex_vec.execute(instr_val_vec, ctx_val)
    if not res.success:
        err(f"Falha: {res.errors}")
        return 1

    vec_val = ctx_val.get_resource("VEC_VAL").payload
    state_val = vec_val.metadata["architecture_state"]
    ok(f"Matriz vetorizada: {vec_val.metadata['score_vector_matrix'].shape}")
    info(f"s₀ aplicado:       {state_val['s0']:+.4f}  (mesmo de PREPARATION)")
    info(f"r̄_F aplicado:     {state_val['r0_probe']:+.4f}")
    info(f"r̄_P aplicado:     {state_val['r0_reference']:+.4f}")
    info(f"contract_fp:       {vec_val.metadata['vectorization_contract_fingerprint'][:16]}…")
    same_state = (
        vec_val.metadata["vectorization_contract_fingerprint"]
        == vec_ds.metadata["vectorization_contract_fingerprint"]
    )
    ok(f"contract_fp idêntico ao da PREPARATION: {'MATCH ✓' if same_state else 'MISMATCH ✗'}")

    # ── 6. VALIDATION: aplicação do calibrador ───────────────────────
    step(
        "VALIDATION #2 — SCORE_CALIBRATION",
        """
        Aplica o calibrador ao vetor de validação. O executor invoca
        CalibrationModelMetadataService.validate_compatibility(…) antes
        do .calibrate_scores(): se vectorization_contract_fingerprint do
        dataset ≠ do calibrador, falha STRICT.
        """,
    )
    instr_calibrate = ScoreCalibrationInstruction(
        instruction_id="calibrate",
        name="Calibrar Validação",
        instruction_type=StrategyInstructionType.SCORE_CALIBRATION,
        input_similarity_scores_resource_id="VEC_VAL",
        trained_calibration_model_resource_id="CAL_MODEL",
        output_calibrated_scores_resource_id="LOG_LRS_VAL",
    )
    frame_val2 = InstructionExecutionFrame(
        phase=StrategyPhase.VALIDATION, instruction_index=1, strategy_id="demo"
    )
    ctx_val2 = ctx.with_frame(frame_val2)

    ex_cal = ScoreCalibrationInstructionExecutor()
    res = ex_cal.execute(instr_calibrate, ctx_val2)
    if not res.success:
        err(f"Falha: {res.errors}")
        return 1

    log_lr_ds = ctx_val2.get_resource("LOG_LRS_VAL").payload
    log_lrs = []
    sps = []
    for s in log_lr_ds.items:
        m = getattr(s, "metadata", {}) or {}
        v = m.get("log_lr") if isinstance(m, dict) else None
        if v is None:
            continue
        log_lrs.append(float(v))
        sps.append(bool(s.same_person))
    log_lrs = np.array(log_lrs)
    sps = np.array(sps)

    gen_lrs = log_lrs[sps]
    imp_lrs = log_lrs[~sps]
    ok(f"Log-LRs aplicados: {len(log_lrs)} pares")
    info(f"GEN  (Hp): n={len(gen_lrs):3d}, mediana={np.median(gen_lrs):+.2f}, "
         f"min={gen_lrs.min():+.2f}, max={gen_lrs.max():+.2f}")
    info(f"IMP  (Hd): n={len(imp_lrs):3d}, mediana={np.median(imp_lrs):+.2f}, "
         f"min={imp_lrs.min():+.2f}, max={imp_lrs.max():+.2f}")
    ws_hp = float((gen_lrs < 0).mean()) * 100
    ws_hd = float((imp_lrs > 0).mean()) * 100
    info(f"WS_Hp (sinal errado em gen): {ws_hp:.2f} %")
    info(f"WS_Hd (sinal errado em imp): {ws_hd:.2f} %")

    # ── 7. Cenário negativo: dataset corrompido tenta usar calibrador ─
    step(
        "VALIDATION #3 — Binding STRICT bloqueia incompatibilidade",
        """
        Forço uma incompatibilidade no fingerprint do dataset vetorizado.
        O SCORE_CALIBRATION em modo STRICT (default) deve falhar antes de
        aplicar o calibrador.
        """,
    )
    # Corrompe o fingerprint do dataset de validação
    vec_val.metadata["vectorization_contract_fingerprint"] = "TAMPERED_FP"
    res = ex_cal.execute(instr_calibrate, ctx_val2)
    if res.success:
        err("O executor deveria ter falhado por mismatch de fingerprint")
        return 1
    ok("STRICT bloqueou a calibração:")
    for e in res.errors:
        info(e)

    # ── Resumo final ─────────────────────────────────────────────────
    hdr("Resumo")
    print(textwrap.dedent(f"""\
      • PREPARATION: 1 vetorização (FIT_FULL_CALIBRATION_LEGACY) + 1 treinamento
        de calibrador. Fingerprint vetor↔calibrador bindado com sucesso.

      • VALIDATION: 1 vetorização (INHERIT — sem refit) preservou
        vectorization_contract_fingerprint = preparação. SCORE_CALIBRATION
        validou o vínculo via CalibrationModelMetadataService antes de aplicar.

      • Defesa: tentativa de aplicação com fingerprint alterado foi
        bloqueada (modo STRICT). Para forçar compat híbrida o usuário
        precisa opt-in explicitamente em LEGACY_PERMISSIVE.

      Resultado: a nova instrução LR_ARCHITECTURE_VECTORIZATION substitui
      a sequência manual SCORE_INTERACTION_FEATURE×6 + SCORE_VECTORIZATION
      por 1 declaração. Cobre o ciclo completo preparação → validação com
      contrato auditável. Pronto para uso na UI da v2 do Strategy Manager.
    """))
    return 0


if __name__ == "__main__":
    sys.exit(main())
