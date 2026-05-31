# faceaudit/executors/instructions/strategy_evaluation_instruction_executor.py

from __future__ import annotations

import time
from typing import List

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_evaluation_instruction import (
    StrategyEvaluationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.validation_math_service import ValidationMathService


class StrategyEvaluationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução STRATEGY_EVALUATION.

    Recebe um SimilarityScoreDataset CALIBRADO (com `metadata["log_lr"]`
    populado por sample) e produz um StrategyValidationResult contendo
    métricas escalares (EER, Cllr, Cllr_min via PAVA, AUC) e dados de
    plot (DET, Tippett, ECE).

    Delega todo o cálculo a `ValidationMathService.calculate_metrics`.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, StrategyEvaluationInstruction):
            return self._failure_result(
                "O executor StrategyEvaluationInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando avaliação de estratégia: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            scores_resource = context.get_resource(
                instruction.input_calibrated_scores_resource_id
            )
            score_dataset: SimilarityScoreDataset = scores_resource.payload

            # Separa log-LRs E raw_scores por classe a partir de sample.label
            # e sample.metadata["log_lr"]. Iteração em CHUNKS com micro-pausa
            # entre eles para liberar a GIL e manter a UI responsiva durante
            # datasets de milhões de pares. Raw_scores ficam alinhados aos
            # log_lrs (mesmos índices i) — viabilizam a Curva de Calibração
            # (5ª aba do ValidationResultWidget).
            #
            # `gen_idx` / `imp_idx` rastreiam as posições originais dos
            # itens em `score_dataset.items` — usados depois pra extrair
            # os score_vectors correspondentes do
            # `metadata["score_vector_matrix"]` (D-dim, quando o input
            # veio de SCORE_VECTORIZATION).
            gen_log_lrs: List[float] = []
            imp_log_lrs: List[float] = []
            gen_raw_scores: List[float] = []
            imp_raw_scores: List[float] = []
            gen_idx: List[int] = []
            imp_idx: List[int] = []
            n_skipped = 0
            items = score_dataset.items
            n = len(items)
            CHUNK = 50_000
            for chunk_start in range(0, n, CHUNK):
                chunk_end = min(chunk_start + CHUNK, n)
                for j in range(chunk_start, chunk_end):
                    sample = items[j]
                    md = getattr(sample, "metadata", None)
                    v = md.get("log_lr") if isinstance(md, dict) else None
                    if v is None:
                        n_skipped += 1
                        continue
                    raw = float(sample.raw_score)
                    if sample.label == 1:
                        gen_log_lrs.append(float(v))
                        gen_raw_scores.append(raw)
                        gen_idx.append(j)
                    else:
                        imp_log_lrs.append(float(v))
                        imp_raw_scores.append(raw)
                        imp_idx.append(j)
                # Micro-pausa: cede GIL à main thread.
                time.sleep(0)

            if n_skipped:
                self._log(
                    context,
                    f"[{instruction.name}] ⚠️ {n_skipped} sample(s) sem log_lr "
                    "ignorado(s) na avaliação."
                )

            if not gen_log_lrs or not imp_log_lrs:
                return self._failure_result(
                    "Avaliação requer ambas as classes (genuínos E impostores) com "
                    f"log_lr preenchido. Encontrado: {len(gen_log_lrs)} genuíno(s), "
                    f"{len(imp_log_lrs)} impostor(es)."
                )

            strategy_id = context.get_metadata("strategy_id") or instruction.instruction_id
            strategy_name = context.get_metadata("strategy_name") or instruction.name

            self._log(
                context,
                f"[{instruction.name}] 📈 Calculando métricas "
                f"(N_gen={len(gen_log_lrs):,}, N_imp={len(imp_log_lrs):,})…"
                .replace(",", ".")
            )

            # Lê telemetria CV do calibrador propagada via SCORE_CALIBRATION.
            # Valores são opcionais — None quando calibrador foi treinado sem
            # CV (k_folds=0). Service só transporta pro ValidationMetrics.
            ds_md = score_dataset.metadata if isinstance(score_dataset.metadata, dict) else {}

            # Score vectors D-dim (se input veio de SCORE_VECTORIZATION).
            # Permite render da Curva de Calibração 2D (heatmap log-LR
            # no plano dos sistemas combinados) ou supressão pra D ≥ 3.
            # `metadata["score_vector_matrix"]` é (N, D) alinhada com
            # `items[i]`; recortamos pelas idx gen/imp coletadas acima.
            gen_score_vectors = None
            imp_score_vectors = None
            axis_labels: List[str] = []
            score_vector_matrix = ds_md.get("score_vector_matrix")
            if score_vector_matrix is not None:
                try:
                    import numpy as _np
                    svm = _np.asarray(score_vector_matrix, dtype=_np.float64)
                    if svm.ndim == 2 and svm.shape[0] == n:
                        gen_score_vectors = svm[gen_idx, :]
                        imp_score_vectors = svm[imp_idx, :]
                        # Labels dos eixos vêm dos source resources (nomes
                        # dos modelos de similaridade que produziram cada
                        # coluna). Quando ausente cai em "x₀", "x₁", ...
                        sources = ds_md.get("score_vector_source_resources")
                        if isinstance(sources, (list, tuple)) and len(sources) == svm.shape[1]:
                            axis_labels = [str(s) for s in sources]
                except Exception as exc:
                    self._log(
                        context,
                        f"[{instruction.name}] ⚠️ Falha ao extrair "
                        f"score_vector_matrix pra plot 2D: {exc}",
                    )

            # Coeficientes da LogReg propagados pelo SCORE_CALIBRATION —
            # permitem superfície analítica + equação no plot 2D.
            calibrator_coef_raw = ds_md.get("calibrator_coef_raw")
            calibrator_intercept_raw = ds_md.get("calibrator_intercept_raw")

            # Estado piecewise-linear (V1, mode=PIECEWISE_LINEAR no LogReg).
            # Permite o math service desenhar a curva analiticamente com
            # quebras visíveis em cada knot — sem isso, np.interp em cima
            # dos observados colapsa as N dobras numa só visualmente.
            piecewise_enabled = bool(ds_md.get("piecewise_enabled"))
            hinge_anchor_s0 = ds_md.get("hinge_anchor_s0")
            piecewise_knots_absolute = ds_md.get("piecewise_knots_absolute")

            service = ValidationMathService()
            result = service.calculate_metrics(
                strategy_id=str(strategy_id),
                strategy_name=str(strategy_name),
                strategy_type="V2",
                genuine_scores=gen_log_lrs,
                impostor_scores=imp_log_lrs,
                genuine_raw_scores=gen_raw_scores,
                impostor_raw_scores=imp_raw_scores,
                genuine_score_vectors=gen_score_vectors,
                impostor_score_vectors=imp_score_vectors,
                score_vector_axis_labels=axis_labels,
                calibrator_coef_raw=calibrator_coef_raw,
                calibrator_intercept_raw=calibrator_intercept_raw,
                piecewise_enabled=piecewise_enabled,
                hinge_anchor_s0=hinge_anchor_s0,
                piecewise_knots_absolute=piecewise_knots_absolute,
                calibrator_cllr_train=ds_md.get("calibrator_cllr_train"),
                calibrator_cllr_train_gen=ds_md.get("calibrator_cllr_train_gen"),
                calibrator_cllr_train_imp=ds_md.get("calibrator_cllr_train_imp"),
                calibrator_cllr_cv=ds_md.get("calibrator_cllr_cv"),
                calibrator_cllr_cv_gen=ds_md.get("calibrator_cllr_cv_gen"),
                calibrator_cllr_cv_imp=ds_md.get("calibrator_cllr_cv_imp"),
            )

            output_resource_id = instruction.output_validation_result_resource_id
            output_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.VALIDATION_RESULT,
                payload=result,
                artifact_ref=None,
            )
            context.set_resource(output_resource)

            metrics = result.metrics
            cal_loss = max(0.0, metrics.cllr - metrics.cllr_min)

            self._log(
                context,
                f"[{instruction.name}] ✅ Avaliação concluída — "
                f"EER={metrics.eer*100:.2f}% | "
                f"Cllr={metrics.cllr:.3f} | "
                f"Cllr_min={metrics.cllr_min:.3f} | "
                f"cal_loss={cal_loss:.3f} | "
                f"AUC={metrics.auc:.4f}"
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_calibrated_scores_resource_id":
                        instruction.input_calibrated_scores_resource_id,
                    "output_validation_result_resource_id": output_resource_id,
                    "n_samples": int(len(gen_log_lrs) + len(imp_log_lrs)),
                    "n_genuine": int(len(gen_log_lrs)),
                    "n_impostor": int(len(imp_log_lrs)),
                    "eer": float(metrics.eer),
                    "eer_threshold": float(metrics.eer_threshold),
                    "cllr": float(metrics.cllr),
                    "cllr_min": float(metrics.cllr_min),
                    "cal_loss": float(cal_loss),
                    "auc": float(metrics.auc) if metrics.auc is not None else None,
                },
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de avaliação "
                f"{instruction.instruction_id}: {exc}"
            )
            return self._failure_result(str(exc))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, StrategyEvaluationInstruction):
            result.add_error(
                "A instrução informada não é do tipo StrategyEvaluationInstruction."
            )
            return result

        # ─── Input: CALIBRATED_SCORES ─────────────────────────────────────
        scores_id = instruction.input_calibrated_scores_resource_id
        if not scores_id:
            result.add_error(
                "A instrução deve definir 'input_calibrated_scores_resource_id'."
            )
        elif not context.has_resource(scores_id):
            result.add_error(
                f"O recurso de scores calibrados não foi encontrado no contexto: "
                f"'{scores_id}'."
            )
        else:
            scores_resource = context.get_resource(scores_id)
            if scores_resource is None:
                result.add_error(
                    f"O recurso '{scores_id}' está registrado mas é nulo no contexto."
                )
            elif scores_resource.payload is None:
                ref_path = (
                    scores_resource.artifact_ref.path
                    if scores_resource.artifact_ref is not None else None
                )
                if ref_path:
                    result.add_error(
                        f"O recurso '{scores_id}' tem artifact_ref ('{ref_path}') "
                        "mas o payload está vazio — falha ao carregar do disco. "
                        "Re-execute a instrução SCORE_CALIBRATION que produz este recurso."
                    )
                else:
                    result.add_error(
                        f"O recurso '{scores_id}' ainda não foi produzido. "
                        "Execute SCORE_CALIBRATION antes."
                    )
            elif not isinstance(scores_resource.payload, SimilarityScoreDataset):
                result.add_error(
                    f"O recurso '{scores_id}' contém um payload do tipo "
                    f"{type(scores_resource.payload).__name__}, esperado "
                    "SimilarityScoreDataset."
                )
            else:
                ds: SimilarityScoreDataset = scores_resource.payload

                # Rejeita explicitamente datasets de casework — sem ground
                # truth não há como calcular EER/Cllr/AUC.
                if ds.metadata.get("is_casework"):
                    result.add_error(
                        f"O recurso '{scores_id}' está marcado como "
                        "is_casework=True (sem ground truth). "
                        "STRATEGY_EVALUATION exige rótulos genuíno/impostor "
                        "para calcular EER/Cllr/AUC e só pode operar sobre "
                        "CALIBRATED_SCORES produzidos por SIMILARITY_SCORING "
                        "com is_casework=False."
                    )
                else:
                    # Verifica se ao menos UM sample tem log_lr.
                    has_log_lr = any(
                        isinstance(getattr(s, "metadata", None), dict)
                        and s.metadata.get("log_lr") is not None
                        for s in ds.items
                    )
                    if not has_log_lr:
                        result.add_error(
                            f"O recurso '{scores_id}' não contém scores calibrados — "
                            "nenhum sample tem 'log_lr' em metadata. Execute "
                            "SCORE_CALIBRATION antes da avaliação."
                        )

        # ─── Output ────────────────────────────────────────────────────────
        if not instruction.output_validation_result_resource_id:
            result.add_error(
                "A instrução deve definir 'output_validation_result_resource_id'."
            )

        if (
            instruction.output_validation_result_resource_id
            and instruction.output_validation_result_resource_id
            == instruction.input_calibrated_scores_resource_id
        ):
            result.add_warning(
                "output_validation_result_resource_id coincide com o recurso "
                "de scores de entrada. O contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
