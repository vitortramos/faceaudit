# faceaudit/executors/instructions/score_calibration_instruction_executor.py

from __future__ import annotations

import copy
import time
from typing import List

import numpy as np

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.score_calibration_instruction import (
    ScoreCalibrationInstruction,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.calibration_model_protocol import CalibrationModelProtocol
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


class ScoreCalibrationInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução SCORE_CALIBRATION.

    Aplica um modelo de calibração TREINADO a um SimilarityScoreDataset
    (raw_scores) e produz um novo dataset com `sample.metadata["log_lr"]`
    populado para cada amostra. O output mantém todos os outros campos
    do input (raw_score, same_person, probe/reference) preservados.
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, ScoreCalibrationInstruction):
            return self._failure_result(
                "O executor ScoreCalibrationInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de calibração de scores: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            scores_resource = context.get_resource(
                instruction.input_similarity_scores_resource_id
            )
            calibrator_resource = context.get_resource(
                instruction.trained_calibration_model_resource_id
            )

            score_dataset: SimilarityScoreDataset = scores_resource.payload
            calibrator: CalibrationModelProtocol = calibrator_resource.payload

            n_samples = len(score_dataset.items)
            self._log(
                context,
                f"[{instruction.name}] 🎚️ Calibrando {n_samples:,} pares "
                f"com {type(calibrator).__name__}…".replace(",", ".")
            )

            # Vetorização: scores e calibração em um único passe.
            # Para datasets produzidos por SCORE_VECTORIZATION, extraímos
            # a matriz (N, D) do vetor de scores em vez do raw_score escalar.
            raw_scores = (
                score_dataset.extract_score_vector()
                if score_dataset.is_score_vector
                else score_dataset.scores  # np.ndarray float64
            )
            log_lrs = np.asarray(
                calibrator.calibrate_scores(raw_scores), dtype=np.float64
            )

            if log_lrs.shape[0] != n_samples:
                return self._failure_result(
                    f"Calibrator retornou {log_lrs.shape[0]} log-LRs para "
                    f"{n_samples} samples — tamanhos incompatíveis."
                )

            # Constrói samples NOVOS para o output em chunks com micro-pausa.
            #
            # Premissa errada da versão anterior: "mutação in-place é segura
            # porque o input não será reusado depois". Falha porque o Run All
            # passa por `_persist_trained_model_resources_inline`, que itera
            # `trained_resources` (TODOS os recursos no contexto, inclusive o
            # input) e regrava os que ainda não tinham `artifact_ref` no
            # início da execução. Como a mutação adicionou `log_lr` no
            # metadata dos items compartilhados, o parquet do input ia parar
            # no disco com a coluna `log_lr` populada — fazendo a validação
            # `SCORE_CALIBRATION_STATE(RAW)` falhar na próxima execução.
            #
            # Solução: shallow copy de cada sample + dict de metadata NOVO.
            # `copy.copy` em ~5M samples leva poucos segundos com chunked
            # sleep — mesma ordem de grandeza da mutação in-place — e libera
            # a GIL periodicamente, evitando o freeze da UI relatado no
            # comentário antigo.
            n = len(score_dataset.items)
            output_items: List = [None] * n
            CHUNK = 50_000
            log_lrs_view = log_lrs  # ndarray indexável
            for chunk_start in range(0, n, CHUNK):
                chunk_end = min(chunk_start + CHUNK, n)
                for j in range(chunk_start, chunk_end):
                    src = score_dataset.items[j]
                    new_sample = copy.copy(src)
                    old_md = src.metadata if isinstance(src.metadata, dict) else {}
                    new_sample.metadata = {
                        **old_md,
                        "log_lr": float(log_lrs_view[j]),
                    }
                    output_items[j] = new_sample
                # Micro-pausa: cede GIL à main thread para drenar eventos Qt.
                time.sleep(0)

            output_name = (
                instruction.output_calibrated_scores_name
                or f"{score_dataset.name}_calibrated"
            )
            # Propaga telemetria de Cllr CV / in-sample (Cllr CV diagnóstico
            # do calibrador no domínio de treino) pra `STRATEGY_EVALUATION`
            # poder anexar tudo em ValidationMetrics.
            calibrator_md = getattr(calibrator, "metadata", None) or {}
            ds_metadata = {
                "source_dataset_name": score_dataset.name,
                "calibrator_type": type(calibrator).__name__,
                # Propaga marcador de casework para downstream — preserva
                # a semântica "sem ground truth" através da calibração.
                "is_casework": bool(score_dataset.metadata.get("is_casework", False)),
            }
            for key in (
                "calibrator_cllr_train",
                "calibrator_cllr_train_gen",
                "calibrator_cllr_train_imp",
                "calibrator_cllr_cv",
                "calibrator_cllr_cv_gen",
                "calibrator_cllr_cv_imp",
                "calibrator_cv_gap",
                "cv_folds_effective",
                "cv_split_mode",
                "cv_hyperparameter_mode",
            ):
                if key in calibrator_md and calibrator_md[key] is not None:
                    ds_metadata[key] = calibrator_md[key]

            # Propaga marcador de vetorização + matriz (N, D) + labels dos
            # sistemas-fonte. Sem isso, STRATEGY_EVALUATION downstream perde
            # a info de que o calibrador é multi-D e infere D=1, colapsando
            # o heatmap log-LR num plot raw×log-LR enganoso (raw_score é só
            # a 1ª componente do vetor).
            if score_dataset.is_score_vector:
                ds_metadata["is_score_vector"] = True
                sv_matrix = score_dataset.metadata.get(
                    SimilarityScoreDataset.SCORE_VECTOR_MATRIX_KEY
                )
                if sv_matrix is not None:
                    ds_metadata[SimilarityScoreDataset.SCORE_VECTOR_MATRIX_KEY] = sv_matrix
                sources = score_dataset.metadata.get("score_vector_source_resources")
                if sources is not None:
                    ds_metadata["score_vector_source_resources"] = sources

            # Propaga coeficientes do calibrador (LogReg apenas) pra
            # STRATEGY_EVALUATION downstream conseguir:
            # (a) calcular a superfície log-LR ANALITICAMENTE no plano 2D
            #     em vez de interpolar dos observados (caso degenerado:
            #     fronteira retorna reta perfeita pra LogReg);
            # (b) exibir a equação β₀ + β₁·x₀ + β₂·x₁ no rodapé do plot;
            # (c) mostrar coefs padronizados (resposta direta a "qual
            #     feature domina?").
            # Pra calibradores não-LogReg (KDE/BiGauss) os campos ficam
            # ausentes; math service cai em fallback de interpolação.
            sk_logreg = getattr(calibrator, "logreg", None)
            sk_coef_attr = getattr(sk_logreg, "coef_", None) if sk_logreg is not None else None
            sk_intercept_attr = getattr(sk_logreg, "intercept_", None) if sk_logreg is not None else None
            piecewise_active = bool(calibrator_md.get("piecewise_enabled"))
            if piecewise_active:
                # Calibrador com expansão piecewise-linear interna: a
                # standardização das features expandidas é feita por uma
                # StandardScaler DENTRO da Pipeline, não pelos atributos
                # `_feature_mean`/`_feature_std` (que ficam None nesse modo).
                # Os coefs em escala bruta já foram pré-calculados pelo
                # `_validate_monotonicity_and_persist_diagnostics` e
                # persistidos no metadata; usamos esses valores diretamente.
                coef_raw_list = calibrator_md.get("calibrator_coef_raw")
                intercept_raw_val = calibrator_md.get("calibrator_intercept_raw")
                if coef_raw_list is not None and intercept_raw_val is not None:
                    ds_metadata["calibrator_coef_raw"] = list(coef_raw_list)
                    ds_metadata["calibrator_intercept_raw"] = float(
                        intercept_raw_val
                    )
                # Propaga estado piecewise pra STRATEGY_EVALUATION
                # downstream conseguir desenhar a curva analiticamente
                # (sem precisar interpolar pontos observados, que colapsa
                # múltiplas dobras numa só visualmente).
                for key in (
                    "piecewise_enabled",
                    "hinge_anchor_s0",
                    "piecewise_knot_offsets",
                    "piecewise_knot_offsets_user",
                    "piecewise_knot_offsets_raw",
                    "piecewise_knot_scale",
                    "piecewise_knot_placement",
                    "piecewise_auto_log_lr_hp_p50",
                    "piecewise_auto_log_lr_hd_p50",
                    "piecewise_knots_absolute",
                    "segment_slopes_raw",
                ):
                    if key in calibrator_md and calibrator_md[key] is not None:
                        ds_metadata[key] = calibrator_md[key]
            elif sk_coef_attr is not None and sk_intercept_attr is not None:
                try:
                    import numpy as _np
                    sk_coef = _np.asarray(sk_coef_attr, dtype=_np.float64).ravel()
                    sk_intercept = float(_np.asarray(sk_intercept_attr, dtype=_np.float64).ravel()[0])
                    fm = getattr(calibrator, "_feature_mean", None)
                    fs = getattr(calibrator, "_feature_std", None)
                    if fm is not None and fs is not None:
                        # Standardização interna ativa: sklearn coef está
                        # na escala Z. Converte pra escala bruta do input.
                        fm_arr = _np.asarray(fm, dtype=_np.float64).ravel()
                        fs_arr = _np.asarray(fs, dtype=_np.float64).ravel()
                        coef_raw = (sk_coef / fs_arr)
                        intercept_raw = float(sk_intercept - _np.sum(sk_coef * fm_arr / fs_arr))
                    else:
                        # Sem standardização: sklearn coef JÁ está em escala bruta.
                        coef_raw = sk_coef
                        intercept_raw = sk_intercept
                    ds_metadata["calibrator_coef_raw"] = coef_raw.tolist()
                    ds_metadata["calibrator_intercept_raw"] = intercept_raw
                except Exception as exc:
                    self._log(
                        context,
                        f"[{instruction.name}] ⚠️ Falha ao propagar coefs do "
                        f"calibrador ({type(exc).__name__}: {exc}); plot 2D vai "
                        f"cair em interpolação."
                    )

            calibrated_dataset = SimilarityScoreDataset(
                name=output_name,
                description=(
                    f"Scores calibrados via {type(calibrator).__name__} "
                    f"de {score_dataset.name}"
                ),
                items=output_items,
                metadata=ds_metadata,
            )

            output_resource_id = instruction.output_calibrated_scores_resource_id
            calibrated_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.CALIBRATED_SCORES,
                payload=calibrated_dataset,
                artifact_ref=None,
            )
            context.set_resource(calibrated_resource)

            # Estatísticas para metadata da execução — diagnóstico rápido.
            labels_arr = score_dataset.labels
            mean_log_lr_genuines = (
                float(np.mean(log_lrs[labels_arr == 1]))
                if np.any(labels_arr == 1) else None
            )
            mean_log_lr_impostors = (
                float(np.mean(log_lrs[labels_arr == 0]))
                if np.any(labels_arr == 0) else None
            )

            self._log(
                context,
                f"[{instruction.name}] ✅ Calibração concluída. "
                f"log-LR médio: gen={mean_log_lr_genuines}, imp={mean_log_lr_impostors}."
            )

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata={
                    "instruction_id": instruction.instruction_id,
                    "instruction_name": instruction.name,
                    "input_similarity_scores_resource_id":
                        instruction.input_similarity_scores_resource_id,
                    "trained_calibration_model_resource_id":
                        instruction.trained_calibration_model_resource_id,
                    "output_calibrated_scores_resource_id": output_resource_id,
                    "calibrator_type": type(calibrator).__name__,
                    "n_samples": int(n_samples),
                    "n_genuine": int(score_dataset.total_genuines),
                    "n_impostor": int(score_dataset.total_impostors),
                    "mean_log_lr_genuines": mean_log_lr_genuines,
                    "mean_log_lr_impostors": mean_log_lr_impostors,
                },
            )

        except Exception as exc:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de calibração de scores "
                f"{instruction.instruction_id}: {exc}"
            )
            return self._failure_result(str(exc))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, ScoreCalibrationInstruction):
            result.add_error(
                "A instrução informada não é do tipo ScoreCalibrationInstruction."
            )
            return result

        # ─── Calibrator ────────────────────────────────────────────────────
        calib_id = instruction.trained_calibration_model_resource_id
        if not calib_id:
            result.add_error(
                "A instrução deve definir 'trained_calibration_model_resource_id'."
            )
        elif not context.has_resource(calib_id):
            result.add_error(
                f"O recurso de calibrador não foi encontrado no contexto: '{calib_id}'."
            )
        else:
            calib_resource = context.get_resource(calib_id)
            if calib_resource is None:
                result.add_error(
                    f"O recurso '{calib_id}' está registrado mas é nulo no contexto."
                )
            elif calib_resource.payload is None:
                ref_path = (
                    calib_resource.artifact_ref.path
                    if calib_resource.artifact_ref is not None else None
                )
                if ref_path:
                    result.add_error(
                        f"O recurso '{calib_id}' tem artifact_ref ('{ref_path}') "
                        "mas o payload está vazio — falha ao desserializar o "
                        "calibrador do disco. Re-execute a instrução "
                        "CALIBRATION_MODEL_TRAINING que produz este modelo."
                    )
                else:
                    result.add_error(
                        f"O recurso '{calib_id}' ainda não foi produzido. "
                        "Execute CALIBRATION_MODEL_TRAINING antes."
                    )
            elif not isinstance(calib_resource.payload, CalibrationModelProtocol):
                result.add_error(
                    f"O recurso '{calib_id}' contém um payload do tipo "
                    f"{type(calib_resource.payload).__name__}, que não satisfaz "
                    "CalibrationModelProtocol."
                )

        # ─── Scores de entrada ─────────────────────────────────────────────
        scores_id = instruction.input_similarity_scores_resource_id
        if not scores_id:
            result.add_error(
                "A instrução deve definir 'input_similarity_scores_resource_id'."
            )
        elif not context.has_resource(scores_id):
            result.add_error(
                f"O recurso de scores de entrada não foi encontrado no contexto: "
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
                        "mas o payload está vazio — falha ao carregar os scores "
                        "do disco. Re-execute a instrução SIMILARITY_SCORING "
                        "que produz estes scores."
                    )
                else:
                    result.add_error(
                        f"O recurso '{scores_id}' ainda não foi produzido. "
                        "Execute SIMILARITY_SCORING antes."
                    )
            elif not isinstance(scores_resource.payload, SimilarityScoreDataset):
                result.add_error(
                    f"O recurso '{scores_id}' contém um payload do tipo "
                    f"{type(scores_resource.payload).__name__}, esperado "
                    "SimilarityScoreDataset."
                )

        # ─── Output ────────────────────────────────────────────────────────
        if not instruction.output_calibrated_scores_resource_id:
            result.add_error(
                "A instrução deve definir 'output_calibrated_scores_resource_id'."
            )

        if (
            instruction.output_calibrated_scores_resource_id
            and instruction.output_calibrated_scores_resource_id
            == instruction.input_similarity_scores_resource_id
        ):
            result.add_warning(
                "output_calibrated_scores_resource_id coincide com o recurso "
                "de scores de entrada. O contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
