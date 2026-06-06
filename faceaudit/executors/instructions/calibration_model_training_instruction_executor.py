# faceaudit/executors/instructions/calibration_model_training_instruction_executor.py

from __future__ import annotations

from typing import Optional

import numpy as np

from faceaudit.enums.calibration_model_type import CalibrationModelType
from faceaudit.enums.calibrator_cv_split_mode import CalibratorCVSplitMode
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.executors.instructions.strategy_instruction_executor_base import (
    StrategyInstructionExecutorBase,
)
from faceaudit.models.calibration.bi_gaussianized_calibration_model import (
    BiGaussianizedCalibrationModel,
)
from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
    BiGaussianizedCalibrationModelConfig,
)
from faceaudit.models.calibration.kde_calibration_model import KDECalibrationModel
from faceaudit.models.calibration.kde_calibration_model_config import (
    KDECalibrationModelConfig,
)
from faceaudit.models.calibration.logistic_regression_calibration_model import (
    LogisticRegressionCalibrationModel,
)
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.models.instructions.calibration_model_training_instruction import (
    CalibrationModelTrainingInstruction,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
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
from faceaudit.services.calibrator_cross_validation_service import (
    compute_calibrator_cllr_cv,
)
from faceaudit.services.cllr_math import compute_cllr


class CalibrationModelTrainingInstructionExecutor(StrategyInstructionExecutorBase):
    """
    Executor concreto da instrução de treinamento/preparação de modelo de calibração.

    Famílias atualmente suportadas:
    - LOGISTIC_REGRESSION
    - KDE
    - BI_GAUSSIANIZED

    Nesta versão, o recurso de entrada pode conter:

    1. Um SimilarityScoreDataset
    2. Um dict no formato:
        {
            "scores": np.ndarray | list,
            "labels": np.ndarray | list
        }

    Convenção de labels:
    - 1 = genuíno / Hp
    - 0 = impostor / Hd
    """

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return self._failure_result(*validation.errors)

        if not isinstance(instruction, CalibrationModelTrainingInstruction):
            return self._failure_result(
                "O executor CalibrationModelTrainingInstructionExecutor recebeu uma instrução incompatível."
            )

        self._log(
            context,
            f"▶ Iniciando execução da instrução de treino de modelo de calibração: "
            f"{instruction.name} ({instruction.instruction_id})"
        )

        try:
            input_resource = context.get_resource(instruction.input_resource_id)
            input_payload = input_resource.payload

            scores, labels = self._extract_scores_and_labels(input_payload)

            # Extrai identidades por par (probe/reference) quando o input
            # é SimilarityScoreDataset — usado por calibradores piecewise
            # para reportar segment_unique_identities_hp/hd no metadata.
            train_probe_identities, train_reference_identities = (
                self._extract_identities(input_payload)
            )

            # PR 3: detecta input vetorizado com sinalização "defer_to_
            # calibrator_per_fold" do LR_ARCHITECTURE_VECTORIZATION em modo
            # FIT_PER_FOLD. Nesse caso, ignora a matriz pré-vetorizada e
            # treina via Pipeline+GridSearchCV com transformer interno —
            # s₀/r̄ aprendidos por fold, eliminando leakage residual.
            input_meta = getattr(input_payload, "metadata", {}) or {}
            per_fold_active = (
                instruction.uses_logistic_regression
                and bool(input_meta.get("defer_to_calibrator_per_fold"))
                and input_meta.get("raw_input_resource_id")
            )

            if per_fold_active:
                trained_model = self._train_logreg_per_fold_with_transformer(
                    instruction=instruction,
                    input_payload=input_payload,
                    context=context,
                )

            elif instruction.uses_logistic_regression:
                # Config embutida no DTO da instrução.
                trained_model = self._train_logistic_regression_calibrator(
                    scores=scores,
                    labels=labels,
                    config_payload=instruction.logreg_config,
                    context=context,
                    train_probe_identities=train_probe_identities,
                    train_reference_identities=train_reference_identities,
                )

            elif instruction.calibration_model_type == CalibrationModelType.KDE:
                # Config embutida no DTO da instrução.
                trained_model = self._train_kde_calibrator(
                    scores=scores,
                    labels=labels,
                    config_payload=instruction.kde_config,
                    context=context,
                )

            elif instruction.uses_bigauss:
                # Config embutida no DTO da instrução.
                trained_model = self._train_bigauss_calibrator(
                    scores=scores,
                    labels=labels,
                    config_payload=instruction.bigauss_config,
                    context=context,
                )

            else:
                return self._failure_result(
                    f"Tipo de modelo de calibração não suportado: "
                    f"{instruction.calibration_model_type}"
                )

            # ── Telemetria pós-fit: Cllr in-sample + CV diagnóstico ──
            # In-sample sempre é computado. CV (out-of-fold) só roda quando
            # config.k_folds > 0 — anexa no metadata pra propagar via
            # SCORE_CALIBRATION até a tabela de métricas do STRATEGY_EVALUATION.
            try:
                self._attach_calibrator_telemetry(
                    trained_model=trained_model,
                    input_payload=input_payload,
                    scores=scores,
                    labels=labels,
                    instruction=instruction,
                    context=context,
                )
            except Exception as e:
                # Telemetria nunca deve quebrar o treino do calibrador.
                # Loga warning e segue. O modelo é persistido sem CV.
                self._log(
                    context,
                    f"⚠ Falha ao computar telemetria do calibrador "
                    f"({type(e).__name__}: {e}). Treino segue sem Cllr CV."
                )

            # Quando o treino foi sobre um score vetorizado, propaga a
            # lista ordenada de modelos de similaridade que produziram
            # cada coluna pro modelo treinado. A regra
            # SCORE_VECTOR_ORDER_CONSISTENCY usa isso em `SCORE_CALIBRATION`
            # pra alertar quando a ordem dos sistemas no dataset de
            # validação diverge da ordem usada no treino do calibrador.
            if (
                isinstance(input_payload, SimilarityScoreDataset)
                and input_payload.is_score_vector
            ):
                model_ids = (input_payload.metadata or {}).get(
                    "score_vector_model_ids"
                )
                if model_ids is not None:
                    setattr(trained_model, "score_vector_model_ids", list(model_ids))

            # PR 2: gate de capacidade multivariada do calibrador
            input_dim = self._infer_input_dim(input_payload)
            self._gate_calibrator_capacity(trained_model, input_dim)

            # PR 2: binding do contract fingerprint para vincular
            # dataset vetorizado ↔ calibrador (CalibrationModelMetadataService).
            try:
                from faceaudit.services.calibration_model_metadata_service import (
                    CalibrationModelMetadataService,
                )
                if hasattr(trained_model, "metadata") and trained_model.metadata is not None:
                    CalibrationModelMetadataService.bind_architecture_state(
                        trained_model.metadata,
                        getattr(input_payload, "metadata", {}) or {},
                    )
            except Exception as exc:  # bind é best-effort — não bloqueia treinar
                self._log(context, f"  ⚠ binding fingerprint falhou: {exc}")

            output_resource_id = instruction.trained_calibration_model_resource_id
            trained_resource = StrategyResource(
                resource_id=output_resource_id,
                name=output_resource_id,
                resource_type=StrategyResourceType.TRAINED_CALIBRATION_MODEL,
                payload=trained_model,
                artifact_ref=None,
            )
            context.set_resource(trained_resource)

            self._log(
                context,
                f"✅ Modelo de calibração registrado como recurso: "
                f"'{output_resource_id}'."
            )

            metadata = {
                "instruction_id": instruction.instruction_id,
                "instruction_name": instruction.name,
                "calibration_model_type": instruction.calibration_model_type.to_str(),
                "input_resource_id": instruction.input_resource_id,
                "input_payload_type": type(input_payload).__name__,
                "trained_calibration_model_resource_id": output_resource_id,
                "trained_model_type": type(trained_model).__name__,
                "n_samples": int(len(scores)),
                "n_genuine": int(np.sum(labels == 1)),
                "n_impostor": int(np.sum(labels == 0)),
            }
            best_c = getattr(trained_model, "best_C_", None)
            if best_c is not None:
                metadata["best_C"] = float(best_c)
                if instruction.uses_logistic_regression and instruction.logreg_config is not None:
                    metadata["used_gridsearch"] = bool(
                        instruction.logreg_config.use_gridsearch
                    )

            sigma_target = getattr(trained_model, "sigma_target", None)
            if sigma_target is not None:
                metadata["sigma_target"] = float(sigma_target)

            # Espelha campos de telemetria CV (vazio se k_folds=0).
            model_md = getattr(trained_model, "metadata", None) or {}
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
                if key in model_md and model_md[key] is not None:
                    metadata[key] = model_md[key]

            return InstructionExecutionResult.ok(
                produced_resource_ids=[output_resource_id],
                metadata=metadata,
            )

        except Exception as e:
            self._log(
                context,
                f"❌ Erro ao executar a instrução de treino de modelo de calibração "
                f"{instruction.instruction_id}: {e}"
            )
            return self._failure_result(str(e))

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not isinstance(instruction, CalibrationModelTrainingInstruction):
            result.add_error(
                "A instrução informada não é do tipo CalibrationModelTrainingInstruction."
            )
            return result

        if not instruction.input_resource_id:
            result.add_error("A instrução deve definir 'input_resource_id'.")
        elif not context.has_resource(instruction.input_resource_id):
            result.add_error(
                f"O recurso de entrada não foi encontrado no contexto: "
                f"'{instruction.input_resource_id}'."
            )
        else:
            input_resource = context.get_resource(instruction.input_resource_id)
            if input_resource is None:
                result.add_error("Falha ao recuperar o recurso de entrada.")
            else:
                try:
                    self._extract_scores_and_labels(input_resource.payload)
                except Exception as e:
                    result.add_error(
                        f"O recurso '{instruction.input_resource_id}' não contém "
                        f"um payload válido de calibração: {e}"
                    )

        # Config embutida no DTO da instrução — verificada conforme o tipo escolhido.
        if instruction.uses_logistic_regression:
            if instruction.logreg_config is None:
                result.add_error(
                    "Logistic Regression exige um 'logreg_config' embutido na instrução."
                )
            elif not isinstance(
                instruction.logreg_config, LogisticRegressionCalibrationModelConfig
            ):
                result.add_error(
                    "O 'logreg_config' embutido não é um "
                    "LogisticRegressionCalibrationModelConfig válido."
                )

        elif instruction.calibration_model_type == CalibrationModelType.KDE:
            if instruction.kde_config is None:
                result.add_error(
                    "KDE exige um 'kde_config' embutido na instrução."
                )
            elif not isinstance(instruction.kde_config, KDECalibrationModelConfig):
                result.add_error(
                    "O 'kde_config' embutido não é um KDECalibrationModelConfig válido."
                )

        elif instruction.uses_bigauss:
            if instruction.bigauss_config is None:
                result.add_error(
                    "Bi-Gaussianized exige um 'bigauss_config' embutido na instrução."
                )
            elif not isinstance(
                instruction.bigauss_config, BiGaussianizedCalibrationModelConfig
            ):
                result.add_error(
                    "O 'bigauss_config' embutido não é um "
                    "BiGaussianizedCalibrationModelConfig válido."
                )

        else:
            result.add_error(
                f"Tipo de modelo de calibração não suportado: "
                f"{instruction.calibration_model_type}"
            )

        if not instruction.trained_calibration_model_resource_id:
            result.add_error(
                "A instrução deve definir 'trained_calibration_model_resource_id'."
            )

        if (
            instruction.trained_calibration_model_resource_id
            and instruction.trained_calibration_model_resource_id
            == instruction.input_resource_id
        ):
            result.add_warning(
                "trained_calibration_model_resource_id coincide com o recurso "
                "de entrada. O contexto irá sobrescrever esse recurso."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _extract_scores_and_labels(self, payload) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrai scores e labels do recurso de entrada.

        Formatos aceitos:
        1. SimilarityScoreDataset (com flag `is_score_vector` no metadata
           para datasets produzidos por SCORE_VECTORIZATION — neste caso
           scores tem shape (N, D) com D features por amostra em vez de
           scores 1D).
        2. dict:
            {
                "scores": [...],   # 1D ou 2D (N, D)
                "labels": [...]
            }
        """
        if isinstance(payload, SimilarityScoreDataset):
            scores = (
                payload.extract_score_vector()
                if payload.is_score_vector
                else payload.scores
            )
            labels = payload.labels
            return self._validate_extracted_arrays(scores, labels)

        if isinstance(payload, dict):
            if "scores" not in payload:
                raise ValueError("payload de calibração não contém chave 'scores'.")

            if "labels" not in payload:
                raise ValueError("payload de calibração não contém chave 'labels'.")

            scores = np.asarray(payload["scores"], dtype=np.float64)
            labels = np.asarray(payload["labels"], dtype=np.int32)
            return self._validate_extracted_arrays(scores, labels)

        raise TypeError(
            "payload de calibração deve ser SimilarityScoreDataset ou dict com 'scores' e 'labels'."
        )

    def _validate_extracted_arrays(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if scores.ndim not in (1, 2):
            raise ValueError("scores deve ser vetor 1D ou matriz (N, D).")

        if labels.ndim != 1:
            raise ValueError("labels deve ser vetor 1D.")

        n_scores = scores.shape[0]
        n_labels = labels.shape[0]

        if n_scores != n_labels:
            raise ValueError("scores e labels devem ter o mesmo número de amostras.")

        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError("labels deve conter apenas 0 e 1.")

        return scores, labels

    def _extract_identities(
        self, payload,
    ) -> tuple[Optional[list], Optional[list]]:
        """Extrai listas de probe_identity e reference_identity por par.

        Retorna `(None, None)` quando o payload não é `SimilarityScoreDataset`
        ou não contém identidades (e.g. dict com só scores + labels).
        Usado pelos calibradores piecewise para reportar
        `segment_unique_identities_hp/hd` no metadata.
        """
        if not isinstance(payload, SimilarityScoreDataset):
            return None, None
        try:
            probe_ids = [s.probe.identity for s in payload.items]
            ref_ids = [s.reference.identity for s in payload.items]
            return probe_ids, ref_ids
        except Exception:
            return None, None

    def _train_logistic_regression_calibrator(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        config_payload: LogisticRegressionCalibrationModelConfig,
        context: StrategyExecutionContextProtocol,
        train_probe_identities: Optional[list] = None,
        train_reference_identities: Optional[list] = None,
    ) -> LogisticRegressionCalibrationModel:
        """
        Instancia e treina um LogisticRegressionCalibrationModel, encaminhando
        os logs para o contexto de execução.

        Quando `train_probe_identities` e `train_reference_identities` são
        fornecidos (caso típico: input é `SimilarityScoreDataset`), são
        injetados em `model.metadata` antes do `fit` para que o calibrador
        reporte `segment_unique_identities_hp/hd` no metadata persistido.
        """
        model = LogisticRegressionCalibrationModel(config=config_payload)
        if train_probe_identities is not None:
            model.metadata["train_probe_identities"] = list(train_probe_identities)
        if train_reference_identities is not None:
            model.metadata["train_reference_identities"] = list(train_reference_identities)
        model.fit(scores=scores, labels=labels, logger_callback=context.log)
        # Após fit, removemos as listas brutas — só os campos derivados
        # (segment_unique_identities_*) permanecem persistidos. Evita
        # inflar o pickle do calibrador com cópia integral das identidades.
        model.metadata.pop("train_probe_identities", None)
        model.metadata.pop("train_reference_identities", None)
        return model

    def _train_logreg_per_fold_with_transformer(
        self,
        instruction,
        input_payload,
        context: StrategyExecutionContextProtocol,
    ) -> LogisticRegressionCalibrationModel:
        """
        PR 3 — Treina LogReg com FIT_PER_FOLD via Pipeline sklearn.

        Carrega o LR_ARCHITECTURE_RAW_INPUT referenciado pelo
        `input_payload.metadata['raw_input_resource_id']`, constrói o
        pipeline `[ArchitectureVectorizerTransformer → StandardScaler →
        LogReg]`, roda `GridSearchCV(cv=IdentityDisjointPairKFold(...))`
        com identidades disjuntas dos dois lados, e extrai o `state_`
        final do transformer pós-refit em D_cal completo.

        O state final é persistido no `model.metadata` para suportar
        INHERIT em VALIDATION/CASEWORK — o LR_ARCHITECTURE_VECTORIZATION
        de validação detecta o calibrador como fonte de architecture_state.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from faceaudit.enums.lr_calibration_architecture import (
            LRCalibrationArchitecture,
        )
        from faceaudit.enums.pair_role_orientation import PairRoleOrientation
        from faceaudit.services.architecture_vectorizer_transformer import (
            ArchitectureVectorizerTransformer,
        )
        from faceaudit.services.identity_disjoint_pair_kfold import (
            IdentityDisjointPairKFold,
        )

        meta = input_payload.metadata or {}
        raw_resource_id = meta["raw_input_resource_id"]
        context.ensure_payload(raw_resource_id)
        raw_resource = context.get_resource(raw_resource_id)
        if raw_resource is None or raw_resource.payload is None:
            raise RuntimeError(
                f"Recurso LR_ARCHITECTURE_RAW_INPUT ausente: {raw_resource_id}"
            )
        raw_payload = raw_resource.payload

        if raw_payload.same_person is None:
            raise RuntimeError(
                "LR_ARCHITECTURE_RAW_INPUT sem same_person; FIT_PER_FOLD "
                "exige rótulos para AUTO_LOGREG"
            )
        if (
            raw_payload.probe_identity_ids is None
            or raw_payload.reference_identity_ids is None
        ):
            raise RuntimeError(
                "LR_ARCHITECTURE_RAW_INPUT sem identidades por par; "
                "FIT_PER_FOLD exige PairIdentityMetadata para CV "
                "identidade-disjunta"
            )

        # Resolve arquitetura herdada do metadata vetorizado
        arch = LRCalibrationArchitecture.from_str(meta["lr_architecture"])

        # Captura proveniência de tipicidade para o state final
        state_dict = meta.get("architecture_state") or {}
        probe_role_str = (
            state_dict.get("pair_role_orientation")
            or meta.get("pair_role_orientation")
            or "PROBE_REFERENCE"
        )
        role_orientation = PairRoleOrientation.from_str(probe_role_str)

        # Constrói pipeline
        # s0_logreg_config foi persistido no state do vetorizador (LR_ARCH_VECT)
        s0_cfg = state_dict.get("s0_logreg_config") or None

        transformer = ArchitectureVectorizerTransformer(
            architecture=arch,
            schema=raw_payload.schema,
            s0_logreg_config=s0_cfg,
            base_score_kind=state_dict.get("base_score_kind") or "cosine",
            pair_role_orientation=role_orientation,
            rarity_probe_score_type=state_dict.get("rarity_probe_score_type"),
            rarity_reference_score_type=state_dict.get("rarity_reference_score_type"),
            rarity_single_score_type=state_dict.get("rarity_single_score_type"),
        )

        cfg = instruction.logreg_config
        c_grid = getattr(cfg, "c_grid", None) or [getattr(cfg, "C", 1.0)]
        cv_folds = max(2, int(getattr(cfg, "cv_folds", 5) or 5))
        solver = str(getattr(cfg, "solver", "liblinear") or "liblinear")
        class_weight = getattr(cfg, "class_weight", "balanced") or "balanced"
        max_iter = int(getattr(cfg, "max_iter", 1000) or 1000)

        steps = [
            ("arch_vec", transformer),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                solver=solver, class_weight=class_weight, max_iter=max_iter,
            )),
        ]
        pipe = Pipeline(steps)

        # CV identidade-disjunta dos dois lados
        splitter = IdentityDisjointPairKFold(
            probe_identities=list(raw_payload.probe_identity_ids),
            reference_identities=list(raw_payload.reference_identity_ids),
            n_splits=cv_folds,
            random_state=42,
        )

        X_raw = raw_payload.matrix  # (N, K) já em <f8
        y = np.asarray(raw_payload.same_person, dtype=int)

        self._log(
            context,
            f"  ↻ FIT_PER_FOLD: Pipeline[Transformer→Scaler→LogReg] em "
            f"GridSearchCV(cv={cv_folds}, splitter=IdentityDisjointPairKFold) "
            f"sobre {len(X_raw)} pares; C_grid={c_grid}",
        )

        # Mesma função objetivo do caminho legacy: NEG_CLLR_BALANCED por
        # default (forense; equivalente a −Cllr balanceado por classe).
        # Permite override via `instruction.logreg_config.scoring`. Sem
        # isso, `neg_log_loss` ficaria dominado pela classe majoritária
        # em datasets desbalanceados (Hd >> Hp) e selecionaria C subótimo.
        from faceaudit.enums.logreg_grid_scoring_metric import LogRegGridScoringMetric
        scoring_enum = getattr(cfg, "scoring", None) or LogRegGridScoringMetric.NEG_CLLR_BALANCED
        scoring_obj = LogisticRegressionCalibrationModel._resolve_scoring(scoring_enum)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid={"logreg__C": list(c_grid)},
            cv=splitter,
            scoring=scoring_obj,
            refit=True,
            n_jobs=1,
        )
        grid.fit(np.asarray(X_raw), y)

        # Após refit em todo D_cal, extrai os componentes finais
        best_pipe = grid.best_estimator_
        transformer_final: ArchitectureVectorizerTransformer = best_pipe.named_steps["arch_vec"]
        scaler_final: StandardScaler = best_pipe.named_steps["scaler"]
        logreg_final: LogisticRegression = best_pipe.named_steps["logreg"]

        state_final = transformer_final.state_

        # Empacota num LogisticRegressionCalibrationModel para preservar
        # contrato downstream (SCORE_CALIBRATION consome este wrapper).
        model = LogisticRegressionCalibrationModel(config=cfg)
        model.logreg = logreg_final
        model._is_fitted = True
        model.best_C_ = float(grid.best_params_["logreg__C"])
        model._feature_mean = np.asarray(scaler_final.mean_, dtype=np.float64)
        model._feature_std = np.asarray(scaler_final.scale_, dtype=np.float64)

        # Persiste fingerprints + state no metadata (consumidos por
        # CalibrationModelMetadataService.bind_architecture_state e por
        # LR_ARCHITECTURE_VECTORIZATION em modo INHERIT na validação)
        model.metadata.update({
            "vectorization_contract_fingerprint":
                state_final.vectorization_contract_fingerprint,
            "lr_architecture": state_final.architecture.value,
            "lr_architecture_formula_version": state_final.formula_version,
            "architecture_spec_fingerprint":
                state_final.architecture_spec_fingerprint,
            "architecture_state": state_final.to_dict(),
            # Provenance do treino per-fold (PR 3)
            "fit_per_fold": True,
            "inner_cv_splitter": "IdentityDisjointPairKFold",
            "inner_cv_fold_count": cv_folds,
            "strict_identity_disjoint": True,
            "final_refit_on_full_calibration": True,
            "selected_C": model.best_C_,
            # Coefs em escala bruta (consumidos pela auditoria de
            # monotonicidade condicional — vide §10.5 do relatório)
            "standardized_coefficients": logreg_final.coef_.ravel().tolist(),
            "scaler_mean": model._feature_mean.tolist(),
            "scaler_scale": model._feature_std.tolist(),
            "raw_scale_coefficients": (
                logreg_final.coef_.ravel() / model._feature_std
            ).tolist(),
            "raw_scale_intercept": float(
                logreg_final.intercept_.ravel()[0]
                - float(np.sum(
                    logreg_final.coef_.ravel() * model._feature_mean
                    / model._feature_std
                ))
            ),
        })

        s0_str = f"{state_final.s0:.4f}" if state_final.s0 is not None else "—"
        r0p_str = f"{state_final.r0_probe:.4f}" if state_final.r0_probe is not None else "—"
        r0r_str = f"{state_final.r0_reference:.4f}" if state_final.r0_reference is not None else "—"
        self._log(
            context,
            f"  ✅ FIT_PER_FOLD concluído: C={model.best_C_}, "
            f"s₀={s0_str}, r̄_F={r0p_str}, r̄_P={r0r_str}",
        )

        return model

    def _train_kde_calibrator(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        config_payload: KDECalibrationModelConfig,
        context: StrategyExecutionContextProtocol,
    ) -> KDECalibrationModel:
        """
        Instancia e treina um KDECalibrationModel, encaminhando
        os logs para o contexto de execução.
        """
        model = KDECalibrationModel(config=config_payload)
        model.fit(scores=scores, labels=labels, logger_callback=context.log)
        return model

    def _train_bigauss_calibrator(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        config_payload: BiGaussianizedCalibrationModelConfig,
        context: StrategyExecutionContextProtocol,
    ) -> BiGaussianizedCalibrationModel:
        """
        Instancia e treina um BiGaussianizedCalibrationModel, encaminhando
        os logs para o contexto de execução.
        """
        model = BiGaussianizedCalibrationModel(config=config_payload)
        model.fit(scores=scores, labels=labels, logger_callback=context.log)
        return model

    # =========================================================================
    # TELEMETRIA: Cllr in-sample + CV diagnóstico
    # =========================================================================

    def _attach_calibrator_telemetry(
        self,
        trained_model,
        input_payload,
        scores: np.ndarray,
        labels: np.ndarray,
        instruction: CalibrationModelTrainingInstruction,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        """Computa Cllr in-sample + (se config.k_folds>0) Cllr CV out-of-fold
        e anexa tudo a `trained_model.metadata`. Idempotente — pode ser
        chamado várias vezes sem efeito colateral."""
        # Garante bucket de metadata mesmo em modelos legados que não tinham.
        if not hasattr(trained_model, "metadata") or trained_model.metadata is None:
            trained_model.metadata = {}

        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int32)

        # In-sample (otimista — mede ajuste, não generalização).
        llrs_train = trained_model.calibrate_scores(scores_arr)
        llrs_train = np.asarray(llrs_train, dtype=np.float64)
        cllr_train, cllr_train_gen, cllr_train_imp = compute_cllr(
            llrs_train[labels_arr == 1],
            llrs_train[labels_arr == 0],
        )
        trained_model.metadata.update({
            "calibrator_cllr_train": float(cllr_train),
            "calibrator_cllr_train_gen": float(cllr_train_gen),
            "calibrator_cllr_train_imp": float(cllr_train_imp),
        })
        self._log(
            context,
            f"📐 Cllr in-sample (treino do calibrador) = {cllr_train:.6f} "
            f"(gen={cllr_train_gen:.6f}, imp={cllr_train_imp:.6f})"
        )

        # Resolve config do calibrador (dependendo da família escolhida).
        config_payload = None
        if instruction.uses_logistic_regression:
            config_payload = instruction.logreg_config
        elif instruction.calibration_model_type == CalibrationModelType.KDE:
            config_payload = instruction.kde_config
        elif instruction.uses_bigauss:
            config_payload = instruction.bigauss_config

        k_folds = int(getattr(config_payload, "k_folds", 0) or 0)
        if k_folds < 2:
            return  # CV desligado, in-sample já anexado.

        split_mode = getattr(
            config_payload, "cv_split_mode", CalibratorCVSplitMode.STRATIFIED_PAIR
        )
        if not isinstance(split_mode, CalibratorCVSplitMode):
            split_mode = CalibratorCVSplitMode.from_str(split_mode)

        groups: Optional[np.ndarray] = None
        if split_mode == CalibratorCVSplitMode.STRATIFIED_GROUP:
            groups = self._extract_groups_for_cv(input_payload)
            if groups is None:
                self._log(
                    context,
                    "⚠ STRATIFIED_GROUP requer dataset com probe_identity por amostra — "
                    "usando STRATIFIED_PAIR como fallback."
                )
                split_mode = CalibratorCVSplitMode.STRATIFIED_PAIR

        random_seed = int(getattr(config_payload, "random_seed", 42) or 42)

        factory = trained_model._make_frozen_factory()
        cv_result = compute_calibrator_cllr_cv(
            scores=scores_arr,
            labels=labels_arr,
            factory=factory,
            k_folds=k_folds,
            random_seed=random_seed,
            split_mode=split_mode,
            groups=groups,
            logger_callback=context.log,
            model_name=getattr(trained_model, "name", type(trained_model).__name__),
        )
        if cv_result is None:
            return

        cllr_cv = float(cv_result["cllr_cv"])
        trained_model.metadata.update({
            "calibrator_cllr_cv": cllr_cv,
            "calibrator_cllr_cv_gen": float(cv_result["cllr_cv_gen"]),
            "calibrator_cllr_cv_imp": float(cv_result["cllr_cv_imp"]),
            "calibrator_cv_gap": cllr_cv - float(cllr_train),
            "cv_folds_effective": int(cv_result["cv_folds_effective"]),
            "cv_split_mode": str(cv_result["cv_split_mode"]),
            # FROZEN = hyperparams (best_C_/bandwidths/sigma) congelados do
            # production model. NÃO é nested CV — seleção foi feita sobre o
            # dataset completo de calibração antes do CV. Pode ser levemente
            # otimista em relação à seleção de hiperparâmetros, mas evita
            # custo computacional de re-rodar GridSearch por fold.
            "cv_hyperparameter_mode": "FROZEN",
            "cv_hyperparameter_selection_scope": "FULL_CALIBRATION_SET",
            # Contagens pra auditoria (ajuste 7 da revisão).
            "cv_n_gen_total": cv_result.get("cv_n_gen_total"),
            "cv_n_imp_total": cv_result.get("cv_n_imp_total"),
            "cv_fold_train_gen_min": cv_result.get("cv_fold_train_gen_min"),
            "cv_fold_train_imp_min": cv_result.get("cv_fold_train_imp_min"),
            "cv_groups_total": cv_result.get("cv_groups_total"),
            "cv_groups_min_test": cv_result.get("cv_groups_min_test"),
        })

    def _extract_groups_for_cv(self, input_payload) -> Optional[np.ndarray]:
        """Extrai probe identity como array (N,) pra StratifiedGroupKFold.
        Retorna None se o payload não for SimilarityScoreDataset, se não
        tiver `items`, ou se alguma probe não tiver identidade setada.
        SimilarityScoreSample.probe é um EmbeddingInfo com .identity."""
        items = getattr(input_payload, "items", None)
        if items is None:
            return None
        try:
            groups = []
            for item in items:
                probe = getattr(item, "probe", None)
                pid = getattr(probe, "identity", None) if probe is not None else None
                if pid is None or pid == "":
                    return None
                groups.append(str(pid))
            return np.asarray(groups)
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    # PR 2: Gate de capacidade multivariada
    # ─────────────────────────────────────────────────────────────────────

    def _infer_input_dim(self, input_payload) -> int:
        """Inferência mínima da dimensão do dataset de entrada.

        Retorna 1 quando o payload é escalar (raw_score), ou D quando há
        `score_vector_matrix` no metadata. Calibradores são treinados via
        `fit(scores, labels)` com `scores` 1D em escalares ou 2D em vetor.
        """
        meta = getattr(input_payload, "metadata", {}) or {}
        sv_dim = meta.get("score_vector_dim") or meta.get("score_vector_n_features")
        if sv_dim:
            try:
                return int(sv_dim)
            except Exception:
                pass
        matrix = meta.get("score_vector_matrix")
        if matrix is not None:
            try:
                return int(matrix.shape[1])
            except Exception:
                pass
        return 1

    def _gate_calibrator_capacity(self, calibrator, input_dim: int) -> None:
        """Levanta ValueError se o calibrador não aceita a dimensão atual.

        Calibradores declaram `supports_score_vector` e `max_input_dim`
        como class attributes (PR 2). Modelos antigos sem essas
        declarações são tratados como `supports_score_vector=False,
        max_input_dim=1` (escalar apenas) — compat conservador.
        """
        supports = getattr(calibrator, "supports_score_vector", False)
        max_dim = getattr(calibrator, "max_input_dim", 1)
        if input_dim > 1 and not supports:
            raise ValueError(
                f"Calibrador {type(calibrator).__name__} não suporta "
                f"score vector (input_dim={input_dim}). Use LogReg para "
                f"multivariado."
            )
        if max_dim is not None and input_dim > max_dim:
            raise ValueError(
                f"Calibrador {type(calibrator).__name__} aceita max_input_dim="
                f"{max_dim}, mas dataset tem D={input_dim}. Use LogReg para "
                f"D > 1."
            )