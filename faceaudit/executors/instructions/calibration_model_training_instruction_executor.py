# faceaudit/executors/instructions/calibration_model_training_instruction_executor.py

from __future__ import annotations

import numpy as np

from faceaudit.enums.calibration_model_type import CalibrationModelType
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

            if instruction.uses_logistic_regression:
                # Config embutida no DTO da instrução.
                trained_model = self._train_logistic_regression_calibrator(
                    scores=scores,
                    labels=labels,
                    config_payload=instruction.logreg_config,
                    context=context,
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

    def _train_logistic_regression_calibrator(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        config_payload: LogisticRegressionCalibrationModelConfig,
        context: StrategyExecutionContextProtocol,
    ) -> LogisticRegressionCalibrationModel:
        """
        Instancia e treina um LogisticRegressionCalibrationModel, encaminhando
        os logs para o contexto de execução.
        """
        model = LogisticRegressionCalibrationModel(config=config_payload)
        model.fit(scores=scores, labels=labels, logger_callback=context.log)
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