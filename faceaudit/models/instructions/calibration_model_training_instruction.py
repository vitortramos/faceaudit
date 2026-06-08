# faceaudit/models/instructions/calibration_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.calibration_model_type import CalibrationModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
    BiGaussianizedCalibrationModelConfig,
)
from faceaudit.models.calibration.kde_calibration_model_config import (
    KDECalibrationModelConfig,
)
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="CalibrationModelTrainingInstruction")


@dataclass
class CalibrationModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de calibração.

    A config do calibrador é embutida no DTO da instrução, em campos específicos
    para cada tipo (logreg_config / kde_config). Apenas o config do tipo
    selecionado é usado pelo executor.
    """

    calibration_model_type: CalibrationModelType = field(
        default=CalibrationModelType.LOGISTIC_REGRESSION,
        metadata={"help": "Tipo de modelo de calibração a ser usado."}
    )

    input_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada usado no treino/preparação do calibrador."}
    )

    logreg_config: Optional[LogisticRegressionCalibrationModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do calibrador por Regressão Logística (quando esse tipo é selecionado)."}
    )

    kde_config: Optional[KDECalibrationModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do calibrador KDE (quando esse tipo é selecionado)."}
    )

    bigauss_config: Optional[BiGaussianizedCalibrationModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do calibrador Bi-Gaussianized (quando esse tipo é selecionado)."}
    )

    trained_calibration_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o calibrador treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.CALIBRATION_MODEL_TRAINING

        if not self.name:
            self.name = "Calibration Model Training"

        # Auto-instancia o config relevante. O config "do outro tipo" pode
        # permanecer no DTO se já tiver sido preenchido — útil para o usuário
        # alternar entre tipos sem perder valores ajustados anteriormente.
        if self.uses_logistic_regression and self.logreg_config is None:
            self.logreg_config = LogisticRegressionCalibrationModelConfig()
        if self.uses_kde and self.kde_config is None:
            self.kde_config = KDECalibrationModelConfig()
        if self.uses_bigauss and self.bigauss_config is None:
            self.bigauss_config = BiGaussianizedCalibrationModelConfig()

        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        triples = CalibrationModelType.choices_with_description()
        return [
            ChoiceFieldSpec(
                field_name="calibration_model_type",
                label="Tipo do Modelo",
                choices=tuple(value for value, _label, _desc in triples),
                enum_type=CalibrationModelType,
                descriptions={value: desc for value, _label, desc in triples},
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (
            StrategyResourceType.SIMILARITY_SCORES,
            StrategyResourceType.ARTIFACT,
        )
        TC = (StrategyResourceType.TRAINED_CALIBRATION_MODEL,)
        return [
            ResourceFieldSpec("input_resource_id", "Scores de Entrada", IN),
            ResourceFieldSpec(
                "trained_calibration_model_resource_id",
                "Calibrador Treinado",
                TC,
                is_output=True,
            ),
        ]

    def prune_unused_fields(self) -> None:
        """
        Zera os configs aninhados dos calibradores que NÃO são o tipo
        atual. Sem isso, mudar de LOGREG pra KDE deixa `logreg_config`
        pendurado (consumindo espaço no JSON, confundindo diagnóstico
        em logs). Chamado só no Apply.
        """
        if not self.uses_logistic_regression:
            self.logreg_config = None
        if not self.uses_kde:
            self.kde_config = None
        if not self.uses_bigauss:
            self.bigauss_config = None

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.input_resource_id:
            inputs.append(self.input_resource_id)

        if self.trained_calibration_model_resource_id:
            outputs.append(self.trained_calibration_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # Repopula defaults após o usuário definir o recurso de scores.
        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Popula `validation_rule_configs` com regras padrão se a lista
        estiver vazia, houver `input_resource_id` definido E o usuário
        não tiver assumido o controle das regras.

        Defaults:
        - MIN_CALIBRATION_PAIRS — protege contra calibradores treinados
          com poucos pares genuínos ou impostores.
        - SCORE_CALIBRATION_STATE(RAW) — evita treino sobre log-LR já
          calibrado, que produziria modelo distorcido.
        - CONDITIONAL_MONOTONICITY — gate forense: ∂log_LR/∂s > 0 em
          observed_rows + central_box (V1: apenas para LogReg, que persiste
          architecture_state quando treinado via LR_ARCHITECTURE_VECTORIZATION
          em FIT_PER_FOLD).
        """
        if self.validation_rules_user_managed:
            return

        # Reset & repopulate: params apontam pro input field, então mudança
        # de recurso tem que refletir nos defaults. Quando user_managed=True
        # (acima), a lista do usuário é preservada.
        self.validation_rule_configs = []

        if not self.input_resource_id:
            return

        defaults = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_CALIBRATION_PAIRS,
                params={"scores_resource_id": self.input_resource_id},
            ),
            ValidationRuleConfig(
                rule_type=ValidationRuleType.SCORE_CALIBRATION_STATE,
                params={
                    "scores_resource_id": self.input_resource_id,
                    "expected": "RAW",
                },
            ),
        ]

        # CONDITIONAL_MONOTONICITY: auditoria embutida no executor (V1.5).
        # Não auto-populamos como regra pré-execução porque a regra precisa
        # do calibrador refitado para avaliar; o executor roda auditoria
        # programaticamente pós-refit e persiste `monotonicity_audit` no
        # metadata do calibrador. Usuários que queiram opt-in da regra como
        # validação extra ainda podem adicioná-la manualmente via UI.

        self.validation_rule_configs = defaults

    @property
    def uses_logistic_regression(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.LOGISTIC_REGRESSION

    @property
    def uses_kde(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.KDE

    @property
    def uses_bigauss(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.BI_GAUSSIANIZED

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if (
            "calibration_model_type" in processed_data
            and processed_data["calibration_model_type"]
        ):
            processed_data["calibration_model_type"] = CalibrationModelType.from_str(
                processed_data["calibration_model_type"]
            )

        if "logreg_config" in processed_data and isinstance(
            processed_data["logreg_config"], dict
        ):
            processed_data["logreg_config"] = (
                LogisticRegressionCalibrationModelConfig.from_dict(
                    processed_data["logreg_config"]
                )
            )

        if "kde_config" in processed_data and isinstance(
            processed_data["kde_config"], dict
        ):
            processed_data["kde_config"] = KDECalibrationModelConfig.from_dict(
                processed_data["kde_config"]
            )

        if "bigauss_config" in processed_data and isinstance(
            processed_data["bigauss_config"], dict
        ):
            processed_data["bigauss_config"] = (
                BiGaussianizedCalibrationModelConfig.from_dict(
                    processed_data["bigauss_config"]
                )
            )

        # Retrocompat: campo removido
        processed_data.pop("calibration_model_config_resource_id", None)

        return cls(**processed_data)
