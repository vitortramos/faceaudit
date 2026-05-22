# faceaudit/models/instructions/knn_typicality_model_training_instruction.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="KNNTypicalityModelTrainingInstruction")


@dataclass
class KNNTypicalityModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Treina um modelo de tipicidade não-paramétrico (kNN esférico) a partir
    de um dataset de referência. O dataset DEVE chegar já estratificado
    pela subpopulação relevante ao caso forense (gênero, etnia, faixa
    etária, condição de captura) — o serviço não estratifica por dentro.

    Hiperparâmetros embutidos no DTO:
      - k: número de vizinhos. Heurística defensável: k ≈ √N.
      - ood_quantile: corte para flag is_in_distribution.
      - population_label: rótulo descritivo (livre, atravessa pro DTO de
        inferência e aparece nos relatórios).
      - embedding_column: qual coluna do dataset usar (default embedding_v2).
    """

    input_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset de referência (já estratificado) usado pra construir o modelo."}
    )

    k: int = field(
        default=50,
        metadata={"help": "Número de vizinhos. Heurística: k ≈ √N. Deve ser >= 1 e < sample_count."}
    )

    ood_quantile: float = field(
        default=0.01,
        metadata={"help": "Quantil de T_k usado como corte de fora-de-distribuição. 0 desativa."}
    )

    population_label: Optional[str] = field(
        default=None,
        metadata={"help": "Rótulo livre da subpopulação (ex.: 'homens_adultos_br_frontal')."}
    )

    embedding_column: str = field(
        default="embedding_v2",
        metadata={"help": "Coluna do dataset que contém o embedding L2-normalizado."}
    )

    trained_typicality_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o modelo de tipicidade treinado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.KNN_TYPICALITY_MODEL_TRAINING

        if not self.name:
            self.name = "kNN Typicality Model Training"

        self._sync_declared_io()

    # ─────────────────────────── form specs ────────────────────────────

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TT = (StrategyResourceType.TRAINED_TYPICALITY_MODEL,)
        return [
            ResourceFieldSpec(
                "input_dataset_resource_id", "Dataset de Referência", DS,
            ),
            ResourceFieldSpec(
                "trained_typicality_model_resource_id",
                "Modelo de Tipicidade Treinado",
                TT,
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="k",
                label="k (vizinhos)",
                value_type="int",
                min_value=1,
                max_value=10000,
                placeholder="50",
            ),
            ValueFieldSpec(
                field_name="ood_quantile",
                label="Quantil OOD",
                value_type="float",
                min_value=0.0,
                max_value=0.49,
                decimals=4,
                placeholder="0.01",
            ),
            ValueFieldSpec(
                field_name="population_label",
                label="Rótulo da Subpopulação",
                value_type="str",
                placeholder="ex.: homens_adultos_br_frontal",
            ),
            ValueFieldSpec(
                field_name="embedding_column",
                label="Coluna de Embedding",
                value_type="str",
                placeholder="embedding_v2",
            ),
        ]

    # ─────────────────────────── IO sync + defaults ────────────────────

    def _sync_declared_io(self) -> None:
        inputs: list[str] = []
        outputs: list[str] = []

        if self.input_dataset_resource_id:
            inputs.append(self.input_dataset_resource_id)
        if self.trained_typicality_model_resource_id:
            outputs.append(self.trained_typicality_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Defaults persistentes (configuráveis pelo usuário):
          - MIN_SAMPLES: pelo menos `k + 1` amostras (requisito do fit LOO);
            warning abaixo de 1000 (referência muito pequena perde poder
            estatístico).
          - MAX_SAMPLES_WARNING: avisa se referência > 50k (custo O(N²) no fit).
        """
        if self.validation_rules_user_managed:
            return

        self.validation_rule_configs = []

        if not self.input_dataset_resource_id:
            return

        self.validation_rule_configs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_SAMPLES,
                params={
                    "dataset_resource_id": self.input_dataset_resource_id,
                    "min_required": max(self.k + 1, 2),
                    "safety_threshold": 1000,
                },
            ),
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MAX_SAMPLES_WARNING,
                params={
                    "dataset_resource_id": self.input_dataset_resource_id,
                    "max_samples": 50_000,
                },
            ),
        ]

    # ─────────────────────────── deserialização ────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        return cls(**processed_data)
