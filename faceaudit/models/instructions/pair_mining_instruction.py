# faceaudit/models/instructions/pair_mining_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="PairMiningInstruction")


@dataclass
class PairMiningInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para curadoria estatística de pares de scores.

    Consome:
    - um SimilarityScoreDataset (output típico de SIMILARITY_SCORING)
    - uma ComparisonMiningConfig (recurso `MINING_CONFIG` do projeto/biblioteca)

    Produz um único SimilarityScoreDataset curado — concatenação de
    impostores e genuínos minerados, mantendo a estrutura de
    SimilarityScoreSample com raw_score, same_person, label e identidades
    preservadas.

    O output é tipicamente consumido por CALIBRATION_MODEL_TRAINING para
    treinar o calibrador em pares mais representativos da dificuldade real
    do caso forense (ex: top-k impostores difíceis em vez de pares aleatórios).
    """

    input_similarity_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada: SimilarityScoreDataset com pares pré-calculados."}
    )

    mining_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso MINING_CONFIG referenciando uma ComparisonMiningConfig."}
    )

    output_mined_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída: SimilarityScoreDataset com pares minerados (gen + imp concatenados)."}
    )

    output_mined_scores_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome amigável opcional do dataset minerado gerado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.PAIR_MINING

        if not self.name:
            self.name = "Pair Mining"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        SCORES = (StrategyResourceType.SIMILARITY_SCORES,)
        MINING = (StrategyResourceType.MINING_CONFIG,)
        return [
            ResourceFieldSpec(
                "input_similarity_scores_resource_id",
                "Scores de Similaridade",
                SCORES,
            ),
            ResourceFieldSpec(
                "mining_config_resource_id",
                "Config de Mineração",
                MINING,
            ),
            ResourceFieldSpec(
                "output_mined_scores_resource_id",
                "Pares Minerados",
                SCORES,
                is_output=True,
            ),
        ]

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.input_similarity_scores_resource_id:
            inputs.append(self.input_similarity_scores_resource_id)

        if self.mining_config_resource_id:
            inputs.append(self.mining_config_resource_id)

        if self.output_mined_scores_resource_id:
            outputs.append(self.output_mined_scores_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, hidratando instruction_type.

        Tolera JSONs antigos que carregavam:
        - `mining_config` embutido (ComparisonMiningConfig) — descartado;
          a estratégia precisa ser refeita apontando para um recurso
          MINING_CONFIG do projeto.
        - Campos da arch antiga (`comparison_dataset_resource_id`,
          `output_impostors_resource_id`, `output_genuines_resource_id`)
          — descartados silenciosamente.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        # Retrocompat: campos legados descartados.
        processed_data.pop("mining_config", None)
        processed_data.pop("comparison_dataset_resource_id", None)
        processed_data.pop("output_impostors_resource_id", None)
        processed_data.pop("output_genuines_resource_id", None)

        return cls(**processed_data)
