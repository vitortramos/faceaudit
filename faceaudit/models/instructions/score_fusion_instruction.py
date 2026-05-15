# faceaudit/models/instructions/score_fusion_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="ScoreFusionInstruction")


@dataclass
class ScoreFusionInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para fusão score-level de dois SimilarityScoreDataset.

    Recebe dois datasets de scores com os MESMOS pares (probe + reference) e
    produz um novo SimilarityScoreDataset onde cada amostra carrega um vetor
    de scores em `metadata["fused_features"]`. O `raw_score` do output é o
    primeiro input (preserva utilidade em mining e casework_viz que esperam
    score escalar).

    O dataset resultante tem `metadata["is_fused"] = True` e é consumido pelo
    CalibrationModelTrainingInstruction com LogReg multivariado, que aprende
    pesos ótimos para cada sistema. Forma forensicamente defensável de
    combinar evidências (Brümmer & du Preez 2006).

    Como os dois sistemas tipicamente usam o mesmo embedding base (e portanto
    não são condicionalmente independentes), score fusion via LogReg captura
    a correlação melhor do que soma simples de log-LRs.
    """

    input_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Primeiro recurso SIMILARITY_SCORES (ex: cosseno)."}
    )

    input_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Segundo recurso SIMILARITY_SCORES (ex: PLDA)."}
    )

    output_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída SIMILARITY_SCORES fundido."}
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome do dataset fundido (opcional)."}
    )

    require_pair_match: bool = field(
        default=True,
        metadata={
            "help": (
                "Quando True, valida que os dois datasets têm os mesmos pares "
                "(probe.face_hash + reference.face_hash) na mesma ordem. "
                "Quando False, reordena B para casar com A por hash. Falha se "
                "algum par de A não estiver em B."
            )
        },
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SCORE_FUSION

        if not self.name:
            self.name = "Score Fusion"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (StrategyResourceType.SIMILARITY_SCORES,)
        OUT = (StrategyResourceType.SIMILARITY_SCORES,)
        return [
            ResourceFieldSpec("input_a_resource_id", "Sistema A", IN),
            ResourceFieldSpec("input_b_resource_id", "Sistema B", IN),
            ResourceFieldSpec(
                "output_resource_id", "Scores Fundidos", OUT, is_output=True,
            ),
        ]

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.input_a_resource_id:
            inputs.append(self.input_a_resource_id)
        if self.input_b_resource_id:
            inputs.append(self.input_b_resource_id)
        if self.output_resource_id:
            outputs.append(self.output_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

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
