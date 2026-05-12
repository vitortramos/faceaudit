# faceaudit/models/instructions/hubness_analysis_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)


T = TypeVar("T", bound="HubnessAnalysisInstruction")


@dataclass
class HubnessAnalysisInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que executa o diagnóstico de hubness completo
    sobre um EmbeddingDataset, produzindo um HubnessReport compósito com
    5 estudos:

    1. analysis (compute_analysis): Nk score, similaridade impostora,
       correlações Pearson/Spearman entre Nk × norma e Nk × impostor.
    2. stratification: 5 faixas de yaw (frontal/leve/moderada/forte/extrema).
    3. paired_study: para cada identidade, frontal × melhor não-frontal.
    4. quality_diagnostics: covariável FIQA / norma × pose.
    5. optimization_study: sweet spot lab/wild/perfil.

    Custo: O(N²) pago uma vez no compute_analysis. Os 4 estudos derivados
    são triviais em cima do resultado base.
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso DATASET ou INTERMEDIATE_DATASET (EmbeddingDataset) a analisar."}
    )

    output_report_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do tipo HUBNESS_REPORT (pickle)."}
    )

    # Hyperparâmetros do KNN exaustivo (compute_analysis).
    k_neighbors: int = field(
        default=20,
        metadata={"help": "Número de vizinhos K para o KNN exaustivo (Nk score)."}
    )

    batch_size: int = field(
        default=1000,
        metadata={"help": "Tamanho do lote de queries no compute_analysis."}
    )

    chunk_size: int = field(
        default=10000,
        metadata={"help": "Tamanho do bloco de referência no compute_analysis."}
    )

    paired_study_threshold_nf: float = field(
        default=30.0,
        metadata={"help": "Threshold em graus de yaw absoluto para considerar 'não-frontal' no estudo pareado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.HUBNESS_ANALYSIS

        if not self.name:
            self.name = "Hubness Analysis"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (StrategyResourceType.DATASET,)
        OUT = (StrategyResourceType.HUBNESS_REPORT,)
        return [
            ResourceFieldSpec("dataset_resource_id", "Dataset", IN),
            ResourceFieldSpec(
                "output_report_resource_id",
                "Relatório de Hubness",
                OUT,
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="k_neighbors",
                label="Vizinhos (K)",
                value_type="int",
                min_value=1, max_value=10**4,
            ),
            ValueFieldSpec(
                field_name="batch_size",
                label="Batch size (queries)",
                value_type="int",
                min_value=1, max_value=10**6,
            ),
            ValueFieldSpec(
                field_name="chunk_size",
                label="Chunk size (refs)",
                value_type="int",
                min_value=1, max_value=10**7,
            ),
            ValueFieldSpec(
                field_name="paired_study_threshold_nf",
                label="Threshold yaw não-frontal (°)",
                value_type="float",
                min_value=0.0, max_value=180.0,
                decimals=1,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "⚠ <b>Operação cara:</b> O(N²) em similaridade. Os parâmetros "
            "<code>batch_size</code> e <code>chunk_size</code> controlam o "
            "uso de memória durante o cálculo (não afetam o resultado, "
            "só performance). Para N grandes, mantenha batch_size baixo."
        )

    def _sync_declared_io(self) -> None:
        inputs: list = []
        outputs: list = []
        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)
        if self.output_report_resource_id:
            outputs.append(self.output_report_resource_id)
        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None
        d = dict(data)
        if "instruction_type" in d and d["instruction_type"]:
            d["instruction_type"] = StrategyInstructionType.from_str(d["instruction_type"])
        return cls(**d)
