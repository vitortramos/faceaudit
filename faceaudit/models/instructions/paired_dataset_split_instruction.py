# faceaudit/models/instructions/paired_dataset_split_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.split_config import GroupingStrategy
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="PairedDatasetSplitInstruction")


@dataclass
class PairedDatasetSplitInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que divide DOIS datasets de forma sincronizada
    por identidade — um único sorteio de IDs é aplicado em ambos.

    Caso de uso típico (cross-pose forense):
    - primary   = BFW Frontais (transformado pelo pipeline)
    - secondary = BFW Laterais (transformado pelo pipeline)
    - Outputs A_primary e A_secondary têm AS MESMAS identidades — usados
      para treinar o PLDA num pool merged.
    - Outputs B_primary e B_secondary têm AS MESMAS identidades (complemento)
      — usados para SIMILARITY_SCORING cross-pose puro alimentando o
      treino do calibrador.

    A interseção de identidades entre primary e secondary é o pool de
    candidatos para o sorteio. Identidades exclusivas de um dos datasets
    são silenciosamente excluídas — emitir warning é responsabilidade do
    executor.
    """

    primary_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Primeiro dataset (ex: frontais transformados)."}
    )

    secondary_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Segundo dataset (ex: laterais transformados). Deve "
                          "compartilhar identidades com o primary."}
    )

    split_ratio: float = field(
        default=0.8,
        metadata={"help": "Proporção das identidades alocadas ao split A "
                          "(treino). O complemento vai para B (calibração)."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente aleatória para reprodutibilidade do sorteio."}
    )

    grouping_strategy: GroupingStrategy = field(
        default=GroupingStrategy.RANDOM_IDENTITIES,
        metadata={"help": "RANDOM_IDENTITIES (sorteio cego) ou PRESERVE_LOOKALIKES "
                          "(agrupa sósias no mesmo split)."}
    )

    cluster_size: int = field(
        default=3,
        metadata={"help": "Tamanho do pacote de sósias quando "
                          "grouping_strategy=PRESERVE_LOOKALIKES "
                          "(ignorado quando RANDOM_IDENTITIES)."}
    )

    min_calibration_pairs: int = field(
        default=3000,
        metadata={"help": "Piso mínimo de pares genuínos no split B "
                          "(somando pares dentro de primary, dentro de "
                          "secondary E pares cross-pose entre os dois)."}
    )

    min_calibration_ids: int = field(
        default=40,
        metadata={"help": "Piso mínimo de identidades no split B."}
    )

    min_samples_per_id: int = field(
        default=2,
        metadata={"help": "Filtro: identidades com menos de N amostras "
                          "(somando primary + secondary) são DESCARTADAS."}
    )

    output_primary_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Saída A do primary (treino — IDs sorteadas)."}
    )

    output_primary_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Saída B do primary (calibração — IDs complementares)."}
    )

    output_secondary_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Saída A do secondary (mesmas IDs do A do primary)."}
    )

    output_secondary_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Saída B do secondary (mesmas IDs do B do primary)."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.PAIRED_DATASET_SPLIT
        if not self.name:
            self.name = "Paired Dataset Split"
        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec(
                field_name="grouping_strategy",
                label="Estratégia de Agrupamento",
                choices=("RANDOM_IDENTITIES", "PRESERVE_LOOKALIKES"),
                enum_type=GroupingStrategy,
                descriptions={
                    "RANDOM_IDENTITIES":
                        "Sorteio cego de identidades. Mais rápido, mas pode "
                        "separar sósias entre os subsets, prejudicando hard "
                        "impostor mining.",
                    "PRESERVE_LOOKALIKES":
                        "Calcula a matriz de confusão e força que cada "
                        "identidade viaje para o mesmo subset junto com seus "
                        "piores sósias (cluster_size identidades por pacote). "
                        "Preserva a dificuldade geométrica para o calibrador.",
                },
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        OUT = (StrategyResourceType.DATASET,)
        return [
            ResourceFieldSpec("primary_dataset_resource_id",   "Primary Dataset", DS),
            ResourceFieldSpec("secondary_dataset_resource_id", "Secondary Dataset", DS),
            ResourceFieldSpec("output_primary_a_resource_id",
                              "Primary A (treino)", OUT, is_output=True),
            ResourceFieldSpec("output_primary_b_resource_id",
                              "Primary B (calibração)", OUT, is_output=True),
            ResourceFieldSpec("output_secondary_a_resource_id",
                              "Secondary A (treino)", OUT, is_output=True),
            ResourceFieldSpec("output_secondary_b_resource_id",
                              "Secondary B (calibração)", OUT, is_output=True),
        ]

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        # Todos os campos numéricos ficam sempre visíveis (consistente com
        # min_calibration_pairs/ids/samples_per_id). cluster_size só tem
        # efeito quando grouping_strategy=PRESERVE_LOOKALIKES — quando
        # RANDOM_IDENTITIES é selecionado, o valor é ignorado pelo executor.
        return [
            ValueFieldSpec(
                field_name="split_ratio",
                label="Proporção do Split (treino)",
                value_type="float",
                min_value=0.0,
                max_value=1.0,
                decimals=3,
            ),
            ValueFieldSpec(
                field_name="cluster_size",
                label="Tamanho do Cluster (sósias)",
                value_type="int",
                min_value=2,
                max_value=20,
            ),
            ValueFieldSpec(
                field_name="min_calibration_pairs",
                label="Mín. Pares Genuínos (B)",
                value_type="int",
                min_value=0,
                max_value=10**9,
            ),
            ValueFieldSpec(
                field_name="min_calibration_ids",
                label="Mín. Identidades (B)",
                value_type="int",
                min_value=0,
                max_value=10**6,
            ),
            ValueFieldSpec(
                field_name="min_samples_per_id",
                label="Mín. Amostras por Identidade",
                value_type="int",
                min_value=1,
                max_value=10**4,
            ),
            ValueFieldSpec(
                field_name="random_seed",
                label="Semente Aleatória",
                value_type="int",
                min_value=0,
                max_value=10**9,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "⚠️ Apenas a INTERSEÇÃO de identidades entre PRIMARY e SECONDARY "
            "é candidata ao sorteio (cross-pose puro). IDs exclusivas de um "
            "lado caem automaticamente no SPLIT A daquele lado. Os splits B "
            "carregam EXATAMENTE as mesmas IDs em primary e secondary — "
            "garantindo pares genuínos cross-dataset na calibração. Para um "
            "caso 100% pareado, faça os dois datasets terem o mesmo conjunto "
            "de identidades antes desta instrução."
        )

    def prune_unused_fields(self) -> None:
        """
        Zera `cluster_size` quando `grouping_strategy=RANDOM_IDENTITIES`
        (cluster_size só faz sentido pra PRESERVE_LOOKALIKES). Chamado
        só no Apply.
        """
        if not self.uses_lookalikes:
            self.cluster_size = 3  # default neutro

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []
        if self.primary_dataset_resource_id:
            inputs.append(self.primary_dataset_resource_id)
        if self.secondary_dataset_resource_id:
            inputs.append(self.secondary_dataset_resource_id)
        for oid in (
            self.output_primary_a_resource_id,
            self.output_primary_b_resource_id,
            self.output_secondary_a_resource_id,
            self.output_secondary_b_resource_id,
        ):
            if oid:
                outputs.append(oid)
        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Default: DATASET_OVERLAP_REQUIRED entre primary e secondary —
        sem overlap de identidades não há candidatos pro sorteio pareado.
        """
        if self.validation_rules_user_managed:
            return

        # Reset & repopulate: os params apontam pros input fields, então
        # quando o usuário troca o recurso a regra tem que acompanhar.
        # `user_managed=True` (acima) preserva escolha do usuário.
        self.validation_rule_configs = []

        if not (
            self.primary_dataset_resource_id
            and self.secondary_dataset_resource_id
        ):
            return

        self.validation_rule_configs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.DATASET_OVERLAP_REQUIRED,
                params={
                    "dataset_a_resource_id": self.primary_dataset_resource_id,
                    "dataset_b_resource_id": self.secondary_dataset_resource_id,
                },
            ),
        ]

    @property
    def uses_lookalikes(self) -> bool:
        return self.grouping_strategy == GroupingStrategy.PRESERVE_LOOKALIKES

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - grouping_strategy
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "grouping_strategy" in processed_data and isinstance(
            processed_data["grouping_strategy"], str
        ):
            try:
                processed_data["grouping_strategy"] = GroupingStrategy(
                    processed_data["grouping_strategy"]
                )
            except ValueError:
                processed_data["grouping_strategy"] = GroupingStrategy.RANDOM_IDENTITIES

        return cls(**processed_data)
