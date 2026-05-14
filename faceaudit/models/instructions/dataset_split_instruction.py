# faceaudit/models/instructions/dataset_split_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.dataset_split_mode import DatasetSplitMode
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


T = TypeVar("T", bound="DatasetSplitInstruction")


@dataclass
class DatasetSplitInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para particionamento de datasets.

    Esta instrução descreve:
    - qual dataset de entrada será particionado
    - qual modo de split será utilizado
    - quais parâmetros controlam o particionamento
    - quais recursos de saída serão produzidos

    Modos atualmente suportados:
    - BY_IDENTITY_RATIO         — split por identidade na proporção informada (sem clustering)
    - BY_IDENTITY_LIST          — split por lista explícita de identidades
    - BY_IDENTITY_ABSOLUTE      — split por pisos rígidos (min_calibration_pairs/ids)
    - BY_INDEX_RATIO            — split por amostras na proporção informada
    - BY_INDEX_LIST             — split por lista explícita de índices
    - BY_SUPERVISED_ADAPTATION  — split SDA (source ∪ target, prioriza interseção)

    Modos por identidade (RATIO, ABSOLUTE, SUPERVISED_ADAPTATION) podem ser
    combinados com `grouping_strategy=PRESERVE_LOOKALIKES` para preservar
    pacotes de sósias no mesmo subset.
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset que será particionado (source em SDA)."}
    )

    target_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset target — usado APENAS em BY_SUPERVISED_ADAPTATION."}
    )

    split_mode: DatasetSplitMode = field(
        default=DatasetSplitMode.BY_IDENTITY_RATIO,
        metadata={"help": "Modo de split do dataset."}
    )

    split_ratio: float = field(
        default=0.8,
        metadata={"help": "Proporção do split quando o modo usar razão."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente aleatória para splits reprodutíveis."}
    )

    identity_list: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista explícita de identidades usada em modos manuais."}
    )

    index_list: List[int] = field(
        default_factory=list,
        metadata={"help": "Lista explícita de índices usada em modos manuais."}
    )

    grouping_strategy: GroupingStrategy = field(
        default=GroupingStrategy.RANDOM_IDENTITIES,
        metadata={"help": "Estratégia de agrupamento de identidades (sorteio cego ou clusters de sósias)."}
    )

    cluster_size: int = field(
        default=3,
        metadata={"help": "Tamanho do pacote de sósias quando grouping_strategy=PRESERVE_LOOKALIKES."}
    )

    min_calibration_pairs: int = field(
        default=3000,
        metadata={"help": "Meta/piso de pares genuínos para B (calibração) em modos por identidade."}
    )

    min_calibration_ids: int = field(
        default=40,
        metadata={"help": "Meta/piso de identidades em B (calibração) em modos por identidade."}
    )

    min_samples_per_id: int = field(
        default=2,
        metadata={"help": (
            "Filtro de qualidade: identidades com MENOS de N amostras são "
            "DESCARTADAS do split — não vão para A nem para B. Aplica-se aos "
            "modos por identidade (RATIO, ABSOLUTE, SDA). Padrão 2 remove "
            "identidades singleton (que não permitem formar pares genuínos)."
        )}
    )

    output_dataset_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do primeiro dataset gerado."}
    )

    output_dataset_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do segundo dataset gerado."}
    )

    output_dataset_a_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do primeiro dataset gerado."}
    )

    output_dataset_b_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do segundo dataset gerado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.DATASET_SPLIT

        if not self.name:
            self.name = "Dataset Split"

        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        specs: list[ChoiceFieldSpec] = [
            ChoiceFieldSpec(
                field_name="split_mode",
                label="Modo do Split",
                choices=(
                    "BY_IDENTITY_RATIO",
                    "BY_IDENTITY_LIST",
                    "BY_IDENTITY_ABSOLUTE",
                    "BY_INDEX_RATIO",
                    "BY_INDEX_LIST",
                    "BY_SUPERVISED_ADAPTATION",
                ),
                enum_type=DatasetSplitMode,
                descriptions={
                    "BY_IDENTITY_RATIO":
                        "Divide POR IDENTIDADES distintas, na proporção informada. "
                        "A recebe ratio×100% das identidades; B recebe o complemento. "
                        "Evita vazamento de identidade entre os subconjuntos.",
                    "BY_IDENTITY_LIST":
                        "Divide com base em uma LISTA EXPLÍCITA de identidades. "
                        "B recebe somente as identidades listadas; A recebe todas as demais. "
                        "Útil para split controlado e reproduzível.",
                    "BY_IDENTITY_ABSOLUTE":
                        "Divide por pisos rígidos: B (calibração) recebe identidades até "
                        "atingir min_calibration_pairs E min_calibration_ids; A (treino) "
                        "recebe o complemento. Não usa split_ratio.",
                    "BY_INDEX_RATIO":
                        "Divide POR AMOSTRAS, na proporção informada. "
                        "A recebe ratio×100% das amostras; B recebe o complemento. "
                        "Mais simples, mas pode espalhar a mesma identidade entre os subsets.",
                    "BY_INDEX_LIST":
                        "Divide com base em uma LISTA EXPLÍCITA de índices. "
                        "B recebe somente as amostras nos índices listados; A recebe todas as demais.",
                    "BY_SUPERVISED_ADAPTATION":
                        "Adaptação Supervisionada (SDA): recebe source + target. "
                        "A (treino PLDA) = (source ∪ target) menos calibração; "
                        "B (calibração) prioriza identidades da interseção source∩target "
                        "e completa com identidades exclusivas do target.",
                },
            ),
        ]

        # Segundo choice condicional: estratégia de agrupamento aplicável a
        # modos por identidade (RATIO/ABSOLUTE/LIST/SDA). Não aparece em modos
        # por índice.
        if self.uses_identity_grouping_capable:
            specs.append(
                ChoiceFieldSpec(
                    field_name="grouping_strategy",
                    label="Estratégia de Agrupamento",
                    choices=("RANDOM_IDENTITIES", "PRESERVE_LOOKALIKES"),
                    enum_type=GroupingStrategy,
                    descriptions={
                        "RANDOM_IDENTITIES":
                            "Sorteio cego de identidades. Mais rápido, mas pode separar "
                            "sósias entre os subsets, prejudicando hard impostor mining.",
                        "PRESERVE_LOOKALIKES":
                            "Calcula a matriz de confusão e força que cada identidade "
                            "viaje para o mesmo subset junto com seus piores sósias "
                            "(cluster_size identidades por pacote). Preserva a dificuldade "
                            "geométrica para o calibrador.",
                    },
                )
            )

        return specs

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        OUT = (StrategyResourceType.DATASET,)

        # Labels dinâmicos por modo, deixando claro qual dataset recebe o quê.
        # Convenção do executor:
        #   *_RATIO         → Dataset A recebe split_ratio×N (principal); B recebe complemento.
        #   *_LIST          → Dataset B recebe os itens listados; A recebe o complemento.
        #   ABSOLUTE / LOOKALIKES / SDA → A = treino (PLDA), B = calibração.
        treino_calib = (
            self.uses_identity_absolute
            or self.uses_supervised_adaptation
            or (self.uses_identity_ratio and self.uses_lookalikes)
        )

        if treino_calib:
            label_a = "Dataset A (treino — PLDA)"
            label_b = "Dataset B (calibração)"
        elif self.uses_identity_ratio or self.uses_index_ratio:
            label_a = "Dataset A (proporção principal)"
            label_b = "Dataset B (complementar)"
        elif self.uses_identity_list:
            label_a = "Dataset A (demais identidades)"
            label_b = "Dataset B (identidades selecionadas)"
        elif self.uses_index_list:
            label_a = "Dataset A (demais índices)"
            label_b = "Dataset B (índices selecionados)"
        else:
            label_a, label_b = "Dataset A", "Dataset B"

        # Em SDA, o dataset principal é o "source" — relabel para clareza.
        input_label = "Dataset Source" if self.uses_supervised_adaptation else "Dataset"

        specs: list[ResourceFieldSpec] = [
            ResourceFieldSpec("dataset_resource_id", input_label, DS),
        ]

        if self.uses_supervised_adaptation:
            specs.append(
                ResourceFieldSpec("target_dataset_resource_id", "Dataset Target", DS)
            )

        specs.extend([
            ResourceFieldSpec("output_dataset_a_resource_id", label_a, OUT, is_output=True),
            ResourceFieldSpec("output_dataset_b_resource_id", label_b, OUT, is_output=True),
        ])

        return specs

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        """
        Retorna campos primitivos editáveis no painel, condicionados ao
        split_mode e à grouping_strategy.

        - Modos *_RATIO e SDA: split_ratio (float, 0-1)
        - BY_IDENTITY_LIST: identity_list (list_str)
        - BY_INDEX_LIST: index_list (list_int)
        - Modos por identidade (RATIO/ABSOLUTE/SDA): min_calibration_pairs,
          min_calibration_ids, min_samples_per_id (int)
        - PRESERVE_LOOKALIKES: cluster_size (int)
        - Sempre: random_seed (int)

        Os campos `output_dataset_a_name`/`output_dataset_b_name` permanecem
        no DTO mas não são expostos na UI — o nome do recurso de saída já
        é configurado na seção Saídas. Quando não preenchidos, o executor
        usa fallback automático ({dataset.name}_A / _B).
        """
        specs: list[ValueFieldSpec] = []

        # split_ratio: usado em RATIO (identity/index) e em SDA.
        # ABSOLUTE não usa — pisos rígidos governam o split.
        if (
            self.uses_identity_ratio
            or self.uses_index_ratio
            or self.uses_supervised_adaptation
        ):
            specs.append(ValueFieldSpec(
                field_name="split_ratio",
                label="Proporção do Split (treino)",
                value_type="float",
                min_value=0.0,
                max_value=1.0,
                decimals=3,
            ))

        if self.uses_identity_list:
            specs.append(ValueFieldSpec(
                field_name="identity_list",
                label="Lista de Identidades",
                value_type="list_str",
                placeholder="ID1, ID2, ID3, …",
            ))

        if self.uses_index_list:
            specs.append(ValueFieldSpec(
                field_name="index_list",
                label="Lista de Índices",
                value_type="list_int",
                placeholder="0, 1, 5, 12, …",
            ))

        # cluster_size aparece quando o agrupamento é por sósias.
        if self.uses_lookalikes and self.uses_identity_grouping_capable:
            specs.append(ValueFieldSpec(
                field_name="cluster_size",
                label="Tamanho do Cluster (sósias)",
                value_type="int",
                min_value=2,
                max_value=20,
            ))

        # Pisos/metas e filtro de qualidade — apenas em modos por identidade
        # com semântica de treino+calibração (RATIO, ABSOLUTE, SDA).
        if (
            self.uses_identity_ratio
            or self.uses_identity_absolute
            or self.uses_supervised_adaptation
        ):
            specs.append(ValueFieldSpec(
                field_name="min_calibration_pairs",
                label="Mín. Pares Genuínos (B)",
                value_type="int",
                min_value=0,
                max_value=10**9,
            ))
            specs.append(ValueFieldSpec(
                field_name="min_calibration_ids",
                label="Mín. Identidades (B)",
                value_type="int",
                min_value=0,
                max_value=10**6,
            ))
            specs.append(ValueFieldSpec(
                field_name="min_samples_per_id",
                label="Mín. Amostras por Identidade",
                value_type="int",
                min_value=1,
                max_value=10**4,
            ))

        specs.append(ValueFieldSpec(
            field_name="random_seed",
            label="Semente Aleatória",
            value_type="int",
            min_value=0,
            max_value=10**9,
        ))

        return specs

    def get_configuration_help(self) -> Optional[str]:
        """
        Aviso contextual exibido no topo da seção Configuração da UI.

        Lembra o usuário que o filtro `min_samples_per_id` descarta
        silenciosamente identidades com menos amostras do que o limite —
        elas não vão para A nem para B. Só aparece nos modos por identidade
        que aplicam esse filtro (RATIO, ABSOLUTE, SDA).
        """
        if (
            self.uses_identity_ratio
            or self.uses_identity_absolute
            or self.uses_supervised_adaptation
        ):
            n = self.min_samples_per_id
            return (
                f"⚠️ Identidades com menos de {n} amostra(s) serão DESCARTADAS "
                "do split (não vão para A nem para B). Ajuste 'Mín. Amostras "
                "por Identidade' se quiser preservá-las."
            )
        return None

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos
        específicos da instrução.

        Em SDA, target entra como segundo input (após source).
        """
        inputs = []
        outputs = []

        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)

        if self.uses_supervised_adaptation and self.target_dataset_resource_id:
            inputs.append(self.target_dataset_resource_id)

        if self.output_dataset_a_resource_id:
            outputs.append(self.output_dataset_a_resource_id)

        if self.output_dataset_b_resource_id:
            outputs.append(self.output_dataset_b_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # Repopula defaults após o usuário definir o dataset e/ou modo.
        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Defaults:
        - Modos BY_IDENTITY_*: MIN_IDENTITIES no source (split por identidade
          exige ≥ 2 identidades pra produzir A+B não-vazios).
        - Modos BY_INDEX_*: MIN_SAMPLES no source.
        - BY_SUPERVISED_ADAPTATION: MIN_IDENTITIES no source + target +
          DATASET_OVERLAP_REQUIRED (SDA precisa de identidades compartilhadas
          pra alinhar pares).
        """
        if self.validation_rules_user_managed:
            return
        if self.validation_rule_configs:
            return
        if not self.dataset_resource_id:
            return

        cfgs = []
        if (
            self.uses_identity_ratio
            or self.uses_identity_list
            or self.uses_identity_absolute
        ):
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_IDENTITIES,
                params={"dataset_resource_id": self.dataset_resource_id},
            ))
        elif self.uses_index_ratio or self.uses_index_list:
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_SAMPLES,
                params={"dataset_resource_id": self.dataset_resource_id},
            ))
        elif self.uses_supervised_adaptation:
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_IDENTITIES,
                params={"dataset_resource_id": self.dataset_resource_id},
            ))
            if self.target_dataset_resource_id:
                cfgs.append(ValidationRuleConfig(
                    rule_type=ValidationRuleType.MIN_IDENTITIES,
                    params={"dataset_resource_id": self.target_dataset_resource_id},
                ))
                cfgs.append(ValidationRuleConfig(
                    rule_type=ValidationRuleType.DATASET_OVERLAP_REQUIRED,
                    params={
                        "dataset_a_resource_id": self.dataset_resource_id,
                        "dataset_b_resource_id": self.target_dataset_resource_id,
                    },
                ))

        if cfgs:
            self.validation_rule_configs = cfgs

    @property
    def uses_identity_ratio(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_IDENTITY_RATIO

    @property
    def uses_identity_list(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_IDENTITY_LIST

    @property
    def uses_identity_absolute(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_IDENTITY_ABSOLUTE

    @property
    def uses_index_ratio(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_INDEX_RATIO

    @property
    def uses_index_list(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_INDEX_LIST

    @property
    def uses_supervised_adaptation(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_SUPERVISED_ADAPTATION

    @property
    def uses_identity_grouping_capable(self) -> bool:
        """
        True se o modo opera por identidade — habilita o choice
        `grouping_strategy` (RANDOM_IDENTITIES vs PRESERVE_LOOKALIKES).
        """
        return (
            self.uses_identity_ratio
            or self.uses_identity_list
            or self.uses_identity_absolute
            or self.uses_supervised_adaptation
        )

    @property
    def uses_lookalikes(self) -> bool:
        return self.grouping_strategy == GroupingStrategy.PRESERVE_LOOKALIKES

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - split_mode
        - grouping_strategy
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "split_mode" in processed_data and processed_data["split_mode"]:
            processed_data["split_mode"] = DatasetSplitMode.from_str(
                processed_data["split_mode"]
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