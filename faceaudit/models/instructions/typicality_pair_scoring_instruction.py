# faceaudit/models/instructions/typicality_pair_scoring_instruction.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_combiner import TypicalityCombiner
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)


T = TypeVar("T", bound="TypicalityPairScoringInstruction")


@dataclass
class TypicalityPairScoringInstruction(StrategyInstructionDefinition):
    """
    Aplica um modelo de tipicidade treinado às faces de cada par de um
    dataset (self) ou de dois datasets (cross), produzindo um
    `SimilarityScoreDataset` onde o score de cada par é a combinação
    simétrica das tipicidades individuais de probe e reference.

    Pareado por construção com a saída de `SIMILARITY_SCORING` da
    mesma combinação de datasets — o `SCORE_VECTORIZATION` pode
    fundi-los em features multivariadas pro calibrador.

    Ordem de enumeração de pares (deve casar com SIMILARITY_SCORING):
      - Self:  for i in range(N): for j in range(i+1, N) — triangular superior.
      - Cross: for i in range(N_probe): for j in range(N_ref) — produto cartesiano.
    """

    trained_typicality_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do modelo de tipicidade treinado."},
    )

    probe_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset probe (mesmo do scoring de similaridade pareado)."},
    )

    reference_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset reference. Opcional — sem ele, faz auto-comparação."
        },
    )

    score_type: TypicalityScoreType = field(
        default=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        metadata={"help": "Qual métrica de tipicidade individual por face combinar."},
    )

    combiner: TypicalityCombiner = field(
        default=TypicalityCombiner.MEAN,
        metadata={"help": "Como combinar as tipicidades de probe e reference por par."},
    )

    output_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída — SimilarityScoreDataset com scores por par."},
    )

    output_scores_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do SimilarityScoreDataset gerado."},
    )

    is_casework: bool = field(
        default=False,
        metadata={
            "help": (
                "Se marcado, os pares são tratados como caso forense (sem "
                "ground truth) — same_person=None nos samples. Útil pra "
                "passar pelo SCORE_VECTORIZATION + ScoreCalibration em "
                "modo casework."
            )
        },
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.TYPICALITY_PAIR_SCORING

        if not self.name:
            self.name = "Typicality Pair Scoring"

        self._sync_declared_io()

    # ─────────────────────────── form specs ────────────────────────────

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TT = (StrategyResourceType.TRAINED_TYPICALITY_MODEL,)
        OUT = (StrategyResourceType.SIMILARITY_SCORES,)
        return [
            ResourceFieldSpec(
                "trained_typicality_model_resource_id",
                "Modelo de Tipicidade",
                TT,
            ),
            ResourceFieldSpec("probe_dataset_resource_id", "Dataset 1", DS),
            ResourceFieldSpec(
                "reference_dataset_resource_id", "Dataset 2 (opcional)", DS,
            ),
            ResourceFieldSpec(
                "output_scores_resource_id",
                "Scores de Saída",
                OUT,
                is_output=True,
            ),
        ]

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec(
                field_name="score_type",
                label="Métrica de Tipicidade",
                choices=tuple(value for value, _ in TypicalityScoreType.choices()),
                enum_type=TypicalityScoreType,
            ),
            ChoiceFieldSpec(
                field_name="combiner",
                label="Combinador (simétrico)",
                choices=tuple(value for value, _, _ in TypicalityCombiner.choices_with_description()),
                enum_type=TypicalityCombiner,
                descriptions={
                    value: desc
                    for value, _label, desc
                    in TypicalityCombiner.choices_with_description()
                },
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="is_casework",
                label="Modo Casework (sem ground truth)",
                value_type="bool",
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "Modo de pareamento (deve casar com SIMILARITY_SCORING da mesma combinação):\n"
            "• Apenas Dataset 1 → auto-comparação (triangular superior, i<j).\n"
            "• Dataset 1 + Dataset 2 → comparação cruzada (N₁ × N₂ pares).\n"
            "\n"
            "O score do par é combiner(score_type(probe), score_type(reference)) — "
            "simétrico por construção."
        )

    # ─────────────────────────── dynamic rules ─────────────────────────

    def build_validation_rules(self) -> list:
        """
        Regras dinâmicas (intrínsecas, não removíveis pelo usuário):
        - MODEL_FITTED no modelo de tipicidade.
        - EMBEDDING_DIM_MATCHES_MODEL nos datasets.
        """
        from faceaudit.validation_rules.model_fitted_rule import ModelFittedRule
        from faceaudit.validation_rules.embedding_dim_matches_model_rule import (
            EmbeddingDimMatchesModelRule,
        )

        rules: list = []
        if not self.trained_typicality_model_resource_id:
            return rules

        rules.append(ModelFittedRule(
            rule_id=f"{self.instruction_id}_model_fitted",
            model_resource_id=self.trained_typicality_model_resource_id,
        ))

        if self.probe_dataset_resource_id:
            rules.append(EmbeddingDimMatchesModelRule(
                rule_id=f"{self.instruction_id}_dim_compat_probe",
                dataset_resource_id=self.probe_dataset_resource_id,
                model_resource_id=self.trained_typicality_model_resource_id,
            ))
        if self.reference_dataset_resource_id:
            rules.append(EmbeddingDimMatchesModelRule(
                rule_id=f"{self.instruction_id}_dim_compat_reference",
                dataset_resource_id=self.reference_dataset_resource_id,
                model_resource_id=self.trained_typicality_model_resource_id,
            ))

        return rules

    # ─────────────────────────── helpers ───────────────────────────────

    @property
    def uses_self_comparison(self) -> bool:
        return bool(self.probe_dataset_resource_id) and not bool(
            self.reference_dataset_resource_id
        )

    @property
    def uses_cross_comparison(self) -> bool:
        return bool(self.probe_dataset_resource_id) and bool(
            self.reference_dataset_resource_id
        )

    def _sync_declared_io(self) -> None:
        inputs: list[str] = []
        outputs: list[str] = []

        if self.trained_typicality_model_resource_id:
            inputs.append(self.trained_typicality_model_resource_id)
        if self.probe_dataset_resource_id:
            inputs.append(self.probe_dataset_resource_id)
        if self.reference_dataset_resource_id:
            inputs.append(self.reference_dataset_resource_id)

        if self.output_scores_resource_id:
            outputs.append(self.output_scores_resource_id)

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

        # Backward compat: campo removido.
        processed_data.pop("embedding_column", None)

        if "score_type" in processed_data and processed_data["score_type"]:
            processed_data["score_type"] = TypicalityScoreType.from_str(
                processed_data["score_type"]
            )

        if "combiner" in processed_data and processed_data["combiner"]:
            processed_data["combiner"] = TypicalityCombiner.from_str(
                processed_data["combiner"]
            )

        return cls(**processed_data)
