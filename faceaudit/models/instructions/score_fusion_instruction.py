# faceaudit/models/instructions/score_fusion_instruction.py

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


T = TypeVar("T", bound="ScoreFusionInstruction")


@dataclass
class ScoreFusionInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para fusão score-level de **N (≥2)** SimilarityScoreDataset.

    Recebe inputs com os MESMOS pares (probe + reference) e produz um novo
    SimilarityScoreDataset com a matriz `(N_pairs, D)` em
    `metadata["fused_features_matrix"]` (D = nº de sistemas combinados).
    O `raw_score` do output é o do input A (preserva utilidade em mining
    e casework_viz que esperam score escalar).

    Os campos canônicos são `input_a_resource_id` (obrigatório) e
    `input_b_resource_id` (obrigatório, mínimo 2 sistemas). Para fundir 3+
    sistemas (ex: Cosseno + PLDA + Quality), preencha
    `extra_input_resource_ids` com os IDs adicionais.

    O dataset resultante tem `metadata["is_fused"] = True` e é consumido
    pelo CalibrationModelTrainingInstruction com LogReg multivariado, que
    aprende um peso por sistema (Brümmer & du Preez 2006). Como os
    sistemas tipicamente compartilham o mesmo embedding base (não são
    condicionalmente independentes), score fusion via LogReg captura a
    correlação melhor do que soma simples de log-LRs.
    """

    input_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Primeiro recurso SIMILARITY_SCORES (ex: cosseno)."}
    )

    input_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Segundo recurso SIMILARITY_SCORES (ex: PLDA)."}
    )

    extra_input_resource_ids: List[str] = field(
        default_factory=list,
        metadata={"help": (
            "Inputs adicionais para fundir 3+ sistemas (lista de "
            "resource_ids separados por vírgula na UI). Cada ID deve ser "
            "um recurso SIMILARITY_SCORES com os mesmos pares de A/B. "
            "Vazio = fusão simples A+B (caso comum)."
        )},
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
                "Quando True, valida que todos os inputs têm os mesmos pares "
                "(probe.face_hash + reference.face_hash) na mesma ordem. "
                "Quando False, reordena cada input B/C/... para casar com A "
                "por hash. Falha se algum par de A não estiver em algum dos "
                "outros inputs."
            )
        },
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SCORE_FUSION

        if not self.name:
            self.name = "Score Fusion"

        self._sync_declared_io()
        self._ensure_default_validation_rules()

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

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="extra_input_resource_ids",
                label="Sistemas Extras (IDs separados por vírgula)",
                value_type="list_str",
                placeholder="ex: scores_quality,scores_extra_3",
            ),
        ]

    def all_input_resource_ids(self) -> List[str]:
        """Lista ordenada de TODOS os inputs: A, B, depois extras.

        Filtra entradas vazias (`None`/`""`) e desduplica preservando
        ordem. Helper canônico — todos os call sites devem usar isso
        em vez de manipular `input_a/input_b/extra_input_resource_ids`
        manualmente.
        """
        ordered: List[str] = []
        for rid in (self.input_a_resource_id, self.input_b_resource_id, *self.extra_input_resource_ids):
            if rid and rid not in ordered:
                ordered.append(rid)
        return ordered

    def _sync_declared_io(self) -> None:
        outputs = []

        if self.output_resource_id:
            outputs.append(self.output_resource_id)

        self.input_resource_ids = self.all_input_resource_ids()
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # Repopula defaults após o usuário definir o input A.
        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """Popula `validation_rule_configs` com defaults persistentes
        quando há `input_a_resource_id` definido E o usuário ainda não
        assumiu o controle das regras.

        Default:
        - **MIN_CALIBRATION_PAIRS** no input A — protege contra fusão
          de datasets com pares insuficientes pra alimentar a calibração
          downstream (LogReg multivariado precisa de gen+imp suficientes).
          Vale só pra A porque a regra dinâmica `FUSED_PAIR_COMPATIBILITY`
          já garante que B tem o mesmo conjunto.
        """
        if self.validation_rules_user_managed:
            return

        if self.validation_rule_configs:
            return

        if not self.input_a_resource_id:
            return

        self.validation_rule_configs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_CALIBRATION_PAIRS,
                params={"scores_resource_id": self.input_a_resource_id},
            ),
        ]

    def build_validation_rules(self) -> list:
        """Regras dinâmicas (não persistidas):

        - **FUSED_PAIR_COMPATIBILITY** — todos os inputs (A, B, extras)
          compartilham exatamente o mesmo conjunto de pares
          `(probe.face_hash, reference.face_hash)`. Catch antecipado pra
          evitar ValueError tardio do executor.
        """
        from faceaudit.validation_rules.fused_pair_compatibility_rule import (
            FusedPairCompatibilityRule,
        )

        rules: list = []
        all_inputs = self.all_input_resource_ids()
        if len(all_inputs) >= 2:
            rules.append(FusedPairCompatibilityRule(
                rule_id=f"{self.instruction_id}_fused_pair_compatibility",
                input_resource_ids=all_inputs,
            ))
        return rules

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
