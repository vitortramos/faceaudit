# faceaudit/models/instructions/casework_visualization_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="CaseworkVisualizationInstruction")


@dataclass
class CaseworkVisualizationInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que produz uma visualização forense comparando a
    distribuição de scores de uma população de referência (gen + imp) com
    os scores do caso questionado (sem ground truth).

    Modos:
    - **CALIBRATED**: ambos dropdowns aceitam CALIBRATED_SCORES; eixo X é
      log-LR; linha de referência em log-LR=0 (limiar neutro).
    - **RAW**: ambos dropdowns aceitam SIMILARITY_SCORES; eixo X é raw_score;
      linha de referência no threshold de EER computado on-the-fly do
      dataset de referência.

    Renderização do caso:
    - n_case ≥ 30: KDE + histograma leve (azul).
    - n_case < 30: linhas verticais tracejadas com valor numérico anotado.

    O output é persistido como recurso CASEWORK_VISUALIZATION (.pkl) — pode
    ser revisitado via botão 📊 Visualizar no painel da instrução, sem
    precisar re-executar.
    """

    score_type: str = field(
        default="CALIBRATED",
        metadata={"help": "Tipo de escore: 'CALIBRATED' (log-LR) ou 'RAW' (raw_score)."}
    )

    reference_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de scores da população de referência (com gen+imp)."}
    )

    case_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de scores do caso forense (is_casework=True)."}
    )

    output_visualization_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída tipo CASEWORK_VISUALIZATION."}
    )

    output_visualization_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome amigável opcional do resultado de visualização."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.CASEWORK_VISUALIZATION

        if not self.name:
            self.name = "Casework Visualization"

        # Normaliza score_type para o conjunto suportado.
        st = (self.score_type or "CALIBRATED").upper()
        if st not in ("CALIBRATED", "RAW"):
            st = "CALIBRATED"
        self.score_type = st

        self._sync_declared_io()

    @property
    def uses_calibrated(self) -> bool:
        return self.score_type == "CALIBRATED"

    @property
    def uses_raw(self) -> bool:
        return self.score_type == "RAW"

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec(
                field_name="score_type",
                label="Tipo de Escore",
                choices=("CALIBRATED", "RAW"),
                enum_type=None,
                descriptions={
                    "CALIBRATED": (
                        "Log-LR calibrado — eixo X em escala de log-LR; "
                        "linha de referência em log-LR=0 (limiar neutro)."
                    ),
                    "RAW": (
                        "Raw score do similarity model (cosseno, PLDA, ...) — "
                        "eixo X em escala bruta; linha de referência no "
                        "threshold de EER calculado on-the-fly."
                    ),
                },
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        if self.uses_calibrated:
            INPUT_TYPES = (StrategyResourceType.CALIBRATED_SCORES,)
        else:  # RAW
            INPUT_TYPES = (StrategyResourceType.SIMILARITY_SCORES,)

        OUTPUT_TYPES = (StrategyResourceType.CASEWORK_VISUALIZATION,)

        return [
            ResourceFieldSpec(
                "reference_scores_resource_id",
                "Escores de Referência (validação)",
                INPUT_TYPES,
            ),
            ResourceFieldSpec(
                "case_scores_resource_id",
                "Escores do Caso",
                INPUT_TYPES,
            ),
            ResourceFieldSpec(
                "output_visualization_resource_id",
                "Visualização de Caso",
                OUTPUT_TYPES,
                is_output=True,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "Sobrepõe a distribuição dos scores do caso forense "
            "(questionada × padrão) sobre as distribuições de referência "
            "(genuínos verde, impostores vermelho) — KDE quando há ≥30 "
            "pares de caso, linhas verticais tracejadas com valor numérico "
            "quando há menos. Permite ao perito mostrar em laudo onde o "
            "log-LR do caso cai em relação à população de referência."
        )

    def build_validation_rules(self) -> list:
        """
        Regras dinâmicas (não persistidas):
        - CASEWORK_STATE(False) na referência — deve ser validação, não caso
        - CASEWORK_STATE(True) no caso — deve estar marcado como casework
        - SCORE_TYPE_MATCH — referência e caso devem ser RAW×RAW ou
          CALIBRATED×CALIBRATED (comparar escalas diferentes é sem sentido)
        """
        from faceaudit.validation_rules.casework_state_rule import CaseworkStateRule
        from faceaudit.validation_rules.score_type_match_rule import (
            ScoreTypeMatchRule,
        )

        rules: list = []
        if self.reference_scores_resource_id:
            rules.append(CaseworkStateRule(
                rule_id=f"{self.instruction_id}_reference_not_casework",
                scores_resource_id=self.reference_scores_resource_id,
                expected=False,
            ))
        if self.case_scores_resource_id:
            rules.append(CaseworkStateRule(
                rule_id=f"{self.instruction_id}_case_is_casework",
                scores_resource_id=self.case_scores_resource_id,
                expected=True,
            ))
        if self.reference_scores_resource_id and self.case_scores_resource_id:
            rules.append(ScoreTypeMatchRule(
                rule_id=f"{self.instruction_id}_score_type_match",
                reference_resource_id=self.reference_scores_resource_id,
                case_resource_id=self.case_scores_resource_id,
            ))
        return rules

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.reference_scores_resource_id:
            inputs.append(self.reference_scores_resource_id)

        if self.case_scores_resource_id:
            inputs.append(self.case_scores_resource_id)

        if self.output_visualization_resource_id:
            outputs.append(self.output_visualization_resource_id)

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
