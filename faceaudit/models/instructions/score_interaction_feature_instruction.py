# faceaudit/models/instructions/score_interaction_feature_instruction.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.s0_mode import S0Mode
from faceaudit.enums.score_interaction_mode import ScoreInteractionMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.calibration.logreg_s0_config import LogRegS0Config
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="ScoreInteractionFeatureInstruction")


@dataclass
class ScoreInteractionFeatureInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para a feature forense de interação
    `(base − s0)·rarity` entre um score base (cosseno/PLDA) e a
    raridade derivada da tipicidade.

    A premissa forense: **raridade sozinha não é evidência de mesma
    origem**. Ela só deve modular a *força* de uma similaridade
    observada. A formulação correta é:

        logLR = β0 + β1·base + β2·f(base, s0)·rarity

    onde `s0` (ponto de evidência neutra, logLR=0) é aprendido no
    domínio de calibração e congelado para validação/casework. O
    output desta instrução é um `SimilarityScoreDataset` com
    `raw_score` 1D igual à feature derivada — consumível por
    `SCORE_VECTORIZATION` para empilhar com `[cosine]` (ou outras
    features) e treinar o calibrador LogReg multivariado.

    A instrução é **agnóstica à origem da raridade**: recebe dois
    `SimilarityScoreDataset` alinháveis por par (base + modifier
    de tipicidade) e produz a feature derivada. Quem decide se o
    modifier é simétrico ou unilateral (REFERENCE_ONLY/PROBE_ONLY)
    é o `TYPICALITY_PAIR_SCORING` upstream. O alinhamento de pares
    usa chave orientada `(probe_hash, reference_hash)` quando o
    modifier vem de combiner unilateral; senão usa
    `frozenset({probe, reference})` (compat com simétricos).

    Recomendações forenses:
    - Default: `interaction_mode=ONE_SIDED_POSITIVE` — raridade só
      amplifica evidência positiva (mais conservador, mais defensável).
    - Para o modelo plenamente geral (invariante à definição de
      raridade): combine downstream com uma segunda instância em modo
      `HINGE_BASE`, gerando `[base, HINGE_BASE, ONE_SIDED_POSITIVE]`.
    - `s0_mode=INHERIT` é **obrigatório** em validação e casework
      para preservar a definição de evidência neutra aprendida na
      calibração. AUTO_LOGREG em casework é bloqueado.
    """

    base_score_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "SimilarityScoreDataset base — tipicamente o COSSENO da "
            "calibração. Pode também ser PLDA (modo experimental — "
            "risco de double-counting com a covariância between-class)."
        )},
    )

    modifier_score_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "SimilarityScoreDataset com tipicidade pareada — output do "
            "TYPICALITY_PAIR_SCORING. Pode ser combiner simétrico "
            "(MEAN/...) ou unilateral (REFERENCE_ONLY/PROBE_ONLY). "
            "Ignorado quando `interaction_mode=HINGE_BASE`."
        )},
    )

    interaction_mode: ScoreInteractionMode = field(
        default=ScoreInteractionMode.ONE_SIDED_POSITIVE,
        metadata={"help": "Fórmula da feature derivada."},
    )

    s0_mode: S0Mode = field(
        default=S0Mode.AUTO_LOGREG,
        metadata={"help": (
            "Como resolver `s0` (ponto de evidência neutra). "
            "AUTO_LOGREG: ajusta calibrador auxiliar univariado no "
            "próprio dataset (calibração). MANUAL: usa "
            "`s0_manual_value`. INHERIT: lê `interaction_s0` do metadata "
            "do recurso `inherit_s0_from_resource_id` (obrigatório em "
            "validação e casework)."
        )},
    )

    s0_manual_value: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Valor fixo de s0 (só usado em `s0_mode=MANUAL`)."
        )},
    )

    inherit_s0_from_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Outro SimilarityScoreDataset interaction de onde herdar "
            "`interaction_s0`. Obrigatório em `s0_mode=INHERIT`."
        )},
    )

    s0_logreg_config_override: Optional[LogRegS0Config] = field(
        default=None,
        metadata={"help": (
            "Override opcional da configuração do LogReg auxiliar em "
            "AUTO_LOGREG. Quando vazio, defaults sãos são aplicados "
            "(C=1.0, solver=liblinear, class_weight=balanced, "
            "max_iter=1000)."
        )},
    )

    invert_rarity: bool = field(
        default=True,
        metadata={"help": (
            "Quando True, converte tipicidade→raridade por `score_type` "
            "do modifier (PERCENTILE: 1 − v, com v já em [0,1] vindo do "
            "TYPICALITY_PAIR_SCORING; MEAN_KNN_SIM/LOG_DENSITY: −v). "
            "Cosmético no caso bilateral, mas não cosmético no "
            "unilateral quando o offset C≠0 (percentil) — manter True "
            "por auditabilidade do sinal do coeficiente."
        )},
    )

    center_rarity: bool = field(
        default=False,
        metadata={"help": (
            "Quando True, centraliza a raridade por r0 antes de aplicar a "
            "fórmula da interação — produz `h × (rarity − r0)` em vez de "
            "`h × rarity`. r0 é calculado/herdado em paralelo a s0: "
            "AUTO_LOGREG → r0 = mean(rarity) no dataset atual; INHERIT → "
            "lê `interaction_r0` do recurso apontado por "
            "`inherit_s0_from_resource_id` (mesmo source que s0). "
            "Implementa o princípio hierárquico de Tukey: ao incluir "
            "interação no modelo [cos, h, h·(rarity−r0)], a decomposição "
            "fica interpretável (β₂ = efeito-base da hinge; β₃ = "
            "modulação específica pela raridade em torno de uma face "
            "típica). Sem efeito em HINGE_BASE (que ignora rarity)."
        )},
    )

    output_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "SimilarityScoreDataset de saída."},
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do dataset de saída."},
    )

    is_casework: bool = field(
        default=False,
        metadata={"help": (
            "Modo casework: zera `same_person` em todos os samples e "
            "marca `metadata.is_casework=True`. Requer "
            "`s0_mode=INHERIT` (AUTO_LOGREG é bloqueado em casework)."
        )},
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SCORE_INTERACTION_FEATURE

        if not self.name:
            self.name = "Score Interaction Feature"

        # Coerções defensivas quando o DTO vem de from_dict externo
        # (parâmetros podem chegar como string ou dict aninhado).
        if not isinstance(self.interaction_mode, ScoreInteractionMode):
            self.interaction_mode = ScoreInteractionMode.from_str(self.interaction_mode)
        if not isinstance(self.s0_mode, S0Mode):
            self.s0_mode = S0Mode.from_str(self.s0_mode)
        if self.s0_logreg_config_override is not None and isinstance(
            self.s0_logreg_config_override, dict
        ):
            self.s0_logreg_config_override = LogRegS0Config.from_dict(
                self.s0_logreg_config_override
            )

        self._sync_declared_io()
        self._ensure_default_validation_rules()

    # ─────────────────────────── form specs ────────────────────────────

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        IN = (StrategyResourceType.SIMILARITY_SCORES,)
        OUT = (StrategyResourceType.SIMILARITY_SCORES,)
        specs: List[ResourceFieldSpec] = [
            ResourceFieldSpec(
                "base_score_resource_id",
                "Score Base (cosseno/PLDA)",
                IN,
                # Trocar o tipo do base não muda outros specs, mas a
                # validação default depende dele — rebuild garante
                # repopulação imediata da regra MIN_CALIBRATION_PAIRS.
                triggers_rebuild=True,
            ),
        ]

        if self.interaction_mode is not ScoreInteractionMode.HINGE_BASE:
            specs.append(
                ResourceFieldSpec(
                    "modifier_score_resource_id",
                    "Modificador (Tipicidade)",
                    IN,
                )
            )

        if self.s0_mode is S0Mode.INHERIT:
            specs.append(
                ResourceFieldSpec(
                    "inherit_s0_from_resource_id",
                    "Herdar s0 de…",
                    IN,
                )
            )

        specs.append(
            ResourceFieldSpec(
                "output_resource_id",
                "Feature de Interação (saída)",
                OUT,
                is_output=True,
            )
        )
        return specs

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec(
                field_name="interaction_mode",
                label="Modo de Interação",
                choices=tuple(v for v, _, _ in ScoreInteractionMode.choices_with_description()),
                enum_type=ScoreInteractionMode,
                descriptions={
                    v: desc
                    for v, _label, desc in ScoreInteractionMode.choices_with_description()
                },
            ),
            ChoiceFieldSpec(
                field_name="s0_mode",
                label="Resolução de s0",
                choices=tuple(v for v, _, _ in S0Mode.choices_with_description()),
                enum_type=S0Mode,
                descriptions={
                    v: desc for v, _label, desc in S0Mode.choices_with_description()
                },
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        specs = [
            ValueFieldSpec(
                field_name="invert_rarity",
                label="Inverter Tipicidade → Raridade",
                value_type="bool",
            ),
            ValueFieldSpec(
                field_name="is_casework",
                label="Modo Casework (sem ground truth)",
                value_type="bool",
            ),
        ]
        if self.s0_mode is S0Mode.MANUAL:
            specs.append(
                ValueFieldSpec(
                    field_name="s0_manual_value",
                    label="Valor de s0 (manual)",
                    value_type="float",
                    decimals=6,
                )
            )
        return specs

    def get_configuration_help(self) -> Optional[str]:
        return (
            "Feature forense de interação: a raridade NÃO entra como "
            "evidência independente — ela modula a força de uma "
            "similaridade observada. A fórmula default "
            "(ONE_SIDED_POSITIVE: max(base−s0, 0)·rarity) garante que "
            "uma face rara que NÃO casa não vira evidência forte "
            "contra (defensável em corte).\n\n"
            "Protocolo de s0:\n"
            "• Calibração: s0_mode=AUTO_LOGREG (ajusta calibrador "
            "auxiliar univariado). O valor de s0 é gravado no metadata.\n"
            "• Validação A/B e Casework: s0_mode=INHERIT, apontando "
            "para a execução de calibração — preserva a definição de "
            "'evidência neutra'.\n\n"
            "Para o modelo plenamente geral (invariante à definição "
            "exata de raridade), combine downstream com uma segunda "
            "instância em modo HINGE_BASE, gerando "
            "[base, HINGE_BASE, ONE_SIDED_POSITIVE] no "
            "SCORE_VECTORIZATION."
        )

    # ─────────────────────────── helpers ───────────────────────────────

    def _sync_declared_io(self) -> None:
        inputs: list[str] = []
        if self.base_score_resource_id:
            inputs.append(self.base_score_resource_id)
        if (
            self.interaction_mode is not ScoreInteractionMode.HINGE_BASE
            and self.modifier_score_resource_id
        ):
            inputs.append(self.modifier_score_resource_id)
        if self.s0_mode is S0Mode.INHERIT and self.inherit_s0_from_resource_id:
            inputs.append(self.inherit_s0_from_resource_id)

        outputs: list[str] = []
        if self.output_resource_id:
            outputs.append(self.output_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # Repopula defaults após o usuário mudar inputs/s0_mode.
        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """Defaults persistentes:

        - MIN_CALIBRATION_PAIRS sobre o base quando `s0_mode=AUTO_LOGREG`
          (precisa de massa estatística para ajustar o calibrador
          auxiliar univariado). Em MANUAL/INHERIT não há fit, então o
          default não se aplica.
        """
        if self.validation_rules_user_managed:
            return

        self.validation_rule_configs = []

        if (
            self.s0_mode is S0Mode.AUTO_LOGREG
            and self.base_score_resource_id
        ):
            self.validation_rule_configs.append(
                ValidationRuleConfig(
                    rule_type=ValidationRuleType.MIN_CALIBRATION_PAIRS,
                    params={"scores_resource_id": self.base_score_resource_id},
                )
            )

    def build_validation_rules(self) -> list:
        """Regras dinâmicas:

        - `ScoreVectorPairCompatibilityRule` entre base e modifier
          (catch antecipado para conjuntos divergentes de pares). A
          regra usa `frozenset` (simétrica), o que é seguro porque o
          executor refaz a checagem com chave orientada quando o
          combiner do modifier é unilateral.
        """
        from faceaudit.validation_rules.score_vector_pair_compatibility_rule import (
            ScoreVectorPairCompatibilityRule,
        )

        rules: list = []
        if (
            self.interaction_mode is not ScoreInteractionMode.HINGE_BASE
            and self.base_score_resource_id
            and self.modifier_score_resource_id
        ):
            rules.append(
                ScoreVectorPairCompatibilityRule(
                    rule_id=f"{self.instruction_id}_pair_compat",
                    input_resource_ids=[
                        self.base_score_resource_id,
                        self.modifier_score_resource_id,
                    ],
                )
            )
        return rules

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed = data.copy()

        if "instruction_type" in processed and processed["instruction_type"]:
            processed["instruction_type"] = StrategyInstructionType.from_str(
                processed["instruction_type"]
            )
        if "interaction_mode" in processed and processed["interaction_mode"]:
            processed["interaction_mode"] = ScoreInteractionMode.from_str(
                processed["interaction_mode"]
            )
        if "s0_mode" in processed and processed["s0_mode"]:
            processed["s0_mode"] = S0Mode.from_str(processed["s0_mode"])
        if (
            "s0_logreg_config_override" in processed
            and isinstance(processed["s0_logreg_config_override"], dict)
        ):
            processed["s0_logreg_config_override"] = LogRegS0Config.from_dict(
                processed["s0_logreg_config_override"]
            )

        return cls(**processed)
