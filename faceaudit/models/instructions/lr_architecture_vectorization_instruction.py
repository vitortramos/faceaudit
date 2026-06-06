# faceaudit/models/instructions/lr_architecture_vectorization_instruction.py
"""
DTO da nova instrução declarativa `LR_ARCHITECTURE_VECTORIZATION`.

Substitui o encadeamento histórico `SCORE_INTERACTION_FEATURE` ×N +
`SCORE_VECTORIZATION` por uma única declaração: o usuário escolhe a
arquitetura (BILATERAL cheia, HIERÁRQUICA, etc.) e o executor cuida da
construção interna das hinges, raridades centradas, vetorização e
binding do fingerprint para o calibrador.

A obrigatoriedade dos campos depende da arquitetura + parameter_mode:
  - Arquiteturas sem tipicidade (COSINE_ONLY/HINGE_ONLY/TWO_SIDED_HINGE_ONLY)
    escondem `rarity_probe_resource_id` e `rarity_reference_resource_id`.
  - parameter_mode != MANUAL esconde `manual_s0` e `manual_r0_*`.
  - parameter_mode != INHERIT esconde `inherit_architecture_state_from_resource_id`.
  - parameter_mode != FIT_FULL_CALIBRATION_LEGACY esconde
    `s0_logreg_config_override`.

Sem `is_casework` no DTO — a fase é derivada do `InstructionExecutionFrame`
pelo executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

from faceaudit.enums.architecture_parameter_mode import ArchitectureParameterMode
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="LRArchitectureVectorizationInstruction")


@dataclass
class LRArchitectureVectorizationInstruction(StrategyInstructionDefinition):
    """
    Declaração de arquitetura de vetorização para o calibrador LR.

    Inputs:
      - base_score_resource_id: SIMILARITY_SCORES com o cosseno (ou PLDA).
      - rarity_probe_resource_id / rarity_reference_resource_id:
        SIMILARITY_SCORES com tipicidade do par (output do
        TYPICALITY_PAIR_SCORING com combiner=PROBE_ONLY ou REFERENCE_ONLY).
        Obrigatórios apenas para arquiteturas que distinguem lados.

    Outputs:
      - output_resource_id: SIMILARITY_SCORES vetorizado, consumido por
        CALIBRATION_MODEL_TRAINING.
      - output_raw_input_resource_id: LR_ARCHITECTURE_RAW_INPUT com a
        matriz bruta (s, r_F, r_P) — necessário para o FIT_PER_FOLD da v2.

    `REACTIVE_FIELDS` ativa rebuild do form quando arquitetura ou
    parameter_mode mudam. `get_hidden_fields()` retorna o conjunto de
    fields a esconder com base na escolha atual.
    """

    REACTIVE_FIELDS: Tuple[str, ...] = ("architecture", "parameter_mode")

    # ── Inputs ──────────────────────────────────────────────────────────
    base_score_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "SIMILARITY_SCORES com o score base (cosseno ou PLDA)."},
    )
    rarity_probe_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "SIMILARITY_SCORES produzido por TYPICALITY_PAIR_SCORING "
            "(combiner=PROBE_ONLY tipicamente) — fornece r_F. "
            "Obrigatório apenas para arquiteturas que consomem r_F."
        )},
    )
    rarity_reference_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "SIMILARITY_SCORES produzido por TYPICALITY_PAIR_SCORING "
            "(combiner=REFERENCE_ONLY tipicamente) — fornece r_P. "
            "Obrigatório apenas para arquiteturas que consomem r_P."
        )},
    )
    rarity_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "SIMILARITY_SCORES com a tipicidade da face única — usado apenas "
            "em arquiteturas ONE_FACE (HIERARCHICAL_CENTERED_ONE_FACE, "
            "BILATERAL_FULL_CENTERED_ONE_FACE, ONE_SIDED_*_ONE_FACE, "
            "LINEAR_*_ONE_FACE). Quando preenchido, rarity_probe_resource_id "
            "e rarity_reference_resource_id devem estar vazios."
        )},
    )

    # ── Arquitetura + parâmetros ────────────────────────────────────────
    architecture: LRCalibrationArchitecture = field(
        default=LRCalibrationArchitecture.COSINE_ONLY,
        metadata={"help": (
            "Arquitetura de vetorização escolhida. Veja "
            "LRCalibrationArchitecture para a descrição matemática de "
            "cada variante. As 9 arquiteturas cobrem o espaço testado no "
            "estudo (COSINE_ONLY baseline a BILATERAL_FULL_CENTERED D=7)."
        )},
    )
    parameter_mode: ArchitectureParameterMode = field(
        default=ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
        metadata={"help": (
            "Modo de obtenção de s₀ e r̄_F/r̄_P. "
            "FIT_FULL_CALIBRATION_LEGACY (default v1) ajusta antes da CV "
            "interna — sujeito a leakage residual. FIT_PER_FOLD é v2. "
            "MANUAL usa manual_s0/manual_r0_*. INHERIT carrega state "
            "completo de execução anterior."
        )},
    )
    manual_s0: Optional[float] = field(default=None)
    manual_r0_probe: Optional[float] = field(default=None)
    manual_r0_reference: Optional[float] = field(default=None)
    inherit_architecture_state_from_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Recurso (SIMILARITY_SCORES vetorizado por esta mesma "
            "instrução numa execução anterior) cujo metadata "
            "`architecture_state` será reaplicado. OBRIGATÓRIO em "
            "VALIDATION e CASEWORK."
        )},
    )
    s0_logreg_config_override: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": (
            "Override opcional dos parâmetros do LogReg auxiliar usado "
            "pelo AUTO_LOGREG (C, solver, class_weight, max_iter). "
            "None usa defaults."
        )},
    )

    # ── Outputs ─────────────────────────────────────────────────────────
    output_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso SIMILARITY_SCORES vetorizado de saída."},
    )
    output_raw_input_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Recurso LR_ARCHITECTURE_RAW_INPUT contendo a matriz bruta "
            "(s, r_F, r_P) + pair_keys + identidades. Necessário para o "
            "FIT_PER_FOLD da v2. Opcional na v1 — se vazio, payload "
            "bruto não é persistido."
        )},
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION
        if not self.name:
            self.name = "Vetorização por Arquitetura LR"
        self._sync_declared_io()
        self._ensure_default_validation_rules()

    # ─────────────────────────────────────────────────────────────────────
    # UI specs
    # ─────────────────────────────────────────────────────────────────────

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        IN = (StrategyResourceType.SIMILARITY_SCORES,)
        OUT_SCORES = (StrategyResourceType.SIMILARITY_SCORES,)
        OUT_RAW = (StrategyResourceType.LR_ARCHITECTURE_RAW_INPUT,)
        return [
            ResourceFieldSpec(
                "base_score_resource_id", "Score base (cosseno/PLDA)", IN,
            ),
            ResourceFieldSpec(
                "rarity_probe_resource_id",
                "Tipicidade do probe (r_F)", IN,
            ),
            ResourceFieldSpec(
                "rarity_reference_resource_id",
                "Tipicidade do reference (r_P)", IN,
            ),
            ResourceFieldSpec(
                "rarity_resource_id",
                "Tipicidade da face única (ONE_FACE)", IN,
            ),
            ResourceFieldSpec(
                "inherit_architecture_state_from_resource_id",
                "Recurso de origem do state (INHERIT)", OUT_SCORES,
            ),
            ResourceFieldSpec(
                "output_resource_id", "Scores vetorizados (saída)", OUT_SCORES,
                is_output=True,
            ),
            ResourceFieldSpec(
                "output_raw_input_resource_id",
                "Payload bruto LR (saída opcional v1, obrigatório v2)",
                OUT_RAW, is_output=True,
            ),
        ]

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        arch_descs = {
            item.value: item.description for item in LRCalibrationArchitecture
        }
        mode_descs = {
            item.value: item.description for item in ArchitectureParameterMode
        }
        return [
            ChoiceFieldSpec(
                field_name="architecture",
                label="Arquitetura",
                choices=tuple(item.value for item in LRCalibrationArchitecture),
                enum_type=LRCalibrationArchitecture,
                descriptions=arch_descs,
            ),
            ChoiceFieldSpec(
                field_name="parameter_mode",
                label="Modo de ajuste de s₀/r₀",
                choices=tuple(item.value for item in ArchitectureParameterMode),
                enum_type=ArchitectureParameterMode,
                descriptions=mode_descs,
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="manual_s0", label="s₀ manual",
                value_type="float", decimals=4,
                placeholder="ex: 0.253",
            ),
            ValueFieldSpec(
                field_name="manual_r0_probe", label="r̄_F manual (centragem do probe)",
                value_type="float", decimals=4,
                placeholder="ex: 0.4566",
            ),
            ValueFieldSpec(
                field_name="manual_r0_reference",
                label="r̄_P manual (centragem do reference)",
                value_type="float", decimals=4,
                placeholder="ex: 0.4962",
            ),
        ]

    # ─────────────────────────────────────────────────────────────────────
    # Visibilidade condicional (REACTIVE_FIELDS)
    # ─────────────────────────────────────────────────────────────────────

    def get_hidden_fields(self) -> Set[str]:
        """Esconde campos irrelevantes para a arquitetura/modo atual."""
        spec = LR_ARCHITECTURE_SPECS.get(self.architecture)
        hidden: Set[str] = set()

        if spec is None:
            return hidden  # estado inicial; nada a esconder ainda

        requires_single = getattr(spec, "requires_typicality_single", False)

        if not spec.requires_typicality_probe:
            hidden.add("rarity_probe_resource_id")
        if not spec.requires_typicality_reference:
            hidden.add("rarity_reference_resource_id")
        if not requires_single:
            hidden.add("rarity_resource_id")

        if self.parameter_mode != ArchitectureParameterMode.MANUAL:
            hidden.update({
                "manual_s0", "manual_r0_probe", "manual_r0_reference",
            })
        else:
            # MANUAL: esconde os manuais não aplicáveis à arquitetura
            if not spec.requires_s0:
                hidden.add("manual_s0")
            # ONE_FACE não usa manual_r0_probe/reference (usa manual_r0_probe
            # como manual_r0_single via override do executor)
            if not spec.requires_typicality_probe or spec.centering_mode.value == "NONE":
                hidden.add("manual_r0_probe")
            if not spec.requires_typicality_reference or spec.centering_mode.value == "NONE":
                hidden.add("manual_r0_reference")
            # ONE_FACE reusa o slot manual_r0_probe semanticamente — UI ainda
            # esconde porque o spec não exige probe; mostraremos manual_r0_probe
            # quando ONE_FACE+MEAN, com label override seria ideal mas aqui
            # vamos simplesmente expor manual_r0_probe como manual_r0_single
            if requires_single and spec.centering_mode.value == "MEAN":
                hidden.discard("manual_r0_probe")

        if self.parameter_mode != ArchitectureParameterMode.INHERIT:
            hidden.add("inherit_architecture_state_from_resource_id")

        if (
            self.parameter_mode != ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY
            or not spec.requires_s0
        ):
            hidden.add("s0_logreg_config_override")

        return hidden

    # ─────────────────────────────────────────────────────────────────────
    # IO + regras
    # ─────────────────────────────────────────────────────────────────────

    def _sync_declared_io(self) -> None:
        inputs: List[str] = []
        for rid in (
            self.base_score_resource_id,
            self.rarity_probe_resource_id,
            self.rarity_reference_resource_id,
            self.rarity_resource_id,
            self.inherit_architecture_state_from_resource_id,
        ):
            if rid and rid not in inputs:
                inputs.append(rid)
        outputs: List[str] = []
        for rid in (self.output_resource_id, self.output_raw_input_resource_id):
            if rid and rid not in outputs:
                outputs.append(rid)

        self.input_resource_ids = inputs
        self.output_resource_ids = outputs

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Defaults persistentes mínimos. A validação contextual rica (orientação,
        fingerprint, fase) ocorre no executor (não no DTO).
        """
        if self.validation_rules_user_managed:
            return
        self.validation_rule_configs = []
        if self.base_score_resource_id:
            # Ajusta s₀ em FIT_FULL_CALIBRATION_LEGACY exige pares mínimos
            if self.parameter_mode == ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY:
                self.validation_rule_configs.append(
                    ValidationRuleConfig(
                        rule_type=ValidationRuleType.MIN_CALIBRATION_PAIRS,
                        params={"scores_resource_id": self.base_score_resource_id},
                    )
                )

    def build_validation_rules(self) -> list:
        """
        Regras dinâmicas. A maioria das validações (orientação, fingerprint,
        fase) é responsabilidade do executor — DTO declara apenas pré-requisitos
        estruturais detectáveis sem contexto de execução.
        """
        return []

    # ─────────────────────────────────────────────────────────────────────
    # Serialização
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None
        processed = dict(data)
        if "architecture" in processed and not isinstance(
            processed["architecture"], LRCalibrationArchitecture
        ):
            processed["architecture"] = LRCalibrationArchitecture.from_str(
                processed["architecture"]
            )
        if "parameter_mode" in processed and not isinstance(
            processed["parameter_mode"], ArchitectureParameterMode
        ):
            processed["parameter_mode"] = ArchitectureParameterMode.from_str(
                processed["parameter_mode"]
            )
        # Garante presença do instruction_type (será re-injetado por
        # __post_init__, mas o construtor dataclass o exige).
        processed["instruction_type"] = StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION
        return cls(**processed)

    def to_dict(self) -> dict:
        d = super().to_dict()
        if isinstance(d.get("architecture"), LRCalibrationArchitecture):
            d["architecture"] = d["architecture"].value
        elif "architecture" in d and hasattr(d["architecture"], "value"):
            d["architecture"] = d["architecture"].value
        if isinstance(d.get("parameter_mode"), ArchitectureParameterMode):
            d["parameter_mode"] = d["parameter_mode"].value
        elif "parameter_mode" in d and hasattr(d["parameter_mode"], "value"):
            d["parameter_mode"] = d["parameter_mode"].value
        return d
