# faceaudit/models/instructions/dataset_balancing_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import DatasetBalancingSelection
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)


T = TypeVar("T", bound="DatasetBalancingInstruction")


@dataclass
class DatasetBalancingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para balancear o número de amostras por identidade.

    Duas dimensões ortogonais:

    1. `balancing_mode` — define o alvo de samples por identidade:
       - CAP_ONLY: cap em `target_n`; IDs com menos amostras são mantidas.
       - FLOOR_AND_CAP: cap + floor; IDs com < `target_n` são descartadas.
       - AUTO_MIN_BALANCE: N = min observado; todas as IDs ficam com N.

    2. `selection_criterion` — define como descartar excesso (quando há):
       - RANDOM: sorteio uniforme reprodutível via `random_seed`.
       - BY_QUALITY: top-N por `quality_metric` (maior = melhor).

    Saída: novo `INTERMEDIATE_DATASET` (input preservado).
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset que será balanceado."}
    )

    output_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset balanceado."}
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do dataset balanceado."}
    )

    balancing_mode: DatasetBalancingMode = field(
        default=DatasetBalancingMode.CAP_ONLY,
        metadata={"help": "Modo de balanceamento (cap, floor+cap ou auto-min)."}
    )

    target_n: int = field(
        default=10,
        metadata={"help": (
            "Número alvo de amostras por identidade. Usado em CAP_ONLY e "
            "FLOOR_AND_CAP. Ignorado em AUTO_MIN_BALANCE (que computa N "
            "como o mínimo observado em runtime)."
        )}
    )

    selection_criterion: DatasetBalancingSelection = field(
        default=DatasetBalancingSelection.RANDOM,
        metadata={"help": "Critério para escolher quais amostras descartar."}
    )

    quality_metric: QualityScoreType = field(
        default=QualityScoreType.OFIQ,
        metadata={"help": (
            "Métrica de qualidade usada quando selection_criterion=BY_QUALITY. "
            "Direção: maior = melhor para todas as métricas suportadas."
        )}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": (
            "Semente aleatória para RANDOM. Em BY_QUALITY é usada apenas "
            "para tie-breaking determinístico em empates."
        )}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.DATASET_BALANCING

        if not self.name:
            self.name = "Dataset Balancing"

        self._sync_declared_io()

    # ── Field specs para a UI ────────────────────────────────────────

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        specs: List[ChoiceFieldSpec] = [
            ChoiceFieldSpec(
                field_name="balancing_mode",
                label="Modo de Balanceamento",
                choices=("CAP_ONLY", "FLOOR_AND_CAP", "AUTO_MIN_BALANCE"),
                enum_type=DatasetBalancingMode,
                descriptions={
                    "CAP_ONLY":
                        "Limita a `target_n` amostras por identidade. "
                        "Identidades com MENOS amostras permanecem inalteradas. "
                        "Mais conservador — preserva o máximo de dados.",
                    "FLOOR_AND_CAP":
                        "Aplica `target_n` como piso E cap. Identidades com "
                        "menos de `target_n` amostras são DESCARTADAS por "
                        "completo. Garante uniformidade do mínimo de cobertura.",
                    "AUTO_MIN_BALANCE":
                        "Computa N = min(samples_per_identity) em runtime e "
                        "aplica como cap. Resultado é perfeitamente balanceado "
                        "(todas as IDs com mesma quantidade), mas pode descartar "
                        "muitos dados se houver IDs com muito poucas amostras.",
                },
            ),
            ChoiceFieldSpec(
                field_name="selection_criterion",
                label="Critério de Seleção",
                choices=("RANDOM", "BY_QUALITY"),
                enum_type=DatasetBalancingSelection,
                descriptions={
                    "RANDOM":
                        "Sorteio uniforme controlado por `random_seed` — "
                        "reprodutível.",
                    "BY_QUALITY":
                        "Mantém as N melhores amostras de cada identidade "
                        "ordenadas por uma métrica de qualidade. Empates "
                        "são quebrados deterministicamente.",
                },
            ),
        ]

        # Métrica de qualidade só aparece quando o critério é BY_QUALITY.
        # Filtragem dinâmica por disponibilidade no dataset será feita pelo
        # painel da UI (lê stats/parquet e desabilita opções inviáveis); o
        # DTO sempre oferece o conjunto completo.
        if self.uses_quality_selection:
            specs.append(
                ChoiceFieldSpec(
                    field_name="quality_metric",
                    label="Métrica de Qualidade",
                    choices=tuple(qt.name for qt in QualityScoreType),
                    enum_type=QualityScoreType,
                    descriptions={
                        "FIQA":
                            "Score do ForensicFace (0.0–1.0). Lê de "
                            "`face.ff_info.fiqa_score`.",
                        "OFIQ":
                            "Score unificado ISO/OFIQ (0.0–100.0). Lê de "
                            "`face.ofiq_info.unified_quality_score_scalar`.",
                        "SERFIQ":
                            "Score SER-FIQ estocástico (0.0–100.0). Lê de "
                            "`face.serfiq_info.score`.",
                        "EMBEDDING_NORM":
                            "Norma L2 do embedding sepaelv2/AdaFace. "
                            "Maior norma = mais confiança no extrator. "
                            "Calculada on-the-fly — não requer coluna no parquet.",
                    },
                )
            )

        return specs

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET, StrategyResourceType.INTERMEDIATE_DATASET)
        OUT = (StrategyResourceType.INTERMEDIATE_DATASET,)

        return [
            ResourceFieldSpec("dataset_resource_id", "Dataset de entrada", DS),
            ResourceFieldSpec(
                "output_dataset_resource_id",
                "Dataset balanceado",
                OUT,
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        specs: List[ValueFieldSpec] = []

        # target_n não é editável em AUTO_MIN_BALANCE — N é computado em runtime.
        if not self.uses_auto_min:
            specs.append(ValueFieldSpec(
                field_name="target_n",
                label="Amostras por Identidade (N)",
                value_type="int",
                min_value=1,
                max_value=10**5,
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
        if self.uses_floor_and_cap:
            n = self.target_n
            return (
                f"⚠️ Identidades com menos de {n} amostra(s) serão DESCARTADAS "
                "por completo (não aparecem no dataset balanceado)."
            )
        if self.uses_auto_min:
            return (
                "ℹ️ N será computado em runtime como o MÍNIMO de amostras "
                "observado entre as identidades. Se houver alguma identidade "
                "com pouquíssimas amostras, todas as outras serão reduzidas "
                "ao mesmo nível — atenção a outliers."
            )
        return None

    # ── Sincronização de IO declarado ────────────────────────────────

    def _sync_declared_io(self) -> None:
        inputs: List[str] = []
        outputs: List[str] = []

        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)

        if self.output_dataset_resource_id:
            outputs.append(self.output_dataset_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    # ── Properties para UI/executor ──────────────────────────────────

    @property
    def uses_cap_only(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.CAP_ONLY

    @property
    def uses_floor_and_cap(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.FLOOR_AND_CAP

    @property
    def uses_auto_min(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.AUTO_MIN_BALANCE

    @property
    def uses_random_selection(self) -> bool:
        return self.selection_criterion == DatasetBalancingSelection.RANDOM

    @property
    def uses_quality_selection(self) -> bool:
        return self.selection_criterion == DatasetBalancingSelection.BY_QUALITY

    # ── Reconstrução a partir de dict ────────────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if not data:
            raise ValueError("data não pode ser vazio")

        d = dict(data)

        # Hidratação de enums (com fallbacks seguros via from_str).
        d["balancing_mode"] = DatasetBalancingMode.from_str(
            d.get("balancing_mode", DatasetBalancingMode.CAP_ONLY.value)
        )
        d["selection_criterion"] = DatasetBalancingSelection.from_str(
            d.get("selection_criterion", DatasetBalancingSelection.RANDOM.value)
        )
        d["quality_metric"] = QualityScoreType.from_str(
            d.get("quality_metric", QualityScoreType.OFIQ.name)
        )

        # instruction_type sempre fixo — força para o tipo correto.
        d["instruction_type"] = StrategyInstructionType.DATASET_BALANCING

        # Filtra apenas chaves válidas para evitar TypeError.
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}

        return cls(**filtered)
