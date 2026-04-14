from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


class MiningStrategy(str, Enum):
    """
    Define a estratégia estatística/geométrica usada para curadoria das comparações.
    """
    RANDOM = "RANDOM"    # Amostragem aleatória uniforme
    HARD = "HARD"        # Busca os pares mais confusos/limítrofes (Top LLR)
    EASY = "EASY"        # Busca os pares mais óbvios/distintos (Bottom LLR)
    TYPICAL = "TYPICAL"  # Busca o comportamento mediano da distribuição (Miolo do LLR)
    MIXED = "MIXED"      # Usa uma composição predefinida das estratégias acima


class MiningFocus(str, Enum):
    """
    Define o nível principal da mineração.
    """
    PAIRS = "PAIRS"
    IDENTITIES = "IDENTITIES"


class GlobalSelectionMode(str, Enum):
    """
    Define como aplicar o corte global final quando max_global_pairs estiver definido.
    """
    GLOBAL_ORDER = "GLOBAL_ORDER"
    ROUND_ROBIN_BY_PIVOT = "ROUND_ROBIN_BY_PIVOT"
    ROUND_ROBIN_BY_IDENTITY_PAIR = "ROUND_ROBIN_BY_IDENTITY_PAIR"


@dataclass
class MixedProportions(JSONSerializableMixin):
    """
    Define as frações usadas quando a estratégia for MIXED.
    A soma das proporções ativas deve totalizar 1.0 (100%).
    """
    hard_ratio: float = 0.30
    typical_ratio: float = 0.20
    random_ratio: float = 0.50
    easy_ratio: float = 0.0

    def is_valid(self) -> bool:
        """Verifica se as proporções somam 1.0 (com tolerância de ponto flutuante)."""
        total = self.hard_ratio + self.typical_ratio + self.random_ratio + self.easy_ratio
        return abs(total - 1.0) < 1e-5
        
    @classmethod
    def from_dict(cls, data: dict):
        if not data:
            return cls()
        return cls(**data)


@dataclass
class MiningRule(JSONSerializableMixin):
    """
    Regra atômica que define como minerar um tipo específico de par
    (Genuíno ou Impostor).
    """

    strategy: MiningStrategy = field(
        default=MiningStrategy.RANDOM,
        metadata={"help": "Estratégia de extração estatística dos pares."}
    )
    
    mixed_proportions: MixedProportions = field(
        default_factory=MixedProportions,
        metadata={"help": "Configuração de frações utilizada apenas se strategy=MIXED."}
    )

    focus: MiningFocus = field(
        default=MiningFocus.PAIRS,
        metadata={"help": "Define se o foco é emparelhamento solto ou agrupado por identidade."}
    )

    max_global_pairs: Optional[int] = field(
        default=None,
        metadata={"help": "Limite rígido total de pares a retornar ao final da mineração."}
    )

    max_pairs_per_identity: Optional[int] = field(
        default=None,
        metadata={"help": "Limita quantos pares podem ser escolhidos por identidade pivô."}
    )

    max_identities: Optional[int] = field(
        default=None,
        metadata={"help": "Limita quantas identidades relacionadas serão escolhidas para cada identidade pivô."}
    )

    global_selection_mode: GlobalSelectionMode = field(
        default=GlobalSelectionMode.GLOBAL_ORDER,
        metadata={"help": "Define como o corte global final será aplicado (ex: Round Robin)."}
    )

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstrói a regra e seus Enums internos."""
        if not data:
            return cls()

        if "strategy" in data and isinstance(data["strategy"], str):
            data["strategy"] = MiningStrategy(data["strategy"])

        if "focus" in data and isinstance(data["focus"], str):
            data["focus"] = MiningFocus(data["focus"])

        if "global_selection_mode" in data and isinstance(data["global_selection_mode"], str):
            data["global_selection_mode"] = GlobalSelectionMode(data["global_selection_mode"])

        if "mixed_proportions" in data and isinstance(data["mixed_proportions"], dict):
            data["mixed_proportions"] = MixedProportions.from_dict(data["mixed_proportions"])

        return cls(**data)


@dataclass
class ComparisonMiningConfig(JSONSerializableMixin):
    """
    Motor de Regras (Rule Engine) para mineração e curadoria de pares biométricos.
    """

    id: str = field(default="")
    name: str = field(default="Nova Estratégia de Mineração")
    description: str = field(default="")

    seed: int = field(default=42)

    exclude_self_comparisons: bool = field(default=True)

    impostors_rule: MiningRule = field(default_factory=MiningRule)
    genuines_rule: MiningRule = field(default_factory=MiningRule)

    @classmethod
    def from_dict(cls, data: dict):
        if not data:
            return None

        if "impostors_rule" in data and isinstance(data["impostors_rule"], dict):
            data["impostors_rule"] = MiningRule.from_dict(data["impostors_rule"])

        if "genuines_rule" in data and isinstance(data["genuines_rule"], dict):
            data["genuines_rule"] = MiningRule.from_dict(data["genuines_rule"])

        return cls(**data)