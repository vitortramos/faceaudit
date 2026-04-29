from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


class SplitMode(str, Enum):
    """
    Define a métrica de alvo (cota) usada para dividir os dados de calibração.
    """
    ABSOLUTE = "ABSOLUTE"  # Usa metas fixas (min_calibration_pairs, min_calibration_ids)
    RATIO = "RATIO"        # Separa uma percentagem das identidades (ex: 20%), garantindo o piso mínimo


class GroupingStrategy(str, Enum):
    """
    Define como as identidades serão selecionadas e agrupadas ANTES de atingir a cota do SplitMode.
    Isso é vital para garantir que a dificuldade do dataset (sósias) seja preservada
    tanto no Treino quanto na Calibração.
    """
    RANDOM_IDENTITIES = "RANDOM_IDENTITIES"
    # Sorteio cego. Risco: Sósias podem ser separados em datasets diferentes, impedindo o Hard Mining.

    PRESERVE_LOOKALIKES = "PRESERVE_LOOKALIKES"
    # Agrupamento inteligente. Calcula a Matriz de Confusão e força que as N identidades mais
    # parecidas entre si viajem juntas para o mesmo dataset (Treino ou Calibração).

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "GroupingStrategy":
        """Reconstrói o enum a partir de string com fallback seguro para RANDOM_IDENTITIES."""
        if not value:
            return cls.RANDOM_IDENTITIES
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[value]
            except KeyError:
                return cls.RANDOM_IDENTITIES


@dataclass
class SplitConfig(JSONSerializableMixin):
    """
    Configuração do Motor de Divisão de Dados (Splitter) para treino.
    Define o que vai para a aprendizagem da matriz (PLDA) e o que vai para o Calibrador (LogReg).
    """
    name: str = field(default="Nova Configuração de Split", metadata={"help": "Nome único para identificar esta configuração."})
    description: Optional[str] = field(default=None, metadata={"help": "Descrição detalhada."})

    # --- Estratégias Core ---
    mode: SplitMode = field(
        default=SplitMode.RATIO,
        metadata={"help": "Se deve usar proporção (RATIO) ou números fixos (ABSOLUTE)."}
    )
    
    grouping_strategy: GroupingStrategy = field(
        default=GroupingStrategy.PRESERVE_LOOKALIKES,
        metadata={"help": "Se deve sortear pessoas isoladas ou clusters de sósias."}
    )
    
    cluster_size: int = field(
        default=3,
        metadata={
            "help": (
                "Válido apenas se grouping_strategy=PRESERVE_LOOKALIKES. "
                "Define o tamanho do pacote de sósias. Ex: 3 = O sorteado + seus 2 piores sósias."
            )
        }
    )

    # --- Parâmetros de Proporção (Se mode=RATIO) ---
    calibration_ratio: float = field(
        default=0.20,
        metadata={"help": "Fração de identidades reservadas para calibração (0.0 a 1.0). Predefinição: 20%."}
    )

    # --- Parâmetros Absolutos e Pisos de Segurança ---
    min_calibration_pairs: int = field(
        default=3000,
        metadata={"help": "Meta rígida de pares genuínos (Modo Absoluto) ou piso de segurança (Modo Ratio)."}
    )
    
    min_calibration_ids: int = field(
        default=40,
        metadata={"help": "Meta rígida de identidades (Modo Absoluto) ou piso de segurança (Modo Ratio)."}
    )

    # --- Filtros de Qualidade ---
    min_samples_per_id: int = field(
        default=2,
        metadata={"help": "Apenas identidades com no mínimo este número de amostras serão elegíveis para o split."}
    )

    # --- Reprodutibilidade ---
    seed: int = field(
        default=42,
        metadata={"help": "Semente para garantir que a divisão das identidades seja sempre idêntica em reexecuções."}
    )

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstrói a configuração e os seus Enums internos a partir do JSON guardado."""
        if not data:
            return cls()

        if "mode" in data and isinstance(data["mode"], str):
            data["mode"] = SplitMode(data["mode"])
            
        if "grouping_strategy" in data and isinstance(data["grouping_strategy"], str):
            data["grouping_strategy"] = GroupingStrategy(data["grouping_strategy"])

        return cls(**data)