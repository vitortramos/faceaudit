from __future__ import annotations
from enum import Enum
from typing import List, Tuple, Optional

class PLDAMode(Enum):
    """
    Define como escolher a dimensionalidade latente (rank) usada no treino do PLDA.

    - FULL_RANK: usa o maior rank permitido pelos dados: min(D, C-1)
    - LOW_RANK: limita o rank a um valor configurável (latent_dim), respeitando min(D, C-1)
    """

    FULL_RANK = (
        "FULL_RANK",
        "Full-rank (automático)",
        "Usa a maior dimensão latente permitida pelos dados (min(D, C-1)). "
        "Preserva o máximo de informação, mas exige muitas identidades para ficar estável.",
    )

    LOW_RANK = (
        "LOW_RANK",
        "Low-rank (controlado)",
        "Limita a dimensão latente ao valor definido (latent_dim), respeitando min(D, C-1). "
        "Geralmente mais estável com menos identidades e com variações fortes de domínio.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "PLDAMode":
        # Definir _value_ aqui (em __new__) é crítico para que a metaclass do
        # Enum registre o membro no _value2member_map_ pelo VALOR string
        # ("LOW_RANK"), e não pela tupla. Isso é o que faz PLDAMode("LOW_RANK")
        # funcionar — incluindo o caminho que pickle usa para reconstruir o
        # enum durante a desserialização. Definir em __init__ era um bug:
        # a metaclass já tinha indexado pelo valor inicial (a tupla) antes
        # de __init__ rodar.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.description = description
        return obj

    def __str__(self):
        """Retorna o label amigável para a interface."""
        return self.label

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna a chave string (ex: 'FULL_RANK') para o JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> PLDAMode:
        """
        Reconstrói o Enum a partir de uma string de forma robusta.
        Incorpora a lógica de limpeza (strip, upper, replace) que você já possuía.
        """
        if not value:
            return cls.FULL_RANK

        if isinstance(value, cls):
            return value

        # Normalização: "full-rank" ou "Full Rank" -> "FULL_RANK"
        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            # Tenta buscar pelo nome da constante (cls['FULL_RANK'])
            return cls[val_norm]
        except KeyError:
            # Tenta buscar pelo valor técnico (.value)
            for m in cls:
                if m.value == val_norm:
                    return m
            
            # Fallback seguro para evitar crash em produção
            return cls.FULL_RANK

    # --- Helpers para UI ---

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Útil para carregar QComboBox: [(value, label), ...]"""
        return [(m.value, m.label) for m in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        """Útil para UI com Tooltips: [(value, label, description), ...]"""
        return [(m.value, m.label, m.description) for m in cls]