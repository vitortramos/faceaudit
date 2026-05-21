from __future__ import annotations
from enum import Enum


class WccnEstimatorType(Enum):
    """
    Define os estimadores de covariância intraclasse (W) para WCCN.
    Contém metadados para UI (label_pt e descrição).
    """

    STANDARD = (
        "standard",
        "Padrão (Média de Outer Products)",
        "Método clássico do WCCN (Hatch et al. 2006). Calcula W como média "
        "das covariâncias por classe, com regularização ridge ε·I.\n\n"
        "Recomendado quando há identidades suficientes com múltiplas amostras.",
    )

    LEDOIT_WOLF = (
        "ledoit_wolf",
        "Ledoit-Wolf (Robusto)",
        "Estimador de covariância com shrinkage analítico (sklearn.covariance.LedoitWolf). "
        "Aplica shrinkage automático sobre os vetores centrados por classe.\n\n"
        "Ideal quando há poucas amostras por identidade, pois evita W mal-condicionada.",
    )

    OAS = (
        "oas",
        "OAS (Oracle Shrinkage)",
        "Oracle Approximating Shrinkage (sklearn.covariance.OAS). "
        "Geralmente supera Ledoit-Wolf quando o número de amostras é muito "
        "menor que a dimensão do embedding.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "WccnEstimatorType":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "WccnEstimatorType":
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.STANDARD
