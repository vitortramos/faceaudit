from __future__ import annotations
from enum import Enum


class LdaEstimatorType(Enum):
    """
    Define os estimadores da matriz de dispersão intra-classe (Sw) usada
    pelo Linear Discriminant Analysis.

    O LDA depende do eig generalizado `(Sb, Sw)`. Sw mal-condicionada é o
    problema típico em alta dimensionalidade / poucas amostras por classe.
    Estimadores com shrinkage analítico (Ledoit-Wolf, OAS) substituem a
    soma manual de outer products por uma forma mais robusta.
    """

    STANDARD = (
        "standard",
        "Padrão (Soma de Outer Products)",
        "Cálculo clássico do Fisher LDA. Sw é a soma das covariâncias por "
        "classe, com regularização ridge ε·I antes da inversão.\n\n"
        "Recomendado quando há muitas amostras por identidade e a "
        "dimensionalidade é moderada.",
    )

    LEDOIT_WOLF = (
        "ledoit_wolf",
        "Ledoit-Wolf (Robusto)",
        "Estimador de covariância com shrinkage analítico "
        "(sklearn.covariance.LedoitWolf). Computa Sw a partir dos vetores "
        "centrados por classe, aplicando shrinkage automaticamente em "
        "direção a um alvo diagonal.\n\n"
        "Ideal quando há poucas amostras por identidade — reduz overfitting "
        "do LDA e produz autovalores mais estáveis no eig generalizado.",
    )

    OAS = (
        "oas",
        "OAS (Oracle Shrinkage)",
        "Oracle Approximating Shrinkage (sklearn.covariance.OAS). Geralmente "
        "supera Ledoit-Wolf quando o número de amostras é muito menor que a "
        "dimensão do embedding. Recomendado em datasets forenses pequenos "
        "(N_amostras << embedding_dim).",
    )

    def __new__(cls, value: str, label: str, description: str) -> "LdaEstimatorType":
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
    def from_str(cls, value: str) -> "LdaEstimatorType":
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.STANDARD
