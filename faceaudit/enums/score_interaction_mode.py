# faceaudit/enums/score_interaction_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class ScoreInteractionMode(Enum):
    """
    Define a fórmula da feature derivada produzida por
    `SCORE_INTERACTION_FEATURE` a partir do score base (cosseno/PLDA) e
    do modificador (raridade derivada de tipicidade).

    Os modos materializam a tese forense de que **raridade sozinha não
    é evidência de mesma origem** — ela só deve *modular a força* de
    uma similaridade observada. `s0` é o ponto de evidência neutra
    (logLR=0) aprendido no domínio de calibração e congelado.

    Variantes:

    - `ONE_SIDED_POSITIVE` (default): `max(base − s0, 0) · rarity`.
      Raridade só amplifica evidência POSITIVA. Conservador — uma face
      rara que NÃO casa não vira evidência forte contra (mais difícil
      de justificar em corte). É o default recomendado.

    - `LINEAR`: `(base − s0) · rarity`. Bilateral; permite raridade
      modular evidência negativa. Matematicamente mais simples;
      experimental por construção.

    - `HINGE_BASE`: `max(base − s0, 0)` (ignora o modificador).
      Feature-companheira para montar o modelo unilateral plenamente
      geral `[base, HINGE_BASE, HINGE_BASE·rarity]` — invariante à
      definição de raridade (offset constante absorvido) e usada
      como controle no A/B para separar "ganho por não-linearidade
      simples" de "ganho específico por raridade".
    """

    ONE_SIDED_POSITIVE = (
        "ONE_SIDED_POSITIVE",
        "Unilateral Positivo (recomendado)",
        "max(base − s0, 0) · rarity. Raridade só amplifica evidência "
        "POSITIVA — uma face rara que não casa NÃO vira evidência "
        "forte contra. Default conservador e defensável.",
    )

    LINEAR = (
        "LINEAR",
        "Linear (bilateral)",
        "(base − s0) · rarity. Bilateral: raridade modula evidência em "
        "ambos os lados de s0. Mais simples matematicamente; "
        "experimental — pode dificultar a defesa quando o caso cai "
        "em região de evidência neutra ou negativa.",
    )

    HINGE_BASE = (
        "HINGE_BASE",
        "Hinge Base (sem raridade)",
        "max(base − s0, 0). Não usa o modificador — feature-companheira "
        "para o modelo unilateral plenamente geral "
        "[base, HINGE_BASE, HINGE_BASE·rarity]. Controle do A/B: "
        "se este modelo sozinho já bate o baseline forte, o ganho "
        "veio da não-linearidade hinge, não da raridade.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @property
    def requires_modifier(self) -> bool:
        """True quando o modo de fato lê a coluna de raridade.

        `HINGE_BASE` ignora o modifier; o validador relaxa a exigência
        de `modifier_score_resource_id` nesse caso.
        """
        return self is not ScoreInteractionMode.HINGE_BASE

    @classmethod
    def from_str(cls, value: str) -> "ScoreInteractionMode":
        if not value:
            return cls.ONE_SIDED_POSITIVE
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.ONE_SIDED_POSITIVE

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
