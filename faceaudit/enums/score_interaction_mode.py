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

    - `ONE_SIDED_NEGATIVE` (V2.13): `max(s0 − base, 0) · rarity`.
      Espelho do `ONE_SIDED_POSITIVE` para o lado abaixo de `s0`. A
      raridade só modula evidência NEGATIVA — duas faces raras com
      cosseno baixo viram evidência mais forte de origens distintas
      que duas faces comuns com mesmo cosseno baixo. Pareada com
      `ONE_SIDED_POSITIVE` no modelo bilateral
      `[base, h_+, h_-, h_+·rarity, h_-·rarity]`. **Apenas
      experimental**: a assertiva forense de que a raridade pode
      reforçar evidência contra a mesma origem em corte ainda exige
      validação adicional.

    - `HINGE_BASE_NEGATIVE` (V2.13): `max(s0 − base, 0)` (ignora o
      modificador). Espelho da `HINGE_BASE` para o lado negativo.
      Feature-companheira para o modelo bilateral plenamente geral
      `[base, h_+, h_-, h_+·rarity, h_-·rarity, h_-·rarity, ...]`
      e controle estrutural análogo à HINGE_BASE para isolar a
      contribuição da modulação bilateral por raridade da
      contribuição da dobra bilateral em si.
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

    ONE_SIDED_NEGATIVE = (
        "ONE_SIDED_NEGATIVE",
        "Unilateral Negativo (experimental)",
        "max(s0 − base, 0) · rarity. Espelho do ONE_SIDED_POSITIVE: "
        "raridade só modula evidência NEGATIVA. Pareado com o positivo "
        "no modelo bilateral [base, h_+, h_-, h_+·rarity, h_-·rarity]. "
        "Apenas experimental — a defensibilidade de que raridade reforça "
        "evidência contra mesma origem ainda exige validação.",
    )

    HINGE_BASE_NEGATIVE = (
        "HINGE_BASE_NEGATIVE",
        "Hinge Base Negativo (sem raridade)",
        "max(s0 − base, 0). Espelho da HINGE_BASE: feature-companheira do "
        "modelo bilateral plenamente geral. Controle estrutural para "
        "isolar a contribuição da dobra bilateral em si da contribuição "
        "da modulação bilateral por raridade.",
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

        `HINGE_BASE` e `HINGE_BASE_NEGATIVE` ignoram o modifier; o
        validador relaxa a exigência de `modifier_score_resource_id`
        nesses casos.
        """
        return self not in (
            ScoreInteractionMode.HINGE_BASE,
            ScoreInteractionMode.HINGE_BASE_NEGATIVE,
        )

    @property
    def is_negative_side(self) -> bool:
        """True para os modos espelhados ao lado abaixo de `s_0`.

        Usado pelo executor para decidir entre `max(base − s_0, 0)`
        (positivo) e `max(s_0 − base, 0)` (negativo).
        """
        return self in (
            ScoreInteractionMode.ONE_SIDED_NEGATIVE,
            ScoreInteractionMode.HINGE_BASE_NEGATIVE,
        )

    @property
    def is_hinge_only(self) -> bool:
        """True para `HINGE_BASE` e `HINGE_BASE_NEGATIVE` (sem
        multiplicação pelo modifier)."""
        return self in (
            ScoreInteractionMode.HINGE_BASE,
            ScoreInteractionMode.HINGE_BASE_NEGATIVE,
        )

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
