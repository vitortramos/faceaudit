# faceaudit/enums/typicality_combiner.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np


class TypicalityCombiner(Enum):
    """
    Define como combinar as tipicidades de probe e reference em
    um único score simétrico por par.

    Simetria é forensicamente obrigatória: LR(A, B) deve igualar
    LR(B, A). Por isso passar `t_probe` e `t_reference` brutos como
    features independentes NÃO é uma opção — usaremos sempre um
    combinador simétrico.

    Cada membro carrega:
    - `value`: identificador canônico (persistido em JSON).
    - `label_pt`: rótulo amigável em PT-BR para a UI.
    - `description`: explicação curta do que o combinador captura.
    """

    MEAN = (
        "MEAN",
        "Média",
        "Tipicidade média do par. Sensível à raridade global das duas faces — "
        "se ambas são típicas, o par 'mora na região densa' do espaço, e o "
        "calibrador sabe que score cosseno alto ali é menos surpreendente.",
    )

    MIN = (
        "MIN",
        "Mínimo",
        "Menor das duas tipicidades. Captura a face MAIS atípica do par — "
        "útil pra detectar quando pelo menos uma das faces é fora do comum, "
        "o que reforça evidência de identidade quando há match.",
    )

    MAX = (
        "MAX",
        "Máximo",
        "Maior das duas tipicidades. Captura a face MAIS típica do par — "
        "complementar a MIN; ambas juntas (via vetorização) caracterizam "
        "o intervalo de tipicidade do par.",
    )

    PRODUCT = (
        "PRODUCT",
        "Produto",
        "Produto das duas tipicidades. Pares cujas duas faces são típicas "
        "produzem valor alto; pares com pelo menos uma face atípica produzem "
        "valor baixo. Cresce não-linearmente com a tipicidade conjunta.",
    )

    ABS_DIFF = (
        "ABS_DIFF",
        "Diferença Absoluta",
        "|t_probe − t_reference|. Captura assimetria de raridade — útil pra "
        "detectar pares onde uma face é muito mais comum que a outra (caso "
        "questionada vs padrão de referência demograficamente distintos).",
    )

    def __init__(self, value: str, label_pt: str, description: str):
        self._value_ = value
        self.label_pt = label_pt
        self.description = description

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "TypicalityCombiner":
        if not value:
            return cls.MEAN
        if isinstance(value, cls):
            return value
        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[val_norm]
        except KeyError:
            for item in cls:
                if item.value == val_norm:
                    return item
            return cls.MEAN

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]

    def combine(self, t_probe: float, t_reference: float) -> float:
        """
        Aplica o combinador a duas tipicidades escalares.

        O caller decide qual métrica passar — `mean_knn_similarity` (intuitiva,
        [-1, 1]), `typicality_percentile/100` ([0, 1]), ou `log_density_knn`
        (escala real, melhor pra LR generativo). MEAN/MIN/MAX/ABS_DIFF
        funcionam em qualquer escala; PRODUCT exige inputs não-negativos
        pra ser sensato (use percentile/100).
        """
        if self is TypicalityCombiner.MEAN:
            return 0.5 * (t_probe + t_reference)
        if self is TypicalityCombiner.MIN:
            return float(np.minimum(t_probe, t_reference))
        if self is TypicalityCombiner.MAX:
            return float(np.maximum(t_probe, t_reference))
        if self is TypicalityCombiner.PRODUCT:
            return float(t_probe * t_reference)
        if self is TypicalityCombiner.ABS_DIFF:
            return float(abs(t_probe - t_reference))
        raise ValueError(f"Combinador não suportado: {self}")
