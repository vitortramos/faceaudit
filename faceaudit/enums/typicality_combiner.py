# faceaudit/enums/typicality_combiner.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple

import numpy as np


class TypicalityCombiner(Enum):
    """
    Define como combinar as tipicidades de probe e reference em
    um único score por par.

    A maior parte dos combinadores é **simétrica** (LR(A,B) = LR(B,A))
    — defensável quando a hipótese forense não privilegia uma das
    faces. Combinadores **unilaterais** (`REFERENCE_ONLY`,
    `PROBE_ONLY`) quebram a simetria propositalmente e existem para
    alimentar a feature de interação `(base − s0)·rarity_ref` em
    `SCORE_INTERACTION_FEATURE`, onde a raridade modula a evidência
    a partir de um papel específico do par (tipicamente o documento
    de referência conhecido). Em self-comparação os papéis
    probe/reference são convencionais — modos unilaterais emitem
    WARNING nesse caso.

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

    REFERENCE_ONLY = (
        "REFERENCE_ONLY",
        "Apenas Referência (assimétrico)",
        "Usa só a tipicidade da face de referência (t_reference) — "
        "quebra simetria propositalmente. Destinado a alimentar a "
        "feature de interação `(base − s0)·rarity_ref` em "
        "SCORE_INTERACTION_FEATURE quando a hipótese forense atribui "
        "raridade ao papel REFERENCE (ex.: documento conhecido). "
        "Em self-comparação os papéis probe/reference são "
        "convencionais — emite WARNING.",
    )

    PROBE_ONLY = (
        "PROBE_ONLY",
        "Apenas Probe (assimétrico)",
        "Usa só a tipicidade da face do probe (t_probe) — espelho "
        "assimétrico de REFERENCE_ONLY. Útil em A/B test e em casos "
        "onde a referência é fixa e o que varia é a face questionada. "
        "Em self-comparação os papéis são convencionais — emite "
        "WARNING.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        # Pattern canônico de Enum com valor composto — ver
        # typicality_reference_mode.py pra explicação completa.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

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

    @property
    def is_symmetric(self) -> bool:
        """True para combinadores simétricos (LR(A,B)=LR(B,A)).

        REFERENCE_ONLY e PROBE_ONLY são unilaterais por construção.
        """
        return self not in (
            TypicalityCombiner.REFERENCE_ONLY,
            TypicalityCombiner.PROBE_ONLY,
        )

    def combine(self, t_probe: float, t_reference: float) -> float:
        """
        Aplica o combinador a duas tipicidades escalares.

        O caller decide qual métrica passar — `mean_knn_similarity` (intuitiva,
        [-1, 1]), `typicality_percentile/100` ([0, 1]), ou `log_density_knn`
        (escala real, melhor pra LR generativo). MEAN/MIN/MAX/ABS_DIFF
        funcionam em qualquer escala; PRODUCT exige inputs não-negativos
        pra ser sensato (use percentile/100).

        Modos unilaterais (`REFERENCE_ONLY`, `PROBE_ONLY`) NÃO usam
        `combine()` — o executor de TYPICALITY_PAIR_SCORING lê uma
        das duas tipicidades diretamente. Chamar `combine()` neles
        levanta `ValueError`.
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
        if self in (
            TypicalityCombiner.REFERENCE_ONLY,
            TypicalityCombiner.PROBE_ONLY,
        ):
            raise ValueError(
                f"{self.value}: modo unilateral não usa combine(); o "
                "executor lê a tipicidade do papel diretamente."
            )
        raise ValueError(f"Combinador não suportado: {self}")
