# faceaudit/enums/logreg_grid_scoring_metric.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class LogRegGridScoringMetric(Enum):
    """
    Métrica de avaliação usada pelo GridSearch ao otimizar o C da
    Regressão Logística no calibrador.

    Padrão segue o de outros enums "ricos" do projeto (PLDATrainingStrategy,
    PLDAMode): cada membro carrega `value` (string compatível com sklearn),
    `label` (texto curto para UI) e `description` (texto longo para tooltip
    e linha contextual).
    """

    NEG_LOG_LOSS = (
        "neg_log_loss",
        "Negative Log-Loss (calibração)",
        "Penaliza miscalibração diretamente — ideal para calibradores. "
        "Modelos com probabilidades 'honestas' (próximas da frequência real) "
        "vencem. É a métrica recomendada para calibração forense.",
    )

    ROC_AUC = (
        "roc_auc",
        "ROC AUC (separabilidade)",
        "Mede a capacidade do modelo de ordenar genuínos acima de impostores "
        "(ranking). Não premia calibração: dois modelos com a mesma curva "
        "de ROC têm AUC idêntica mesmo se um for mal calibrado. Útil quando "
        "o foco é separação/classificação, não LR honesto.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "LogRegGridScoringMetric":
        # Pattern padrão do projeto: define _value_ em __new__ para que o
        # _value2member_map_ da metaclass seja populado com a string simples
        # — o que permite construtor por valor (ex: cls("neg_log_loss")) e
        # pickle round-trip funcionarem corretamente.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        # label_pt alias — `label` já está em PT-BR. Mantido por consistência
        # de API com outros enums do projeto.
        obj.label_pt = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "LogRegGridScoringMetric":
        if not value:
            return cls.NEG_LOG_LOSS
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[str(value).strip().upper()]
            except KeyError:
                return cls.NEG_LOG_LOSS

    # --- Helpers para UI ---

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(m.value, m.label) for m in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(m.value, m.label, m.description) for m in cls]
