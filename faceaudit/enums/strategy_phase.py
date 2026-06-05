# faceaudit/enums/strategy_phase.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class StrategyPhase(Enum):
    """
    Fase atual de execução de uma estratégia FaceAudit.

    O `LRStrategyDefinition` carrega 3 listas de instruções, uma por fase.
    Quando o `StrategyExecutor` itera sobre as instruções, ele cria um
    `InstructionExecutionFrame` contendo a fase atual e o passa ao
    executor via `context.with_frame(frame)`.

    A fase é usada pela nova instrução `LR_ARCHITECTURE_VECTORIZATION` para
    derivar comportamento — bloqueando ajuste automático de s₀/r₀ em
    VALIDATION e CASEWORK. Executores legados ignoram o frame e mantêm o
    comportamento histórico.
    """

    PREPARATION = (
        "PREPARATION",
        "Preparação",
        "Treinamento de modelos, pipelines e calibradores sobre D_cal.",
    )

    VALIDATION = (
        "VALIDATION",
        "Validação",
        "Avaliação supervisionada sobre dataset holdout. Sem ajuste de "
        "parâmetros — apenas aplicação de estado herdado da preparação.",
    )

    CASEWORK = (
        "CASEWORK",
        "Análise de caso",
        "Aplicação operacional ao caso questionado. Sem ground truth; "
        "obrigatoriamente usa estado herdado da preparação.",
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

    @classmethod
    def from_str(cls, value: str) -> "StrategyPhase":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.PREPARATION
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"StrategyPhase desconhecida: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @property
    def allows_parameter_fitting(self) -> bool:
        """True apenas em PREPARATION — VALIDATION/CASEWORK só aplicam state herdado."""
        return self == StrategyPhase.PREPARATION
