# faceaudit/enums/architecture_parameter_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class ArchitectureParameterMode(Enum):
    """
    Modo de operação atual da instrução `LR_ARCHITECTURE_VECTORIZATION`
    quanto a s₀ e r₀.

    Distinto de `ArchitectureFitScope`: este enum descreve o que a
    INSTRUÇÃO está fazendo agora (DTO-level); `ArchitectureFitScope`
    descreve como o ESTADO foi originalmente ajustado.

    Modos:

    - `FIT_FULL_CALIBRATION_LEGACY` (default v1): ajusta s₀ via AUTO_LOGREG
      sobre todo D_cal antes da CV interna do calibrador. Mantém compatibilidade
      com o pipeline legado e com as estratégias históricas. Sujeito a
      leakage residual na seleção de C (vide §10.5 do relatório). Marcado
      como transitório — será substituído por FIT_PER_FOLD no v2.

    - `FIT_PER_FOLD` (v2, indisponível no v1): ajusta s₀ e r₀ dentro de cada
      train fold via `ArchitectureVectorizerTransformer` no Pipeline sklearn.
      Elimina o leakage residual. Tentativa de uso no v1 → erro explícito.

    - `MANUAL`: usa valores fornecidos pelo usuário (`manual_s0`, `manual_r0_*`).
      A obrigatoriedade dos campos depende da arquitetura.

    - `INHERIT`: obrigatório em VALIDATION e CASEWORK. Lê o `LRArchitectureState`
      completo de um recurso vetorizado já produzido (preparação). Sem
      recálculo de s₀ ou r₀.
    """

    FIT_FULL_CALIBRATION_LEGACY = (
        "FIT_FULL_CALIBRATION_LEGACY",
        "Ajuste em todo D_cal (legado)",
        "Ajusta s₀ via AUTO_LOGREG sobre todo D_cal antes da CV interna. "
        "Comportamento padrão da v1 — preserva equivalência com estratégias "
        "históricas. Marcado como transitório: será substituído por "
        "FIT_PER_FOLD na v2 que elimina leakage residual.",
    )

    FIT_PER_FOLD = (
        "FIT_PER_FOLD",
        "Ajuste por fold (v2)",
        "Ajusta s₀ e r₀ dentro de cada train fold via sklearn.Pipeline. "
        "Elimina leakage residual. Indisponível no v1 — uso resulta em erro.",
    )

    MANUAL = (
        "MANUAL",
        "Valores manuais",
        "Usa valores fornecidos pelo usuário em manual_s0 e manual_r0_*. "
        "A obrigatoriedade dos campos depende da arquitetura escolhida.",
    )

    INHERIT = (
        "INHERIT",
        "Herdado de execução anterior",
        "Lê o LRArchitectureState completo de um recurso vetorizado já "
        "produzido. OBRIGATÓRIO em VALIDATION e CASEWORK para preservar a "
        "definição de evidência neutra aprendida na calibração.",
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
    def from_str(cls, value: str) -> "ArchitectureParameterMode":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.FIT_FULL_CALIBRATION_LEGACY
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"ArchitectureParameterMode desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @property
    def requires_inherit_resource(self) -> bool:
        return self == ArchitectureParameterMode.INHERIT

    @property
    def is_fit_mode(self) -> bool:
        return self in (
            ArchitectureParameterMode.FIT_FULL_CALIBRATION_LEGACY,
            ArchitectureParameterMode.FIT_PER_FOLD,
        )
