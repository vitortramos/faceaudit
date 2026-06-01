# faceaudit/enums/logreg_gridsearch_cv_mode.py

"""Modo de partição dos folds no GridSearch interno do LogReg.

Distinto de `CalibratorCVSplitMode` (que governa o CV diagnóstico opt-in
com `k_folds > 0`). Este modo controla o splitter usado pelo
`GridSearchCV` que seleciona o hiperparâmetro `C` sob restrição de
monotonicidade — o CV "principal" do calibrador.
"""

from __future__ import annotations

from enum import Enum


class LogRegGridSearchCVMode(Enum):
    """Estratégia de partição de folds no GridSearch interno do LogReg.

    Os modos diferem no nível de disjunção entre treino e held-out dentro
    de cada fold:

    - `AUTO`: escolhe `GROUPED_PROBE` quando há `train_probe_identities`
      no metadata e cai em `PAIR` caso contrário. Default operacional.
    - `PAIR`: `StratifiedKFold` por par — sem agrupamento por identidade.
      Pares correlacionados podem cruzar folds (cobertura `none`).
    - `GROUPED_PROBE`: `StratifiedGroupKFold` por `probe_identity`. Probe
      disjunto entre folds; a identidade de referência dos pares Hd ainda
      pode aparecer em múltiplos folds (cobertura `probe_only`).
    - `STRICT_IDENTITY`: splitter customizado
      `IdentityDisjointPairKFold` que produz folds com identidades
      totalmente disjuntas; pares Hd boundary (uma identidade em cada
      lado) são excluídos só do fold corrente, não globalmente —
      preservando ≈ ((K−1)/K)² do conjunto Hd no treino de cada fold
      (cobertura `full`).
    """

    AUTO = (
        "AUTO",
        "Automático (recomendado)",
        "Escolhe GROUPED_PROBE quando o calibrador recebe identidades de "
        "probe via metadata; cai em PAIR quando não. Default operacional.",
    )
    PAIR = (
        "PAIR",
        "Estratificado por par",
        "sklearn.StratifiedKFold sem agrupamento. Pares da mesma "
        "identidade podem aparecer em ambos os lados (cobertura `none`). "
        "Otimista — útil apenas como fallback.",
    )
    GROUPED_PROBE = (
        "GROUPED_PROBE",
        "Agrupado por identidade do probe",
        "sklearn.StratifiedGroupKFold por probe_identity. Probe disjunto "
        "entre folds (cobertura `probe_only`). Default V2.8. Identidade "
        "de referência dos pares Hd pode ainda aparecer em múltiplos "
        "folds — leakage residual.",
    )
    STRICT_IDENTITY = (
        "STRICT_IDENTITY",
        "Estrito por identidade (folds dinâmicos)",
        "Splitter customizado IdentityDisjointPairKFold. Particiona "
        "identidades em K grupos e, para cada fold, define treino/teste "
        "com identidades totalmente disjuntas; pares Hd boundary (uma "
        "identidade em cada lado do split) são excluídos apenas no fold "
        "corrente, preservando ≈ ((K−1)/K)² do conjunto Hd no treino. "
        "Refit final em todo D_cal. Cobertura `full`. V2.10.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "LogRegGridSearchCVMode":
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
    def from_str(cls, value) -> "LogRegGridSearchCVMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.AUTO
        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        # backward-compat: "STRICT_BY_IDENTITY" (metadata V2.9) → STRICT_IDENTITY
        if val_norm == "STRICT_BY_IDENTITY":
            return cls.STRICT_IDENTITY
        try:
            return cls[val_norm]
        except KeyError:
            for m in cls:
                if m.value == val_norm:
                    return m
            return cls.AUTO
