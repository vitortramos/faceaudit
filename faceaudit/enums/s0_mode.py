# faceaudit/enums/s0_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class S0Mode(Enum):
    """
    Como `s0` (ponto de evidência neutra) é resolvido em
    `SCORE_INTERACTION_FEATURE`.

    `s0` é o valor do score base no qual o calibrador univariado
    cruza logLR=0 — a "fronteira" entre evidência a favor de Hp e a
    favor de Hd. Ele deve ser **aprendido no domínio de calibração**
    e **congelado** para validação/casework. Refit silencioso em
    validação ou casework recalibra a definição de evidência neutra
    e contamina o LR final.

    Modos:

    - `AUTO_LOGREG` (default): ajusta um LogReg univariado
      `base → label` no próprio dataset e calcula
      `s0 = −intercept/coef`. **Calibrador auxiliar**, não
      necessariamente igual ao calibrador final do projeto (que pode
      usar GridSearch). Exige `same_person ∈ {0, 1}` e duas classes;
      BLOQUEADO em casework. Emite WARNING de CV-leakage quando o CV
      diagnóstico do calibrador final roda sobre o mesmo dataset.

    - `MANUAL`: usa `s0_manual_value` diretamente. Útil para fixar s0
      vindo de literatura, calibrador externo já analisado ou regra
      de negócio explícita.

    - `INHERIT`: lê `interaction_s0` do metadata de um
      `SimilarityScoreDataset` herdado (gerado por uma execução
      anterior desta instrução). Uso obrigatório em validação e
      casework para preservar a definição de evidência neutra. O
      executor compara `base_score_type`/`base_similarity_model_*` do
      herdado vs atual e emite WARNING em mismatch.

    Evolução futura (não implementada na v1):
    `FROM_CALIBRATOR_RESOURCE` lerá `coef`/`intercept` de um
    `TRAINED_CALIBRATION_MODEL` univariado já existente — alinharia
    100% com o calibrador final do projeto.
    """

    AUTO_LOGREG = (
        "AUTO_LOGREG",
        "Auto (LogReg univariado)",
        "Ajusta LogReg univariado (calibrador auxiliar) e calcula "
        "s0 = −intercept/coef. Default para o domínio de calibração. "
        "BLOQUEADO em casework (sem ground truth).",
    )

    MANUAL = (
        "MANUAL",
        "Manual (valor fixo)",
        "Usa o valor de `s0_manual_value` informado pelo usuário. "
        "Útil para fixar s0 a partir de literatura ou calibrador "
        "externo já analisado.",
    )

    INHERIT = (
        "INHERIT",
        "Herdado de execução anterior",
        "Lê `interaction_s0` do metadata de um SimilarityScoreDataset "
        "já produzido por SCORE_INTERACTION_FEATURE. OBRIGATÓRIO em "
        "validação e casework para preservar a definição de "
        "evidência neutra aprendida na calibração.",
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
    def from_str(cls, value: str) -> "S0Mode":
        if not value:
            return cls.AUTO_LOGREG
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.AUTO_LOGREG

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
