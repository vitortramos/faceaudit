# faceaudit/models/calibration/logreg_s0_config.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="LogRegS0Config")


@dataclass
class LogRegS0Config(JSONSerializableMixin):
    """
    Configuração do **calibrador auxiliar univariado** usado em
    `SCORE_INTERACTION_FEATURE` no modo `AUTO_LOGREG` para estimar
    `s0 = −intercept/coef`.

    É um calibrador AUXILIAR — não necessariamente igual ao calibrador
    final do projeto (que pode usar GridSearch de C). Default é
    `class_weight=balanced, solver=liblinear, C=1.0, max_iter=1000` —
    razoável, robusto e rápido para a maioria dos datasets de
    calibração. Persistido inteiro no metadata do dataset produzido
    (`interaction_s0_C`, `interaction_s0_training_method`).

    Para alinhar 100% com o calibrador final, considere a evolução
    futura `S0Mode.FROM_CALIBRATOR_RESOURCE` (não implementada na v1).
    """

    C: float = field(
        default=1.0,
        metadata={"help": "Parâmetro de regularização inversa do LogReg auxiliar."},
    )

    solver: str = field(
        default="liblinear",
        metadata={"help": "Solver do sklearn (`liblinear`, `lbfgs`, ...)."},
    )

    max_iter: int = field(
        default=1000,
        metadata={"help": "Número máximo de iterações."},
    )

    class_weight: Optional[str] = field(
        default="balanced",
        metadata={"help": (
            "Pesos de classe — 'balanced' (default) ou None. "
            "'balanced' compensa desbalanceamento gen/imp; vale a "
            "pena manter como default para datasets onde impostores "
            "dominam."
        )},
    )

    def __post_init__(self) -> None:
        if self.C <= 0:
            raise ValueError("LogRegS0Config.C deve ser > 0.")
        if self.max_iter < 1:
            raise ValueError("LogRegS0Config.max_iter deve ser >= 1.")
        if self.class_weight is not None and self.class_weight not in {"balanced"}:
            # Aceita None ou "balanced" — sklearn aceita dict mas a UI atual
            # não tem como editar dict; mantém a API restrita pra evitar
            # configs inválidas via JSON livre.
            raise ValueError(
                "LogRegS0Config.class_weight deve ser 'balanced' ou None."
            )

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        return cls(
            C=float(data.get("C", 1.0)),
            solver=str(data.get("solver", "liblinear")),
            max_iter=int(data.get("max_iter", 1000)),
            class_weight=(
                None if data.get("class_weight") is None
                else str(data.get("class_weight", "balanced"))
            ),
        )
