# faceaudit/models/calibration/piecewise_linear_calibration_config.py

"""Configuração da expansão piecewise-linear interna do calibrador.

A expansão é aplicada **dentro** do calibrador como passo de feature
engineering antes da regressão logística. Não tem instrução nem
recurso próprio — o usuário ativa via `mode = PIECEWISE_LINEAR` no
config do calibrador.

Estrutura preparada para múltiplas dobras (`knot_offsets` list), embora
a UI da V1 exponha apenas o caso de uma dobra no ponto neutro
(`knot_offsets = [0.0]`).

Princípios de design:

- `mode` em vez de `enabled: bool` — permite estender pra `SPLINE_MONOTONIC`,
  `GAUSSIAN_KERNELS` etc. sem mudar semântica do pickle.
- `hinge_anchor_s0` sempre aprendido fold-aware (sem leakage em CV).
- Monotonicidade validada na escala bruta após o fit final — não negociável.
- `knot_offsets` ordenado, deduplicado, finito — validado no `__post_init__`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar
import math

from faceaudit.enums.calibration_basis_mode import CalibrationBasisMode
from faceaudit.enums.hinge_anchor_mode import HingeAnchorMode
from faceaudit.enums.monotonicity_policy import MonotonicityPolicy
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="PiecewiseLinearCalibrationConfig")


@dataclass
class PiecewiseLinearCalibrationConfig(JSONSerializableMixin):
    """Configuração da expansão piecewise-linear interna do calibrador.

    Quando `mode = PIECEWISE_LINEAR`, o calibrador (LogReg) é alimentado
    com features expandidas via `PiecewiseLinearScoreTransformer`:

        z = score − hinge_anchor_s0
        features = [z, max(z − offset_j, 0)  for offset_j in knot_offsets]

    O `hinge_anchor_s0` é aprendido fold-aware (via LogReg auxiliar
    univariado em AUTO_LOGREG ou informado em MANUAL), garantindo que
    GridSearchCV de `C` não vaze dados de holdout.

    Pickle do calibrador final armazena `hinge_anchor_s0`, `knot_offsets`,
    `knots_absolute`, `segment_slopes_raw`, `is_monotonic` e
    `effective_log_lr_zero_crossing` para auditoria forense.

    Restrição V1: aplicável apenas a score 1D (cosseno, PLDA escalar).
    Calibrador multivariado (`SCORE_VECTORIZATION` com D > 1) **rejeita**
    `mode = PIECEWISE_LINEAR` — usar `SCORE_INTERACTION_FEATURE` externa.
    """

    mode: CalibrationBasisMode = field(
        default=CalibrationBasisMode.NONE,
        metadata={"help": (
            "Modo de expansão das basis functions. NONE = nenhuma "
            "expansão (calibrador vê score direto, backward-compat). "
            "PIECEWISE_LINEAR = aplica expansão hinge interna."
        )},
    )

    hinge_anchor_mode: HingeAnchorMode = field(
        default=HingeAnchorMode.AUTO_LOGREG,
        metadata={"help": (
            "Como `hinge_anchor_s0` é resolvido: AUTO_LOGREG estima via "
            "LogReg auxiliar (fold-aware no CV); MANUAL usa "
            "`hinge_anchor_manual_value`."
        )},
    )

    hinge_anchor_manual_value: Optional[float] = field(
        default=None,
        metadata={"help": (
            "Valor manual de `hinge_anchor_s0` (só usado em "
            "`hinge_anchor_mode = MANUAL`)."
        )},
    )

    knot_offsets: List[float] = field(
        default_factory=lambda: [0.0],
        metadata={"help": (
            "Offsets das dobras em relação a `hinge_anchor_s0`. "
            "Uma única hinge no ponto neutro: [0.0] (default). Três "
            "dobras: [-δ, 0, +δ]. UI V1 expõe só [0.0]; múltiplas "
            "dobras disponíveis programaticamente."
        )},
    )

    require_monotonicity: bool = field(
        default=True,
        metadata={"help": (
            "Quando True, valida que todas as inclinações cumulativas "
            "na escala bruta são > min_allowed_slope após o fit final. "
            "Falha (ou avisa) conforme `monotonicity_policy`."
        )},
    )

    monotonicity_policy: MonotonicityPolicy = field(
        default=MonotonicityPolicy.ERROR,
        metadata={"help": (
            "Política quando a curva treinada não é monotônica: ERROR "
            "rejeita (default operacional, defensável forensicamente); "
            "WARNING permite com alerta forte (apenas para diagnóstico)."
        )},
    )

    min_allowed_slope: float = field(
        default=1e-8,
        metadata={"help": (
            "Inclinação mínima aceita por trecho na escala bruta. "
            "Valores ≤ este threshold contam como 'não monotônico'. "
            "Default 1e-8 — captura inclinações efetivamente nulas sem "
            "ser sensível a ruído numérico."
        )},
    )

    def __post_init__(self) -> None:
        # Coerções defensivas pra strings vindas de JSON serializado.
        if not isinstance(self.mode, CalibrationBasisMode):
            self.mode = CalibrationBasisMode.from_str(self.mode)
        if not isinstance(self.hinge_anchor_mode, HingeAnchorMode):
            self.hinge_anchor_mode = HingeAnchorMode.from_str(self.hinge_anchor_mode)
        if not isinstance(self.monotonicity_policy, MonotonicityPolicy):
            self.monotonicity_policy = MonotonicityPolicy.from_str(
                self.monotonicity_policy
            )

        # Normalização e validação de knot_offsets.
        # - ordenar
        # - rejeitar duplicatas (mesmo offset numa mesma posição é uma
        #   feature colinear inútil; provavelmente erro de config)
        # - exigir valores finitos
        if not isinstance(self.knot_offsets, list):
            self.knot_offsets = list(self.knot_offsets)
        offsets = [float(x) for x in self.knot_offsets]
        for x in offsets:
            if not math.isfinite(x):
                raise ValueError(
                    f"knot_offsets contém valor não finito: {x!r}"
                )
        dedup = sorted(set(offsets))
        if len(dedup) != len(offsets):
            raise ValueError(
                f"knot_offsets contém duplicatas: {offsets!r} → "
                f"deduplicado vira {dedup!r}. Cada knot deve ser único."
            )
        self.knot_offsets = dedup

        # Validações cross-field.
        if self.mode is CalibrationBasisMode.PIECEWISE_LINEAR:
            if not self.knot_offsets:
                raise ValueError(
                    "mode=PIECEWISE_LINEAR requer knot_offsets não vazio. "
                    "Use [0.0] para uma hinge única no ponto neutro."
                )
            if (
                self.hinge_anchor_mode is HingeAnchorMode.MANUAL
                and self.hinge_anchor_manual_value is None
            ):
                raise ValueError(
                    "hinge_anchor_mode=MANUAL requer "
                    "hinge_anchor_manual_value não nulo."
                )
            if (
                self.hinge_anchor_mode is HingeAnchorMode.MANUAL
                and self.hinge_anchor_manual_value is not None
                and not math.isfinite(float(self.hinge_anchor_manual_value))
            ):
                raise ValueError(
                    "hinge_anchor_manual_value deve ser número finito."
                )

        if self.min_allowed_slope < 0:
            raise ValueError("min_allowed_slope deve ser >= 0.")

    # ─────────────────────── hooks de UI ───────────────────────

    # Fields cuja mudança dispara rebuild do form (ver NestedDTOInlineWidget):
    # - mode: muda a visibilidade de todos os outros fields (NONE esconde tudo).
    # - hinge_anchor_mode: condiciona a visibilidade de hinge_anchor_manual_value.
    REACTIVE_FIELDS = ("mode", "hinge_anchor_mode")

    def get_hidden_fields(self) -> set:
        """Visibilidade dinâmica de fields no formulário inline.

        - Quando `mode = NONE`: esconde tudo exceto o próprio combo `mode`.
          O usuário ativa o piecewise antes de configurá-lo.
        - Quando `hinge_anchor_mode != MANUAL`: esconde `hinge_anchor_manual_value`
          (sem efeito quando não está em modo manual).
        - `knot_offsets` permanece visível para usuários avançados.
        - `monotonicity_policy` e `min_allowed_slope` são opções de segurança
          sempre relevantes quando o piecewise está ativo.
        """
        hidden: set = set()
        if self.mode is not CalibrationBasisMode.PIECEWISE_LINEAR:
            hidden.update({
                "hinge_anchor_mode",
                "hinge_anchor_manual_value",
                "knot_offsets",
                "require_monotonicity",
                "monotonicity_policy",
                "min_allowed_slope",
            })
        else:
            if self.hinge_anchor_mode is not HingeAnchorMode.MANUAL:
                hidden.add("hinge_anchor_manual_value")
        return hidden

    # ─────────────────────── conveniências ───────────────────────

    @property
    def is_enabled(self) -> bool:
        """True quando a expansão piecewise-linear deve ser aplicada."""
        return self.mode is CalibrationBasisMode.PIECEWISE_LINEAR

    @property
    def n_knots(self) -> int:
        return len(self.knot_offsets)

    def knots_absolute(self, hinge_anchor_s0: float) -> List[float]:
        """Posição absoluta dos knots dado o `hinge_anchor_s0` aprendido."""
        return [float(hinge_anchor_s0) + offset for offset in self.knot_offsets]

    # ─────────────────────── (de)serialização ───────────────────────

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.to_str(),
            "hinge_anchor_mode": self.hinge_anchor_mode.to_str(),
            "hinge_anchor_manual_value": (
                None if self.hinge_anchor_manual_value is None
                else float(self.hinge_anchor_manual_value)
            ),
            "knot_offsets": list(self.knot_offsets),
            "require_monotonicity": bool(self.require_monotonicity),
            "monotonicity_policy": self.monotonicity_policy.to_str(),
            "min_allowed_slope": float(self.min_allowed_slope),
        }

    @classmethod
    def from_dict(cls: Type[T], data: Optional[dict]) -> T:
        if not data:
            return cls()
        return cls(
            mode=CalibrationBasisMode.from_str(
                data.get("mode", "NONE")
            ),
            hinge_anchor_mode=HingeAnchorMode.from_str(
                data.get("hinge_anchor_mode", "AUTO_LOGREG")
            ),
            hinge_anchor_manual_value=(
                None if data.get("hinge_anchor_manual_value") is None
                else float(data["hinge_anchor_manual_value"])
            ),
            knot_offsets=list(data.get("knot_offsets", [0.0])),
            require_monotonicity=bool(data.get("require_monotonicity", True)),
            monotonicity_policy=MonotonicityPolicy.from_str(
                data.get("monotonicity_policy", "ERROR")
            ),
            min_allowed_slope=float(data.get("min_allowed_slope", 1e-8)),
        )
