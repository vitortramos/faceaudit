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
from faceaudit.enums.knot_placement import KnotPlacement
from faceaudit.enums.knot_scale import KnotScale
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
            "Offsets das dobras em relação a `hinge_anchor_s0`, expressos "
            "na unidade definida por `knot_scale`. Uma única hinge no "
            "ponto neutro: [0.0] (default). Três dobras: [-δ, 0, +δ]. "
            "Em AUX_LOG_LR, [-ln(10), 0, +ln(10)] ≈ [-2.30, 0, +2.30] "
            "= dobras em LR auxiliar 10:1 contra Hp, neutro, e 10:1 a "
            "favor de Hp. Em RAW, são offsets no domínio do score."
        )},
    )

    knot_scale: KnotScale = field(
        default=KnotScale.AUX_LOG_LR,
        metadata={"help": (
            "Escala em que `knot_offsets` é expresso. AUX_LOG_LR (default) "
            "usa unidades de log-LR do calibrador auxiliar (nats); "
            "converte raw_offset = aux_log_lr / aux_coef no fit, "
            "fold-aware. RAW usa diretamente o domínio do score "
            "(backward-compat com calibradores antigos). AUX_LOG_LR "
            "exige hinge_anchor_mode=AUTO_LOGREG."
        )},
    )

    knot_placement: KnotPlacement = field(
        default=KnotPlacement.MANUAL,
        metadata={"help": (
            "Como os knots são posicionados. MANUAL (default): usuário "
            "escreve knot_offsets. Configuração operacional recomendada: "
            "knot_offsets=[0.0] em knot_scale=AUX_LOG_LR (uma única dobra "
            "no ponto neutro do auxiliar). AUTO_FROM_MIRRORED_MEDIANS: "
            "transformer popula knot_offsets no fit com 2 knots espelhados "
            "em torno de zero — heurística experimental, opt-in. Em AUTO, "
            "knot_offsets/knot_scale do DTO são ignorados."
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
        if not isinstance(self.knot_scale, KnotScale):
            self.knot_scale = KnotScale.from_str(self.knot_scale)
        if not isinstance(self.knot_placement, KnotPlacement):
            self.knot_placement = KnotPlacement.from_str(self.knot_placement)

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
            # Em AUTO_FROM_MEDIANS, knot_offsets é preenchido pelo
            # transformer no fit — pode estar vazio no DTO.
            if (
                self.knot_placement is KnotPlacement.MANUAL
                and not self.knot_offsets
            ):
                raise ValueError(
                    "mode=PIECEWISE_LINEAR com knot_placement=MANUAL "
                    "requer knot_offsets não vazio. Use [0.0] para uma "
                    "hinge única no ponto neutro, ou troque para "
                    "knot_placement=AUTO_FROM_MEDIANS."
                )
            if (
                self.hinge_anchor_mode is HingeAnchorMode.MANUAL
                and self.hinge_anchor_manual_value is None
            ):
                raise ValueError(
                    "hinge_anchor_mode=MANUAL requer "
                    "hinge_anchor_manual_value não nulo."
                )
            # AUTO_FROM_MEDIANS depende do aux_coef do LogReg auxiliar
            # pra mapear medianas raw → log-LR. Indisponível em MANUAL.
            if (
                self.knot_placement is KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS
                and self.hinge_anchor_mode is HingeAnchorMode.MANUAL
            ):
                raise ValueError(
                    "knot_placement=AUTO_FROM_MEDIANS exige "
                    "hinge_anchor_mode=AUTO_LOGREG (precisa do aux_coef "
                    "do LogReg auxiliar pra calcular medianas log-LR)."
                )
            if (
                self.hinge_anchor_mode is HingeAnchorMode.MANUAL
                and self.hinge_anchor_manual_value is not None
                and not math.isfinite(float(self.hinge_anchor_manual_value))
            ):
                raise ValueError(
                    "hinge_anchor_manual_value deve ser número finito."
                )
            # Conversão aux_log_lr → raw depende de `aux_coef` do LogReg
            # auxiliar, indisponível em MANUAL. Em MANUAL, o usuário
            # precisa expressar os offsets diretamente no domínio do
            # score (RAW).
            if (
                self.knot_scale is KnotScale.AUX_LOG_LR
                and self.hinge_anchor_mode is HingeAnchorMode.MANUAL
            ):
                raise ValueError(
                    "knot_scale=AUX_LOG_LR exige hinge_anchor_mode=AUTO_LOGREG "
                    "(conversão aux_log_lr → score bruto usa o coef do LogReg "
                    "auxiliar). Em MANUAL, use knot_scale=RAW e expresse "
                    "os offsets diretamente no domínio do score."
                )

        if self.min_allowed_slope < 0:
            raise ValueError("min_allowed_slope deve ser >= 0.")

    # ─────────────────────── hooks de UI ───────────────────────

    # Fields cuja mudança dispara rebuild do form (ver NestedDTOInlineWidget):
    # - mode: muda a visibilidade de todos os outros fields (NONE esconde tudo).
    # - hinge_anchor_mode: condiciona visibilidade de hinge_anchor_manual_value
    #   e a aceitação de knot_scale=LOG_LR.
    # - knot_scale: muda o significado numérico de knot_offsets (label dinâmico).
    REACTIVE_FIELDS = (
        "mode",
        "hinge_anchor_mode",
        "knot_scale",
        "knot_placement",
    )

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
                "knot_scale",
                "knot_placement",
                "require_monotonicity",
                "monotonicity_policy",
                "min_allowed_slope",
            })
        else:
            if self.hinge_anchor_mode is not HingeAnchorMode.MANUAL:
                hidden.add("hinge_anchor_manual_value")
            # Em MANUAL anchor, LOG_LR scale é proibido E AUTO_FROM_MEDIANS
            # placement é proibido (ambos exigem aux_coef do auxiliar).
            # Esconde knot_scale e knot_placement nesse caso.
            if self.hinge_anchor_mode is HingeAnchorMode.MANUAL:
                hidden.add("knot_scale")
                hidden.add("knot_placement")
            # Em AUTO_FROM_MEDIANS, o transformer popula knot_offsets/
            # knot_scale automaticamente — esconde os dois pra evitar
            # confusão visual ("estou preenchendo isso e nada acontece").
            if self.knot_placement is KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS:
                hidden.add("knot_offsets")
                hidden.add("knot_scale")
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
        """Posição absoluta dos knots dado o `hinge_anchor_s0` aprendido.

        ATENÇÃO: usa `knot_offsets` em sua escala literal — só faz sentido
        quando `knot_scale=RAW`. Quando `knot_scale=LOG_LR`, a posição
        absoluta só é definida após o `fit` (depende de `aux_coef`); use
        `PiecewiseLinearScoreTransformer.knots_absolute_` em vez.
        """
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
            "knot_scale": self.knot_scale.to_str(),
            "knot_placement": self.knot_placement.to_str(),
            "require_monotonicity": bool(self.require_monotonicity),
            "monotonicity_policy": self.monotonicity_policy.to_str(),
            "min_allowed_slope": float(self.min_allowed_slope),
        }

    @classmethod
    def from_dict(cls: Type[T], data: Optional[dict]) -> T:
        if not data:
            return cls()
        # Backward-compat: JSONs antigos sem `knot_scale` assumem RAW
        # (semântica pré-existente). Novos defaults via construtor sem
        # args usam LOG_LR — só impacta usuários que criem do zero.
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
            knot_scale=KnotScale.from_str(data.get("knot_scale", "RAW")),
            knot_placement=KnotPlacement.from_str(
                data.get("knot_placement", "MANUAL")
            ),
            require_monotonicity=bool(data.get("require_monotonicity", True)),
            monotonicity_policy=MonotonicityPolicy.from_str(
                data.get("monotonicity_policy", "ERROR")
            ),
            min_allowed_slope=float(data.get("min_allowed_slope", 1e-8)),
        )
