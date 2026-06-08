# faceaudit/protocols/monotonicity_derivative_provider_protocol.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class MonotonicityRegionResult:
    """
    Resultado da auditoria de monotonicidade para uma região piecewise.

    Uma região é uma faixa de cosseno (`s > s_0`, `s < s_0`, ou segmento
    entre knots em piecewise univariado) sobre a qual a derivada
    `∂ log_LR / ∂ s` tem forma fixa (constante ou função linear das
    raridades). Cada provider retorna uma ou mais regiões; a regra agrega.

    Para arquiteturas sem dependência de raridade (COSINE_ONLY, HINGE_ONLY,
    TWO_SIDED_HINGE_ONLY), `slope_value` é o slope constante na região e
    os campos baseados em raridades ficam None.

    Para arquiteturas com dependência de raridade (HIERARCHICAL_CENTERED,
    BILATERAL_FULL_CENTERED, etc.), `slope_at_observed_rows` traz o slope
    avaliado em cada par observado em D_cal dentro da região, e
    `slope_at_central_box_corners` traz o slope nos cantos do retângulo
    P5-P95 das raridades observadas.
    """

    # ── Identificação ──
    region_id: str
    region_description: str

    # ── Suporte ──
    n_observed_rows_in_region: int

    # ── Caso simples (derivada constante na região) ──
    slope_value: Optional[float] = None

    # ── Caso multivariado (derivada depende de raridades) ──
    slope_at_observed_rows: Optional[np.ndarray] = None
    slope_at_central_box_corners: Optional[List[float]] = None
    corner_labels: Optional[List[str]] = None

    # Identificação do pior observed_row e pior corner
    worst_observed_row_index: Optional[int] = None
    worst_observed_row_features: Optional[dict] = None
    worst_central_corner_label: Optional[str] = None
    worst_central_corner_values: Optional[dict] = None

    # ── Diagnósticos ──
    feature_variance_in_region: dict = field(default_factory=dict)
    zero_variance_terms: List[str] = field(default_factory=list)
    near_zero_variance_terms: List[str] = field(default_factory=list)

    def min_slope_observed(self) -> Optional[float]:
        """Menor slope sobre todos os pares observados na região."""
        if self.slope_value is not None:
            return self.slope_value
        if self.slope_at_observed_rows is not None and self.slope_at_observed_rows.size:
            return float(np.min(self.slope_at_observed_rows))
        return None

    def min_slope_central_box(self) -> Optional[float]:
        """Menor slope sobre os cantos da caixa central."""
        if self.slope_value is not None:
            return self.slope_value
        if self.slope_at_central_box_corners:
            return float(min(self.slope_at_central_box_corners))
        return None

    def max_slope_observed(self) -> Optional[float]:
        if self.slope_value is not None:
            return self.slope_value
        if self.slope_at_observed_rows is not None and self.slope_at_observed_rows.size:
            return float(np.max(self.slope_at_observed_rows))
        return None


class MonotonicityDerivativeProviderProtocol(Protocol):
    """
    Provê as derivadas `∂ log_LR / ∂ s` por região para auditoria de
    monotonicidade do calibrador.

    Cada implementação cobre uma família matemática de arquiteturas:
        - LRArchitectureDerivativeProvider: arquiteturas declarativas de
          LR_ARCHITECTURE_VECTORIZATION (V1).
        - PiecewiseLinearDerivativeProvider: piecewise univariado com K
          knots arbitrários (V1.5).

    A regra `ConditionalMonotonicityRule` seleciona o provider apropriado
    via dispatch sobre o tipo da arquitetura do calibrador.
    """

    def evaluate_regions(
        self,
        *,
        calibrator_metadata: dict,
        raw_input_payload: Any,
        box_quantile_low: float = 0.05,
        box_quantile_high: float = 0.95,
    ) -> List[MonotonicityRegionResult]:
        """
        Retorna lista de resultados por região piecewise.

        Args:
            calibrator_metadata: dict com `architecture_state`,
                coeficientes (β_z + scaler) e fingerprints.
            raw_input_payload: LRArchitectureRawInputPayload com matriz
                `(N, K)` de features brutas e schema.
            box_quantile_low, box_quantile_high: limites do
                `central_box` em quantis das raridades observadas.

        Returns:
            Lista de MonotonicityRegionResult, uma por região piecewise.
        """
        ...
