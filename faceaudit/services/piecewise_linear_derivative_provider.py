# faceaudit/services/piecewise_linear_derivative_provider.py

"""V1.5 — `MonotonicityDerivativeProviderProtocol` para calibradores
piecewise legado (`LogisticRegressionCalibrationModel` com
`piecewise_linear_config.mode = PIECEWISE_LINEAR`).

Estrutura piecewise univariada:
    z = score − s_0
    features = [z, max(z − offset_1, 0), ..., max(z − offset_D, 0)]
    log_LR(s) = β_raw[0]·z + Σ_{j: s > k_j} β_raw[j+1]·(z − offset_j)

onde `k_j = s_0 + offset_j` são os knots absolutos. A derivada
∂ log_LR / ∂ s é constante por região:
    segmento 0 (s ≤ k_1):           slope = β_raw[0]
    segmento 1 (k_1 < s ≤ k_2):     slope = β_raw[0] + β_raw[1]
    ...
    segmento D (s > k_D):           slope = β_raw[0] + Σ β_raw[j]

O calibrador já persiste `segment_slopes_raw` e `piecewise_knots_absolute`
no metadata após o `_assert_monotonic_after_fit`. Este provider apenas
embrulha esses valores em `MonotonicityRegionResult` por segmento e
adiciona a verificação contra `reportable_box["s"]` (V2).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from faceaudit.protocols.monotonicity_derivative_provider_protocol import (
    MonotonicityRegionResult,
)


class PiecewiseLinearDerivativeProvider:
    """Provider de derivadas ∂log_LR/∂s para calibradores piecewise-linear
    univariados (V1.5 da regra CONDITIONAL_MONOTONICITY).

    Lê os campos persistidos pelo `LogisticRegressionCalibrationModel`:
        - `piecewise_knots_absolute`: List[float] em escala bruta de score
        - `segment_slopes_raw`: List[float] de tamanho len(knots)+1
        - `hinge_anchor_s0`: float (para identificação das regiões)

    Reportable_box (V2): aceita chave `"s"` (ou `"base_score"`) mapeada
    para (lo, hi) em escala bruta de score. Cantos são apenas (lo) e (hi)
    — domínio 1D. Slope avaliado em cada um corresponde ao slope do
    segmento que contém o ponto.
    """

    PROVIDER_VERSION = "1.0"

    def evaluate_regions(
        self,
        *,
        calibrator_metadata: dict,
        raw_input_payload: Any = None,
        box_quantile_low: float = 0.05,
        box_quantile_high: float = 0.95,
        reportable_box: Optional[dict] = None,
    ) -> List[MonotonicityRegionResult]:
        # 1. Extrair knots e slopes do metadata
        knots = calibrator_metadata.get("piecewise_knots_absolute")
        slopes = calibrator_metadata.get("segment_slopes_raw")
        if knots is None or slopes is None:
            raise ValueError(
                "calibrator_metadata sem 'piecewise_knots_absolute' ou "
                "'segment_slopes_raw' — auditoria piecewise requer "
                "calibrador V3+ com piecewise_enabled=True."
            )
        knots = [float(k) for k in knots]
        slopes = [float(s) for s in slopes]
        if len(slopes) != len(knots) + 1:
            raise ValueError(
                f"Inconsistência piecewise: {len(knots)} knots mas "
                f"{len(slopes)} slopes (esperado {len(knots) + 1})."
            )

        # 2. Scores observados (opcional — fonte de "observed_rows")
        observed_scores = self._extract_observed_scores(raw_input_payload)

        # 3. Box central a partir dos observed_scores (P5–P95)
        central_box_s: Optional[Tuple[float, float]] = None
        if observed_scores is not None and observed_scores.size:
            central_box_s = (
                float(np.quantile(observed_scores, box_quantile_low)),
                float(np.quantile(observed_scores, box_quantile_high)),
            )

        # 4. Reportable_box: aceita "s" ou "base_score"
        reportable_s: Optional[Tuple[float, float]] = None
        if reportable_box:
            for key in ("s", "base_score"):
                if key in reportable_box:
                    reportable_s = tuple(reportable_box[key])  # type: ignore
                    break

        # 5. Construir regiões — uma por segmento
        results: List[MonotonicityRegionResult] = []
        for j, slope in enumerate(slopes):
            lo_k = knots[j - 1] if j > 0 else float("-inf")
            hi_k = knots[j] if j < len(knots) else float("+inf")
            region_id = f"segment_{j}"
            region_desc = self._describe_segment(j, lo_k, hi_k)

            # Quantos observed_rows caem nesta região
            n_in_region = 0
            slopes_obs: Optional[np.ndarray] = None
            worst_idx: Optional[int] = None
            worst_features: Optional[dict] = None
            if observed_scores is not None and observed_scores.size:
                in_seg = (observed_scores >= lo_k) & (observed_scores < hi_k)
                n_in_region = int(in_seg.sum())
                if n_in_region:
                    # Slope constante no segmento — array preenchido
                    slopes_obs = np.full(n_in_region, slope, dtype=np.float64)
                    # Pior par == primeiro (todos iguais), só pra identificação
                    global_idx = int(np.where(in_seg)[0][0])
                    worst_idx = global_idx
                    worst_features = {
                        "s": float(observed_scores[global_idx]),
                        "slope": slope,
                    }

            # Central box: avalia slope nos cantos da intersecção entre
            # box e segmento (se houver intersecção). Como o slope é
            # constante no segmento, ambos os cantos têm mesmo slope.
            corner_slopes_central: Optional[List[float]] = None
            corner_labels_central: Optional[List[str]] = None
            worst_central: Optional[Tuple[str, dict]] = None
            if central_box_s is not None:
                cs = self._slopes_in_box_intersection(
                    central_box_s, lo_k, hi_k, slope,
                )
                if cs is not None:
                    corner_slopes_central, corner_labels_central, worst_central = cs

            # Reportable box: mesma lógica
            corner_slopes_rep: Optional[List[float]] = None
            corner_labels_rep: Optional[List[str]] = None
            worst_rep: Optional[Tuple[str, dict]] = None
            if reportable_s is not None:
                rs = self._slopes_in_box_intersection(
                    reportable_s, lo_k, hi_k, slope,
                )
                if rs is not None:
                    corner_slopes_rep, corner_labels_rep, worst_rep = rs

            results.append(MonotonicityRegionResult(
                region_id=region_id,
                region_description=region_desc,
                n_observed_rows_in_region=n_in_region,
                slope_value=slope,
                slope_at_observed_rows=slopes_obs,
                slope_at_central_box_corners=corner_slopes_central,
                corner_labels=corner_labels_central,
                slope_at_reportable_box_corners=corner_slopes_rep,
                reportable_corner_labels=corner_labels_rep,
                worst_observed_row_index=worst_idx,
                worst_observed_row_features=worst_features,
                worst_central_corner_label=worst_central[0] if worst_central else None,
                worst_central_corner_values=worst_central[1] if worst_central else None,
                worst_reportable_corner_label=worst_rep[0] if worst_rep else None,
                worst_reportable_corner_values=worst_rep[1] if worst_rep else None,
                feature_variance_in_region={},  # piecewise não tem features
                zero_variance_terms=[],
                near_zero_variance_terms=[],
            ))

        return results

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_observed_scores(raw_input_payload: Any) -> Optional[np.ndarray]:
        """Extrai vetor de scores observados de várias fontes possíveis.

        V1.5: piecewise legado tipicamente recebe `SimilarityScoreDataset`
        como input (não `LRArchitectureRawInputPayload`). Tenta múltiplas
        formas para ser robusto a interfaces.
        """
        if raw_input_payload is None:
            return None
        # Forma 1: SimilarityScoreDataset com items[].raw_score
        items = getattr(raw_input_payload, "items", None)
        if items is not None:
            try:
                scores = np.asarray(
                    [float(it.raw_score) for it in items], dtype=np.float64
                )
                return scores if scores.size else None
            except (AttributeError, TypeError, ValueError):
                pass
        # Forma 2: LRArchitectureRawInputPayload com matrix + schema
        matrix = getattr(raw_input_payload, "matrix", None)
        schema = getattr(raw_input_payload, "schema", None)
        if matrix is not None and schema is not None:
            base_col = getattr(schema, "base_score_column", None)
            if base_col is not None and 0 <= base_col < matrix.shape[1]:
                return np.asarray(matrix[:, base_col], dtype=np.float64)
        # Forma 3: array direto
        if hasattr(raw_input_payload, "shape"):
            arr = np.asarray(raw_input_payload, dtype=np.float64)
            if arr.ndim == 1:
                return arr
        return None

    @staticmethod
    def _describe_segment(j: int, lo: float, hi: float) -> str:
        if lo == float("-inf"):
            return f"s ≤ {hi:.4f} (segmento {j})"
        if hi == float("+inf"):
            return f"s > {lo:.4f} (segmento {j})"
        return f"{lo:.4f} < s ≤ {hi:.4f} (segmento {j})"

    @staticmethod
    def _slopes_in_box_intersection(
        box: Tuple[float, float],
        seg_lo: float,
        seg_hi: float,
        slope: float,
    ) -> Optional[Tuple[List[float], List[str], Tuple[str, dict]]]:
        """Para um segmento [seg_lo, seg_hi] e uma caixa [box_lo, box_hi],
        retorna slopes nos cantos da intersecção. Se não há intersecção,
        retorna None — o segmento não é coberto pelo domínio declarado
        e a auditoria não opina sobre ele nesse box.
        """
        box_lo, box_hi = box
        inter_lo = max(seg_lo, box_lo)
        inter_hi = min(seg_hi, box_hi)
        if inter_lo > inter_hi:
            return None
        # Slope é constante no segmento — ambos os cantos têm mesmo slope
        labels = [f"s={inter_lo:.4f}", f"s={inter_hi:.4f}"]
        slopes = [slope, slope]
        worst = (labels[0], {"s": inter_lo, "slope": slope})
        return slopes, labels, worst
