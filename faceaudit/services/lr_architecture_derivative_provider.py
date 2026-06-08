# faceaudit/services/lr_architecture_derivative_provider.py

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from faceaudit.models.lr.lr_architecture_spec import LRArchitectureSpec
from faceaudit.models.lr.lr_architecture_state import (
    LRArchitectureState,
    LRArchitectureStateFactory,
)
from faceaudit.protocols.monotonicity_derivative_provider_protocol import (
    MonotonicityRegionResult,
)
from faceaudit.services.architecture_feature_builder import (
    derivative_matrix_wrt_base,
)


class LRArchitectureDerivativeProvider:
    """
    Provider de derivadas `∂ log_LR / ∂ s` para arquiteturas declarativas de
    `LR_ARCHITECTURE_VECTORIZATION` (V1 da regra CONDITIONAL_MONOTONICITY).

    Estratégia:
        ∂ log_LR / ∂ s = Σ_j β_raw[j] · ∂φ_j / ∂s
    onde:
        β_raw[j] = β_z[j] / σ_j  (recupera escala bruta após StandardScaler)
        ∂φ_j / ∂s vem de `derivative_matrix_wrt_base(spec, state, ...)`
        do `architecture_feature_builder`.

    Regiões piecewise são geradas a partir da estrutura do spec:
        - Arquitetura sem hinge_positive: 1 região global ("global")
        - Arquitetura com hinge_positive: 2 regiões ("s_gt_s0", "s_le_s0")
        - Arquitetura com hinge_negative também: 2 regiões (mesma divisão
          em s_0 — h⁺ e h⁻ são complementares no mesmo knot)

    Para cada região avalia:
        - slope_at_observed_rows: slope por par observado em D_cal na região
        - slope_at_central_box_corners: slope nos cantos do retângulo
          [P_low(r̃_F), P_high(r̃_F)] × [P_low(r̃_P), P_high(r̃_P)]
          das raridades observadas (se aplicável)

    Convenção em s = s_0 (kink): usa derivada à direita
    (mesmo padrão de `derivative_matrix_wrt_base`).
    """

    def __init__(self, spec: LRArchitectureSpec) -> None:
        self._spec = spec

    def evaluate_regions(
        self,
        *,
        calibrator_metadata: dict,
        raw_input_payload: Any,
        box_quantile_low: float = 0.05,
        box_quantile_high: float = 0.95,
    ) -> List[MonotonicityRegionResult]:
        # 1. Reconstrói LRArchitectureState e β_raw
        state_dict = calibrator_metadata.get("architecture_state")
        if state_dict is None:
            raise ValueError(
                "calibrator_metadata['architecture_state'] ausente — "
                "auditoria de monotonicidade requer state persistido"
            )
        state = LRArchitectureStateFactory.from_dict(state_dict)

        beta_raw = self._extract_beta_raw(calibrator_metadata)
        if beta_raw.shape[0] != len(self._spec.feature_terms):
            raise ValueError(
                f"β_raw com {beta_raw.shape[0]} dims; spec tem "
                f"{len(self._spec.feature_terms)} feature_terms"
            )

        # 2. Extrai matriz observada de D_cal: base + raridades
        matrix, schema = self._extract_matrix_and_schema(raw_input_payload)
        base = matrix[:, schema.base_score_column]
        rarity_probe = self._safe_col(matrix, schema.rarity_probe_column)
        rarity_reference = self._safe_col(matrix, schema.rarity_reference_column)
        rarity_single = self._safe_col(matrix, getattr(schema, "rarity_single_column", None))

        # 3. Calcula derivada por par observado
        deriv_matrix = derivative_matrix_wrt_base(
            self._spec, state, base,
            rarity_probe=rarity_probe,
            rarity_reference=rarity_reference,
            rarity_single=rarity_single,
        )
        # ∂ log_LR / ∂ s = β_raw · ∂φ/∂s
        slope_per_pair = deriv_matrix @ beta_raw  # (N,)

        # 4. Particiona em regiões
        s0 = self._spec_s0(state)
        if self._has_hinge_kink():
            regions_def = [
                ("s_gt_s0", "s > s_0 (hinge positiva ativa)", base > s0),
                ("s_le_s0", "s ≤ s_0 (hinge negativa ativa)", base <= s0),
            ]
        else:
            regions_def = [
                ("global", "região global (sem kink em s_0)", np.ones_like(base, dtype=bool)),
            ]

        # 5. Computa central_box dos cantos para raridade probe e reference
        central_box = self._compute_central_box(
            rarity_probe, rarity_reference, rarity_single,
            box_quantile_low, box_quantile_high, state,
        )

        results: List[MonotonicityRegionResult] = []
        for region_id, region_desc, mask in regions_def:
            n_in_region = int(mask.sum())
            slopes_obs = slope_per_pair[mask] if n_in_region else np.array([])

            # Cantos avaliados nesta região
            corner_slopes, corner_labels, worst_corner = self._evaluate_corners(
                state=state, region_id=region_id, beta_raw=beta_raw,
                central_box=central_box,
            )

            # Pior par observado (menor slope)
            worst_idx, worst_features = None, None
            if slopes_obs.size:
                local_min = int(np.argmin(slopes_obs))
                # global_indices: onde o mask é True
                global_indices = np.where(mask)[0]
                worst_idx = int(global_indices[local_min])
                worst_features = {
                    "s": float(base[worst_idx]),
                    "r_probe": float(rarity_probe[worst_idx]) if rarity_probe is not None else None,
                    "r_reference": float(rarity_reference[worst_idx]) if rarity_reference is not None else None,
                    "r_single": float(rarity_single[worst_idx]) if rarity_single is not None else None,
                    "slope": float(slopes_obs[local_min]),
                }

            # Variância das features na região
            feat_variance, zero_var, near_zero_var = self._feature_variance_in_region(
                deriv_matrix, mask, beta_raw,
            )

            results.append(MonotonicityRegionResult(
                region_id=region_id,
                region_description=region_desc,
                n_observed_rows_in_region=n_in_region,
                slope_value=None,  # arquiteturas LR_ARCH_VEC: depende de raridades
                slope_at_observed_rows=slopes_obs if slopes_obs.size else None,
                slope_at_central_box_corners=corner_slopes if corner_slopes else None,
                corner_labels=corner_labels if corner_labels else None,
                worst_observed_row_index=worst_idx,
                worst_observed_row_features=worst_features,
                worst_central_corner_label=worst_corner[0] if worst_corner else None,
                worst_central_corner_values=worst_corner[1] if worst_corner else None,
                feature_variance_in_region=feat_variance,
                zero_variance_terms=zero_var,
                near_zero_variance_terms=near_zero_var,
            ))

        return results

    # =========================================================================
    # Helpers internos
    # =========================================================================

    @staticmethod
    def _extract_beta_raw(calibrator_metadata: dict) -> np.ndarray:
        """Recupera β_raw da metadata persistida pelo
        LogisticRegressionCalibrationModel.

        V2.14+ persiste `raw_scale_coefficients` (e `final_raw_scale_coefficients`
        é o nome teórico do design doc — V2.14 usa o nome sem prefixo). Aceita
        ambos para retrocompatibilidade. Fallback final: β_z / σ.
        """
        for key in ("raw_scale_coefficients", "final_raw_scale_coefficients"):
            raw = calibrator_metadata.get(key)
            if raw is not None:
                return np.asarray(raw, dtype=np.float64)

        # Fallback: reconstrói β_raw a partir de β_z e σ
        beta_z = (calibrator_metadata.get("standardized_coefficients")
                  or calibrator_metadata.get("final_standardized_coefficients"))
        sigma = (calibrator_metadata.get("scaler_scale")
                 or calibrator_metadata.get("final_scaler_scale"))
        if beta_z is None or sigma is None:
            raise ValueError(
                "calibrator_metadata sem raw_scale_coefficients nem "
                "(standardized_coefficients + scaler_scale) — "
                "não dá pra reconstruir β_raw"
            )
        beta_z = np.asarray(beta_z, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        return beta_z / np.maximum(sigma, 1e-30)

    @staticmethod
    def _extract_matrix_and_schema(raw_input_payload: Any) -> Tuple[np.ndarray, Any]:
        matrix = np.asarray(raw_input_payload.matrix, dtype=np.float64)
        schema = raw_input_payload.schema
        return matrix, schema

    @staticmethod
    def _safe_col(matrix: np.ndarray, col: Optional[int]) -> Optional[np.ndarray]:
        if col is None or col < 0 or col >= matrix.shape[1]:
            return None
        return matrix[:, col]

    @staticmethod
    def _spec_s0(state: LRArchitectureState) -> float:
        return float(state.s0) if state.s0 is not None else 0.0

    def _has_hinge_kink(self) -> bool:
        """True se a arquitetura tem termo dependente de s_0
        (HINGE_POSITIVE, HINGE_NEGATIVE, ou produto com h⁺/h⁻)."""
        return self._spec.requires_s0

    def _compute_central_box(
        self,
        rarity_probe: Optional[np.ndarray],
        rarity_reference: Optional[np.ndarray],
        rarity_single: Optional[np.ndarray],
        q_low: float, q_high: float,
        state: LRArchitectureState,
    ) -> dict:
        """Box de raridades observadas em quantis [P_low, P_high]
        (escala bruta — não centrada)."""
        box: dict = {}
        if rarity_probe is not None:
            box["r_probe"] = (float(np.quantile(rarity_probe, q_low)),
                              float(np.quantile(rarity_probe, q_high)))
        if rarity_reference is not None:
            box["r_reference"] = (float(np.quantile(rarity_reference, q_low)),
                                  float(np.quantile(rarity_reference, q_high)))
        if rarity_single is not None:
            box["r_single"] = (float(np.quantile(rarity_single, q_low)),
                               float(np.quantile(rarity_single, q_high)))
        return box

    def _evaluate_corners(
        self,
        *,
        state: LRArchitectureState,
        region_id: str,
        beta_raw: np.ndarray,
        central_box: dict,
    ) -> Tuple[List[float], List[str], Optional[Tuple[str, dict]]]:
        """Avalia slope nos cantos da caixa central usando um único par
        sintético em cada canto. Retorna (slopes, labels, worst).

        Para arquiteturas sem raridade, a caixa é vazia e retornamos
        (lista vazia, lista vazia, None) — o slope nessa região já
        está coberto por slope_at_observed_rows.
        """
        if not central_box:
            return [], [], None

        # Gera cantos cartesianos do box
        var_names = list(central_box.keys())
        bounds = [central_box[v] for v in var_names]
        corners_values = self._cartesian_corners(bounds)
        labels = []
        for c in corners_values:
            parts = []
            for vname, val in zip(var_names, c):
                lab = "min" if val == central_box[vname][0] else "max"
                parts.append(f"{vname}={lab}")
            labels.append(",".join(parts))

        # Para cada canto, monta vetor sintético (s representativo da região)
        # e avalia derivada.
        # Escolha de s: meio da região (s_0 + epsilon para s > s_0; s_0 - epsilon para s < s_0)
        s0 = self._spec_s0(state)
        s_repr = s0 + 0.1 if region_id == "s_gt_s0" else (s0 - 0.1 if region_id == "s_le_s0" else 0.5)

        slope_at_corners = []
        for c in corners_values:
            r_probe = c[var_names.index("r_probe")] if "r_probe" in var_names else None
            r_ref = c[var_names.index("r_reference")] if "r_reference" in var_names else None
            r_single = c[var_names.index("r_single")] if "r_single" in var_names else None
            base = np.array([s_repr], dtype=np.float64)
            r_probe_arr = np.array([r_probe], dtype=np.float64) if r_probe is not None else None
            r_ref_arr = np.array([r_ref], dtype=np.float64) if r_ref is not None else None
            r_single_arr = np.array([r_single], dtype=np.float64) if r_single is not None else None
            deriv = derivative_matrix_wrt_base(
                self._spec, state, base,
                rarity_probe=r_probe_arr,
                rarity_reference=r_ref_arr,
                rarity_single=r_single_arr,
            )
            slope = float((deriv @ beta_raw)[0])
            slope_at_corners.append(slope)

        if slope_at_corners:
            i_worst = int(np.argmin(slope_at_corners))
            worst_corner = (labels[i_worst], {
                v: corners_values[i_worst][k] for k, v in enumerate(var_names)
            })
            return slope_at_corners, labels, worst_corner

        return slope_at_corners, labels, None

    @staticmethod
    def _cartesian_corners(bounds: List[Tuple[float, float]]) -> List[Tuple[float, ...]]:
        """Produto cartesiano dos extremos. 1-dim → 2 cantos; 2-dim → 4; 3-dim → 8."""
        if not bounds:
            return []
        out: List[Tuple[float, ...]] = [()]
        for lo, hi in bounds:
            out = [t + (lo,) for t in out] + [t + (hi,) for t in out]
        return out

    @staticmethod
    def _feature_variance_in_region(
        deriv_matrix: np.ndarray,
        mask: np.ndarray,
        beta_raw: np.ndarray,
        zero_threshold: float = 1e-30,
        near_zero_threshold: float = 1e-6,
    ) -> Tuple[dict, list, list]:
        """Variância de cada coluna da matriz de derivadas dentro da região.
        Termos com variância zero/quase zero podem indicar features
        empiricamente não-identificadas."""
        feat_var = {}
        zero_var: list = []
        near_zero_var: list = []
        sub = deriv_matrix[mask]
        if sub.size == 0:
            return feat_var, zero_var, near_zero_var
        for j in range(deriv_matrix.shape[1]):
            v = float(np.var(sub[:, j]))
            feat_var[f"term_{j}"] = v
            if v < zero_threshold:
                zero_var.append(f"term_{j}")
            elif v < near_zero_threshold:
                near_zero_var.append(f"term_{j}")
        return feat_var, zero_var, near_zero_var
