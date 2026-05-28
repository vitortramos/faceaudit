# faceaudit/models/calibration/bi_gaussianized_calibration_model.py

"""Modelo de calibração Bi-Gaussianized (Morrison 2024).

Algoritmo (Morrison, *Law, Probability and Risk* 2024, §2):

  1. Calcular scores genuínos/impostores no treino.
  2. (Opcional) Calibrador inicial — LogReg, KDE, ou NONE (PLDA já em log-LR).
  3. Calcular Cllr ou EER do output do passo 2.
  4. Resolver para σ² do sistema bi-gaussiano alvo:
       - via Cllr: σ² = -(1/c)·ln((1/b)·ln(Cllr) + 1), b=17.7, c=9.33e-3.
       - via EER: σ = -2·Φ⁻¹(EER), analítico.
  5. Construir CDF empírica G(x) com pesos 0.5/(Nd+1) e 0.5/(Ns+1).
  6. Mapear: ln(Λ) = F⁻¹(G(x)), onde F é a mistura 0.5·N(+σ²/2,σ²) + 0.5·N(-σ²/2,σ²).

EER e Cllr são portados localmente pra evitar acoplamento com `ValidationMathService`.
"""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
    BiGaussGxMethod,
    BiGaussianizedCalibrationModelConfig,
    BiGaussInitialCalibrator,
    BiGaussInversionMethod,
    BiGaussScoreDirection,
    BiGaussSigmaEstimationMode,
    BiGaussTargetMetric,
)
from faceaudit.protocols.calibration_model_protocol import CalibrationModelProtocol


# Constantes do fit ln(Cllr) = b·(exp(-c·σ²) - 1) (Morrison §2.3, valores empíricos
# obtidos por regressão sobre integração numérica da Eq. (6) do paper).
_CLLR_FIT_B = 17.7
_CLLR_FIT_C = 9.33e-3

# Cap de segurança para σ²; valores acima viram cauda numericamente intratável.
_SIGMA_SQ_CAP = 1.0e3
_EPS = 1.0e-12


class BiGaussianizedCalibrationModel(CalibrationModelProtocol):
    """Calibrador Bi-Gaussianized (Morrison 2024).

    Convenção de labels:
      - 1 = genuíno / Hp
      - 0 = impostor / Hd
    """

    MODEL_VERSION = 3  # v3: + calibrator_cllr_train/_cv (CV diagnóstico via executor)

    def __init__(
        self,
        config: Optional[BiGaussianizedCalibrationModelConfig] = None,
        name: str = "Bi-Gaussianized Calibration Model",
    ):
        self.config = config or BiGaussianizedCalibrationModelConfig()
        self._name = name
        self._is_fitted = False

        # Estado treinado
        self._sigma_target: Optional[float] = None
        self._g_interp: Optional[interp1d] = None
        self._x_min: Optional[float] = None
        self._x_max: Optional[float] = None
        self._n_total: int = 0

        # Inversa da CDF da mistura (preenchido se INTERP1D)
        self._mixture_inv_interp: Optional[interp1d] = None

        # Score direction: True se "lower is genuine" e o modelo deve flipar
        # o sinal dos scores em fit + calibrate_scores. Decidido em fit().
        self._flip_sign: bool = False

        self.metadata: dict = {}

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def requires_training(self) -> bool:
        return True

    @property
    def sigma_target(self) -> Optional[float]:
        """σ do sistema bi-gaussiano alvo (depois do shrinkage)."""
        return self._sigma_target

    # =========================================================================
    # TREINO
    # =========================================================================

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        scores_arr, labels_arr = self._normalize_inputs(scores, labels)

        # ── Passo 0: direção dos scores ────────────────────────────────────────
        # Resolve score_direction (AUTO / HIGHER / LOWER). Se LOWER, flipa o
        # sinal de scores_arr antes de tudo. _flip_sign é guardado e aplicado
        # em calibrate_scores depois.
        scores_arr, direction_resolved = self._resolve_score_direction(
            scores_arr, labels_arr, logger_callback
        )

        gen = scores_arr[labels_arr == 1]
        imp = scores_arr[labels_arr == 0]

        if gen.size == 0 or imp.size == 0:
            raise ValueError(
                "Bi-Gaussianized calibration requer amostras genuínas e impostoras."
            )

        n_gen, n_imp = int(gen.size), int(imp.size)
        logger_callback(
            f"[{self.name}] 🧠 Treinando "
            f"(N_gen={n_gen} | N_imp={n_imp} | "
            f"metric={self.config.target_metric.value} | "
            f"initial={self.config.initial_calibrator.value} | "
            f"sigma_est={self.config.sigma_estimation_mode.value})..."
        )

        # ── Passos 2 e 3: calibrador inicial + métrica ─────────────────────────
        cllr_value: Optional[float] = None
        eer_value: Optional[float] = None

        if self.config.target_metric == BiGaussTargetMetric.CLLR:
            llr_gen, llr_imp = self._compute_llrs_for_sigma_estimation(
                gen, imp, logger_callback
            )
            cllr_value = self._compute_cllr(llr_gen, llr_imp)
            logger_callback(
                f"[{self.name}] Cllr "
                f"({self.config.sigma_estimation_mode.value.lower()}) = {cllr_value:.6f}"
            )
            sigma_sq = self._sigma_sq_from_cllr(cllr_value)
        else:
            # EER usa scores diretamente — calibrador inicial não é necessário.
            eer_value = self._compute_eer(scores_arr, labels_arr)
            logger_callback(f"[{self.name}] EER no treino = {eer_value:.6f}")
            sigma_sq = self._sigma_sq_from_eer(eer_value)

        sigma_sq = float(min(max(sigma_sq, _EPS), _SIGMA_SQ_CAP))
        sigma = float(np.sqrt(sigma_sq)) * float(self.config.sigma_shrinkage_factor)
        self._sigma_target = sigma

        logger_callback(
            f"[{self.name}] σ alvo = {sigma:.4f} "
            f"(σ²={sigma * sigma:.4f}, shrinkage={self.config.sigma_shrinkage_factor})"
        )

        # ── Passo 5: CDF empírica G(x) ─────────────────────────────────────────
        self._build_g_cdf(scores_arr, labels_arr, logger_callback=logger_callback)

        # ── Pré-computa F⁻¹ da mistura se INTERP1D ─────────────────────────────
        if self.config.inversion_method == BiGaussInversionMethod.INTERP1D:
            logger_callback(
                f"[{self.name}] Construindo F⁻¹ da mistura "
                f"(grid {int(self.config.interpolation_points)} pontos)..."
            )
            self._mixture_inv_interp = self._build_mixture_inverse(sigma)
        else:
            self._mixture_inv_interp = None

        # ── Marca como fitted ANTES do diagnóstico pós-fit (calibrate_scores
        # exige _ensure_fitted).
        self._n_total = n_gen + n_imp
        self._is_fitted = True

        # ── Diagnóstico pós-fit: Cllr realizado vs Cllr predito pela fórmula ───
        # cllr_after: Cllr do output completo (após bi-gaussianização) no MESMO
        # conjunto de treino. Bias-otimista mas útil pra detectar implementação
        # quebrada — divergência grande contra `cllr_predicted` indica problema
        # em G(x), F⁻¹ ou clipping agressivo.
        cllr_after, cllr_predicted = self._compute_post_fit_diagnostic(
            scores_arr, labels_arr, sigma_sq, logger_callback,
        )

        self.metadata = {
            "model_version": self.MODEL_VERSION,
            "sigma_target": float(sigma),
            "sigma_target_squared": float(sigma * sigma),
            "target_metric": self.config.target_metric.value,
            "initial_calibrator": self.config.initial_calibrator.value,
            "inversion_method": self.config.inversion_method.value,
            "gx_method": self.config.gx_method.value,
            "shrinkage_factor": float(self.config.sigma_shrinkage_factor),
            "sigma_estimation_mode": self.config.sigma_estimation_mode.value,
            "score_direction_config": self.config.score_direction.value,
            "score_direction_resolved": direction_resolved,
            "flip_sign_applied": bool(self._flip_sign),
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "cllr_at_fit": float(cllr_value) if cllr_value is not None else None,
            "eer_at_fit": float(eer_value) if eer_value is not None else None,
            "cllr_after": float(cllr_after) if cllr_after is not None else None,
            "cllr_predicted_from_sigma": (
                float(cllr_predicted) if cllr_predicted is not None else None
            ),
            "clipping_enabled": bool(self.config.enable_clipping),
            "min_log_lr": (
                float(self.config.min_log_lr)
                if self.config.enable_clipping else None
            ),
            "max_log_lr": (
                float(self.config.max_log_lr)
                if self.config.enable_clipping else None
            ),
        }
        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

    # =========================================================================
    # INFERÊNCIA
    # =========================================================================

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        log_lr = self.calibrate_scores(scores)
        # P(Hp|x) = sigmoid(log_lr) com prior 0.5; P(Hd|x) = 1 - P(Hp|x).
        # Estável para log_lr extremo.
        p_hp = np.where(
            log_lr >= 0,
            1.0 / (1.0 + np.exp(-log_lr)),
            np.exp(log_lr) / (1.0 + np.exp(log_lr)),
        )
        p_hd = 1.0 - p_hp
        return np.column_stack([p_hd, p_hp]).astype(np.float64)

    def calibrate_scores(self, scores: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        score_vec = self._normalize_scores_only(scores)

        # Aplica flip de sinal SE foi determinado em fit (lower-is-genuine).
        if self._flip_sign:
            score_vec = -score_vec

        # G(x) → u ∈ (0, 1); clamp pra evitar 0/1 exatos.
        u = self._g_apply(score_vec)
        u = np.clip(u, _EPS, 1.0 - _EPS)

        # F⁻¹(u) → ln(Λ).
        if self.config.inversion_method == BiGaussInversionMethod.INTERP1D:
            log_lr = self._mixture_inv_interp(u).astype(np.float64)
        else:
            log_lr = self._mixture_inv_brent(u, self._sigma_target)

        if self.config.enable_clipping:
            return np.clip(log_lr, self.config.min_log_lr, self.config.max_log_lr)
        return log_lr

    # =========================================================================
    # DIAGNÓSTICO
    # =========================================================================

    def diagnose_scores(self, scores: np.ndarray) -> Dict[str, Any]:
        """Diagnóstico forense de aplicação do calibrador a um conjunto de scores.

        Retorna um dict com:
          - n_scores, score_min_seen, score_max_seen
          - training_score_range (x_min, x_max do treino)
          - pct_below_training_range, pct_above_training_range
              (extrapolação fora do range observado em treino)
          - pct_at_min_log_lr, pct_at_max_log_lr
              (saturação no clipping configurado)
          - training_log_lr_range (min_log_lr, max_log_lr)
          - flip_sign_applied (lower-is-genuine ativado)

        Útil pra produção forense: alerta se scores do caso estão fora da
        faixa do treino (mapeados pra valores de boundary) ou se muitos
        log-LRs saturam no clipping (perda de informação extrema).
        """
        self._ensure_fitted()
        score_vec = self._normalize_scores_only(scores)
        # Aplica flip ANTES de comparar com x_min/x_max (que estão no espaço
        # internamente armazenado, já flipado se foi o caso).
        score_for_range = -score_vec if self._flip_sign else score_vec

        n = int(score_vec.size)
        if n == 0:
            return {
                "n_scores": 0,
                "training_score_range": (
                    float(self._x_min) if self._x_min is not None else None,
                    float(self._x_max) if self._x_max is not None else None,
                ),
            }

        log_lr = self.calibrate_scores(scores)

        min_seen = float(score_vec.min())
        max_seen = float(score_vec.max())
        below = int((score_for_range < self._x_min).sum())
        above = int((score_for_range > self._x_max).sum())

        # Saturação em min/max só faz sentido quando clipping está ligado.
        clipping_on = bool(self.config.enable_clipping)
        if clipping_on:
            clip_lo = float(self.config.min_log_lr)
            clip_hi = float(self.config.max_log_lr)
            eps_clip = 1e-9
            at_min = int((log_lr <= clip_lo + eps_clip).sum())
            at_max = int((log_lr >= clip_hi - eps_clip).sum())
            training_log_lr_range = (clip_lo, clip_hi)
            pct_at_min = 100.0 * at_min / n
            pct_at_max = 100.0 * at_max / n
        else:
            at_min = at_max = 0
            training_log_lr_range = None
            pct_at_min = pct_at_max = None

        return {
            "n_scores": n,
            "score_min_seen": min_seen,
            "score_max_seen": max_seen,
            "training_score_range": (float(self._x_min), float(self._x_max)),
            "training_log_lr_range": training_log_lr_range,
            "clipping_enabled": clipping_on,
            "pct_below_training_range": 100.0 * below / n,
            "pct_above_training_range": 100.0 * above / n,
            "pct_at_min_log_lr": pct_at_min,
            "pct_at_max_log_lr": pct_at_max,
            "n_below_training_range": below,
            "n_above_training_range": above,
            "n_at_min_log_lr": at_min,
            "n_at_max_log_lr": at_max,
            "flip_sign_applied": bool(self._flip_sign),
        }

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def save(self, path: str) -> None:
        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "bi_gaussianized_calibration_model",
            "name": self._name,
            "config": self.config.to_dict(),
            "sigma_target": self._sigma_target,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "n_total": self._n_total,
            "flip_sign": bool(self._flip_sign),  # v2: score direction state
            # Salva os pontos do interpolador G(x), não o objeto interp1d
            # (mais portável e pequeno).
            "g_interp_x": (
                self._g_interp.x.copy() if self._g_interp is not None else None
            ),
            "g_interp_y": (
                self._g_interp.y.copy() if self._g_interp is not None else None
            ),
            "metadata": self.metadata,
            "is_fitted": self._is_fitted,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo do calibrador não encontrado: {path_obj}")

        with open(path_obj, "rb") as f:
            payload = pickle.load(f)

        self._name = payload.get("name", self._name)
        self.config = BiGaussianizedCalibrationModelConfig.from_dict(
            payload.get("config", {})
        )
        self._sigma_target = payload.get("sigma_target")
        self._x_min = payload.get("x_min")
        self._x_max = payload.get("x_max")
        self._n_total = int(payload.get("n_total", 0))
        self.metadata = payload.get("metadata", {})
        self._is_fitted = bool(payload.get("is_fitted", False))
        # v2: flip_sign — payloads de v1 não tinham → default False
        # (comportamento idêntico ao v1: assume higher-is-genuine sem flip).
        self._flip_sign = bool(payload.get("flip_sign", False))

        # Reconstrói o interpolador G(x).
        gx = payload.get("g_interp_x")
        gy = payload.get("g_interp_y")
        if gx is not None and gy is not None:
            self._g_interp = interp1d(
                gx, gy,
                kind="linear",
                bounds_error=False,
                fill_value=(float(gy[0]), float(gy[-1])),
                assume_sorted=True,
            )
        else:
            self._g_interp = None

        # Reconstrói F⁻¹ se INTERP1D.
        if (
            self._is_fitted
            and self._sigma_target is not None
            and self.config.inversion_method == BiGaussInversionMethod.INTERP1D
        ):
            self._mixture_inv_interp = self._build_mixture_inverse(self._sigma_target)
        else:
            self._mixture_inv_interp = None

    # =========================================================================
    # FACTORY DE FOLD (CV diagnóstico)
    # =========================================================================

    def _make_frozen_factory(self) -> Callable[[], "BiGaussianizedCalibrationModel"]:
        """Factory pra k-fold CV diagnóstico: clona o config com k_folds=0
        pra impedir recursão. Os hiperparâmetros do BiGauss (target_metric,
        initial_calibrator, sigma_estimation_mode, gx_method, sigma_cv_folds,
        bw_kde_factor, etc.) já são auto-contidos no DTO — não precisam ser
        congelados como C/bandwidth do LogReg/KDE. O fit por fold roda o
        pipeline interno completo (initial calibrator + σ² + G(x) + F⁻¹)
        com hiperparâmetros idênticos ao production model."""
        self._ensure_fitted()
        frozen_config = copy.deepcopy(self.config)
        frozen_config.k_folds = 0

        base_name = self._name

        def _make() -> "BiGaussianizedCalibrationModel":
            return BiGaussianizedCalibrationModel(
                config=copy.deepcopy(frozen_config),
                name=f"{base_name} [CV-fold]",
            )

        return _make

    # =========================================================================
    # CÁLCULO DE Cllr e EER (PORTADOS LOCALMENTE; SEM ACOPLAR ValidationMathService)
    # =========================================================================

    @staticmethod
    def _compute_cllr(llr_gen: np.ndarray, llr_imp: np.ndarray) -> float:
        """Cllr (log-likelihood-ratio cost) numericamente estável.

        Cllr = 0.5·(mean_gen log2(1 + 1/Λ) + mean_imp log2(1 + Λ))
             = 0.5·(mean_gen log2(1 + e^(-llr)) + mean_imp log2(1 + e^(llr)))

        Usa log1p + np.where pra evitar overflow em llr extremo.
        """
        llr_gen = np.asarray(llr_gen, dtype=np.float64)
        llr_imp = np.asarray(llr_imp, dtype=np.float64)

        # log2(1 + e^(-llr)) genuínos: softplus(-llr) / ln(2)
        # log2(1 + e^( llr)) impostores: softplus( llr) / ln(2)
        def _softplus(x: np.ndarray) -> np.ndarray:
            # log(1 + e^x), estável: max(x, 0) + log1p(e^(-|x|))
            return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))

        ln2 = float(np.log(2.0))
        term_gen = _softplus(-llr_gen).mean() / ln2 if llr_gen.size else 0.0
        term_imp = _softplus(llr_imp).mean() / ln2 if llr_imp.size else 0.0
        return 0.5 * (term_gen + term_imp)

    @staticmethod
    def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
        """EER via varredura de thresholds. Mesma lógica do algoritmo do paper §2.5
        (Morrison): sort por score, FAR/MR a cada step, encontra cruzamento."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Sort decrescente por score; threshold passa por cada score.
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]

        # A cada step i (threshold = sorted_scores[i]), aceita-se os primeiros i+1.
        # FAR = false positives / n_neg; FRR (miss rate) = false negatives / n_pos.
        cum_pos = np.cumsum(sorted_labels == 1)  # TP até index i (inclusivo)
        cum_neg = np.cumsum(sorted_labels == 0)  # FP até index i
        far = cum_neg / n_neg
        frr = 1.0 - (cum_pos / n_pos)

        # Procura cruzamento: primeiro i onde frr <= far.
        diff = frr - far
        crossing = np.where(diff <= 0)[0]
        if len(crossing) == 0:
            # Nunca cruzou: cauda do EER no extremo
            return float(min(far[-1], frr[0]))
        idx = int(crossing[0])
        if idx == 0:
            return float(0.5 * (far[0] + frr[0]))
        # Interpolação linear entre idx-1 e idx
        f1, f2 = far[idx - 1], far[idx]
        m1, m2 = frr[idx - 1], frr[idx]
        denom = (f2 - f1) - (m2 - m1)
        if abs(denom) < _EPS:
            return float(0.5 * (f2 + m2))
        t = (m1 - f1) / denom
        eer = float(f1 + t * (f2 - f1))
        return max(0.0, min(0.5, eer))

    # =========================================================================
    # SCORE DIRECTION (defensive)
    # =========================================================================

    def _resolve_score_direction(
        self,
        scores_arr: np.ndarray,
        labels_arr: np.ndarray,
        logger_callback: Callable[[str], None],
    ) -> Tuple[np.ndarray, str]:
        """Detecta/aplica a direção dos scores (higher vs lower is genuine).

        Em AUTO: compara médias gen vs imp pra inferir. Loga decisão.
        Em LOWER_IS_GENUINE: flipa sinal de scores_arr.
        Em HIGHER_IS_GENUINE: sem flip.

        Retorna (scores_arr possivelmente flipados, label da direção resolvida).
        Seta self._flip_sign no estado do modelo.
        """
        direction_cfg = self.config.score_direction
        gen = scores_arr[labels_arr == 1]
        imp = scores_arr[labels_arr == 0]

        if direction_cfg == BiGaussScoreDirection.AUTO:
            if gen.size > 0 and imp.size > 0:
                gen_mean = float(gen.mean())
                imp_mean = float(imp.mean())
                if gen_mean < imp_mean:
                    resolved = "LOWER_IS_GENUINE"
                    self._flip_sign = True
                    logger_callback(
                        f"[{self.name}] ⚠ Direção AUTO-detectada: LOWER_IS_GENUINE "
                        f"(gen_mean={gen_mean:.4f} < imp_mean={imp_mean:.4f}). "
                        "Aplicando flip de sinal."
                    )
                    scores_arr = -scores_arr
                else:
                    resolved = "HIGHER_IS_GENUINE"
                    self._flip_sign = False
                    logger_callback(
                        f"[{self.name}] Direção AUTO-detectada: HIGHER_IS_GENUINE "
                        f"(gen_mean={gen_mean:.4f} > imp_mean={imp_mean:.4f})."
                    )
            else:
                resolved = "HIGHER_IS_GENUINE"
                self._flip_sign = False
        elif direction_cfg == BiGaussScoreDirection.LOWER_IS_GENUINE:
            resolved = "LOWER_IS_GENUINE"
            self._flip_sign = True
            logger_callback(
                f"[{self.name}] Direção configurada: LOWER_IS_GENUINE. "
                "Aplicando flip de sinal."
            )
            scores_arr = -scores_arr
        else:
            resolved = "HIGHER_IS_GENUINE"
            self._flip_sign = False

        return scores_arr, resolved

    # =========================================================================
    # ESTIMAÇÃO DE σ² (in-sample ou cross-fitted)
    # =========================================================================

    def _compute_llrs_for_sigma_estimation(
        self,
        gen: np.ndarray,
        imp: np.ndarray,
        logger_callback: Callable[[str], None],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retorna (llr_gen, llr_imp) usados pra computar Cllr que alimenta σ².

        IN_SAMPLE: fit + predict do calibrador inicial no mesmo dataset.
        CROSS_FITTED: k-fold estratificado; predições held-out concatenadas.
                       Sem efeito se initial=NONE (não há nada pra cross-fittar).
        """
        mode = self.config.sigma_estimation_mode
        kind = self.config.initial_calibrator

        if mode == BiGaussSigmaEstimationMode.IN_SAMPLE or kind == BiGaussInitialCalibrator.NONE:
            return self._apply_initial_calibrator(gen, imp, logger_callback)

        # CROSS_FITTED com calibrador inicial não-trivial
        return self._cross_fitted_initial_llrs(gen, imp, logger_callback)

    def _cross_fitted_initial_llrs(
        self,
        gen: np.ndarray,
        imp: np.ndarray,
        logger_callback: Callable[[str], None],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """k-fold estratificado: cada amostra recebe llr predito por inicial
        treinado em outras k-1 dobras. Sem data leak. Cllr resultante é
        não-viesado pra estimação de σ².
        """
        from sklearn.model_selection import StratifiedKFold

        scores_all = np.concatenate([gen, imp])
        labels_all = np.concatenate([
            np.ones(gen.size, dtype=np.int32),
            np.zeros(imp.size, dtype=np.int32),
        ])

        # Reduz k se a classe minoritária tem poucos exemplos
        k_cfg = int(self.config.sigma_cv_folds)
        min_class = int(min(gen.size, imp.size))
        k = min(k_cfg, max(2, min_class))
        if k < k_cfg:
            logger_callback(
                f"[{self.name}] ⚠ Reduzindo k de {k_cfg} para {k} "
                f"(classe minoritária tem {min_class} exemplos)."
            )

        logger_callback(
            f"[{self.name}] Cross-fitted Cllr estimation "
            f"(k={k} folds estratificados, initial={self.config.initial_calibrator.value})"
        )

        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=int(self.config.random_seed),
        )
        held_out_llrs = np.empty(scores_all.size, dtype=np.float64)

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(scores_all, labels_all), start=1
        ):
            train_scores = scores_all[train_idx]
            train_labels = labels_all[train_idx]
            train_gen = train_scores[train_labels == 1]
            train_imp = train_scores[train_labels == 0]
            test_scores = scores_all[test_idx]

            logger_callback(
                f"[{self.name}]   • Fold {fold_idx}/{k}: "
                f"train n_gen={train_gen.size:,}, n_imp={train_imp.size:,}; "
                f"test n={test_scores.size:,}"
            )

            fitted = self._fit_initial_only(train_gen, train_imp, logger_callback)
            held_out_llrs[test_idx] = self._predict_initial(fitted, test_scores)

        llr_gen = held_out_llrs[labels_all == 1]
        llr_imp = held_out_llrs[labels_all == 0]
        return llr_gen, llr_imp

    def _fit_initial_only(
        self,
        gen: np.ndarray,
        imp: np.ndarray,
        logger_callback: Callable[[str], None],
    ) -> Tuple[str, Any]:
        """Treina o calibrador inicial e retorna um handle opaco.
        Não usado pelo path IN_SAMPLE — só pelo CROSS_FITTED, que precisa de
        fit+predict separados.
        """
        kind = self.config.initial_calibrator

        if kind == BiGaussInitialCalibrator.NONE:
            return ("none", None)

        if kind == BiGaussInitialCalibrator.LOGREG:
            from sklearn.linear_model import LogisticRegression
            x = np.concatenate([gen, imp]).reshape(-1, 1)
            y = np.concatenate([
                np.ones(gen.size, dtype=np.int32),
                np.zeros(imp.size, dtype=np.int32),
            ])
            lr = LogisticRegression(
                C=float(self.config.initial_logreg_C),
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
            )
            lr.fit(x, y)
            return ("logreg", lr)

        if kind == BiGaussInitialCalibrator.KDE:
            from sklearn.neighbors import KernelDensity
            max_per_class = int(self.config.initial_kde_max_samples_per_class)
            rng = np.random.default_rng(int(self.config.random_seed))

            def _maybe_subsample(arr: np.ndarray) -> np.ndarray:
                if max_per_class <= 0 or arr.size <= max_per_class:
                    return arr
                idx = rng.choice(arr.size, size=max_per_class, replace=False)
                return arr[idx]

            gen_fit = _maybe_subsample(gen)
            imp_fit = _maybe_subsample(imp)
            bw_cfg = float(self.config.initial_kde_bandwidth)
            bw_gen = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(gen_fit)
            bw_imp = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(imp_fit)
            kde_g = KernelDensity(kernel="gaussian", bandwidth=bw_gen).fit(
                gen_fit.reshape(-1, 1)
            )
            kde_i = KernelDensity(kernel="gaussian", bandwidth=bw_imp).fit(
                imp_fit.reshape(-1, 1)
            )
            return ("kde", (kde_g, kde_i))

        raise ValueError(f"initial_calibrator desconhecido: {kind}")

    def _predict_initial(
        self,
        fitted: Tuple[str, Any],
        scores: np.ndarray,
    ) -> np.ndarray:
        """Aplica o calibrador inicial treinado a `scores`, retorna llr 1D."""
        kind, model = fitted
        if kind == "none":
            return scores.astype(np.float64)
        if kind == "logreg":
            return model.decision_function(scores.reshape(-1, 1)).astype(np.float64)
        if kind == "kde":
            kde_g, kde_i = model
            log_p_g = kde_g.score_samples(scores.reshape(-1, 1))
            log_p_i = kde_i.score_samples(scores.reshape(-1, 1))
            return (log_p_g - log_p_i).astype(np.float64)
        raise ValueError(f"handle desconhecido: {kind}")

    # =========================================================================
    # DIAGNÓSTICO PÓS-FIT
    # =========================================================================

    def _compute_post_fit_diagnostic(
        self,
        scores_arr: np.ndarray,
        labels_arr: np.ndarray,
        sigma_sq: float,
        logger_callback: Callable[[str], None],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calcula Cllr realizado pós-bi-gaussianização vs Cllr predito pela
        fórmula ln(Cllr) = b·(exp(-c·σ²) - 1). Divergência grande indica:
          - G(x) não cobrindo bem as caudas
          - F⁻¹ com precisão insuficiente
          - Clipping min/max_log_lr ativo demais

        Retorna (cllr_after, cllr_predicted). Ambos None se cálculo falhar.
        """
        try:
            # `calibrate_scores` espera scores no espaço ORIGINAL (não-flipado),
            # porque ele aplica `_flip_sign` internamente. Mas aqui já estamos
            # com `scores_arr` flipado (pós-_resolve_score_direction). Então
            # invertemos pra passar input "original".
            input_scores = -scores_arr if self._flip_sign else scores_arr
            log_lrs = self.calibrate_scores(input_scores)
            cllr_after = float(self._compute_cllr(
                log_lrs[labels_arr == 1],
                log_lrs[labels_arr == 0],
            ))
            # Cllr predito pela fórmula: Cllr = exp(b·(exp(-c·σ²) - 1))
            cllr_predicted = float(np.exp(
                _CLLR_FIT_B * (np.exp(-_CLLR_FIT_C * sigma_sq) - 1.0)
            ))
            logger_callback(
                f"[{self.name}] Diagnóstico pós-fit: "
                f"Cllr_after={cllr_after:.6f} | Cllr_predicted={cllr_predicted:.6f} "
                f"(divergência {abs(cllr_after - cllr_predicted):.4f})"
            )
            return cllr_after, cllr_predicted
        except Exception as exc:
            logger_callback(
                f"[{self.name}] ⚠ Falha no diagnóstico pós-fit: {exc}"
            )
            return None, None

    # =========================================================================
    # σ ↔ MÉTRICA
    # =========================================================================

    @staticmethod
    def _sigma_sq_from_cllr(cllr: float) -> float:
        """Inverte ln(Cllr) = b·(exp(-c·σ²) - 1) → σ² = -(1/c)·ln((1/b)·ln(Cllr) + 1)."""
        cllr = max(min(float(cllr), 1.0 - _EPS), _EPS)
        inner = (np.log(cllr) / _CLLR_FIT_B) + 1.0
        if inner <= 0:
            # Cllr muito ruim (próximo de 1 ou acima) → σ² grande.
            return _SIGMA_SQ_CAP
        return float(-np.log(inner) / _CLLR_FIT_C)

    @staticmethod
    def _sigma_sq_from_eer(eer: float) -> float:
        """Forma analítica: σ = -2·Φ⁻¹(EER) (Morrison Eq. 10)."""
        eer = max(min(float(eer), 0.5 - _EPS), _EPS)
        sigma = -2.0 * float(norm.ppf(eer))
        return float(sigma * sigma)

    # =========================================================================
    # CDF EMPÍRICA G(x)
    # =========================================================================

    def _build_g_cdf(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """Dispatcher: constrói G(x) conforme `config.gx_method`."""
        if self.config.gx_method == BiGaussGxMethod.KDE_SMOOTHED:
            self._build_smoothed_cdf(scores, labels, logger_callback=logger_callback)
        else:
            self._build_empirical_cdf(scores, labels)

    def _build_empirical_cdf(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """G(x) via CDF empírica piecewise-linear (paper Morrison §2.7).

        Peso 0.5/(Nd+1) por imp e 0.5/(Ns+1) por gen; pesos somam < 1 para
        evitar G(x_max) = 1 exato.
        """
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        weights = np.where(
            labels == 1,
            0.5 / (n_pos + 1),
            0.5 / (n_neg + 1),
        ).astype(np.float64)

        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_weights = weights[order]
        cum = np.cumsum(sorted_weights)

        # Deduplica scores iguais mantendo o último valor da CDF.
        unique_mask = np.concatenate([
            np.diff(sorted_scores) > 0, np.array([True])
        ])
        gx = sorted_scores[unique_mask]
        gy = cum[unique_mask]

        self._x_min = float(gx[0])
        self._x_max = float(gx[-1])
        self._g_interp = interp1d(
            gx, gy,
            kind="linear",
            bounds_error=False,
            fill_value=(float(gy[0]), float(gy[-1])),
            assume_sorted=True,
        )

    def _build_smoothed_cdf(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """G(x) via KDE-smoothing (variante deste projeto, não-canônica).

        Constrói G(x) = 0.5·F̂_g(x) + 0.5·F̂_i(x) onde F̂_c é a CDF do KDE
        Gaussiano por classe (kernel Φ tem CDF de forma fechada). Bandwidth
        independente por classe via Silverman (default) ou config.

        Vantagem: elimina stairsteps no output quando há poucos genuínos.
        Custo: introduz bandwidth como hiperparâmetro; pode levemente
        distorcer regiões de transição abrupta.

        Compatibilidade com paper §2.7:
        - Pesos 0.5/0.5 entre classes preservados.
        - Não-extrapolação preservada se `gx_clamp_to_training_range=True`
          (interp1d.fill_value clampa nos extremos do grid).
        """
        gen = scores[labels == 1]
        imp = scores[labels == 0]

        bw_gen = float(self.config.gx_kde_bandwidth_gen)
        if bw_gen <= 0:
            bw_gen = self._silverman_bandwidth(gen)
        bw_imp = float(self.config.gx_kde_bandwidth_imp)
        if bw_imp <= 0:
            bw_imp = self._silverman_bandwidth(imp)

        # Multiplicador uniforme aplicado em cima da bandwidth resolvida
        # (Silverman ou explícita). Permite oversmoothing sem mexer no fluxo
        # automático — defensável em laudo via `gx_kde_bandwidth_factor` no JSON.
        bw_factor = float(self.config.gx_kde_bandwidth_factor)
        if bw_factor != 1.0:
            bw_gen *= bw_factor
            bw_imp *= bw_factor

        x_min = float(scores.min())
        x_max = float(scores.max())
        n_grid = int(self.config.gx_grid_points)
        clamp = bool(self.config.gx_clamp_to_training_range)

        if clamp:
            # Grid exatamente no range do treino — interp1d.fill_value clampa
            # nos boundary values, preservando a não-extrapolação do paper.
            grid_lo, grid_hi = x_min, x_max
        else:
            # Estende grid em ±5·max(h) pra captar a cauda do KDE; valores
            # ainda mais distantes recebem 0/1 via fill_value.
            margin = 5.0 * max(bw_gen, bw_imp, 1e-6)
            grid_lo = x_min - margin
            grid_hi = x_max + margin

        x_grid = np.linspace(grid_lo, grid_hi, n_grid)

        logger_callback(
            f"[{self.name}] Construindo G(x) smoothed "
            f"(N_grid={n_grid}, h_gen={bw_gen:.4f}, h_imp={bw_imp:.4f})..."
        )

        # G(x) = 0.5·F̂_g(x) + 0.5·F̂_i(x) onde F̂_c = (1/N_c) Σ Φ((x - x_ci)/h_c).
        # Broadcasting é O(N_grid · N_c) — chunkado pra evitar matrizes grandes.
        logger_callback(
            f"[{self.name}]   • F̂_g(x): {gen.size:,} genuínos × {n_grid:,} grid pts..."
        )
        f_gen = self._kernel_cdf_grid(x_grid, gen, bw_gen)
        logger_callback(
            f"[{self.name}]   • F̂_i(x): {imp.size:,} impostores × {n_grid:,} grid pts..."
        )
        f_imp = self._kernel_cdf_grid(x_grid, imp, bw_imp)
        g_grid = 0.5 * f_gen + 0.5 * f_imp

        # Garante monotonia estrita (proteção numérica)
        g_grid = np.maximum.accumulate(g_grid)

        # Re-normaliza pra preservar a propriedade do paper §2.7 de que
        # G(x_max) < 1.0 (com a margem 0.5/(N+1) somando 1 - eps).
        # Aqui usamos o equivalente: força G(x_max) = 1 - 0.5/(N_total + 1).
        n_total = scores.size
        eps_top = 0.5 / (n_total + 1)
        eps_bot = eps_top
        # Rescala linear pra [eps_bot, 1 - eps_top]
        g_min = float(g_grid[0])
        g_max = float(g_grid[-1])
        if g_max - g_min > _EPS:
            g_grid = eps_bot + (g_grid - g_min) * (1.0 - eps_top - eps_bot) / (g_max - g_min)
        else:
            g_grid = np.full_like(g_grid, 0.5)

        self._x_min = x_min
        self._x_max = x_max
        self._g_interp = interp1d(
            x_grid, g_grid,
            kind="linear",
            bounds_error=False,
            fill_value=(float(g_grid[0]), float(g_grid[-1])),
            assume_sorted=True,
        )

    @staticmethod
    def _kernel_cdf_grid(
        x_grid: np.ndarray,
        samples: np.ndarray,
        bw: float,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """F̂(x) = (1/N) Σ_i Φ((x - x_i)/bw) computada em chunks pra economizar RAM.

        Memory cap aproximado: chunk_size × len(samples) × 8 bytes.
        Com chunk=256 e samples=10k, ~20MB por chunk — seguro.
        """
        from scipy.stats import norm
        out = np.empty(len(x_grid), dtype=np.float64)
        n = samples.size
        if n == 0 or bw <= 0:
            return np.full_like(x_grid, 0.5, dtype=np.float64)
        for i in range(0, len(x_grid), chunk_size):
            chunk = x_grid[i:i + chunk_size]
            z = (chunk[:, None] - samples[None, :]) / bw
            out[i:i + chunk_size] = norm.cdf(z).mean(axis=1)
        return out

    def _g_apply(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._g_interp(x), dtype=np.float64)

    # =========================================================================
    # INVERSA F⁻¹ DA MISTURA 0.5·N(+σ²/2,σ²) + 0.5·N(-σ²/2,σ²)
    # =========================================================================

    def _build_mixture_inverse(self, sigma: float) -> interp1d:
        """Pré-computa um interpolador F⁻¹(u) → ln(Λ) usando grid uniforme em ln(Λ).

        F(z) = 0.5·Φ((z - σ²/2)/σ) + 0.5·Φ((z + σ²/2)/σ).
        F é monotônica → invertemos numericamente via grid + interp1d.
        """
        sigma = float(sigma)
        n_points = int(self.config.interpolation_points)
        half_sigma_sq = 0.5 * sigma * sigma

        # Range em ln(Λ) que cobre as caudas; baseado em ±k·σ a partir das médias.
        max_range = max(
            self.config.brent_search_range,
            half_sigma_sq + 6.0 * sigma + 1.0,
        )
        z_grid = np.linspace(-max_range, max_range, n_points)

        # CDF da mistura
        u_grid = 0.5 * norm.cdf(z_grid, loc=+half_sigma_sq, scale=sigma) + \
                 0.5 * norm.cdf(z_grid, loc=-half_sigma_sq, scale=sigma)

        # Garante monotonia estrita (numpy.linspace + CDF é monotônico, mas
        # pode ter ties nas caudas devido a underflow).
        u_grid = np.maximum.accumulate(u_grid)
        eps_mono = 1e-15
        for i in range(1, len(u_grid)):
            if u_grid[i] <= u_grid[i - 1]:
                u_grid[i] = u_grid[i - 1] + eps_mono

        return interp1d(
            u_grid, z_grid,
            kind="linear",
            bounds_error=False,
            fill_value=(float(z_grid[0]), float(z_grid[-1])),
            assume_sorted=True,
        )

    def _mixture_cdf_scalar(self, z: float, sigma: float) -> float:
        half_sigma_sq = 0.5 * sigma * sigma
        return 0.5 * float(norm.cdf(z, loc=+half_sigma_sq, scale=sigma)) + \
               0.5 * float(norm.cdf(z, loc=-half_sigma_sq, scale=sigma))

    def _mixture_inv_brent(self, u: np.ndarray, sigma: float) -> np.ndarray:
        """Inverte F via Brent. Lento (~ms por ponto) mas mais preciso."""
        u = np.asarray(u, dtype=np.float64)
        R = float(self.config.brent_search_range)
        xtol = float(self.config.brent_xtol)
        out = np.empty_like(u)
        for i, ui in enumerate(u):
            ui_clip = float(min(max(ui, _EPS), 1.0 - _EPS))
            try:
                out[i] = brentq(
                    lambda z: self._mixture_cdf_scalar(z, sigma) - ui_clip,
                    -R, R, xtol=xtol, maxiter=200,
                )
            except ValueError:
                # bracket não engloba o zero → satura no extremo
                out[i] = -R if ui_clip < 0.5 else R
        return out

    # =========================================================================
    # CALIBRADOR INICIAL (passos 2-3 do paper)
    # =========================================================================

    def _apply_initial_calibrator(
        self,
        gen: np.ndarray,
        imp: np.ndarray,
        logger_callback: Callable[[str], None],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica calibrador inicial e retorna llr_gen, llr_imp.

        NONE   → assume scores já são ln(Λ); identidade.
        LOGREG → ajusta logistic regression univariada (1D); retorna decision_function.
        KDE    → ajusta dois KDEs com bandwidth Silverman (ou config); retorna log(ratio).
        """
        kind = self.config.initial_calibrator

        if kind == BiGaussInitialCalibrator.NONE:
            return gen.astype(np.float64), imp.astype(np.float64)

        if kind == BiGaussInitialCalibrator.LOGREG:
            # LogReg simples sem GridSearch (config.initial_logreg_C).
            from sklearn.linear_model import LogisticRegression
            logger_callback(f"[{self.name}] Calibrador inicial: LogReg")
            x = np.concatenate([gen, imp]).reshape(-1, 1)
            y = np.concatenate([np.ones(gen.size), np.zeros(imp.size)]).astype(np.int32)
            lr = LogisticRegression(
                C=float(self.config.initial_logreg_C),
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
            )
            lr.fit(x, y)
            # decision_function = log-odds posteriori; com class_weight=balanced
            # equivale a log-LR sob prior 0.5 (Brümmer).
            llr_gen = lr.decision_function(gen.reshape(-1, 1)).astype(np.float64)
            llr_imp = lr.decision_function(imp.reshape(-1, 1)).astype(np.float64)
            return llr_gen, llr_imp

        if kind == BiGaussInitialCalibrator.KDE:
            # KDE simples sobre os scores brutos (não Fisher-transformados),
            # bandwidth via Silverman se config.initial_kde_bandwidth == 0.
            #
            # Subsampling: sklearn.KernelDensity é O(N²) no pior caso (atol=0
            # default). Com 40k+ impostores o fit + score_samples leva minutos
            # sem feedback. O resultado é usado apenas para estimar Cllr que
            # alimenta σ² — uma amostra de 10k já é estatisticamente farto.
            from sklearn.neighbors import KernelDensity
            logger_callback(f"[{self.name}] Calibrador inicial: KDE")

            max_per_class = int(self.config.initial_kde_max_samples_per_class)
            rng = np.random.default_rng(int(self.config.random_seed))

            def _maybe_subsample(arr: np.ndarray, label: str) -> np.ndarray:
                if max_per_class <= 0 or arr.size <= max_per_class:
                    return arr
                logger_callback(
                    f"[{self.name}]   • Subsampling {label}: "
                    f"{arr.size:,} → {max_per_class:,} (seed={self.config.random_seed})"
                )
                idx = rng.choice(arr.size, size=max_per_class, replace=False)
                return arr[idx]

            gen_fit = _maybe_subsample(gen, "genuínos")
            imp_fit = _maybe_subsample(imp, "impostores")

            bw_cfg = float(self.config.initial_kde_bandwidth)
            bw_gen = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(gen_fit)
            bw_imp = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(imp_fit)

            logger_callback(
                f"[{self.name}]   • Fitting KDE_gen "
                f"(N={gen_fit.size:,}, h={bw_gen:.4f})..."
            )
            kde_g = KernelDensity(kernel="gaussian", bandwidth=bw_gen).fit(
                gen_fit.reshape(-1, 1)
            )
            logger_callback(
                f"[{self.name}]   • Fitting KDE_imp "
                f"(N={imp_fit.size:,}, h={bw_imp:.4f})..."
            )
            kde_i = KernelDensity(kernel="gaussian", bandwidth=bw_imp).fit(
                imp_fit.reshape(-1, 1)
            )

            # Scoring: avalia no MESMO conjunto subsampled usado no fit.
            # Cllr é estimado sobre esse subsample — consistente e rápido.
            logger_callback(f"[{self.name}]   • Scoring KDEs em {gen_fit.size:,} gen + {imp_fit.size:,} imp...")
            log_p_g_on_gen = kde_g.score_samples(gen_fit.reshape(-1, 1))
            log_p_i_on_gen = kde_i.score_samples(gen_fit.reshape(-1, 1))
            log_p_g_on_imp = kde_g.score_samples(imp_fit.reshape(-1, 1))
            log_p_i_on_imp = kde_i.score_samples(imp_fit.reshape(-1, 1))
            llr_gen = (log_p_g_on_gen - log_p_i_on_gen).astype(np.float64)
            llr_imp = (log_p_g_on_imp - log_p_i_on_imp).astype(np.float64)
            return llr_gen, llr_imp

        raise ValueError(f"initial_calibrator desconhecido: {kind}")

    @staticmethod
    def _silverman_bandwidth(x: np.ndarray) -> float:
        """Bandwidth de Silverman (rule-of-thumb). Robusto pra menor que IQR."""
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        if n < 2:
            return 1.0
        sigma = float(np.std(x, ddof=1))
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        spread = min(sigma, iqr / 1.349) if iqr > 0 else sigma
        if spread <= 0:
            spread = 1.0
        bw = 0.9 * spread * (n ** (-1.0 / 5.0))
        return max(bw, 1e-6)

    # =========================================================================
    # AUX
    # =========================================================================

    def _normalize_inputs(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int32)
        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            scores_arr = scores_arr.reshape(-1)
        elif scores_arr.ndim != 1:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")
        if labels_arr.ndim != 1:
            raise ValueError("labels deve ser vetor 1D.")
        if scores_arr.shape[0] != labels_arr.shape[0]:
            raise ValueError("scores e labels devem ter o mesmo número de amostras.")
        unique = set(np.unique(labels_arr).tolist())
        if not unique.issubset({0, 1}):
            raise ValueError("labels deve conter apenas 0 e 1.")
        return scores_arr, labels_arr

    def _normalize_scores_only(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            scores_arr = scores_arr.reshape(-1)
        elif scores_arr.ndim != 1:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")
        return scores_arr

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.name}] Modelo não treinado. Chame fit() antes."
            )
