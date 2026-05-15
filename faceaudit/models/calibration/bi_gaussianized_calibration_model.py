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

import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import norm

from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
    BiGaussGxMethod,
    BiGaussianizedCalibrationModelConfig,
    BiGaussInitialCalibrator,
    BiGaussInversionMethod,
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

    MODEL_VERSION = 1

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
            f"initial={self.config.initial_calibrator.value})..."
        )

        # ── Passos 2 e 3: calibrador inicial + métrica ─────────────────────────
        cllr_value: Optional[float] = None
        eer_value: Optional[float] = None

        if self.config.target_metric == BiGaussTargetMetric.CLLR:
            llr_gen, llr_imp = self._apply_initial_calibrator(gen, imp, logger_callback)
            cllr_value = self._compute_cllr(llr_gen, llr_imp)
            logger_callback(f"[{self.name}] Cllr no treino = {cllr_value:.6f}")
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
        self._build_g_cdf(scores_arr, labels_arr)

        # ── Pré-computa F⁻¹ da mistura se INTERP1D ─────────────────────────────
        if self.config.inversion_method == BiGaussInversionMethod.INTERP1D:
            self._mixture_inv_interp = self._build_mixture_inverse(sigma)
        else:
            self._mixture_inv_interp = None

        self.metadata = {
            "model_version": self.MODEL_VERSION,
            "sigma_target": float(sigma),
            "sigma_target_squared": float(sigma * sigma),
            "target_metric": self.config.target_metric.value,
            "initial_calibrator": self.config.initial_calibrator.value,
            "inversion_method": self.config.inversion_method.value,
            "gx_method": self.config.gx_method.value,
            "shrinkage_factor": float(self.config.sigma_shrinkage_factor),
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "cllr_at_fit": float(cllr_value) if cllr_value is not None else None,
            "eer_at_fit": float(eer_value) if eer_value is not None else None,
            "min_log_lr": float(self.config.min_log_lr),
            "max_log_lr": float(self.config.max_log_lr),
        }
        self._n_total = n_gen + n_imp
        self._is_fitted = True
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

        # G(x) → u ∈ (0, 1); clamp pra evitar 0/1 exatos.
        u = self._g_apply(score_vec)
        u = np.clip(u, _EPS, 1.0 - _EPS)

        # F⁻¹(u) → ln(Λ).
        if self.config.inversion_method == BiGaussInversionMethod.INTERP1D:
            log_lr = self._mixture_inv_interp(u).astype(np.float64)
        else:
            log_lr = self._mixture_inv_brent(u, self._sigma_target)

        return np.clip(log_lr, self.config.min_log_lr, self.config.max_log_lr)

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

    def _build_g_cdf(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Dispatcher: constrói G(x) conforme `config.gx_method`."""
        if self.config.gx_method == BiGaussGxMethod.KDE_SMOOTHED:
            self._build_smoothed_cdf(scores, labels)
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

    def _build_smoothed_cdf(self, scores: np.ndarray, labels: np.ndarray) -> None:
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

        # G(x) = 0.5·F̂_g(x) + 0.5·F̂_i(x) onde F̂_c = (1/N_c) Σ Φ((x - x_ci)/h_c).
        # Broadcasting é O(N_grid · N_c) — chunkado pra evitar matrizes grandes.
        f_gen = self._kernel_cdf_grid(x_grid, gen, bw_gen)
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
            from sklearn.neighbors import KernelDensity
            logger_callback(f"[{self.name}] Calibrador inicial: KDE")
            bw_cfg = float(self.config.initial_kde_bandwidth)
            bw_gen = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(gen)
            bw_imp = bw_cfg if bw_cfg > 0 else self._silverman_bandwidth(imp)
            kde_g = KernelDensity(kernel="gaussian", bandwidth=bw_gen).fit(
                gen.reshape(-1, 1)
            )
            kde_i = KernelDensity(kernel="gaussian", bandwidth=bw_imp).fit(
                imp.reshape(-1, 1)
            )
            log_p_g_on_gen = kde_g.score_samples(gen.reshape(-1, 1))
            log_p_i_on_gen = kde_i.score_samples(gen.reshape(-1, 1))
            log_p_g_on_imp = kde_g.score_samples(imp.reshape(-1, 1))
            log_p_i_on_imp = kde_i.score_samples(imp.reshape(-1, 1))
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
