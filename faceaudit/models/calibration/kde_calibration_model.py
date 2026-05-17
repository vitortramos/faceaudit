# faceaudit/models/calibration/kde_calibration_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from faceaudit.models.calibration.kde_calibration_model_config import (
    KDECalibrationModelConfig,
)
from faceaudit.protocols.calibration_model_protocol import CalibrationModelProtocol


class KDECalibrationModel(CalibrationModelProtocol):
    """
    Modelo de calibração baseado em KDE.

    Responsabilidades:
    - treinar dois KDEs separados:
        - genuínos (label=1 / Hp)
        - impostores (label=0 / Hd)
    - converter score bruto em probabilidades calibradas
    - converter score bruto em log-LR calibrado
    - persistir e restaurar seu estado

    Convenção de labels:
    - 1 = genuíno / Hp
    - 0 = impostor / Hd

    Pré-processamento:
    Aplica z-score normalization (`(x - μ) / σ`) com μ/σ aprendidos em
    `fit()` a partir de todos os scores de treino concatenados. Funciona
    pra QUALQUER escala de score (cosseno em [-1, 1], PLDA em ranges
    arbitrários, etc). A transformação é determinística e persistida
    junto com o modelo — `calibrate_scores` em validação/casework aplica
    exatamente a mesma normalização.
    """

    MODEL_VERSION = 2

    def __init__(
        self,
        config: KDECalibrationModelConfig | None = None,
        name: str = "KDE Calibration Model",
    ):
        self.config = config or KDECalibrationModelConfig()
        self._name = name
        self._is_fitted = False

        self.kde_genuine: Optional[KernelDensity] = None
        self.kde_impostor: Optional[KernelDensity] = None
        self.lr_interpolator: Optional[interp1d] = None

        # Parâmetros da z-score normalization aprendidos em fit().
        # Aplicados a TODOS os scores antes do KDE (treino + inferência).
        self._score_mean: Optional[float] = None
        self._score_std: Optional[float] = None
        # Range coberto pela curva interpoladora no espaço raw. Definido
        # em fit() pra cobrir os dados de treino com folga sem desperdiçar
        # pontos na grade em regiões vazias.
        self._raw_min: Optional[float] = None
        self._raw_max: Optional[float] = None

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

    # =========================================================================
    # TREINO
    # =========================================================================

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Treina o calibrador KDE a partir de scores brutos e labels binários.
        """
        scores_arr, labels_arr = self._normalize_inputs(scores, labels)

        genuine_scores_raw = scores_arr[labels_arr == 1]
        impostor_scores_raw = scores_arr[labels_arr == 0]

        if genuine_scores_raw.size == 0 or impostor_scores_raw.size == 0:
            raise ValueError(
                "O treino do calibrador KDE requer amostras genuínas e impostoras."
            )

        logger_callback(
            f"[{self.name}] 🧠 Treinando calibrador KDE "
            f"(N={scores_arr.size} | Gen={genuine_scores_raw.size} | "
            f"Imp={impostor_scores_raw.size} | kernel={self.config.kernel})..."
        )

        genuine_scores_raw = self._downsample_if_needed(
            genuine_scores_raw, "Genuínos", logger_callback
        )
        impostor_scores_raw = self._downsample_if_needed(
            impostor_scores_raw, "Impostores", logger_callback
        )

        self._log_score_stats(genuine_scores_raw, "Genuínos RAW", logger_callback)
        self._log_score_stats(impostor_scores_raw, "Impostores RAW", logger_callback)

        # Z-score normalization aprendida a partir de TODOS os scores de
        # treino (gen + imp concatenados). Funciona pra qualquer escala
        # de score sem assumir range [-1, 1] como o Fisher transform
        # antigo exigia. Persistida no save() pra que validação/casework
        # apliquem exatamente a mesma transformação.
        all_train = np.concatenate([genuine_scores_raw, impostor_scores_raw])
        self._score_mean = float(np.mean(all_train))
        std = float(np.std(all_train))
        # Guarda mínimo positivo pra evitar divisão por zero em datasets
        # degenerados (ex: todos scores iguais). Cenário improvável mas
        # protege contra NaN/Inf na inferência.
        self._score_std = max(std, 1e-12)

        # Cobertura da curva interpoladora no espaço raw: usa os extremos
        # dos dados de treino com margem de ±5σ, garantindo que scores
        # de validação um pouco fora do range visto ainda caiam dentro
        # de pontos interpolados (em vez de bordas clampadas).
        margin = 5.0 * self._score_std
        self._raw_min = float(all_train.min()) - margin
        self._raw_max = float(all_train.max()) + margin

        genuine_scores_z = self._to_normalized_space(genuine_scores_raw)
        impostor_scores_z = self._to_normalized_space(impostor_scores_raw)

        self._log_score_stats(genuine_scores_z, "Genuínos z-score", logger_callback)
        self._log_score_stats(impostor_scores_z, "Impostores z-score", logger_callback)

        bw_gen = float(self.config.bandwidth_genuine)
        bw_imp = float(self.config.bandwidth_impostor)

        if self.config.use_gridsearch:
            logger_callback(f"[{self.name}] Executando GridSearch para bandwidth no espaço z-score...")
            bw_gen = self._optimize_bandwidth(
                genuine_scores_z, "Genuínos", logger_callback
            )
            bw_imp = self._optimize_bandwidth(
                impostor_scores_z, "Impostores", logger_callback
            )
        else:
            bw_gen = max(bw_gen, self.config.min_bandwidth_abs)
            bw_imp = max(bw_imp, self.config.min_bandwidth_abs)
            logger_callback(
                f"[{self.name}] Usando bandwidth manual: Gen={bw_gen:.6f}, Imp={bw_imp:.6f}"
            )

        self.kde_genuine = KernelDensity(
            kernel=self.config.kernel,
            bandwidth=bw_gen,
        ).fit(genuine_scores_z.reshape(-1, 1))

        self.kde_impostor = KernelDensity(
            kernel=self.config.kernel,
            bandwidth=bw_imp,
        ).fit(impostor_scores_z.reshape(-1, 1))

        self._regenerate_interpolator()

        self.metadata = {
            "kernel": self.config.kernel,
            "use_gridsearch": bool(self.config.use_gridsearch),
            "transform": "z_score",
            "bw_genuine_z": float(bw_gen),
            "bw_impostor_z": float(bw_imp),
            "n_genuine": int(genuine_scores_raw.size),
            "n_impostor": int(impostor_scores_raw.size),
            "score_mean": float(self._score_mean),
            "score_std": float(self._score_std),
            "raw_min": float(self._raw_min),
            "raw_max": float(self._raw_max),
            "min_log_lr": float(self.config.min_log_lr),
            "max_log_lr": float(self.config.max_log_lr),
        }

        self._is_fitted = True

        logger_callback(
            f"[{self.name}] ✅ Treinamento concluído "
            f"(μ={self._score_mean:.6f}, σ={self._score_std:.6f}, "
            f"h_gen={bw_gen:.6f}, h_imp={bw_imp:.6f})."
        )

    # =========================================================================
    # INFERÊNCIA
    # =========================================================================

    def predict_proba(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna probabilidades calibradas [Hd, Hp] a partir do LR estimado.
        """
        self._ensure_fitted()

        score_vec = self._normalize_scores_only(scores)
        llr = self.calibrate_scores(score_vec)
        lr = np.exp(llr)

        # Conversão simples LR -> probabilidades assumindo odds relativas:
        # p_hp = lr / (1 + lr), p_hd = 1 / (1 + lr)
        p_hp = lr / (1.0 + lr)
        p_hd = 1.0 / (1.0 + lr)

        probs = np.column_stack([p_hd, p_hp])
        return np.asarray(probs, dtype=np.float64)

    def calibrate_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna log-LR calibrado.
        """
        self._ensure_fitted()

        score_vec = self._normalize_scores_only(scores)

        try:
            lr = np.asarray(self.lr_interpolator(score_vec), dtype=np.float64)
        except ValueError:
            xs = self.lr_interpolator.x
            ys = self.lr_interpolator.y
            lr = np.array(
                [float(ys[0]) if s < xs[0] else float(ys[-1]) for s in score_vec],
                dtype=np.float64,
            )

        lr = np.clip(
            lr,
            np.exp(self.config.min_log_lr),
            np.exp(self.config.max_log_lr),
        )
        return np.log(lr)

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Persiste o estado do calibrador em disco.
        """
        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "kde_calibration_model",
            "name": self._name,
            "config": self.config.to_dict(),
            "kde_genuine": self.kde_genuine,
            "kde_impostor": self.kde_impostor,
            "metadata": self.metadata,
            "is_fitted": self._is_fitted,
            # Z-score normalization aprendida em fit() — sem isso o
            # calibrador em inferência (validação/casework) não consegue
            # reproduzir a mesma transformação que viu no treino.
            "score_mean": self._score_mean,
            "score_std": self._score_std,
            "raw_min": self._raw_min,
            "raw_max": self._raw_max,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """
        Restaura o estado do calibrador a partir do disco.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo do calibrador não encontrado: {path_obj}")

        with open(path_obj, "rb") as f:
            payload = pickle.load(f)

        self._name = payload.get("name", self._name)
        self.config = KDECalibrationModelConfig.from_dict(payload.get("config", {}))
        self.kde_genuine = payload.get("kde_genuine")
        self.kde_impostor = payload.get("kde_impostor")
        self.metadata = payload.get("metadata", {})
        self._is_fitted = bool(payload.get("is_fitted", False))

        # Restaura params da z-score normalization. Calibradores legados
        # (MODEL_VERSION=1, antes do switch pra z-score) não têm esses
        # campos — `_regenerate_interpolator` vai falhar com mensagem
        # explícita e usuário re-treina.
        self._score_mean = payload.get("score_mean")
        self._score_std = payload.get("score_std")
        self._raw_min = payload.get("raw_min")
        self._raw_max = payload.get("raw_max")

        if self._is_fitted:
            self._regenerate_interpolator()

    # =========================================================================
    # AUXILIARES DE ENTRADA
    # =========================================================================

    def _normalize_inputs(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int32)

        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            score_vec = scores_arr.reshape(-1)
        elif scores_arr.ndim == 1:
            score_vec = scores_arr
        else:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")

        if labels_arr.ndim != 1:
            raise ValueError("labels deve ser vetor 1D.")

        if score_vec.shape[0] != labels_arr.shape[0]:
            raise ValueError("scores e labels devem ter o mesmo número de amostras.")

        unique_labels = set(np.unique(labels_arr).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError("labels deve conter apenas 0 e 1.")

        return score_vec, labels_arr

    def _normalize_scores_only(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float64)

        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            return scores_arr.reshape(-1)
        if scores_arr.ndim == 1:
            return scores_arr

        raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")

    # =========================================================================
    # HELPERS MATEMÁTICOS
    # =========================================================================

    def _to_normalized_space(self, x_raw: np.ndarray) -> np.ndarray:
        """Aplica z-score normalization: `(x - μ) / σ` com μ/σ aprendidos
        em fit() a partir de todos os scores de treino concatenados.
        """
        if self._score_mean is None or self._score_std is None:
            raise RuntimeError(
                "Normalização não foi aprendida — chame fit() antes."
            )
        x = np.asarray(x_raw, dtype=np.float64)
        return (x - self._score_mean) / self._score_std

    def _regenerate_interpolator(self) -> None:
        if self.kde_genuine is None or self.kde_impostor is None:
            raise RuntimeError("KDEs não inicializados.")
        if self._raw_min is None or self._raw_max is None:
            raise RuntimeError(
                "Range raw não foi computado — chame fit() antes."
            )

        # Grade de pontos no espaço raw, cobrindo [min_train - 5σ,
        # max_train + 5σ]. Scores de validação um pouco fora do range
        # visto ainda caem dentro da curva interpolada — sem isso, eles
        # seriam clampados nas bordas, achatando log-LRs extremos.
        x_raw = np.linspace(
            self._raw_min,
            self._raw_max,
            self.config.interpolation_points,
            dtype=np.float64,
        ).reshape(-1, 1)
        x_z = self._to_normalized_space(x_raw)

        log_pdf_gen = self.kde_genuine.score_samples(x_z)
        log_pdf_imp = self.kde_impostor.score_samples(x_z)

        log_lr_curve = log_pdf_gen - log_pdf_imp
        log_lr_curve = np.clip(
            log_lr_curve,
            self.config.min_log_lr,
            self.config.max_log_lr,
        )
        lr_curve = np.exp(log_lr_curve)

        self.lr_interpolator = interp1d(
            x_raw.flatten(),
            lr_curve,
            kind="linear",
            bounds_error=False,
            fill_value=(float(lr_curve[0]), float(lr_curve[-1])),
            assume_sorted=True,
        )

    def _downsample_if_needed(
        self,
        data: np.ndarray,
        label: str,
        logger_callback: Callable[[str], None],
    ) -> np.ndarray:
        limit = int(self.config.max_stored_samples)
        data = np.asarray(data, dtype=np.float64).reshape(-1)

        if data.size <= limit:
            return data

        logger_callback(
            f"[{self.name}] 📉 Reduzindo {label} de {data.size} para {limit} pontos "
            f"(seed={self.config.random_seed})."
        )
        rng = np.random.default_rng(self.config.random_seed)
        sampled = rng.choice(data, size=limit, replace=False)
        return np.asarray(sampled, dtype=np.float64)

    def _optimize_bandwidth(
        self,
        data: np.ndarray,
        label: str,
        logger_callback: Callable[[str], None],
    ) -> float:
        sample = np.asarray(data, dtype=np.float64).reshape(-1, 1)

        max_samples = int(min(self.config.max_samples_for_gridsearch, 10000))
        if len(sample) > max_samples:
            logger_callback(
                f"[{self.name}] ⚠️ Amostrando {max_samples} pontos para GridSearch "
                f"({label}) ({len(sample)} -> {max_samples})."
            )
            rng = np.random.default_rng(self.config.random_seed)
            sample = rng.choice(sample.flatten(), size=max_samples, replace=False).reshape(-1, 1)

        n = len(sample)
        if n < 30:
            std = float(np.std(sample))
            h = max(std * 0.2, self.config.min_bandwidth_abs)
            logger_callback(
                f"[{self.name}] ⚠️ Poucas amostras para CV ({label}, n={n}). "
                f"Usando h≈{h:.6f} (heurístico)."
            )
            return h

        cv = 5 if n >= 200 else 3
        std = float(np.std(sample))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0

        base_min = max(self.config.min_bandwidth_abs, std * 0.02)
        base_max = max(base_min * 50.0, std * 2.0)
        bandwidths = np.logspace(np.log10(base_min), np.log10(base_max), 30)

        def run_grid(bws: np.ndarray) -> Tuple[float, float, float]:
            grid = GridSearchCV(
                KernelDensity(kernel=self.config.kernel),
                {"bandwidth": bws},
                cv=cv,
                n_jobs=-1,
            )
            grid.fit(sample)
            best = float(grid.best_params_["bandwidth"])
            return best, float(np.min(bws)), float(np.max(bws))

        best, min_bw, max_bw = run_grid(bandwidths)

        expansions = 0
        while expansions < self.config.max_grid_expansions:
            hit_low = np.isclose(best, min_bw, rtol=1e-6, atol=0.0)
            hit_high = np.isclose(best, max_bw, rtol=1e-6, atol=0.0)

            if not (hit_low or hit_high):
                break

            expansions += 1

            if hit_high:
                new_min = min_bw
                new_max = max_bw * 10.0
                logger_callback(
                    f"[{self.name}] ⚠️ GridSearch ({label}) bateu no teto "
                    f"(best={best:.6f}). Expandindo para [{new_min:.6f}, {new_max:.6f}] "
                    f"(tentativa {expansions})."
                )
            else:
                new_min = max(self.config.min_bandwidth_abs, min_bw / 10.0)
                new_max = max_bw
                logger_callback(
                    f"[{self.name}] ⚠️ GridSearch ({label}) bateu no piso "
                    f"(best={best:.6f}). Expandindo para [{new_min:.6f}, {new_max:.6f}] "
                    f"(tentativa {expansions})."
                )

            bandwidths = np.logspace(np.log10(new_min), np.log10(new_max), 30)
            best, min_bw, max_bw = run_grid(bandwidths)

        best_safe = max(best, self.config.min_bandwidth_abs)

        logger_callback(
            f"[{self.name}] Melhor h para {label}: {best_safe:.6f} "
            f"(cv={cv}, n={n}, std={std:.6f}, grid=[{min_bw:.6f}..{max_bw:.6f}])."
        )
        return best_safe

    def _log_score_stats(
        self,
        arr: np.ndarray,
        label: str,
        logger_callback: Callable[[str], None],
    ) -> None:
        x = np.asarray(arr, dtype=np.float64).reshape(-1)
        if x.size == 0:
            logger_callback(f"[{self.name}] {label}: vazio.")
            return

        mn = float(np.min(x))
        mx = float(np.max(x))
        mu = float(np.mean(x))
        sd = float(np.std(x))

        logger_callback(
            f"[{self.name}] {label}: n={x.size} mean={mu:.6f} "
            f"std={sd:.6f} min={mn:.6f} max={mx:.6f}"
        )

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self.kde_genuine is None or self.kde_impostor is None:
            raise RuntimeError(
                "Modelo KDECalibrationModel não treinado. Execute fit() primeiro."
            )