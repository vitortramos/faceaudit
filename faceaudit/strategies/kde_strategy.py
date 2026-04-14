# faceaudit/strategies/kde_strategy.py

import numpy as np
import pickle
from typing import Optional, Dict, Any, Tuple, Callable

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.interpolate import interp1d

# Imports do Projeto
from faceaudit.models.embedding_comparison_dataset import EmbeddingComparisonDataset
from faceaudit.protocols.lr_model_protocol import LRModelProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo, CaseResult
from faceaudit.models.kde_training_artifacts import KDETrainingArtifacts
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
from faceaudit.parameters.kde_parameters import KDEParameters
from faceaudit.services import ComparisonService
from faceaudit.services.comparison_mining_service import ComparisonMiningService

class KDEStrategy(LRModelProtocol):
    """
    Estratégia de cálculo de Likelihood Ratio (LR) baseada em Estimativa de Densidade por Kernel (KDE).

    Versão robusta integrada ao DTO de Mineração:
    - O Motor de Curadoria e as Configurações de Mineração são injetados. A estratégia não acessa disco.
    - Usa transformação de Fisher (arctanh) para reduzir viés de borda do domínio [-1, 1].
    - GridSearch adaptativo ao spread dos dados (std) e com expansão automática do grid se "bater no teto".
    - CV adaptativo para maior estabilidade.
    """

    # Limites para evitar overflow/underflow numérico
    _MAX_LOG_LR: float = 20.0
    _MIN_LOG_LR: float = -20.0

    # --- Robustez numérica / borda ---
    _FISHER_EPS: float = 1e-6  # evita arctanh(±1)
    _MIN_BW_ABS: float = 1e-3  # mínimo absoluto de bandwidth no espaço transformado
    _MAX_GRID_EXPANSIONS: int = 2  # quantas vezes pode expandir o grid se cair na borda

    def __init__(
        self,
        comparison_service: ComparisonService,
        mining_service: ComparisonMiningService, # <--- INJEÇÃO DE DEPENDÊNCIA AQUI
        parameters: KDEParameters,
        name: str = "KDE_Strategy"
    ):
        self._name = name
        self.comp_svc = comparison_service
        self.mining_svc = mining_service         # <--- SALVA O SERVIÇO NA INSTÂNCIA
        self.params = parameters

        # Estado interno do modelo
        self.kde_gen: Optional[KernelDensity] = None
        self.kde_imp: Optional[KernelDensity] = None

        # Interpolador final: recebe score RAW em [-1,1] e retorna LR
        self.lr_interpolator: Optional[interp1d] = None

        self.metadata: Dict[str, Any] = {}
        self.fitted = False

        # =====================================================================
        # PASSO 2: cache VOLÁTIL (não vai para o pickle do modelo)
        # =====================================================================
        self._train_gen_scores_raw_1d: Optional[np.ndarray] = None
        self._train_imp_scores_raw_1d: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_train_dataset(self) -> bool:
        return True

    @property
    def requires_calibration_dataset(self) -> bool:
        return True

    # =========================================================================
    # FIT
    # =========================================================================

    def fit(
        self,
        train_dataset: Optional[EmbeddingDataset],
        calibration_dataset: EmbeddingDataset,
        mining_config: ComparisonMiningConfig,   # <--- DTO DE MINERAÇÃO RECEBIDO PRONTO
        logger_callback: Callable[[str], None] = print
    ) -> None:
        if train_dataset is None:
            raise ValueError(f"[{self.name}] Erro: 'train_dataset' (Probe) é obrigatório.")

        logger_callback(
            f"[{self.name}] Iniciando Fit Matemático (Regra de Mineração: {mining_config.name}).\n"
            f"    -> Probe (M): {len(train_dataset)} amostras\n"
            f"    -> Reference (N): {len(calibration_dataset)} amostras"
        )

        # 1. Geração de Scores (M x N)
        ds_scores = self.comp_svc.process_comparison(
            probe_dataset=train_dataset,
            reference_dataset=calibration_dataset,
            comparison_name=f"KDE_Fit_{self.name}"
        )

        # 2. Seleção e Curadoria de Pares (Delegado ao Serviço de Mineração)
        logger_callback(f"[{self.name}] Aplicando Curadoria Forense (Motor: {mining_config.name})...")
        impostors_ds, genuines_ds = self.mining_svc.mine_pairs(ds_scores, mining_config)

        # 3. Extração de Arrays Numpy (RAW scores em [-1,1])
        gen_scores_raw = np.array([c.cosine_similarity for c in genuines_ds.items], dtype=np.float64).reshape(-1, 1)
        imp_scores_raw = np.array([c.cosine_similarity for c in impostors_ds.items], dtype=np.float64).reshape(-1, 1)

        # Validação Mínima Baseada no KDEParameters (Para evitar explosão matemática)
        if gen_scores_raw.size < self.params.genuines_min_count or imp_scores_raw.size < self.params.impostors_min_count:
            raise ValueError(
                f"[{self.name}] Dados insuficientes após mineração!\n"
                f"    -> Genuínos Finais: {gen_scores_raw.size} (mínimo exigido: {self.params.genuines_min_count})\n"
                f"    -> Impostores Finais: {imp_scores_raw.size} (mínimo exigido: {self.params.impostors_min_count})\n"
                f"Ajuste os filtros da regra '{mining_config.name}' ou verifique a compatibilidade dos datasets."
            )

        # 4. Downsample para modelo final (preserva distribuição da mineração, usa seed do DTO)
        limit_stored = int(self.params.max_stored_samples)

        def _smart_downsample(data: np.ndarray, label: str) -> np.ndarray:
            if len(data) > limit_stored:
                logger_callback(
                    f"[{self.name}] 📉 Reduzindo {label} de {len(data)} para {limit_stored} pontos "
                    f"(Amostragem Aleatória via Seed: {mining_config.seed})."
                )
                rng = np.random.default_rng(mining_config.seed)
                return rng.choice(data.flatten(), size=limit_stored, replace=False).reshape(-1, 1)
            return data

        gen_final_raw = _smart_downsample(gen_scores_raw, "Genuínos")
        imp_final_raw = _smart_downsample(imp_scores_raw, "Impostores")

        # --- LOG útil para diagnosticar ---
        self._log_score_stats(gen_final_raw, "Genuínos RAW", logger_callback)
        self._log_score_stats(imp_final_raw, "Impostores RAW", logger_callback)

        # 5. TRANSFORMAÇÃO ROBUSTA (Fisher z)
        gen_final = self._to_fisher_space(gen_final_raw)
        imp_final = self._to_fisher_space(imp_final_raw)

        self._log_score_stats(gen_final, "Genuínos Fisher(z)", logger_callback)
        self._log_score_stats(imp_final, "Impostores Fisher(z)", logger_callback)

        # 6. Otimização de Bandwidth (no espaço Fisher)
        bw_gen = float(self.params.bandwidth_genuines)
        bw_imp = float(self.params.bandwidth_impostors)

        if self.params.use_gridsearch:
            logger_callback(f"[{self.name}] Executando GridSearch robusto para otimizar Bandwidth (Fisher space)...")
            bw_gen = self._optimize_bandwidth(gen_final, "Genuines", logger_callback)
            bw_imp = self._optimize_bandwidth(imp_final, "Impostors", logger_callback)
        else:
            bw_gen = max(bw_gen, self._MIN_BW_ABS)
            bw_imp = max(bw_imp, self._MIN_BW_ABS)
            logger_callback(f"[{self.name}] Usando Bandwidth manual: Gen={bw_gen}, Imp={bw_imp}")

        # 7. Fit final do KDE (no espaço Fisher)
        kernel_name = self.params.kernel.value
        logger_callback(f"[{self.name}] Ajustando curvas finais (Kernel={kernel_name}, Fisher space)...")

        self.kde_gen = KernelDensity(kernel=kernel_name, bandwidth=bw_gen).fit(gen_final)
        self.kde_imp = KernelDensity(kernel=kernel_name, bandwidth=bw_imp).fit(imp_final)

        # 8. Geração do Interpolador FINAL em score RAW [-1, 1]
        self._regenerate_interpolator()

        # Metadados do modelo (persistidos no .pkl do modelo)
        self.metadata = {
            "mining_rule_name": mining_config.name, # <-- Registra qual regra gerou este modelo
            "kernel": kernel_name,
            "grid_search": bool(self.params.use_gridsearch),
            "transform": "fisher_arctanh",
            "bw_gen_fisher": float(bw_gen),
            "bw_imp_fisher": float(bw_imp),
            "n_genuines": int(gen_final_raw.size),
            "n_impostors": int(imp_final_raw.size),
            "fisher_eps": float(self._FISHER_EPS),
        }
        self.fitted = True

        # =====================================================================
        # PASSO 2: cache VOLÁTIL (para visualização) — EXATAMENTE o fit final
        # =====================================================================
        self._train_gen_scores_raw_1d = np.asarray(gen_final_raw, dtype=np.float64).reshape(-1).copy()
        self._train_imp_scores_raw_1d = np.asarray(imp_final_raw, dtype=np.float64).reshape(-1).copy()

        logger_callback(
            f"[{self.name}] ✅ Treino Concluído.\n"
            f"    -> Pares Finais: {gen_final_raw.size} Gen / {imp_final_raw.size} Imp\n"
            f"    -> Bandwidths (h) em Fisher(z): {bw_gen:.6f} / {bw_imp:.6f}"
        )

    # =========================================================================
    # EXPORT DOS ARTEFATOS
    # =========================================================================

    def has_training_artifacts_cache(self) -> bool:
        return self._train_gen_scores_raw_1d is not None and self._train_imp_scores_raw_1d is not None

    def export_training_artifacts(self) -> KDETrainingArtifacts:
        if not self.has_training_artifacts_cache():
            raise RuntimeError(
                f"[{self.name}] Training artifacts indisponíveis. "
                f"Execute fit() nesta instância antes de exportar."
            )

        return KDETrainingArtifacts.from_arrays(
            genuines_scores_raw=self._train_gen_scores_raw_1d,
            impostors_scores_raw=self._train_imp_scores_raw_1d,
            metadata=self.metadata
        )

    def clear_training_artifacts_cache(self) -> None:
        self._train_gen_scores_raw_1d = None
        self._train_imp_scores_raw_1d = None

    # =========================================================================
    # PREDICT / COMPARE
    # =========================================================================

    def predict(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> float:
        if not self.fitted or self.lr_interpolator is None:
            raise RuntimeError(f"[{self.name}] Estratégia não treinada.")

        # Score RAW (cosine)
        score = float(np.dot(probe.embedding, reference.embedding))

        # Lookup LR(score_raw) na tabela interpolada
        try:
            lr = float(self.lr_interpolator(score))
        except ValueError:
            xs = self.lr_interpolator.x
            ys = self.lr_interpolator.y
            lr = float(ys[0]) if score < xs[0] else float(ys[-1])

        # Clamp de segurança (LR mínimo)
        return max(lr, float(np.exp(self._MIN_LOG_LR)))
    
    def infer(self, comparison_dataset: EmbeddingComparisonDataset) -> np.ndarray:
        """
        Calcula LRs em lote para um dataset de confrontos.
        Aproveita as similaridades já calculadas para máxima performance.
        """
        if not self.fitted or self.lr_interpolator is None:
            raise RuntimeError(f"[{self.name}] Estratégia não treinada.")

        scores = np.array([c.cosine_similarity for c in comparison_dataset.items], dtype=np.float64)
        lrs = self.lr_interpolator(scores)
        lrs_clipped = np.clip(lrs, np.exp(self._MIN_LOG_LR), np.exp(self._MAX_LOG_LR))
        
        # Interpolação vetorizada (Scipy interp1d aceita arrays e é ultra rápida)
        lrs = self.lr_interpolator(scores)
        
        # Clamp de segurança
        return np.log(lrs_clipped)

    def compare(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> CaseResult:
        lr = self.predict(probe, reference)
        raw_score = float(np.dot(probe.embedding, reference.embedding))
        verbal = self._get_verbal_support(lr)

        return CaseResult(
            strategy_name=self.name,
            probe=probe,
            reference=reference,
            score=raw_score,
            lr=float(lr),
            verbal_support=verbal,
            ground_truth=None
        )

    # =========================================================================
    # SAVE / LOAD
    # =========================================================================

    def save(self, path: str) -> None:
        state = {
            "kde_gen": self.kde_gen,
            "kde_imp": self.kde_imp,
            "fitted": self.fitted,
            "metadata": self.metadata,
            "params_dict": self.params.to_dict(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"[{self.name}] ✅ Modelo salvo em {path}.")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.kde_gen = state["kde_gen"]
            self.kde_imp = state["kde_imp"]
            self.fitted = state.get("fitted", True)
            self.metadata = state.get("metadata", {})

            self.clear_training_artifacts_cache()

            if self.fitted:
                print(f"[{self.name}] Recriando interpolador...")
                self._regenerate_interpolator()

        print(f"[{self.name}] ✅ Modelo carregado.")

    # =========================================================================
    # HELPERS INTERNOS MATEMÁTICOS
    # =========================================================================

    def _to_fisher_space(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.float64)
        x = np.clip(x, -1.0 + self._FISHER_EPS, 1.0 - self._FISHER_EPS)
        return np.arctanh(x)

    def _regenerate_interpolator(self) -> None:
        if self.kde_gen is None or self.kde_imp is None:
            raise RuntimeError(f"[{self.name}] KDEs não inicializados.")

        x_raw = np.linspace(-1.0, 1.0, 10000, dtype=np.float64).reshape(-1, 1)
        x_fisher = self._to_fisher_space(x_raw)

        log_pdf_gen = self.kde_gen.score_samples(x_fisher)
        log_pdf_imp = self.kde_imp.score_samples(x_fisher)

        log_lr_curve = log_pdf_gen - log_pdf_imp
        log_lr_curve = np.clip(log_lr_curve, self._MIN_LOG_LR, self._MAX_LOG_LR)
        lr_curve = np.exp(log_lr_curve)

        self.lr_interpolator = interp1d(
            x_raw.flatten(),
            lr_curve,
            kind="linear",
            bounds_error=False,
            fill_value=(float(lr_curve[0]), float(lr_curve[-1])),
            assume_sorted=True
        )

    def _optimize_bandwidth(
        self,
        data: np.ndarray,
        label: str,
        logger_callback: Callable[[str], None] = print
    ) -> float:
        data = np.asarray(data, dtype=np.float64).reshape(-1, 1)

        max_samples_param = int(self.params.max_samples_for_gridsearch)
        max_samples = int(min(max_samples_param, 10_000))
        sample = data

        if len(sample) > max_samples:
            logger_callback(
                f"[{self.name}] ⚠️ Amostrando {max_samples} pontos para GridSearch ({label}) "
                f"({len(sample)} -> {max_samples})."
            )
            rng = np.random.default_rng(42)
            sample = rng.choice(sample.flatten(), size=max_samples, replace=False).reshape(-1, 1)

        n = len(sample)
        if n < 30:
            std = float(np.std(sample))
            h = max(std * 0.2, self._MIN_BW_ABS)
            logger_callback(f"[{self.name}] ⚠️ Poucas amostras para CV ({label}, n={n}). Usando h≈{h:.6f} (heurístico).")
            return h

        cv = 5 if n >= 200 else 3
        kernel_val = self.params.kernel.value
        std = float(np.std(sample))
        if not np.isfinite(std) or std <= 0.0:
            std = 1.0

        base_min = max(self._MIN_BW_ABS, std * 0.02)
        base_max = max(base_min * 50.0, std * 2.0)
        bandwidths = np.logspace(np.log10(base_min), np.log10(base_max), 30)

        def run_grid(bws: np.ndarray) -> Tuple[float, float, float]:
            grid = GridSearchCV(
                KernelDensity(kernel=kernel_val),
                {"bandwidth": bws},
                cv=cv,
                n_jobs=-1
            )
            grid.fit(sample)
            best = float(grid.best_params_["bandwidth"])
            min_bw = float(np.min(bws))
            max_bw = float(np.max(bws))
            return best, min_bw, max_bw

        best, min_bw, max_bw = run_grid(bandwidths)

        expansions = 0
        while expansions < self._MAX_GRID_EXPANSIONS:
            hit_low = np.isclose(best, min_bw, rtol=1e-6, atol=0.0)
            hit_high = np.isclose(best, max_bw, rtol=1e-6, atol=0.0)

            if not (hit_low or hit_high):
                break

            expansions += 1
            if hit_high:
                new_min = min_bw
                new_max = max_bw * 10.0
                logger_callback(
                    f"[{self.name}] ⚠️ GridSearch ({label}) bateu no TETO (best={best:.6f}). "
                    f"Expandindo grid: max {max_bw:.6f} -> {new_max:.6f} (tentativa {expansions})."
                )
            else:
                new_min = max(self._MIN_BW_ABS, min_bw / 10.0)
                new_max = max_bw
                logger_callback(
                    f"[{self.name}] ⚠️ GridSearch ({label}) bateu no PISO (best={best:.6f}). "
                    f"Expandindo grid: min {min_bw:.6f} -> {new_min:.6f} (tentativa {expansions})."
                )

            bandwidths = np.logspace(np.log10(new_min), np.log10(new_max), 30)
            best, min_bw, max_bw = run_grid(bandwidths)

        best_safe = max(best, self._MIN_BW_ABS)
        logger_callback(
            f"[{self.name}] Melhor h para {label} (Fisher space): {best_safe:.6f} "
            f"(cv={cv}, n={n}, std={std:.6f}, grid=[{min_bw:.6f}..{max_bw:.6f}])"
        )
        return best_safe

    def _log_score_stats(self, arr: np.ndarray, label: str, logger_callback: Callable[[str], None]) -> None:
        x = np.asarray(arr, dtype=np.float64).reshape(-1)
        if x.size == 0:
            logger_callback(f"[{self.name}] {label}: vazio.")
            return
        mn = float(np.min(x))
        mx = float(np.max(x))
        mu = float(np.mean(x))
        sd = float(np.std(x))
        logger_callback(f"[{self.name}] {label}: n={x.size} mean={mu:.6f} std={sd:.6f} min={mn:.6f} max={mx:.6f}")

    def _get_verbal_support(self, lr: float) -> str:
        if lr < 1:
            return "Exclusão"
        if lr < 10:
            return "Suporte Fraco"
        if lr < 100:
            return "Suporte Moderado"
        if lr < 1000:
            return "Suporte Moderadamente Forte"
        if lr < 10000:
            return "Suporte Forte"
        return "Suporte Muito Forte"