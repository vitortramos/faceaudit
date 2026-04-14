import numpy as np
import pickle
from typing import Optional, Tuple, Callable, Dict, List
from collections import defaultdict
from itertools import combinations

from sklearn.linear_model import LogisticRegression

# SpeechBrain
from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB, Ndx, fast_PLDA_scoring

# Seu projeto
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.models import EmbeddingDataset, EmbeddingInfo, CaseResult
from faceaudit.protocols.lr_model_protocol import LRModelProtocol


class PLDASnormStrategy(LRModelProtocol):
    """
    PLDA (SpeechBrain) + S-norm (Z/T/S-norm) + calibração por Logistic Regression.
    PIPELINE RECOMENDADO: 
        L2 → centering → L2
        L2 → centering → PCA (ex: 256) → L2

    Características:
    - Implementa LRModelProtocol diretamente (estratégia separada; não altera PLDAStrategy existente).
    - Treino: usa train_dataset para treinar PLDA.
    - Cohort (S-norm): construído a partir do train_dataset (para não “roubar” amostras do calibration_dataset).
    - Calibração: usa calibration_dataset (tipicamente pequeno no seu controller) para gerar pares e treinar LogReg.
    - Opção B: pré-cálculo das estatísticas Z/T por embedding do calibration_dataset (usando cohort do treino),
      permitindo S-norm eficiente em milhares de pares e em hard negative mining.
    - Inferência 1:1: aplica S-norm on-the-fly (reference vs cohort e cohort vs probe), e então LogReg → LR.

    Convenção:
    - predict(probe, reference) internamente calcula score PLDA como: reference=enroll, probe=test.
    """

    MODEL_VERSION = 1

    def __init__(
        self,
        name: str = "PLDA_LLR_Snorm_LogReg",
        em_iterations: int = 20,
        plda_mode: PLDAMode = PLDAMode.FULL_RANK,
        latent_dim: int = 128,

        # --- Calibração ---
        logreg_C: float = 1.0,
        min_impostors: int = 10000,
        impostor_hard_fraction: float = 0.30,   # 0..1
        hard_candidate_multiplier: int = 5,
        max_genuine_pairs_per_id: Optional[int] = 500,

        # --- Performance (pares esparsos) ---
        score_block_size: int = 512,

        # --- S-norm ---
        cohort_size: int = 2000,
        snorm_eps: float = 1e-6,
        stats_block_size: int = 256,            # chunk para pré-cálculo de stats com cohort

        # --- Reprodutibilidade ---
        seed: Optional[int] = 42,

        # --- Telemetria ---
        logger: Optional[Callable[[str], None]] = None,
        compute_train_diagnostics: bool = False,
        diagnostics_compute_cllr: bool = True,
    ):
        self._name = name
        self.em_iterations = int(em_iterations)

        self.plda_mode = plda_mode if isinstance(plda_mode, PLDAMode) else PLDAMode(plda_mode)

        self.latent_dim = int(latent_dim)
        if self.latent_dim < 1:
            raise ValueError("latent_dim deve ser >= 1.")

        self.min_impostors = int(min_impostors)
        if self.min_impostors < 0:
            raise ValueError("min_impostors deve ser >= 0.")

        self.impostor_hard_fraction = float(impostor_hard_fraction)
        if not (0.0 <= self.impostor_hard_fraction <= 1.0):
            raise ValueError("impostor_hard_fraction deve estar em [0, 1].")

        self.hard_candidate_multiplier = int(hard_candidate_multiplier)
        if self.hard_candidate_multiplier < 1:
            raise ValueError("hard_candidate_multiplier deve ser >= 1.")

        self.max_genuine_pairs_per_id = None if max_genuine_pairs_per_id is None else int(max_genuine_pairs_per_id)
        if self.max_genuine_pairs_per_id is not None and self.max_genuine_pairs_per_id < 1:
            raise ValueError("max_genuine_pairs_per_id deve ser >= 1 ou None.")

        self.score_block_size = int(score_block_size)
        if self.score_block_size < 32:
            raise ValueError("score_block_size muito pequeno. Use >= 32 (recomendado 256/512/1024).")

        # S-norm params
        self.cohort_size = int(cohort_size)
        if self.cohort_size < 50:
            raise ValueError("cohort_size muito pequeno. Use algo como 500..5000.")
        self.snorm_eps = float(snorm_eps)
        self.stats_block_size = int(stats_block_size)
        if self.stats_block_size < 16:
            raise ValueError("stats_block_size muito pequeno. Use >= 16 (recomendado 128/256).")

        # RNG / logger
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._log = logger or print

        self.compute_train_diagnostics = bool(compute_train_diagnostics)
        self.diagnostics_compute_cllr = bool(diagnostics_compute_cllr)

        # Backend PLDA + calibrador
        self.plda: Optional[PLDA] = None
        self.logreg = LogisticRegression(
            C=float(logreg_C),
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )

        self.fitted = False
        self.trained_rank: Optional[int] = None
        self.trained_dim: Optional[int] = None

        # S-norm state
        self.cohort_X: Optional[np.ndarray] = None  # (M, D)

        # Stats cache (Opção B) para X_cal (calibração)
        self._z_mu: Optional[np.ndarray] = None     # (N_cal,)
        self._z_std: Optional[np.ndarray] = None    # (N_cal,)
        self._t_mu: Optional[np.ndarray] = None     # (N_cal,)
        self._t_std: Optional[np.ndarray] = None    # (N_cal,)

    # -----------------------------
    # LRModelProtocol
    # -----------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_train_dataset(self) -> bool:
        return True

    @property
    def requires_calibration_dataset(self) -> bool:
        return True

    # -----------------------------
    # Helpers (SpeechBrain)
    # -----------------------------
    def _make_stat_object(self, X: np.ndarray, ids: np.ndarray) -> StatObject_SB:
        X = np.asarray(X, dtype=np.float64)
        ids = np.asarray(ids).astype(str)

        n = X.shape[0]
        stat0 = np.ones((n, 1), dtype=np.float64)
        none_arr = np.array([None] * n)

        return StatObject_SB(
            modelset=ids.astype("|O"),
            segset=ids.astype("|O"),
            start=none_arr,
            stop=none_arr,
            stat0=stat0,
            stat1=X,
        )

    def _decide_rank_f(self, D: int, C: int) -> int:
        max_allowed = max(1, min(D, C - 1))
        if self.plda_mode == PLDAMode.FULL_RANK:
            return max_allowed
        return max(1, min(self.latent_dim, max_allowed))

    def _score_pairs_llr_batched(self, X: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Score PLDA LLR para lista arbitrária de pares (i,j) em blocos.
        Interpreta (i,j) como (enroll_idx, test_idx) dentro do array X.
        """
        if self.plda is None:
            raise RuntimeError("PLDA não treinado.")
        if not pairs:
            return np.array([], dtype=np.float64)

        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.empty(len(pairs), dtype=np.float64)

        B = self.score_block_size
        block_map: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = defaultdict(list)
        for k, (i, j) in enumerate(pairs):
            if not (0 <= i < n and 0 <= j < n):
                raise IndexError(f"Par fora do range: ({i},{j}) com n={n}")
            be = i // B
            bt = j // B
            block_map[(be, bt)].append((k, i, j))

        for (be, bt), entries in block_map.items():
            en_start = be * B
            en_end = min((be + 1) * B, n)
            te_start = bt * B
            te_end = min((bt + 1) * B, n)

            X_en = X[en_start:en_end]
            X_te = X[te_start:te_end]

            en_ids = np.array([f"en_{en_start + idx}" for idx in range(X_en.shape[0])], dtype="|O")
            te_ids = np.array([f"te_{te_start + idx}" for idx in range(X_te.shape[0])], dtype="|O")

            enroll = self._make_stat_object(X_en, en_ids)
            test = self._make_stat_object(X_te, te_ids)

            ndx = Ndx(models=en_ids, testsegs=te_ids)
            ndx.trialmask = np.ones((en_ids.shape[0], te_ids.shape[0]), dtype=bool)

            scores = fast_PLDA_scoring(enroll, test, ndx, self.plda.mean, self.plda.F, self.plda.Sigma)
            S = scores.scoremat

            for pair_idx, i, j in entries:
                ii = i - en_start
                jj = j - te_start
                out[pair_idx] = float(S[ii, jj])

        return out

    def _score_enroll_many_test_many(self, X_en: np.ndarray, X_te: np.ndarray) -> np.ndarray:
        """
        Matriz densa (len(X_en) x len(X_te)) com fast_PLDA_scoring.
        """
        if self.plda is None:
            raise RuntimeError("PLDA não treinado.")
        X_en = np.asarray(X_en, dtype=np.float64)
        X_te = np.asarray(X_te, dtype=np.float64)

        en_ids = np.array([f"en_{i}" for i in range(X_en.shape[0])], dtype="|O")
        te_ids = np.array([f"te_{i}" for i in range(X_te.shape[0])], dtype="|O")

        enroll = self._make_stat_object(X_en, en_ids)
        test = self._make_stat_object(X_te, te_ids)

        ndx = Ndx(models=en_ids, testsegs=te_ids)
        ndx.trialmask = np.ones((en_ids.shape[0], te_ids.shape[0]), dtype=bool)

        scores = fast_PLDA_scoring(enroll, test, ndx, self.plda.mean, self.plda.F, self.plda.Sigma)
        return scores.scoremat.astype(np.float64)

    # -----------------------------
    # S-norm helpers (Opção B)
    # -----------------------------
    def _build_cohort_from_train(self, train_dataset: EmbeddingDataset) -> np.ndarray:
        X = train_dataset.matrix.astype(np.float64)
        if X.size == 0:
            raise ValueError("Train dataset vazio para construir cohort.")
        M = min(self.cohort_size, X.shape[0])
        idx = self._rng.choice(X.shape[0], size=M, replace=False)
        cohort = X[idx].astype(np.float64)
        self._log(f"[{self.name}]    ↳ Cohort construído do TRAIN: {cohort.shape[0]} embeddings")
        return cohort

    def _compute_zt_stats(self, X_cal: np.ndarray, cohort_X: np.ndarray) -> None:
        """
        Pré-calcula stats Z/T para cada embedding em X_cal:
        - Z: score(enroll=X_cal[i], test=cohort)  -> média/std por linha
        - T: score(enroll=cohort, test=X_cal[i]) -> média/std por coluna
        """
        X_cal = np.asarray(X_cal, dtype=np.float64)
        cohort_X = np.asarray(cohort_X, dtype=np.float64)

        N = X_cal.shape[0]
        M = cohort_X.shape[0]
        B = max(16, int(self.stats_block_size))

        self._log(f"[{self.name}]    ↳ Pré-cálculo S-norm stats: N_cal={N} | cohort={M} | block={B}")

        z_mu = np.empty(N, dtype=np.float64)
        z_std = np.empty(N, dtype=np.float64)

        for s in range(0, N, B):
            e = min(s + B, N)
            S = self._score_enroll_many_test_many(X_cal[s:e], cohort_X)  # (b x M)
            z_mu[s:e] = np.mean(S, axis=1)
            z_std[s:e] = np.std(S, axis=1) + self.snorm_eps

        t_mu = np.empty(N, dtype=np.float64)
        t_std = np.empty(N, dtype=np.float64)

        for s in range(0, N, B):
            e = min(s + B, N)
            S = self._score_enroll_many_test_many(cohort_X, X_cal[s:e])  # (M x b)
            t_mu[s:e] = np.mean(S, axis=0)
            t_std[s:e] = np.std(S, axis=0) + self.snorm_eps

        self._z_mu, self._z_std, self._t_mu, self._t_std = z_mu, z_std, t_mu, t_std

    def _snorm_from_stats(self, scores_raw: np.ndarray, enroll_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        if self._z_mu is None or self._z_std is None or self._t_mu is None or self._t_std is None:
            raise RuntimeError("Stats S-norm não inicializadas.")

        s = np.asarray(scores_raw, dtype=np.float64)
        ei = np.asarray(enroll_idx, dtype=np.int64)
        ti = np.asarray(test_idx, dtype=np.int64)

        z = (s - self._z_mu[ei]) / self._z_std[ei]
        t = (s - self._t_mu[ti]) / self._t_std[ti]
        return 0.5 * (z + t)

    def _snorm_1to1(self, llr_raw: float, probe: np.ndarray, reference: np.ndarray) -> float:
        """
        S-norm on-the-fly para 1:1:
        - Z: reference vs cohort
        - T: cohort vs probe
        """
        if self.cohort_X is None or self.cohort_X.size == 0:
            return float(llr_raw)

        ref = np.asarray(reference, dtype=np.float64).reshape(1, -1)
        prb = np.asarray(probe, dtype=np.float64).reshape(1, -1)

        z_scores = self._score_enroll_many_test_many(ref, self.cohort_X)[0]
        mu_z = float(np.mean(z_scores))
        sd_z = float(np.std(z_scores)) + self.snorm_eps
        z = (llr_raw - mu_z) / sd_z

        t_scores = self._score_enroll_many_test_many(self.cohort_X, prb)[:, 0]
        mu_t = float(np.mean(t_scores))
        sd_t = float(np.std(t_scores)) + self.snorm_eps
        t = (llr_raw - mu_t) / sd_t

        return float(0.5 * (z + t))

    # -----------------------------
    # Diagnósticos (opcional)
    # -----------------------------
    @staticmethod
    def _cllr_from_llr(llr_genuine: np.ndarray, llr_impostor: np.ndarray) -> float:
        llr_genuine = np.asarray(llr_genuine, dtype=np.float64)
        llr_impostor = np.asarray(llr_impostor, dtype=np.float64)
        ln2 = np.log(2.0)

        a = np.mean(np.log1p(np.exp(-llr_genuine)) / ln2) if llr_genuine.size else np.nan
        b = np.mean(np.log1p(np.exp(llr_impostor)) / ln2) if llr_impostor.size else np.nan
        return float(0.5 * (a + b))

    @staticmethod
    def _percentiles(x: np.ndarray, ps=(1, 5, 10, 50, 90, 95, 99)) -> Dict[int, float]:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return {int(p): float("nan") for p in ps}
        vals = np.percentile(x, ps)
        return {int(p): float(v) for p, v in zip(ps, vals)}

    # -----------------------------
    # FIT (assinatura do LRModelProtocol)
    # -----------------------------
    def fit(self, train_dataset: Optional[EmbeddingDataset], calibration_dataset: EmbeddingDataset) -> None:
        if train_dataset is None:
            raise ValueError("PLDASnormStrategy requer train_dataset (não pode ser None).")

        self._log(
            f"[{self.name}] 🧠 Treinando PLDA+S-norm (SpeechBrain PLDA LLR | EM={self.em_iterations} | "
            f"Mode={self.plda_mode.value} | latent_dim={self.latent_dim} | seed={self.seed})..."
        )

        # ---- Treino PLDA (train_dataset)
        X_all = train_dataset.matrix.astype(np.float64)
        y_all = np.array([item.identity for item in train_dataset.items], dtype=object)

        if X_all.size == 0:
            raise ValueError("Dataset de treino vazio.")

        mask_labeled = np.array([id_ is not None for id_ in y_all], dtype=bool)
        X = X_all[mask_labeled]
        y = np.array([str(id_) for id_ in y_all[mask_labeled]], dtype=str)

        dropped_none = int(np.sum(~mask_labeled))
        if dropped_none > 0:
            self._log(f"[{self.name}] ℹ️ Removidos {dropped_none} embeddings sem identity (identity=None).")

        if X.shape[0] < 2:
            raise ValueError("Dados insuficientes para treinar PLDA após remover identity=None.")

        unique_ids, counts = np.unique(y, return_counts=True)
        valid_ids = set(unique_ids[counts > 1])
        mask_multi = np.array([idi in valid_ids for idi in y], dtype=bool)

        X_train, y_train = X[mask_multi], y[mask_multi]
        dropped_singletons = int(np.sum(~mask_multi))
        if dropped_singletons > 0:
            self._log(f"[{self.name}] ℹ️ Filtrados {dropped_singletons} embeddings 'singletons' (1 amostra/ID).")

        if X_train.shape[0] < 2:
            raise ValueError("Dados insuficientes para treinar PLDA (requer IDs com >=2 embeddings).")

        D = int(X_train.shape[1])
        C = int(len(np.unique(y_train)))
        N = int(X_train.shape[0])
        N_within = int(N - C)

        rank_f = self._decide_rank_f(D, C)
        self.trained_rank = rank_f
        self.trained_dim = D

        self._log(f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D} | N_within={N_within} | rank_f={rank_f}")
        if C < 5 * rank_f:
            self._log(
                f"[{self.name}] ⚠️ Aviso: Classes={C} pode ser pouco para rank_f={rank_f}. "
                f"Considere LOW_RANK menor ou mais identidades."
            )
        if N_within < 10 * rank_f:
            self._log(
                f"[{self.name}] ⚠️ Aviso: N_within={N_within} pode ser pouco para rank_f={rank_f}. "
                f"Considere mais amostras por identidade."
            )

        self._log(f"[{self.name}] ⚙️ Treinando PLDA (SpeechBrain nb_iter={self.em_iterations})...")
        train_stat = self._make_stat_object(X_train, y_train)
        self.plda = PLDA(rank_f=rank_f, nb_iter=self.em_iterations)

        # whiten=False: você já prepara o embedding externamente (centering -> L2 etc.)
        self.plda.plda(train_stat, whiten=False)

        # ---- Cohort vem do TRAIN (não “rouba” do calibration)
        self.cohort_X = self._build_cohort_from_train(train_dataset)

        # ---- Calibração com S-norm (Opção B) usando cohort do treino
        self._fit_calibrator_snorm(calibration_dataset)

        self.fitted = True
        self._log(f"[{self.name}] ✅ Treinamento concluído.")

    def _fit_calibrator_snorm(self, dataset: EmbeddingDataset) -> None:
        if self.plda is None:
            raise RuntimeError("PLDA não treinado.")
        if self.cohort_X is None or self.cohort_X.size == 0:
            raise RuntimeError("Cohort vazio. Não é possível calibrar S-norm.")

        self._log(
            f"[{self.name}] ⚖️ Calibrando LogReg (scores = PLDA LLR + S-norm | "
            f"min_impostors={self.min_impostors} | hard_frac={self.impostor_hard_fraction:.2f} | "
            f"max_gen_per_id={self.max_genuine_pairs_per_id} | block={self.score_block_size} | "
            f"cohort={self.cohort_X.shape[0]} | stats_block={self.stats_block_size})..."
        )

        X_all = dataset.matrix.astype(np.float64)
        y_all = np.array([item.identity for item in dataset.items], dtype=object)
        if X_all.size == 0:
            raise ValueError("Dataset de calibração vazio.")

        mask_labeled = np.array([id_ is not None for id_ in y_all], dtype=bool)
        X_cal = X_all[mask_labeled]
        ids_cal = np.array([str(id_) for id_ in y_all[mask_labeled]], dtype=str)

        dropped_none = int(np.sum(~mask_labeled))
        if dropped_none > 0:
            self._log(f"[{self.name}] ℹ️ Removidos {dropped_none} embeddings sem identity na calibração.")

        if X_cal.shape[0] < 4:
            raise ValueError("Calibração insuficiente (poucos embeddings após remover identity=None).")

        # ---- agrupamento do set de calibração
        grouped = defaultdict(list)
        for i, identity in enumerate(ids_cal):
            grouped[identity].append(i)

        all_identities = list(grouped.keys())
        if len(all_identities) < 2:
            raise ValueError("Calibração requer >= 2 identidades distintas (com labels).")

        # ---- pré-cálculo stats Z/T (Opção B) no calibration set, usando cohort do treino
        self._compute_zt_stats(X_cal, self.cohort_X)

        # -----------------
        # 1) pares genuínos (enroll,test)
        # -----------------
        genuine_pairs: List[Tuple[int, int]] = []
        self._log(f"[{self.name}]    ↳ Gerando pares genuínos...")
        for identity, idxs in grouped.items():
            if len(idxs) < 2:
                continue

            all_pairs = [(i, j) for i, j in combinations(idxs, 2)]
            if self.max_genuine_pairs_per_id is not None and len(all_pairs) > self.max_genuine_pairs_per_id:
                sel = self._rng.choice(len(all_pairs), size=self.max_genuine_pairs_per_id, replace=False)
                genuine_pairs.extend([all_pairs[int(k)] for k in sel])
            else:
                genuine_pairs.extend(all_pairs)

        num_gen = len(genuine_pairs)
        self._log(f"[{self.name}]    ↳ Genuínos: {num_gen}")

        # -----------------
        # 2) impostores (random + hard)
        # -----------------
        imp_target = max(num_gen, self.min_impostors)
        hard_target = int(round(imp_target * self.impostor_hard_fraction))
        rand_target = imp_target - hard_target
        self._log(f"[{self.name}]    ↳ Impostores alvo: {imp_target} (random={rand_target}, hard={hard_target})")

        impostor_pairs_random: List[Tuple[int, int]] = []
        if rand_target > 0:
            for _ in range(rand_target):
                id1, id2 = self._rng.choice(all_identities, size=2, replace=False)
                en = int(self._rng.choice(grouped[id1]))
                te = int(self._rng.choice(grouped[id2]))
                impostor_pairs_random.append((en, te))

        impostor_pairs_hard: List[Tuple[int, int]] = []
        if hard_target > 0:
            cand_n = hard_target * self.hard_candidate_multiplier
            self._log(f"[{self.name}]    ↳ Hard impostors: gerando {cand_n} candidatos...")

            candidates: List[Tuple[int, int]] = []
            for _ in range(cand_n):
                id1, id2 = self._rng.choice(all_identities, size=2, replace=False)
                en = int(self._rng.choice(grouped[id1]))
                te = int(self._rng.choice(grouped[id2]))
                candidates.append((en, te))

            self._log(f"[{self.name}]    ↳ Pontuando candidatos (raw, batch)...")
            cand_raw = self._score_pairs_llr_batched(X_cal, candidates)

            en_idx = np.fromiter((p[0] for p in candidates), dtype=np.int64, count=len(candidates))
            te_idx = np.fromiter((p[1] for p in candidates), dtype=np.int64, count=len(candidates))
            cand_snorm = self._snorm_from_stats(cand_raw, en_idx, te_idx)

            top_idx = np.argpartition(cand_snorm, -hard_target)[-hard_target:]
            top_idx = top_idx[np.argsort(cand_snorm[top_idx])[::-1]]
            impostor_pairs_hard = [candidates[int(k)] for k in top_idx.tolist()]

        impostor_pairs = impostor_pairs_random + impostor_pairs_hard
        num_imp = len(impostor_pairs)

        # -----------------
        # 3) scoring + snorm
        # -----------------
        self._log(f"[{self.name}]    ↳ Scoring batch: genuínos + impostores...")
        llr_gen_raw = self._score_pairs_llr_batched(X_cal, genuine_pairs)
        llr_imp_raw = self._score_pairs_llr_batched(X_cal, impostor_pairs)

        gen_en = np.fromiter((p[0] for p in genuine_pairs), dtype=np.int64, count=len(genuine_pairs))
        gen_te = np.fromiter((p[1] for p in genuine_pairs), dtype=np.int64, count=len(genuine_pairs))
        imp_en = np.fromiter((p[0] for p in impostor_pairs), dtype=np.int64, count=len(impostor_pairs))
        imp_te = np.fromiter((p[1] for p in impostor_pairs), dtype=np.int64, count=len(impostor_pairs))

        llr_gen = self._snorm_from_stats(llr_gen_raw, gen_en, gen_te)
        llr_imp = self._snorm_from_stats(llr_imp_raw, imp_en, imp_te)

        # -----------------
        # 4) treina LogReg
        # -----------------
        self._log(f"[{self.name}]    ↳ Ajustando calibrador LogReg...")
        scores = np.concatenate([llr_gen, llr_imp], axis=0).reshape(-1, 1)
        labels = np.concatenate(
            [np.ones_like(llr_gen, dtype=np.int32), np.zeros_like(llr_imp, dtype=np.int32)],
            axis=0
        )
        self.logreg.fit(scores, labels)

        self._log(f"[{self.name}] ⚖️ Calibrador pronto (Genuínos: {num_gen}, Impostores: {num_imp}).")

        # -----------------
        # 5) diagnósticos (opcional)
        # -----------------
        if self.compute_train_diagnostics:
            try:
                gen_p = self._percentiles(llr_gen)
                imp_p = self._percentiles(llr_imp)

                self._log(f"[{self.name}] 📈 Diagnostics (S-norm scores usados na calibração):")
                self._log(f"[{self.name}]    percentis genuíno: {gen_p}")
                self._log(f"[{self.name}]    percentis impostor: {imp_p}")

                if self.diagnostics_compute_cllr:
                    cllr_before = self._cllr_from_llr(llr_gen, llr_imp)

                    probs_gen = self.logreg.predict_proba(llr_gen.reshape(-1, 1))
                    probs_imp = self.logreg.predict_proba(llr_imp.reshape(-1, 1))
                    eps = 1e-15

                    p1g = np.clip(probs_gen[:, 1], eps, 1 - eps)
                    llr_gen_cal = np.log(p1g) - np.log(1 - p1g)

                    p1i = np.clip(probs_imp[:, 1], eps, 1 - eps)
                    llr_imp_cal = np.log(p1i) - np.log(1 - p1i)

                    cllr_after = self._cllr_from_llr(llr_gen_cal, llr_imp_cal)

                    self._log(f"[{self.name}]    Cllr (antes calib): {cllr_before:.4f}")
                    self._log(f"[{self.name}]    Cllr (depois calib): {cllr_after:.4f}")
            except Exception as e:
                self._log(f"[{self.name}] ⚠️ Diagnostics falharam: {e}")

    # -----------------------------
    # Predict / Compare
    # -----------------------------
    def _predict_llr_used_and_raw(self, probe_vec: np.ndarray, ref_vec: np.ndarray) -> Tuple[float, float]:
        if not self.fitted or self.plda is None:
            raise RuntimeError("Modelo PLDASnormStrategy não treinado. Execute 'fit' primeiro.")
        if self.cohort_X is None or self.cohort_X.size == 0:
            raise RuntimeError("Cohort não disponível. Re-treine (fit) para usar S-norm.")

        probe = np.asarray(probe_vec, dtype=np.float64)
        reference = np.asarray(ref_vec, dtype=np.float64)

        # Score bruto: reference=enroll, probe=test
        X = np.vstack([reference, probe]).astype(np.float64)
        llr_raw = float(self._score_pairs_llr_batched(X, [(0, 1)])[0])

        llr_used = self._snorm_1to1(llr_raw, probe=probe, reference=reference)
        return llr_used, llr_raw

    def predict(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> float:
        llr_used, _llr_raw = self._predict_llr_used_and_raw(probe.embedding, reference.embedding)

        probs = self.logreg.predict_proba([[llr_used]])[0]
        prob_hp = float(probs[1])
        prob_hd = float(probs[0])

        if prob_hd < 1e-15:
            return float("inf")
        if prob_hp < 1e-15:
            return 0.0
        return float(prob_hp / prob_hd)

    def compare(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> CaseResult:
        llr_used, llr_raw = self._predict_llr_used_and_raw(probe.embedding, reference.embedding)

        probs = self.logreg.predict_proba([[llr_used]])[0]
        prob_hp = float(probs[1])
        prob_hd = float(probs[0])

        if prob_hd < 1e-15:
            lr = float("inf")
        elif prob_hp < 1e-15:
            lr = 0.0
        else:
            lr = prob_hp / prob_hd

        return CaseResult(
            strategy_name=self.name,
            probe=probe,
            reference=reference,
            score=float(llr_raw),   # score bruto PLDA LLR (debug/telemetria)
            lr=float(lr),
            verbal_support=self._get_verbal_support(float(lr)),
        )

    # -----------------------------
    # Persistência
    # -----------------------------
    def save(self, path: str) -> None:
        payload = {
            "model_version": self.MODEL_VERSION,
            "plda_backend": "speechbrain",
            "plda": self.plda,
            "logreg": self.logreg,
            "cohort_X": self.cohort_X,
            "config": {
                "name": self._name,
                "em_iterations": self.em_iterations,
                "plda_mode": self.plda_mode.value,
                "latent_dim": self.latent_dim,
                "trained_rank": self.trained_rank,
                "trained_dim": self.trained_dim,
                "min_impostors": self.min_impostors,
                "impostor_hard_fraction": self.impostor_hard_fraction,
                "hard_candidate_multiplier": self.hard_candidate_multiplier,
                "max_genuine_pairs_per_id": self.max_genuine_pairs_per_id,
                "score_block_size": self.score_block_size,
                "cohort_size": self.cohort_size,
                "snorm_eps": self.snorm_eps,
                "stats_block_size": self.stats_block_size,
                "seed": self.seed,
                "compute_train_diagnostics": self.compute_train_diagnostics,
                "diagnostics_compute_cllr": self.diagnostics_compute_cllr,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)

        self.plda = d.get("plda", None)
        self.logreg = d.get("logreg", self.logreg)
        self.cohort_X = d.get("cohort_X", None)

        conf = d.get("config", {})
        self._name = conf.get("name", self._name)
        self.em_iterations = int(conf.get("em_iterations", self.em_iterations))

        saved_mode = conf.get("plda_mode", self.plda_mode.value)
        self.plda_mode = saved_mode if isinstance(saved_mode, PLDAMode) else PLDAMode(saved_mode)

        self.latent_dim = int(conf.get("latent_dim", self.latent_dim))
        self.trained_rank = conf.get("trained_rank", None)
        self.trained_dim = conf.get("trained_dim", None)

        self.min_impostors = int(conf.get("min_impostors", self.min_impostors))
        self.impostor_hard_fraction = float(conf.get("impostor_hard_fraction", self.impostor_hard_fraction))
        self.hard_candidate_multiplier = int(conf.get("hard_candidate_multiplier", self.hard_candidate_multiplier))
        self.max_genuine_pairs_per_id = conf.get("max_genuine_pairs_per_id", self.max_genuine_pairs_per_id)
        self.score_block_size = int(conf.get("score_block_size", self.score_block_size))

        self.cohort_size = int(conf.get("cohort_size", self.cohort_size))
        self.snorm_eps = float(conf.get("snorm_eps", self.snorm_eps))
        self.stats_block_size = int(conf.get("stats_block_size", self.stats_block_size))

        self.seed = conf.get("seed", self.seed)
        self._rng = np.random.default_rng(self.seed)

        self.compute_train_diagnostics = bool(conf.get("compute_train_diagnostics", self.compute_train_diagnostics))
        self.diagnostics_compute_cllr = bool(conf.get("diagnostics_compute_cllr", self.diagnostics_compute_cllr))

        # stats cache não precisa persistir (depende do calibration set). Mantém None.
        self._z_mu = self._z_std = self._t_mu = self._t_std = None

        self.fitted = self.plda is not None

    # -----------------------------
    # Suporte verbal
    # -----------------------------
    def _get_verbal_support(self, lr: float) -> str:
        if np.isinf(lr) or np.isnan(lr):
            return "Suporte Indeterminado"
        if lr < 1:
            return "Exclusão / Suporte para Diferente"
        if lr < 10:
            return "Suporte Fraco"
        if lr < 100:
            return "Suporte Moderado"
        if lr < 1000:
            return "Suporte Moderadamente Forte"
        if lr < 10000:
            return "Suporte Forte"
        return "Suporte Muito Forte"