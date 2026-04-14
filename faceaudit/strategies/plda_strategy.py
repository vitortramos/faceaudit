import numpy as np
import pickle
import random
from typing import Optional, Tuple, Callable, Dict, List

from sklearn.linear_model import LogisticRegression

# Backend SpeechBrain
from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB, Ndx, fast_PLDA_scoring

# Enums e Protocolos
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.models import EmbeddingDataset, EmbeddingInfo, CaseResult
from faceaudit.protocols.lr_model_protocol import LRModelProtocol
from faceaudit.models.embedding_comparison_dataset import EmbeddingComparisonDataset
from faceaudit.models.embedding_comparison import EmbeddingComparison

# Imports de Mineração para tipagem
from faceaudit.services.comparison_mining_service import ComparisonMiningService
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig


class PLDAStrategy(LRModelProtocol):
    """
    Implementação robusta de PLDA (Probabilistic Linear Discriminant Analysis) usando backend SpeechBrain.
    
    Esta estratégia realiza:
    1. Redução de Dimensionalidade Supervisionada (opcional via PLDAMode).
    2. Modelagem Generativa Gaussiana (Between-Class vs Within-Class Covariance).
    3. Cálculo de Scores LLR (Log-Likelihood Ratio).
    4. Calibração de Scores para Probabilidades (Likelihood Ratio) via Regressão Logística.

    Destaques Técnicos:
    -------------------
    - **SpeechBrain Backend:** Utiliza a implementação canônica e otimizada de PLDA.
    - **Batch Scoring:** Otimização crítica de performance via Matrizes C++.
    - **Injeção de Curadoria:** A calibração utiliza o ComparisonMiningService para extração balanceada.
    """

    MODEL_VERSION = 5  # Incrementado devido à purificação do construtor e injeção forte

    def __init__(
        self,
        name: str = "PLDA_LLR_LogReg",
        em_iterations: int = 20,
        plda_mode: PLDAMode = PLDAMode.FULL_RANK,
        latent_dim: int = 128,

        # --- Calibração Matemática ---
        logreg_C: float = 1.0,

        # --- Performance ---
        score_block_size: int = 512,

        # --- Reprodutibilidade ---
        seed: Optional[int] = 42,

        # --- Telemetria ---
        compute_train_diagnostics: bool = False,
        diagnostics_compute_cllr: bool = True,
    ):
        self._name = name
        self.em_iterations = int(em_iterations)

        self.plda_mode = plda_mode if isinstance(plda_mode, PLDAMode) else PLDAMode(plda_mode)
        
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 1:
            raise ValueError("latent_dim deve ser >= 1.")

        self.score_block_size = int(score_block_size)
        if self.score_block_size < 32:
            raise ValueError("score_block_size muito pequeno. Use >= 32 (recomendado 256/512/1024).")

        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.compute_train_diagnostics = bool(compute_train_diagnostics)
        self.diagnostics_compute_cllr = bool(diagnostics_compute_cllr)

        # Backend PLDA (SpeechBrain)
        self.plda: Optional[PLDA] = None
        self.fitted = False
        self.trained_rank: Optional[int] = None
        self.trained_dim: Optional[int] = None

        # Calibrador (LLR -> Probabilidade)
        self.logreg = LogisticRegression(
            C=float(logreg_C),
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )

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
    # Helpers SpeechBrain & Math
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

    # -----------------------------
    # Util: Diagnósticos (Cllr)
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
    # Pipeline Principal: FIT
    # -----------------------------
    def fit(
        self,
        train_dataset: EmbeddingDataset,
        calibration_dataset: EmbeddingDataset,
        logger_callback: Callable[[str], None] = print,
        mining_service: Optional[ComparisonMiningService] = None,
        mining_config: Optional[ComparisonMiningConfig] = None
    ) -> None:
        """
        Executa o treinamento completo (PLDA + Calibração).
        Usa Injeção de Dependência pelo Método para aplicar a regra do usuário.
        """
        logger_callback(
            f"[{self.name}] 🧠 Treinando (SpeechBrain PLDA LLR | EM={self.em_iterations} | "
            f"Mode={self.plda_mode.value} | latent_dim={self.latent_dim} | seed={self.seed})..."
        )

        # ---- Etapa 1: Limpeza do Treino (Training Set)
        X_all = train_dataset.matrix.astype(np.float64)
        y_all = np.array([item.identity for item in train_dataset.items], dtype=object)

        if X_all.size == 0:
            raise ValueError("Dataset de treino vazio.")

        mask_labeled = np.array([id_ is not None for id_ in y_all], dtype=bool)
        X = X_all[mask_labeled]
        y = np.array([str(id_) for id_ in y_all[mask_labeled]], dtype=str)

        if X.shape[0] < 2:
            raise ValueError("Dados insuficientes para treinar PLDA após remover identity=None.")

        unique_ids, counts = np.unique(y, return_counts=True)
        valid_ids = set(unique_ids[counts > 1])
        mask_multi = np.array([idi in valid_ids for idi in y], dtype=bool)

        X_train, y_train = X[mask_multi], y[mask_multi]
        if X_train.shape[0] < 2:
            raise ValueError("Dados insuficientes para treinar PLDA (requer IDs com >=2 embeddings).")

        # ---- Etapa 2: Definição de Rank e Diagnósticos
        D = int(X_train.shape[1])
        C = int(len(np.unique(y_train)))
        N = int(X_train.shape[0])

        rank_f = self._decide_rank_f(D, C)
        self.trained_rank = rank_f
        self.trained_dim = D

        logger_callback(f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D} | rank_f={rank_f}")

        # ---- Etapa 3: Treino PLDA (Core)
        logger_callback(f"[{self.name}] ⚙️ Treinando Matriz PLDA (nb_iter={self.em_iterations})...")
        train_stat = self._make_stat_object(X_train, y_train)
        self.plda = PLDA(rank_f=rank_f, nb_iter=self.em_iterations)
        self.plda.plda(train_stat, whiten=False) 

        # ---- Etapa 4: Calibração Rigorosa
        if mining_service is None or mining_config is None:
            raise ValueError("Um ComparisonMiningService e um ComparisonMiningConfig devem ser injetados para o treinamento de Calibração do PLDA.")

        self._fit_calibrator_with_service(
            calibration_dataset=calibration_dataset,
            mining_service=mining_service,
            mining_config=mining_config,
            logger_callback=logger_callback
        )

        self.fitted = True
        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

    # -----------------------------
    # Calibração Injetada
    # -----------------------------
    def _fit_calibrator_with_service(
        self, 
        calibration_dataset: EmbeddingDataset, 
        mining_service: ComparisonMiningService,
        mining_config: ComparisonMiningConfig,
        logger_callback: Callable[[str], None]
    ) -> None:
        """
        Gera os scores iniciais da matriz completa de calibração, repassa ao 
        Serviço de Mineração e treina a Regressão Logística com os dados limpos.
        """
        logger_callback(f"[{self.name}] 📐 Pontuando matriz densa de calibração (isso pode levar alguns segundos)...")
        
        # 1. Gerar todos contra todos (rápido via C++) SEM CALIBRAR
        X_calib = calibration_dataset.matrix.astype(np.float64)
        n_c = X_calib.shape[0]
        
        c_ids = np.array([f"c_{i:08d}" for i in range(n_c)], dtype="|O")
        c_stat = self._make_stat_object(X_calib, c_ids)
        
        ndx = Ndx(models=c_ids, testsegs=c_ids)
        ndx.trialmask = np.ones((n_c, n_c), dtype=bool)
        
        scores_obj = fast_PLDA_scoring(c_stat, c_stat, ndx, self.plda.mean, self.plda.F, self.plda.Sigma)
        S_llr = scores_obj.scoremat # Matriz N x N com os scores brutos
        
        # 2. Empacotar tudo em EmbeddingComparisons para o serviço ler
        logger_callback(f"[{self.name}] 📦 Empacotando {(n_c * (n_c - 1)) // 2} pares válidos para o Serviço de Mineração...")
        items = calibration_dataset.items
        raw_comparisons = []
        
        # Extrai apenas o triângulo superior (sem diagonal)
        for i in range(n_c):
            for j in range(i + 1, n_c):
                comp = EmbeddingComparison(
                    embedding_1=items[i],
                    embedding_2=items[j],
                    cosine_similarity=float(S_llr[i, j]) # Usamos a propriedade existente para transitar o LLR Bruto
                )
                raw_comparisons.append(comp)
                
        raw_comp_dataset = EmbeddingComparisonDataset(name="Raw_LLR_Pairs", items=raw_comparisons)
        
        # 3. Chamar o serviço com o arquivo Global
        logger_callback(f"[{self.name}] ⛏️ Delegando curadoria usando regra '{mining_config.name}'...")
        ds_imp, ds_gen = mining_service.mine_pairs(raw_comp_dataset, mining_config)
        
        # 4. Extrair os scores filtrados
        llr_imp = np.array([c.cosine_similarity for c in ds_imp.items], dtype=np.float64)
        llr_gen = np.array([c.cosine_similarity for c in ds_gen.items], dtype=np.float64)

        num_imp, num_gen = len(llr_imp), len(llr_gen)
        logger_callback(f"[{self.name}] ⚖️ Ajustando calibrador LogReg (Genuínos: {num_gen}, Impostores: {num_imp})...")
        
        if num_imp == 0 or num_gen == 0:
            raise ValueError(f"Mineração retornou {num_gen} genuínos e {num_imp} impostores. O Treino requer pelo menos 1 de cada.")

        scores = np.concatenate([llr_gen, llr_imp], axis=0).reshape(-1, 1)
        labels = np.concatenate([np.ones_like(llr_gen, dtype=np.int32), np.zeros_like(llr_imp, dtype=np.int32)], axis=0)

        self.logreg.fit(scores, labels)

        # 5. Diagnósticos
        if self.compute_train_diagnostics:
            try:
                gen_p = self._percentiles(llr_gen)
                imp_p = self._percentiles(llr_imp)

                logger_callback(f"[{self.name}] 📈 Diagnostics (TRAIN/CALIB pairs):")
                logger_callback(f"[{self.name}]    LLR genuíno percentis: {gen_p}")
                logger_callback(f"[{self.name}]    LLR impostor percentis: {imp_p}")

                if self.diagnostics_compute_cllr:
                    cllr_before = self._cllr_from_llr(llr_gen, llr_imp)

                    probs_gen = self.logreg.predict_proba(llr_gen.reshape(-1, 1))
                    probs_imp = self.logreg.predict_proba(llr_imp.reshape(-1, 1))
                    eps = 1e-15
                    
                    p1g = np.clip(probs_gen[:, 1], eps, 1 - eps)
                    p0g = 1 - p1g
                    llr_gen_cal = np.log(p1g) - np.log(p0g)

                    p1i = np.clip(probs_imp[:, 1], eps, 1 - eps)
                    p0i = 1 - p1i
                    llr_imp_cal = np.log(p1i) - np.log(p0i)

                    cllr_after = self._cllr_from_llr(llr_gen_cal, llr_imp_cal)

                    logger_callback(f"[{self.name}]    Cllr (antes calib): {cllr_before:.4f}")
                    logger_callback(f"[{self.name}]    Cllr (depois calib): {cllr_after:.4f}")
            except Exception as e:
                logger_callback(f"[{self.name}] ⚠️ Diagnostics falharam: {e}")

    # -----------------------------
    # Predict / Compare
    # -----------------------------
    def _predict(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float]:
        if not self.fitted or self.plda is None:
            raise RuntimeError("Modelo PLDAStrategy não treinado. Execute 'fit' primeiro.")

        try:
            # Fake IDs
            p_ids = np.array([f"p_00000001"], dtype="|O")
            r_ids = np.array([f"r_00000001"], dtype="|O")
            
            p_stat = self._make_stat_object(v1.reshape(1, -1), p_ids)
            r_stat = self._make_stat_object(v2.reshape(1, -1), r_ids)
            
            ndx = Ndx(models=p_ids, testsegs=r_ids)
            ndx.trialmask = np.ones((1, 1), dtype=bool)
            
            scores = fast_PLDA_scoring(p_stat, r_stat, ndx, self.plda.mean, self.plda.F, self.plda.Sigma)
            llr = float(scores.scoremat[0, 0])

            probs = self.logreg.predict_proba([[llr]])[0]
            prob_hp = float(probs[1]) 
            prob_hd = float(probs[0]) 

            if prob_hd < 1e-15:
                lr = float("inf")
            elif prob_hp < 1e-15:
                lr = 0.0
            else:
                lr = prob_hp / prob_hd

            return float(lr), llr
        except Exception as e:
            print(f"[{self.name}] ERRO CRÍTICO em _predict: {e}")
            return 1.0, 0.0

    def compare(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> CaseResult:
        lr, raw_score = self._predict(probe.embedding, reference.embedding)
        return CaseResult(
            strategy_name=self.name,
            probe=probe,
            reference=reference,
            score=float(raw_score),
            lr=float(lr),
            verbal_support=self._get_verbal_support(lr),
        )

    def predict(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> float:
        lr, _ = self._predict(probe.embedding, reference.embedding)
        return float(lr)

    def predict_full_matrix(self, probe_matrix: np.ndarray, ref_matrix: np.ndarray) -> np.ndarray:
        if not self.fitted or self.plda is None:
            raise RuntimeError("Modelo PLDAStrategy não treinado. Execute 'fit' primeiro.")

        P = np.asarray(probe_matrix, dtype=np.float64)
        R = np.asarray(ref_matrix, dtype=np.float64)
        
        n_p, n_r = P.shape[0], R.shape[0]
        
        if n_p == 0 or n_r == 0:
            return np.empty((n_p, n_r), dtype=np.float64)
            
        p_ids = np.array([f"p_{i:08d}" for i in range(n_p)], dtype="|O")
        r_ids = np.array([f"r_{i:08d}" for i in range(n_r)], dtype="|O")
        
        p_stat = self._make_stat_object(P, p_ids)
        r_stat = self._make_stat_object(R, r_ids)
        
        ndx = Ndx(models=p_ids, testsegs=r_ids)
        ndx.trialmask = np.ones((n_p, n_r), dtype=bool)
        
        scores = fast_PLDA_scoring(p_stat, r_stat, ndx, self.plda.mean, self.plda.F, self.plda.Sigma)
        S_llr = scores.scoremat 
        
        probs = self.logreg.predict_proba(S_llr.reshape(-1, 1))
        prob_hd = probs[:, 0]
        prob_hp = probs[:, 1]
        
        eps = 1e-15
        prob_hd = np.clip(prob_hd, eps, 1.0)
        prob_hp = np.clip(prob_hp, eps, 1.0)
        
        lr_flat = prob_hp / prob_hd
        
        return lr_flat.reshape((n_p, n_r))
        
    # -----------------------------
    # Persistência
    # -----------------------------
    def save(self, path: str) -> None:
        payload = {
            "model_version": self.MODEL_VERSION,
            "plda_backend": "speechbrain",
            "plda": self.plda,
            "logreg": self.logreg,
            "config": {
                "name": self._name,
                "em_iterations": self.em_iterations,
                "plda_mode": self.plda_mode.name,
                "latent_dim": self.latent_dim,
                "trained_rank": self.trained_rank,
                "trained_dim": self.trained_dim,
                "score_block_size": self.score_block_size,
                "seed": self.seed,
                "compute_train_diagnostics": self.compute_train_diagnostics,
                "diagnostics_compute_cllr": self.diagnostics_compute_cllr,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)

            if isinstance(d, dict) and "plda_backend" not in d and "plda" in d and "logreg" in d:
                self.logreg = d["logreg"]
                self.fitted = False
                self.plda = None
                return

            self.plda = d.get("plda", None)
            self.logreg = d.get("logreg", self.logreg)

            conf = d.get("config", {})
            self._name = conf.get("name", self._name)
            self.em_iterations = int(conf.get("em_iterations", self.em_iterations))
            
            saved_mode = conf.get("plda_mode", self.plda_mode.name)
            self.plda_mode = saved_mode if isinstance(saved_mode, PLDAMode) else PLDAMode.from_str(saved_mode)
            
            self.latent_dim = int(conf.get("latent_dim", self.latent_dim))
            self.trained_rank = conf.get("trained_rank", None)
            self.trained_dim = conf.get("trained_dim", None)

            self.score_block_size = int(conf.get("score_block_size", self.score_block_size))
            self.seed = conf.get("seed", self.seed)
            self._rng = np.random.default_rng(self.seed)

            self.compute_train_diagnostics = bool(conf.get("compute_train_diagnostics", self.compute_train_diagnostics))
            self.diagnostics_compute_cllr = bool(conf.get("diagnostics_compute_cllr", self.diagnostics_compute_cllr))

            self.fitted = self.plda is not None
        except Exception:
            self.fitted = False
            raise

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