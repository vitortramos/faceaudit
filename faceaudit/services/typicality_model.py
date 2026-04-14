# faceaudit/services/typicality_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Union, Tuple, List

import numpy as np
from scipy.special import ive, logsumexp

from faceaudit.models.typicality_info import TypicalityInfo


ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]


@dataclass
class _MoVMFParams:
    mus: np.ndarray          # (K, D)
    kappas: np.ndarray       # (K,)
    weights: np.ndarray      # (K,)
    dim: int
    sample_count: int
    n_components: int


class TypicalityModel:
    """
    Modelo de typicality baseado em mistura de distribuições von Mises-Fisher (MoVMF)
    para embeddings L2-normalizados na hiperesfera.

    Recursos:
    - Suporte opcional à centralização dos embeddings. (Atenção geométrica: centralizar
      pela média euclidiana global e renormalizar distorce levemente a métrica esférica
      original, mas reduz o efeito de 'hubness' em espaços de biometria).
    - n_components = 1 -> equivalente à vMF única.
    - n_components > 1 -> mistura de vMF treinada por EM com inicialização K-Means++ esférico.
    - Seleção automática de K via AIC/BIC usando o método de classe `fit_best_model`.
    - Cálculo de log-likelihood da mistura e percentil empírico suavizado.
    - Exportação e importação explícita de estado via dicionário seguro para JSON.
    - Métodos de diagnóstico para observabilidade em produção (summary, get_degenerate_components).
    """

    def __init__(
        self,
        population_label: Optional[str] = None,
        model_name: str = "TypicalityModel",
        eps: float = 1e-12,
        center_embeddings: bool = True,
        n_components: int = 1,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int = 42,
        min_component_weight: float = 1e-6,
        min_kappa: float = 1e-6,
        max_kappa: float = 1e4,
    ):
        if n_components < 1:
            raise ValueError("n_components deve ser >= 1.")

        self.population_label = population_label
        self.model_name = model_name
        self.eps = eps
        self.center_embeddings = center_embeddings

        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.min_component_weight = float(min_component_weight)
        self.min_kappa = float(min_kappa)
        self.max_kappa = float(max_kappa)

        # Estado interno pós-treino
        self._params: Optional[_MoVMFParams] = None
        self._train_log_likelihoods: Optional[np.ndarray] = None
        self._mean_vector: Optional[np.ndarray] = None
        self._train_avg_log_likelihood: Optional[float] = None
        self._n_iter_: Optional[int] = None
        self._converged_: Optional[bool] = None

    # =========================================================================
    # SELEÇÃO DE MODELO (AIC/BIC) E DIAGNÓSTICO
    # =========================================================================

    @property
    def aic_(self) -> Optional[float]:
        """
        Critério de Informação de Akaike (menor é melhor).
        Nota metodológica: Em misturas vMF, o número de graus de liberdade é uma aproximação prática
        devido a restrições geométricas e identificabilidade.
        """
        if not self.is_fitted():
            return None
        
        if self._train_avg_log_likelihood is None or not np.isfinite(self._train_avg_log_likelihood):
            return None

        n = self._params.sample_count
        k = self._params.n_components
        d = self._params.dim
        
        # Graus de liberdade aproximados:
        # - K*(D-1) para as direções médias na hiperesfera
        # - K para os kappas
        # - (K-1) para os pesos da mistura
        # Total aproximado: K*D + K - 1
        p = k * d + k - 1 
        total_ll = self._train_avg_log_likelihood * n
        return -2 * total_ll + 2 * p

    @property
    def bic_(self) -> Optional[float]:
        """
        Critério de Informação Bayesiano (menor é melhor).
        Nota metodológica: O número de parâmetros penalizados baseia-se em uma aproximação
        para a dimensionalidade efetiva da mistura na hiperesfera.
        """
        if not self.is_fitted():
            return None
            
        if self._train_avg_log_likelihood is None or not np.isfinite(self._train_avg_log_likelihood):
            return None

        n = self._params.sample_count
        k = self._params.n_components
        d = self._params.dim
        
        # Graus de liberdade aproximados: K*D + K - 1
        p = k * d + k - 1 
        total_ll = self._train_avg_log_likelihood * n
        return -2 * total_ll + p * float(np.log(n))

    @classmethod
    def fit_best_model(
        cls, 
        embeddings: ArrayLike, 
        component_candidates: Iterable[int], 
        criterion: str = "bic", 
        **kwargs
    ) -> "TypicalityModel":
        """
        Testa múltiplos valores de n_components iterativamente e retorna a instância
        do modelo que minimiza o critério escolhido ('aic' ou 'bic').
        Ignora componentes que falham numericamente ou que produzem critérios não finitos.

        Args:
            embeddings: Matriz de dados de treino.
            component_candidates: Lista de inteiros com os valores de K a serem testados.
            criterion: String 'aic' ou 'bic'.
            **kwargs: Demais hiperparâmetros passados ao construtor da classe.
        """
        criterion = criterion.lower()
        if criterion not in {"aic", "bic"}:
            raise ValueError("O argumento 'criterion' deve ser 'aic' ou 'bic'.")

        # Normaliza os candidatos para evitar duplicidade e garantir ordem de execução
        component_candidates = sorted(set(int(k) for k in component_candidates))

        best_model = None
        best_score = float('inf')
        errors = []

        for k in component_candidates:
            model = cls(n_components=k, **kwargs)
            try:
                model.fit(embeddings)
                score = model.bic_ if criterion == "bic" else model.aic_
                
                if score is None or not np.isfinite(score):
                    errors.append((k, "Critério resultante não é finito ou é nulo."))
                    continue

                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                errors.append((k, str(e)))
                continue
                
        if best_model is None:
            raise RuntimeError(
                f"Nenhum modelo convergiu com sucesso. Histórico de falhas por K: {errors}"
            )
            
        return best_model

    def get_degenerate_components(self, weight_threshold: float = 1e-3) -> List[int]:
        """
        Identifica componentes da mistura que podem estar degenerados.
        Um componente é considerado degenerado se o seu peso for extremamente baixo
        (indicando cluster vazio) ou se a concentração kappa atingir o teto máximo
        (indicando uma singularidade ou um único ponto repetido).
        """
        self._ensure_fitted()
        assert self._params is not None
        
        degenerates = []
        for j in range(self._params.n_components):
            w = self._params.weights[j]
            k = self._params.kappas[j]
            if w < weight_threshold or k >= self.max_kappa:
                degenerates.append(j)
                
        return degenerates

    def summary(self, degenerate_weight_threshold: float = 1e-3) -> dict:
        """
        Retorna um dicionário com estatísticas chave sobre o ajuste do modelo.
        Extremamente útil para logs, auditoria ou monitoramento em produção.
        """
        if not self.is_fitted():
            return {"status": "unfitted"}
            
        assert self._params is not None
        
        return {
            "status": "fitted",
            "model_name": self.model_name,
            "population_label": self.population_label,
            "converged": self._converged_,
            "n_iter": self._n_iter_,
            "sample_count": self._params.sample_count,
            "avg_train_log_likelihood": self._train_avg_log_likelihood,
            "aic": self.aic_,
            "bic": self.bic_,
            "n_components_configured": self.n_components,
            "n_components_fitted": self._params.n_components,
            "weights": self._params.weights.tolist(),
            "kappas": self._params.kappas.tolist(),
            "center_embeddings": self.center_embeddings,
            "mean_vector_norm": float(np.linalg.norm(self._mean_vector)) if self._mean_vector is not None else None,
            "has_percentile_reference": self.has_percentile_reference(),
            "degenerate_weight_threshold": degenerate_weight_threshold,
            "degenerate_components_indices": self.get_degenerate_components(weight_threshold=degenerate_weight_threshold)
        }

    # =========================================================================
    # SERIALIZAÇÃO E DESSERIALIZAÇÃO JSON
    # =========================================================================

    def to_dict(self) -> dict:
        """
        Exporta o estado e os hiperparâmetros do modelo ajustado para um formato
        de dicionário estritamente compatível com serialização JSON.
        """
        if not self.is_fitted():
            raise RuntimeError("Não é possível exportar um modelo que ainda não foi treinado.")
            
        assert self._params is not None
            
        return {
            "format_version": 1,
            "hyperparameters": {
                "population_label": self.population_label,
                "model_name": self.model_name,
                "eps": self.eps,
                "center_embeddings": self.center_embeddings,
                "n_components": self.n_components,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
                "min_component_weight": self.min_component_weight,
                "min_kappa": self.min_kappa,
                "max_kappa": self.max_kappa,
            },
            "learned_params": {
                "mus": self._params.mus.tolist(),
                "kappas": self._params.kappas.tolist(),
                "weights": self._params.weights.tolist(),
                "dim": self._params.dim,
                "sample_count": self._params.sample_count,
                "n_components": self._params.n_components,
            },
            "state": {
                "mean_vector": self._mean_vector.tolist() if self._mean_vector is not None else None,
                "train_log_likelihoods": self._train_log_likelihoods.tolist() if self._train_log_likelihoods is not None else None,
                "train_avg_log_likelihood": self._train_avg_log_likelihood,
                "n_iter_": self._n_iter_,
                "converged_": self._converged_
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TypicalityModel":
        """
        Reconstrói uma instância de TypicalityModel com seu estado perfeitamente
        restaurado a partir de um dicionário gerado pelo método `to_dict`.
        """
        version = data.get("format_version", 1)
        if version != 1:
            raise ValueError(f"Versão de formato não suportada: {version}")

        model = cls(**data["hyperparameters"])
        
        lp = data["learned_params"]
        model._params = _MoVMFParams(
            mus=np.array(lp["mus"], dtype=np.float64),
            kappas=np.array(lp["kappas"], dtype=np.float64),
            weights=np.array(lp["weights"], dtype=np.float64),
            dim=lp["dim"],
            sample_count=lp["sample_count"],
            n_components=lp["n_components"]
        )
        
        st = data["state"]
        
        model._mean_vector = (
            np.array(st["mean_vector"], dtype=np.float64) 
            if st["mean_vector"] is not None 
            else None
        )
        
        model._train_log_likelihoods = (
            np.array(st["train_log_likelihoods"], dtype=np.float64) 
            if st["train_log_likelihoods"] is not None 
            else None
        )
        
        model._train_avg_log_likelihood = st["train_avg_log_likelihood"]
        model._n_iter_ = st["n_iter_"]
        model._converged_ = st["converged_"]
        
        return model

    # =========================================================================
    # API PÚBLICA DE TREINO E INFERÊNCIA
    # =========================================================================

    def fit(self, embeddings: ArrayLike) -> "TypicalityModel":
        """
        Ajusta os parâmetros da mistura MoVMF aos dados fornecidos.

        Fluxo:
        1. Valida e garante a normalização L2.
        2. Calcula a média global (mean_vector) se centralização estiver ativa.
        3. Se n_components == 1, realiza aproximação fechada.
        4. Se n_components > 1, treina via Expectation-Maximization (EM).
        5. Computa as estatísticas empíricas do conjunto de treino.
        """
        x_raw = self._validate_and_normalize_2d(embeddings)

        if x_raw.shape[0] < 2:
            raise ValueError("São necessárias pelo menos 2 amostras para treinar o TypicalityModel.")

        n_samples, dim = x_raw.shape

        if self.n_components > n_samples:
            raise ValueError(
                f"O número de componentes ({self.n_components}) não pode ser maior "
                f"que a quantidade de amostras de treino ({n_samples})."
            )

        if self.center_embeddings:
            self._mean_vector = np.mean(x_raw, axis=0).astype(np.float64)
        else:
            self._mean_vector = np.zeros(dim, dtype=np.float64)

        x = self._apply_training_transform(x_raw)

        if self.n_components == 1:
            self._fit_single_component(x)
            self._n_iter_ = 1
            self._converged_ = True
        else:
            self._fit_mixture_em(x)

        train_ll = self.score_samples(x_raw)
        if not np.all(np.isfinite(train_ll)):
            raise RuntimeError("Instabilidade detectada: O modelo ajustado gerou log-likelihoods não finitos nos dados de treino.")

        self._train_log_likelihoods = np.sort(train_ll.astype(np.float64))
        self._train_avg_log_likelihood = float(np.mean(train_ll))

        return self

    def is_fitted(self) -> bool:
        """Retorna True se o modelo possui parâmetros ajustados em memória."""
        return self._params is not None

    def has_percentile_reference(self) -> bool:
        """Retorna True se a matriz de referência empírica do treino existe."""
        return self._train_log_likelihoods is not None

    def score_samples(self, embeddings: ArrayLike) -> np.ndarray:
        """
        Calcula o log-likelihood total da mistura para cada embedding fornecido.
        Aplica a mesma transformação (centralização geométrica) aprendida no fit().
        """
        _, x_transformed = self._prepare_inference_matrix(embeddings)
        
        params = self._params
        assert params is not None

        if x_transformed.shape[1] != params.dim:
            raise ValueError(
                f"A dimensão informada na inferência ({x_transformed.shape[1]}) é diferente "
                f"da aprendida no treinamento ({params.dim})."
            )

        log_comp = self._component_logpdf_matrix(x_transformed)
        if not np.all(np.isfinite(log_comp)):
            raise RuntimeError("Instabilidade numérica durante a inferência: a matriz log_comp contém valores não finitos.")

        ll = logsumexp(log_comp, axis=1)
        if not np.all(np.isfinite(ll)):
            raise RuntimeError("Instabilidade numérica durante a inferência: log-likelihoods resultantes não são finitos.")

        return ll

    def get_percentiles(self, embeddings: ArrayLike) -> np.ndarray:
        """
        Retorna o percentil empírico (de 0.0 a 100.0) de cada embedding comparando
        o seu log-likelihood gerado contra a distribuição de log-likelihoods salvos do treino.
        Usa interpolação no meio dos empates (mid-rank) para maior precisão estatística.
        """
        self._ensure_fitted()
        if self._train_log_likelihoods is None:
            raise RuntimeError(
                "O TypicalityModel não possui a referência de percentis empíricos de treino."
            )

        ll = self.score_samples(embeddings)
        ref = self._train_log_likelihoods
        n = len(ref)

        left = np.searchsorted(ref, ll, side="left")
        right = np.searchsorted(ref, ll, side="right")
        mid_rank = (left + right) / 2.0

        percentiles = 100.0 * mid_rank / n
        return percentiles.astype(np.float64)

    def get_component_posteriors(self, embeddings: ArrayLike) -> np.ndarray:
        """
        Retorna uma matriz (N, K) contendo as probabilidades posteriores de cada componente
        da mistura dado cada embedding. A soma por linha garante ser igual a 1.0.
        """
        _, x_transformed = self._prepare_inference_matrix(embeddings)

        log_comp = self._component_logpdf_matrix(x_transformed)
        if not np.all(np.isfinite(log_comp)):
            raise RuntimeError("Erro ao calcular posteriors: matriz de pdf das componentes corrompida (não finita).")

        log_norm = logsumexp(log_comp, axis=1, keepdims=True)
        responsibilities = np.exp(log_comp - log_norm)

        if not np.all(np.isfinite(responsibilities)):
            raise RuntimeError("Erro numérico: a matriz resultante de probabilidades posteriores (responsibilities) contém falhas.")

        return responsibilities

    def get_typicality_info(self, embedding: ArrayLike) -> TypicalityInfo:
        """
        Avalia um único embedding biométrico e encapsula suas métricas, 
        estatísticas e propriedades do modelo no Data Transfer Object `TypicalityInfo`.
        """
        self._ensure_fitted()
        x_raw = self._validate_and_normalize_1d(embedding)
        
        # Converte em batch de tamanho 1 para usar as rotinas vetorizadas
        x_raw_2d = x_raw.reshape(1, -1)
        _, x_transformed = self._prepare_inference_matrix(x_raw_2d)
        
        x = x_transformed[0]
        params = self._params
        assert params is not None

        post = self.get_component_posteriors(x_raw_2d)[0]
        dominant_idx = int(np.argmax(post))

        mu_dom = params.mus[dominant_idx]
        kappa_dom = float(params.kappas[dominant_idx])
        weight_dom = float(params.weights[dominant_idx])

        cosine_to_dom = float(np.clip(np.dot(x, mu_dom), -1.0, 1.0))
        log_likelihood = float(self.score_samples(x_raw_2d)[0])
        percentile = float(self.get_percentiles(x_raw_2d)[0])

        mean_vector_norm = float(np.linalg.norm(self._mean_vector)) if self._mean_vector is not None else 0.0

        return TypicalityInfo(
            model_name=self.model_name,
            population_label=self.population_label,
            embedding_dim=params.dim,
            log_likelihood=log_likelihood,
            percentile=percentile,
            typicality_score=percentile,
            sample_count=params.sample_count,
            is_fitted=True,
            avg_log_likelihood_train=self._train_avg_log_likelihood,
            converged=self._converged_,
            center_embeddings_used=self.center_embeddings,
            mean_vector_norm=mean_vector_norm,
            n_components=params.n_components,
            dominant_component_index=dominant_idx,
            dominant_component_posterior=float(post[dominant_idx]),
            dominant_component_weight=weight_dom,
            dominant_component_kappa=kappa_dom,
            cosine_to_dominant_mean_direction=cosine_to_dom,
            kappa=kappa_dom,
            cosine_to_mean_direction=cosine_to_dom,
            component_posteriors=post.tolist(),
            component_weights=params.weights.tolist(),
            component_kappas=params.kappas.tolist(),
            notes=(
                f"dominant_component={dominant_idx}, "
                f"posterior={post[dominant_idx]:.6f}, "
                f"weight={weight_dom:.6f}"
            )
        )

    def get_typicality_infos(self, embeddings: ArrayLike) -> list[TypicalityInfo]:
        """
        Executa a inferência e retorna uma lista independente de Data Transfer Objects 
        `TypicalityInfo` para um batch contendo múltiplos embeddings.
        """
        x_raw, x_transformed = self._prepare_inference_matrix(embeddings)
        params = self._params
        assert params is not None

        posts = self.get_component_posteriors(x_raw)
        lls = self.score_samples(x_raw)
        pcts = self.get_percentiles(x_raw)

        mean_vector_norm = float(np.linalg.norm(self._mean_vector)) if self._mean_vector is not None else 0.0

        infos: list[TypicalityInfo] = []
        for i in range(len(x_raw)):
            dominant_idx = int(np.argmax(posts[i]))
            mu_dom = params.mus[dominant_idx]
            kappa_dom = float(params.kappas[dominant_idx])
            weight_dom = float(params.weights[dominant_idx])
            cosine_to_dom = float(np.clip(np.dot(x_transformed[i], mu_dom), -1.0, 1.0))

            infos.append(
                TypicalityInfo(
                    model_name=self.model_name,
                    population_label=self.population_label,
                    embedding_dim=params.dim,
                    log_likelihood=float(lls[i]),
                    percentile=float(pcts[i]),
                    typicality_score=float(pcts[i]),
                    sample_count=params.sample_count,
                    is_fitted=True,
                    avg_log_likelihood_train=self._train_avg_log_likelihood,
                    converged=self._converged_,
                    center_embeddings_used=self.center_embeddings,
                    mean_vector_norm=mean_vector_norm,
                    n_components=params.n_components,
                    dominant_component_index=dominant_idx,
                    dominant_component_posterior=float(posts[i, dominant_idx]),
                    dominant_component_weight=weight_dom,
                    dominant_component_kappa=kappa_dom,
                    cosine_to_dominant_mean_direction=cosine_to_dom,
                    kappa=kappa_dom,
                    cosine_to_mean_direction=cosine_to_dom,
                    component_posteriors=posts[i].tolist(),
                    component_weights=params.weights.tolist(),
                    component_kappas=params.kappas.tolist(),
                    notes=(
                        f"dominant_component={dominant_idx}, "
                        f"posterior={posts[i, dominant_idx]:.6f}, "
                        f"weight={weight_dom:.6f}"
                    ),
                )
            )
        return infos

    # =========================================================================
    # ACESSO DE PROPRIEDADES (READ-ONLY)
    # =========================================================================

    @property
    def mean_vector(self) -> Optional[np.ndarray]:
        if self._mean_vector is None:
            return None
        return self._mean_vector.copy()

    @property
    def n_iter_(self) -> Optional[int]:
        return self._n_iter_

    @property
    def converged_(self) -> Optional[bool]:
        return self._converged_

    @property
    def component_weights(self) -> Optional[np.ndarray]:
        if self._params is None:
            return None
        return self._params.weights.copy()

    @property
    def component_kappas(self) -> Optional[np.ndarray]:
        if self._params is None:
            return None
        return self._params.kappas.copy()

    @property
    def component_mus(self) -> Optional[np.ndarray]:
        if self._params is None:
            return None
        return self._params.mus.copy()

    # =========================================================================
    # FIT INTERNO (EM E K-MEANS++)
    # =========================================================================

    def _fit_single_component(self, x: np.ndarray) -> None:
        """Procedimento simplificado para treinar um modelo de vMF univariado."""
        n_samples, dim = x.shape
        resultant = np.sum(x, axis=0)
        resultant_norm = np.linalg.norm(resultant)

        if resultant_norm < self.eps or not np.isfinite(resultant_norm):
            mu = np.zeros(dim, dtype=np.float64)
            mu[0] = 1.0
            r_bar = 0.0
        else:
            mu = resultant / resultant_norm
            r_bar = resultant_norm / n_samples

        kappa = self._estimate_kappa(r_bar=r_bar, dim=dim)

        self._params = _MoVMFParams(
            mus=mu.reshape(1, -1).astype(np.float64),
            kappas=np.array([float(kappa)], dtype=np.float64),
            weights=np.array([1.0], dtype=np.float64),
            dim=int(dim),
            sample_count=int(n_samples),
            n_components=1,
        )

    def _fit_mixture_em(self, x: np.ndarray) -> None:
        """
        Núcleo do Algoritmo Expectation-Maximization (EM) para misturas vMF.
        Garante verificação exata de overflow/underflow em cada passo da iteração.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, dim = x.shape
        k = self.n_components

        mus, kappas, weights = self._initialize_mixture_params(x, rng)
        prev_avg_ll = -np.inf
        converged = False
        n_iter_done = 0

        for it in range(1, self.max_iter + 1):
            
            # -----------------------------------------------------------------
            # E-step
            # -----------------------------------------------------------------
            log_comp = self._component_logpdf_matrix_from_params(x, mus, kappas, weights)
            if not np.all(np.isfinite(log_comp)):
                raise RuntimeError("Instabilidade detectada no EM (E-step): matriz log_comp corrompida.")

            log_px = logsumexp(log_comp, axis=1)
            if not np.all(np.isfinite(log_px)):
                raise RuntimeError("Instabilidade detectada no EM (E-step): vetor log_px (probabilidade marginal) corrompido.")

            responsibilities = np.exp(log_comp - log_px[:, None])
            if not np.all(np.isfinite(responsibilities)):
                raise RuntimeError("Instabilidade detectada no EM (E-step): matriz de responsibilities não finita.")

            avg_ll = float(np.mean(log_px))
            if not np.isfinite(avg_ll):
                raise RuntimeError("Instabilidade detectada no EM (E-step): log-likelihood médio do ciclo não finito.")

            if it > 1 and abs(avg_ll - prev_avg_ll) < self.tol:
                converged = True
                n_iter_done = it
                break
                
            prev_avg_ll = avg_ll

            # -----------------------------------------------------------------
            # M-step
            # -----------------------------------------------------------------
            nk = responsibilities.sum(axis=0)
            nk = np.maximum(nk, self.eps)
            
            weights = nk / n_samples
            weights = np.maximum(weights, self.min_component_weight)
            weights_sum = np.sum(weights)

            if not np.isfinite(weights_sum) or weights_sum <= 0:
                weights = np.full(k, 1.0 / k, dtype=np.float64)
            else:
                weights = weights / weights_sum

            new_mus = np.zeros((k, dim), dtype=np.float64)
            new_kappas = np.zeros(k, dtype=np.float64)

            for j in range(k):
                if nk[j] <= self.eps or not np.isfinite(nk[j]):
                    new_mus[j] = x[int(rng.integers(0, n_samples))]
                    new_kappas[j] = self.min_kappa
                    continue

                resultant = responsibilities[:, j] @ x
                resultant_norm = float(np.linalg.norm(resultant))

                if resultant_norm < self.eps or not np.isfinite(resultant_norm):
                    new_mus[j] = x[int(rng.integers(0, n_samples))]
                    r_bar = 0.0
                else:
                    new_mus[j] = resultant / resultant_norm
                    r_bar = resultant_norm / nk[j]

                kappa_j = self._estimate_kappa(r_bar=float(r_bar), dim=dim)
                if not np.isfinite(kappa_j):
                    kappa_j = self.min_kappa
                    
                new_kappas[j] = float(np.clip(kappa_j, self.min_kappa, self.max_kappa))

            mus = self._renormalize_rows(new_mus)
            kappas = new_kappas
            n_iter_done = it

        if not (np.all(np.isfinite(mus)) and np.all(np.isfinite(kappas)) and np.all(np.isfinite(weights))):
            raise RuntimeError("Instabilidade fatal: Parâmetros computados ao final da mistura EM não são finitos.")

        self._params = _MoVMFParams(
            mus=mus.astype(np.float64),
            kappas=kappas.astype(np.float64),
            weights=weights.astype(np.float64),
            dim=int(dim),
            sample_count=int(n_samples),
            n_components=int(k),
        )
        self._n_iter_ = n_iter_done
        self._converged_ = converged

    def _initialize_mixture_params(
        self, x: np.ndarray, rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spherical K-Means++ Initialization:
        Seleciona sementes garantindo que os clusters iniciais sejam separados 
        angularmente pelo máximo de distância cosseno possível.
        """
        n_samples, dim = x.shape
        k = self.n_components

        mus = np.zeros((k, dim), dtype=np.float64)
        mus[0] = x[int(rng.integers(0, n_samples))]

        for j in range(1, k):
            sims = x @ mus[:j].T
            max_sim = np.max(sims, axis=1)
            dist = 1.0 - max_sim
            dist = np.maximum(dist, 0.0)
            
            dist_sum = np.sum(dist)
            if dist_sum > 0:
                probs = dist / dist_sum
                mus[j] = x[rng.choice(n_samples, p=probs)]
            else:
                mus[j] = x[int(rng.integers(0, n_samples))]

        mus = self._renormalize_rows(mus)
        similarities = x @ mus.T
        labels = np.argmax(similarities, axis=1)

        kappas = np.zeros(k, dtype=np.float64)
        weights = np.zeros(k, dtype=np.float64)

        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                mus[j] = x[int(rng.integers(0, n_samples))]
                kappas[j] = self.min_kappa
                weights[j] = 1.0 / k
                continue

            cluster_x = x[idx]
            weights[j] = len(idx) / n_samples
            resultant = cluster_x.sum(axis=0)
            resultant_norm = float(np.linalg.norm(resultant))

            if resultant_norm < self.eps or not np.isfinite(resultant_norm):
                mus[j] = x[int(rng.integers(0, n_samples))]
                r_bar = 0.0
            else:
                mus[j] = resultant / resultant_norm
                r_bar = resultant_norm / len(idx)

            kappa_j = self._estimate_kappa(r_bar=float(r_bar), dim=dim)
            if not np.isfinite(kappa_j): 
                kappa_j = self.min_kappa
                
            kappas[j] = float(np.clip(kappa_j, self.min_kappa, self.max_kappa))

        weights = np.maximum(weights, self.min_component_weight)
        weights_sum = np.sum(weights)
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            weights = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            weights = weights / weights_sum

        return self._renormalize_rows(mus), kappas, weights

    # =========================================================================
    # HELPER METHODS (Validação, Transformação e Matemática Base)
    # =========================================================================
    
    def _prepare_inference_matrix(self, embeddings: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centraliza rotinas de pré-processamento para métodos de inferência (DRY).
        Garante que os dados crus sejam validados e a transformação de geometria seja segura.
        """
        self._ensure_fitted()
        x_raw = self._validate_and_normalize_2d(embeddings)
        x_transformed = self._apply_inference_transform(x_raw)
        return x_raw, x_transformed

    def _component_logpdf_matrix(self, x: np.ndarray) -> np.ndarray:
        params = self._params
        assert params is not None
        return self._component_logpdf_matrix_from_params(x, params.mus, params.kappas, params.weights)

    def _component_logpdf_matrix_from_params(
        self, x: np.ndarray, mus: np.ndarray, kappas: np.ndarray, weights: np.ndarray,
    ) -> np.ndarray:
        dot = np.clip(x @ mus.T, -1.0, 1.0)
        log_cs = np.array(
            [self._log_vmf_normalizing_constant(dim=x.shape[1], kappa=float(kj)) for kj in kappas],
            dtype=np.float64,
        )
        
        if not np.all(np.isfinite(log_cs)):
            raise RuntimeError("Instabilidade numérica: constantes de normalização vMF não finitas.")

        safe_weights = np.maximum(weights, self.eps)
        log_weights = np.log(safe_weights)
        
        if not np.all(np.isfinite(log_weights)):
            raise RuntimeError("Instabilidade numérica: pesos logarítmicos não finitos.")

        return log_weights[None, :] + log_cs[None, :] + dot * kappas[None, :]

    def _ensure_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("O TypicalityModel ainda não foi treinado. Por favor, chame fit() antes.")

    def _validate_and_normalize_2d(self, embeddings: ArrayLike) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 1: 
            x = x.reshape(1, -1)
            
        if x.ndim != 2: 
            raise ValueError("O formato esperado para os embeddings é (N, D) ou (D,).")
            
        return self._renormalize_rows(x)

    def _validate_and_normalize_1d(self, embedding: ArrayLike) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float64)
        if x.ndim == 2:
            if x.shape[0] != 1: 
                raise ValueError("Espera-se fornecer apenas um único embedding nesta chamada.")
            x = x[0]
            
        if x.ndim != 1: 
            raise ValueError("O formato de entrada deve ser estritamente um vetor 1D.")
            
        norm = max(np.linalg.norm(x), self.eps)
        return x / norm

    def _apply_training_transform(self, x_raw: np.ndarray) -> np.ndarray:
        if self.center_embeddings:
            if self._mean_vector is None:
                raise RuntimeError("Falha interna de arquitetura: mean_vector foi perdido antes da transformação de treino.")
            return self._renormalize_rows(x_raw - self._mean_vector.reshape(1, -1))
            
        return self._renormalize_rows(x_raw.copy())

    def _apply_inference_transform(self, x_raw: np.ndarray) -> np.ndarray:
        if self.center_embeddings:
            if self._mean_vector is None:
                raise RuntimeError("Falha de inferência: modelo requer center_embeddings, mas mean_vector não foi instanciado.")
            return self._renormalize_rows(x_raw - self._mean_vector.reshape(1, -1))
            
        return self._renormalize_rows(x_raw.copy())

    def _renormalize_rows(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        out = x / np.maximum(norms, self.eps)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("Renormalização gerou valores não finitos.")
        return out

    def _estimate_kappa(self, r_bar: float, dim: int) -> float:
        r_bar = float(np.clip(r_bar, 0.0, 1.0 - 1e-10))
        if not np.isfinite(r_bar) or r_bar < 1e-8: 
            return self.min_kappa

        denom = max(1e-12, (1.0 - r_bar**2))
        kappa = (r_bar * dim - r_bar**3) / denom
        
        if not np.isfinite(kappa): 
            return self.min_kappa
            
        kappa = max(kappa, self.min_kappa)

        for _ in range(30):
            a = self._A_d(kappa, dim)
            if not np.isfinite(a): 
                return self.min_kappa
                
            diff = a - r_bar
            if abs(diff) < 1e-8: 
                break

            deriv = 1.0 - a**2 - ((dim - 1) / max(kappa, 1e-12)) * a
            if not np.isfinite(deriv) or abs(deriv) < 1e-12: 
                break

            new_kappa = kappa - diff / deriv
            if not np.isfinite(new_kappa) or new_kappa <= 0:
                new_kappa = max(kappa / 2.0, self.min_kappa)

            if abs(new_kappa - kappa) < 1e-8:
                kappa = new_kappa
                break
            kappa = new_kappa

        if not np.isfinite(kappa): 
            return self.min_kappa
            
        return float(np.clip(kappa, self.min_kappa, self.max_kappa))

    def _A_d(self, kappa: float, dim: int) -> float:
        if kappa <= 0 or not np.isfinite(kappa): 
            return 0.0
            
        nu = dim / 2.0 - 1.0
        num = ive(nu + 1.0, kappa)
        den = ive(nu, kappa)
        
        if not np.isfinite(num) or not np.isfinite(den) or abs(den) < self.eps: 
            return 0.0
            
        val = num / den
        return float(val) if np.isfinite(val) else 0.0

    def _log_vmf_normalizing_constant(self, dim: int, kappa: float) -> float:
        from math import lgamma, log, pi
        if not np.isfinite(kappa): 
            return np.nan
            
        if kappa <= 1e-3:
            return float(lgamma(dim / 2.0) - log(2.0) - (dim / 2.0) * log(pi))

        nu = dim / 2.0 - 1.0
        ive_val = ive(nu, kappa)
        
        if not np.isfinite(ive_val) or ive_val <= 0:
            return float(lgamma(dim / 2.0) - log(2.0) - (dim / 2.0) * log(pi))

        log_bessel = kappa + np.log(max(ive_val, self.eps))
        val = nu * log(kappa) - (dim / 2.0) * log(2.0 * pi) - log_bessel
        
        return float(val) if np.isfinite(val) else float(lgamma(dim / 2.0) - log(2.0) - (dim / 2.0) * log(pi))