# faceaudit/services/movmf_typicality_model.py

"""
Modelo paramétrico de typicality baseado em mistura de distribuições von Mises-Fisher (MoVMF)
para embeddings L2-normalizados na hiperesfera.

Fornece densidade probabilística explícita p_MoVMF(x) = Σ π_m C_d(κ_m) exp(κ_m μ_mᵀ x).
Backend alternativo ao `KNNTypicalityModel` (não-paramétrico local). Comparação dos dois
backends é cientificamente relevante; ver `docs/typicality_models.md`.

NOTAS METODOLÓGICAS IMPORTANTES (ver docstrings dos métodos para detalhes):

1. CENTRALIZAÇÃO: A centralização euclidiana seguida de renormalização L2 *não*
   preserva a métrica esférica e distorce a interpretação probabilística da vMF.
   Por isso, o default agora é "none". Para mitigar hubness, oferecemos a opção
   "frechet" (centralização geodésica via média intrínseca esférica), que preserva
   a estrutura. A opção "euclidean_legacy" é mantida apenas para retrocompatibilidade
   e emite um warning.

2. AIC/BIC: Os graus de liberdade são contados como K*(D-1) para as direções
   médias (que vivem em S^{D-1}, não em R^D), K para os kappas, e (K-1) para os
   pesos da mistura. Total: K*D - 1.

3. PERCENTIL vs LOG-LIKELIHOOD: O percentil empírico é uma estatística de rank,
   apropriada para comunicação ("face no 5º percentil"). Para cálculo de LR
   forense, sempre use a log-likelihood absoluta (campo `log_likelihood`).
   O alias `typicality_score` é mantido apenas para retrocompatibilidade.

4. DETECÇÃO DE FORA-DA-DISTRIBUIÇÃO: O modelo agora reporta explicitamente quando
   um embedding cai fora do range observado em treino, evitando que log-likelihoods
   extrapolados sejam interpretados como tipicidade real.

5. ESTIMAÇÃO INTRA-CLASSE: Este modelo estima apenas a densidade marginal de
   identidades (p(x | H_d)). Para construir LR generativo full, é necessário
   estimar adicionalmente kappa_intra a partir de um conjunto auxiliar com múltiplas
   fotos por pessoa (não suportado por este módulo).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Iterable, Sequence, Union, Tuple, List, Literal

import numpy as np
from scipy.special import ive, logsumexp

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.models.typicality_info import TypicalityInfo
from faceaudit.models.typicality.typicality_model_summary import (
    TypicalityModelSummary,
)
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)
from faceaudit.services.unit_norm_validator import (
    apply_unit_norm_policy,
)


ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]

CenteringMode = Literal["none", "frechet", "euclidean_legacy"]


@dataclass
class _MoVMFParams:
    mus: np.ndarray          # (K, D) — cada linha em S^{D-1}
    kappas: np.ndarray       # (K,)
    weights: np.ndarray      # (K,) — somam 1
    dim: int
    sample_count: int
    n_components: int        # K efetivo após eventual drop de componentes mortos


class MoVMFTypicalityModel:
    """
    MoVMF para embeddings na hiperesfera unitária.

    Configuração principal:
    - centering: "none" (default), "frechet" (geodésica), "euclidean_legacy"
    - n_components: 1 (vMF única) ou >1 (mistura via EM)
    - n_init: número de restarts aleatórios do EM (a melhor LL é mantida)

    Diagnóstico:
    - aic_, bic_, summary(), get_degenerate_components()
    - is_in_distribution: presente no DTO TypicalityInfo

    Treino:
    - fit(X) com X (N, D) L2-normalizado (será renormalizado se não estiver)
    - fit_best_model(X, candidates, criterion={'aic','bic'})
    """

    # =========================================================================
    # CONSTRUTOR
    # =========================================================================

    def __init__(
        self,
        population_label: Optional[str] = None,
        model_name: str = "MoVMFTypicalityModel",
        eps: float = 1e-12,
        centering: CenteringMode = "none",
        n_components: int = 1,
        max_iter: int = 200,
        tol: float = 1e-5,
        convergence_patience: int = 3,
        random_state: int = 42,
        n_init: int = 1,
        min_component_weight: float = 1e-6,
        min_kappa: float = 1e-6,
        max_kappa: float = 1e4,
        ood_quantile: float = 0.001,
        # PR 4: endurecimento experimental
        unit_norm_policy: UnitNormPolicy = UnitNormPolicy.NORMALIZE_SILENTLY,
        unit_norm_tolerance: float = 1e-4,
        population_weighting_mode: PopulationWeightingMode = (
            PopulationWeightingMode.FACE_LEVEL_RAW
        ),
        # Backwards-compat (deprecated):
        center_embeddings: Optional[bool] = None,
    ):
        if n_components < 1:
            raise ValueError("n_components deve ser >= 1.")
        if convergence_patience < 1:
            raise ValueError("convergence_patience deve ser >= 1.")
        if n_init < 1:
            raise ValueError("n_init deve ser >= 1.")
        if not (0.0 <= ood_quantile < 0.5):
            raise ValueError("ood_quantile deve estar em [0, 0.5).")

        # Compatibilidade com a API antiga (center_embeddings: bool)
        if center_embeddings is not None:
            warnings.warn(
                "O parâmetro 'center_embeddings' está obsoleto. Use 'centering' "
                "com valor 'none' (default), 'frechet' ou 'euclidean_legacy'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if center_embeddings:
                centering = "euclidean_legacy"
            else:
                centering = "none"

        if centering not in ("none", "frechet", "euclidean_legacy"):
            raise ValueError(
                f"centering inválido: {centering!r}. "
                "Use 'none', 'frechet' ou 'euclidean_legacy'."
            )

        if centering == "euclidean_legacy":
            warnings.warn(
                "centering='euclidean_legacy' não preserva a métrica esférica e "
                "distorce a interpretação probabilística da vMF. Mantido apenas "
                "para retrocompatibilidade. Considere 'none' ou 'frechet'.",
                UserWarning,
                stacklevel=2,
            )

        self.population_label = population_label
        self.model_name = model_name
        self.eps = eps
        self.centering: CenteringMode = centering

        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.convergence_patience = int(convergence_patience)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.min_component_weight = float(min_component_weight)
        self.min_kappa = float(min_kappa)
        self.max_kappa = float(max_kappa)
        self.ood_quantile = float(ood_quantile)
        # PR 4
        self.unit_norm_policy = (
            UnitNormPolicy.from_str(unit_norm_policy)
            if not isinstance(unit_norm_policy, UnitNormPolicy)
            else unit_norm_policy
        )
        self.unit_norm_tolerance = float(unit_norm_tolerance)
        self.population_weighting_mode = (
            PopulationWeightingMode.from_str(population_weighting_mode)
            if not isinstance(population_weighting_mode, PopulationWeightingMode)
            else population_weighting_mode
        )

        # Estado interno pós-treino
        self._params: Optional[_MoVMFParams] = None
        self._train_log_likelihoods: Optional[np.ndarray] = None
        self._reference_point: Optional[np.ndarray] = None  # ponto de referência (Fréchet ou euclidiano)
        self._train_avg_log_likelihood: Optional[float] = None
        self._n_iter_: Optional[int] = None
        self._converged_: Optional[bool] = None
        self._ood_log_likelihood_threshold: Optional[float] = None
        self._n_init_used: Optional[int] = None
        # PR 4: estado de balanceamento populacional
        self._n_input_faces: Optional[int] = None
        self._n_singleton_identities: Optional[int] = None
        # PR 5: estado do INVERSE_IDENTITY_FREQUENCY
        self._n_identities_inv_idfreq: Optional[int] = None
        self._pending_sample_weights: Optional[np.ndarray] = None
        self._train_log_likelihoods_weights: Optional[np.ndarray] = None
        self._train_log_likelihoods_cdf: Optional[np.ndarray] = None
        # P0-1 (revisão externa 2026-06-09): proveniência da referência de scores.
        # Quando `install_reference_score_calibration` é chamado (caso típico:
        # após CROSS_FIT_IDENTITY do `ReferenceScoreCalibrationService`), os
        # campos `_train_log_likelihoods*` e `_ood_log_likelihood_threshold`
        # passam a refletir a referência externa (OOF), não a in-sample do
        # `fit`. Esses dois campos auditam quem instalou a referência atual.
        self._reference_score_calibration_mode = None  # ReferenceScoreCalibrationMode | None
        self._split_fingerprint: Optional[str] = None

    # =========================================================================
    # COMPATIBILIDADE COM A API ANTIGA
    # =========================================================================

    @property
    def center_embeddings(self) -> bool:
        """[DEPRECATED] True se algum tipo de centralização está ativo."""
        return self.centering != "none"

    # =========================================================================
    # SELEÇÃO DE MODELO (AIC/BIC) E DIAGNÓSTICO
    # =========================================================================

    def _free_parameters(self) -> Optional[int]:
        """
        Graus de liberdade corretos para mistura vMF:
          - K * (D - 1) para as direções médias mu_j ∈ S^{D-1}
          - K para os kappas (>= 0)
          - (K - 1) para os pesos da mistura (somam 1)
        Total: K*D - 1.
        """
        if not self.is_fitted():
            return None
        assert self._params is not None
        k = self._params.n_components
        d = self._params.dim
        return k * (d - 1) + k + (k - 1)

    @property
    def aic_(self) -> Optional[float]:
        """Akaike (menor é melhor). Ver `_free_parameters` para a contagem."""
        if not self.is_fitted() or self._train_avg_log_likelihood is None:
            return None
        if not np.isfinite(self._train_avg_log_likelihood):
            return None
        assert self._params is not None
        n = self._params.sample_count
        p = self._free_parameters()
        assert p is not None
        total_ll = self._train_avg_log_likelihood * n
        return -2.0 * total_ll + 2.0 * p

    @property
    def bic_(self) -> Optional[float]:
        """Bayesian Information Criterion (menor é melhor)."""
        if not self.is_fitted() or self._train_avg_log_likelihood is None:
            return None
        if not np.isfinite(self._train_avg_log_likelihood):
            return None
        assert self._params is not None
        n = self._params.sample_count
        p = self._free_parameters()
        assert p is not None
        total_ll = self._train_avg_log_likelihood * n
        return -2.0 * total_ll + p * float(np.log(n))

    @classmethod
    def fit_best_model(
        cls,
        embeddings: ArrayLike,
        component_candidates: Iterable[int],
        criterion: str = "bic",
        **kwargs,
    ) -> "MoVMFTypicalityModel":
        """
        Testa múltiplos K e retorna o modelo que minimiza o critério escolhido.
        Falhas numéricas em K específicos são ignoradas e listadas em caso de erro total.
        """
        criterion = criterion.lower()
        if criterion not in {"aic", "bic"}:
            raise ValueError("criterion deve ser 'aic' ou 'bic'.")

        component_candidates = sorted({int(k) for k in component_candidates})
        best_model: Optional["MoVMFTypicalityModel"] = None
        best_score = float("inf")
        errors: List[Tuple[int, str]] = []

        for k in component_candidates:
            model = cls(n_components=k, **kwargs)
            try:
                model.fit(embeddings)
                score = model.bic_ if criterion == "bic" else model.aic_
                if score is None or not np.isfinite(score):
                    errors.append((k, "Critério não finito ou nulo."))
                    continue
                if score < best_score:
                    best_score = score
                    best_model = model
            except Exception as e:  # noqa: BLE001 - queremos capturar qualquer falha numérica/EM
                errors.append((k, str(e)))
                continue

        if best_model is None:
            raise RuntimeError(
                f"Nenhum modelo convergiu com sucesso. Falhas por K: {errors}"
            )
        return best_model

    def get_degenerate_components(self, weight_threshold: float = 1e-3) -> List[int]:
        """
        Componentes considerados degenerados (peso quase nulo OU kappa no teto).
        """
        self._ensure_fitted()
        assert self._params is not None
        out: List[int] = []
        for j in range(self._params.n_components):
            w = self._params.weights[j]
            kappa = self._params.kappas[j]
            if w < weight_threshold or kappa >= self.max_kappa:
                out.append(j)
        return out

    def summary(self, degenerate_weight_threshold: float = 1e-3) -> dict:
        """Estatísticas do modelo ajustado para auditoria/observabilidade."""
        if not self.is_fitted():
            return {"status": "unfitted"}
        assert self._params is not None
        ref_norm = (
            float(np.linalg.norm(self._reference_point))
            if self._reference_point is not None
            else None
        )
        return {
            "status": "fitted",
            "model_name": self.model_name,
            "population_label": self.population_label,
            "centering": self.centering,
            "converged": self._converged_,
            "n_iter": self._n_iter_,
            "n_init_used": self._n_init_used,
            "sample_count": self._params.sample_count,
            "avg_train_log_likelihood": self._train_avg_log_likelihood,
            "ood_log_likelihood_threshold": self._ood_log_likelihood_threshold,
            "ood_quantile": self.ood_quantile,
            "aic": self.aic_,
            "bic": self.bic_,
            "free_parameters": self._free_parameters(),
            "n_components_configured": self.n_components,
            "n_components_fitted": self._params.n_components,
            "weights": self._params.weights.tolist(),
            "kappas": self._params.kappas.tolist(),
            "reference_point_norm": ref_norm,
            "has_percentile_reference": self.has_percentile_reference(),
            "degenerate_weight_threshold": degenerate_weight_threshold,
            "degenerate_components_indices": self.get_degenerate_components(
                weight_threshold=degenerate_weight_threshold
            ),
        }

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    def to_dict(self) -> dict:
        """Exporta estado e hiperparâmetros para JSON.

        PR 4: componentes ordenados canonicamente (peso ↓ κ ↓ hash_μ)
        antes da serialização → modelos matematicamente equivalentes
        têm `to_dict()` byte-by-byte equivalente.
        """
        if not self.is_fitted():
            raise RuntimeError("Não é possível exportar um modelo não treinado.")
        assert self._params is not None
        order = self._canonical_component_order()
        return {
            "format_version": 4,
            "hyperparameters": {
                "population_label": self.population_label,
                "model_name": self.model_name,
                "eps": self.eps,
                "centering": self.centering,
                "n_components": self.n_components,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "convergence_patience": self.convergence_patience,
                "random_state": self.random_state,
                "n_init": self.n_init,
                "min_component_weight": self.min_component_weight,
                "min_kappa": self.min_kappa,
                "max_kappa": self.max_kappa,
                "ood_quantile": self.ood_quantile,
                # PR 4
                "unit_norm_policy": self.unit_norm_policy.to_str(),
                "unit_norm_tolerance": self.unit_norm_tolerance,
                "population_weighting_mode":
                    self.population_weighting_mode.to_str(),
            },
            "learned_params": {
                "mus": self._params.mus[order].tolist(),
                "kappas": self._params.kappas[order].tolist(),
                "weights": self._params.weights[order].tolist(),
                "dim": self._params.dim,
                "sample_count": self._params.sample_count,
                "n_components": self._params.n_components,
            },
            "state": {
                "reference_point": (
                    self._reference_point.tolist()
                    if self._reference_point is not None
                    else None
                ),
                "train_log_likelihoods": (
                    self._train_log_likelihoods.tolist()
                    if self._train_log_likelihoods is not None
                    else None
                ),
                "train_avg_log_likelihood": self._train_avg_log_likelihood,
                "n_iter_": self._n_iter_,
                "converged_": self._converged_,
                "ood_log_likelihood_threshold": self._ood_log_likelihood_threshold,
                "n_init_used": self._n_init_used,
                # P0-3 (format_version=4): persistência completa do estado
                # ponderado (INVERSE_IDENTITY_FREQUENCY) + da proveniência
                # da referência de scores (CROSS_FIT_IDENTITY etc.). Antes,
                # `_train_log_likelihoods_weights`/`_train_log_likelihoods_cdf`
                # e `_reference_score_calibration_mode`/`_split_fingerprint`
                # eram descartados na serialização — round-trip silenciosamente
                # devolvia rank uniforme.
                "train_log_likelihoods_weights": (
                    self._train_log_likelihoods_weights.tolist()
                    if self._train_log_likelihoods_weights is not None
                    else None
                ),
                "train_log_likelihoods_cdf": (
                    self._train_log_likelihoods_cdf.tolist()
                    if self._train_log_likelihoods_cdf is not None
                    else None
                ),
                "n_input_faces": self._n_input_faces,
                "n_identities_inv_idfreq": self._n_identities_inv_idfreq,
                "reference_score_calibration_mode": (
                    self._reference_score_calibration_mode.to_str()
                    if self._reference_score_calibration_mode is not None
                    else None
                ),
                "split_fingerprint": self._split_fingerprint,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MoVMFTypicalityModel":
        """
        Reconstrói modelo a partir de dict gerado por to_dict().
        Aceita format_version 1, 2, 3, 4.
        Valida invariantes geométricos e numéricos.
        """
        version = data.get("format_version", 1)
        if version not in (1, 2, 3, 4):
            raise ValueError(f"Versão de formato não suportada: {version}")

        hp = dict(data["hyperparameters"])

        # Migração v1 -> v2: 'center_embeddings' (bool) -> 'centering' (str)
        if version == 1:
            ce = hp.pop("center_embeddings", False)
            hp["centering"] = "euclidean_legacy" if ce else "none"
            # Campos novos da v2 com defaults seguros:
            hp.setdefault("convergence_patience", 3)
            hp.setdefault("n_init", 1)
            hp.setdefault("ood_quantile", 0.001)

        # Migração v2 -> v3 (PR 4): defaults dos novos campos
        # reproduzem comportamento V2.14.
        if version <= 2:
            hp.setdefault("unit_norm_policy", "NORMALIZE_SILENTLY")
            hp.setdefault("unit_norm_tolerance", 1e-4)
            hp.setdefault("population_weighting_mode", "FACE_LEVEL_RAW")

        # Construtor não deve emitir warnings de deprecação ao desserializar
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=(DeprecationWarning, UserWarning))
            model = cls(**hp)

        lp = data["learned_params"]
        params = _MoVMFParams(
            mus=np.array(lp["mus"], dtype=np.float64),
            kappas=np.array(lp["kappas"], dtype=np.float64),
            weights=np.array(lp["weights"], dtype=np.float64),
            dim=int(lp["dim"]),
            sample_count=int(lp["sample_count"]),
            n_components=int(lp["n_components"]),
        )

        # ---- Validação de invariantes ----
        k = params.n_components
        d = params.dim
        if params.mus.shape != (k, d):
            raise ValueError(f"mus shape {params.mus.shape} != ({k}, {d})")
        if params.kappas.shape != (k,):
            raise ValueError(f"kappas shape {params.kappas.shape} != ({k},)")
        if params.weights.shape != (k,):
            raise ValueError(f"weights shape {params.weights.shape} != ({k},)")
        if not np.all(np.isfinite(params.mus)):
            raise ValueError("mus contém valores não finitos.")
        if not np.all(np.isfinite(params.kappas)) or np.any(params.kappas < 0):
            raise ValueError("kappas contém valores não finitos ou negativos.")
        if not np.all(np.isfinite(params.weights)) or np.any(params.weights < 0):
            raise ValueError("weights contém valores não finitos ou negativos.")
        if not np.isclose(float(np.sum(params.weights)), 1.0, atol=1e-6):
            raise ValueError(
                f"weights não somam 1: soma = {float(np.sum(params.weights))}"
            )
        mu_norms = np.linalg.norm(params.mus, axis=1)
        if not np.allclose(mu_norms, 1.0, atol=1e-6):
            raise ValueError(
                f"mus não estão em S^{d-1}: norma min={mu_norms.min()}, max={mu_norms.max()}"
            )
        # ----------------------------------

        model._params = params
        st = data["state"]

        # 'reference_point' (v2) ou 'mean_vector' (v1)
        ref = st.get("reference_point", st.get("mean_vector", None))
        model._reference_point = (
            np.array(ref, dtype=np.float64) if ref is not None else None
        )

        if st.get("train_log_likelihoods") is not None:
            model._train_log_likelihoods = np.array(
                st["train_log_likelihoods"], dtype=np.float64
            )
        else:
            model._train_log_likelihoods = None

        model._train_avg_log_likelihood = st.get("train_avg_log_likelihood")
        model._n_iter_ = st.get("n_iter_")
        model._converged_ = st.get("converged_")
        model._ood_log_likelihood_threshold = st.get("ood_log_likelihood_threshold")
        model._n_init_used = st.get("n_init_used")

        # P0-3 (format_version=4): restaura estado ponderado + proveniência
        # da referência. Em pickles v1/v2/v3 esses campos não existem;
        # `st.get(...)` retorna None — comportamento idempotente ao da
        # fit() não-ponderada (rank uniforme via `get_percentiles`).
        if st.get("train_log_likelihoods_weights") is not None:
            model._train_log_likelihoods_weights = np.array(
                st["train_log_likelihoods_weights"], dtype=np.float64
            )
        if st.get("train_log_likelihoods_cdf") is not None:
            model._train_log_likelihoods_cdf = np.array(
                st["train_log_likelihoods_cdf"], dtype=np.float64
            )
        model._n_input_faces = st.get("n_input_faces")
        model._n_identities_inv_idfreq = st.get("n_identities_inv_idfreq")
        ref_mode_str = st.get("reference_score_calibration_mode")
        if ref_mode_str:
            from faceaudit.enums.reference_score_calibration_mode import (
                ReferenceScoreCalibrationMode,
            )
            model._reference_score_calibration_mode = (
                ReferenceScoreCalibrationMode.from_str(ref_mode_str)
            )
        model._split_fingerprint = st.get("split_fingerprint")

        # Computa OOD threshold se ausente (v1) e tivermos referência empírica
        if (
            model._ood_log_likelihood_threshold is None
            and model._train_log_likelihoods is not None
            and model.ood_quantile > 0
        ):
            model._ood_log_likelihood_threshold = float(
                np.quantile(model._train_log_likelihoods, model.ood_quantile)
            )

        return model

    # =========================================================================
    # API PÚBLICA DE TREINO E INFERÊNCIA
    # =========================================================================

    def fit(
        self,
        embeddings: ArrayLike,
        *,
        identity_ids: Optional[Sequence[str]] = None,
    ) -> "MoVMFTypicalityModel":
        """
        Ajusta a mistura.

        Pipeline:
          1. Validação + L2-norm.
          2. **PR 4: aplica `population_weighting_mode`**:
             - `FACE_LEVEL_RAW`: usa embeddings como vêm.
             - `IDENTITY_CENTROIDS`: agrega por identidade
               (centróide L2-renormalizado) e treina em centróides.
             - `INVERSE_IDENTITY_FREQUENCY`: NÃO implementado em V2.15;
               levanta `NotImplementedError` com referência a
               `docs/typicality_models.md` §4.1.
          3. Cálculo do ponto de referência (centralização Fréchet/euclidiana se aplicável).
          4. Single-component ou EM com n_init restarts (mantém o melhor LL).
          5. Score do treino, threshold OOD.
        """
        x_raw = self._validate_and_normalize_2d(embeddings)
        if x_raw.shape[0] < 2:
            raise ValueError("São necessárias pelo menos 2 amostras para treinar.")

        # PR 4: weighting populacional
        if self.population_weighting_mode == PopulationWeightingMode.IDENTITY_CENTROIDS:
            if identity_ids is None:
                raise ValueError(
                    "PopulationWeightingMode.IDENTITY_CENTROIDS exige "
                    "`identity_ids` paralelo a `embeddings`."
                )
            if len(identity_ids) != x_raw.shape[0]:
                raise ValueError(
                    f"len(identity_ids)={len(identity_ids)} difere de "
                    f"N_embeddings={x_raw.shape[0]}."
                )
            x_raw = self._aggregate_by_identity(x_raw, identity_ids)
        elif self.population_weighting_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
            # PR 5: EM ponderado. `sample_weights[i] = 1 / n_id_i`.
            if identity_ids is None:
                raise ValueError(
                    "PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY exige "
                    "`identity_ids` paralelo a `embeddings`."
                )
            if len(identity_ids) != x_raw.shape[0]:
                raise ValueError(
                    f"len(identity_ids)={len(identity_ids)} difere de "
                    f"N_embeddings={x_raw.shape[0]}."
                )
            ids_arr = np.asarray(identity_ids)
            unique_ids, inverse_idx, id_counts = np.unique(
                ids_arr, return_inverse=True, return_counts=True,
            )
            sample_weights = (1.0 / id_counts[inverse_idx]).astype(np.float64)
            self._n_input_faces = int(x_raw.shape[0])
            self._n_identities_inv_idfreq = int(unique_ids.shape[0])
            # Anexa pesos para o resto do fit usar
            self._pending_sample_weights = sample_weights

        n_samples, dim = x_raw.shape
        if self.n_components > n_samples:
            raise ValueError(
                f"n_components ({self.n_components}) não pode exceder amostras ({n_samples})."
            )

        self._compute_reference_point(x_raw)
        x = self._apply_transform(x_raw)

        # PR 5: pega pesos pendentes (de INVERSE_IDENTITY_FREQUENCY)
        sample_weights = getattr(self, "_pending_sample_weights", None)
        if sample_weights is not None:
            # Limpa para não vazar para outros fits
            self._pending_sample_weights = None

        if self.n_components == 1:
            self._fit_single_component(x, sample_weights=sample_weights)
            self._n_iter_ = 1
            self._converged_ = True
            self._n_init_used = 1
        else:
            self._fit_mixture_em_with_restarts(x, sample_weights=sample_weights)

        # Score do treino + estatísticas de referência
        train_ll = self.score_samples(x_raw)
        if not np.all(np.isfinite(train_ll)):
            raise RuntimeError(
                "Instabilidade: log-likelihoods de treino não são todos finitos."
            )

        # PR 5: CDF e avg_ll ponderados quando INVERSE_IDENTITY_FREQUENCY
        if sample_weights is not None:
            # Ordena por LL e alinha pesos
            order = np.argsort(train_ll)
            self._train_log_likelihoods = train_ll[order].astype(np.float64)
            sorted_w = sample_weights[order]
            self._train_log_likelihoods_weights = sorted_w
            self._train_log_likelihoods_cdf = np.cumsum(sorted_w) / sorted_w.sum()
            # avg_ll ponderado
            self._train_avg_log_likelihood = float(
                np.sum(sample_weights * train_ll) / sample_weights.sum()
            )
            if self.ood_quantile > 0.0:
                # Threshold via quantil ponderado
                idx = int(np.searchsorted(
                    self._train_log_likelihoods_cdf, self.ood_quantile, side="left"
                ))
                idx = min(max(idx, 0), len(self._train_log_likelihoods) - 1)
                self._ood_log_likelihood_threshold = float(
                    self._train_log_likelihoods[idx]
                )
            else:
                self._ood_log_likelihood_threshold = None
        else:
            self._train_log_likelihoods = np.sort(train_ll.astype(np.float64))
            self._train_log_likelihoods_weights = None
            self._train_log_likelihoods_cdf = None
            self._train_avg_log_likelihood = float(np.mean(train_ll))

            if self.ood_quantile > 0.0:
                self._ood_log_likelihood_threshold = float(
                    np.quantile(self._train_log_likelihoods, self.ood_quantile)
                )
            else:
                self._ood_log_likelihood_threshold = None

        return self

    def is_fitted(self) -> bool:
        return self._params is not None

    def has_percentile_reference(self) -> bool:
        return self._train_log_likelihoods is not None

    @property
    def reference_log_likelihoods(self) -> Optional[np.ndarray]:
        """Cópia das log-verossimilhanças de treino (já ordenadas crescentes
        pelo `fit`). Análogo a `KNNTypicalityModel.loo_distribution` —
        diagnóstico da distribuição de referência sob `H_d`. Retorna None
        antes do fit ou quando o modelo não persistiu o vetor."""
        if self._train_log_likelihoods is None:
            return None
        return self._train_log_likelihoods.copy()

    @property
    def ood_threshold(self) -> Optional[float]:
        """Limiar OOD em log-verossimilhança (`q_ood`-ésimo percentil da
        distribuição de referência). Alias unificado com
        `KNNTypicalityModel.ood_threshold` para consumo polimórfico
        (plot helper, exposição de diagnóstico)."""
        return self._ood_log_likelihood_threshold

    def score_samples(self, embeddings: ArrayLike) -> np.ndarray:
        """Log-likelihood total da mistura (vetorizado)."""
        _, x_transformed = self._prepare_inference_matrix(embeddings)
        params = self._params
        assert params is not None

        if x_transformed.shape[1] != params.dim:
            raise ValueError(
                f"Dim de inferência ({x_transformed.shape[1]}) != dim de treino ({params.dim})."
            )

        log_comp = self._component_logpdf_matrix(x_transformed)
        if not np.all(np.isfinite(log_comp)):
            raise RuntimeError("Inferência: log_comp contém não finitos.")

        ll = logsumexp(log_comp, axis=1)
        if not np.all(np.isfinite(ll)):
            raise RuntimeError("Inferência: log-likelihoods finais não finitos.")
        return ll

    def get_percentiles(
        self, embeddings: ArrayLike, *, extrapolate: bool = True
    ) -> np.ndarray:
        """
        Percentil empírico (0..100) do log-likelihood vs distribuição de
        referência. Mid-rank para empates.

        P0-2 (revisão externa 2026-06-09): quando o modelo tem CDF
        ponderada (`_train_log_likelihoods_cdf` populada pelo
        `INVERSE_IDENTITY_FREQUENCY` ou pelo
        `install_reference_score_calibration`), o percentil é calculado
        consultando a CDF — não rank uniforme. Antes desta correção,
        `get_percentiles` ignorava silenciosamente os pesos e produzia
        percentil uniforme mesmo após cross-fit.

        Se extrapolate=False, valores fora do range observado em treino
        retornam NaN (recomendado para uso forense — ver nota 3 do
        header).
        """
        self._ensure_fitted()
        if self._train_log_likelihoods is None:
            raise RuntimeError("Sem referência de percentis de treino.")

        ll = self.score_samples(embeddings)
        ref = self._train_log_likelihoods
        n = len(ref)

        cdf = self._train_log_likelihoods_cdf
        if cdf is not None:
            # CDF ponderada: rank ponderado por mid-CDF entre side="left" e
            # side="right". Mesmo padrão do KNN (`_compute_info_single`).
            left = np.searchsorted(ref, ll, side="left")
            right = np.searchsorted(ref, ll, side="right")
            left_clipped = np.clip(left - 1, 0, n - 1)
            right_clipped = np.clip(right - 1, 0, n - 1)
            left_cdf = np.where(left > 0, cdf[left_clipped], 0.0)
            right_cdf = np.where(right > 0, cdf[right_clipped], 0.0)
            percentiles = 100.0 * 0.5 * (left_cdf + right_cdf)
        else:
            # Rank uniforme (sem pesos)
            left = np.searchsorted(ref, ll, side="left")
            right = np.searchsorted(ref, ll, side="right")
            mid_rank = (left + right) / 2.0
            percentiles = 100.0 * mid_rank / n

        if not extrapolate:
            below = ll < ref[0]
            above = ll > ref[-1]
            percentiles = percentiles.astype(np.float64)
            percentiles[below | above] = np.nan

        return percentiles.astype(np.float64)

    def install_reference_score_calibration(
        self,
        *,
        reference_scores,
        reference_weights=None,
        ood_threshold=None,
        calibration_mode,
        split_fingerprint=None,
    ) -> None:
        """Sobrescreve atomicamente a referência interna de scores do modelo.

        P0-1 (revisão externa 2026-06-09): este método é o gancho usado
        pelo `ReferenceScoreCalibrationService` após `CROSS_FIT_IDENTITY`
        para instalar a distribuição OOF (out-of-fold) + threshold OOD no
        modelo persistido. Antes desta correção, o serviço retornava
        `oof_cdf`/`ood_threshold` AO LADO de um `refit_model` que ainda
        carregava a distribuição IN-SAMPLE no próprio
        `_train_log_likelihoods`/`_ood_log_likelihood_threshold` —
        `get_percentiles()` e `is_in_distribution()` consultavam os
        valores in-sample, descartando o cross-fit.

        Parâmetros:
            reference_scores: vetor 1D de scores (log-likelihood no MoVMF).
            reference_weights: pesos opcionais por score. Quando None,
                rank uniforme. Quando dados, é o caminho ponderado
                (INVERSE_IDENTITY_FREQUENCY ou OOF identitário).
            ood_threshold: limiar OOD calculado externamente. Quando None,
                deixa `is_in_distribution` permissivo.
            calibration_mode: `ReferenceScoreCalibrationMode` — registra
                como esta referência foi obtida (entra no contract
                fingerprint).
            split_fingerprint: hash do split identidade→fold (auditoria).

        Atômico: ou substitui todos os campos derivados, ou levanta erro
        sem mutar estado.
        """
        self._ensure_fitted()
        from faceaudit.enums.reference_score_calibration_mode import (
            ReferenceScoreCalibrationMode,
        )

        scores_arr = np.asarray(reference_scores, dtype=np.float64).ravel()
        if scores_arr.size == 0:
            raise ValueError(
                "install_reference_score_calibration: reference_scores vazio."
            )
        if not np.all(np.isfinite(scores_arr)):
            raise ValueError(
                "install_reference_score_calibration: reference_scores "
                "contém valores não finitos."
            )

        if reference_weights is not None:
            weights_arr = np.asarray(reference_weights, dtype=np.float64).ravel()
            if weights_arr.shape != scores_arr.shape:
                raise ValueError(
                    f"install_reference_score_calibration: shape de "
                    f"reference_weights {weights_arr.shape} ≠ scores "
                    f"{scores_arr.shape}."
                )
            if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0):
                raise ValueError(
                    "install_reference_score_calibration: reference_weights "
                    "deve conter valores não-negativos finitos."
                )
            total_w = float(weights_arr.sum())
            if total_w <= 0.0:
                raise ValueError(
                    "install_reference_score_calibration: reference_weights "
                    "soma 0."
                )
        else:
            weights_arr = None
            total_w = 0.0

        if not isinstance(calibration_mode, ReferenceScoreCalibrationMode):
            calibration_mode = ReferenceScoreCalibrationMode.from_str(
                calibration_mode
            )

        if ood_threshold is not None:
            ood_threshold_val: Optional[float] = float(ood_threshold)
            if not np.isfinite(ood_threshold_val):
                raise ValueError(
                    "install_reference_score_calibration: ood_threshold "
                    "não finito."
                )
        else:
            ood_threshold_val = None

        # Ordena scores + pesos paralelamente; computa CDF.
        order = np.argsort(scores_arr)
        scores_sorted = np.ascontiguousarray(
            scores_arr[order], dtype=np.float64
        )
        if weights_arr is not None:
            weights_sorted = np.ascontiguousarray(
                weights_arr[order], dtype=np.float64
            )
            cdf_arr = np.cumsum(weights_sorted) / total_w
        else:
            weights_sorted = None
            cdf_arr = None

        # Substitui atomicamente.
        self._train_log_likelihoods = scores_sorted
        self._train_log_likelihoods_weights = weights_sorted
        self._train_log_likelihoods_cdf = cdf_arr
        self._ood_log_likelihood_threshold = ood_threshold_val
        self._reference_score_calibration_mode = calibration_mode
        self._split_fingerprint = (
            str(split_fingerprint) if split_fingerprint else None
        )

    def is_in_distribution(self, embeddings: ArrayLike) -> np.ndarray:
        """
        Vetor booleano indicando se cada embedding está dentro do range de treino,
        usando o threshold OOD definido por ood_quantile.
        """
        self._ensure_fitted()
        ll = self.score_samples(embeddings)
        if self._ood_log_likelihood_threshold is None:
            return np.ones_like(ll, dtype=bool)
        return ll >= self._ood_log_likelihood_threshold

    def get_component_posteriors(self, embeddings: ArrayLike) -> np.ndarray:
        """Posteriores P(componente | x), shape (N, K). Linhas somam 1."""
        _, x_transformed = self._prepare_inference_matrix(embeddings)
        log_comp = self._component_logpdf_matrix(x_transformed)
        if not np.all(np.isfinite(log_comp)):
            raise RuntimeError("Posteriors: log_comp não finitos.")
        log_norm = logsumexp(log_comp, axis=1, keepdims=True)
        responsibilities = np.exp(log_comp - log_norm)
        if not np.all(np.isfinite(responsibilities)):
            raise RuntimeError("Posteriors: responsibilities não finitos.")
        return responsibilities

    def get_typicality_info(self, embedding: ArrayLike) -> TypicalityInfo:
        """Avalia um único embedding e empacota tudo num TypicalityInfo."""
        self._ensure_fitted()
        x_raw = self._validate_and_normalize_1d(embedding).reshape(1, -1)
        infos = self.get_typicality_infos(x_raw)
        return infos[0]

    def get_typicality_infos(self, embeddings: ArrayLike) -> List[TypicalityInfo]:
        """Avalia um batch de embeddings e retorna lista de TypicalityInfo."""
        x_raw, x_transformed = self._prepare_inference_matrix(embeddings)
        params = self._params
        assert params is not None

        posts = self.get_component_posteriors(x_raw)
        lls = self.score_samples(x_raw)
        pcts = self.get_percentiles(x_raw, extrapolate=True)
        in_dist = self.is_in_distribution(x_raw)

        ref_norm = (
            float(np.linalg.norm(self._reference_point))
            if self._reference_point is not None
            else 0.0
        )

        infos: List[TypicalityInfo] = []
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
                    typicality_score=float(pcts[i]),  # alias legado
                    sample_count=params.sample_count,
                    is_fitted=True,
                    avg_log_likelihood_train=self._train_avg_log_likelihood,
                    converged=self._converged_,
                    center_embeddings_used=(self.centering != "none"),
                    mean_vector_norm=ref_norm,
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
                        f"centering={self.centering}, "
                        f"in_distribution={bool(in_dist[i])}, "
                        f"dominant_component={dominant_idx}, "
                        f"posterior={posts[i, dominant_idx]:.6f}, "
                        f"weight={weight_dom:.6f}"
                    ),
                )
            )
        return infos

    # =========================================================================
    # TypicalityScoringModelProtocol (PR 1b)
    # =========================================================================
    #
    # MoVMF habilita o subset universal de score types (sem
    # `MEAN_KNN_SIMILARITY`, que é uma noção local do k-NN sem análogo
    # paramétrico aqui). O `score_semantics_id` no batch deixa explícita
    # a diferença matemática entre os dois backends que compartilham o
    # mesmo enum value `LOG_DENSITY_ESTIMATE`:
    # `LOG_LIKELIHOOD_MOVMF` (mistura paramétrica) vs
    # `LOG_DENSITY_ESTIMATE_LQ_SPHERICAL` (Loftsgaarden-Quesenberry k-NN).

    @property
    def model_type(self) -> TypicalityModelType:
        return TypicalityModelType.MOVMF

    @property
    def enabled_score_types(self) -> frozenset[TypicalityScoreType]:
        if not self.is_fitted():
            return frozenset()
        # MoVMF: TYPICALITY_PERCENTILE (sobre log-likelihood) +
        # LOG_DENSITY_ESTIMATE (universal — semantics_id distingue
        # log_likelihood paramétrico de L-Q esférico não-paramétrico).
        return frozenset({
            TypicalityScoreType.TYPICALITY_PERCENTILE,
            TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        })

    @property
    def embedding_dim(self) -> int:
        self._ensure_fitted()
        return int(self._params.dim)  # type: ignore[union-attr]

    @property
    def sample_count(self) -> int:
        self._ensure_fitted()
        return int(self._params.sample_count)  # type: ignore[union-attr]

    def _canonical_component_order(self) -> np.ndarray:
        """Índices que ordenam os componentes pela chave lex composta:
        peso decrescente → κ decrescente → hash lex de μ.

        Garante invariância do `model_contract_fingerprint` à permutação
        de componentes (mistura é simétrica em permutação).
        """
        import hashlib
        p = self._params
        assert p is not None
        mu_hashes = [
            hashlib.sha256(
                np.ascontiguousarray(p.mus[j], dtype=np.float64).tobytes(order="C")
            ).hexdigest()
            for j in range(p.n_components)
        ]
        # Chave de ordenação: (-peso, -κ, hash_μ). Quanto MENOR a chave,
        # mais "à esquerda" — peso/κ negativos invertem para decrescente.
        keys = [
            (-float(p.weights[j]), -float(p.kappas[j]), mu_hashes[j])
            for j in range(p.n_components)
        ]
        return np.array(sorted(range(p.n_components), key=lambda j: keys[j]), dtype=np.int64)

    @property
    def model_state_fingerprint(self) -> str:
        """SHA-256 do estado numérico (componentes canonicalizados,
        CDF, threshold, reference_point).

        PR 4: ordem dos componentes canonicalizada (peso ↓ κ ↓ hash_μ)
        antes do hash — invariância à permutação.
        """
        self._ensure_fitted()
        import hashlib
        p = self._params
        assert p is not None
        order = self._canonical_component_order()
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(p.mus[order], dtype=np.float64).tobytes(order="C"))
        h.update(np.ascontiguousarray(p.kappas[order], dtype=np.float64).tobytes(order="C"))
        h.update(np.ascontiguousarray(p.weights[order], dtype=np.float64).tobytes(order="C"))
        if self._train_log_likelihoods is not None:
            h.update(
                np.ascontiguousarray(self._train_log_likelihoods, dtype=np.float64).tobytes(order="C")
            )
        # P0-3 (revisão externa 2026-06-09): pesos e CDF da referência
        # ponderada entram no state fingerprint. Sem isso, dois modelos
        # com mesma distribuição mas pesos diferentes (INV_ID_FREQ vs
        # uniforme; OOF vs IN_SAMPLE) tinham o mesmo state fingerprint.
        if self._train_log_likelihoods_weights is not None:
            h.update(np.ascontiguousarray(
                self._train_log_likelihoods_weights, dtype=np.float64
            ).tobytes(order="C"))
        if self._train_log_likelihoods_cdf is not None:
            h.update(np.ascontiguousarray(
                self._train_log_likelihoods_cdf, dtype=np.float64
            ).tobytes(order="C"))
        ood = self._ood_log_likelihood_threshold
        h.update(f"ood={ood!r}".encode("utf-8"))
        # Proveniência da referência (CROSS_FIT_IDENTITY ↔ IN_SAMPLE_LEGACY
        # ↔ None pré-fit) — duas referências com mesma distribuição numérica
        # produzida por protocolos diferentes não devem compartilhar
        # fingerprint.
        if self._reference_score_calibration_mode is not None:
            h.update(
                f"calmode={self._reference_score_calibration_mode.to_str()}"
                .encode("utf-8")
            )
        if self._split_fingerprint:
            h.update(f"split={self._split_fingerprint}".encode("utf-8"))
        if self._reference_point is not None:
            h.update(
                np.ascontiguousarray(self._reference_point, dtype=np.float64).tobytes(order="C")
            )
        h.update(f"K={p.n_components};D={p.dim};N={p.sample_count}".encode("utf-8"))
        return h.hexdigest()

    @property
    def model_contract_fingerprint(self) -> str:
        """SHA-256 do contrato semântico (sem o estado bruto).

        PR 4: inclui `unit_norm_policy` e `population_weighting_mode`
        — mudanças em qualquer um deslocam o contract fingerprint
        downstream (LR architecture INHERIT rejeita).
        """
        self._ensure_fitted()
        import hashlib
        p = self._params
        assert p is not None
        semantics_ids = sorted([
            self._score_semantics_id_for(t) for t in self.enabled_score_types
        ])
        contract_parts = [
            f"model_type={TypicalityModelType.MOVMF.to_str()}",
            f"semantics_ids={','.join(semantics_ids)}",
            f"centering={self.centering}",
            f"n_components_fitted={p.n_components}",
            f"dim={p.dim}",
            f"ood_quantile={self.ood_quantile:.17g}",
            # PR 4
            f"unit_norm_policy={self.unit_norm_policy.to_str()}",
            f"population_weighting_mode={self.population_weighting_mode.to_str()}",
            # P0-3 (revisão externa 2026-06-09): protocolo de calibração da
            # referência (CROSS_FIT_IDENTITY ↔ IN_SAMPLE_LEGACY ↔ None)
            # entra no contrato. Aplicar modelo OOF a dataset que espera
            # IN_SAMPLE_LEGACY (ou vice-versa) muda interpretação dos
            # percentis e do `is_in_distribution`.
            (
                "reference_score_calibration_mode="
                + (
                    self._reference_score_calibration_mode.to_str()
                    if self._reference_score_calibration_mode is not None
                    else "none"
                )
            ),
        ]
        h = hashlib.sha256()
        h.update("\n".join(contract_parts).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _score_semantics_id_for(t: TypicalityScoreType) -> str:
        if t == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            return "LOG_LIKELIHOOD_MOVMF"
        if t == TypicalityScoreType.TYPICALITY_PERCENTILE:
            return "PERCENTILE_OF_LOG_LIKELIHOOD_MOVMF"
        raise ValueError(f"score_type sem semantics id para MoVMF: {t}")

    def compute_scores(
        self,
        embeddings: ArrayLike,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch:
        from faceaudit.protocols.typicality_scoring_model_protocol import (
            UnsupportedTypicalityScoreTypeError,
        )

        self._ensure_fitted()
        if score_type not in self.enabled_score_types:
            raise UnsupportedTypicalityScoreTypeError(
                requested=score_type,
                enabled=self.enabled_score_types,
                model_type=self.model_type,
            )

        if score_type == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            arr = self.score_samples(embeddings)
            percentile_basis = None
            ref_dist_fp = None
        elif score_type == TypicalityScoreType.TYPICALITY_PERCENTILE:
            arr = self.get_percentiles(embeddings, extrapolate=True)
            # Base do percentil MoVMF: log-likelihood. Em PR 3 isso vira
            # LOG_DENSITY_ESTIMATE (mesma identidade enum, nome novo).
            percentile_basis = TypicalityScoreType.LOG_DENSITY_ESTIMATE
            ref_dist_fp = self._reference_score_distribution_fingerprint()
        else:
            raise ValueError(f"score_type não esperado: {score_type}")

        return TypicalityScoreBatch(
            scores=np.asarray(arr, dtype=np.float64),
            n=int(arr.shape[0]),
            embedding_dim=self.embedding_dim,
            score_type=score_type,
            score_semantics_id=self._score_semantics_id_for(score_type),
            percentile_basis_score_type=percentile_basis,
            reference_score_distribution_fingerprint=ref_dist_fp,
            model_type=self.model_type,
            model_state_fingerprint=self.model_state_fingerprint,
            model_contract_fingerprint=self.model_contract_fingerprint,
        )

    def _reference_score_distribution_fingerprint(self) -> str:
        import hashlib
        self._ensure_fitted()
        if self._train_log_likelihoods is None:
            return ""
        loo = np.ascontiguousarray(self._train_log_likelihoods, dtype=np.float64)
        return hashlib.sha256(loo.tobytes(order="C")).hexdigest()

    def summary_protocol(self) -> TypicalityModelSummary:
        """Implementação do `summary()` do protocolo.

        PR 4: health checks expandidos (10 itens documentados em
        `docs/typicality_models.md` §7).
        Coexistente com o método `summary()` legado que retorna dict.
        """
        self._ensure_fitted()
        p = self._params
        assert p is not None

        warnings_list: list[str] = []
        health = "OK"

        # Health 5: log-likelihood finita
        if self._train_avg_log_likelihood is None or not np.isfinite(
            self._train_avg_log_likelihood
        ):
            warnings_list.append("avg_train_log_likelihood não-finito")
            health = "CRITICAL"
        # Health 6: convergência em pelo menos uma inicialização
        if self._converged_ is False:
            warnings_list.append("MoVMF não convergiu na inicialização final")
            health = "WARNING" if health == "OK" else health
        # Health 1: n_components ≤ effective_population_size
        if p.n_components > p.sample_count:
            warnings_list.append(
                f"n_components ({p.n_components}) > sample_count "
                f"({p.sample_count}) — modelo super-parametrizado"
            )
            health = "CRITICAL"
        # Health 3: nenhum peso abaixo de min_component_weight
        below = np.sum(p.weights < self.min_component_weight)
        if below > 0:
            warnings_list.append(
                f"{int(below)} componente(s) com peso < min_component_weight"
            )
            health = "WARNING" if health == "OK" else health
        # Health 4: nenhum kappa saturou max_kappa
        saturated = np.sum(p.kappas >= self.max_kappa - 1e-9)
        if saturated > 0:
            warnings_list.append(
                f"{int(saturated)} componente(s) com κ saturado em max_kappa "
                f"({self.max_kappa}) — pode indicar componente delta"
            )
            health = "WARNING" if health == "OK" else health
        # Health 2: componentes degenerados (peso quase nulo OU κ saturado)
        degenerate = self.get_degenerate_components(weight_threshold=1e-3)
        if degenerate:
            warnings_list.append(
                f"Componentes degenerados (índices canônicos pós-ordenação "
                f"podem diferir): {degenerate}"
            )
            health = "WARNING" if health == "OK" else health

        return TypicalityModelSummary(
            model_type=TypicalityModelType.MOVMF,
            model_name=self.model_name,
            embedding_dim=int(p.dim),
            sample_count=int(p.sample_count),
            enabled_score_types=tuple(
                sorted(self.enabled_score_types, key=lambda t: t.value)
            ),
            health_status=health,
            warnings=tuple(warnings_list),
            backend_details={
                "n_components": int(p.n_components),
                "centering": self.centering,
                "weights": p.weights.tolist(),
                "kappas": p.kappas.tolist(),
                "converged": self._converged_,
                "n_iter": self._n_iter_,
                "n_init_used": self._n_init_used,
                "avg_train_log_likelihood": self._train_avg_log_likelihood,
                "aic": self.aic_,
                "bic": self.bic_,
                "ood_quantile": float(self.ood_quantile),
                "ood_log_likelihood_threshold": self._ood_log_likelihood_threshold,
                # PR 4: novos campos auditáveis
                "unit_norm_policy": self.unit_norm_policy.to_str(),
                "unit_norm_tolerance": self.unit_norm_tolerance,
                "population_weighting_mode":
                    self.population_weighting_mode.to_str(),
                "n_input_faces": self._n_input_faces,
                "n_singleton_identities": self._n_singleton_identities,
                "random_state": self.random_state,
            },
        )

    # =========================================================================
    # ACESSO READ-ONLY
    # =========================================================================

    @property
    def reference_point(self) -> Optional[np.ndarray]:
        """Ponto usado para centralização (Fréchet ou euclidiano), ou None."""
        if self._reference_point is None:
            return None
        return self._reference_point.copy()

    @property
    def mean_vector(self) -> Optional[np.ndarray]:
        """[DEPRECATED] alias para reference_point."""
        return self.reference_point

    @property
    def n_iter_(self) -> Optional[int]:
        return self._n_iter_

    @property
    def converged_(self) -> Optional[bool]:
        return self._converged_

    @property
    def component_weights(self) -> Optional[np.ndarray]:
        return None if self._params is None else self._params.weights.copy()

    @property
    def component_kappas(self) -> Optional[np.ndarray]:
        return None if self._params is None else self._params.kappas.copy()

    @property
    def component_mus(self) -> Optional[np.ndarray]:
        return None if self._params is None else self._params.mus.copy()

    # =========================================================================
    # FIT INTERNO: SINGLE COMPONENT
    # =========================================================================

    def _fit_single_component(
        self, x: np.ndarray, *, sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        n_samples, dim = x.shape
        if sample_weights is not None:
            # Resultante e r_bar ponderados
            resultant = (sample_weights[:, None] * x).sum(axis=0)
            total_w = float(sample_weights.sum())
        else:
            resultant = np.sum(x, axis=0)
            total_w = float(n_samples)
        resultant_norm = np.linalg.norm(resultant)

        if resultant_norm < self.eps or not np.isfinite(resultant_norm):
            mu = np.zeros(dim, dtype=np.float64)
            mu[0] = 1.0
            r_bar = 0.0
        else:
            mu = resultant / resultant_norm
            r_bar = float(resultant_norm / max(total_w, self.eps))

        kappa = self._estimate_kappa_robust(r_bar=r_bar, dim=dim)
        self._params = _MoVMFParams(
            mus=mu.reshape(1, -1).astype(np.float64),
            kappas=np.array([kappa], dtype=np.float64),
            weights=np.array([1.0], dtype=np.float64),
            dim=int(dim),
            sample_count=int(n_samples),
            n_components=1,
        )

    # =========================================================================
    # FIT INTERNO: EM COM RESTARTS
    # =========================================================================

    def _fit_mixture_em_with_restarts(
        self, x: np.ndarray, *,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        """Roda n_init EMs e mantém o melhor (maior log-likelihood final)."""
        best: Optional[Tuple[float, _MoVMFParams, int, bool]] = None
        for init_idx in range(self.n_init):
            seed = self.random_state + init_idx
            try:
                params, n_iter, converged, avg_ll = self._fit_mixture_em_once(
                    x, seed=seed, sample_weights=sample_weights,
                )
            except Exception:  # noqa: BLE001
                # Esta inicialização falhou; tenta próxima.
                continue
            if best is None or avg_ll > best[0]:
                best = (avg_ll, params, n_iter, converged)

        if best is None:
            raise RuntimeError(
                f"Todas as {self.n_init} inicializações do EM falharam."
            )

        _, best_params, best_iter, best_conv = best
        self._params = best_params
        self._n_iter_ = best_iter
        self._converged_ = best_conv
        self._n_init_used = self.n_init

    def _fit_mixture_em_once(
        self, x: np.ndarray, *, seed: int,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[_MoVMFParams, int, bool, float]:
        """Uma execução completa de EM. Retorna (params, n_iter, converged, avg_ll)."""
        rng = np.random.default_rng(seed)
        n_samples, dim = x.shape
        k = self.n_components

        mus, kappas, weights = self._initialize_mixture_params(x, rng)

        prev_avg_ll = -np.inf
        consecutive_below_tol = 0
        converged = False
        n_iter_done = 0
        avg_ll = -np.inf
        total_w = (
            float(sample_weights.sum()) if sample_weights is not None
            else float(n_samples)
        )

        for it in range(1, self.max_iter + 1):
            # ---- E-step ----
            log_comp = self._component_logpdf_from(x, mus, kappas, weights)
            if not np.all(np.isfinite(log_comp)):
                raise RuntimeError("EM (E-step): log_comp não finito.")

            log_px = logsumexp(log_comp, axis=1)
            if not np.all(np.isfinite(log_px)):
                raise RuntimeError("EM (E-step): log_px não finito.")

            responsibilities = np.exp(log_comp - log_px[:, None])
            if not np.all(np.isfinite(responsibilities)):
                raise RuntimeError("EM (E-step): responsibilities não finitos.")

            # avg_ll ponderado em INVERSE_IDENTITY_FREQUENCY
            if sample_weights is not None:
                avg_ll = float(np.sum(sample_weights * log_px) / total_w)
            else:
                avg_ll = float(np.mean(log_px))
            if not np.isfinite(avg_ll):
                raise RuntimeError("EM (E-step): avg_ll não finito.")

            if it > 1 and abs(avg_ll - prev_avg_ll) < self.tol:
                consecutive_below_tol += 1
                if consecutive_below_tol >= self.convergence_patience:
                    converged = True
                    n_iter_done = it
                    break
            else:
                consecutive_below_tol = 0
            prev_avg_ll = avg_ll

            # ---- M-step ----
            mus, kappas, weights = self._em_m_step(
                x, responsibilities, mus, kappas, weights, rng,
                sample_weights=sample_weights,
            )
            n_iter_done = it

        if not (
            np.all(np.isfinite(mus))
            and np.all(np.isfinite(kappas))
            and np.all(np.isfinite(weights))
        ):
            raise RuntimeError("EM: parâmetros finais não finitos.")

        # Drop de componentes degenerados (peso < min) e renormalização final
        keep = weights >= self.min_component_weight
        if not np.all(keep):
            mus = mus[keep]
            kappas = kappas[keep]
            weights = weights[keep]
            wsum = float(np.sum(weights))
            if wsum <= 0 or not np.isfinite(wsum):
                raise RuntimeError("EM: todos os componentes morreram.")
            weights = weights / wsum
            k = int(weights.shape[0])

        params = _MoVMFParams(
            mus=mus.astype(np.float64),
            kappas=kappas.astype(np.float64),
            weights=weights.astype(np.float64),
            dim=int(dim),
            sample_count=int(n_samples),
            n_components=int(k),
        )
        return params, n_iter_done, converged, avg_ll

    def _em_m_step(
        self,
        x: np.ndarray,
        responsibilities: np.ndarray,
        mus: np.ndarray,
        kappas: np.ndarray,
        weights: np.ndarray,
        rng: np.random.Generator,
        *,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        M-step com handling de componentes mortos via split do componente
        mais massivo (preserva entropia da mistura).

        PR 5: quando `sample_weights` é fornecido, EM ponderado:
        `n_k = Σ_i w_i · r_ik` (em vez de `Σ_i r_ik`).
        """
        n_samples, dim = x.shape
        k = mus.shape[0]

        if sample_weights is not None:
            # Responsibilities ponderadas: r_ponderado[i,k] = w_i * r_ik
            r_w = responsibilities * sample_weights[:, None]      # (N, K)
            nk = r_w.sum(axis=0)                                  # (K,)
            total_w = float(sample_weights.sum())
            denom_w = max(total_w, self.eps)
        else:
            r_w = responsibilities
            nk = responsibilities.sum(axis=0)
            denom_w = float(n_samples)

        # Componentes vivos vs mortos
        alive = nk > self.eps

        # Pesos preliminares: π_k = n_k / Σ w_i (ou n_samples).
        new_weights = np.where(alive, nk / denom_w, 0.0)

        # Mus e r_bars vetorizados — por componente
        new_mus = np.zeros((k, dim), dtype=np.float64)
        r_bars = np.zeros(k, dtype=np.float64)

        # r_w.T @ x  -> (K, D)
        resultants = r_w.T @ x  # (K, D)
        resultant_norms = np.linalg.norm(resultants, axis=1)  # (K,)

        for j in range(k):
            if not alive[j] or resultant_norms[j] < self.eps:
                # Componente morto: marcar para split posterior
                new_mus[j] = mus[j]  # placeholder; será reescrito no split
                r_bars[j] = 0.0
            else:
                new_mus[j] = resultants[j] / resultant_norms[j]
                r_bars[j] = resultant_norms[j] / max(nk[j], self.eps)

        # Estimação de kappa vetorizada (recebe array de r_bars)
        new_kappas = self._estimate_kappa_vectorized(r_bars, dim)

        # Componentes vivos com kappa válido
        alive_final = alive & np.isfinite(new_kappas) & (new_kappas >= self.min_kappa)

        # Split: para cada componente morto, clona o vivo de maior peso
        # (com perturbação angular pequena) para preservar K.
        if not np.all(alive_final):
            n_dead = int(np.sum(~alive_final))
            if np.any(alive_final):
                # Ordena vivos por nk descendente para selecionar doadores
                alive_indices = np.where(alive_final)[0]
                donor_order = alive_indices[np.argsort(-nk[alive_indices])]

                for d_idx, dead_j in enumerate(np.where(~alive_final)[0]):
                    donor = donor_order[d_idx % len(donor_order)]
                    # Perturbação angular: amostra direção tangente aleatória,
                    # normaliza e mistura com mu do doador.
                    tangent = rng.normal(size=dim)
                    tangent -= np.dot(tangent, new_mus[donor]) * new_mus[donor]
                    tnorm = np.linalg.norm(tangent)
                    if tnorm > self.eps:
                        tangent = tangent / tnorm
                    perturb_angle = 0.05  # ~3 graus
                    new_mu = (
                        np.cos(perturb_angle) * new_mus[donor]
                        + np.sin(perturb_angle) * tangent
                    )
                    nm = np.linalg.norm(new_mu)
                    if nm > self.eps:
                        new_mu = new_mu / nm
                    new_mus[dead_j] = new_mu
                    new_kappas[dead_j] = new_kappas[donor]
                    # Doa metade do peso ao filho recém-nascido
                    transfer = new_weights[donor] * 0.5
                    new_weights[donor] -= transfer
                    new_weights[dead_j] = transfer
            else:
                # Todos os componentes morreram: reseed completo
                idx = rng.integers(0, n_samples, size=k)
                new_mus = x[idx].copy()
                new_kappas = np.full(k, self.min_kappa, dtype=np.float64)
                new_weights = np.full(k, 1.0 / k, dtype=np.float64)
                _ = n_dead  # silenciar lint

        # Clip e renormaliza pesos
        new_weights = np.maximum(new_weights, self.min_component_weight)
        wsum = float(np.sum(new_weights))
        if not np.isfinite(wsum) or wsum <= 0:
            new_weights = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            new_weights = new_weights / wsum

        # Clip kappas e renormaliza mus
        new_kappas = np.clip(new_kappas, self.min_kappa, self.max_kappa)
        new_mus = self._renormalize_rows(new_mus)

        return new_mus, new_kappas, new_weights

    def _initialize_mixture_params(
        self, x: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """K-Means++ esférico."""
        n_samples, dim = x.shape
        k = self.n_components

        mus = np.zeros((k, dim), dtype=np.float64)
        mus[0] = x[int(rng.integers(0, n_samples))]

        for j in range(1, k):
            sims = x @ mus[:j].T  # (N, j)
            max_sim = np.max(sims, axis=1)
            dist = np.maximum(1.0 - max_sim, 0.0)
            dsum = float(np.sum(dist))
            if dsum > 0:
                mus[j] = x[rng.choice(n_samples, p=dist / dsum)]
            else:
                mus[j] = x[int(rng.integers(0, n_samples))]

        mus = self._renormalize_rows(mus)
        labels = np.argmax(x @ mus.T, axis=1)

        weights = np.zeros(k, dtype=np.float64)
        r_bars = np.zeros(k, dtype=np.float64)

        for j in range(k):
            idx = np.where(labels == j)[0]
            if len(idx) == 0:
                mus[j] = x[int(rng.integers(0, n_samples))]
                weights[j] = 1.0 / k
                r_bars[j] = 0.0
                continue
            cluster_x = x[idx]
            weights[j] = len(idx) / n_samples
            resultant = cluster_x.sum(axis=0)
            rnorm = float(np.linalg.norm(resultant))
            if rnorm < self.eps:
                mus[j] = x[int(rng.integers(0, n_samples))]
                r_bars[j] = 0.0
            else:
                mus[j] = resultant / rnorm
                r_bars[j] = rnorm / len(idx)

        kappas = self._estimate_kappa_vectorized(r_bars, dim)
        kappas = np.clip(kappas, self.min_kappa, self.max_kappa)

        weights = np.maximum(weights, self.min_component_weight)
        wsum = float(np.sum(weights))
        if not np.isfinite(wsum) or wsum <= 0:
            weights = np.full(k, 1.0 / k, dtype=np.float64)
        else:
            weights = weights / wsum

        return self._renormalize_rows(mus), kappas, weights

    # =========================================================================
    # CENTRALIZAÇÃO E TRANSFORMAÇÃO
    # =========================================================================

    def _compute_reference_point(self, x_raw: np.ndarray) -> None:
        """
        Decide e calcula o ponto de referência conforme self.centering.

        - 'none': sem centralização. _reference_point fica None.
        - 'frechet': média intrínseca esférica (média de Fréchet) iterativa.
            Preserva a métrica esférica via log/exp maps.
        - 'euclidean_legacy': média euclidiana tradicional (legado, não preserva métrica).
        """
        if self.centering == "none":
            self._reference_point = None
        elif self.centering == "euclidean_legacy":
            self._reference_point = np.mean(x_raw, axis=0).astype(np.float64)
        elif self.centering == "frechet":
            self._reference_point = self._frechet_mean_sphere(x_raw)
        else:
            raise ValueError(f"centering desconhecido: {self.centering}")

    def _frechet_mean_sphere(
        self, x: np.ndarray, max_iter: int = 50, tol: float = 1e-8
    ) -> np.ndarray:
        """
        Média de Fréchet (Karcher mean) na hiperesfera unitária via iteração de
        gradiente: começa pela média euclidiana renormalizada e itera a média
        no espaço tangente, aplicando exp_map de volta à esfera.
        """
        # Ponto inicial: média euclidiana renormalizada
        mu = np.mean(x, axis=0)
        mu_norm = np.linalg.norm(mu)
        if mu_norm < self.eps:
            # Distribuição quase isotrópica — qualquer ponto serve
            mu = x[0].copy()
        else:
            mu = mu / mu_norm

        for _ in range(max_iter):
            # log_map: v_i = (x_i - cos(d_i) * mu) * d_i / sin(d_i), onde d_i = arccos(mu·x_i)
            cos_d = np.clip(x @ mu, -1.0, 1.0)
            # Estabilidade numérica:
            d = np.arccos(cos_d)
            sin_d = np.sin(d)

            # Onde sin_d ≈ 0 (pontos coincidentes com mu), v_i = 0
            tangent = np.zeros_like(x)
            mask = sin_d > self.eps
            if np.any(mask):
                # (x - cos_d * mu) projeta x no plano tangente em mu;
                # multiplicado por d/sin_d dá o vetor tangente de norma d (a distância geodésica).
                tangent[mask] = (
                    (x[mask] - cos_d[mask, None] * mu[None, :])
                    * (d[mask] / sin_d[mask])[:, None]
                )

            mean_tangent = np.mean(tangent, axis=0)
            step_norm = float(np.linalg.norm(mean_tangent))
            if step_norm < tol:
                break

            # exp_map: mu_new = cos(s)*mu + sin(s)*(v/||v||), s = ||mean_tangent||
            v_dir = mean_tangent / step_norm
            mu_new = np.cos(step_norm) * mu + np.sin(step_norm) * v_dir
            mu_new_norm = np.linalg.norm(mu_new)
            if mu_new_norm < self.eps or not np.isfinite(mu_new_norm):
                break
            mu = mu_new / mu_new_norm

        return mu.astype(np.float64)

    def _apply_transform(self, x_raw: np.ndarray) -> np.ndarray:
        """
        Aplica a centralização escolhida.
        Usado tanto em treino quanto em inferência (mesma transformação).
        """
        if self.centering == "none" or self._reference_point is None:
            return x_raw  # já está L2-normalizado pelo _validate_and_normalize_2d

        if self.centering == "euclidean_legacy":
            return self._renormalize_rows(
                x_raw - self._reference_point.reshape(1, -1)
            )

        if self.centering == "frechet":
            # Reorientação geodésica: rotaciona x para que o reference_point vá
            # para o "polo norte" canônico e_1, preservando ângulos. Usa rotação
            # de Householder (reflexão composta) por simplicidade.
            return self._rotate_to_pole(x_raw, self._reference_point)

        raise ValueError(f"centering desconhecido na transformação: {self.centering}")

    @staticmethod
    def _rotate_to_pole(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """
        Rotação ortogonal R tal que R @ mu = e_1. Aplicada a cada linha de x.
        Preserva a métrica esférica.

        Usa reflexão de Householder: R = I - 2 v v^T / (v^T v), com v = mu - e_1
        (e ajuste de sinal). Reflexão é ortogonal e cumpre R @ mu = e_1 (a menos
        de sinal global, irrelevante para a vMF que é invariante a reflexões).
        """
        d = mu.shape[0]
        e1 = np.zeros(d, dtype=np.float64)
        e1[0] = 1.0
        v = mu - e1
        v_norm_sq = float(np.dot(v, v))
        if v_norm_sq < 1e-20:
            # mu ≈ e_1 já: identidade
            return x.copy()
        # x_rot = x - 2 (x @ v) v / v_norm_sq
        coef = (x @ v) * (2.0 / v_norm_sq)
        x_rot = x - coef[:, None] * v[None, :]
        return x_rot

    # =========================================================================
    # MATEMÁTICA DA vMF (vetorizada)
    # =========================================================================

    def _component_logpdf_matrix(self, x: np.ndarray) -> np.ndarray:
        params = self._params
        assert params is not None
        return self._component_logpdf_from(x, params.mus, params.kappas, params.weights)

    def _component_logpdf_from(
        self,
        x: np.ndarray,
        mus: np.ndarray,
        kappas: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna matriz (N, K) com log(w_j) + log C_d(kappa_j) + kappa_j * (x_i^T mu_j).
        """
        d = x.shape[1]
        dot = np.clip(x @ mus.T, -1.0, 1.0)  # (N, K)

        log_cs = self._log_vmf_normalizing_constant_vec(d, kappas)
        if not np.all(np.isfinite(log_cs)):
            raise RuntimeError("Constantes log C_d(kappa) não finitas.")

        safe_weights = np.maximum(weights, self.eps)
        log_weights = np.log(safe_weights)
        if not np.all(np.isfinite(log_weights)):
            raise RuntimeError("log(weights) não finitos.")

        return log_weights[None, :] + log_cs[None, :] + dot * kappas[None, :]

    def _log_vmf_normalizing_constant_vec(
        self, dim: int, kappas: np.ndarray
    ) -> np.ndarray:
        """
        log C_d(kappa) vetorizado.

        log C_d(kappa) = (d/2 - 1) log(kappa) - (d/2) log(2π) - log I_{d/2-1}(kappa)
                      = (d/2 - 1) log(kappa) - (d/2) log(2π) - kappa - log(ive_{d/2-1}(kappa))

        onde ive(nu, kappa) = I_nu(kappa) * exp(-kappa) é numericamente estável em alta dimensão.

        Para kappa < eps_safe, usa o limite uniforme:
            log C_d(0) = log Γ(d/2) - log 2 - (d/2) log π.
        """
        from math import lgamma, log, pi

        kappas = np.asarray(kappas, dtype=np.float64)
        log_uniform = lgamma(dim / 2.0) - log(2.0) - (dim / 2.0) * log(pi)
        out = np.full(kappas.shape, log_uniform, dtype=np.float64)

        nu = dim / 2.0 - 1.0
        eps_safe = 1e-8
        active = kappas > eps_safe
        if not np.any(active):
            return out

        ka = kappas[active]
        ive_vals = ive(nu, ka)  # vetor

        # Onde ive falha (não finito ou <= 0), mantém o uniforme.
        good = np.isfinite(ive_vals) & (ive_vals > 0)
        if np.any(good):
            ka_good = ka[good]
            log_bessel_part = ka_good + np.log(ive_vals[good])
            vals = (
                nu * np.log(ka_good)
                - (dim / 2.0) * log(2.0 * pi)
                - log_bessel_part
            )
            # Substitui não finitos pelo uniforme
            vals = np.where(np.isfinite(vals), vals, log_uniform)
            # Encaixa de volta na posição certa
            target_indices = np.where(active)[0][good]
            out[target_indices] = vals
        return out

    def _estimate_kappa_vectorized(
        self, r_bars: np.ndarray, dim: int
    ) -> np.ndarray:
        """Versão vetorizada de _estimate_kappa_robust para arrays de r_bar."""
        r_bars = np.asarray(r_bars, dtype=np.float64)
        out = np.empty_like(r_bars)
        for i, r in enumerate(r_bars):
            out[i] = self._estimate_kappa_robust(float(r), dim)
        return out

    def _estimate_kappa_robust(self, r_bar: float, dim: int) -> float:
        """
        Estimação robusta de kappa a partir de r_bar = ||resultant|| / N.

        Estratégia (em alta dimensão, d=512, Newton ingênuo é instável):
          1. Aproximação de Banerjee como ponto inicial.
          2. Newton com damping (line search por backtracking).
          3. Fallback para bisseção monotônica caso Newton oscile.
          4. Para r_bar >= 1 - eps, retorna max_kappa diretamente
             (cap superior; r_bar=1 é a degenerescência delta).
        """
        # Sanitização do r_bar
        r_bar = float(np.clip(r_bar, 0.0, 1.0 - 1e-12))
        if not np.isfinite(r_bar) or r_bar < 1e-8:
            return self.min_kappa
        if r_bar > 1.0 - 1e-6:
            return self.max_kappa

        # 1) Banerjee initial guess
        denom = max(1e-12, 1.0 - r_bar * r_bar)
        kappa = (r_bar * dim - r_bar ** 3) / denom
        if not np.isfinite(kappa) or kappa <= 0:
            kappa = self.min_kappa
        kappa = float(np.clip(kappa, self.min_kappa, self.max_kappa))

        # 2) Newton com damping
        kappa = self._newton_kappa_damped(kappa, r_bar, dim)

        # 3) Verifica resíduo; se ruim, cai para bisseção
        a = self._A_d(kappa, dim)
        if not np.isfinite(a) or abs(a - r_bar) > 1e-4:
            kappa = self._bisect_kappa(r_bar, dim, kappa_hint=kappa)

        # Final clip
        if not np.isfinite(kappa):
            return self.min_kappa
        return float(np.clip(kappa, self.min_kappa, self.max_kappa))

    def _newton_kappa_damped(
        self, kappa0: float, r_bar: float, dim: int, max_iter: int = 50
    ) -> float:
        """Newton com backtracking line search. Mantém monotonicidade do resíduo |a - r_bar|."""
        kappa = kappa0
        for _ in range(max_iter):
            a = self._A_d(kappa, dim)
            if not np.isfinite(a):
                return kappa
            diff = a - r_bar
            if abs(diff) < 1e-9:
                return kappa
            # Derivada: d/dk A_d(k) = 1 - A_d^2 - ((d-1)/k) * A_d
            deriv = 1.0 - a * a - ((dim - 1) / max(kappa, 1e-12)) * a
            if not np.isfinite(deriv) or abs(deriv) < 1e-15:
                return kappa
            full_step = diff / deriv

            # Backtracking: encontra alpha em (0, 1] que reduz |residual|
            alpha = 1.0
            best_kappa = kappa
            best_resid = abs(diff)
            improved = False
            for _bt in range(20):
                trial = kappa - alpha * full_step
                if trial <= 0 or not np.isfinite(trial):
                    alpha *= 0.5
                    continue
                trial = float(np.clip(trial, self.min_kappa, self.max_kappa))
                a_trial = self._A_d(trial, dim)
                if not np.isfinite(a_trial):
                    alpha *= 0.5
                    continue
                resid_trial = abs(a_trial - r_bar)
                if resid_trial < best_resid:
                    best_resid = resid_trial
                    best_kappa = trial
                    improved = True
                    break
                alpha *= 0.5

            if not improved:
                return kappa
            if abs(best_kappa - kappa) < 1e-9:
                return best_kappa
            kappa = best_kappa
        return kappa

    def _bisect_kappa(
        self, r_bar: float, dim: int, kappa_hint: float, max_iter: int = 80
    ) -> float:
        """
        Bisseção monotônica em kappa.
        A_d(kappa) é monótona crescente em kappa, então a busca por A_d(k*) = r_bar é bem-posta.
        Usa o hint para encontrar bracket inicial.
        """
        # Determinar bracket [lo, hi] tal que A_d(lo) < r_bar < A_d(hi)
        lo = max(self.min_kappa, kappa_hint / 1024.0)
        hi = min(self.max_kappa, kappa_hint * 1024.0)

        a_lo = self._A_d(lo, dim)
        a_hi = self._A_d(hi, dim)

        # Expande hi se ainda abaixo de r_bar
        expand_count = 0
        while np.isfinite(a_hi) and a_hi < r_bar and hi < self.max_kappa and expand_count < 30:
            hi = min(self.max_kappa, hi * 4.0)
            a_hi = self._A_d(hi, dim)
            expand_count += 1

        # Encolhe lo se a_lo > r_bar (raro mas possível)
        shrink_count = 0
        while np.isfinite(a_lo) and a_lo > r_bar and lo > self.min_kappa and shrink_count < 30:
            lo = max(self.min_kappa, lo / 4.0)
            a_lo = self._A_d(lo, dim)
            shrink_count += 1

        if not (np.isfinite(a_lo) and np.isfinite(a_hi)):
            return float(np.clip(kappa_hint, self.min_kappa, self.max_kappa))
        if a_hi < r_bar:
            return self.max_kappa
        if a_lo > r_bar:
            return self.min_kappa

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            a_mid = self._A_d(mid, dim)
            if not np.isfinite(a_mid):
                return mid
            if abs(a_mid - r_bar) < 1e-9 or (hi - lo) < 1e-9 * max(1.0, mid):
                return mid
            if a_mid < r_bar:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _A_d(self, kappa: float, dim: int) -> float:
        """A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) — razão de Bessels via ive."""
        if kappa <= 0 or not np.isfinite(kappa):
            return 0.0
        nu = dim / 2.0 - 1.0
        num = ive(nu + 1.0, kappa)
        den = ive(nu, kappa)
        if not np.isfinite(num) or not np.isfinite(den) or abs(den) < self.eps:
            return 0.0
        val = num / den
        return float(val) if np.isfinite(val) else 0.0

    # =========================================================================
    # VALIDAÇÃO E NORMALIZAÇÃO
    # =========================================================================

    def _prepare_inference_matrix(
        self, embeddings: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_fitted()
        x_raw = self._validate_and_normalize_2d(embeddings)
        x_transformed = self._apply_transform(x_raw)
        return x_raw, x_transformed

    def _ensure_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("MoVMFTypicalityModel não foi treinado. Chame fit() antes.")

    def _validate_and_normalize_2d(self, embeddings: ArrayLike) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("Formato esperado: (N, D) ou (D,).")
        # PR 4: política UnitNormPolicy explícita (ERROR | NORMALIZE_*)
        return apply_unit_norm_policy(
            x, policy=self.unit_norm_policy,
            tolerance=self.unit_norm_tolerance, eps=self.eps,
        )

    def _validate_and_normalize_1d(self, embedding: ArrayLike) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float64)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("Esperado um único embedding (1, D) ou (D,).")
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Formato 1D esperado.")
        # PR 4: aplica a política de norma sob o ponto único do batch
        return apply_unit_norm_policy(
            x.reshape(1, -1), policy=self.unit_norm_policy,
            tolerance=self.unit_norm_tolerance, eps=self.eps,
        )[0]

    def _aggregate_by_identity(
        self, embeddings: np.ndarray, identities: Sequence[str],
    ) -> np.ndarray:
        """Centróide L2-renormalizado por identidade.

        Reusa a fórmula do `KNNTypicalityModel._aggregate_by_identity`.
        Persiste no estado interno o número de identidades singleton
        para auditoria em `summary_protocol`.
        """
        ids_arr = np.asarray(identities)
        unique_ids, inverse_idx = np.unique(ids_arr, return_inverse=True)
        n_ids = unique_ids.shape[0]
        d = embeddings.shape[1]

        sums = np.zeros((n_ids, d), dtype=np.float64)
        counts = np.zeros(n_ids, dtype=np.int64)
        np.add.at(sums, inverse_idx, embeddings)
        np.add.at(counts, inverse_idx, 1)

        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        centroids = sums / np.maximum(norms, self.eps)

        if not np.all(np.isfinite(centroids)):
            raise RuntimeError(
                "Centroides por identidade contêm valores não finitos."
            )

        self._n_input_faces = int(embeddings.shape[0])
        self._n_singleton_identities = int(np.sum(counts == 1))
        return centroids.astype(np.float64)

    def _renormalize_rows(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        out = x / np.maximum(norms, self.eps)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("Renormalização gerou valores não finitos.")
        return out