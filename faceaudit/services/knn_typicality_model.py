# faceaudit/services/knn_typicality_model.py

"""
Modelo de tipicidade não-paramétrico baseado em k-NN esférico.

Filosofia:
  Em vez de assumir uma família distribucional (vMF, Power Spherical),
  esse modelo carrega a população de referência inteira e computa,
  para cada query, três métricas complementares de tipicidade:

    1. mean_knn_similarity: T_k(x) = (1/k) * sum_{i=1..k} cos(x, y_(i))
       Cosseno médio aos k vizinhos mais próximos. Cresce com a densidade
       local. Intuitivo e cheap de computar.

    2. typicality_percentile: rank empírico de T_k(x) na distribuição
       leave-one-out de T_k(y_j) sobre a referência. 0..100.
       Defensível em corte como estatística não-paramétrica de ranking
       ("X é mais típico que 73% da população de referência").

    3. log_density_knn: estimador de Loftsgaarden-Quesenberry na esfera.
       log p̂(x) ≈ log k - log N - (d-1) * log(theta_k)
       Pode entrar como p(x | H_d) num LR generativo.

Modos de referência (reference_mode):
  - PER_IDENTITY (default): a referência tem 1 ponto por identidade
    (centroide L2-renormalizado dos embeddings dela). Tipicidade mede
    "esta pessoa é típica na população", insensível a desbalanceamento
    de fotos por pessoa. Identidades singleton mantidas — centroide
    é a própria face única.
  - PER_FACE: a referência tem 1 ponto por face. Mede "esta foto é
    típica no conjunto de fotos", vulnerável a hubness intra-identidade.

Vantagens vs MoVMF:
  - Sem suposição de simetria rotacional (que hubness viola em FR).
  - Sem inversão de Bessel/Newton.
  - Hubness vira sinal: faces em região densa têm T_k alto.

Custo:
  - O "modelo" carrega a referência inteira na memória/disco.
  - Para PER_IDENTITY com 5 fotos/id médio, a referência fica 5× menor
    que PER_FACE — modelos compactos sem perda de poder forense.
  - Inferência: O(N·D) por query. Para uso forense exato e auditável,
    é o preço justo. Backends aproximados (FAISS) ficam para uma V2 se
    N >> 100k tornar isso problemático.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.knn_typicality_info import KNNTypicalityInfo
from faceaudit.models.typicality.typicality_model_summary import (
    TypicalityModelSummary,
)
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)


ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]


def _freeze_array(
    arr: Optional[np.ndarray],
    *,
    dtype: Optional[np.dtype] = None,
) -> Optional[np.ndarray]:
    """P1-R3-7 (revisão externa 2026-06-09 round 3): congela arrays
    internos do state como read-only.

    Faz cópia (defensiva — caller não pode escrever no mesmo buffer
    depois), garante C-contiguous, opcionalmente normaliza dtype e zera
    a flag de write. Acesso posterior tenta-escrever levanta
    `ValueError: assignment destination is read-only`.

    Aceita None (passa adiante) para simplificar uso em campos
    opcionais (pesos, CDF).
    """
    if arr is None:
        return None
    out = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None \
        else np.ascontiguousarray(arr)
    if out is arr:
        # ascontiguousarray pode devolver o próprio array sem copiar
        # quando já é C-contiguous + dtype compatível. Forçamos cópia
        # para garantir snapshot defensivo independente do caller.
        out = out.copy()
    out.setflags(write=False)
    return out


@dataclass
class _KNNRefState:
    """Estado interno congelado pós-fit. Imutável conceitualmente."""
    reference: np.ndarray              # (N_ref, D), L2-normalizado, float32
    loo_mean_knn_sorted: np.ndarray    # (N_ref,), float64, ordem crescente
    ood_threshold: float
    k: int
    dim: int
    reference_mode: TypicalityReferenceMode
    # Diagnóstico — quantas faces de origem foram agrupadas no PER_IDENTITY,
    # ou shape original em PER_FACE.
    n_input_faces: int
    n_singleton_identities: int  # só relevante em PER_IDENTITY
    # PR 5: pesos por face de referência (INVERSE_IDENTITY_FREQUENCY).
    # Quando None, modo legado (FACE_LEVEL_RAW/IDENTITY_CENTROIDS).
    # Quando definido, paralelo a `reference` e a `loo_mean_knn_sorted_weights`.
    reference_score_weights: Optional[np.ndarray] = None
    # Pesos correspondentes à `loo_mean_knn_sorted` (mesma ordenação).
    loo_mean_knn_sorted_weights: Optional[np.ndarray] = None
    # CDF empírica ponderada (cumsum normalizada de
    # `loo_mean_knn_sorted_weights`). Quando None, usa cumsum
    # uniforme = (1..N)/N.
    loo_cdf: Optional[np.ndarray] = None
    # ID estável da fórmula (`INV_ID_FREQ_v1`), entra no contract
    # fingerprint para auditoria.
    weighted_T_k_formula_id: Optional[str] = None
    # P0-1 (revisão externa 2026-06-09): proveniência da referência de
    # scores. Quando `install_reference_score_calibration` é chamado
    # (caso típico: pós-CROSS_FIT_IDENTITY do
    # `ReferenceScoreCalibrationService`), esses campos auditam quem
    # instalou a referência atual. Em fit() puro permanecem None.
    reference_score_calibration_mode: Optional[Any] = None  # ReferenceScoreCalibrationMode
    split_fingerprint: Optional[str] = None


class KNNTypicalityModel:
    """
    Tipicidade não-paramétrica via k-NN esférico.

    Uso:
        model = KNNTypicalityModel(k=50, reference_mode=TypicalityReferenceMode.PER_IDENTITY)
        model.fit(reference_embeddings, identities=reference_identities)
        info = model.get_typicality_info(query_embedding)
        print(info.typicality_percentile, info.log_density_knn)
    """

    def __init__(
        self,
        k: int = 50,
        ood_quantile: float = 0.01,
        reference_mode: TypicalityReferenceMode = TypicalityReferenceMode.PER_IDENTITY,
        model_name: str = "KNNTypicalityModel",
        eps: float = 1e-12,
        # PR 4
        unit_norm_policy=None,
        unit_norm_tolerance: float = 1e-4,
        # PR 5: ponderação populacional. Quando None, deriva de
        # `reference_mode` (PER_FACE → FACE_LEVEL_RAW; PER_IDENTITY →
        # IDENTITY_CENTROIDS).
        population_weighting_mode=None,
    ):
        if k < 1:
            raise ValueError("k deve ser >= 1.")
        if not (0.0 <= ood_quantile < 0.5):
            raise ValueError("ood_quantile deve estar em [0, 0.5).")

        self.k = int(k)
        self.ood_quantile = float(ood_quantile)
        self.reference_mode = TypicalityReferenceMode.from_str(reference_mode) \
            if not isinstance(reference_mode, TypicalityReferenceMode) \
            else reference_mode
        self.model_name = model_name
        self.eps = float(eps)
        # PR 4: política UnitNormPolicy.
        # P2-R2-15 (revisão externa round 2): construtor direto default
        # `NORMALIZE_SILENTLY` por back-compat com o caminho legado;
        # diverge do default do DTO (`KNNTypicalityModelConfig` usa
        # `ERROR`, defensável forensicamente). O caminho oficial de
        # treinamento via Strategy Engine passa pelos DTOs e portanto
        # adota ERROR; instâncias diretas (scripts, notebooks de
        # desenvolvimento) ficam em SILENTLY salvo override explícito.
        # A política efetivamente aplicada é persistida no estado e
        # entra no `model_contract_fingerprint`, então auditoria pode
        # detectar se foi NORMALIZE_SILENTLY no laudo.
        from faceaudit.enums.unit_norm_policy import UnitNormPolicy
        if unit_norm_policy is None:
            unit_norm_policy = UnitNormPolicy.NORMALIZE_SILENTLY
        self.unit_norm_policy = (
            UnitNormPolicy.from_str(unit_norm_policy)
            if not isinstance(unit_norm_policy, UnitNormPolicy)
            else unit_norm_policy
        )
        self.unit_norm_tolerance = float(unit_norm_tolerance)

        # PR 5: PopulationWeightingMode. Quando None, deriva do
        # reference_mode para preservar compat.
        from faceaudit.enums.population_weighting_mode import (
            PopulationWeightingMode,
        )
        if population_weighting_mode is None:
            if self.reference_mode is TypicalityReferenceMode.PER_IDENTITY:
                population_weighting_mode = PopulationWeightingMode.IDENTITY_CENTROIDS
            else:
                population_weighting_mode = PopulationWeightingMode.FACE_LEVEL_RAW
        self.population_weighting_mode = (
            PopulationWeightingMode.from_str(population_weighting_mode)
            if not isinstance(population_weighting_mode, PopulationWeightingMode)
            else population_weighting_mode
        )

        # Estado pós-fit. None até `fit()` ser chamado.
        self._state: Optional[_KNNRefState] = None
        # P2-R2-13 + P0-R3-6: telemetria de normalização separada por
        # escopo (fit vs inference). Auditoria pericial usa
        # `_fit_unit_norm_telemetry`; o snapshot da última inferência
        # fica em `_inference_unit_norm_telemetry_last`.
        self._fit_unit_norm_telemetry: Optional[dict] = None
        self._inference_unit_norm_telemetry_last: Optional[dict] = None
        # R4 (revisão externa 2026-06-09 round 4): métricas preditivas
        # OOF persistidas pelo `install_reference_score_calibration`.
        # No k-NN com basis=MEAN_KNN_SIMILARITY, esses números NÃO são
        # log-likelihood — são T_k médio na população OOF.
        self._oof_predictive_avg_score: Optional[float] = None
        self._oof_predictive_avg_score_by_fold: Optional[tuple] = None
        self._oof_predictive_score_type: Optional[str] = None
        # P0-R5-6 (revisão externa round 5): diagnósticos por fold
        # persistidos. Antes só viviam no `ReferenceScoreCalibrationResult`.
        self._n_identities_by_fold: Optional[tuple] = None
        self._n_scores_by_fold: Optional[tuple] = None
        self._ood_threshold_by_fold: Optional[tuple] = None
        self._fold_seeds: Optional[tuple] = None
        self._n_folds_used: Optional[int] = None
        # P0-R5-3 (revisão externa round 5): fingerprint auditável das
        # métricas OOF. Recalculado e validado pelo `from_dict`.
        self._audit_metrics_fingerprint: Optional[str] = None

    # =========================================================================
    # API PÚBLICA
    # =========================================================================

    def fit(
        self,
        embeddings: ArrayLike,
        *,
        identities: Optional[Sequence[str]] = None,
        identity_ids: Optional[Sequence[str]] = None,
    ) -> "KNNTypicalityModel":
        """
        Armazena a referência (faces ou centroides por identidade) e
        pré-computa a CDF empírica leave-one-out de T_k pra suporte a
        percentile lookup.

        Args:
            embeddings: (N, D). Será L2-normalizado se ainda não estiver.
            identities: sequência paralela a embeddings com o rótulo de
                identidade de cada face. Obrigatório em PER_IDENTITY,
                ignorado em PER_FACE.
            identity_ids: alias canônico de `identities` definido pelo
                `FittableTypicalityModelProtocol` (PR 1b). Quando
                fornecido sem `identities`, é usado como tal. Erro
                quando ambos forem fornecidos com valores divergentes.
        """
        if identity_ids is not None and identities is not None and identity_ids != identities:
            raise ValueError(
                "KNNTypicalityModel.fit recebeu `identities` e "
                "`identity_ids` divergentes — use apenas um dos dois."
            )
        if identities is None and identity_ids is not None:
            identities = identity_ids
        x_raw = self._validate_and_normalize_2d(embeddings)
        n_input, d = x_raw.shape

        # PR 5: dispatch via PopulationWeightingMode.
        from faceaudit.enums.population_weighting_mode import (
            PopulationWeightingMode,
        )

        weights_per_face: Optional[np.ndarray] = None

        if self.population_weighting_mode == PopulationWeightingMode.IDENTITY_CENTROIDS:
            if identities is None:
                raise ValueError(
                    "PopulationWeightingMode.IDENTITY_CENTROIDS exige "
                    "`identities`/`identity_ids` paralelo a `embeddings`."
                )
            if len(identities) != n_input:
                raise ValueError(
                    f"len(identities)={len(identities)} difere de "
                    f"len(embeddings)={n_input}."
                )
            reference, n_singletons = self._aggregate_by_identity(x_raw, identities)
        elif self.population_weighting_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
            if identities is None:
                raise ValueError(
                    "PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY exige "
                    "`identities`/`identity_ids` paralelo a `embeddings`."
                )
            if len(identities) != n_input:
                raise ValueError(
                    f"len(identities)={len(identities)} difere de "
                    f"len(embeddings)={n_input}."
                )
            reference = x_raw.copy()
            n_singletons = 0
            # w_i = 1/n_id_i (faces de identidades raras pesam mais).
            ids_arr = np.asarray(identities)
            unique_ids, inverse_idx, id_counts = np.unique(
                ids_arr, return_inverse=True, return_counts=True,
            )
            weights_per_face = (1.0 / id_counts[inverse_idx]).astype(np.float64)
            self._n_input_faces_inv = int(n_input)
            self._n_identities_inv = int(unique_ids.shape[0])
        else:
            # FACE_LEVEL_RAW
            reference = x_raw.copy()
            n_singletons = 0

        n_ref = reference.shape[0]
        if n_ref < self.k + 1:
            raise ValueError(
                f"k+1 ({self.k + 1}) pontos de referência são necessários para LOO. "
                f"Obtidos: {n_ref} (modo={self.population_weighting_mode.to_str()})."
            )

        # Similaridade N_ref × N_ref. Exclui self via diagonal = -inf.
        sims = reference @ reference.T
        np.fill_diagonal(sims, -np.inf)

        top_k_idx = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
        top_k_sim = np.take_along_axis(sims, top_k_idx, axis=1)

        if weights_per_face is not None:
            # T_k ponderado: w_j_renormalizado = w_j / Σ w_k_vizinhos
            top_k_w = weights_per_face[top_k_idx]                 # (N, k)
            row_w_sum = top_k_w.sum(axis=1, keepdims=True)        # (N, 1)
            renormalized = top_k_w / np.maximum(row_w_sum, self.eps)
            loo_T_k = (top_k_sim * renormalized).sum(axis=1).astype(np.float64)
        else:
            loo_T_k = top_k_sim.mean(axis=1).astype(np.float64)

        if not np.all(np.isfinite(loo_T_k)):
            raise RuntimeError(
                "Cálculo LOO produziu valores não finitos. Verifique se a "
                "referência tem pontos degenerados (norma zero)."
            )

        if weights_per_face is not None:
            # CDF ponderada: ordena T_k, alinha pesos, e cumsum normalizado.
            order = np.argsort(loo_T_k)
            loo_sorted = loo_T_k[order]
            sorted_weights = weights_per_face[order]
            total_w = sorted_weights.sum()
            cdf = np.cumsum(sorted_weights) / total_w
            # Threshold OOD via quantil ponderado (busca onde cdf >= q)
            if self.ood_quantile > 0.0:
                idx = int(np.searchsorted(cdf, self.ood_quantile, side="left"))
                idx = min(max(idx, 0), len(loo_sorted) - 1)
                ood_threshold = float(loo_sorted[idx])
            else:
                ood_threshold = float(-np.inf)
            reference_score_weights = weights_per_face
            loo_sorted_weights = sorted_weights
            formula_id = "INV_ID_FREQ_v1"
        else:
            loo_sorted = np.sort(loo_T_k)
            ood_threshold = (
                float(np.quantile(loo_sorted, self.ood_quantile))
                if self.ood_quantile > 0.0
                else float(-np.inf)
            )
            cdf = None
            reference_score_weights = None
            loo_sorted_weights = None
            formula_id = None

        self._state = _KNNRefState(
            reference=_freeze_array(reference, dtype=np.float32),
            loo_mean_knn_sorted=_freeze_array(loo_sorted, dtype=np.float64),
            ood_threshold=ood_threshold,
            k=self.k,
            dim=int(d),
            reference_mode=self.reference_mode,
            n_input_faces=int(n_input),
            n_singleton_identities=int(n_singletons),
            reference_score_weights=_freeze_array(
                reference_score_weights, dtype=np.float64,
            ),
            loo_mean_knn_sorted_weights=_freeze_array(
                loo_sorted_weights, dtype=np.float64,
            ),
            loo_cdf=_freeze_array(cdf, dtype=np.float64),
            weighted_T_k_formula_id=formula_id,
        )
        return self

    def is_fitted(self) -> bool:
        return self._state is not None

    def cross_fit_training_contract_snapshot(self) -> dict:
        """P0-R6-1 (revisão externa round 6): captura os hiperparâmetros
        de treino que devem ser idênticos entre peek/fold/refit do
        cross-fit. Em particular, `k` precisa ser estável — antes uma
        factory adversarial podia alternar `k=2` e `k=5` entre folds
        e o serviço aceitava.
        """
        return {
            "model_type": self.model_type.to_str(),
            "k": int(self.k),
            "ood_quantile": float(self.ood_quantile),
            "reference_mode": self.reference_mode.to_str(),
            "population_weighting_mode":
                self.population_weighting_mode.to_str(),
            "unit_norm_policy": self.unit_norm_policy.to_str(),
            "unit_norm_tolerance": float(self.unit_norm_tolerance),
            "eps": float(self.eps),
            "reference_score_basis_type": (
                self.reference_score_basis_type.to_str()
                if hasattr(self.reference_score_basis_type, "to_str")
                else str(self.reference_score_basis_type)
            ),
        }

    def install_reference_score_calibration(
        self,
        *,
        reference_scores,
        reference_weights=None,
        ood_threshold=None,
        calibration_mode,
        split_fingerprint=None,
        # R4 (revisão externa 2026-06-09 round 4): métricas preditivas
        # agregadas. Persistidas para exibir no `summary()`. No k-NN
        # (basis=MEAN_KNN_SIMILARITY) NÃO é log-likelihood — é a média
        # de T_k (similaridade média aos k vizinhos mais próximos) na
        # população OOF. Útil como métrica relativa entre k's.
        oof_predictive_avg_score=None,
        oof_predictive_avg_score_by_fold=None,
        oof_predictive_score_type=None,
        # P0-R5-3/6 (revisão externa round 5): diagnósticos por fold
        # persistidos no modelo + fingerprint auditável recalculado em
        # from_dict (defesa contra adulteração de pickle).
        n_identities_by_fold=None,
        n_scores_by_fold=None,
        ood_threshold_by_fold=None,
        fold_seeds=None,
        n_folds_used=None,
    ) -> None:
        """Sobrescreve atomicamente a referência interna de scores T_k.

        P0-1 (revisão externa 2026-06-09): gancho usado pelo
        `ReferenceScoreCalibrationService` após `CROSS_FIT_IDENTITY` para
        instalar a distribuição OOF de T_k + threshold OOD no modelo
        persistido. Antes, a CDF/threshold do `_state` ficavam IN-SAMPLE
        do refit final e os valores OOF retornados ao serviço nunca
        chegavam ao modelo.

        Análogo do `install_reference_score_calibration` do MoVMF. Reusa
        os mesmos parâmetros + semântica. Cria novo `_KNNRefState` (a
        instância é frozen-by-convention).

        Parâmetros:
            reference_scores: vetor 1D de T_k (já agregado por face, na
                escala bruta `[-1, +1]`).
            reference_weights: pesos opcionais por score. Quando None,
                rank uniforme.
            ood_threshold: limiar OOD calculado externamente.
            calibration_mode: `ReferenceScoreCalibrationMode` que produziu
                esta referência (entra no contract fingerprint).
            split_fingerprint: hash do split identidade→fold (auditoria).
        """
        self._ensure_fitted()
        assert self._state is not None
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
            ood_val: float = float(ood_threshold)
            if not np.isfinite(ood_val):
                raise ValueError(
                    "install_reference_score_calibration: ood_threshold "
                    "não finito."
                )
        else:
            ood_val = float("-inf")  # permissivo — consistente com KNN

        # Ordena scores + pesos paralelamente; computa CDF.
        order = np.argsort(scores_arr)
        scores_sorted = np.ascontiguousarray(scores_arr[order], dtype=np.float64)
        if weights_arr is not None:
            weights_sorted = np.ascontiguousarray(
                weights_arr[order], dtype=np.float64
            )
            cdf_arr = np.cumsum(weights_sorted) / total_w
        else:
            weights_sorted = None
            cdf_arr = None

        # P0-R5-2: valida finitude + comprimentos antes de instalar.
        from faceaudit.services.oof_audit_metrics import (
            _compute_audit_metrics_fingerprint,
            _normalize_n_folds_used,
            _normalize_oof_avg_score,
            _normalize_oof_per_fold_floats,
            _normalize_oof_per_fold_floats_optional,
            _normalize_oof_per_fold_ints,
            _validate_oof_per_fold_lengths,
        )
        norm_oof_avg = _normalize_oof_avg_score(
            oof_predictive_avg_score, field="oof_predictive_avg_score",
        )
        norm_oof_by_fold = _normalize_oof_per_fold_floats(
            oof_predictive_avg_score_by_fold,
            field="oof_predictive_avg_score_by_fold",
        )
        norm_n_ids_by_fold = _normalize_oof_per_fold_ints(
            n_identities_by_fold, field="n_identities_by_fold",
        )
        norm_n_scores_by_fold = _normalize_oof_per_fold_ints(
            n_scores_by_fold, field="n_scores_by_fold",
        )
        norm_ood_thr_by_fold = _normalize_oof_per_fold_floats_optional(
            ood_threshold_by_fold, field="ood_threshold_by_fold",
        )
        norm_fold_seeds = _normalize_oof_per_fold_ints(
            fold_seeds, field="fold_seeds",
        )
        # P1-R6-6: usa o helper estrito (rejeita float fracionário).
        norm_n_folds_used = _normalize_n_folds_used(n_folds_used)
        _validate_oof_per_fold_lengths(
            n_folds_used=norm_n_folds_used,
            oof_by_fold=norm_oof_by_fold,
            n_identities_by_fold=norm_n_ids_by_fold,
            n_scores_by_fold=norm_n_scores_by_fold,
            ood_threshold_by_fold=norm_ood_thr_by_fold,
            fold_seeds=norm_fold_seeds,
        )

        self._state = dataclass_replace(
            self._state,
            loo_mean_knn_sorted=_freeze_array(scores_sorted, dtype=np.float64),
            ood_threshold=ood_val,
            loo_mean_knn_sorted_weights=_freeze_array(
                weights_sorted, dtype=np.float64,
            ),
            loo_cdf=_freeze_array(cdf_arr, dtype=np.float64),
            reference_score_calibration_mode=calibration_mode,
            split_fingerprint=str(split_fingerprint) if split_fingerprint else None,
        )
        # R4: persiste métricas preditivas como atributos da instância
        # (não no `_KNNRefState` para evitar quebrar pickles antigos —
        # `_state` é restaurado por `from_dict` validando shapes).
        self._oof_predictive_avg_score = norm_oof_avg
        self._oof_predictive_avg_score_by_fold = norm_oof_by_fold
        self._oof_predictive_score_type = (
            str(oof_predictive_score_type)
            if oof_predictive_score_type is not None else None
        )
        # R5: diagnósticos por fold + fingerprint auditável.
        self._n_identities_by_fold = norm_n_ids_by_fold
        self._n_scores_by_fold = norm_n_scores_by_fold
        self._ood_threshold_by_fold = norm_ood_thr_by_fold
        self._fold_seeds = norm_fold_seeds
        self._n_folds_used = norm_n_folds_used
        self._audit_metrics_fingerprint = _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=self._oof_predictive_avg_score,
            oof_predictive_avg_score_by_fold=self._oof_predictive_avg_score_by_fold,
            oof_predictive_score_type=self._oof_predictive_score_type,
            n_identities_by_fold=self._n_identities_by_fold,
            n_scores_by_fold=self._n_scores_by_fold,
            ood_threshold_by_fold=self._ood_threshold_by_fold,
            split_fingerprint=self._state.split_fingerprint,
            fold_seeds=self._fold_seeds,
            n_folds_used=self._n_folds_used,
        )

    def score_samples(self, embeddings: ArrayLike) -> np.ndarray:
        """Retorna log_density_knn pra cada embedding (batch)."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.log_density_knn for i in infos], dtype=np.float64)

    def get_percentiles(self, embeddings: ArrayLike) -> np.ndarray:
        """Retorna o percentile (0..100) pra cada embedding (batch)."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.typicality_percentile for i in infos], dtype=np.float64)

    def is_in_distribution(self, embeddings: ArrayLike) -> np.ndarray:
        """Boolean por embedding: T_k >= ood_threshold."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.is_in_distribution for i in infos], dtype=bool)

    def get_typicality_info(
        self, embedding: ArrayLike, *, include_top_k: bool = True,
    ) -> KNNTypicalityInfo:
        """Avalia um único embedding e empacota em KNNTypicalityInfo."""
        self._ensure_fitted()
        x = self._validate_and_normalize_1d(embedding)
        return self._compute_info_single(x, include_top_k=include_top_k)

    def get_typicality_infos(
        self, embeddings: ArrayLike, *, include_top_k: bool = False,
    ) -> List[KNNTypicalityInfo]:
        """
        Batch. Default include_top_k=False pra economizar memória — em
        scoring de pares (~milhões), guardar 50 floats por face explode.
        """
        self._ensure_fitted()
        x = self._validate_and_normalize_2d(embeddings)
        return [
            self._compute_info_single(x[i], include_top_k=include_top_k)
            for i in range(x.shape[0])
        ]

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    def to_dict(self) -> dict:
        """Exporta o modelo treinado para um dict JSON-serializável.

        PR 5: campos `reference_score_weights`/`loo_cdf`/
        `weighted_T_k_formula_id` persistem quando
        `INVERSE_IDENTITY_FREQUENCY` está ativo.
        """
        self._ensure_fitted()
        st = self._state
        assert st is not None
        return {
            "format_version": 4,
            "hyperparameters": {
                "k": self.k,
                "ood_quantile": self.ood_quantile,
                "reference_mode": self.reference_mode.to_str(),
                "model_name": self.model_name,
                "eps": self.eps,
                # PR 4 + PR 5
                "unit_norm_policy": self.unit_norm_policy.to_str(),
                "unit_norm_tolerance": self.unit_norm_tolerance,
                "population_weighting_mode":
                    self.population_weighting_mode.to_str(),
            },
            "learned_state": {
                "reference": st.reference.tolist(),
                "loo_mean_knn_sorted": st.loo_mean_knn_sorted.tolist(),
                "loo_mean_knn_sorted_weights": (
                    st.loo_mean_knn_sorted_weights.tolist()
                    if st.loo_mean_knn_sorted_weights is not None else None
                ),
                "loo_cdf": (
                    st.loo_cdf.tolist() if st.loo_cdf is not None else None
                ),
                "reference_score_weights": (
                    st.reference_score_weights.tolist()
                    if st.reference_score_weights is not None else None
                ),
                "weighted_T_k_formula_id": st.weighted_T_k_formula_id,
                "ood_threshold": st.ood_threshold,
                "k": st.k,
                "dim": st.dim,
                "reference_mode": st.reference_mode.to_str(),
                "n_input_faces": st.n_input_faces,
                "n_singleton_identities": st.n_singleton_identities,
                # P0-1/P0-3 (format_version=4): proveniência da referência
                # de scores. Identifica se o modelo carrega CDF OOF
                # (CROSS_FIT_IDENTITY) ou IN_SAMPLE_LEGACY ou pura `fit()`.
                "reference_score_calibration_mode": (
                    st.reference_score_calibration_mode.to_str()
                    if st.reference_score_calibration_mode is not None
                    else None
                ),
                "split_fingerprint": st.split_fingerprint,
                # R4: métricas preditivas OOF persistidas no envelope.
                "oof_predictive_avg_score": self._oof_predictive_avg_score,
                "oof_predictive_avg_score_by_fold": (
                    list(self._oof_predictive_avg_score_by_fold)
                    if self._oof_predictive_avg_score_by_fold is not None
                    else None
                ),
                "oof_predictive_score_type":
                    self._oof_predictive_score_type,
                # R5: diagnósticos por fold + fingerprint auditável.
                "n_identities_by_fold": (
                    list(self._n_identities_by_fold)
                    if self._n_identities_by_fold is not None else None
                ),
                "n_scores_by_fold": (
                    list(self._n_scores_by_fold)
                    if self._n_scores_by_fold is not None else None
                ),
                "ood_threshold_by_fold": (
                    list(self._ood_threshold_by_fold)
                    if self._ood_threshold_by_fold is not None else None
                ),
                "fold_seeds": (
                    list(self._fold_seeds)
                    if self._fold_seeds is not None else None
                ),
                "n_folds_used": self._n_folds_used,
                "audit_metrics_fingerprint": self._audit_metrics_fingerprint,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KNNTypicalityModel":
        version = data.get("format_version", 1)
        if version not in (1, 2, 3, 4):
            raise ValueError(f"Versão de formato não suportada: {version}")

        hp = dict(data["hyperparameters"])

        # Migração v1 → v2: campos removidos (population_label) + novo
        # campo reference_mode default PER_FACE (era o comportamento
        # implícito antes desse toggle existir).
        if version == 1:
            hp.pop("population_label", None)
            hp.setdefault("reference_mode", TypicalityReferenceMode.PER_FACE.to_str())

        # Migração v2 → v3 (PR 4/5): defaults V2.14
        if version <= 2:
            hp.setdefault("unit_norm_policy", "NORMALIZE_SILENTLY")
            hp.setdefault("unit_norm_tolerance", 1e-4)
            # PR 5 ainda não existia: derive de reference_mode
            ref_mode_str = hp.get(
                "reference_mode", TypicalityReferenceMode.PER_IDENTITY.to_str()
            )
            if ref_mode_str.upper() == "PER_FACE":
                hp.setdefault("population_weighting_mode", "FACE_LEVEL_RAW")
            else:
                hp.setdefault("population_weighting_mode", "IDENTITY_CENTROIDS")

        model = cls(
            k=int(hp["k"]),
            ood_quantile=float(hp["ood_quantile"]),
            reference_mode=TypicalityReferenceMode.from_str(
                hp.get("reference_mode", TypicalityReferenceMode.PER_IDENTITY.to_str())
            ),
            model_name=hp.get("model_name", "KNNTypicalityModel"),
            eps=float(hp.get("eps", 1e-12)),
            unit_norm_policy=hp.get("unit_norm_policy"),
            unit_norm_tolerance=float(hp.get("unit_norm_tolerance", 1e-4)),
            population_weighting_mode=hp.get("population_weighting_mode"),
        )

        ls = data["learned_state"]
        reference = np.asarray(ls["reference"], dtype=np.float32)
        loo = np.asarray(ls["loo_mean_knn_sorted"], dtype=np.float64)

        if reference.ndim != 2:
            raise ValueError("reference deve ser 2D.")
        if reference.shape[0] != loo.shape[0]:
            raise ValueError(
                f"Inconsistência: reference tem {reference.shape[0]} "
                f"pontos, loo_mean_knn_sorted tem {loo.shape[0]}."
            )
        norms = np.linalg.norm(reference, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                f"reference não está L2-normalizada (norms min={norms.min()}, "
                f"max={norms.max()})."
            )

        # v1 não carregava reference_mode no state — inferimos PER_FACE.
        state_mode = TypicalityReferenceMode.from_str(
            ls.get("reference_mode", TypicalityReferenceMode.PER_FACE.to_str())
        )

        # PR 5: estado opcional dos pesos.
        weights_raw = ls.get("reference_score_weights")
        ref_weights = (
            np.asarray(weights_raw, dtype=np.float64)
            if weights_raw is not None else None
        )
        sorted_weights_raw = ls.get("loo_mean_knn_sorted_weights")
        sorted_weights = (
            np.asarray(sorted_weights_raw, dtype=np.float64)
            if sorted_weights_raw is not None else None
        )
        cdf_raw = ls.get("loo_cdf")
        cdf = (
            np.asarray(cdf_raw, dtype=np.float64)
            if cdf_raw is not None else None
        )

        # P0-1/P0-3 (format_version=4): proveniência da referência de
        # scores. Pickles v1/v2/v3 não têm os campos — `ls.get(...)`
        # devolve None, preservando comportamento legado.
        ref_mode_str = ls.get("reference_score_calibration_mode")
        if ref_mode_str:
            from faceaudit.enums.reference_score_calibration_mode import (
                ReferenceScoreCalibrationMode,
            )
            ref_cal_mode = ReferenceScoreCalibrationMode.from_str(ref_mode_str)
        else:
            ref_cal_mode = None

        model._state = _KNNRefState(
            reference=_freeze_array(reference, dtype=np.float32),
            loo_mean_knn_sorted=_freeze_array(loo, dtype=np.float64),
            ood_threshold=float(ls["ood_threshold"]),
            k=int(ls["k"]),
            dim=int(ls["dim"]),
            reference_mode=state_mode,
            n_input_faces=int(ls.get("n_input_faces", reference.shape[0])),
            n_singleton_identities=int(ls.get("n_singleton_identities", 0)),
            reference_score_weights=_freeze_array(ref_weights, dtype=np.float64),
            loo_mean_knn_sorted_weights=_freeze_array(
                sorted_weights, dtype=np.float64,
            ),
            loo_cdf=_freeze_array(cdf, dtype=np.float64),
            weighted_T_k_formula_id=ls.get("weighted_T_k_formula_id"),
            reference_score_calibration_mode=ref_cal_mode,
            split_fingerprint=ls.get("split_fingerprint"),
        )

        # P1-R2-7 (revisão externa round 2): valida integridade
        # estrutural da distribuição de referência restaurada antes
        # de devolver o modelo. Sem isso, pickles corrompidos
        # (scores não-ordenados, pesos negativos, CDF incoerente)
        # passavam silenciosamente e davam erro só no momento do
        # get_percentiles.
        from faceaudit.services.reference_score_distribution_validator import (
            validate_reference_score_distribution,
        )
        # Centroides L2-normalizados: validação extra.
        ref_norms = np.linalg.norm(reference, axis=1)
        if not np.allclose(ref_norms, 1.0, atol=1e-3):
            raise ValueError(
                f"KNN.from_dict: reference matrix não está L2-normalizada "
                f"(norms min={ref_norms.min()}, max={ref_norms.max()})."
            )
        validate_reference_score_distribution(
            scores=loo,
            weights=sorted_weights,
            cdf=cdf,
            ood_threshold=float(ls["ood_threshold"]),
        )

        # R4: hidrata métricas preditivas OOF. Pickles antigos não
        # têm esses campos — devolvem None.
        # P0-R7-1: passa VALORES BRUTOS aos normalizadores (sem `int(x)`
        # defensivo). Coerções silenciosas truncavam `1.9 → 1` e
        # neutralizavam a validação estrita de R6.
        raw_oof_avg = ls.get("oof_predictive_avg_score")
        raw_oof_by_fold = ls.get("oof_predictive_avg_score_by_fold")
        raw_oof_by_fold = (
            tuple(raw_oof_by_fold) if raw_oof_by_fold is not None else None
        )
        raw_n_ids = ls.get("n_identities_by_fold")
        raw_n_ids = tuple(raw_n_ids) if raw_n_ids is not None else None
        raw_n_scores = ls.get("n_scores_by_fold")
        raw_n_scores = tuple(raw_n_scores) if raw_n_scores is not None else None
        raw_ood_thr = ls.get("ood_threshold_by_fold")
        raw_ood_thr = tuple(raw_ood_thr) if raw_ood_thr is not None else None
        raw_fs = ls.get("fold_seeds")
        raw_fs = tuple(raw_fs) if raw_fs is not None else None
        raw_n_folds = ls.get("n_folds_used")
        model._audit_metrics_fingerprint = ls.get("audit_metrics_fingerprint")
        # P0-R6-3 + P0-R7-1: re-aplica validações estruturais ANTES de
        # armazenar nos atributos.
        from faceaudit.services.oof_audit_metrics import (
            _normalize_n_folds_used,
            _normalize_oof_per_fold_floats_optional,
            _normalize_oof_per_fold_ints,
            _revalidate_oof_payload_or_raise,
            _validate_cross_fit_audit_completeness_or_raise,
            _verify_audit_metrics_fingerprint_or_raise,
        )
        _revalidate_oof_payload_or_raise(
            oof_predictive_avg_score=raw_oof_avg,
            oof_predictive_avg_score_by_fold=raw_oof_by_fold,
            n_identities_by_fold=raw_n_ids,
            n_scores_by_fold=raw_n_scores,
            ood_threshold_by_fold=raw_ood_thr,
            fold_seeds=raw_fs,
            n_folds_used=raw_n_folds,
        )
        # Validações OK — agora armazena com coerção segura.
        model._oof_predictive_avg_score = (
            float(raw_oof_avg) if raw_oof_avg is not None else None
        )
        model._oof_predictive_avg_score_by_fold = (
            tuple(float(x) for x in raw_oof_by_fold)
            if raw_oof_by_fold is not None else None
        )
        model._oof_predictive_score_type = ls.get("oof_predictive_score_type")
        model._n_identities_by_fold = _normalize_oof_per_fold_ints(
            raw_n_ids, field="n_identities_by_fold",
        )
        model._n_scores_by_fold = _normalize_oof_per_fold_ints(
            raw_n_scores, field="n_scores_by_fold",
        )
        model._ood_threshold_by_fold = _normalize_oof_per_fold_floats_optional(
            raw_ood_thr, field="ood_threshold_by_fold",
        )
        model._fold_seeds = _normalize_oof_per_fold_ints(
            raw_fs, field="fold_seeds",
        )
        model._n_folds_used = _normalize_n_folds_used(raw_n_folds)
        # P0-R7-2/3: em CROSS_FIT_IDENTITY, exige bloco completo +
        # score_type esperado por backend.
        _validate_cross_fit_audit_completeness_or_raise(
            backend="KNN",
            reference_score_calibration_mode=
                model._state.reference_score_calibration_mode,
            persisted_fingerprint=model._audit_metrics_fingerprint,
            oof_predictive_avg_score=model._oof_predictive_avg_score,
            oof_predictive_avg_score_by_fold=model._oof_predictive_avg_score_by_fold,
            oof_predictive_score_type=model._oof_predictive_score_type,
            n_identities_by_fold=model._n_identities_by_fold,
            n_scores_by_fold=model._n_scores_by_fold,
            ood_threshold_by_fold=model._ood_threshold_by_fold,
            split_fingerprint=model._state.split_fingerprint,
            fold_seeds=model._fold_seeds,
            n_folds_used=model._n_folds_used,
        )
        # P0-R5-3: valida fingerprint auditável das métricas OOF.
        _verify_audit_metrics_fingerprint_or_raise(
            persisted_fingerprint=model._audit_metrics_fingerprint,
            oof_predictive_avg_score=model._oof_predictive_avg_score,
            oof_predictive_avg_score_by_fold=model._oof_predictive_avg_score_by_fold,
            oof_predictive_score_type=model._oof_predictive_score_type,
            n_identities_by_fold=model._n_identities_by_fold,
            n_scores_by_fold=model._n_scores_by_fold,
            ood_threshold_by_fold=model._ood_threshold_by_fold,
            split_fingerprint=model._state.split_fingerprint,
            fold_seeds=model._fold_seeds,
            n_folds_used=model._n_folds_used,
        )
        return model

    # =========================================================================
    # DIAGNÓSTICO / OBSERVABILIDADE
    # =========================================================================

    def summary_legacy(self) -> dict:
        """Resumo dict ad-hoc — preservado por back-compat com a UI
        (`ResourceInfoWidget._populate_typicality_details`).

        P0-5 (revisão externa 2026-06-09): a versão canônica do método
        `summary()` agora retorna `TypicalityModelSummary` para cumprir o
        protocolo — vide doc da nova `summary()`.
        """
        return self._summary_dict()

    def _summary_dict(self) -> dict:
        """Estatísticas de auditoria do modelo treinado."""
        if not self.is_fitted():
            return {"status": "unfitted"}
        st = self._state
        assert st is not None
        loo = st.loo_mean_knn_sorted
        return {
            "status": "fitted",
            "model_name": self.model_name,
            "reference_mode": st.reference_mode.to_str(),
            "k": st.k,
            "sample_count": st.reference.shape[0],
            "n_input_faces": st.n_input_faces,
            "n_singleton_identities": st.n_singleton_identities,
            "embedding_dim": st.dim,
            "ood_quantile": self.ood_quantile,
            "ood_threshold": st.ood_threshold,
            "loo_T_k_min": float(loo[0]),
            "loo_T_k_max": float(loo[-1]),
            "loo_T_k_median": float(np.median(loo)),
            "loo_T_k_mean": float(np.mean(loo)),
            "loo_T_k_std": float(np.std(loo)),
        }

    # =========================================================================
    # PROPRIEDADES READ-ONLY (útil pra UI / inspeção)
    # =========================================================================

    @property
    def sample_count(self) -> Optional[int]:
        """Tamanho da referência (faces em PER_FACE, identidades em PER_IDENTITY)."""
        return int(self._state.reference.shape[0]) if self._state is not None else None

    @property
    def n_input_faces(self) -> Optional[int]:
        """Quantas faces foram passadas pro fit (antes de qualquer agregação)."""
        return self._state.n_input_faces if self._state is not None else None

    @property
    def n_singleton_identities(self) -> Optional[int]:
        """Em PER_IDENTITY: quantas identidades têm uma única face."""
        return self._state.n_singleton_identities if self._state is not None else None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._state.dim if self._state is not None else None

    @property
    def ood_threshold(self) -> Optional[float]:
        return self._state.ood_threshold if self._state is not None else None

    @property
    def loo_distribution(self) -> Optional[np.ndarray]:
        """Cópia da CDF empírica de T_k — útil pra plot."""
        if self._state is None:
            return None
        return self._state.loo_mean_knn_sorted.copy()

    # =========================================================================
    # TypicalityScoringModelProtocol (PR 1b)
    # =========================================================================

    @property
    def model_type(self) -> TypicalityModelType:
        return TypicalityModelType.KNN

    @property
    def reference_score_basis_type(self) -> TypicalityScoreType:
        """Base canônica do KNN: similaridade média aos k vizinhos (T_k).

        P0-R2-2 (revisão externa 2026-06-09 round 2): consultada pelo
        `ReferenceScoreCalibrationService` para pedir scores OOF na
        escala que `_state.loo_mean_knn_sorted` e `get_percentiles`
        consomem. Antes, o serviço escolhia `LOG_DENSITY_ESTIMATE`
        sem distinguir backend — instalava scores em log-densidade
        num estado cujo `_compute_info_single` compara contra T_k,
        produzindo percentis em 100% para todas as consultas.
        """
        return TypicalityScoreType.MEAN_KNN_SIMILARITY

    @property
    def enabled_score_types(self) -> frozenset[TypicalityScoreType]:
        """Quais score types esta instância está pronta para produzir.

        k-NN treinado disponibiliza T_k bruto e percentil empírico
        sobre T_k. Log-densidade L-Q esférica é exposta APENAS em
        modos não-ponderados (FACE_LEVEL_RAW, IDENTITY_CENTROIDS).

        P1-R2-11 (revisão externa 2026-06-09 round 2): em
        INVERSE_IDENTITY_FREQUENCY, T_k é ponderado (pesos
        renormalizados sobre os k vizinhos), mas a estimativa
        Loftsgaarden-Quesenberry segue `log k − log N − (d−1) log θ_k`
        sem incorporar pesos. Versão ponderada formal de L-Q em S^d
        não está definida no projeto; expor a versão não-ponderada
        em modo ponderado produziria valores incoerentes com o T_k
        que alimenta a CDF. Solução: ocultar `LOG_DENSITY_ESTIMATE`
        de `enabled_score_types` em INV_ID_FREQ até definição
        formal.
        """
        if not self.is_fitted():
            return frozenset()
        from faceaudit.enums.population_weighting_mode import (
            PopulationWeightingMode,
        )
        types = {
            TypicalityScoreType.MEAN_KNN_SIMILARITY,
            TypicalityScoreType.TYPICALITY_PERCENTILE,
        }
        if self.population_weighting_mode != PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
            types.add(TypicalityScoreType.LOG_DENSITY_ESTIMATE)
        return frozenset(types)

    @property
    def embedding_dim(self) -> int:
        self._ensure_fitted()
        return int(self._state.dim)  # type: ignore[union-attr]

    @property
    def sample_count(self) -> int:
        self._ensure_fitted()
        return int(self._state.reference.shape[0])  # type: ignore[union-attr]

    @property
    def model_state_fingerprint(self) -> str:
        """SHA-256 do estado numérico (reference matrix + CDF LOO + threshold).

        Recalculado sob demanda; não é cacheado (consistente com
        movmf/PLDA fingerprints existentes em outros pontos do projeto).
        """
        self._ensure_fitted()
        import hashlib
        st = self._state
        assert st is not None
        h = hashlib.sha256()
        # Referência canonicaliza para float32 C-contígua (formato persistido)
        ref = np.ascontiguousarray(st.reference, dtype=np.float32)
        h.update(ref.tobytes(order="C"))
        loo = np.ascontiguousarray(st.loo_mean_knn_sorted, dtype=np.float64)
        h.update(loo.tobytes(order="C"))
        # P0-3 (revisão externa 2026-06-09): pesos e CDF da referência
        # ponderada entram no state fingerprint. Sem isso, dois modelos
        # com mesma distribuição de T_k mas pesos diferentes
        # (INV_ID_FREQ vs uniforme; OOF vs IN_SAMPLE) tinham o mesmo
        # state fingerprint.
        if st.loo_mean_knn_sorted_weights is not None:
            h.update(np.ascontiguousarray(
                st.loo_mean_knn_sorted_weights, dtype=np.float64
            ).tobytes(order="C"))
        if st.loo_cdf is not None:
            h.update(np.ascontiguousarray(
                st.loo_cdf, dtype=np.float64
            ).tobytes(order="C"))
        # P0-R2-5 (round 2): `reference_score_weights` afeta o T_k
        # inferido (pesos renormalizados sobre os k vizinhos). Antes
        # estava fora do state fingerprint — alterando os pesos sem
        # tocar mais nada produzia mesmo hash mas T_k diferente.
        if st.reference_score_weights is not None:
            h.update(np.ascontiguousarray(
                st.reference_score_weights, dtype=np.float64
            ).tobytes(order="C"))
        h.update(f"{st.ood_threshold:.17g}".encode("utf-8"))
        h.update(f"k={st.k};dim={st.dim};rmode={st.reference_mode.to_str()}".encode("utf-8"))
        # Proveniência da referência (CROSS_FIT_IDENTITY ↔
        # IN_SAMPLE_LEGACY ↔ None pré-install).
        if st.reference_score_calibration_mode is not None:
            h.update(
                f"calmode={st.reference_score_calibration_mode.to_str()}"
                .encode("utf-8")
            )
        if st.split_fingerprint:
            h.update(f"split={st.split_fingerprint}".encode("utf-8"))
        return h.hexdigest()

    @property
    def model_contract_fingerprint(self) -> str:
        """SHA-256 do contrato semântico (sem o estado bruto).

        Inclui o tipo do modelo, score_semantics_ids habilitados,
        reference_mode, k, ood_quantile. Excluímos `eps` e `model_name`
        porque são cosmeticamente relevantes mas não mudam a matemática.
        """
        self._ensure_fitted()
        import hashlib
        st = self._state
        assert st is not None
        semantics_ids = sorted([
            self._score_semantics_id_for(t) for t in self.enabled_score_types
        ])
        contract_parts = [
            f"model_type={TypicalityModelType.KNN.to_str()}",
            f"semantics_ids={','.join(semantics_ids)}",
            f"reference_mode={st.reference_mode.to_str()}",
            f"k={st.k}",
            f"dim={st.dim}",
            f"ood_quantile={self.ood_quantile:.17g}",
            # PR 4
            f"unit_norm_policy={self.unit_norm_policy.to_str()}",
            # P1-11 (revisão externa 2026-06-09): tolerance entra no
            # contrato — dois modelos com tolerâncias diferentes aceitam
            # universos diferentes de embeddings como esféricos.
            f"unit_norm_tolerance={self.unit_norm_tolerance:.17g}",
            # PR 5
            f"population_weighting_mode={self.population_weighting_mode.to_str()}",
            f"weighted_T_k_formula_id={st.weighted_T_k_formula_id or 'none'}",
            # P0-3 (revisão externa 2026-06-09): protocolo de calibração
            # da referência (CROSS_FIT_IDENTITY ↔ IN_SAMPLE_LEGACY ↔
            # None) entra no contrato. Aplicar modelo OOF a dataset que
            # espera IN_SAMPLE_LEGACY muda interpretação dos percentis.
            (
                "reference_score_calibration_mode="
                + (
                    st.reference_score_calibration_mode.to_str()
                    if st.reference_score_calibration_mode is not None
                    else "none"
                )
            ),
        ]
        h = hashlib.sha256()
        h.update("\n".join(contract_parts).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _score_semantics_id_for(t: TypicalityScoreType) -> str:
        if t == TypicalityScoreType.MEAN_KNN_SIMILARITY:
            return "MEAN_KNN_SIMILARITY_SPHERICAL"
        if t == TypicalityScoreType.TYPICALITY_PERCENTILE:
            return "PERCENTILE_OF_MEAN_KNN_SIMILARITY"
        if t == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            return "LOG_DENSITY_ESTIMATE_LQ_SPHERICAL"
        raise ValueError(f"score_type sem semantics id: {t}")

    def compute_scores(
        self,
        embeddings: ArrayLike,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch:
        """Aplica o k-NN aos embeddings e retorna o batch tipado.

        Levanta `UnsupportedTypicalityScoreTypeError` se o score_type não
        estiver em `enabled_score_types`.
        """
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

        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        if score_type == TypicalityScoreType.MEAN_KNN_SIMILARITY:
            arr = np.array([i.mean_knn_similarity for i in infos], dtype=np.float64)
            percentile_basis = None
            ref_dist_fp = None
        elif score_type == TypicalityScoreType.TYPICALITY_PERCENTILE:
            arr = np.array([i.typicality_percentile for i in infos], dtype=np.float64)
            # Base histórica do k-NN: o percentil é sobre MEAN_KNN_SIMILARITY
            # (LOO T_k da referência).
            percentile_basis = TypicalityScoreType.MEAN_KNN_SIMILARITY
            # Fingerprint da CDF de referência (loo_mean_knn_sorted).
            ref_dist_fp = self._reference_score_distribution_fingerprint()
        elif score_type == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            arr = np.array([i.log_density_knn for i in infos], dtype=np.float64)
            percentile_basis = None
            ref_dist_fp = None
        else:
            raise ValueError(f"score_type desconhecido: {score_type}")

        return TypicalityScoreBatch(
            scores=arr,
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
        """Fingerprint da distribuição de referência.

        P0-R2-4 (revisão externa 2026-06-09 round 2): inclui scores
        ordenados + pesos + CDF + threshold OOD + proveniência da
        calibração + base canônica. Antes hashava apenas
        `loo_mean_knn_sorted` — dois modelos com pesos distintos
        compartilhavam fingerprint apesar de produzirem percentis
        diferentes.
        """
        import hashlib
        self._ensure_fitted()
        st = self._state
        assert st is not None
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(
            st.loo_mean_knn_sorted, dtype=np.float64
        ).tobytes(order="C"))
        if st.loo_mean_knn_sorted_weights is not None:
            h.update(np.ascontiguousarray(
                st.loo_mean_knn_sorted_weights, dtype=np.float64
            ).tobytes(order="C"))
        if st.loo_cdf is not None:
            h.update(np.ascontiguousarray(
                st.loo_cdf, dtype=np.float64
            ).tobytes(order="C"))
        h.update(f"ood={st.ood_threshold:.17g}".encode("utf-8"))
        if st.reference_score_calibration_mode is not None:
            h.update(
                f"calmode={st.reference_score_calibration_mode.to_str()}"
                .encode("utf-8")
            )
        if st.split_fingerprint:
            h.update(f"split={st.split_fingerprint}".encode("utf-8"))
        h.update(
            f"basis={self.reference_score_basis_type.to_str()}".encode("utf-8")
        )
        return h.hexdigest()

    def summary(self) -> TypicalityModelSummary:
        """Implementação do `summary()` do protocolo.

        Nome distinto do método legado `summary()` (que retorna dict) para
        não quebrar consumidores existentes. PR 2 unifica para `summary()`.
        """
        self._ensure_fitted()
        st = self._state
        assert st is not None
        loo = st.loo_mean_knn_sorted

        warnings_list: list[str] = []
        health = "OK"
        if st.n_singleton_identities > 0 and st.reference_mode is TypicalityReferenceMode.PER_IDENTITY:
            warnings_list.append(
                f"{st.n_singleton_identities} identidades singleton (centroide == face única)"
            )

        return TypicalityModelSummary(
            model_type=TypicalityModelType.KNN,
            model_name=self.model_name,
            embedding_dim=int(st.dim),
            sample_count=int(st.reference.shape[0]),
            enabled_score_types=tuple(sorted(self.enabled_score_types, key=lambda t: t.value)),
            health_status=health,
            warnings=tuple(warnings_list),
            backend_details={
                "k": int(st.k),
                "reference_mode": st.reference_mode.to_str(),
                "ood_quantile": float(self.ood_quantile),
                "ood_threshold": float(st.ood_threshold),
                "n_input_faces": int(st.n_input_faces),
                "n_singleton_identities": int(st.n_singleton_identities),
                "loo_T_k_min": float(loo[0]),
                "loo_T_k_max": float(loo[-1]),
                "loo_T_k_median": float(np.median(loo)),
                "loo_T_k_mean": float(np.mean(loo)),
                "loo_T_k_std": float(np.std(loo)),
                # R4: métricas preditivas OOF. NÃO são log-likelihood
                # no k-NN (basis=MEAN_KNN_SIMILARITY) — são T_k
                # (similaridade média aos k vizinhos mais próximos)
                # médio ponderado entre os scores OOF. Úteis como
                # métrica relativa entre valores de k.
                "oof_predictive_avg_score": self._oof_predictive_avg_score,
                "oof_predictive_avg_score_by_fold":
                    self._oof_predictive_avg_score_by_fold,
                "oof_predictive_score_type": self._oof_predictive_score_type,
                # R5: diagnósticos completos por fold + fingerprint.
                "n_identities_by_fold": self._n_identities_by_fold,
                "n_scores_by_fold": self._n_scores_by_fold,
                "ood_threshold_by_fold": self._ood_threshold_by_fold,
                "fold_seeds": self._fold_seeds,
                "n_folds_used": self._n_folds_used,
                "audit_metrics_fingerprint": self._audit_metrics_fingerprint,
            },
        )

    # =========================================================================
    # AGREGAÇÃO POR IDENTIDADE
    # =========================================================================

    def _aggregate_by_identity(
        self, embeddings: np.ndarray, identities: Sequence[str],
    ) -> Tuple[np.ndarray, int]:
        """
        Soma os embeddings de cada identidade e L2-renormaliza para obter
        o centroide esférico (direção média na hiperesfera unitária).

        Identidades singleton viram centroide = a face única (renormalizada,
        mantém norma 1). Não há descarte — toda identidade pesa igual no
        espaço de referência.

        Retorna:
            (matriz_centroides, n_singletons_count)
        """
        ids_arr = np.asarray(identities)
        unique_ids, inverse_idx = np.unique(ids_arr, return_inverse=True)

        n_ids = unique_ids.shape[0]
        d = embeddings.shape[1]
        sums = np.zeros((n_ids, d), dtype=np.float64)
        counts = np.zeros(n_ids, dtype=np.int64)

        np.add.at(sums, inverse_idx, embeddings)
        np.add.at(counts, inverse_idx, 1)

        # L2-renormaliza cada centroide. Identidades singleton já estão
        # normalizadas (sums = a face única), mas a divisão segura cobre
        # tudo.
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        centroids = sums / np.maximum(norms, self.eps)

        if not np.all(np.isfinite(centroids)):
            raise RuntimeError(
                "Centroides por identidade contêm valores não finitos. "
                "Verifique embeddings degenerados."
            )

        n_singletons = int(np.sum(counts == 1))
        return centroids.astype(np.float64), n_singletons

    # =========================================================================
    # CORE COMPUTATION
    # =========================================================================

    def _compute_info_single(
        self, x_normalized: np.ndarray, *, include_top_k: bool,
    ) -> KNNTypicalityInfo:
        """
        x_normalized: (D,) L2-normalizado.
        """
        st = self._state
        assert st is not None

        if x_normalized.shape[0] != st.dim:
            raise ValueError(
                f"Dimensão do query ({x_normalized.shape[0]}) difere da "
                f"referência ({st.dim})."
            )

        sims = st.reference @ x_normalized.astype(st.reference.dtype)  # (N,)

        # P1-R2-8 (revisão externa 2026-06-09 round 2): SEM heurística
        # de self-match por cosseno. A versão anterior descartava o
        # vizinho mais próximo quando `cos > 1 - 1e-6`, mas isso
        # eliminava também vizinhos LEGÍTIMOS quase-idênticos (face
        # quase-duplicata, consulta externa coincidente com referência,
        # etc.) durante inferência externa.
        #
        # `_compute_info_single` é exclusivo da inferência sobre faces
        # NOVAS — self-match não é uma possibilidade legítima neste
        # caminho. O LOO interno do `fit` (cálculo de
        # `loo_mean_knn_sorted`) já exclui self por índice via
        # `np.fill_diagonal(sims, -np.inf)` ao construir o estado.
        n = sims.shape[0]
        k = min(self.k, n)
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_sim = sims[top_idx]
        desc_order = np.argsort(-top_sim)
        top_sim = top_sim[desc_order]
        top_idx_desc = top_idx[desc_order]

        # PR 5: T_k ponderado quando reference_score_weights estiver definido.
        if st.reference_score_weights is not None:
            top_w = np.asarray(
                st.reference_score_weights, dtype=np.float64
            )[top_idx_desc]
            row_w_sum = float(top_w.sum())
            renormalized = top_w / max(row_w_sum, self.eps)
            T_k = float(np.sum(top_sim * renormalized))
        else:
            T_k = float(np.mean(top_sim))
        theta_k = float(np.arccos(np.clip(top_sim[-1], -1.0, 1.0)))

        # PR 5: percentile via CDF ponderada quando disponível.
        if st.loo_cdf is not None:
            # rank(T_k) = cdf empírica avaliada em T_k.
            # Usa busca por valor + média entre side="left" e side="right"
            # para empatar (mid_rank ponderado).
            sorted_T = st.loo_mean_knn_sorted
            cdf = st.loo_cdf
            left = int(np.searchsorted(sorted_T, T_k, side="left"))
            right = int(np.searchsorted(sorted_T, T_k, side="right"))
            # cdf[left-1] é a fração acumulada de pesos com valor < T_k;
            # cdf[right-1] é a fração com valor ≤ T_k. mid-rank pondera
            # os dois.
            left_cdf = cdf[left - 1] if left > 0 else 0.0
            right_cdf = cdf[right - 1] if right > 0 else 0.0
            percentile = 100.0 * 0.5 * (left_cdf + right_cdf)
        else:
            rank = float(np.searchsorted(st.loo_mean_knn_sorted, T_k, side="left"))
            rank_right = float(np.searchsorted(st.loo_mean_knn_sorted, T_k, side="right"))
            mid_rank = 0.5 * (rank + rank_right)
            percentile = 100.0 * mid_rank / len(st.loo_mean_knn_sorted)

        d = st.dim
        theta_safe = max(theta_k, 1e-12)
        log_density = (
            np.log(self.k)
            - np.log(st.reference.shape[0])
            - (d - 1) * np.log(theta_safe)
        )

        return KNNTypicalityInfo(
            model_name=self.model_name,
            embedding_dim=d,
            k=self.k,
            sample_count=st.reference.shape[0],
            mean_knn_similarity=T_k,
            typicality_percentile=percentile,
            log_density_knn=float(log_density),
            angular_radius_to_kth_nn=theta_k,
            is_in_distribution=bool(T_k >= st.ood_threshold),
            knn_similarities=top_sim.tolist() if include_top_k else None,
            notes=None,
        )

    # =========================================================================
    # HELPERS DE VALIDAÇÃO / NORMALIZAÇÃO
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("KNNTypicalityModel não foi treinado. Chame fit() antes.")

    def _validate_and_normalize_2d(self, embeddings: ArrayLike) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("Formato esperado: (N, D) ou (D,).")
        # PR 4: aplica UnitNormPolicy
        # P2-R2-13 + P0-R3-6: telemetria. Snapshot do fit fica
        # imutável após o treino; chamadas subsequentes (inferência)
        # populam o slot separado.
        from faceaudit.services.unit_norm_validator import (
            apply_unit_norm_policy,
        )
        telemetry: dict = {}
        out = apply_unit_norm_policy(
            x, policy=self.unit_norm_policy,
            tolerance=self.unit_norm_tolerance, eps=self.eps,
            telemetry=telemetry,
        )
        if self.is_fitted():
            self._inference_unit_norm_telemetry_last = telemetry
        else:
            self._fit_unit_norm_telemetry = telemetry
        return out

    def _validate_and_normalize_1d(self, embedding: ArrayLike) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float64)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("Esperado um único embedding (1, D) ou (D,).")
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Formato esperado: vetor 1D.")
        # PR 4: aplica UnitNormPolicy via batch de tamanho 1
        from faceaudit.services.unit_norm_validator import (
            apply_unit_norm_policy,
        )
        return apply_unit_norm_policy(
            x.reshape(1, -1), policy=self.unit_norm_policy,
            tolerance=self.unit_norm_tolerance, eps=self.eps,
        )[0]
