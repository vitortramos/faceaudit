# faceaudit/services/cross_condition_two_covariance_plda_service.py

"""
Estimador PLDA cross-condition (two-covariance) para comparações forenses
entre condições distintas (ex.: foto-padrão frontal vs frame de CFTV lateral).

**Problema.** O `fit_two_covariance` clássico junta amostras de A e B em um
único dataset. Quando as condições são fortemente desbalanceadas em amostras
por identidade (ex.: 10 frontais e 2 laterais por pessoa), o W empírico é
dominado por resíduos da condição majoritária — a variabilidade cross-
condição que importa no caso real fica subponderada e o LR vira super-
confiante.

**Solução.** Estimar W per-condição e combinar com peso igual:

    W_A = scatter intra-classe na condição A (só de classes com ≥2 amostras em A)
    W_B = scatter intra-classe na condição B (só de classes com ≥2 amostras em B)
    W   = (W_A + W_B) / 2

Para B (between-class) e μ, balancear por classe em vez de por amostra:

    μ_c = (μ_c^A + μ_c^B) / 2     (se a identidade está nas duas condições;
                                   caso contrário, a média da condição
                                   disponível)
    μ   = mean_c(μ_c)
    B   = (1/C) Σ_c (μ_c − μ)(μ_c − μ)^T

**Scoring permanece inalterado.** O objeto final é um `TwoCovariancePLDAState`
e os kernels Q, P, const são pré-computados pela mesma rotina do
`fit_two_covariance` clássico. Só a estimação de (μ, B, W) muda.

**Dois modos de centragem do W** (configuráveis via `CrossConditionWithinMode`):

- PER_CONDITION: resíduos centrados em μ_c^cond (a média da condição
  daquela classe). Remove o deslocamento per-identidade do W — assume que
  CORAL/Whitening eliminou o shift cross-condição.

- BALANCED_IDENTITY (default): resíduos centrados em μ_c balanceada.
  O W absorve o deslocamento per-identidade da identidade — mais
  conservador forensicamente, reduz risco de LR superconfiante.

**Política de singletons (V1).** Identidades com 1 amostra em uma condição:
- entram em B/μ via μ_c balanceada;
- NÃO contribuem direta nem indiretamente como linha em W_cond
  (não há resíduo intra-classe estimável a partir de 1 amostra).

Em `BALANCED_IDENTITY`, um singleton pareado em A influencia *indiretamente*
os resíduos da OUTRA condição: por exemplo, ID com 10 amostras em A e 1
amostra em B tem μ_c balanceada puxada pela única lateral, e os resíduos
das 10 frontais ficam centrados nesse ponto puxado. Esta é a propriedade
desejada do estimador (a meia-soma puxa para o ponto cross-condição), mas
não é óbvia — daí o registro explícito.

**Convenções de normalização** (divergem por design do `fit_two_covariance`):

- W per-cond: `W_cond = S_w_cond / N_residuals_cond`, onde
  N_residuals_cond = Σ n_{c,cond} para classes c com n_{c,cond} ≥ 2 na
  condição cond. Alinha exatamente com `fit_two_covariance` que após filtrar
  singletons usa `N = X_valid.shape[0]`. Singletons não contribuem nem ao
  numerador nem ao denominador.
- W global: `W = (W_A + W_B) / 2` — média não-ponderada (vs `S_w/N` total
  no clássico) — peso igual por condição é o objetivo.
- B: `B = (1/C) Σ_c (μ_c − μ)(μ_c − μ)^T` — não-ponderada por n_c (vs
  ponderada no clássico) — peso igual por classe é o objetivo.

Magnitude relativa B/W cross-condition é comparável (não idêntica) à do
`fit_two_covariance` em datasets balanceados; assimetria do merge é
justamente o que o estimador remove.

**Shrinkage Ledoit-Wolf em cross-condition.** Para evitar que o estimador
automático reflita o desbalanceamento amostral, α é estimado per-condição
e combinado por média não-ponderada: `α = (α_A + α_B) / 2`. Espelha a
mesma lógica do W cross.

**Fora de escopo V1**: scorer heterocedástico W_probe ≠ W_reference.

Referências:
- Brümmer & de Villiers (2010) — formulação original two-covariance.
- O nome "cross-condition" segue a literatura speaker verification (NIST SRE)
  para PLDA treinado entre condições assimétricas (ex.: telefone vs microfone).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from faceaudit.enums.covariance_shrinkage_mode import CovarianceShrinkageMode
from faceaudit.enums.cross_condition_within_mode import CrossConditionWithinMode
from faceaudit.services.shrinkage_two_covariance_plda_service import (
    SHRINKAGE_TARGET_NAME,
    apply_shrinkage_to_w,
    ledoit_wolf_alpha,
)
from faceaudit.services.two_covariance_plda_service import TwoCovariancePLDAState


def _group_by_identity(y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Agrupa índices por label de identidade (string original). Retorna
    {identity_str: array_of_row_indices}. Labels None devem ter sido
    filtradas pelo caller.
    """
    groups: Dict[str, List[int]] = {}
    for i, label in enumerate(y):
        # Preserva a representação string original — não fatoriza para int,
        # senão a interseção entre A e B fica errada quando ordens divergem.
        key = str(label)
        groups.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def _compute_class_means(
    X: np.ndarray,
    groups: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """μ_c por identidade na condição. Singletons incluídos (a média é a própria amostra)."""
    return {ident: X[idxs].mean(axis=0) for ident, idxs in groups.items()}


def _accumulate_within_scatter(
    X: np.ndarray,
    groups: Dict[str, np.ndarray],
    center_fn: Callable[[str, np.ndarray], np.ndarray],
    D: int,
) -> tuple:
    """
    Acumula S_w para a condição, somando apenas classes com ≥2 amostras.

    `center_fn(identity, mu_c_cond) -> centro` decide o ponto de centragem:
    PER_CONDITION usa μ_c da própria condição; BALANCED_IDENTITY usa μ_c
    balanceada (passada do caller via closure).

    Retorna (S_w, N_residuals, residuals_all_stacked).
    """
    S_w = np.zeros((D, D), dtype=np.float64)
    N_residuals = 0
    residuals_chunks: List[np.ndarray] = []

    for ident, idxs in groups.items():
        if idxs.shape[0] < 2:
            # Singleton: não contribui pra W_cond (nem numerador nem denominador).
            # Comentário V1: política uniforme nas duas centragens.
            continue
        Xc = X[idxs]
        mu_c_cond = Xc.mean(axis=0)
        center = center_fn(ident, mu_c_cond)
        residuals = Xc - center  # (n_c, D)
        S_w += residuals.T @ residuals
        N_residuals += residuals.shape[0]
        residuals_chunks.append(residuals)

    residuals_all = (
        np.vstack(residuals_chunks)
        if residuals_chunks
        else np.empty((0, D), dtype=np.float64)
    )
    return S_w, N_residuals, residuals_all


def fit_cross_condition_two_covariance(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    *,
    within_mode: CrossConditionWithinMode = CrossConditionWithinMode.BALANCED_IDENTITY,
    shrinkage_mode: CovarianceShrinkageMode = CovarianceShrinkageMode.NONE,
    shrinkage_intensity: float = 0.1,
    regularization: float = 0.0,
    logger: Optional[Callable[[str], None]] = None,
) -> TwoCovariancePLDAState:
    """
    Estima (μ, B, W) cross-condition a partir de dois datasets condicionais.

    Args:
        X_a, y_a: matriz (N_A, D) + rótulos string para a condição A.
                  Singletons preservados pelo caller — não removidos aqui.
        X_b, y_b: análogo para a condição B. Deve ter mesma D.
        within_mode: PER_CONDITION ou BALANCED_IDENTITY (default).
        shrinkage_mode: NONE / LEDOIT_WOLF / MANUAL aplicado a W combinada.
                        Em LEDOIT_WOLF, α = (α_A + α_B) / 2.
        shrinkage_intensity: α em [0,1] usado apenas em MANUAL.
        regularization: ε ≥ 0 — Tikhonov final em W.
        logger: callable opcional pra diagnóstico verboso.

    Returns:
        TwoCovariancePLDAState com flags cross-condition populadas.

    Raises:
        ValueError: dims A ≠ B, dados insuficientes, W não positiva definida.
    """
    if logger is None:
        logger = lambda _msg: None  # noqa: E731

    X_a = np.asarray(X_a, dtype=np.float64)
    X_b = np.asarray(X_b, dtype=np.float64)
    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)

    if X_a.ndim != 2 or X_b.ndim != 2:
        raise ValueError("X_a e X_b devem ser 2D (N, D).")

    if X_a.shape[1] != X_b.shape[1]:
        raise ValueError(
            f"Dimensões incompatíveis entre condições: "
            f"A.D={X_a.shape[1]} vs B.D={X_b.shape[1]}."
        )

    if X_a.shape[0] != y_a.shape[0] or X_b.shape[0] != y_b.shape[0]:
        raise ValueError("X e y devem ter o mesmo número de linhas em cada condição.")

    if X_a.shape[0] < 1 or X_b.shape[0] < 1:
        raise ValueError("Ambas as condições devem ter ≥ 1 amostra.")

    if regularization < 0:
        raise ValueError("regularization deve ser ≥ 0.")

    if not (0.0 <= shrinkage_intensity <= 1.0):
        raise ValueError("shrinkage_intensity deve estar em [0, 1].")

    D = X_a.shape[1]

    # Agrupamento por labels string originais — preserva interseção correta
    # mesmo quando A e B têm ordens distintas.
    groups_a = _group_by_identity(y_a)
    groups_b = _group_by_identity(y_b)

    ids_a = set(groups_a.keys())
    ids_b = set(groups_b.keys())
    common_ids = ids_a & ids_b
    all_ids = ids_a | ids_b

    if len(all_ids) < 2:
        raise ValueError(
            "Treino cross-condition exige pelo menos 2 identidades distintas "
            "na união A ∪ B."
        )

    # μ_c por condição (inclui singletons — usado em B/μ).
    mu_c_a = _compute_class_means(X_a, groups_a)
    mu_c_b = _compute_class_means(X_b, groups_b)

    # μ_c balanceada: média das médias-por-condição disponíveis.
    mu_c_balanced: Dict[str, np.ndarray] = {}
    for ident in all_ids:
        means_available = []
        if ident in mu_c_a:
            means_available.append(mu_c_a[ident])
        if ident in mu_c_b:
            means_available.append(mu_c_b[ident])
        mu_c_balanced[ident] = np.mean(np.stack(means_available, axis=0), axis=0)

    # μ global = média não-ponderada das μ_c balanceadas.
    mu_balanced = np.mean(np.stack(list(mu_c_balanced.values()), axis=0), axis=0)

    # B = (1/C) Σ_c (μ_c − μ)(μ_c − μ)^T — peso igual por classe (não por n_c).
    C = len(all_ids)
    between_cov = np.zeros((D, D), dtype=np.float64)
    for ident in all_ids:
        delta = (mu_c_balanced[ident] - mu_balanced).reshape(-1, 1)
        between_cov += delta @ delta.T
    between_cov /= C

    # Função de centragem dos resíduos por within_mode.
    if within_mode == CrossConditionWithinMode.PER_CONDITION:
        center_fn_a = lambda _ident, mu_c_cond: mu_c_cond  # noqa: E731
        center_fn_b = lambda _ident, mu_c_cond: mu_c_cond  # noqa: E731
    elif within_mode == CrossConditionWithinMode.BALANCED_IDENTITY:
        # Os resíduos são centrados em μ_c balanceada — esse é o ponto onde
        # um singleton pareado em B influencia *indiretamente* W_A: μ_c
        # balanceada puxa para o ponto cross-condição e os resíduos de A
        # ficam centrados nesse ponto puxado. Comportamento desejado.
        center_fn_a = lambda ident, _mu_c_cond: mu_c_balanced[ident]  # noqa: E731
        center_fn_b = lambda ident, _mu_c_cond: mu_c_balanced[ident]  # noqa: E731
    else:
        raise ValueError(f"within_mode não reconhecido: {within_mode!r}")

    S_w_a, N_res_a, residuals_a = _accumulate_within_scatter(
        X_a, groups_a, center_fn_a, D
    )
    S_w_b, N_res_b, residuals_b = _accumulate_within_scatter(
        X_b, groups_b, center_fn_b, D
    )

    if N_res_a == 0 and N_res_b == 0:
        raise ValueError(
            "Nenhuma condição tem identidades com ≥ 2 amostras — impossível "
            "estimar W cross-condition."
        )
    if N_res_a == 0:
        raise ValueError(
            "Condição A (primária) não tem identidades com ≥ 2 amostras — "
            "impossível estimar W_A."
        )
    if N_res_b == 0:
        raise ValueError(
            "Condição B (secundária) não tem identidades com ≥ 2 amostras — "
            "impossível estimar W_B."
        )

    within_cov_a = S_w_a / N_res_a
    within_cov_b = S_w_b / N_res_b
    within_cov_cross = 0.5 * (within_cov_a + within_cov_b)

    # Shrinkage em W combinado. Em LEDOIT_WOLF, α = média per-condição.
    alpha_used: Optional[float] = None
    if shrinkage_mode == CovarianceShrinkageMode.NONE:
        within_cov_shrunk = within_cov_cross
        alpha_used = 0.0
    elif shrinkage_mode == CovarianceShrinkageMode.MANUAL:
        within_cov_shrunk, alpha_used = apply_shrinkage_to_w(
            W_emp=within_cov_cross,
            residuals=np.empty((0, D), dtype=np.float64),  # ignorado em MANUAL
            mode=CovarianceShrinkageMode.MANUAL,
            manual_intensity=shrinkage_intensity,
        )
        logger(f"[cross-condition] shrinkage MANUAL α={alpha_used:.4f}")
    elif shrinkage_mode == CovarianceShrinkageMode.LEDOIT_WOLF:
        alpha_a = ledoit_wolf_alpha(residuals_a)
        alpha_b = ledoit_wolf_alpha(residuals_b)
        # Média não-ponderada — espelha a lógica de W = (W_A + W_B)/2.
        # Evita subamostragem (sensível a ordenação) e empilhamento de
        # resíduos (re-introduz desbalanceamento).
        alpha_balanced = 0.5 * (alpha_a + alpha_b)
        logger(
            f"[cross-condition] LEDOIT_WOLF α_A={alpha_a:.4f}, "
            f"α_B={alpha_b:.4f}, α_balanced={alpha_balanced:.4f}"
        )
        within_cov_shrunk, alpha_used = apply_shrinkage_to_w(
            W_emp=within_cov_cross,
            residuals=np.empty((0, D), dtype=np.float64),
            mode=CovarianceShrinkageMode.MANUAL,
            manual_intensity=alpha_balanced,
        )
    else:
        raise ValueError(f"Modo de shrinkage não reconhecido: {shrinkage_mode!r}")

    # Tikhonov final só em W (mesma decisão metodológica do shrinkage
    # service: não injeta jitter em B).
    if regularization > 0:
        within_cov_shrunk = within_cov_shrunk + regularization * np.eye(
            D, dtype=np.float64
        )

    # Cholesky check com erro útil.
    try:
        np.linalg.cholesky(within_cov_shrunk)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Matriz W cross-condition não é positiva definida. "
            "Tente aumentar `shrinkage_intensity` (MANUAL) ou `regularization`. "
            f"(erro: {exc})"
        ) from exc

    # n_samples cross-condition: total de amostras válidas (singletons
    # incluídos — eles existem no dataset, só não contribuem pra W).
    n_samples_total = X_a.shape[0] + X_b.shape[0]
    # n_classes: união das identidades distintas.
    n_classes_total = len(all_ids)

    shrinkage_mode_str: Optional[str] = (
        shrinkage_mode.value
        if shrinkage_mode != CovarianceShrinkageMode.NONE
        else None
    )
    shrinkage_target_str: Optional[str] = (
        SHRINKAGE_TARGET_NAME if shrinkage_mode_str else None
    )
    shrinkage_alpha_w: Optional[float] = (
        float(alpha_used) if shrinkage_mode_str else None
    )

    logger(
        f"[cross-condition] fit done: D={D}, n_classes={n_classes_total}, "
        f"n_ids_A={len(ids_a)}, n_ids_B={len(ids_b)}, "
        f"n_common={len(common_ids)}, N_res_A={N_res_a}, N_res_B={N_res_b}, "
        f"within_mode={within_mode.value}"
    )

    return TwoCovariancePLDAState(
        mu=mu_balanced,
        B=between_cov,
        W=within_cov_shrunk,
        n_classes=int(n_classes_total),
        n_samples=int(n_samples_total),
        D=int(D),
        shrinkage_mode=shrinkage_mode_str,
        shrinkage_target=shrinkage_target_str,
        shrinkage_alpha_w=shrinkage_alpha_w,
        shrinkage_alpha_b=None,
        cross_condition=True,
        n_ids_condition_a=int(len(ids_a)),
        n_ids_condition_b=int(len(ids_b)),
        n_common_identities=int(len(common_ids)),
        cross_condition_within_mode=within_mode.value,
    )
