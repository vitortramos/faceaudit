# faceaudit/services/shrinkage_two_covariance_plda_service.py

"""
Variante do PLDA de Duas Covariâncias com shrinkage de Ledoit-Wolf (ou α
manual) aplicado à matriz intra-classe W.

Decisões metodológicas:

(a) **Escopo W-only.** O shrinkage é aplicado APENAS à covariância intra-classe
    W. B (entre-classes) é mantida empírica para não injetar variância
    entre-classes artificial — alterar B mexeria diretamente na noção de
    "quão dispersa é a população de identidades", que é exatamente a
    quantidade que o LLR está medindo.

(b) **Target hardcoded scaled identity.** Single target da literatura forense
    (Ledoit & Wolf 2004): `F = (tr(W)/D) · I`. Outros targets (DIAGONAL,
    BLOCK_DIAGONAL) ficam para uma versão futura — YAGNI.

(c) **`regularization` aplicado SÓ em W.** Na variante TWO_COVARIANCE pura,
    `regularization` (ε·I, Tikhonov) é somado em B e W. Aqui, é somado
    apenas em W. Mesma razão de (a): não queremos jitter artificial em B.

(d) **Scoring reutilizado.** A formulação de scoring é idêntica ao
    TWO_COVARIANCE puro (mesmos kernels Q, P, const). Importamos as
    funções de `two_covariance_plda_service` em vez de duplicar.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from faceaudit.enums.covariance_shrinkage_mode import CovarianceShrinkageMode
from faceaudit.services.two_covariance_plda_service import TwoCovariancePLDAState


SHRINKAGE_TARGET_NAME = "SCALED_IDENTITY"


def ledoit_wolf_alpha(residuals: np.ndarray) -> float:
    """
    Computa α ótimo de Ledoit-Wolf (2004) para shrinkage contra o alvo
    scaled-identity `F = (tr(S)/D) · I`, onde S = R.T @ R / N é a covariância
    empírica dos resíduos.

    Fórmula:
        S = R.T @ R / N
        μ = tr(S) / D
        d² = ||S − μI||²_F / D
        b̄² = (1/N²) · Σ_n ||r_n r_nᵀ − S||²_F / D
        b² = min(d², b̄²)
        α  = clip(b² / d², 0, 1)

    Otimização: o termo ||r_n r_nᵀ − S||²_F é expandido como
        ||r_n||⁴ − 2·(r_nᵀ S r_n) + ||S||²_F
    para evitar materializar outer products O(D²) por amostra.

    Args:
        residuals: (N, D) matriz de resíduos centrados por classe (x_i - μ_c_i).

    Returns:
        α ∈ [0, 1].
    """
    R = np.asarray(residuals, dtype=np.float64)
    if R.ndim != 2 or R.shape[0] < 2:
        return 0.0

    N, D = R.shape

    # Covariância empírica dos resíduos.
    S = (R.T @ R) / N
    mu_trace = np.trace(S) / D

    # d² = ||S − μI||²_F / D.
    S_minus_target = S - mu_trace * np.eye(D, dtype=np.float64)
    d2 = float(np.sum(S_minus_target * S_minus_target)) / D

    if d2 <= 0.0:
        # Já é proporcional ao alvo — nenhum shrinkage adicional ajuda.
        return 0.0

    # b̄² = média de ||r_n r_nᵀ − S||²_F / D, vetorizado via identidade
    # ||r r^T − S||²_F = ||r||⁴ − 2·rᵀ S r + ||S||²_F.
    r_norm_sq = np.einsum("ij,ij->i", R, R)       # (N,) = ||r_n||²
    r_norm_4 = r_norm_sq * r_norm_sq               # (N,) = ||r_n||⁴
    rSr = np.einsum("ij,jk,ik->i", R, S, R)       # (N,) = r_nᵀ S r_n
    S_fro_sq = float(np.sum(S * S))                # ||S||²_F

    per_sample = r_norm_4 - 2.0 * rSr + S_fro_sq   # (N,)
    b2_bar = float(np.mean(per_sample)) / D / N    # divide por N → equivalente a 1/N²·Σ/D

    b2 = min(d2, b2_bar)
    return float(np.clip(b2 / d2, 0.0, 1.0))


def apply_shrinkage_to_w(
    W_emp: np.ndarray,
    residuals: np.ndarray,
    mode: CovarianceShrinkageMode,
    manual_intensity: float,
) -> Tuple[np.ndarray, float]:
    """
    Aplica shrinkage à covariância intra-classe W contra o alvo
    `(tr(W_emp)/D) · I`.

    Retorna `(W_shrunk, alpha_used)`.
    """
    W = np.asarray(W_emp, dtype=np.float64)
    D = W.shape[0]

    if mode == CovarianceShrinkageMode.NONE:
        return W, 0.0

    if mode == CovarianceShrinkageMode.MANUAL:
        alpha = float(manual_intensity)
    elif mode == CovarianceShrinkageMode.LEDOIT_WOLF:
        alpha = ledoit_wolf_alpha(residuals)
    else:
        raise ValueError(f"Modo de shrinkage não reconhecido: {mode!r}")

    alpha = float(np.clip(alpha, 0.0, 1.0))

    if alpha == 0.0:
        return W, 0.0

    mu_trace = np.trace(W) / D
    target = mu_trace * np.eye(D, dtype=np.float64)
    W_shrunk = (1.0 - alpha) * W + alpha * target
    return W_shrunk, alpha


def fit_shrinkage_two_covariance(
    X: np.ndarray,
    y: np.ndarray,
    shrinkage_mode: CovarianceShrinkageMode = CovarianceShrinkageMode.LEDOIT_WOLF,
    shrinkage_intensity: float = 0.1,
    regularization: float = 0.0,
) -> TwoCovariancePLDAState:
    """
    Estima μ, B e W a partir de embeddings rotulados, com shrinkage aplicado
    a W (B permanece empírica). Mesma filtragem e estimação base de
    `fit_two_covariance` — a única diferença é o tratamento de W.

    Args:
        X: (N, D) matriz de embeddings.
        y: (N,) rótulos de classe (identidades singleton são descartadas).
        shrinkage_mode: NONE / LEDOIT_WOLF (default) / MANUAL.
        shrinkage_intensity: α ∈ [0, 1] usado apenas em MANUAL.
        regularization: ε ≥ 0 — Tikhonov final SOMENTE em W (diferente da
            variante TWO_COVARIANCE pura, que aplica em B e W).

    Returns:
        TwoCovariancePLDAState com os campos de auditoria de shrinkage
        populados (`shrinkage_mode`, `shrinkage_target`, `shrinkage_alpha_w`,
        `shrinkage_alpha_b=None`).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError(f"X deve ser 2D (N, D), recebeu shape {X.shape}.")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X e y devem ter o mesmo nº de amostras: "
            f"{X.shape[0]} vs {y.shape[0]}."
        )

    if X.shape[0] < 2:
        raise ValueError("Treino exige pelo menos 2 amostras.")

    if regularization < 0:
        raise ValueError("regularization deve ser ≥ 0.")

    if not (0.0 <= shrinkage_intensity <= 1.0):
        raise ValueError("shrinkage_intensity deve estar em [0, 1].")

    # Filtra identidades com menos de 2 amostras — não contribuem para W.
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_mask = np.isin(y, unique_classes[counts >= 2])

    if valid_mask.sum() < 2:
        raise ValueError(
            "Após filtrar identidades singleton, restaram menos de 2 amostras."
        )

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    unique_classes = np.unique(y_valid)
    n_classes = unique_classes.shape[0]
    N = X_valid.shape[0]
    D = X_valid.shape[1]

    if n_classes < 2:
        raise ValueError(
            "Treino exige pelo menos 2 classes distintas com ≥ 2 amostras cada."
        )

    mu = X_valid.mean(axis=0)

    # Acumuladores S_w e S_b + coleta de resíduos intra-classe centrados.
    # Os resíduos alimentam o estimador Ledoit-Wolf.
    S_w = np.zeros((D, D), dtype=np.float64)
    S_b = np.zeros((D, D), dtype=np.float64)
    residuals_chunks = []

    for c in unique_classes:
        mask = y_valid == c
        Xc = X_valid[mask]
        n_c = Xc.shape[0]

        mu_c = Xc.mean(axis=0)
        residuals = Xc - mu_c            # (n_c, D)
        S_w += residuals.T @ residuals
        residuals_chunks.append(residuals)

        delta = (mu_c - mu).reshape(-1, 1)
        S_b += n_c * (delta @ delta.T)

    W_emp = S_w / N
    B = S_b / N

    residuals_all = np.vstack(residuals_chunks) if residuals_chunks else np.empty((0, D))

    W_shrunk, alpha_w = apply_shrinkage_to_w(
        W_emp=W_emp,
        residuals=residuals_all,
        mode=shrinkage_mode,
        manual_intensity=shrinkage_intensity,
    )

    # Regularização Tikhonov final — SOMENTE em W. B permanece empírica.
    if regularization > 0:
        W_shrunk = W_shrunk + regularization * np.eye(D, dtype=np.float64)

    # Cholesky check em W (B só precisa ser semidefinida positiva).
    try:
        np.linalg.cholesky(W_shrunk)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Matriz W (intra-classe) não é positiva definida mesmo após shrinkage. "
            "Tente um modo MANUAL com α maior, ou aumente `regularization`."
            f" (erro: {exc})"
        ) from exc

    return TwoCovariancePLDAState(
        mu=mu,
        B=B,
        W=W_shrunk,
        n_classes=int(n_classes),
        n_samples=int(N),
        D=int(D),
        shrinkage_mode=shrinkage_mode.value,
        shrinkage_target=SHRINKAGE_TARGET_NAME,
        shrinkage_alpha_w=float(alpha_w),
        shrinkage_alpha_b=None,
    )
