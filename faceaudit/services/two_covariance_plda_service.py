# faceaudit/services/two_covariance_plda_service.py

"""
Implementação do PLDA de Duas Covariâncias (Brümmer & de Villiers 2010).

Modelo generativo:
    y_c   ~ N(μ, B)        # média de classe c
    x|y_c ~ N(y_c, W)      # observação dada a classe

onde B (between-class) e W (within-class) são matrizes de covariância
completas D×D — sem subspace de baixo rank, ao contrário da formulação
simplificada usada pelo SpeechBrain.

Treino: forma fechada (estimadores de máxima verossimilhança baseados nos
scatter matrices intra-classe e entre-classes). Sem EM iterativo.

Scoring: log likelihood ratio direto via inversão dos blocos covariantes.
Para um par (x1, x2):
    log LR = (1/2) z1^T Q z1 + (1/2) z2^T Q z2 + (1/2) z1^T (N-M) z2 + const
onde z = (x - μ), K = inv(W+B), M = inv(W+2B), N = inv(W),
Q = K - 0.5*(M+N), e const = log|W+B| - 0.5 log|W+2B| - 0.5 log|W|.

Referências:
- Niko Brümmer & Edward de Villiers, "The speaker partitioning problem",
  Odyssey Speaker and Language Recognition Workshop 2010.
- Brümmer, "Application-independent evaluation of speaker detection",
  Computer Speech & Language 2006 (raízes da formulação log-LR).

Implementação 100% numpy. Testada por roundtrip simétrico (LR(x1,x2) =
LR(x2,x1)) e pelo ganho assintótico esperado em embeddings sintéticos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TwoCovariancePLDAState:
    """
    Estado treinado do PLDA de duas covariâncias.

    Atributos:
        mu: vetor médio global (D,)
        B: covariância entre-classes (D, D), simétrica positiva semi-definida
        W: covariância intra-classe  (D, D), simétrica positiva definida
        n_classes: número de classes (com ≥ 2 amostras) usadas no treino
        n_samples: total de amostras usadas no treino
        D: dimensionalidade dos embeddings
    """
    mu: np.ndarray
    B: np.ndarray
    W: np.ndarray
    n_classes: int
    n_samples: int
    D: int


def fit_two_covariance(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 0.0,
) -> TwoCovariancePLDAState:
    """
    Estima μ, B e W a partir de embeddings rotulados.

    Args:
        X: (N, D) — matriz de embeddings (float64 recomendado).
        y: (N,)  — rótulos de classe (strings ou ints). Identidades com
                   < 2 amostras são DESCARTADAS (não contribuem para W).
        regularization: ε ≥ 0 — multiplicador para somar a B e W como
                   ε * I (estabilização numérica em alta dimensionalidade
                   ou poucas classes). Padrão 0 = sem regularização.

    Retorna:
        TwoCovariancePLDAState pronto para uso em score_*.

    Raises:
        ValueError: dados insuficientes ou matrizes degeneradas.
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
        raise ValueError(
            "Treino exige pelo menos 2 amostras."
        )

    # Filtra identidades com menos de 2 amostras — não contribuem para W
    # (variância intra-classe não estimável a partir de uma única amostra).
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_mask = np.isin(y, unique_classes[counts >= 2])

    if valid_mask.sum() < 2:
        raise ValueError(
            "Após filtrar identidades singleton, restaram menos de 2 amostras. "
            "Treino inviável."
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

    # Média global (μ).
    mu = X_valid.mean(axis=0)

    # Acumuladores para os scatter matrices.
    # S_w = Σ_c Σ_i (x_i - μ_c)(x_i - μ_c)^T  → within-class scatter
    # S_b = Σ_c n_c (μ_c - μ)(μ_c - μ)^T       → between-class scatter
    S_w = np.zeros((D, D), dtype=np.float64)
    S_b = np.zeros((D, D), dtype=np.float64)

    for c in unique_classes:
        mask = y_valid == c
        Xc = X_valid[mask]
        n_c = Xc.shape[0]

        mu_c = Xc.mean(axis=0)
        residuals = Xc - mu_c            # (n_c, D)
        S_w += residuals.T @ residuals   # within accumulator

        delta = (mu_c - mu).reshape(-1, 1)   # (D, 1)
        S_b += n_c * (delta @ delta.T)        # between accumulator (weighted)

    # Estimadores de máxima verossimilhança (biased) — convenção da
    # literatura de PLDA. Funcionam bem na prática; a alternativa unbiased
    # introduz ajustes pequenos que raramente mudam decisões forenses.
    W = S_w / N
    B = S_b / N

    # Regularização (Tikhonov) — opcional. Útil quando D é alto comparado
    # a n_classes ou N, ou quando há colinearidade nos embeddings.
    if regularization > 0:
        eye = np.eye(D, dtype=np.float64)
        W = W + regularization * eye
        B = B + regularization * eye

    # Sanidade: W deve ser positiva definida (invertível). Se não for,
    # o usuário deve ajustar regularization.
    try:
        np.linalg.cholesky(W)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Matriz W (intra-classe) não é positiva definida. "
            "Tente aumentar `regularization` ou verificar duplicatas no "
            f"dataset. (erro: {exc})"
        ) from exc

    return TwoCovariancePLDAState(
        mu=mu,
        B=B,
        W=W,
        n_classes=int(n_classes),
        n_samples=int(N),
        D=int(D),
    )


# =============================================================================
# Scoring
# =============================================================================


def _compute_scoring_kernels(
    state: TwoCovariancePLDAState,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pré-computa os kernels Q, P e a constante log-determinante para scoring.

    Retorna:
        Q: (D, D) — operador quadrático aplicado a cada lado do par.
        P: (D, D) — operador bilinear (cross-term) entre os dois lados.
        const: escalar — termo log-determinante constante do log-LR.

    Fórmula:
        K = inv(W + B), M = inv(W + 2B), N_inv = inv(W)
        Q = K - 0.5 * (M + N_inv)
        P = N_inv - M
        const = log|W+B| - 0.5 log|W+2B| - 0.5 log|W|
    """
    W = state.W
    B = state.B

    K = np.linalg.inv(W + B)
    M = np.linalg.inv(W + 2.0 * B)
    N_inv = np.linalg.inv(W)

    Q = K - 0.5 * (M + N_inv)
    P = N_inv - M

    sign_kb, logdet_kb = np.linalg.slogdet(W + B)
    sign_k2b, logdet_k2b = np.linalg.slogdet(W + 2.0 * B)
    sign_w, logdet_w = np.linalg.slogdet(W)

    if not (sign_kb > 0 and sign_k2b > 0 and sign_w > 0):
        raise ValueError(
            "Determinantes de (W+B), (W+2B) ou W não positivos — matrizes "
            "degeneradas. Considere usar regularization > 0 no fit."
        )

    const = logdet_kb - 0.5 * logdet_k2b - 0.5 * logdet_w
    return Q, P, float(const)


def score_two_covariance_pair(
    state: TwoCovariancePLDAState,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """
    Computa o log-LR para um único par (v1, v2).

    Para uso massivo (matriz completa de pares), prefira
    `score_two_covariance_matrix` que pré-computa Q/P uma vez só.
    """
    v1 = np.asarray(v1, dtype=np.float64).ravel()
    v2 = np.asarray(v2, dtype=np.float64).ravel()

    if v1.shape[0] != v2.shape[0] != state.D:
        raise ValueError(
            f"Dimensões incompatíveis: v1.shape={v1.shape}, "
            f"v2.shape={v2.shape}, treinada D={state.D}."
        )

    Q, P, const = _compute_scoring_kernels(state)
    z1 = v1 - state.mu
    z2 = v2 - state.mu

    quad1 = float(z1 @ Q @ z1)
    quad2 = float(z2 @ Q @ z2)
    cross = float(z1 @ P @ z2)

    return 0.5 * (quad1 + quad2 + cross) + const


def score_two_covariance_matrix(
    state: TwoCovariancePLDAState,
    probe: np.ndarray,
    reference: np.ndarray,
    kernels: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
) -> np.ndarray:
    """
    Computa a matriz completa de log-LRs entre `probe` (n_p × D) e
    `reference` (n_r × D), retornando shape (n_p, n_r).

    Se `kernels` for passado (resultado de `_compute_scoring_kernels`),
    evita recomputar — útil em loops externos. Caso contrário, computa
    on-the-fly.
    """
    P_mat = np.asarray(probe, dtype=np.float64)
    R_mat = np.asarray(reference, dtype=np.float64)

    if P_mat.ndim != 2 or R_mat.ndim != 2:
        raise ValueError("probe e reference devem ser 2D.")

    if P_mat.shape[1] != state.D or R_mat.shape[1] != state.D:
        raise ValueError(
            f"Dimensão dos embeddings ({P_mat.shape[1]}, {R_mat.shape[1]}) "
            f"diferente da dim treinada ({state.D})."
        )

    n_p = P_mat.shape[0]
    n_r = R_mat.shape[0]
    if n_p == 0 or n_r == 0:
        return np.empty((n_p, n_r), dtype=np.float64)

    Q, P, const = kernels if kernels is not None else _compute_scoring_kernels(state)

    Z1 = P_mat - state.mu  # (n_p, D)
    Z2 = R_mat - state.mu  # (n_r, D)

    # Termos quadráticos: x^T Q x para cada linha. einsum vetorizado.
    Z1_quad = np.einsum("ij,jk,ik->i", Z1, Q, Z1)  # (n_p,)
    Z2_quad = np.einsum("ij,jk,ik->i", Z2, Q, Z2)  # (n_r,)

    # Termo bilinear cross: Z1 P Z2^T → (n_p, n_r)
    Cross = Z1 @ P @ Z2.T

    # Combina e adiciona constante.
    S = 0.5 * (Z1_quad[:, None] + Z2_quad[None, :] + Cross) + const
    return S
