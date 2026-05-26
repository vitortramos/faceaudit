# faceaudit/services/calibrator_cross_validation_service.py

"""
Cross-validation k-fold do calibrador final (LogReg/KDE/BiGauss).

Caso de uso: estimar `Cllr CV` no domínio de calibração, isolado de
domain shift (treino calib ≠ validação externa) e da limitação do score
underlying. Usado pelo `CalibrationModelTrainingInstructionExecutor`
quando `config.k_folds > 0`.

**Semântica do Cllr CV produzido**: out-of-fold no domínio do treino do
calibrador. NÃO é uma estimativa da performance em dados de validação
externos (BFW, etc.) — se houver mismatch de distribuição entre
calibração (CFP) e validação (BFW), o Cllr CV continua medindo apenas
a generalização dentro de CFP. O gap `cllr_validation - cllr_cv` mede
domain shift, separado do `cv_gap = cllr_cv - cllr_train` que mede
overfit in-domain.

**Hyperparameter mode**: v1 só suporta FROZEN — `factory()` cria
calibrador com hyperparams já resolvidos no fit principal (best_C_,
bandwidths finais, etc.). NESTED (re-roda GridSearch por fold) fica
documentado como future work via `cv_hyperparameter_mode = "FROZEN"`
no metadata pra auditoria.

**Split modes**:
- STRATIFIED_PAIR: sklearn.StratifiedKFold (preserva proporção
  genuíno/impostor). Otimista em datasets onde pares da mesma
  identidade aparecem em treino e held-out.
- STRATIFIED_GROUP: sklearn.StratifiedGroupKFold por identidade do
  probe. Forense-defensável — nenhuma identidade do probe usada pra
  treinar o calibrador no fold aparece no held-out. Mais pessimista
  (mais honesto).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from faceaudit.enums.calibrator_cv_split_mode import CalibratorCVSplitMode
from faceaudit.protocols.calibration_model_protocol import CalibrationModelProtocol
from faceaudit.services.cllr_math import compute_cllr


def compute_calibrator_cllr_cv(
    scores: np.ndarray,
    labels: np.ndarray,
    factory: Callable[[], CalibrationModelProtocol],
    k_folds: int,
    random_seed: int,
    split_mode: CalibratorCVSplitMode,
    groups: Optional[np.ndarray],
    logger_callback: Callable[[str], None],
    model_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Roda k-fold CV do calibrador.

    Args:
        scores: array (N,) ou (N, D) de scores brutos do dataset de treino do
            calibrador.
        labels: array (N,) com 1=genuíno, 0=impostor.
        factory: callable que cria uma NOVA instância do calibrador com
            hyperparams CONGELADOS (best_C_/bandwidths/sigma já resolvidos
            no fit principal — sem GridSearch interno por fold). Cada call
            retorna instância nova, virgem.
        k_folds: k desejado.
        random_seed: seed pra reproducibilidade do shuffle do KFold.
        split_mode: STRATIFIED_PAIR ou STRATIFIED_GROUP.
        groups: obrigatório se split_mode=STRATIFIED_GROUP; ignorado caso
            contrário. Tipicamente probe_identity por amostra.
        logger_callback: chamado com strings de progresso por fold.
        model_name: pra prefixar logs ("[<name>] ...").

    Returns:
        Dict com:
            "cllr_cv": float
            "cllr_cv_gen": float
            "cllr_cv_imp": float
            "cv_folds_effective": int  (reduzido se classe minoritária < k)
            "cv_split_mode": str
        Ou None se:
            - k_folds < 2
            - classe minoritária < 2 (sem CV possível)
            - STRATIFIED_GROUP sem groups (fallback NÃO automático —
              caller é responsável por fallback p/ PAIR antes de chamar)
    """
    if k_folds < 2:
        return None

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int32)

    n_gen = int((labels_arr == 1).sum())
    n_imp = int((labels_arr == 0).sum())
    min_class = min(n_gen, n_imp)
    if min_class < 2:
        logger_callback(
            f"[{model_name}] ⚠ Cllr CV: classe minoritária com {min_class} "
            f"amostra(s) — CV não aplicável, skip."
        )
        return None

    # Reduz k automaticamente se classe minoritária for menor.
    k_effective = min(k_folds, min_class)
    if k_effective < k_folds:
        logger_callback(
            f"[{model_name}] ⚠ Cllr CV: reduzindo k de {k_folds} para {k_effective} "
            f"(classe minoritária tem {min_class} amostras)."
        )

    # Escolhe splitter.
    if split_mode == CalibratorCVSplitMode.STRATIFIED_GROUP:
        if groups is None:
            logger_callback(
                f"[{model_name}] ⚠ STRATIFIED_GROUP requer 'groups' — "
                "skip CV. Caller deveria ter feito fallback pra STRATIFIED_PAIR."
            )
            return None
        from sklearn.model_selection import StratifiedGroupKFold
        groups_arr = np.asarray(groups)
        if groups_arr.shape[0] != labels_arr.shape[0]:
            logger_callback(
                f"[{model_name}] ⚠ groups tem {groups_arr.shape[0]} entradas, "
                f"esperado {labels_arr.shape[0]} — skip CV."
            )
            return None
        # StratifiedGroupKFold do sklearn não aceita random_state direto se
        # shuffle=False (default). Pra reproducibilidade, shuffle=True.
        splitter = StratifiedGroupKFold(
            n_splits=k_effective, shuffle=True, random_state=int(random_seed),
        )
        split_iter = splitter.split(scores_arr, labels_arr, groups=groups_arr)
        split_label = "STRATIFIED_GROUP"
    else:
        from sklearn.model_selection import StratifiedKFold
        splitter = StratifiedKFold(
            n_splits=k_effective, shuffle=True, random_state=int(random_seed),
        )
        split_iter = splitter.split(scores_arr, labels_arr)
        split_label = "STRATIFIED_PAIR"

    logger_callback(
        f"[{model_name}] 🧪 Cllr CV start: k={k_effective}, mode={split_label}, "
        f"N_gen={n_gen}, N_imp={n_imp}"
    )

    held_out_llrs = np.empty(labels_arr.shape[0], dtype=np.float64)
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_scores = scores_arr[train_idx]
        train_labels = labels_arr[train_idx]
        test_scores = scores_arr[test_idx]

        n_tr_gen = int((train_labels == 1).sum())
        n_tr_imp = int((train_labels == 0).sum())
        logger_callback(
            f"[{model_name}]   • Fold {fold_idx}/{k_effective}: "
            f"train n_gen={n_tr_gen}, n_imp={n_tr_imp}; "
            f"test n={test_scores.shape[0]}"
        )

        # Calibrador FRESH a cada fold — `factory()` retorna instância nova
        # com hyperparams já congelados do production model.
        calibrator = factory()
        try:
            calibrator.fit(
                scores=train_scores, labels=train_labels,
                logger_callback=lambda _msg: None,  # silencia fit interno
            )
            held_out_llrs[test_idx] = calibrator.calibrate_scores(test_scores)
        except Exception as exc:
            logger_callback(
                f"[{model_name}] ⚠ Fold {fold_idx} falhou: {exc} — skip CV."
            )
            return None

    cv_gen = held_out_llrs[labels_arr == 1]
    cv_imp = held_out_llrs[labels_arr == 0]
    cllr_cv, cllr_cv_gen, cllr_cv_imp = compute_cllr(cv_gen, cv_imp)

    logger_callback(
        f"[{model_name}] ✅ Cllr CV (k={k_effective}, {split_label}) = "
        f"{cllr_cv:.6f}  (gen={cllr_cv_gen:.6f}, imp={cllr_cv_imp:.6f})"
    )

    return {
        "cllr_cv": cllr_cv,
        "cllr_cv_gen": cllr_cv_gen,
        "cllr_cv_imp": cllr_cv_imp,
        "cv_folds_effective": int(k_effective),
        "cv_split_mode": split_label,
    }
