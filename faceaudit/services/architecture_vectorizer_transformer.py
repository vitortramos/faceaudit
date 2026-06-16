# faceaudit/services/architecture_vectorizer_transformer.py
"""
Transformer sklearn-compat que materializa a arquitetura de calibração
LR dentro de um Pipeline.

Permite que s₀ e r̄_F/r̄_P sejam ajustados POR FOLD durante o
GridSearchCV interno do calibrador (PR 3 / v2), eliminando o leakage
residual que existe quando esses parâmetros são estimados sobre todo
D_cal antes da CV (modo FIT_FULL_CALIBRATION_LEGACY do PR 1).

Contrato esperado de X
----------------------
`X` é uma matriz `(N, K)` onde as colunas seguem o schema
`LRArchitectureRawFeatureSchema`:

  * `X[:, schema.base_score_column]` → score base (cosseno).
  * `X[:, schema.rarity_probe_column]` → raridade canônica do probe
    (None na arquitetura quando não consumida).
  * `X[:, schema.rarity_reference_column]` → idem reference.

O `LR_ARCHITECTURE_VECTORIZATION` em modo `FIT_PER_FOLD` emite o
`LR_ARCHITECTURE_RAW_INPUT` com essa matriz e o `CALIBRATION_MODEL_TRAINING`
monta o pipeline `[ArchitectureVectorizerTransformer → StandardScaler →
LogReg]` dentro do `GridSearchCV(cv=IdentityDisjointPairKFold(...))`.

Após a escolha de C, o `best_estimator_` é refitado em todo D_cal
automaticamente pelo `GridSearchCV(refit=True)`. O `state_` final do
transformer (acessível via `pipeline.named_steps["architecture_vectorizer"]
.state_`) é exatamente o ajuste sobre D_cal completo — usado para
persistir o `architecture_state` no metadata do calibrador e suportar
INHERIT em VALIDATION/CASEWORK.

Sentinela de leakage
--------------------
A propriedade central a verificar é que perturbar APENAS as linhas que
caem no test fold do CV NÃO altera `state_` aprendido no train fold.
Isso é estrutural — o transformer só vê `X_train` durante `.fit`. Ver
`tests/test_architecture_vectorizer_transformer.py::test_no_leakage`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from faceaudit.enums.architecture_fit_scope import ArchitectureFitScope
from faceaudit.enums.lr_calibration_architecture import LRCalibrationArchitecture
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.s0_resolution_method import S0ResolutionMethod
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    LRArchitectureRawFeatureSchema,
)
from faceaudit.models.lr.lr_architecture_spec import LR_ARCHITECTURE_SPECS
from faceaudit.models.lr.lr_architecture_state import LRArchitectureStateFactory
from faceaudit.services import architecture_feature_builder as _builder
from faceaudit.services.s0_resolution_service import (
    S0ResolutionError,
    resolve_s0_auto_logreg,
)


class ArchitectureVectorizerTransformer(BaseEstimator, TransformerMixin):
    """
    Pipeline component sklearn-compat para vetorização por arquitetura LR.

    Parâmetros (`__init__`) são clonáveis pelo sklearn — qualquer mudança
    em `architecture`/`schema`/`s0_logreg_config` é replicada no clone.
    Aprendizagem (`state_`) só existe em instâncias fitadas (convenção
    sklearn do sufixo `_`).

    Parameters
    ----------
    architecture : LRCalibrationArchitecture
        Arquitetura escolhida. Consulta `LR_ARCHITECTURE_SPECS` no fit.
    schema : LRArchitectureRawFeatureSchema
        Layout de colunas de `X` esperado.
    s0_logreg_config : dict, opcional
        Override dos params do LogReg auxiliar do AUTO_LOGREG
        (C, solver, class_weight, max_iter, coef_tolerance). Default usa
        valores estáveis do `resolve_s0_auto_logreg`.
    base_score_kind : str
        Documenta o tipo do score base ("cosine"/"plda") no state.
        Default "cosine".
    pair_role_orientation : PairRoleOrientation
        Orientação semântica para arquiteturas ordenadas. Default
        PROBE_REFERENCE (cross-pose canônico).
    rarity_probe_score_type : str, opcional
        Identificador do score_type da tipicidade do probe (para
        rastreabilidade no state — vide validate_state_compatibility).
    rarity_reference_score_type : str, opcional
        Idem reference.

    Attributes
    ----------
    state_ : LRArchitectureState
        Estado ajustado pós-`fit`. Carrega s₀, r̄_F, r̄_P, fingerprints
        e proveniência completa. Cria-se com `fit_scope=
        PER_FOLD_REFIT_FULL_CALIBRATION` para distinguir do modo legado.
    """

    def __init__(
        self,
        architecture: LRCalibrationArchitecture = LRCalibrationArchitecture.COSINE_ONLY,
        schema: Optional[LRArchitectureRawFeatureSchema] = None,
        s0_logreg_config: Optional[dict] = None,
        base_score_kind: str = "cosine",
        pair_role_orientation: PairRoleOrientation = PairRoleOrientation.PROBE_REFERENCE,
        rarity_probe_score_type: Optional[str] = None,
        rarity_reference_score_type: Optional[str] = None,
        rarity_single_score_type: Optional[str] = None,
    ):
        self.architecture = architecture
        self.schema = schema
        self.s0_logreg_config = s0_logreg_config
        self.base_score_kind = base_score_kind
        self.pair_role_orientation = pair_role_orientation
        self.rarity_probe_score_type = rarity_probe_score_type
        self.rarity_reference_score_type = rarity_reference_score_type
        self.rarity_single_score_type = rarity_single_score_type

    # ── sklearn protocol ────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "ArchitectureVectorizerTransformer":
        """
        Ajusta s₀ via AUTO_LOGREG e r̄_F/r̄_P como média em `X`.

        Crítico para a sentinela de leakage: `X` é apenas o `train_fold`
        recebido pelo GridSearchCV. O state aprendido aqui depende
        exclusivamente das linhas de treino — perturbar dados do test
        fold (fora do `X` passado a este `fit`) não afeta `state_`.
        """
        if self.schema is None:
            raise ValueError("schema é obrigatório (LRArchitectureRawFeatureSchema)")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X deve ser 2D, recebido shape={X_arr.shape}")
        if X_arr.shape[1] != self.schema.n_columns:
            raise ValueError(
                f"X.shape[1]={X_arr.shape[1]} != schema.n_columns="
                f"{self.schema.n_columns}"
            )

        spec = LR_ARCHITECTURE_SPECS[self.architecture]
        requires_single = getattr(spec, "requires_typicality_single", False)

        base_scores = X_arr[:, self.schema.base_score_column]
        rarity_probe = (
            X_arr[:, self.schema.rarity_probe_column]
            if self.schema.rarity_probe_column is not None
            else None
        )
        rarity_reference = (
            X_arr[:, self.schema.rarity_reference_column]
            if self.schema.rarity_reference_column is not None
            else None
        )
        rarity_single = (
            X_arr[:, self.schema.rarity_single_column]
            if self.schema.rarity_single_column is not None
            else None
        )

        # ── s₀ ──
        s0_value = None
        s0_fit_summary = None
        s0_resolution_method = S0ResolutionMethod.NOT_APPLICABLE
        if spec.requires_s0:
            if y is None:
                raise ValueError(
                    "Arquitetura exige s₀ via AUTO_LOGREG, mas y (labels) "
                    "não foi fornecido a fit()"
                )
            cfg = self.s0_logreg_config or {}
            try:
                summary = resolve_s0_auto_logreg(
                    base_scores,
                    np.asarray(y, dtype=int),
                    C=float(cfg.get("C", 1.0)),
                    solver=str(cfg.get("solver", "liblinear")),
                    class_weight=cfg.get("class_weight", "balanced"),
                    max_iter=int(cfg.get("max_iter", 1000)),
                    coef_tolerance=float(cfg.get("coef_tolerance", 1e-8)),
                )
            except S0ResolutionError as exc:
                raise ValueError(
                    f"AUTO_LOGREG falhou no fit do transformer: {exc}"
                ) from exc
            s0_value = summary.s0_value
            s0_fit_summary = summary.to_dict_hex()
            s0_resolution_method = S0ResolutionMethod.AUTO_LOGREG

        # ── r̄_F, r̄_P, r̄_single ──
        r0_probe = None
        r0_reference = None
        r0_single = None
        r0_probe_estimator = None
        r0_reference_estimator = None
        r0_single_estimator = None
        if spec.centering_mode.value == "MEAN":
            if spec.requires_typicality_probe:
                if rarity_probe is None:
                    raise ValueError(
                        "Arquitetura usa centragem MEAN no probe, mas "
                        "schema não declara rarity_probe_column"
                    )
                r0_probe = float(rarity_probe.mean())
                r0_probe_estimator = "MEAN"
            if spec.requires_typicality_reference:
                if rarity_reference is None:
                    raise ValueError(
                        "Arquitetura usa centragem MEAN no reference, mas "
                        "schema não declara rarity_reference_column"
                    )
                r0_reference = float(rarity_reference.mean())
                r0_reference_estimator = "MEAN"
            if requires_single:
                if rarity_single is None:
                    raise ValueError(
                        "Arquitetura ONE_FACE usa centragem MEAN, mas "
                        "schema não declara rarity_single_column"
                    )
                r0_single = float(rarity_single.mean())
                r0_single_estimator = "MEAN"

        # ── contagens (provenance) ──
        if y is not None:
            y_arr = np.asarray(y, dtype=int)
            n_genuine = int((y_arr == 1).sum())
            n_impostor = int((y_arr == 0).sum())
        else:
            n_genuine = None
            n_impostor = None

        # ── monta state ──
        self.state_ = LRArchitectureStateFactory.create(
            spec=spec,
            s0=s0_value,
            r0_probe=r0_probe,
            r0_reference=r0_reference,
            r0_single=r0_single,
            base_score_kind=self.base_score_kind,
            pair_role_orientation=(
                self.pair_role_orientation
                if spec.requires_ordered_pair_roles
                else PairRoleOrientation.NOT_APPLICABLE
            ),
            rarity_probe_score_type=self.rarity_probe_score_type,
            rarity_reference_score_type=self.rarity_reference_score_type,
            rarity_single_score_type=self.rarity_single_score_type,
            fit_scope=ArchitectureFitScope.PER_FOLD_REFIT_FULL_CALIBRATION,
            s0_resolution_method=s0_resolution_method,
            r0_probe_estimator=r0_probe_estimator,
            r0_reference_estimator=r0_reference_estimator,
            r0_single_estimator=r0_single_estimator,
            n_fitted_pairs=len(X_arr),
            n_fitted_genuine_pairs=n_genuine,
            n_fitted_impostor_pairs=n_impostor,
            s0_logreg_config=self.s0_logreg_config,
            s0_fit_summary=s0_fit_summary,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Aplica state_ aprendido. Pode ser chamado sobre test_fold (CV)
        ou sobre todo D_cal (refit final) — state_ é o mesmo em qualquer
        invocação após `fit`.
        """
        check_is_fitted(self, "state_")
        if self.schema is None:
            raise ValueError("schema é obrigatório")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X deve ser 2D, recebido shape={X_arr.shape}")
        if X_arr.shape[1] != self.schema.n_columns:
            raise ValueError(
                f"X.shape[1]={X_arr.shape[1]} != schema.n_columns="
                f"{self.schema.n_columns}"
            )

        spec = LR_ARCHITECTURE_SPECS[self.architecture]
        base_scores = X_arr[:, self.schema.base_score_column]
        rarity_probe = (
            X_arr[:, self.schema.rarity_probe_column]
            if self.schema.rarity_probe_column is not None
            else None
        )
        rarity_reference = (
            X_arr[:, self.schema.rarity_reference_column]
            if self.schema.rarity_reference_column is not None
            else None
        )
        rarity_single = (
            X_arr[:, self.schema.rarity_single_column]
            if self.schema.rarity_single_column is not None
            else None
        )
        return _builder.build(
            spec, self.state_, base_scores, rarity_probe, rarity_reference,
            rarity_single,
        )

    def get_feature_names_out(
        self, input_features: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Convenção sklearn — retorna nomes das features produzidas pelo
        `transform`. Útil para ColumnTransformer e debugging.
        """
        check_is_fitted(self, "state_")
        return np.asarray(self.state_.feature_names, dtype=object)
