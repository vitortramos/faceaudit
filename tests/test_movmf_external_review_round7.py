"""Regression tests para os 3 P0 + 3 P1 + 2 P2 da sétima rodada da
revisão externa (2026-06-09 round 7).

P0:
- R7-1: loaders NÃO coerçam silenciosamente `1.9 → 1` antes da
        validação (bypass do `_normalize_oof_per_fold_ints` estrito).
- R7-2: modelo CROSS_FIT_IDENTITY exige bloco OOF completo +
        `n_folds_used >= 2`.
- R7-3: `oof_predictive_score_type` deve bater com o backend
        (`LOG_DENSITY_ESTIMATE` em MoVMF, `MEAN_KNN_SIMILARITY` em
        k-NN). Outros valores → fail-closed.

P1:
- R7-4: comparação de `training_contract` é simétrica (chaves extras
        em `other` também são divergência).
- R7-5: factory que ignora `seed` (random_state hardcoded) é
        detectada via `EffectiveSeedMismatchError`.
- R7-6: detecção de colisão de IDs também no caminho legado
        (`IN_SAMPLE_LEGACY`).

P2:
- R7-7: `split_fingerprint` via JSON canônico — identidades contendo
        `:` ou `\\n` não geram mais colisão de hash.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.oof_audit_metrics import (
    IncompleteCrossFitAuditPayloadError,
    UnexpectedOofScoreTypeError,
)
from faceaudit.services.reference_score_calibration_service import (
    CrossFitContractMismatchError,
    EffectiveSeedMismatchError,
    IdentityCanonicalizationCollisionError,
    ReferenceScoreCalibrationService,
)


def _make_data(n_id=12, per=20, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d); c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(f"id_{i}")
    return np.asarray(X), tuple(ids)


def _good_factory_movmf(seed=None):
    return MoVMFTypicalityModel(
        n_components=1, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        random_state=seed if seed is not None else 0,
    )


def _train_and_dump(X, ids):
    """Treina via cross-fit e devolve o payload to_dict."""
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=_good_factory_movmf,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    return result.refit_model.to_dict()


# ─── P0-R7-1 ───────────────────────────────────────────────────────────


def test_from_dict_rejects_fractional_per_fold_values_before_cast():
    """`n_identities_by_fold=[3.9, 3, 3]` no payload — antes era
    truncado para `(3, 3, 3)` pelo `int(x)` defensivo do loader."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    # Adultera com float fracionário.
    payload["state"]["n_identities_by_fold"] = [3.9, 3, 3]
    with pytest.raises(ValueError, match="n_identities_by_fold"):
        MoVMFTypicalityModel.from_dict(payload)


def test_from_dict_rejects_fractional_n_folds_used():
    """`n_folds_used=3.9` no payload."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    payload["state"]["n_folds_used"] = 3.9
    with pytest.raises(ValueError, match="n_folds_used"):
        MoVMFTypicalityModel.from_dict(payload)


def test_from_dict_rejects_fractional_fold_seeds():
    """`fold_seeds=[0.5, 1, 2]` no payload."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    payload["state"]["fold_seeds"] = [0.5, 1, 2]
    with pytest.raises(ValueError, match="fold_seeds"):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R7-2 ───────────────────────────────────────────────────────────


def test_cross_fit_requires_complete_oof_audit_block():
    """Remover TODOS os diagnósticos OOF + fingerprint mantendo o
    modo CROSS_FIT_IDENTITY deve falhar."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    # Remove o bloco completo:
    for key in (
        "oof_predictive_avg_score", "oof_predictive_avg_score_by_fold",
        "oof_predictive_score_type", "n_identities_by_fold",
        "n_scores_by_fold", "ood_threshold_by_fold", "fold_seeds",
        "n_folds_used", "audit_metrics_fingerprint",
    ):
        payload["state"][key] = None
    # Mantém o modo CROSS_FIT_IDENTITY:
    assert payload["state"]["reference_score_calibration_mode"] == "CROSS_FIT_IDENTITY"
    with pytest.raises(IncompleteCrossFitAuditPayloadError):
        MoVMFTypicalityModel.from_dict(payload)


def test_cross_fit_rejects_n_folds_used_lt_2():
    """`n_folds_used = 1` é semanticamente IN_SAMPLE; rejeitado."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    payload["state"]["n_folds_used"] = 1
    # Trunca os arrays por-fold para 1 (caso contrário rejeita por
    # length mismatch antes da regra n_folds_used >= 2).
    for key in (
        "oof_predictive_avg_score_by_fold", "n_identities_by_fold",
        "n_scores_by_fold", "ood_threshold_by_fold", "fold_seeds",
    ):
        payload["state"][key] = list(payload["state"][key])[:1]
    # Recalcula fingerprint com os campos curtos para chegar à
    # validação de completude.
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    payload["state"]["audit_metrics_fingerprint"] = (
        _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=payload["state"]["oof_predictive_avg_score"],
            oof_predictive_avg_score_by_fold=tuple(
                payload["state"]["oof_predictive_avg_score_by_fold"]
            ),
            oof_predictive_score_type=payload["state"]["oof_predictive_score_type"],
            n_identities_by_fold=tuple(payload["state"]["n_identities_by_fold"]),
            n_scores_by_fold=tuple(payload["state"]["n_scores_by_fold"]),
            ood_threshold_by_fold=tuple(payload["state"]["ood_threshold_by_fold"]),
            split_fingerprint=payload["state"]["split_fingerprint"],
            fold_seeds=tuple(payload["state"]["fold_seeds"]),
            n_folds_used=1,
        )
    )
    with pytest.raises(IncompleteCrossFitAuditPayloadError, match="n_folds_used"):
        MoVMFTypicalityModel.from_dict(payload)


# ─── P0-R7-3 ───────────────────────────────────────────────────────────


def test_movmf_rejects_unknown_oof_score_type():
    """MoVMF deve persistir `LOG_DENSITY_ESTIMATE`. Outro valor → erro."""
    X, ids = _make_data()
    payload = _train_and_dump(X, ids)
    payload["state"]["oof_predictive_score_type"] = "ARBITRARY_SCALE"
    # Recalcula o fingerprint com o tipo malicioso para chegar à
    # validação de score_type.
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    payload["state"]["audit_metrics_fingerprint"] = (
        _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=payload["state"]["oof_predictive_avg_score"],
            oof_predictive_avg_score_by_fold=tuple(
                payload["state"]["oof_predictive_avg_score_by_fold"]
            ),
            oof_predictive_score_type="ARBITRARY_SCALE",
            n_identities_by_fold=tuple(payload["state"]["n_identities_by_fold"]),
            n_scores_by_fold=tuple(payload["state"]["n_scores_by_fold"]),
            ood_threshold_by_fold=tuple(payload["state"]["ood_threshold_by_fold"]),
            split_fingerprint=payload["state"]["split_fingerprint"],
            fold_seeds=tuple(payload["state"]["fold_seeds"]),
            n_folds_used=payload["state"]["n_folds_used"],
        )
    )
    with pytest.raises(UnexpectedOofScoreTypeError):
        MoVMFTypicalityModel.from_dict(payload)


def test_knn_rejects_unknown_oof_score_type():
    """k-NN deve persistir `MEAN_KNN_SIMILARITY`."""
    X, ids = _make_data()
    factory = lambda **kw: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.05,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    payload = result.refit_model.to_dict()
    payload["learned_state"]["oof_predictive_score_type"] = "LOG_DENSITY_ESTIMATE"
    from faceaudit.services.oof_audit_metrics import (
        _compute_audit_metrics_fingerprint,
    )
    payload["learned_state"]["audit_metrics_fingerprint"] = (
        _compute_audit_metrics_fingerprint(
            oof_predictive_avg_score=payload["learned_state"]["oof_predictive_avg_score"],
            oof_predictive_avg_score_by_fold=tuple(
                payload["learned_state"]["oof_predictive_avg_score_by_fold"]
            ),
            oof_predictive_score_type="LOG_DENSITY_ESTIMATE",
            n_identities_by_fold=tuple(payload["learned_state"]["n_identities_by_fold"]),
            n_scores_by_fold=tuple(payload["learned_state"]["n_scores_by_fold"]),
            ood_threshold_by_fold=tuple(payload["learned_state"]["ood_threshold_by_fold"]),
            split_fingerprint=payload["learned_state"]["split_fingerprint"],
            fold_seeds=tuple(payload["learned_state"]["fold_seeds"]),
            n_folds_used=payload["learned_state"]["n_folds_used"],
        )
    )
    with pytest.raises(UnexpectedOofScoreTypeError):
        KNNTypicalityModel.from_dict(payload)


# ─── P1-R7-4 ───────────────────────────────────────────────────────────


def test_contract_rejects_extra_keys_in_fold_snapshot():
    """Snapshot do fold com chave adicional → divergência."""
    X, ids = _make_data()
    call_count = [0]

    def factory_with_drift(*, seed=None):
        i = call_count[0]
        call_count[0] += 1
        m = MoVMFTypicalityModel(
            n_components=1, ood_quantile=0.0,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            random_state=seed if seed is not None else 0,
        )
        # No 2º fold (i=2: peek=0, fold0=1, fold1=2) injeta chave extra
        # no contrato.
        if i >= 2:
            original = m.cross_fit_training_contract_snapshot
            def with_extra():
                snap = original()
                snap["adversarial_extra_key"] = "boom"
                return snap
            m.cross_fit_training_contract_snapshot = with_extra
        return m

    with pytest.raises(
        CrossFitContractMismatchError,
        match=r"training_contract\.adversarial_extra_key",
    ):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=factory_with_drift,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=0,
        )


# ─── P1-R7-5 ───────────────────────────────────────────────────────────


def test_factory_must_propagate_effective_seed():
    """Factory que aceita `seed` mas instancia com `random_state=42`
    fixo → `EffectiveSeedMismatchError`."""
    X, ids = _make_data()

    def adversarial_factory(*, seed=None):
        # Aceita o argumento mas IGNORA: hardcoda random_state=42.
        return MoVMFTypicalityModel(
            n_components=1, ood_quantile=0.0,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            random_state=42,
        )

    with pytest.raises(EffectiveSeedMismatchError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=adversarial_factory,
            embeddings=X, identity_ids=ids,
            n_folds=3, random_state=10,  # seeds esperadas: 10, 11, 12
        )


# ─── P1-R7-6 ───────────────────────────────────────────────────────────


def test_legacy_mode_rejects_identity_string_collisions():
    """`IN_SAMPLE_LEGACY` agora também detecta colisão (1, "1")."""
    rng = np.random.default_rng(0)
    n_per_id, d = 6, 12
    X, ids = [], []
    raw_ids = [1, "1", 2, "2"]
    centers = [rng.normal(size=d) for _ in range(4)]
    for c in centers:
        c /= np.linalg.norm(c)
    for orig_id, center in zip(raw_ids, centers):
        for _ in range(n_per_id):
            v = center + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(orig_id)
    X = np.asarray(X)
    factory = lambda **kw: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    with pytest.raises(IdentityCanonicalizationCollisionError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
            backend=TypicalityModelType.KNN,
            trainer_factory=factory,
            embeddings=X, identity_ids=tuple(ids),
            n_folds=0, random_state=0,
        )


# ─── P2-R7-7 ───────────────────────────────────────────────────────────


def test_split_fingerprint_uses_unambiguous_serialization():
    """Identidades com `:` produzem JSON canônico bem-formado.
    Distintos mapas de identidade → folds geram fingerprints distintos.
    """
    # Cenário A: id "a:1" -> fold 0, "b" -> fold 1
    # Cenário B: id "a" -> fold 0, "1:b" -> fold 1
    # Sob a serialização antiga `f"{id}:{fold}"`, ambos viravam algo como
    # `a:1:0\nb:1` e `a:0\n1:b:1` — strings diferentes mas o padrão
    # ambíguo. JSON canônico não tem essa fragilidade.
    rng = np.random.default_rng(0)
    d = 16
    centers = [rng.normal(size=d), rng.normal(size=d)]
    centers = [c / np.linalg.norm(c) for c in centers]
    X = []
    raw_ids_A = []
    raw_ids_B = []
    for c, idA, idB in zip(centers, ["a:1", "b"], ["a", "1:b"]):
        for _ in range(8):
            v = c + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v)
            raw_ids_A.append(idA)
            raw_ids_B.append(idB)
    X = np.asarray(X)
    factory = _good_factory_movmf
    rA = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=tuple(raw_ids_A),
        n_folds=2, random_state=0,
    )
    rB = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X, identity_ids=tuple(raw_ids_B),
        n_folds=2, random_state=0,
    )
    # Os dois fingerprints devem ser distintos (mapas distintos).
    assert rA.split_fingerprint != rB.split_fingerprint
    # Sanity: ambos hex SHA-256 (64 chars).
    assert len(rA.split_fingerprint) == 64
    assert len(rB.split_fingerprint) == 64
