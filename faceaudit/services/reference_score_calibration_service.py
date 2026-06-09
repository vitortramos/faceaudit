# faceaudit/services/reference_score_calibration_service.py

"""Calibração da distribuição de referência de scores (CDF + threshold OOD).

Em PR 2 o serviço é esqueleto: implementa o caminho `IN_SAMPLE_LEGACY`
para o k-NN (que delega o fit nativo, que já tem LOO de imagem) e
levanta `NotImplementedError` para `CROSS_FIT_IDENTITY` — entra em PR 4.

A separação de responsabilidades segue a decisão do plano: modelo
matemático faz só o fit paramétrico (EM no MoVMF; armazenamento de
referência no k-NN). Serviço orquestra folds, scores OOF, CDF empírica,
threshold OOD, refit final em toda a população. Permite reuso futuro
para LOIO real do k-NN (substituir LOO de imagem).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.protocols.typicality_scoring_model_protocol import (
    FittableTypicalityModelProtocol,
)


def oof_quantile_in_range(q: float) -> bool:
    """0 < q < 0.5 — protege contra divisão por zero ou quantis acima da
    metade que não fazem sentido como threshold OOD inferior."""
    return q is not None and 0.0 < q < 0.5


@dataclass(frozen=True)
class ReferenceScoreCalibrationResult:
    """Resultado da calibração de score de referência.

    Em PR 2 (k-NN, IN_SAMPLE_LEGACY):
        - `refit_model`: o k-NN treinado nativamente (LOO de imagem).
        - `oof_cdf`: None (não há scores OOF nessa via).
        - `ood_threshold`: derivado do próprio modelo.
        - `split_fingerprint`: None.

    Em PR 4 (MoVMF, CROSS_FIT_IDENTITY):
        - `refit_model`: modelo treinado em toda a população (persistido).
        - `oof_cdf`: vetor (N_OOF,) ordenado de scores out-of-fold.
        - `ood_threshold`: quantil do `oof_cdf`.
        - `split_fingerprint`: hash do mapeamento identidade → fold.
        - `fold_seeds`: lista de seeds por fold.
    """

    refit_model: FittableTypicalityModelProtocol
    mode: ReferenceScoreCalibrationMode
    oof_cdf: Optional[np.ndarray] = None
    ood_threshold: Optional[float] = None
    split_fingerprint: Optional[str] = None
    fold_seeds: Optional[Sequence[int]] = None
    n_folds_used: int = 0


class ReferenceScoreCalibrationService:
    """Serviço de calibração de score de referência.

    Implementação por backend dispatchada via `TypicalityModelType`.
    Em PR 2 só `IN_SAMPLE_LEGACY` está implementado.
    """

    def calibrate(
        self,
        *,
        backend: TypicalityModelType,
        mode: ReferenceScoreCalibrationMode,
        trainer_factory,                                   # callable: () -> FittableTypicalityModelProtocol
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
        n_folds: int,
        random_state: int,
    ) -> ReferenceScoreCalibrationResult:
        """Dispatch principal.

        Args:
            backend: tipo do modelo (KNN / MOVMF).
            mode: CROSS_FIT_IDENTITY ou IN_SAMPLE_LEGACY.
            trainer_factory: callable sem args que constrói uma instância
                Fittable nova do backend, com hyperparams já fixados.
            embeddings: matriz (N, D) L2-normalizada (verificação fica no
                modelo conforme a `unit_norm_policy` do config).
            identity_ids: identidades por face. Obrigatório para
                `CROSS_FIT_IDENTITY` e para os modos populacionais
                baseados em identidade.
            n_folds: número de folds quando `CROSS_FIT_IDENTITY`.
            random_state: semente do split por identidade.
        """
        if mode == ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY:
            return self._calibrate_in_sample_legacy(
                trainer_factory=trainer_factory,
                embeddings=embeddings,
                identity_ids=identity_ids,
            )
        if mode == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY:
            return self._calibrate_cross_fit_identity(
                backend=backend,
                trainer_factory=trainer_factory,
                embeddings=embeddings,
                identity_ids=identity_ids,
                n_folds=n_folds,
                random_state=random_state,
            )
        raise ValueError(
            f"ReferenceScoreCalibrationService: modo não suportado: {mode}."
        )

    @staticmethod
    def _calibrate_cross_fit_identity(
        *,
        backend: TypicalityModelType,
        trainer_factory,
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
        n_folds: int,
        random_state: int,
    ) -> ReferenceScoreCalibrationResult:
        """K-fold por identidade → scores OOF → CDF + threshold; refit final.

        Procedimento (documentado em `docs/typicality_models.md` §5.1):

        1. Particiona identidades únicas em `n_folds` folds disjuntos.
        2. Para cada fold: treina backend nos folds train; pontua faces
           cujas identidades estão no fold test (scores OOF).
        3. Concatena OOF, ordena → CDF empírica; threshold OOD =
           quantile.
        4. Refit final em toda a população: o modelo persistido é este.

        Em PR 4 a implementação cobre k-NN (LOIO real, substituindo LOO
        de imagem) e MoVMF (k-fold por identidade). O `trainer_factory`
        é chamado uma vez por fold + uma vez no refit final.
        """
        import hashlib

        from faceaudit.enums.typicality_score_type import TypicalityScoreType

        if identity_ids is None or len(identity_ids) != embeddings.shape[0]:
            raise ValueError(
                "CROSS_FIT_IDENTITY exige `identity_ids` paralelo a "
                "`embeddings` (não pode ser None nem ter tamanho diferente)."
            )

        ids_arr = np.asarray(identity_ids, dtype=object)
        unique_ids = np.unique(ids_arr)
        if len(unique_ids) < n_folds:
            raise ValueError(
                f"CROSS_FIT_IDENTITY: {len(unique_ids)} identidades únicas "
                f"insuficientes para {n_folds} folds — reduza cv_n_folds "
                f"ou aumente a referência."
            )

        rng = np.random.default_rng(random_state)
        shuffled_ids = unique_ids.copy()
        rng.shuffle(shuffled_ids)
        # Splits balanceados por identidade
        fold_assignments = np.array_split(shuffled_ids, n_folds)

        # Fingerprint do split (identidade → fold index)
        identity_to_fold = {}
        for fold_idx, fold_ids in enumerate(fold_assignments):
            for ident in fold_ids:
                identity_to_fold[str(ident)] = fold_idx
        split_fp_payload = "\n".join(
            f"{ident}:{idx}"
            for ident, idx in sorted(identity_to_fold.items())
        )
        split_fingerprint = hashlib.sha256(
            split_fp_payload.encode("utf-8")
        ).hexdigest()

        # P0-4 (revisão externa 2026-06-09): coletamos scores OOF E os
        # identity_ids correspondentes. Antes só guardávamos scores e o
        # peso da CDF OOF era uniforme face-a-face — uma identidade com
        # 20 imagens contribuía 20× mais à CDF mesmo em
        # INVERSE_IDENTITY_FREQUENCY (frequência de coleta vazando
        # silenciosamente na referência populacional).
        oof_score_chunks: list[np.ndarray] = []
        oof_ids_chunks: list[np.ndarray] = []
        fold_seeds: list[int] = []
        for fold_idx, fold_ids in enumerate(fold_assignments):
            fold_seed = int(random_state) + fold_idx
            fold_seeds.append(fold_seed)
            train_mask = np.array(
                [str(i) not in set(map(str, fold_ids)) for i in ids_arr],
                dtype=bool,
            )
            test_mask = ~train_mask
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            # Treina backend no fold train
            fold_model = trainer_factory()
            train_ids = tuple(str(i) for i in ids_arr[train_mask])
            fold_model.fit(
                embeddings[train_mask], identity_ids=train_ids,
            )

            # Score faces do fold test usando o backend.
            # Para o cálculo de CDF/OOD escolhemos LOG_DENSITY_ESTIMATE
            # quando disponível (universal entre k-NN e MoVMF). Se o
            # backend não suportar, cai em TYPICALITY_PERCENTILE como
            # fallback. Para o k-NN ambos estão disponíveis após o fit.
            preferred_types = [
                TypicalityScoreType.LOG_DENSITY_ESTIMATE,
                TypicalityScoreType.MEAN_KNN_SIMILARITY,
            ]
            chosen_type = None
            for t in preferred_types:
                if t in fold_model.enabled_score_types:
                    chosen_type = t
                    break
            if chosen_type is None:
                raise RuntimeError(
                    "CROSS_FIT_IDENTITY: backend não expõe score type "
                    "compatível para construir CDF OOF."
                )

            batch = fold_model.compute_scores(
                embeddings[test_mask], chosen_type,
            )
            oof_score_chunks.append(np.asarray(batch.scores, dtype=np.float64))
            oof_ids_chunks.append(ids_arr[test_mask].copy())

        if not oof_score_chunks:
            raise RuntimeError(
                "CROSS_FIT_IDENTITY: nenhum score OOF produzido — verifique "
                "splits ou trainer."
            )
        oof_scores_unsorted = np.concatenate(oof_score_chunks)
        oof_ids_unsorted = np.concatenate(oof_ids_chunks)

        # Refit final em toda a população
        refit_model = trainer_factory()
        all_ids = tuple(str(i) for i in ids_arr)
        refit_model.fit(embeddings, identity_ids=all_ids)

        # P0-4: pesos OOF derivados do `population_weighting_mode` do modelo
        # treinado. Lógica:
        #   INVERSE_IDENTITY_FREQUENCY → w_ij = 1/n_i_oof por face (cada
        #     identidade contribui o mesmo, independente de quantas imagens
        #     têm em OOF).
        #   IDENTITY_CENTROIDS → agrega scores OOF por identidade (média) →
        #     1 ponto por identidade, w = 1. Coerente com o fit() de
        #     IDENTITY_CENTROIDS que vê 1 centroide por identidade.
        #   FACE_LEVEL_RAW → uniforme face-a-face (comportamento legado).
        pop_mode = getattr(refit_model, "population_weighting_mode", None)
        if pop_mode is not None:
            from faceaudit.enums.population_weighting_mode import (
                PopulationWeightingMode,
            )
            if pop_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY:
                _, inv_idx, id_counts = np.unique(
                    oof_ids_unsorted, return_inverse=True, return_counts=True,
                )
                oof_weights = (1.0 / id_counts[inv_idx]).astype(np.float64)
                oof_scores_final = oof_scores_unsorted
            elif pop_mode == PopulationWeightingMode.IDENTITY_CENTROIDS:
                uniq_ids, inv_idx = np.unique(
                    oof_ids_unsorted, return_inverse=True,
                )
                agg = np.zeros(len(uniq_ids), dtype=np.float64)
                cnt = np.zeros(len(uniq_ids), dtype=np.int64)
                for i, score in zip(inv_idx, oof_scores_unsorted):
                    agg[i] += score
                    cnt[i] += 1
                oof_scores_final = agg / np.maximum(cnt, 1)
                oof_weights = np.ones(len(uniq_ids), dtype=np.float64)
            else:  # FACE_LEVEL_RAW ou ausente
                oof_weights = np.ones_like(oof_scores_unsorted, dtype=np.float64)
                oof_scores_final = oof_scores_unsorted
        else:
            oof_weights = np.ones_like(oof_scores_unsorted, dtype=np.float64)
            oof_scores_final = oof_scores_unsorted

        # Ordena scores + pesos paralelamente para a CDF/threshold.
        order = np.argsort(oof_scores_final)
        oof_arr_sorted = oof_scores_final[order]
        oof_weights_sorted = oof_weights[order]

        # Threshold OOD = quantil ponderado em q = ood_quantile
        ood_quantile = float(getattr(refit_model, "ood_quantile", 0.01))
        if oof_quantile_in_range(ood_quantile):
            total_w = oof_weights_sorted.sum()
            cdf_arr = np.cumsum(oof_weights_sorted) / total_w
            idx = int(np.searchsorted(cdf_arr, ood_quantile, side="left"))
            idx = min(max(idx, 0), len(oof_arr_sorted) - 1)
            threshold = float(oof_arr_sorted[idx])
        else:
            threshold = None

        # P0-1: instala a referência OOF no modelo final. Após este
        # método retornar, `refit_model.get_percentiles()` e
        # `is_in_distribution()` consultam a distribuição OOF — não a
        # in-sample do refit. O `ReferenceScoreCalibrationResult`
        # continua retornando os mesmos arrays para auditoria.
        refit_model.install_reference_score_calibration(
            reference_scores=oof_arr_sorted,
            reference_weights=oof_weights_sorted,
            ood_threshold=threshold,
            calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            split_fingerprint=split_fingerprint,
        )

        return ReferenceScoreCalibrationResult(
            refit_model=refit_model,
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            oof_cdf=oof_arr_sorted,
            ood_threshold=threshold,
            split_fingerprint=split_fingerprint,
            fold_seeds=tuple(fold_seeds),
            n_folds_used=int(n_folds),
        )

    @staticmethod
    def _calibrate_in_sample_legacy(
        *,
        trainer_factory,
        embeddings: np.ndarray,
        identity_ids: Optional[Sequence[str]],
    ) -> ReferenceScoreCalibrationResult:
        """k-NN nativo já faz LOO de imagem internamente."""
        model = trainer_factory()
        if identity_ids is not None:
            model.fit(embeddings, identity_ids=tuple(str(i) for i in identity_ids))
        else:
            model.fit(embeddings)
        # Threshold OOD persistido no próprio modelo via `ood_quantile`.
        return ReferenceScoreCalibrationResult(
            refit_model=model,
            mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
            oof_cdf=None,
            ood_threshold=None,
            split_fingerprint=None,
            fold_seeds=None,
            n_folds_used=0,
        )
