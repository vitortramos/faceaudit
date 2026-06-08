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
            # PR 4 implementa; PR 2 fail-fast com mensagem explícita.
            raise NotImplementedError(
                "ReferenceScoreCalibrationService: CROSS_FIT_IDENTITY entra "
                "em PR 4. Em PR 2 use IN_SAMPLE_LEGACY (default migratório do "
                "k-NN); novas estratégias devem aguardar PR 4 para "
                "CROSS_FIT_IDENTITY."
            )
        raise ValueError(
            f"ReferenceScoreCalibrationService: modo não suportado: {mode}."
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
