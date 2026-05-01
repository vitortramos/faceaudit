# faceaudit/models/casework_visualization_result.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CaseworkVisualizationResult:
    """
    DTO de resultado da instrução CASEWORK_VISUALIZATION.

    Carrega os arrays prontos para plot — distribuição de scores genuínos
    e impostores da população de referência (validação) + scores do caso
    forense — junto com metadata de rastreabilidade e o threshold de EER
    computado on-the-fly (apenas no modo RAW).

    Persistido em disco via pickle (preserva ndarray nativamente sem
    perda de precisão).
    """

    # Modo de operação: "RAW" (raw_score brutos do similarity model) ou
    # "CALIBRATED" (log-LR calibrado). Define o eixo X do plot.
    score_type: str

    # Arrays da população de referência (validação) — ground truth conhecido.
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray

    # Array do caso forense — sem ground truth.
    case_scores: np.ndarray

    # Rastreabilidade: nomes dos datasets de origem.
    reference_dataset_name: str = ""
    case_dataset_name: str = ""

    # Threshold de decisão sobreposto no plot:
    # - CALIBRATED: sempre 0 (limiar neutro de log-LR); este campo fica None
    #   para sinalizar "use 0 como referência".
    # - RAW: threshold de EER computado on-the-fly do dataset de referência
    #   (FNR=FPR crossover). Plot mostra linha tracejada nesse ponto.
    eer_threshold: Optional[float] = None

    # Contagens (cacheadas para summarizer e header do dialog).
    n_genuine: int = 0
    n_impostor: int = 0
    n_case: int = 0

    # Sumário estatístico do array de caso — pré-computado no executor para
    # exibição no header do dialog sem custo a cada reabertura.
    case_summary: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Garantia defensiva: arrays como np.ndarray, contagens consistentes.
        self.genuine_scores = np.asarray(self.genuine_scores, dtype=np.float64)
        self.impostor_scores = np.asarray(self.impostor_scores, dtype=np.float64)
        self.case_scores = np.asarray(self.case_scores, dtype=np.float64)

        if self.n_genuine == 0:
            self.n_genuine = int(self.genuine_scores.size)
        if self.n_impostor == 0:
            self.n_impostor = int(self.impostor_scores.size)
        if self.n_case == 0:
            self.n_case = int(self.case_scores.size)

        if not self.case_summary and self.case_scores.size > 0:
            self.case_summary = {
                "min":    float(np.min(self.case_scores)),
                "max":    float(np.max(self.case_scores)),
                "mean":   float(np.mean(self.case_scores)),
                "median": float(np.median(self.case_scores)),
            }
