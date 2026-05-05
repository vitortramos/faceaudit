# interface/tabs/strategy_manager_v2/comparison/comparison_kind.py

"""
Enum dos 3 tipos de comparação suportados pela ComparisonWindow.

Usado pelo `comparison_left_panel` (kind selector) e por `availability_resolver`
pra decidir quais (estratégia, recurso) podem ser comparados em cada modo.
"""

from __future__ import annotations

from enum import Enum


class ComparisonKind(Enum):
    """
    Cada membro carrega `label` (texto exibido na UI) e `description` (tooltip).
    """
    CALIBRATION_DATASET_DISTRIBUTION = (
        "Distribuição do dataset de calibração",
        "Compara o histograma + KDE dos scores genuínos vs impostores que "
        "alimentaram cada calibrador. Útil para entender o quão (dis)similares "
        "são as populações usadas por cada estratégia.",
    )
    VALIDATION_RESULTS = (
        "Resultados de Validação",
        "Compara métricas (EER, AUC, Cllr, Cllr_min, cal_loss) e gráficos "
        "(DET, Tippett, ECE, distribuição de log-LR) das instruções de "
        "validação executadas.",
    )
    CASEWORK_RESULTS = (
        "Resultados do Caso",
        "Compara as visualizações forenses do caso (referência gen/imp + "
        "scores do caso sobrepostos) produzidas pela instrução "
        "CASEWORK_VISUALIZATION.",
    )

    def __init__(self, label: str, description: str):
        self.label = label
        self.description = description
