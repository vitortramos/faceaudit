# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/__init__.py

"""
Widgets de visualização de resultados de instrução, reutilizáveis tanto
inline (na coluna de Resultados do editor de estratégias) quanto wrapped
em QDialog (para visualização em tela cheia).

Cada widget:
- Recebe o objeto de resultado da instrução (dataset / validation /
  casework / etc.) via construtor ou `set_result(...)`.
- Constrói um `matplotlib.Figure` populado com o(s) gráfico(s).
- Expõe um `FigureCanvas` para embedding em layouts Qt.
- Pode opcionalmente exibir um `NavigationToolbar2QT` (ativável via flag
  `show_toolbar`) — útil em modo dialog/expandido, dispensável inline.

Esses widgets foram extraídos de `validation_result_viewer_dialog.py`,
`casework_visualization_viewer_dialog.py` e `_draw_score_distribution`
(originalmente no controller V2) sem alterar comportamento — os dialogs
existentes agora são wrappers finos sobre eles.
"""

from .score_distribution_widget import ScoreDistributionWidget
from .validation_result_widget import ValidationResultWidget
from .casework_visualization_widget import CaseworkVisualizationWidget
from .hubness_report_widget import HubnessReportWidget
from .hubness_ablation_widget import HubnessAblationWidget
from .score_comparison_table_widget import ScoreComparisonTableWidget

__all__ = [
    "ScoreDistributionWidget",
    "ValidationResultWidget",
    "CaseworkVisualizationWidget",
    "HubnessReportWidget",
    "HubnessAblationWidget",
    "ScoreComparisonTableWidget",
]
