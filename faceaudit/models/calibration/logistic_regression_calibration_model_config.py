# faceaudit/models/calibration/logistic_regression_calibration_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.logreg_grid_scoring_metric import LogRegGridScoringMetric
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="LogisticRegressionCalibrationModelConfig")


@dataclass
class LogisticRegressionCalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração por Regressão Logística.

    Esta configuração representa apenas parâmetros do calibrador, sem misturar:
    - modelo de similaridade
    - pipeline de transformação
    - estratégia de execução
    - mineração de pares

    Suporta dois modos de seleção do hiperparâmetro `C`:
    - Fixo (padrão): usa o valor de `C` informado.
    - GridSearch: ignora `C` e busca o melhor valor em `c_grid` via cross-validation
      estratificada com `cv_folds` folds, otimizando `scoring` (ex: "neg_log_loss").
    """

    C: float = field(
        default=1.0,
        metadata={"help": "Parâmetro de regularização inversa. Ignorado quando use_gridsearch=True."}
    )

    use_gridsearch: bool = field(
        default=True,
        metadata={"help": (
            "Se ATIVO (padrão), busca o melhor C via GridSearch + cross-validation "
            "estratificada. Recomendado: o C ótimo varia muito conforme o dataset "
            "e dificilmente é 1.0."
        )}
    )

    c_grid: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0],
        metadata={"help": "Grade de valores de C testados quando use_gridsearch=True."}
    )

    cv_folds: int = field(
        default=5,
        metadata={"help": "Número de folds da cross-validation (StratifiedKFold) usada no GridSearch."}
    )

    scoring: LogRegGridScoringMetric = field(
        default=LogRegGridScoringMetric.NEG_LOG_LOSS,
        metadata={"help": (
            "Métrica de avaliação no GridSearch. NEG_LOG_LOSS (padrão) é ideal "
            "para calibração — penaliza miscalibração diretamente. ROC_AUC "
            "otimiza separabilidade/ranking, não calibração."
        )}
    )

    standardize_features: bool = field(
        default=False,
        metadata={"help": (
            "Quando True E o input X for matriz multivariada (N, D) com D > 1, "
            "aplica z-score por feature (μ, σ aprendidos no fit) antes do "
            "LogReg. Necessário em score vectorization quando os sistemas "
            "têm escalas dispares (ex: Cosseno em [-0.5, +0.5] + PLDA em "
            "[-200, +60]). Sem standardização, a L2 do sklearn penaliza pesos "
            "uniformemente, distorcendo a contribuição relativa das features. "
            "Para X 1D (calibração univariada normal), o flag é ignorado. "
            "μ/σ são persistidos com o modelo e aplicados automaticamente em "
            "score_calibration (treino/validação/casework)."
        )}
    )

    def __post_init__(self) -> None:
        """
        Validação defensiva mínima do DTO.
        """
        if self.C <= 0:
            raise ValueError("C deve ser maior que zero.")
        if self.use_gridsearch:
            if not self.c_grid or any(c <= 0 for c in self.c_grid):
                raise ValueError(
                    "c_grid não pode ser vazio e todos os valores devem ser > 0 "
                    "quando use_gridsearch=True."
                )
            if self.cv_folds < 2:
                raise ValueError("cv_folds deve ser >= 2 quando use_gridsearch=True.")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO de configuração a partir de dicionário.
        Coerce de c_grid: aceita string "0.01,0.1,1.0" (UI), lista mista, etc.
        Coerce de scoring: aceita string ("neg_log_loss") ou enum.
        """
        if not data:
            return cls()

        processed = data.copy()

        if "c_grid" in processed:
            raw = processed["c_grid"]
            if isinstance(raw, str):
                processed["c_grid"] = [
                    float(tok.strip()) for tok in raw.split(",") if tok.strip()
                ]
            elif isinstance(raw, (list, tuple)):
                processed["c_grid"] = [float(v) for v in raw]

        if "scoring" in processed and isinstance(processed["scoring"], str):
            processed["scoring"] = LogRegGridScoringMetric.from_str(processed["scoring"])

        return cls(**processed)