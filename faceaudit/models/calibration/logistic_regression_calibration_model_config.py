# faceaudit/models/calibration/logistic_regression_calibration_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.calibrator_cv_split_mode import CalibratorCVSplitMode
from faceaudit.enums.logreg_grid_scoring_metric import LogRegGridScoringMetric
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.calibration.piecewise_linear_calibration_config import (
    PiecewiseLinearCalibrationConfig,
)


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

    # ── Cross-validation diagnóstico (opt-in) ────────────────────────────
    k_folds: int = field(
        default=0,
        metadata={"help": (
            "<html><b>🧪 K-fold CV diagnóstico (opt-in)</b> — pra estimar "
            "Cllr CV out-of-fold no <b>domínio de calibração</b>.<br/><br/>"
            "<code>0</code> (padrão) — desativado, comportamento atual idêntico.<br/>"
            "<code>5</code> — recomendado quando ligado. Roda 5 folds "
            "estratificados após o fit principal usando os hiperparâmetros "
            "do production model (C congelado). Anexa <code>calibrator_cllr_cv</code> "
            "ao metadata, propagado pra ValidationMetrics.<br/><br/>"
            "<b>Atenção</b>: o Cllr CV mede generalização DENTRO do domínio "
            "de calibração, NÃO performance em validação externa. Para LogReg, "
            "o gap CV − in-sample tende a ser pequeno (calibrador robusto)."
            "</html>"
        )}
    )

    cv_split_mode: CalibratorCVSplitMode = field(
        default=CalibratorCVSplitMode.STRATIFIED_PAIR,
        metadata={"help": (
            "<html><b>Estratégia de partição dos folds</b> em k-fold CV "
            "(só aplicável quando <code>k_folds &gt; 0</code>).<br/><br/>"
            "<b>STRATIFIED_PAIR</b> (padrão) — sklearn.StratifiedKFold. "
            "Preserva proporção genuíno/impostor. Pares da mesma identidade "
            "podem aparecer em ambos os folds → otimista.<br/><br/>"
            "<b>STRATIFIED_GROUP</b> — sklearn.StratifiedGroupKFold por "
            "<code>probe_identity</code>. Nenhuma identidade de probe usada "
            "no treino do fold aparece no held-out. <b>Forense-defensável</b>, "
            "mais pessimista, mais honesto. Recomendado em laudos.</html>"
        )}
    )

    piecewise_linear_config: PiecewiseLinearCalibrationConfig = field(
        default_factory=PiecewiseLinearCalibrationConfig,
        metadata={"help": (
            "Configuração da expansão piecewise-linear interna. Default = "
            "mode NONE (backward-compat: calibrador opera direto sobre o "
            "score sem expansão). Quando ativada (mode=PIECEWISE_LINEAR), "
            "o calibrador aprende internamente `hinge_anchor_s0` fold-aware "
            "e expande a feature em [z, max(z-offset_j, 0)]. Restrita a "
            "score 1D na V1 — multivariado deve usar SCORE_INTERACTION_FEATURE."
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
        # k_folds = 0 desativa CV diagnóstico; >=2 ativa; 1 inválido.
        if self.k_folds < 0:
            raise ValueError("k_folds não pode ser negativo.")
        if self.k_folds == 1:
            raise ValueError("k_folds deve ser 0 (desativado) ou >= 2.")
        # Normaliza enum vindo de string (caso from_dict externo)
        if not isinstance(self.cv_split_mode, CalibratorCVSplitMode):
            self.cv_split_mode = CalibratorCVSplitMode.from_str(self.cv_split_mode)
        # Normaliza PiecewiseLinearCalibrationConfig vindo de dict
        if isinstance(self.piecewise_linear_config, dict):
            self.piecewise_linear_config = (
                PiecewiseLinearCalibrationConfig.from_dict(
                    self.piecewise_linear_config
                )
            )
        elif not isinstance(
            self.piecewise_linear_config, PiecewiseLinearCalibrationConfig
        ):
            self.piecewise_linear_config = PiecewiseLinearCalibrationConfig()

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

        if "cv_split_mode" in processed:
            processed["cv_split_mode"] = CalibratorCVSplitMode.from_str(
                processed["cv_split_mode"]
            )

        if "piecewise_linear_config" in processed and isinstance(
            processed["piecewise_linear_config"], dict
        ):
            processed["piecewise_linear_config"] = (
                PiecewiseLinearCalibrationConfig.from_dict(
                    processed["piecewise_linear_config"]
                )
            )

        return cls(**processed)