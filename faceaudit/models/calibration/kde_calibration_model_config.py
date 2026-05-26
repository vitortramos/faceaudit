# faceaudit/models/calibration/kde_calibration_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.calibrator_cv_split_mode import CalibratorCVSplitMode
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="KDECalibrationModelConfig")


@dataclass
class KDECalibrationModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de calibração KDE.

    Este calibrador modela separadamente:
    - a densidade dos scores genuínos
    - a densidade dos scores impostores

    e transforma score bruto em LR/log-LR calibrado.

    A configuração aqui é propositalmente enxuta e focada no calibrador em si,
    sem misturar:
    - mineração de pares
    - comparação entre embeddings
    - estratégia de similaridade
    """

    bandwidth_genuine: float = field(
        default=0.1,
        metadata={"help": "Bandwidth do KDE para scores genuínos."}
    )

    bandwidth_impostor: float = field(
        default=0.1,
        metadata={"help": "Bandwidth do KDE para scores impostores."}
    )

    use_gridsearch: bool = field(
        default=True,
        metadata={"help": "Indica se o calibrador deve otimizar bandwidth via GridSearch."}
    )

    max_samples_for_gridsearch: int = field(
        default=10000,
        metadata={"help": "Máximo de amostras usadas no GridSearch do bandwidth."}
    )

    max_stored_samples: int = field(
        default=50000,
        metadata={"help": "Máximo de amostras armazenadas por classe para ajuste final do KDE."}
    )

    min_bandwidth_abs: float = field(
        default=1e-3,
        metadata={"help": "Bandwidth mínimo absoluto no espaço transformado."}
    )

    max_grid_expansions: int = field(
        default=2,
        metadata={"help": "Número máximo de expansões do grid se o melhor bandwidth cair na borda."}
    )

    min_log_lr: float = field(
        default=-20.0,
        metadata={"help": "Limite inferior do log-LR calibrado."}
    )

    max_log_lr: float = field(
        default=20.0,
        metadata={"help": "Limite superior do log-LR calibrado."}
    )

    interpolation_points: int = field(
        default=10000,
        metadata={"help": "Quantidade de pontos usados para construir a curva interpolada score -> LR."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente usada em amostragens internas quando aplicável."}
    )

    kernel: str = field(
        default="gaussian",
        metadata={"help": "Kernel do sklearn.neighbors.KernelDensity."}
    )

    # ── Cross-validation diagnóstico (opt-in) ────────────────────────────
    k_folds: int = field(
        default=0,
        metadata={"help": (
            "<html><b>🧪 K-fold CV diagnóstico (opt-in)</b> — pra estimar "
            "Cllr CV out-of-fold no <b>domínio de calibração</b>.<br/><br/>"
            "<code>0</code> (padrão) — desativado, comportamento atual idêntico.<br/>"
            "<code>5</code> — recomendado quando ligado. Roda 5 folds "
            "estratificados após o fit principal usando os bandwidths do "
            "production model (congelados). Anexa <code>calibrator_cllr_cv</code> "
            "ao metadata, propagado pra ValidationMetrics.<br/><br/>"
            "<b>Atenção</b>: o Cllr CV mede generalização DENTRO do domínio "
            "de calibração, NÃO performance em validação externa. Pra KDE, "
            "o gap CV − in-sample tende a ser maior (calibrador de alta "
            "capacidade overfita mais que LogReg)."
            "</html>"
        )}
    )

    cv_split_mode: CalibratorCVSplitMode = field(
        default=CalibratorCVSplitMode.STRATIFIED_PAIR,
        metadata={"help": (
            "<html><b>Estratégia de partição dos folds</b> em k-fold CV "
            "(só aplicável quando <code>k_folds &gt; 0</code>).<br/><br/>"
            "<b>STRATIFIED_PAIR</b> (padrão) — sklearn.StratifiedKFold. "
            "Preserva proporção gen/imp. Pares da mesma identidade podem "
            "aparecer em ambos os folds → otimista.<br/><br/>"
            "<b>STRATIFIED_GROUP</b> — sklearn.StratifiedGroupKFold por "
            "<code>probe_identity</code>. Nenhuma identidade de probe usada "
            "no treino do fold aparece no held-out. <b>Forense-defensável</b>, "
            "mais pessimista. Recomendado em laudos.</html>"
        )}
    )

    def __post_init__(self) -> None:
        if self.bandwidth_genuine <= 0:
            raise ValueError("bandwidth_genuine deve ser maior que zero.")

        if self.bandwidth_impostor <= 0:
            raise ValueError("bandwidth_impostor deve ser maior que zero.")

        if self.max_samples_for_gridsearch <= 0:
            raise ValueError("max_samples_for_gridsearch deve ser maior que zero.")

        if self.max_stored_samples <= 0:
            raise ValueError("max_stored_samples deve ser maior que zero.")

        if self.min_bandwidth_abs <= 0:
            raise ValueError("min_bandwidth_abs deve ser maior que zero.")

        if self.max_grid_expansions < 0:
            raise ValueError("max_grid_expansions não pode ser negativo.")

        if self.interpolation_points < 100:
            raise ValueError("interpolation_points deve ser pelo menos 100.")

        if self.min_log_lr >= self.max_log_lr:
            raise ValueError("min_log_lr deve ser menor que max_log_lr.")

        if not self.kernel:
            raise ValueError("kernel não pode ser vazio.")

        # k_folds = 0 desativa CV diagnóstico; >=2 ativa; 1 inválido.
        if self.k_folds < 0:
            raise ValueError("k_folds não pode ser negativo.")
        if self.k_folds == 1:
            raise ValueError("k_folds deve ser 0 (desativado) ou >= 2.")
        if not isinstance(self.cv_split_mode, CalibratorCVSplitMode):
            self.cv_split_mode = CalibratorCVSplitMode.from_str(self.cv_split_mode)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO de configuração a partir de dicionário.

        Descarta silenciosamente campos legados (ex: `fisher_eps` antes do
        switch pra z-score normalization) — assim configs antigas seguem
        carregando sem erro.
        """
        if not data:
            return cls()

        legacy_fields = {"fisher_eps"}
        cleaned = {k: v for k, v in data.items() if k not in legacy_fields}
        if "cv_split_mode" in cleaned:
            cleaned["cv_split_mode"] = CalibratorCVSplitMode.from_str(
                cleaned["cv_split_mode"]
            )
        return cls(**cleaned)