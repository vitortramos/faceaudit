import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.wccn_config import WCCNConfig
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType


class WCCNTransformation(EmbeddingTransformationProtocol):
    """
    Within-Class Covariance Normalization (Hatch, Kajarekar & Stolcke 2006).

    Calcula a matriz W intraclasse a partir de identidades rotuladas, regulariza
    via ε·I + shrinkage opcional e produz B = chol(inv(W_reg)). A transformação
    aplica x' = x @ B, reduzindo a variabilidade intra-identidade.
    """

    def __init__(
        self,
        math_service: EmbeddingMathService,
        name: str = "WCCNTransformation",
        config: Optional[WCCNConfig] = None,
    ):
        self._name = name
        self.math_service = math_service
        self.config = config if config is not None else WCCNConfig()

        self._wccn_matrix: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.WCCN_TRANSFORMATION

    @property
    def requires_target_dataset(self) -> bool:
        return False

    def fit(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset] = None,
    ) -> None:
        print(
            f"[{self.name}] Treinando WCCN "
            f"(estimador={self.config.estimator.label}, "
            f"grid_search={self.config.use_grid_search})..."
        )

        if not source_dataset.items:
            raise ValueError(f"[{self.name}] source_dataset vazio.")

        self._wccn_matrix, self._metrics = self.math_service.calculate_wccn_parameters(
            dataset=source_dataset,
            config=self.config,
        )
        self._fitted = True

        print(
            f"[{self.name}] Concluído: "
            f"{self._metrics.get('n_classes_used', '?')} ids, "
            f"ε={self._metrics.get('final_regularization', float('nan')):.1e}, "
            f"cond={self._metrics.get('condition_number', float('nan')):.2e}"
        )

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Não treinado. Chame fit() primeiro.")
        return self.math_service.apply_wccn_transformation(dataset, self._wccn_matrix)

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Não treinado.")
        temp = EmbeddingDataset(name="temp", items=[embedding_info])
        return self.transform(temp).items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save(self, path: str) -> None:
        if not self._fitted:
            return
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "wccn_matrix": self._wccn_matrix,
            "config_dict": self.config.to_dict(),
            "metrics": self._metrics,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"[{self.name}] Arquivo não encontrado: {file_path}")
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        self._wccn_matrix = state["wccn_matrix"]
        self._metrics = state.get("metrics", {})
        config_dict = state.get("config_dict")
        self.config = WCCNConfig.from_dict(config_dict) if config_dict else WCCNConfig()
        self._fitted = True

    def get_state(self) -> Dict[str, Any]:
        return {
            "_wccn_matrix": self._wccn_matrix,
            "config_dict": self.config.to_dict(),
            "_metrics": self._metrics,
            "_fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._wccn_matrix = state.get("_wccn_matrix")
        config_dict = state.get("config_dict")
        self.config = WCCNConfig.from_dict(config_dict) if config_dict else WCCNConfig()
        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))
