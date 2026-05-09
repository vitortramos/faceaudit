"""Radial Gaussianization (RG) — transformação não-linear que mapeia
a marginal radial dos embeddings para a chi-density de uma gaussiana
esférica unitária.

Baseado em Lyu & Simoncelli (2009), "Nonlinear Extraction of
Independent Components of Natural Images Using Radial Gaussianization",
Neural Computation 21(6):1485–1519.

Algoritmo (paper §3.3 + §4.2):
  1. Assume entrada whitened (na prática, deve vir após
     WhiteningTransformation no pipeline).
  2. Fit: armazena `r_sorted = sort(‖x_i‖)` — CDF empírica radial.
  3. Transform por face:
       r = ‖x‖
       u = F̂_r(r) = searchsorted(r_sorted, r) / N      ∈ [0, 1]
       u' = clip(u, eps, 1-eps)                          (evita inf)
       r' = F_χ⁻¹(u'; df=d)                              (chi.ppf)
       x' = x · r' / r                                   (preserva direção)

Stateless: depende apenas dos r_sorted + dim. Suporta DUAL mode via
get_state/set_state.

⚠ Paper §3.4: RG só REMOVE dependências em fontes elipticamente
simétricas (ESD). Em fontes próximas de factoriais, pode AUMENTAR
a dependência. Recomenda-se medir mutual information antes/depois.
"""

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import chi

from faceaudit.enums.transformation_type import TransformationType
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.models.radial_gaussianization_config import (
    RadialGaussianizationConfig,
)
from faceaudit.protocols.embedding_transformation_protocol import (
    EmbeddingTransformationProtocol,
)


class RadialGaussianizationTransformation(EmbeddingTransformationProtocol):
    """Aplica Radial Gaussianization a um EmbeddingDataset."""

    def __init__(
        self,
        math_service=None,  # uniforme com WhiteningTransformation; ignorado aqui
        name: str = "RadialGaussianizationTransformation",
        config: Optional[RadialGaussianizationConfig] = None,
    ):
        self._name = name
        # `math_service` recebido pra compatibilidade com o factory do
        # pipeline; RG não delega operações matemáticas a ele (usa
        # numpy + scipy.stats.chi direto).
        self.config = config if config is not None else RadialGaussianizationConfig()

        # State aprendido no fit
        self._r_sorted: Optional[np.ndarray] = None
        self._dim: Optional[int] = None
        self._n_train: int = 0
        self._fitted: bool = False
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.RADIAL_GAUSSIANIZATION

    @property
    def requires_target_dataset(self) -> bool:
        return False

    # ── FIT ───────────────────────────────────────────────────────────────

    def fit(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset] = None,
    ) -> None:
        """Estima a CDF empírica radial a partir das normas dos
        vetores de treino. Não usa target_dataset (RG é unimodal)."""
        print(f"[{self.name}] Treinando Radial Gaussianization…")

        X = source_dataset.matrix  # (N, d)
        if X is None or X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(
                f"[{self.name}] source_dataset vazio ou shape inválido."
            )

        n, d = X.shape

        # Aviso defensivo: dados não whitened? RG funciona melhor após
        # whitening (paper §3.3). Verifica via diagonal da covariância.
        if self.config.assume_whitened:
            self._warn_if_not_whitened(X)

        # CDF empírica radial — apenas ordena as normas
        r = np.linalg.norm(X, axis=1)
        self._r_sorted = np.sort(r).astype(np.float64)
        self._dim = int(d)
        self._n_train = int(n)

        # Métricas pra debug / UI
        self._metrics = {
            "step_type": self.transformation_type.value,
            "n_train": self._n_train,
            "dim": self._dim,
            "r_min": float(self._r_sorted[0]),
            "r_max": float(self._r_sorted[-1]),
            "r_median": float(np.median(self._r_sorted)),
            "r_mean": float(np.mean(self._r_sorted)),
            "expected_chi_mean": float(chi.mean(df=self._dim)),
        }
        print(
            f"[{self.name}] N={self._n_train}, d={self._dim}, "
            f"r̄={self._metrics['r_mean']:.3f}, "
            f"E[χ_d]={self._metrics['expected_chi_mean']:.3f}"
        )

        self._fitted = True

    def _warn_if_not_whitened(self, X: np.ndarray) -> None:
        """Verifica se a covariância está próxima de I. Se a média
        da diagonal desvia mais que `whitening_check_tolerance` de 1.0,
        emite warning. Não impede execução."""
        try:
            cov = np.cov(X, rowvar=False)
            mean_diag = float(np.mean(np.diag(cov)))
            tol = float(self.config.whitening_check_tolerance)
            if abs(mean_diag - 1.0) > tol:
                warnings.warn(
                    f"[{self.name}] Source não parece whitened "
                    f"(mean diag covariância = {mean_diag:.3f}, "
                    f"esperado ≈ 1.0 ± {tol}). "
                    "RG funciona melhor com Whitening prévio no pipeline. "
                    "Defina assume_whitened=False pra suprimir.",
                    stacklevel=3,
                )
        except Exception:
            pass  # falha no check não é crítica

    # ── TRANSFORM ─────────────────────────────────────────────────────────

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """Aplica RG vetorizado em todo o dataset."""
        if not self._fitted:
            raise RuntimeError(
                f"[{self.name}] Não treinado. Chame `fit()` primeiro."
            )

        X = dataset.matrix  # (N, d)
        r_old = np.linalg.norm(X, axis=1)

        # Mapeamento radial: r → r'
        r_new = self._map_radius(r_old)

        # Reconstrução: x' = x · r'/r. r=0 produz vetor zero (defensivo).
        scale = np.where(r_old > 0, r_new / np.maximum(r_old, 1e-12), 0.0)
        X_new = X * scale[:, None]

        return self._build_output_dataset(dataset, X_new)

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """Aplica RG numa face única — wrapper que reusa o vetorizado."""
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Não treinado.")
        temp = EmbeddingDataset("temp", [embedding_info])
        return self.transform(temp).items[0]

    def _map_radius(self, r: np.ndarray) -> np.ndarray:
        """Mapeamento r → r' via composição F_χ⁻¹ ∘ F̂_r.

        F̂_r: CDF empírica baseada em searchsorted (paper §4.2).
        F_χ⁻¹: ICDF da distribuição chi com `_dim` graus de liberdade,
        via scipy.stats.chi.ppf (forma fechada — não precisa estimar).
        """
        # Quantis empíricos: u_i = rank(r_i) / N
        # Usa side='right' pra que o maior valor mapeie pra ~1.0
        # (não exatamente 1.0; eps_clip cuida disso).
        n = len(self._r_sorted)
        u = np.searchsorted(self._r_sorted, r, side="right") / n

        # Clip pra evitar inf (chi.ppf(1.0) = ∞)
        eps = float(self.config.eps_clip)
        u = np.clip(u, eps, 1.0 - eps)

        # ICDF da chi com d graus de liberdade
        return chi.ppf(u, df=self._dim).astype(np.float64)

    @staticmethod
    def _build_output_dataset(
        src: EmbeddingDataset, X_new: np.ndarray,
    ) -> EmbeddingDataset:
        """Constrói EmbeddingDataset preservando metadata dos items."""
        new_items = []
        for item, vec in zip(src.items, X_new):
            new_items.append(EmbeddingInfo(
                embedding=vec.astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores,
            ))
        return EmbeddingDataset(
            name=f"{src.name}_RG",
            description=f"Radial Gaussianization de {src.name}",
            items=new_items,
        )

    # ── METRICS / PERSIST / STATE ─────────────────────────────────────────

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save(self, path: str) -> None:
        if not self._fitted:
            return
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "r_sorted": self._r_sorted,
            "dim": self._dim,
            "n_train": self._n_train,
            "config_dict": self.config.to_dict(),
            "metrics": self._metrics,
        }
        with open(file_path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(
                f"[{self.name}] Arquivo de estado não encontrado: {file_path}"
            )
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        self._r_sorted = state["r_sorted"]
        self._dim = state["dim"]
        self._n_train = state["n_train"]
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = RadialGaussianizationConfig.from_dict(config_dict)
        self._metrics = state.get("metrics", {})
        self._fitted = True

    def get_state(self) -> Dict[str, Any]:
        return {
            "_r_sorted": self._r_sorted,
            "_dim": self._dim,
            "_n_train": self._n_train,
            "config_dict": self.config.to_dict(),
            "_metrics": self._metrics,
            "_fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._r_sorted = state.get("_r_sorted")
        self._dim = state.get("_dim")
        self._n_train = state.get("_n_train", 0)
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = RadialGaussianizationConfig.from_dict(config_dict)
        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))
