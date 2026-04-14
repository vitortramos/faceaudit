from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pickle
import numpy as np

@dataclass
class KDETrainingArtifacts:
    """
    Armazena os dados usados no fit do KDE para reprodução de plots.
    Mantido em Pickle devido ao volume de dados numéricos (np.ndarray).
    """

    genuines_scores_raw: np.ndarray
    impostors_scores_raw: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _ensure_1d_float(scores: np.ndarray) -> np.ndarray:
        arr = np.asarray(scores)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @classmethod
    def from_arrays(
        cls,
        genuines_scores_raw: np.ndarray,
        impostors_scores_raw: np.ndarray,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "KDETrainingArtifacts":
        return cls(
            genuines_scores_raw=cls._ensure_1d_float(genuines_scores_raw),
            impostors_scores_raw=cls._ensure_1d_float(impostors_scores_raw),
            metadata=dict(metadata or {}),
        )

    @property
    def n_genuines(self) -> int:
        return int(self.genuines_scores_raw.size)

    @property
    def n_impostors(self) -> int:
        return int(self.impostors_scores_raw.size)

    def to_pickle(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "genuines_scores_raw": self.genuines_scores_raw,
            "impostors_scores_raw": self.impostors_scores_raw,
            "metadata": self.metadata,
        }

        with open(p, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        return str(p)

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "KDETrainingArtifacts":
        """
        Carrega o artefato e garante a normalização das dimensões dos arrays.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"KDETrainingArtifacts não encontrado: {p}")

        with open(p, "rb") as f:
            payload = pickle.load(f)

        # A "correção" está aqui: usar from_arrays em vez de cls(...) direto
        # para garantir que o _ensure_1d_float seja executado no carregamento.
        return cls.from_arrays(
            genuines_scores_raw=payload["genuines_scores_raw"],
            impostors_scores_raw=payload["impostors_scores_raw"],
            metadata=payload.get("metadata", {}),
        )