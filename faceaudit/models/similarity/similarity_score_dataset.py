# faceaudit/models/similarity/similarity_score_dataset.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample


T = TypeVar("T", bound="SimilarityScoreDataset")


@dataclass
class SimilarityScoreDataset(JSONSerializableMixin):
    """
    Coleção de amostras de scores de similaridade.

    Esta classe é a ponte natural entre:
    - modelos de similaridade, que produzem scores brutos
    - modelos de calibração, que aprendem a partir de scores e labels
    - rotinas de auditoria, análise de erro e exportação

    Diferentemente do DTO antigo preso ao cosseno, aqui usamos `raw_score`
    de forma neutra para suportar:
    - cosine
    - PLDA
    - outros modelos futuros
    """

    name: str
    description: Optional[str] = None
    items: List[SimilarityScoreSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, sample: SimilarityScoreSample) -> None:
        self.items.append(sample)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> SimilarityScoreSample:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    @property
    def total_genuines(self) -> int:
        return sum(1 for item in self.items if item.same_person is True)

    @property
    def total_impostors(self) -> int:
        return sum(1 for item in self.items if item.same_person is False)

    @property
    def scores(self) -> np.ndarray:
        return np.asarray([float(item.raw_score) for item in self.items], dtype=np.float64)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([int(item.label) for item in self.items], dtype=np.int32)

    def to_pandas(self) -> pd.DataFrame:
        """
        Converte todas as amostras em um DataFrame achatado.
        """
        if not self.items:
            return pd.DataFrame()

        rows = [item.to_dict() for item in self.items]
        df = pd.DataFrame(rows)

        df.insert(0, "similarity_score_dataset_name", self.name)

        if self.description is not None:
            df.insert(1, "similarity_score_dataset_description", self.description)

        return df

    def to_parquet(self, path: Union[str, Path]) -> None:
        """
        Salva o dataset em disco para auditoria posterior.
        """
        df = self.to_pandas()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path_obj, index=False)

    def get_lowest_score_genuines(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares genuínos com os menores scores brutos.
        """
        df = self.to_pandas()
        if df.empty:
            return df
        genuines = df[df["same_person"] == True]
        return genuines.nsmallest(top_n, "raw_score")

    def get_highest_score_impostors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares impostores com os maiores scores brutos.
        """
        df = self.to_pandas()
        if df.empty:
            return df
        impostors = df[df["same_person"] == False]
        return impostors.nlargest(top_n, "raw_score")

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o dataset hidratando corretamente os itens.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "items" in processed_data and isinstance(processed_data["items"], list):
            processed_data["items"] = [
                SimilarityScoreSample.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["items"]
            ]

        return cls(**processed_data)