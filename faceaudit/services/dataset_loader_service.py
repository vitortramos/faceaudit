# faceaudit/services/dataset_loader_service.py
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union

from faceaudit.models import EmbeddingDataset, EmbeddingInfo, FaceInfo
from faceaudit.models.face_dataset import FaceDataset

class DatasetLoaderService:
    """
    Serviço responsável por carregar Datasets do disco (Parquet ou Pickle).
    Implementa um cache em memória para evitar I/O repetitivo.
    """

    def __init__(self):
        # Cache simples: { "caminho_string": EmbeddingDataset }
        self._cache: Dict[str, EmbeddingDataset] = {}

    def load(self, path: Path, force_reload: bool = False) -> EmbeddingDataset:
        """
        Carrega um dataset do caminho especificado.
        
        Args:
            path (Path): Caminho do arquivo (.parquet ou .pkl).
            force_reload (bool): Se True, ignora o cache e relê do disco.
        
        Returns:
            EmbeddingDataset: O objeto carregado.
        """
        path_str = str(path.absolute())

        # 1. Verifica Cache
        if not force_reload and path_str in self._cache:
            return self._cache[path_str]

        if not path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {path}")

        # 2. Carrega do Disco
        dataset = self._read_file(path)
        
        # 3. Salva no Cache
        self._cache[path_str] = dataset
        return dataset

    def _read_file(self, path: Path) -> Union[FaceDataset, EmbeddingDataset]:
        """
        Lógica interna de leitura. Identifica se é um FaceDataset (Parquet)
        ou um EmbeddingDataset (Pickle).
        """
        suffix = path.suffix.lower()
        
        if suffix == ".parquet":
            # Usamos o método da classe FaceDataset que já faz o mapeamento de colunas
            return FaceDataset.from_parquet(path)

        elif suffix == ".pkl":
            with open(path, "rb") as f:
                data = pickle.load(f)
                return data
        
        raise ValueError(f"Formato não suportado: {suffix}")

    def clear_cache(self):
        self._cache.clear()