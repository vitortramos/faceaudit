# faceaudit/services/dataset_stats_service.py
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Dict, Set

# Models
from faceaudit.models.dataset_stats import DatasetStats
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.embedding_dataset import EmbeddingDataset

class DatasetStatsService:
    """
    Serviço centralizado para acesso a metadados e estatísticas de Datasets (Disco).
    
    Responsável por ler o cabeçalho dos arquivos (Parquet/Pickle) sem carregar
    as imagens ou vetores pesados, garantindo performance na listagem.
    Serviço 100% Stateless: não possui conhecimento sobre o ambiente ou diretórios padrão.
    """

    def __init__(self):
        # Cache em memória: { "/caminho/absoluto/do/dataset.parquet": ObjetoDatasetStats }
        self._cache: Dict[str, DatasetStats] = {}

    def get_stats(self, dataset_name: str, target_directory: Path) -> Optional[DatasetStats]:
        """
        Obtém estatísticas (DTO) do dataset buscando no diretório especificado.
        """
        if not dataset_name or dataset_name == "Selecione...":
            return None
            
        if not target_directory or not target_directory.exists():
            return None

        # 1. Busca Arquivo
        file_path = target_directory / f"{dataset_name}.parquet"
        is_parquet = True
        
        if not file_path.exists():
            file_path = target_directory / f"{dataset_name}.pkl"
            is_parquet = False
            
        if not file_path.exists():
            return None

        # 2. Cache Hit (Usando o caminho absoluto resolvido como chave)
        cache_key = str(file_path.resolve())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 3. Cache Miss - Executa leitura do disco
        try:
            stats = None
            if is_parquet:
                stats = self._read_parquet_stats(file_path)
            else:
                stats = self._read_pickle_stats(file_path)

            if stats:
                self._cache[cache_key] = stats
                return stats
            
            return None
            
        except Exception as e:
            print(f"❌ [DatasetStatsService] Erro ao ler {dataset_name} em {target_directory}: {e}")
            return None
    
    def compute_from_embedding_dataset(self, dataset: EmbeddingDataset) -> DatasetStats:
        """Computa DatasetStats a partir de um EmbeddingDataset já carregado."""
        if not dataset or not dataset.items:
            return DatasetStats()

        identity_set = {
            item.identity
            for item in dataset.items
            if item.identity is not None
        }

        embedding_dim = 0
        if dataset.items:
            emb = dataset.items[0].embedding
            if emb is not None and hasattr(emb, "shape"):
                embedding_dim = int(emb.shape[0])
            elif emb is not None:
                embedding_dim = len(emb)

        origin_counts: Dict[str, int] = {}
        samples_per_identity: Dict[str, int] = {}
        for item in dataset.items:
            origin = item.origin or item.source
            if origin:
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
            if item.identity is not None:
                key = str(item.identity)
                samples_per_identity[key] = samples_per_identity.get(key, 0) + 1

        return DatasetStats(
            n_imgs=len(dataset.items),
            n_ids=len(identity_set),
            embedding_dim=embedding_dim,
            identity_set=identity_set,
            origin_counts=origin_counts,
            samples_per_identity=samples_per_identity,
        )

    def compute_from_dataset(self, dataset: FaceDataset) -> DatasetStats:
        """Gera o DTO de estatísticas a partir de um objeto FaceDataset já carregado."""
        if not dataset or len(dataset) == 0:
            return DatasetStats()

        df = dataset.to_pandas()
        n_imgs = len(dataset)

        identities = set()
        if 'identity' in df.columns:
            series = df['identity'].dropna().astype(str)
            identities = set(series[series != ""].unique())
        n_ids = len(identities)

        hashes = set()
        if 'hash' in df.columns:
            hash_series = df['hash'].dropna().astype(str)
            hashes = set(hash_series[hash_series != ""].unique())

        emb_cols = {col for col in df.columns if 'embedding' in col or 'vector' in col}

        gender_counts = self._get_counts(df, 'gender')
        ethnicity_counts = self._get_counts(df, 'ethnicity')

        origin_counts = {}
        if 'origin' in df.columns:
            origin_counts = self._get_counts(df, 'origin')
        elif 'source' in df.columns:
            origin_counts = self._get_counts(df, 'source')

        samples_per_identity = self._get_counts(df, 'identity')

        return DatasetStats(
            n_imgs=n_imgs,
            n_ids=n_ids,
            identity_set=identities,
            hash_set=hashes,
            embedding_columns=emb_cols,
            gender_counts=gender_counts,
            ethnicity_counts=ethnicity_counts,
            origin_counts=origin_counts,
            samples_per_identity=samples_per_identity,
        )

    def _read_parquet_stats(self, file_path: Path) -> DatasetStats:
        """Leitura Otimizada de Parquet (Colunar)."""
        pq_file = pq.ParquetFile(file_path)
        all_cols = set(pq_file.schema.names)
        n_imgs = pq_file.metadata.num_rows
        
        emb_cols = {c for c in all_cols if 'embedding' in c or 'vector' in c}
        
        target_cols = ['identity', 'gender', 'ethnicity', 'origin', 'source', 'hash']
        cols_to_read = [c for c in target_cols if c in all_cols]
        
        df = pd.read_parquet(file_path, columns=cols_to_read)
        
        n_ids = 0
        identities = set()
        if 'identity' in df.columns:
            ids_series = df['identity'].dropna()
            ids_series = ids_series[ids_series != ""]
            identities = set(ids_series.unique())
            n_ids = len(identities)

        hashes = set()
        if 'hash' in df.columns:
            hash_series = df['hash'].dropna()
            hash_series = hash_series[hash_series != ""]
            hashes = set(hash_series.unique())

        return DatasetStats(
            n_imgs=n_imgs,
            n_ids=n_ids,
            identity_set=identities,
            hash_set=hashes,
            embedding_columns=emb_cols,
            gender_counts=self._get_counts(df, 'gender'),
            ethnicity_counts=self._get_counts(df, 'ethnicity'),
            origin_counts=self._get_counts(df, 'origin') if 'origin' in df.columns else self._get_counts(df, 'source'),
            samples_per_identity=self._get_counts(df, 'identity'),
        )

    def _read_pickle_stats(self, file_path: Path) -> DatasetStats:
        """Leitura Legada de Pickle."""
        df = pd.read_pickle(file_path)
        
        if not isinstance(df, pd.DataFrame):
            if hasattr(df, 'to_pandas'):
                df = df.to_pandas()
            else:
                return DatasetStats(n_imgs=0)

        emb_cols = {c for c in df.columns if 'embedding' in c}
        identities = set(df['identity'].dropna().unique()) if 'identity' in df.columns else set()
        
        hashes = set()
        if 'hash' in df.columns:
            hash_series = df['hash'].dropna()
            hashes = set(hash_series[hash_series != ""].unique())

        return DatasetStats(
            n_imgs=len(df),
            n_ids=len(identities),
            identity_set=identities,
            hash_set=hashes,
            embedding_columns=emb_cols,
            gender_counts=self._get_counts(df, 'gender'),
            ethnicity_counts=self._get_counts(df, 'ethnicity'),
            origin_counts=self._get_counts(df, 'origin') if 'origin' in df.columns else self._get_counts(df, 'source'),
            samples_per_identity=self._get_counts(df, 'identity'),
        )

    def _get_counts(self, df: pd.DataFrame, col_name: str) -> Dict[str, int]:
        if col_name not in df.columns:
            return {}
        
        counts = df[col_name].value_counts().to_dict()
        clean_counts = {}
        for k, v in counts.items():
            k_str = str(k).strip()
            if k_str and k_str.lower() not in ["nan", "none", "null", ""]:
                clean_counts[k_str] = v
                
        return clean_counts