# faceaudit/services/dataset_stats_service.py
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from faceaudit.models.dataset_stats import ColumnStats, DatasetStats
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset


class DatasetStatsService:
    """
    Serviço centralizado para acesso a metadados e estatísticas de Datasets (Disco).

    Responsável por ler o cabeçalho dos arquivos (Parquet/Pickle) sem carregar
    as imagens ou vetores pesados, garantindo performance na listagem.
    Serviço 100% stateless: não possui conhecimento sobre o ambiente ou
    diretórios padrão.

    Cobertura por coluna (`column_coverage`) é computada nos 4 caminhos —
    leitura de parquet, pickle, FaceDataset em memória, EmbeddingDataset em
    memória — e alimenta tanto o ResourceInfoWidget quanto (em PR 3) a
    metadata do parquet, eliminando o probing colunar ad hoc do
    QualityMetricAvailabilityService.
    """

    # Limite de cardinalidade pra computar `distinct_count` em colunas string.
    # Acima desse valor, `distinct_count` fica como None (evita varredura cara
    # em colunas como `image_path` que tem N distintos por definição).
    _DISTINCT_CAP = 100_000

    def __init__(self):
        # Cache em memória: { "/caminho/absoluto/do/dataset.parquet": DatasetStats }
        self._cache: Dict[str, DatasetStats] = {}

    # ─────────────────────── API pública ───────────────────────

    def get_stats(self, dataset_name: str, target_directory: Path) -> Optional[DatasetStats]:
        """Obtém estatísticas (DTO) do dataset buscando no diretório especificado."""
        if not dataset_name or dataset_name == "Selecione...":
            return None

        if not target_directory or not target_directory.exists():
            return None

        # 1. Busca arquivo
        file_path = target_directory / f"{dataset_name}.parquet"
        is_parquet = True

        if not file_path.exists():
            file_path = target_directory / f"{dataset_name}.pkl"
            is_parquet = False

        if not file_path.exists():
            return None

        # 2. Cache hit (chave = caminho absoluto resolvido)
        cache_key = str(file_path.resolve())
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 3. Cache miss — leitura do disco
        try:
            stats = (
                self._read_parquet_stats(file_path)
                if is_parquet
                else self._read_pickle_stats(file_path)
            )
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

        items = dataset.items
        n_imgs = len(items)

        identity_set = {item.identity for item in items if item.identity is not None}

        embedding_dim = 0
        first_emb = items[0].embedding
        if first_emb is not None:
            embedding_dim = int(first_emb.shape[0]) if hasattr(first_emb, "shape") else len(first_emb)

        origin_counts: Dict[str, int] = {}
        samples_per_identity: Dict[str, int] = {}
        for item in items:
            origin = item.origin or item.source
            if origin:
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
            if item.identity is not None:
                key = str(item.identity)
                samples_per_identity[key] = samples_per_identity.get(key, 0) + 1

        # Cobertura por coluna — derivada da iteração (sem nova passada)
        column_coverage = self._coverage_from_embedding_items(items, embedding_dim)

        # PR 22 — categorical_counts genérico via reflection no EmbeddingInfo:
        # `origin` é a única coluna categórica de baixa cardinalidade tipicamente
        # presente. Estendido conforme demanda futura.
        categorical_counts: Dict[str, Dict[str, int]] = {}
        if origin_counts and len(origin_counts) <= _CATEGORICAL_MAX_CARDINALITY:
            categorical_counts["origin"] = dict(origin_counts)

        return DatasetStats(
            n_imgs=n_imgs,
            n_ids=len(identity_set),
            embedding_dim=embedding_dim,
            identity_set=identity_set,
            origin_counts=origin_counts,
            categorical_counts=categorical_counts,
            samples_per_identity=samples_per_identity,
            embedding_columns={"embedding"} if embedding_dim > 0 else set(),
            column_coverage=column_coverage,
            last_modified=_now_iso(),
        )

    def compute_from_dataset(self, dataset: FaceDataset) -> DatasetStats:
        """Gera o DTO de estatísticas a partir de um objeto FaceDataset já carregado."""
        if not dataset or len(dataset) == 0:
            return DatasetStats()

        df = dataset.to_pandas()
        return self._stats_from_dataframe(df, total_rows=len(dataset))

    # ─────────────────────── Leitura de disco ───────────────────────

    def _read_parquet_stats(self, file_path: Path) -> DatasetStats:
        """Leitura otimizada de parquet — column-aware via pyarrow."""
        pq_file = pq.ParquetFile(file_path)
        all_cols = list(pq_file.schema.names)
        n_imgs = pq_file.metadata.num_rows
        schema = pq_file.schema_arrow

        # Cobertura por coluna via pyarrow (vetorizada, sem carregar tudo)
        column_coverage = self._coverage_from_parquet(file_path, schema, n_imgs)

        # Colunas categóricas baratas — leitura colunar de strings
        target_cols = ["identity", "gender", "ethnicity", "origin", "source", "hash"]
        cols_to_read = [c for c in target_cols if c in all_cols]
        df = pd.read_parquet(file_path, columns=cols_to_read) if cols_to_read else pd.DataFrame()

        identities = _string_unique(df, "identity")
        hashes = _string_unique(df, "hash")
        emb_cols = {name for name, cs in column_coverage.items() if cs.dim is not None}

        return DatasetStats(
            n_imgs=n_imgs,
            n_ids=len(identities),
            embedding_dim=_first_dim(column_coverage),
            identity_set=identities,
            hash_set=hashes,
            embedding_columns=emb_cols,
            gender_counts=_get_counts(df, "gender"),
            ethnicity_counts=_get_counts(df, "ethnicity"),
            origin_counts=_get_counts(df, "origin") or _get_counts(df, "source"),
            categorical_counts=_get_categorical_counts(df),
            samples_per_identity=_get_counts(df, "identity"),
            column_coverage=column_coverage,
            last_modified=_iso_from_path(file_path),
        )

    def _read_pickle_stats(self, file_path: Path) -> DatasetStats:
        """Leitura legada de pickle (lê tudo)."""
        df = pd.read_pickle(file_path)

        if not isinstance(df, pd.DataFrame):
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            else:
                return DatasetStats(n_imgs=0)

        return self._stats_from_dataframe(df, total_rows=len(df), file_path=file_path)

    # ─────────────────────── Helpers internos ───────────────────────

    def _stats_from_dataframe(
        self,
        df: pd.DataFrame,
        total_rows: int,
        file_path: Optional[Path] = None,
    ) -> DatasetStats:
        """Caminho compartilhado por compute_from_dataset e _read_pickle_stats."""
        identities = _string_unique(df, "identity")
        hashes = _string_unique(df, "hash")

        column_coverage = self._coverage_from_dataframe(df, total_rows)
        emb_cols = {name for name, cs in column_coverage.items() if cs.dim is not None}

        return DatasetStats(
            n_imgs=total_rows,
            n_ids=len(identities),
            embedding_dim=_first_dim(column_coverage),
            identity_set=identities,
            hash_set=hashes,
            embedding_columns=emb_cols,
            gender_counts=_get_counts(df, "gender"),
            ethnicity_counts=_get_counts(df, "ethnicity"),
            origin_counts=_get_counts(df, "origin") or _get_counts(df, "source"),
            categorical_counts=_get_categorical_counts(df),
            samples_per_identity=_get_counts(df, "identity"),
            column_coverage=column_coverage,
            last_modified=_iso_from_path(file_path) if file_path else _now_iso(),
        )

    def _coverage_from_parquet(
        self,
        file_path: Path,
        schema: pa.Schema,
        n_rows: int,
    ) -> Dict[str, ColumnStats]:
        """Computa ColumnStats para cada coluna via pyarrow column-only reads.

        Estratégia de uma passada por tipo:
          - vetores (list/large_list): só `dim` (lê 1 linha)
          - numéricos: min/max/mean/std + non_null
          - strings: distinct_count (capped) + non_null
        """
        coverage: Dict[str, ColumnStats] = {}

        # Pass 1: list/large_list → só uma linha pra obter dim. Resto via aggregate.
        list_cols, scalar_cols = [], []
        for field in schema:
            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                list_cols.append(field.name)
            else:
                scalar_cols.append(field.name)

        # Pass list cols — leitura de 1 linha
        if list_cols:
            try:
                pq_file = pq.ParquetFile(file_path)
                first_batch = next(pq_file.iter_batches(batch_size=1, columns=list_cols), None)
                if first_batch is not None:
                    for name in list_cols:
                        try:
                            arr = first_batch.column(first_batch.schema.get_field_index(name))
                            sample = arr[0].as_py() if len(arr) else None
                            dim = len(sample) if sample is not None and hasattr(sample, "__len__") else None
                            # non_null_count via leitura colunar adicional
                            non_null = self._count_non_null_pyarrow(file_path, name, n_rows)
                            coverage[name] = ColumnStats(
                                name=name,
                                non_null_count=non_null,
                                total_count=n_rows,
                                dim=dim,
                            )
                        except Exception:
                            coverage[name] = ColumnStats(name=name, non_null_count=0, total_count=n_rows)
            except Exception:
                pass

        # Pass scalar cols
        if scalar_cols:
            try:
                table = pq.read_table(file_path, columns=scalar_cols)
                for name in scalar_cols:
                    try:
                        col = table.column(name)
                        coverage[name] = self._column_stats_from_arrow(name, col, n_rows)
                    except Exception:
                        coverage[name] = ColumnStats(name=name, non_null_count=0, total_count=n_rows)
            except Exception:
                pass

        return coverage

    def _coverage_from_dataframe(self, df: pd.DataFrame, total_rows: int) -> Dict[str, ColumnStats]:
        """Computa ColumnStats a partir de um pandas DataFrame em memória."""
        coverage: Dict[str, ColumnStats] = {}
        for name in df.columns:
            try:
                series = df[name]
                non_null = int(series.notna().sum())

                # Detecta vetor (list/ndarray na primeira célula não-nula)
                first_value = next((v for v in series.dropna().head(1)), None)
                if isinstance(first_value, (list, tuple, np.ndarray)):
                    dim = len(first_value) if hasattr(first_value, "__len__") else None
                    coverage[name] = ColumnStats(
                        name=name, non_null_count=non_null, total_count=total_rows, dim=dim
                    )
                    continue

                if pd.api.types.is_numeric_dtype(series):
                    arr = series.dropna().to_numpy(dtype=float, na_value=np.nan)
                    if arr.size > 0:
                        coverage[name] = ColumnStats(
                            name=name,
                            non_null_count=non_null,
                            total_count=total_rows,
                            min_value=float(np.min(arr)),
                            max_value=float(np.max(arr)),
                            mean_value=float(np.mean(arr)),
                            std_value=float(np.std(arr)),
                        )
                    else:
                        coverage[name] = ColumnStats(
                            name=name, non_null_count=0, total_count=total_rows
                        )
                    continue

                # Strings/categóricas
                distinct = None
                try:
                    nunique = int(series.dropna().nunique())
                    if nunique <= self._DISTINCT_CAP:
                        distinct = nunique
                except Exception:
                    pass
                coverage[name] = ColumnStats(
                    name=name,
                    non_null_count=non_null,
                    total_count=total_rows,
                    distinct_count=distinct,
                )
            except Exception:
                coverage[name] = ColumnStats(name=name, non_null_count=0, total_count=total_rows)
        return coverage

    def _coverage_from_embedding_items(self, items, embedding_dim: int) -> Dict[str, ColumnStats]:
        """Computa coverage para campos relevantes de EmbeddingInfo iterando 1x.

        Cobre os campos numéricos (yaw, pitch, embedding_norm), embedding (dim),
        identity (distinct), file_name (non-null). Não cobre 100% do schema —
        campos pouco usados ficam de fora pra não pagar O(N) extra.
        """
        n = len(items)
        if n == 0:
            return {}

        non_null = {"identity": 0, "file_name": 0, "image_path": 0, "yaw": 0, "pitch": 0, "embedding": 0}
        identity_distinct = set()
        yaws, pitches = [], []

        for item in items:
            if item.identity is not None:
                non_null["identity"] += 1
                identity_distinct.add(str(item.identity))
            if getattr(item, "file_name", None):
                non_null["file_name"] += 1
            if getattr(item, "image_path", None):
                non_null["image_path"] += 1
            if getattr(item, "yaw", None) is not None:
                non_null["yaw"] += 1
                yaws.append(float(item.yaw))
            if getattr(item, "pitch", None) is not None:
                non_null["pitch"] += 1
                pitches.append(float(item.pitch))
            if item.embedding is not None:
                non_null["embedding"] += 1

        coverage: Dict[str, ColumnStats] = {}
        coverage["identity"] = ColumnStats(
            name="identity", non_null_count=non_null["identity"], total_count=n,
            distinct_count=len(identity_distinct) if len(identity_distinct) <= self._DISTINCT_CAP else None,
        )
        coverage["file_name"] = ColumnStats(
            name="file_name", non_null_count=non_null["file_name"], total_count=n,
        )
        coverage["image_path"] = ColumnStats(
            name="image_path", non_null_count=non_null["image_path"], total_count=n,
        )
        if non_null["yaw"] > 0:
            coverage["yaw"] = ColumnStats(
                name="yaw", non_null_count=non_null["yaw"], total_count=n,
                min_value=min(yaws), max_value=max(yaws),
                mean_value=sum(yaws) / len(yaws),
                std_value=float(np.std(yaws)) if yaws else None,
            )
        if non_null["pitch"] > 0:
            coverage["pitch"] = ColumnStats(
                name="pitch", non_null_count=non_null["pitch"], total_count=n,
                min_value=min(pitches), max_value=max(pitches),
                mean_value=sum(pitches) / len(pitches),
                std_value=float(np.std(pitches)) if pitches else None,
            )
        coverage["embedding"] = ColumnStats(
            name="embedding", non_null_count=non_null["embedding"], total_count=n,
            dim=embedding_dim if embedding_dim > 0 else None,
        )
        return coverage

    def _column_stats_from_arrow(self, name: str, col: pa.ChunkedArray, n_rows: int) -> ColumnStats:
        """Stats vetorizadas pra uma coluna scalar pyarrow."""
        non_null = n_rows - int(col.null_count)

        col_type = col.type
        if pa.types.is_integer(col_type) or pa.types.is_floating(col_type):
            try:
                # min_max retorna struct {min, max}; conversão para Python float
                mm = pc.min_max(col)
                mn = mm["min"].as_py()
                mx = mm["max"].as_py()
                mean = pc.mean(col).as_py()
                std = pc.stddev(col).as_py()
                return ColumnStats(
                    name=name, non_null_count=non_null, total_count=n_rows,
                    min_value=_safe_float(mn),
                    max_value=_safe_float(mx),
                    mean_value=_safe_float(mean),
                    std_value=_safe_float(std),
                )
            except Exception:
                return ColumnStats(name=name, non_null_count=non_null, total_count=n_rows)

        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            distinct = None
            try:
                nunique = int(pc.count_distinct(col).as_py())
                if nunique <= self._DISTINCT_CAP:
                    distinct = nunique
            except Exception:
                pass
            return ColumnStats(
                name=name, non_null_count=non_null, total_count=n_rows,
                distinct_count=distinct,
            )

        if pa.types.is_boolean(col_type):
            return ColumnStats(
                name=name, non_null_count=non_null, total_count=n_rows,
                distinct_count=2,
            )

        return ColumnStats(name=name, non_null_count=non_null, total_count=n_rows)

    def _count_non_null_pyarrow(self, file_path: Path, col_name: str, n_rows: int) -> int:
        """Conta não-nulos numa coluna lendo só ela. Usado pra colunas list/vector."""
        try:
            table = pq.read_table(file_path, columns=[col_name])
            return n_rows - int(table.column(col_name).null_count)
        except Exception:
            return 0


# ─────────────────────── helpers livres do módulo ───────────────────────

def _get_counts(df: pd.DataFrame, col_name: str) -> Dict[str, int]:
    if col_name not in df.columns:
        return {}
    counts = df[col_name].value_counts().to_dict()
    clean: Dict[str, int] = {}
    for k, v in counts.items():
        k_str = str(k).strip()
        if k_str and k_str.lower() not in ("nan", "none", "null", ""):
            clean[k_str] = int(v)
    return clean


# PR 22 (Tema 4.5) — Cardinalidade máxima pra considerar uma coluna
# string como "categórica" no histograma genérico. Acima disso, vira
# free-text e a contagem ficaria gigante (ex: identity, file_name).
_CATEGORICAL_MAX_CARDINALITY = 50

# Colunas que nunca contam como categóricas, independente da cardinalidade.
# Identity/path/hash/file_name são identificadoras — histograma delas não
# tem semântica útil. Embeddings são vetores — também não fazem sentido
# em histograma categórico.
def _excluded_from_categorical() -> frozenset:
    """Build excluded set em runtime pra incluir TODOS os parquet_columns
    do `SepaelModel` enum (v2..v6+) — sem hard-coding."""
    from faceaudit.enums.sepael_model import SepaelModel
    return frozenset(
        {"identity", "image_path", "hash", "file_name",
         "embedding", "embedding_serfiq"}
        | {m.parquet_column for m in SepaelModel}
    )


_EXCLUDED_FROM_CATEGORICAL = _excluded_from_categorical()


def _get_categorical_counts(
    df: pd.DataFrame,
    column_specs: Optional[Dict[str, "object"]] = None,
    max_cardinality: int = _CATEGORICAL_MAX_CARDINALITY,
) -> Dict[str, Dict[str, int]]:
    """Computa histogramas para todas as colunas categóricas do df.

    Critério de inclusão:
      - dtype object/string + nunique ∈ [1, max_cardinality]
    Critério de exclusão:
      - Coluna em `_EXCLUDED_FROM_CATEGORICAL`.
      - Quando `column_specs` presente, role IDENTITY/IMAGE_PATH/FILE_HASH/
        EMBEDDING são excluídas independente do dtype/cardinalidade.
    """
    out: Dict[str, Dict[str, int]] = {}
    for col in df.columns:
        if col in _EXCLUDED_FROM_CATEGORICAL:
            continue
        if column_specs:
            spec = column_specs.get(col)
            if spec is not None:
                role = getattr(getattr(spec, "role", None), "value", None)
                if role in ("identity", "image_path", "file_hash", "embedding"):
                    continue
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        try:
            nunique = int(series.dropna().nunique())
        except Exception:
            continue
        if nunique == 0 or nunique > max_cardinality:
            continue
        counts = _get_counts(df, col)
        if counts:
            out[col] = counts
    return out


def _string_unique(df: pd.DataFrame, col_name: str) -> set:
    if col_name not in df.columns:
        return set()
    series = df[col_name].dropna().astype(str)
    return set(series[series != ""].unique())


def _first_dim(coverage: Dict[str, ColumnStats]) -> int:
    """Retorna o dim do primeiro embedding encontrado (compat com embedding_dim legado)."""
    for cs in coverage.values():
        if cs.dim is not None and cs.dim > 0:
            return cs.dim
    return 0


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_from_path(path: Optional[Path]) -> Optional[str]:
    if path is None or not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
    except Exception:
        return None
