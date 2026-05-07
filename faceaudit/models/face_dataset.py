from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, TYPE_CHECKING, Set
from pathlib import Path
import pandas as pd
import numpy as np
import random
import copy

# Imports relativos
from .face_info import FaceInfo
from .ff_info import FFInfo
from .ofiq_info import OFIQInfo
from .serfiq_info import SERFIQInfo
from .embedding_dataset import EmbeddingDataset
from .embedding_info import EmbeddingInfo
from .filter_config import FilterConfig
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.quality_score_type import QualityScoreType


# PR 21 (Tema 4.2) — predicado pra identificar colunas canônicas que
# `FaceInfo.to_flat_dict` emite. Tudo fora dessa lista é capturado como
# `face.extra_data` em `from_dataframe` e re-emitido em `to_flat_dict` —
# round-trip pra preservar colunas custom em save/load/filter.
_CANONICAL_FACE_COLUMNS = frozenset({
    # FaceInfo metadata
    "image_path", "file_name", "hash", "origin", "source",
    "ethnicity", "identity", "gender",
    # FFInfo (exclude_heavy=True) + renomeados em to_flat_dict
    "ipd", "fiqa_score", "det_score", "gender_est", "age_est",
    "yaw", "pitch", "roll",
    # FFInfo embeddings (preservados como colunas planas)
    "embedding", "embedding_v2", "embedding_v4", "embedding_serfiq",
})


def _is_canonical_face_column(name: str) -> bool:
    """True quando `name` pertence ao mapeamento canônico de FaceInfo."""
    if name in _CANONICAL_FACE_COLUMNS:
        return True
    # OFIQ/SERFIQ emitem múltiplas colunas com prefixo — todas canônicas
    return name.startswith("ofiq_") or name.startswith("serfiq_")


@dataclass
class FaceDataset:
    """
    Representa um conjunto de dados de faces (Dataset) em memória.
    
    Esta classe atua como o container principal de dados do sistema, centralizando
    a lógica de manipulação de listas de `FaceInfo`. Ela abstrai operações complexas
    como filtragem, merge, estatísticas e persistência em disco (Parquet).

    Attributes:
        name (str): Nome identificador do dataset (ex: 'BFW_filtered').
        description (Optional[str]): Metadados descritivos sobre a origem ou filtros aplicados.
        hash (Optional[str]): Checksum para controle de versão e integridade.
        faces (List[FaceInfo]): Lista contendo os objetos de dados faciais.
    """
    name: str
    description: Optional[str] = None
    hash: Optional[str] = None
    faces: List[FaceInfo] = field(default_factory=list)

    # Schema-by-role: dict de nome → ColumnSpec persistido na metadata do
    # parquet em `to_parquet`. Quando vazio (default), `to_parquet` constrói
    # specs a partir do `column_catalog` filtrados pelas colunas presentes.
    # Quando o parquet é lido, `from_parquet` popula este campo com os specs
    # da metadata — esse passa a ser fonte de verdade para a UI tipo planilha.
    column_specs: Dict[str, Any] = field(default_factory=dict)

    def add(self, face: FaceInfo) -> None:
        """
        Adiciona um registro de face ao dataset.

        Args:
            face (FaceInfo): O objeto contendo dados da imagem, identidades e métricas.
        """
        self.faces.append(face)

    def __len__(self) -> int:
        """Retorna a quantidade total de faces no dataset."""
        return len(self.faces)
    
    def __getitem__(self, index) -> FaceInfo:
        """Permite acesso indexado às faces (ex: dataset[0])."""
        return self.faces[index]

    def __iter__(self):
        """Permite iteração direta sobre as faces (ex: for face in dataset)."""
        return iter(self.faces)

    @property
    def number_of_identities(self) -> int:
        """
        Calcula o número de identidades únicas presentes no dataset.
        
        Returns:
            int: Contagem de indivíduos únicos.
        """
        return len(self.unique_identities)

    @property
    def identities(self) -> List[Optional[str]]:
        """
        Retorna uma lista com os IDs de todas as faces, preservando a ordem.
        Pode conter duplicatas e valores None.
        """
        return [face.identity for face in self.faces]

    @property
    def unique_identities(self) -> List[str]:
        """
        Retorna uma lista ordenada contendo apenas as identidades únicas e válidas (não nulas).
        """
        u_ids = {face.identity for face in self.faces if face.identity is not None}
        return sorted(list(u_ids))

    # =========================================================================
    # MÉTODOS DE FILTRAGEM (BACKEND)
    # =========================================================================

    def apply_filter(self, config: FilterConfig, callback: Optional[Callable[[int], None]] = None) -> 'FaceDataset':
        """
        Gera um novo FaceDataset contendo apenas as faces que satisfazem os critérios de filtro.

        Args:
            config (FilterConfig): Objeto contendo os ranges e critérios de aceitação.
            callback (Optional[Callable]): Função para reportar progresso (0-100).

        Returns:
            FaceDataset: Uma nova instância (deep copy) com os dados filtrados.
        """
        filtered_faces = []
        total = len(self.faces)

        for i, face in enumerate(self.faces):
            if self._face_matches_config(face, config):
                filtered_faces.append(copy.deepcopy(face))
            
            if callback and total > 0 and i % 100 == 0: 
                callback(int((i / total) * 100))

        if callback: callback(100)
        
        new_name = f"{self.name}_filtered"
        return FaceDataset(name=new_name, faces=filtered_faces, description=f"Filtro baseado em: {config.name}")
    

    def _face_matches_config(self, face: FaceInfo, config: FilterConfig) -> bool:
        """
        Valida se uma única face atende a todos os critérios definidos no FilterConfig.
        Verifica faixas numéricas (qualidade, pose) e correspondências categóricas (gênero, etnia).
        """
        # 1. MAPEAMENTO NUMÉRICO (Valor Atual, Alvo Exato, Min, Max, Usar Modulo Absoluto)
        numeric_map = [
            (face.ff_info.fiqa_score if face.ff_info else None, config.fiqa_score, config.min_fiqa_score, config.max_fiqa_score, config.abs_fiqa_score),
            (face.ofiq_info.unified_quality_score_scalar if face.ofiq_info else None, config.ofiq_score, config.min_ofiq_score, config.max_ofiq_score, config.abs_ofiq_score),
            (face.serfiq_info.score if face.serfiq_info else None, None, config.min_serfiq_score, config.max_serfiq_score, config.abs_serfiq_score),
            
            # Poses OFIQ
            (face.ofiq_info.head_pose_yaw_raw if face.ofiq_info else None, config.ofiq_yaw, config.min_ofiq_yaw, config.max_ofiq_yaw, config.abs_ofiq_yaw),
            (face.ofiq_info.head_pose_pitch_raw if face.ofiq_info else None, config.ofiq_pitch, config.min_ofiq_pitch, config.max_ofiq_pitch, config.abs_ofiq_pitch),
            (face.ofiq_info.head_pose_roll_raw if face.ofiq_info else None, config.ofiq_roll, config.min_ofiq_roll, config.max_ofiq_roll, config.abs_ofiq_roll),
            
            # Poses ForensicFace
            (face.ff_info.yaw if face.ff_info else None, config.ff_yaw, config.min_ff_yaw, config.max_ff_yaw, config.abs_ff_yaw),
            (face.ff_info.pitch if face.ff_info else None, config.ff_pitch, config.min_ff_pitch, config.max_ff_pitch, config.abs_ff_pitch),
            (face.ff_info.roll if face.ff_info else None, config.ff_roll, config.min_ff_roll, config.max_ff_roll, config.abs_ff_roll),
            
            # Outros — `age`/`ipd` (sem prefixo) não são fields dataclass
            # do FilterConfig, só são criados quando a UI faz setattr. Lemos
            # via getattr pra suportar uso programático (testes e código
            # que não passa pela UI).
            (face.ff_info.age if face.ff_info else None, getattr(config, "age", None), config.min_age, config.max_age, config.abs_age),
            (face.ff_info.ipd if face.ff_info else None, getattr(config, "ipd", None), config.min_ipd, config.max_ipd, config.abs_ipd),
        ]

        for val, target, c_min, c_max, use_abs in numeric_map:
            # Se o filtro não foi configurado para este atributo, ignora
            if target is None and c_min is None and c_max is None:
                continue
            
            # Se o filtro existe, mas o dado na face é nulo, reprova
            if val is None:
                return False

            compare_val = abs(float(val)) if use_abs else float(val)

            # Verificação de Alvo Exato (com tolerância float)
            if target is not None:
                if not (target - 0.001 <= compare_val <= target + 0.001):
                    return False
            
            # Verificação de Range
            if c_min is not None and compare_val < c_min:
                return False
            if c_max is not None and compare_val > c_max:
                return False

        # 2. MAPEAMENTO CATEGÓRICO
        categorical_map = [
            (face.gender, config.gender, config.gender_list),
            (face.ethnicity, config.ethnicity, config.ethnicity_list),
            (face.identity, config.identity, config.identity_list),
            (face.origin, config.source, config.source_list),
        ]

        for val, search_str, inclusion_list in categorical_map:
            if not search_str and not inclusion_list:
                continue

            val_str = str(val).strip() if val is not None else ""

            # Verificação de Lista (Multi-seleção)
            if inclusion_list:
                if val_str not in inclusion_list:
                    return False

            # Verificação de String Única (Combobox/Texto)
            elif search_str:
                if search_str.lower() != val_str.lower():
                    return False

        # 3. FILTROS GENÉRICOS POR COLUNA CUSTOM (PR 21 — Tema 4.2)
        # Lê valores de `face.extra_data` (custom columns) e compara com
        # `config.column_filters`. Mesmas semânticas dos canônicos:
        #   value → match exato (case-insensitive p/ string, ±0.001 p/ num)
        #   list  → multi-select inclusion
        #   min/max + abs → range numérico (com módulo opcional)
        if config.column_filters:
            for col_name, spec in config.column_filters.items():
                if not spec:
                    continue
                val = face.extra_data.get(col_name) if face.extra_data else None
                target_value = spec.get("value")
                target_list = spec.get("list") or []
                c_min = spec.get("min")
                c_max = spec.get("max")
                use_abs = bool(spec.get("abs"))

                # Categórico: value/list
                if target_list:
                    val_str = str(val).strip() if val is not None else ""
                    if val_str not in target_list:
                        return False
                elif target_value not in (None, ""):
                    val_str = str(val).strip().lower() if val is not None else ""
                    if str(target_value).strip().lower() != val_str:
                        return False

                # Numérico: min/max
                if c_min is not None or c_max is not None:
                    if val is None:
                        return False
                    try:
                        num = abs(float(val)) if use_abs else float(val)
                    except (TypeError, ValueError):
                        return False
                    if c_min is not None and num < c_min:
                        return False
                    if c_max is not None and num > c_max:
                        return False

        return True

    # =========================================================================
    # MÉTODOS DE CONCATENAÇÃO E ESTATÍSTICAS
    # =========================================================================

    def concat(self, other: 'FaceDataset', new_name: Optional[str] = None) -> 'FaceDataset':
        """
        Une dois datasets em um novo.
        """
        if not isinstance(other, FaceDataset):
            raise TypeError(f"Não é possível concatenar FaceDataset com {type(other)}")
        new_faces = self.faces + other.faces
        final_name = new_name if new_name else f"{self.name}_{other.name}_Merged"
        return FaceDataset(name=final_name, faces=new_faces)

    def __add__(self, other: 'FaceDataset') -> 'FaceDataset':
        """Sobrecarga do operador + para concatenação."""
        return self.concat(other)

    def get_all_statistics(self) -> Dict[str, pd.Series]:
        """
        Gera estatísticas de contagem para atributos categóricos (Gênero, Etnia, Identidade, Origem).
        Recalcula o DataFrame interno para garantir dados atualizados.
        """
        df = self.to_pandas()
        stats = {}
        
        def get_counts(col_name):
            if col_name in df.columns:
                series = df[col_name].fillna("N/A")
                series = series[series != ""]
                return series.value_counts()
            return pd.Series()

        stats['identity'] = get_counts('identity')
        stats['gender'] = get_counts('gender')
        stats['ethnicity'] = get_counts('ethnicity')
        
        if 'origin' in df.columns:
            stats['origin'] = get_counts('origin')
        elif 'source' in df.columns:
            stats['source'] = get_counts('source')
            stats['origin'] = get_counts('source')
            
        return stats

    # =========================================================================
    # SPLIT E CONVERSÃO
    # =========================================================================

    def split_by_identity(self, split_ratio: float = 0.8, random_seed: int = 42) -> Tuple['FaceDataset', 'FaceDataset']:
        """
        Divide o dataset em dois (Treino/Teste) garantindo que as identidades sejam disjuntas.
        Isso evita data leakage (vazamento de identidades) entre os conjuntos.
        """
        unique_ids = self.unique_identities
        rng = random.Random(random_seed)
        rng.shuffle(unique_ids)
        
        split_idx = int(len(unique_ids) * split_ratio)
        ids_a = set(unique_ids[:split_idx])
        
        faces_a = [f for f in self.faces if f.identity in ids_a]
        faces_b = [f for f in self.faces if f.identity not in ids_a]
        
        return (FaceDataset(f"{self.name}_A", faces=faces_a), 
                FaceDataset(f"{self.name}_B", faces=faces_b))

    def split_by_identity_list(self, ids_for_dataset_b: Union[Set[str], List[str]], name_a: str = "", name_b: str = "") -> Tuple['FaceDataset', 'FaceDataset']:
        """
        Divide o dataset cirurgicamente em dois baseando-se em uma lista explícita de identidades
        que devem compor o Dataset B. As identidades não listadas irão para o Dataset A.

        Args:
            ids_for_dataset_b: Conjunto ou lista contendo as identidades que devem ser movidas para o Dataset B.
            name_a: Nome opcional para o Dataset A gerado.
            name_b: Nome opcional para o Dataset B gerado.

        Returns:
            Tuple[FaceDataset, FaceDataset]: (Dataset A, Dataset B)
        """
        if not isinstance(ids_for_dataset_b, set):
            ids_for_dataset_b = set(ids_for_dataset_b)

        faces_a = []
        faces_b = []

        for face in self.faces:
            if face.identity in ids_for_dataset_b:
                faces_b.append(face)
            else:
                faces_a.append(face)

        n_a = name_a if name_a else f"{self.name}_A"
        n_b = name_b if name_b else f"{self.name}_B"

        return (
            FaceDataset(name=n_a, faces=faces_a, description=f"Split manual de {self.name} (Restante)"),
            FaceDataset(name=n_b, faces=faces_b, description=f"Split manual de {self.name} (Específicos)")
        )

    def to_pandas(self) -> pd.DataFrame:
        """
        Converte a lista de objetos FaceInfo em um pandas DataFrame tabular.
        Útil para visualização em tabelas (QTableView) e exportação.
        """
        if not self.faces:
            return pd.DataFrame()
        
        data = [face.to_flat_dict() for face in self.faces]
        return pd.DataFrame(data)
            
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "Dataset",
                       embedding_column: Optional[str] = None) -> 'FaceDataset':
        """
        Reconstrói um FaceDataset a partir de um pandas DataFrame.
        Realiza o parsing de colunas flat para as estruturas aninhadas (FFInfo, OFIQInfo).
        """
        if df.empty:
            return cls(name=name)

        dataset = cls(name=name)
        
        # 1. Detecção da Coluna de Embedding principal
        col_emb = None
        if embedding_column and embedding_column in df.columns:
            col_emb = embedding_column  # uso direto — configurado na estratégia
        else:
            for candidate in ['embedding', 'embeddings', 'feature', 'vector', 'embedding_v2']:
                if candidate in df.columns:
                    col_emb = candidate
                    break

        # 2. Mapa de Tradução (Parquet -> Atributo Interno)
        parquet_mapping = {
            'yaw_raw': 'head_pose_yaw_raw', 'yaw_scalar': 'head_pose_yaw_scalar',
            'pitch_raw': 'head_pose_pitch_raw', 'pitch_scalar': 'head_pose_pitch_scalar',
            'roll_raw': 'head_pose_roll_raw', 'roll_scalar': 'head_pose_roll_scalar',
            'score_raw': 'unified_quality_score_raw', 'score': 'unified_quality_score_scalar',
            'ipd_raw': 'inter_eye_distance_raw', 'ipd_scalar': 'inter_eye_distance_scalar',
            'luminance_raw': 'luminance_mean_raw', 'luminance_scalar': 'luminance_mean_scalar',
            'lum_var_raw': 'luminance_variance_raw', 'lum_var_scalar': 'luminance_variance_scalar',
            'underexposure_raw': 'under_exposure_raw', 'underexposure_scalar': 'under_exposure_scalar',
            'overexposure_raw': 'over_exposure_raw', 'overexposure_scalar': 'over_exposure_scalar',
            'bg_uniformity_raw': 'background_uniformity_raw', 'bg_uniformity_scalar': 'background_uniformity_scalar',
            'illum_uniformity_raw': 'illumination_uniformity_raw', 'illum_uniformity_scalar': 'illumination_uniformity_scalar',
            'compression_raw': 'compression_artifacts_raw', 'compression_scalar': 'compression_artifacts_scalar',
            'single_face_raw': 'single_face_present_raw', 'single_face_scalar': 'single_face_present_scalar',
            'expression_raw': 'expression_neutrality_raw', 'expression_scalar': 'expression_neutrality_scalar',
            'no_head_covering_scalar': 'no_head_coverings_scalar', 'no_head_covering_raw': 'no_head_coverings_raw'
        }
        
        valid_ofiq_fields = set(OFIQInfo.__annotations__.keys())

        # 3. Iteração e Reconstrução
        for _, row in df.iterrows():
            def clean(val):
                return None if pd.isna(val) or val == "" else val

            # A. FaceInfo Básico
            face = FaceInfo(
                image_path=str(row.get('image_path', '')),
                file_name=clean(row.get('file_name')),
                identity=clean(row.get('identity')),
                ethnicity=clean(row.get('ethnicity')),
                gender=clean(row.get('gender')),
                origin=clean(row.get('origin', row.get('source', 'Local'))),
                hash=clean(row.get('hash'))
            )
            
            # B. Embedding e FFInfo
            emb_array = None
            if col_emb and isinstance(row[col_emb], (list, np.ndarray)):
                 emb_array = np.array(row[col_emb], dtype=np.float32)

            if pd.notna(row.get('fiqa_score')) or emb_array is not None:
                face.ff_info = FFInfo(
                    fiqa_score=clean(row.get('fiqa_score')),
                    yaw=clean(row.get('yaw')),
                    pitch=clean(row.get('pitch')),
                    roll=clean(row.get('roll')),
                    age=clean(row.get('age_est')),
                    gender=clean(row.get('gender_est')),
                    ipd=clean(row.get('ipd')),
                    embedding_sepaelv2=emb_array 
                )
            
            # C. OFIQInfo
            ofiq_values = {}
            extra_metrics = {}
            for col in df.columns:
                if col.startswith('ofiq_'):
                    val = row[col]
                    if pd.notna(val) and val != "":
                        raw_key = col[5:]
                        attr_name = parquet_mapping.get(raw_key, raw_key)
                        if attr_name in valid_ofiq_fields:
                            try:
                                ofiq_values[attr_name] = float(val)
                            except (ValueError, TypeError):
                                ofiq_values[attr_name] = val
                        else:
                            extra_metrics[raw_key] = val

            if ofiq_values:
                info = OFIQInfo(**ofiq_values)
                info._extra_metrics = extra_metrics
                face.ofiq_info = info
            
            # D. SERFIQ
            if pd.notna(row.get('serfiq_score')):
                face.serfiq_info = SERFIQInfo(
                    score=float(row['serfiq_score']),
                    t=10,
                    alpha=1.5
                )

            # E. Custom columns (PR 21 — Tema 4.2). Captura colunas que
            # não pertencem ao mapeamento canônico (FaceInfo / FFInfo /
            # OFIQInfo / SERFIQInfo) em `face.extra_data`. Round-trip via
            # `to_flat_dict` → `to_pandas` → `to_parquet` → `from_dataframe`
            # mantém custom columns vivas em save/load/filter.
            extra_data: Dict[str, Any] = {}
            for col in df.columns:
                if _is_canonical_face_column(col):
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                # Mantém apenas escalares — listas/arrays/dicts ficam fora
                # do round-trip (consistente com to_flat_dict que só emite
                # custom values via update simples).
                if isinstance(val, (list, np.ndarray, dict)):
                    continue
                extra_data[col] = clean(val)
            if extra_data:
                face.extra_data = extra_data

            dataset.add(face)

        return dataset

    @classmethod
    def from_parquet(cls, path: Union[str, Path],
                     embedding_column: Optional[str] = None) -> 'FaceDataset':
        """Carrega um dataset do disco — auto-detecta formato novo vs legado.

        - Novo: parquet escrito por to_parquet pós-PR 3 contém `b"faceaudit:dataset_type"`
          em schema.metadata. Nome/descrição/column_specs vêm da metadata.
        - Legado: parquet sem essa key. Lê via pandas, extrai dataset_name da
          coluna repetida (linha 0), emite warning recomendando migração.
        """
        import warnings
        import pyarrow.parquet as pq

        path_obj = Path(path)
        if not path_obj.exists():
            return cls(name=path_obj.stem)

        try:
            table = pq.read_table(path_obj)
        except Exception:
            # Fallback total para o caminho pandas em caso de erro pyarrow.
            df = pd.read_parquet(path_obj)
            return cls.from_dataframe(df, name=path_obj.stem, embedding_column=embedding_column)

        meta = _decode_schema_metadata(table.schema.metadata)
        is_new_format = "faceaudit:dataset_type" in meta

        if is_new_format:
            ds = cls._from_new_format(table, meta, path_obj, embedding_column)
        else:
            warnings.warn(
                f"Parquet legado detectado em '{path_obj.name}' — sem column_specs na metadata. "
                f"Recomenda-se migrar via `python -m faceaudit.scripts.migrate_legacy_parquet "
                f"{path_obj}`.",
                stacklevel=2,
            )
            ds = cls._from_legacy_format(table, path_obj, embedding_column)

        return ds

    @classmethod
    def _from_new_format(cls, table, meta: Dict[str, str], path_obj: Path,
                         embedding_column: Optional[str]) -> 'FaceDataset':
        """Lê parquet escrito pelo to_parquet pós-PR 3."""
        from faceaudit.models.column_catalog import specs_from_json

        ds_name = meta.get("faceaudit:dataset_name") or path_obj.stem
        ds_description = meta.get("faceaudit:dataset_description") or None

        column_specs: Dict[str, Any] = {}
        specs_payload = meta.get("faceaudit:column_specs")
        if specs_payload:
            try:
                column_specs = dict(specs_from_json(specs_payload))
            except Exception:
                column_specs = {}

        df = table.to_pandas()
        ds = cls.from_dataframe(df, name=ds_name, embedding_column=embedding_column)
        ds.description = ds_description
        ds.column_specs = column_specs
        return ds

    @classmethod
    def _from_legacy_format(cls, table, path_obj: Path,
                            embedding_column: Optional[str]) -> 'FaceDataset':
        """Lê parquet legado (pré-PR 3): nome vem da coluna `dataset_name` linha 0."""
        df = table.to_pandas()
        ds_name = path_obj.stem
        if not df.empty and "dataset_name" in df.columns:
            saved_name = df.iloc[0].get("dataset_name")
            if saved_name and pd.notna(saved_name):
                ds_name = str(saved_name)
        return cls.from_dataframe(df, name=ds_name, embedding_column=embedding_column)

    def to_parquet(self, path: Union[str, Path]):
        """Persiste o dataset em parquet com metadata role-aware (novo formato).

        Diferente do save antigo:
          - Não duplica `dataset_name`/`description`/`hash` como colunas — vão
            para `schema.metadata`.
          - Dropa o alias retrocompat `embedding` (= `embedding_v2`).
          - Persiste `column_specs`, `column_coverage` e `samples_per_identity`
            na metadata, eliminando o sidecar JSON em PR 4.
        """
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq

        from faceaudit.models.column_catalog import (
            DatasetType, get_default_specs, specs_to_json,
        )
        from faceaudit.services.dataset_stats_service import DatasetStatsService

        path_obj = Path(path)
        if not path_obj.parent.exists():
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_pandas()

        # Caso vazio — escreve apenas a metadata mínima e o schema vazio.
        if df.empty:
            empty_table = pa.Table.from_pandas(df, preserve_index=False)
            empty_table = empty_table.replace_schema_metadata(_build_metadata_bytes({
                "faceaudit:dataset_type": DatasetType.FACE.value,
                "faceaudit:dataset_name": self.name,
                "faceaudit:dataset_description": self.description or "",
                "faceaudit:path_mode": "absolute",
            }))
            pq.write_table(empty_table, path_obj)
            return

        # Dropa o alias `embedding` quando `embedding_v2` está presente (são o
        # mesmo dado; alias só existia para retrocompat). Migração efetiva.
        if "embedding" in df.columns and "embedding_v2" in df.columns:
            df = df.drop(columns=["embedding"])

        # Constrói column_specs: usa self.column_specs se populado (caso já
        # tenha sido lido de um parquet novo ou customizado pela UI); caso
        # contrário, deriva do catálogo built-in filtrado pelas colunas
        # presentes.
        if self.column_specs:
            specs_to_persist = {
                name: spec for name, spec in self.column_specs.items()
                if name in df.columns
            }
        else:
            defaults = get_default_specs(DatasetType.FACE)
            specs_to_persist = {
                name: spec for name, spec in defaults.items()
                if name in df.columns
            }
            # Atualiza self.column_specs em memória (consistência pós-save)
            self.column_specs = dict(specs_to_persist)

        # Computa column_coverage + samples_per_identity via o serviço
        stats_service = DatasetStatsService()
        stats = stats_service._stats_from_dataframe(df, total_rows=len(df))

        # Constrói a metadata do schema
        meta_payload = {
            "faceaudit:dataset_type": DatasetType.FACE.value,
            "faceaudit:dataset_name": self.name,
            "faceaudit:dataset_description": self.description or "",
            "faceaudit:path_mode": "absolute",
            "faceaudit:column_specs": specs_to_json(specs_to_persist),
            "faceaudit:column_coverage": json.dumps({
                name: cs.to_dict() for name, cs in stats.column_coverage.items()
            }, ensure_ascii=False, separators=(",", ":")),
            "faceaudit:samples_per_identity": json.dumps(
                stats.samples_per_identity, ensure_ascii=False, separators=(",", ":")
            ),
            "faceaudit:n_imgs": str(stats.n_imgs),
            "faceaudit:n_ids": str(stats.n_ids),
            "faceaudit:last_modified": stats.last_modified or "",
        }

        try:
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = table.replace_schema_metadata(_build_metadata_bytes(meta_payload))
            pq.write_table(table, path_obj)
            print(f"✅ Dataset '{self.name}' salvo em: {path_obj} ({len(df)} registros, "
                  f"{len(specs_to_persist)} column_specs)")
        except Exception as e:
            print(f"❌ Erro ao salvar parquet: {e}")
            raise

    # =========================================================================
    # CONVERSÃO PARA EMBEDDING DATASET
    # =========================================================================

    def to_embedding_dataset(self, embedding_type: Optional['EmbeddingType'] = None) -> EmbeddingDataset:
        """
        Converte o dataset rico (FaceDataset) em dataset leve (EmbeddingDataset).
        Extrai o vetor solicitado, métricas de qualidade e metadados de pose.
        """
        # Import local dos Enums e Modelos necessários
        from faceaudit.enums.quality_score_type import QualityScoreType
        from faceaudit.models.embedding_info import EmbeddingInfo
        from faceaudit.models.embedding_dataset import EmbeddingDataset
        
        emb_items = []
        
        for face in self.faces:
            vec = None
            
            # 1. Extração do Vetor (V2, V4, etc.)
            if embedding_type is not None:
                vec = embedding_type.extract_from(face)
            elif hasattr(face, 'to_embedding'):
                info = face.to_embedding()
                if info: vec = info.embedding
            
            if vec is not None:
                # 2. Extração de Múltiplos Scores de Qualidade
                # Itera sobre o Enum QualityScoreType (FIQA, OFIQ, SERFIQ)
                scores = {}
                for qt in QualityScoreType:
                    val = qt.extract_from(face)
                    if val is not None:
                        scores[qt.name.lower()] = val

                # 3. Extração Explícita de Pose (Yaw e Pitch)
                # Tenta extrair do FFInfo (ForensicFace) ou do OFIQInfo
                yaw_val = 0.0
                pitch_val = 0.0
                
                if face.ff_info:
                    yaw_val = face.ff_info.yaw if face.ff_info.yaw is not None else 0.0
                    pitch_val = face.ff_info.pitch if face.ff_info.pitch is not None else 0.0
                elif face.ofiq_info:
                    yaw_val = face.ofiq_info.head_pose_yaw_raw if face.ofiq_info.head_pose_yaw_raw is not None else 0.0
                    pitch_val = face.ofiq_info.head_pose_pitch_raw if face.ofiq_info.head_pose_pitch_raw is not None else 0.0

                # 4. Criação do EmbeddingInfo com os novos atributos de Pose
                emb_items.append(EmbeddingInfo(
                    embedding=vec,
                    face_hash=face.hash or "",
                    image_path=face.image_path,
                    identity=face.identity,
                    file_name=face.file_name or "",
                    origin=face.origin,
                    yaw=float(yaw_val),
                    pitch=float(pitch_val),
                    quality_scores=scores
                ))
        
        desc_suffix = f" (Source: {embedding_type.display_name})" if embedding_type else ""
        
        return EmbeddingDataset(
            name=self.name,
            items=emb_items,
            description=f"{self.description or ''}{desc_suffix}"
        )
    
    # =========================================================================
    # AUDITORIA DE INTEGRIDADE E QUALIDADE
    # =========================================================================

    def analyze_hash_integrity(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Varre o dataset agrupando imagens pelo hash do arquivo (duplicatas exatas).
        Classifica os achados em dois níveis de alerta baseados na anotação de identidade.

        Returns:
            Dict com duas listas:
            - 'consistent': Duplicatas benignas (mesmo hash, mesma identidade).
            - 'inconsistent': Contaminação crítica (mesmo hash, identidades diferentes).
        """
        from collections import defaultdict

        hash_groups = defaultdict(list)

        # 1. Agrupa todas as faces por hash (ignorando quem não tem hash calculado)
        for face in self.faces:
            if face.hash and str(face.hash).strip():
                hash_groups[face.hash].append(face)

        results = {
            "consistent": [],
            "inconsistent": []
        }

        # 2. Avalia a consistência de identidade para grupos com colisão (>1 imagem)
        for hsh, faces in hash_groups.items():
            if len(faces) > 1:
                # Extrai as identidades, tratando valores nulos para evitar falhas na auditoria
                unique_ids = {face.identity if face.identity else "Desconhecido" for face in faces}

                group_info = {
                    "hash": hsh,
                    "count": len(faces),
                    "identities": sorted(list(unique_ids)),
                    "faces": faces  # Mantemos a referência aos objetos para a UI recuperar caminhos/arquivos
                }

                # Se há mais de uma identidade para a exata mesma foto, é uma anomalia grave
                if len(unique_ids) > 1:
                    results["inconsistent"].append(group_info)
                else:
                    results["consistent"].append(group_info)

        # 3. Ordena os piores ofensores no topo (grupos com mais repetições)
        results["inconsistent"].sort(key=lambda x: x["count"], reverse=True)
        results["consistent"].sort(key=lambda x: x["count"], reverse=True)

        return results
    
    def export_file_names_and_hashes_to_csv(self, path: Union[str, Path]) -> None:
        """
        Exporta um CSV contendo apenas file_name e hash das imagens do dataset.

        Args:
            path: Caminho do arquivo CSV de saída.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for face in self.faces:
            rows.append({
                "identity":face.identity,
                "file_name": face.file_name,
                "hash": face.hash
            })

        df = pd.DataFrame(rows, columns=["identity","file_name", "hash"])
        df.to_csv(path_obj, index=False, encoding="utf-8-sig")

        print(f"✅ CSV exportado com sucesso em: {path_obj} ({len(df)} registros)")


# ───────────────── helpers de metadata pyarrow ─────────────────

def _decode_schema_metadata(raw_meta: Optional[Dict[bytes, bytes]]) -> Dict[str, str]:
    """Decodifica `schema.metadata` (Dict[bytes, bytes]) para Dict[str, str]."""
    if not raw_meta:
        return {}
    decoded: Dict[str, str] = {}
    for k, v in raw_meta.items():
        try:
            key = k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)
            val = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
            decoded[key] = val
        except Exception:
            continue
    return decoded


def _build_metadata_bytes(payload: Dict[str, str]) -> Dict[bytes, bytes]:
    """Encoda Dict[str, str] para o formato do `replace_schema_metadata`."""
    return {k.encode("utf-8"): v.encode("utf-8") for k, v in payload.items()}