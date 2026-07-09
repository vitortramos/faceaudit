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
from faceaudit.models.filter import FilterConfig
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.path_mode import PathMode


# PR 21 (Tema 4.2) — predicado pra identificar colunas canônicas que
# `FaceInfo.to_flat_dict` emite. Tudo fora dessa lista é capturado como
# `face.extra_data` em `from_dataframe` e re-emitido em `to_flat_dict` —
# round-trip pra preservar colunas custom em save/load/filter.
def _canonical_embedding_columns() -> frozenset:
    """Conjunto de nomes de coluna de embedding canônicos derivado do
    `SepaelModel` enum em runtime. Import local pra evitar ciclo de
    import com `enums.embedding_type` que já depende de `face_dataset`.
    Inclui o alias genérico `embedding` (v2 retrocompat) e o `embedding_serfiq`
    histórico que não está no enum."""
    from faceaudit.enums.sepael_model import SepaelModel
    return frozenset(
        {"embedding", "embedding_serfiq"}
        | {member.parquet_column for member in SepaelModel}
    )


_CANONICAL_FACE_COLUMNS = frozenset({
    # FaceInfo metadata
    "image_path", "file_name", "hash", "origin", "source",
    "ethnicity", "identity", "gender",
    # Rastreamento de erros (criação/enriquecimento)
    "error", "error_stage", "error_message",
    # FFInfo (exclude_heavy=True) + renomeados em to_flat_dict
    "ipd", "fiqa_score", "det_score", "gender_est", "age_est",
    "yaw", "pitch", "roll",
})


def _is_canonical_face_column(name: str) -> bool:
    """True quando `name` pertence ao mapeamento canônico de FaceInfo.

    Colunas de embedding são consultadas via `_canonical_embedding_columns()`
    em runtime pra cobrir TODOS os membros do `SepaelModel` enum — sem
    isso, embeddings v3/v5/v6 (qualquer modelo fora dos hard-coded v2/v4)
    caíam no loop de `extra_data` e quebrar com `pd.isna(array)` ambíguo.
    """
    if name in _CANONICAL_FACE_COLUMNS:
        return True
    if name in _canonical_embedding_columns():
        return True
    # OFIQ/SERFIQ emitem múltiplas colunas com prefixo — todas canônicas
    return name.startswith("ofiq_") or name.startswith("serfiq_")


# Chaves de proveniência estruturada (sem o prefixo `faceaudit:`) que o
# `to_parquet`/`from_parquet` aceita gravar/reler em `provenance_metadata`.
# Allowlist deliberada — evita virar saco genérico de metadata. Ex.: aviso
# de viés de seleção de datasets extraídos de vídeo (VIDEO_TARGET_EXTRACTION).
_PROVENANCE_METADATA_KEYS = frozenset({
    "dataset_origin_type",
    "selection_conditioned_on_embedding",
    "selection_anchor_model",
    "selection_threshold",
    "selection_warning",
    "source_run_id",
    "source_run_manifest",
})


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

    # PR L1a — modo de path. ABSOLUTE preserva comportamento legado;
    # RELATIVE habilita portabilidade via DatasetLocalizationService.
    # `path_root` é a subpasta dentro do diretório do parquet onde
    # estão as imagens (default `.` quando flat). Persistidos na
    # metadata via `to_parquet` e relidos por `from_parquet`. Ver
    # `faceaudit/services/image_path_resolver.py` para a resolução.
    path_mode: PathMode = PathMode.ABSOLUTE
    path_root: Optional[str] = None
    # True quando `image_path` aponta pra faces JÁ alinhadas (112×112
    # crop pós-FF.align_face_only) — caso típico: dataset gerado pelo
    # Image Pipeline com `input_source=ALIGNED_FACE`. Quando True, o
    # FFService usa `process_face_aligned` (pula detect+align, vai
    # direto pro embedding extraction). Persistido em
    # `faceaudit:is_aligned` na metadata do parquet.
    is_aligned: bool = False
    # Proveniência estruturada (controlada por allowlist) gravada na
    # metadata do parquet como chaves `faceaudit:*`. Usada, por exemplo,
    # para o aviso de viés de seleção de datasets extraídos de vídeo.
    # Ver `_PROVENANCE_METADATA_KEYS`. Persiste/relê em to_parquet/from_parquet.
    provenance_metadata: Dict[str, str] = field(default_factory=dict)
    # Runtime — diretório do parquet de origem, set por `from_parquet`.
    # Não persiste; não entra em comparações; não aparece em repr.
    _parquet_dir: Optional[Path] = field(
        default=None, repr=False, compare=False,
    )

    def resolve_image_path(self, face_or_path: Any) -> Path:
        """Atalho que delega à free-function
        `faceaudit.services.image_path_resolver.resolve_image_path`,
        passando os parâmetros de resolução do próprio dataset.

        Use isso na UI / código que tem o dataset em mãos. Services
        que recebem só o face (sem dataset) devem chamar a
        free-function diretamente com `parquet_dir`/`path_mode`/
        `path_root` explícitos.
        """
        from faceaudit.services.image_path_resolver import (
            resolve_image_path as _resolve,
        )
        return _resolve(
            face_or_path,
            parquet_dir=self._parquet_dir,
            path_mode=self.path_mode,
            path_root=self.path_root,
        )

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

    def apply_filter(
        self,
        config: 'FilterConfig',
        callback: Optional[Callable[[int], None]] = None,
    ) -> 'FaceDataset':
        """
        Gera um novo FaceDataset contendo apenas as faces que satisfazem
        os critérios do `FilterConfig` (lista polimórfica de
        `ColumnFilter`). Combinação entre filters é AND.

        Delega ao `DatasetFilterService` — toda a lógica de matching
        vive lá. Filtros inválidos (coluna ausente) são silenciosamente
        ignorados; caller deve usar `DatasetFilterService.validate(...)`
        antes de aplicar pra avisar o usuário sobre filters quebrados.

        Args:
            config: `FilterConfig` com a lista de `ColumnFilter`.
            callback: progresso (0–100), chamado a cada 5%.

        Returns:
            FaceDataset: nova instância com as faces filtradas
            (referências compartilhadas, não deep-copy).
        """
        from faceaudit.services.dataset_filter_service import (
            DatasetFilterService,
        )
        return DatasetFilterService.apply(
            self, config, progress_callback=callback,
        )

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

    def to_pandas(self, include_embeddings: bool = True) -> pd.DataFrame:
        """
        Converte a lista de objetos FaceInfo em um pandas DataFrame tabular.
        Útil para visualização em tabelas (QTableView) e exportação.

        `include_embeddings=False` produz um DataFrame **metadata-only** (sem
        as colunas de vetor), usado no caminho de EXIBIÇÃO da tabela — a
        tabela nunca mostra o embedding e mantê-lo no df custa gigabytes em
        datasets de milhões de faces. O default `True` preserva o
        comportamento de `to_parquet`/estatísticas (round-trip completo).
        Ver `FaceInfo.to_flat_dict`.
        """
        if not self.faces:
            return pd.DataFrame()

        data = [
            face.to_flat_dict(include_embeddings=include_embeddings)
            for face in self.faces
        ]
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
            # Erro: lê bool com cuidado — pandas pode retornar np.bool_
            # ou strings "True"/"False" em casos de coerção. Normaliza.
            raw_error = clean(row.get('error'))
            if raw_error is None:
                error_val: Optional[bool] = None
            elif isinstance(raw_error, str):
                error_val = raw_error.strip().lower() in ("true", "1", "yes")
            else:
                error_val = bool(raw_error)

            face = FaceInfo(
                image_path=str(row.get('image_path', '')),
                file_name=clean(row.get('file_name')),
                identity=clean(row.get('identity')),
                ethnicity=clean(row.get('ethnicity')),
                gender=clean(row.get('gender')),
                origin=clean(row.get('origin', row.get('source', 'Local'))),
                hash=clean(row.get('hash')),
                error=error_val,
                error_stage=clean(row.get('error_stage')),
                error_message=clean(row.get('error_message')),
            )
            
            # B. Embedding e FFInfo
            # Primário (sepaelv2, geralmente "embedding"/"embedding_v2"/etc).
            emb_array = None
            if col_emb and isinstance(row[col_emb], (list, np.ndarray)):
                 emb_array = np.array(row[col_emb], dtype=np.float32)

            # Embeddings adicionais por SepaelModel — itera o enum em vez
            # de hard-codear nomes. Sem isso, parquets com `embedding_v3`/
            # `embedding_v5`/`embedding_v6` perdiam silenciosamente esses
            # vetores no load e a auditoria reportava 100% de faltantes
            # mesmo com dados presentes. Para v2, mantém o `emb_array` já
            # resolvido por `col_emb` (que respeita aliases legados como
            # `embedding` puro).
            from faceaudit.enums.sepael_model import SepaelModel

            embeddings_per_model: dict[str, Optional[np.ndarray]] = {}
            for member in SepaelModel:
                col = member.parquet_column
                if col not in df.columns:
                    continue
                val = row[col]
                if isinstance(val, (list, np.ndarray)):
                    embeddings_per_model[member.ffinfo_slot] = (
                        np.array(val, dtype=np.float32)
                    )

            # v2 tem ainda o caminho legado via `col_emb` (cobre parquets
            # antigos com coluna `embedding` sem o sufixo `_v2`).
            if emb_array is not None:
                embeddings_per_model.setdefault(
                    SepaelModel.SEPAEL_V2.ffinfo_slot, emb_array,
                )

            if pd.notna(row.get('fiqa_score')) or embeddings_per_model:
                face.ff_info = FFInfo(
                    fiqa_score=clean(row.get('fiqa_score')),
                    yaw=clean(row.get('yaw')),
                    pitch=clean(row.get('pitch')),
                    roll=clean(row.get('roll')),
                    age=clean(row.get('age_est')),
                    gender=clean(row.get('gender_est')),
                    ipd=clean(row.get('ipd')),
                    **embeddings_per_model,
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
                # Skip de array/list/dict ANTES de pd.isna — sem isso,
                # `pd.isna(array)` retorna array de bools e o `if` quebra
                # com "truth value of an array is ambiguous". Acontecia
                # quando uma coluna de embedding não-canônica chegava aqui.
                if isinstance(val, (list, np.ndarray, dict)):
                    continue
                if pd.isna(val):
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

        # PR L1a — lê path_mode/path_root da metadata. Defaults preservam
        # comportamento legado: parquets sem path_mode são tratados como
        # ABSOLUTE (era o que o writer hardcoded gravava antes).
        path_mode_raw = meta.get("faceaudit:path_mode") or PathMode.ABSOLUTE.value
        try:
            path_mode = PathMode(path_mode_raw)
        except ValueError:
            path_mode = PathMode.ABSOLUTE
        path_root = meta.get("faceaudit:path_root") or None
        # PR L1c.1 — normaliza defensivamente pra POSIX (parquets gerados
        # no Windows pré-fix podem ter `\` no path_root).
        if path_root:
            from faceaudit.services.image_path_resolver import normalize_stored_path
            path_root = normalize_stored_path(path_root)

        # is_aligned: bool persistido como string "true"/"false" em
        # parquet metadata. Default False preserva comportamento legado
        # pra parquets pré-feature (criados antes deste flag).
        is_aligned_raw = meta.get("faceaudit:is_aligned") or ""
        is_aligned = is_aligned_raw.strip().lower() in ("true", "1", "yes")

        df = table.to_pandas()
        ds = cls.from_dataframe(df, name=ds_name, embedding_column=embedding_column)
        ds.description = ds_description
        ds.column_specs = column_specs
        ds.path_mode = path_mode
        ds.path_root = path_root
        ds.is_aligned = is_aligned
        ds.provenance_metadata = {
            k: meta[f"faceaudit:{k}"]
            for k in _PROVENANCE_METADATA_KEYS
            if f"faceaudit:{k}" in meta
        }
        # Set _parquet_dir runtime — usado por resolve_image_path em modo
        # RELATIVE pra ancorar paths.
        ds._parquet_dir = path_obj.parent.resolve()
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
        ds = cls.from_dataframe(df, name=ds_name, embedding_column=embedding_column)
        # PR L1a — parquets legados são sempre ABSOLUTE; ainda assim
        # registramos `_parquet_dir` pra que código upstream que use
        # `resolve_image_path` em datasets recém-abertos não dependa
        # do modo (e pra suportar futura migração in-place).
        ds._parquet_dir = path_obj.parent.resolve()
        return ds

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
            empty_meta = {
                "faceaudit:dataset_type": DatasetType.FACE.value,
                "faceaudit:dataset_name": self.name,
                "faceaudit:dataset_description": self.description or "",
                "faceaudit:path_mode": self.path_mode.value,
            }
            # PR L1a — `path_root` só é gravado em modo RELATIVE; em
            # ABSOLUTE não faz sentido (e mantém metadata enxuta).
            # PR L1c.1 — normaliza pra POSIX antes de gravar (cross-platform).
            if self.path_mode == PathMode.RELATIVE and self.path_root:
                from faceaudit.services.image_path_resolver import normalize_stored_path
                empty_meta["faceaudit:path_root"] = normalize_stored_path(self.path_root)
            if self.is_aligned:
                empty_meta["faceaudit:is_aligned"] = "true"
            for _k, _v in (self.provenance_metadata or {}).items():
                if _k in _PROVENANCE_METADATA_KEYS:
                    empty_meta[f"faceaudit:{_k}"] = str(_v)
            empty_table = pa.Table.from_pandas(df, preserve_index=False)
            empty_table = empty_table.replace_schema_metadata(
                _build_metadata_bytes(empty_meta),
            )
            pq.write_table(empty_table, path_obj)
            return

        # Dropa o alias `embedding` quando `embedding_v2` está presente (são o
        # mesmo dado; alias só existia para retrocompat). Migração efetiva.
        if "embedding" in df.columns and "embedding_v2" in df.columns:
            df = df.drop(columns=["embedding"])

        # Constrói column_specs: começa com self.column_specs se populado
        # (caso já tenha sido lido de um parquet novo ou customizado pela
        # UI), e completa com defaults do catálogo pra colunas novas que
        # apareceram via enrichment (ff_info → embedding_v2/fiqa_score/
        # det_score/age_est/...; ofiq_info → ofiq_*; serfiq_info →
        # serfiq_score). Sem esse complemento, `to_parquet` salvava só
        # os specs originais e o "Mostrar/Ocultar Colunas" da UI ficava
        # cego pras novas colunas.
        defaults = get_default_specs(DatasetType.FACE)
        specs_to_persist: Dict[str, Any] = {}
        if self.column_specs:
            # PR 23 (Tema 4.3) — colunas computadas não têm coluna real no
            # df (valor é avaliado on-the-fly), mas o spec com `computed_expr`
            # precisa ser persistido pra que a fórmula sobreviva ao
            # save/reload.
            for name, spec in self.column_specs.items():
                if name in df.columns or getattr(spec, "computed_expr", None):
                    specs_to_persist[name] = spec
        # Adiciona defaults pra colunas presentes no df mas sem spec.
        for name, spec in defaults.items():
            if name in df.columns and name not in specs_to_persist:
                specs_to_persist[name] = spec
        # Atualiza self.column_specs em memória — consistência pós-save
        # garante que a UI veja os novos specs sem reabrir o parquet.
        self.column_specs = dict(specs_to_persist)

        # Computa column_coverage + samples_per_identity via o serviço
        stats_service = DatasetStatsService()
        stats = stats_service._stats_from_dataframe(df, total_rows=len(df))

        # Constrói a metadata do schema
        meta_payload = {
            "faceaudit:dataset_type": DatasetType.FACE.value,
            "faceaudit:dataset_name": self.name,
            "faceaudit:dataset_description": self.description or "",
            "faceaudit:path_mode": self.path_mode.value,
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

        # Proveniência estruturada (allowlist) — ex.: aviso de viés de seleção.
        for _k, _v in (self.provenance_metadata or {}).items():
            if _k in _PROVENANCE_METADATA_KEYS:
                meta_payload[f"faceaudit:{_k}"] = str(_v)

        # PR L1a — `path_root` só entra na metadata em modo RELATIVE.
        # PR L1c.1 — normaliza pra POSIX antes de gravar (cross-platform).
        if self.path_mode == PathMode.RELATIVE and self.path_root:
            from faceaudit.services.image_path_resolver import normalize_stored_path
            meta_payload["faceaudit:path_root"] = normalize_stored_path(self.path_root)

        # is_aligned só entra na metadata quando True — enxuto pra
        # datasets de fotos originais (a maioria).
        if self.is_aligned:
            meta_payload["faceaudit:is_aligned"] = "true"

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