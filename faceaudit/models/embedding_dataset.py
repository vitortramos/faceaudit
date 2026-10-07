# faceaudit/models/embedding_dataset.py
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Tuple, Set
import numpy as np
import pandas as pd
import pickle
import json
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Import relativo
from .embedding_info import EmbeddingInfo
from .aggregation_provenance import (
    COLUNA_COUNT, COLUNA_HASHES, COLUNA_PATHS,
)


def _prov_count(row_dict):
    """Número de faces agregadas, lido da linha do parquet.

    `None` em face comum. O parquet devolve NaN quando a coluna existe por
    causa de OUTRA linha do mesmo arquivo, e NaN tem de virar ausente.
    """
    bruto = row_dict.get(COLUNA_COUNT)
    if bruto is None or (isinstance(bruto, float) and bruto != bruto):
        return None
    try:
        valor = int(bruto)
    except (TypeError, ValueError):
        return None
    return valor if valor > 0 else None


def _prov_lista(bruto):
    """Lista de strings tolerante ao que o parquet devolve.

    Coluna de lista volta como `numpy.ndarray`, e NaN aparece nas linhas
    de face comum do mesmo arquivo.
    """
    if bruto is None:
        return None
    if isinstance(bruto, float) and bruto != bruto:
        return None
    try:
        valores = [str(v) for v in bruto if v is not None]
    except TypeError:
        return None
    return valores or None


@dataclass
class EmbeddingDataset:
    """
    Coleção otimizada de embeddings para processamento matemático e persistência.
    """
    name: str
    items: List[EmbeddingInfo] = field(default_factory=list)
    description: Optional[str] = None

    # Metadados lidos do arquivo (opcional)
    loaded_metadata: Dict[str, str] = field(default_factory=dict)



    # =========================================================================
    # ESTRATIFICAÇÃO BALANCEADA (QUALIDADE E NORMA L2)
    # =========================================================================

    def get_balanced_subsets_by_quality(self, min_images: int = 3) -> Tuple['EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset']:
        """
        Gera quatro subsets balanceados do dataset atual com base no score de qualidade (FIQA/OFIQ).

        Este método é fundamental para isolar o viés de densidade populacional (desbalanceamento
        demográfico) durante avaliações biométricas. Para cada identidade que possui pelo menos 
        `min_images`, o algoritmo extrai cirurgicamente três amostras representativas da variância 
        do indivíduo: a melhor foto, a foto mediana e a pior foto.

        Args:
            min_images (int): Número mínimo de imagens que uma identidade deve possuir na base 
                              original para ser incluída no particionamento. O padrão é 3.

        Returns:
            Tuple['EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset']:
                Uma tupla contendo 4 novos objetos EmbeddingDataset isolados:
                - [0] ds_total: O dataset balanceado completo (contendo as 3 imagens de cada ID válida).
                - [1] ds_best: Apenas as imagens de MAIOR qualidade (Top 1) de cada ID.
                - [2] ds_median: Apenas as imagens de qualidade INTERMEDIÁRIA de cada ID.
                - [3] ds_worst: Apenas as imagens de PIOR qualidade (Bottom 1) de cada ID.

        Warning:
            Para auditoria de Hubness Topológico, a busca de vizinhos (K-NN) deve ser executada 
            EXCLUSIVAMENTE sobre o `ds_total`. Se os subsets isolados (best, worst) forem 
            processados separadamente pelo K-NN, o ecossistema espacial é quebrado e o viés 
            de atração será mascarado.
        """
        from collections import defaultdict
        
        if not self.items:
            empty = EmbeddingDataset(name=f"{self.name}_Empty", loaded_metadata=self.loaded_metadata.copy())
            return empty, empty, empty, empty

        # Agrupa índices e scores (ignorando quem não tem score)
        id_to_scores = defaultdict(list)
        for i, item in enumerate(self.items):
            if item.identity is not None and item.quality_score is not None:
                id_to_scores[item.identity].append((i, float(item.quality_score)))
                
        best_idx, median_idx, worst_idx = [], [], []
        
        for identity, tuples in id_to_scores.items():
            if len(tuples) >= min_images:
                # Ordena decrescente pela qualidade (maior para menor)
                sorted_tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
                
                best_idx.append(sorted_tuples[0][0])
                median_idx.append(sorted_tuples[len(sorted_tuples) // 2][0])
                worst_idx.append(sorted_tuples[-1][0])
                
        # Constrói os vetores de itens
        best_items = [self.items[i] for i in best_idx]
        median_items = [self.items[i] for i in median_idx]
        worst_items = [self.items[i] for i in worst_idx]
        all_items = best_items + median_items + worst_items
        
        meta = self.loaded_metadata.copy()
        
        ds_total = EmbeddingDataset(name=f"{self.name}_Qual_Total", items=all_items, loaded_metadata=meta)
        ds_best = EmbeddingDataset(name=f"{self.name}_Qual_Best", items=best_items, loaded_metadata=meta)
        ds_median = EmbeddingDataset(name=f"{self.name}_Qual_Median", items=median_items, loaded_metadata=meta)
        ds_worst = EmbeddingDataset(name=f"{self.name}_Qual_Worst", items=worst_items, loaded_metadata=meta)
        
        return ds_total, ds_best, ds_median, ds_worst

    def get_balanced_subsets_by_norm(self, min_images: int = 3) -> Tuple['EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset']:
        """
        Gera quatro subsets balanceados do dataset atual usando a Magnitude do Embedding 
        (Norma L2) como um proxy biométrico para a qualidade da imagem e confiança da rede.

        Em modelos de Deep Metric Learning (ArcFace, AdaFace), a norma do vetor bruto é um 
        termômetro direto da ativação da rede. Este método separa as identidades em estratos de 
        magnitude, permitindo investigar anomalias de ativação (Out-of-Distribution) e degradação.
        

        Args:
            min_images (int): Número mínimo de imagens que uma identidade deve possuir na base 
                              original para participar do balanceamento. O padrão é 3.

        Returns:
            Tuple['EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset', 'EmbeddingDataset']:
                Uma tupla contendo 4 novos objetos EmbeddingDataset:
                - [0] ds_total: O dataset balanceado completo (contendo as 3 imagens de cada ID válida).
                - [1] ds_best: Imagens com a MAIOR Norma L2 de cada ID (Proxy HQ / Altas Ativações).
                - [2] ds_median: Imagens com a Norma L2 MEDIANA de cada ID (Casos ideais comportados).
                - [3] ds_worst: Imagens com a MENOR Norma L2 de cada ID (Proxy LQ / Baixa Confiança).

        Note:
            Embora o K-NN deva rodar no `ds_total` para avaliar a topologia (Hubness), os datasets 
            retornados avulsos (`ds_best`, `ds_worst`) são ideais para rodar testes padrão de 
            Acurácia de Reconhecimento Facial (ex: Curvas ROC, TAR@FAR) de forma isolada por qualidade.
        """
        from collections import defaultdict
        import numpy as np
        
        if not self.items:
            empty = EmbeddingDataset(name=f"{self.name}_Empty", loaded_metadata=self.loaded_metadata.copy())
            return empty, empty, empty, empty

        raw_norms = np.linalg.norm(self.matrix, axis=1)
        
        id_to_indices = defaultdict(list)
        for i, item in enumerate(self.items):
            if item.identity is not None:
                id_to_indices[item.identity].append(i)
                
        best_idx, median_idx, worst_idx = [], [], []
        
        for identity, indices in id_to_indices.items():
            if len(indices) >= min_images:
                # Ordena decrescente pela norma L2 (maior para menor)
                sorted_indices = sorted(indices, key=lambda idx: raw_norms[idx], reverse=True)
                
                best_idx.append(sorted_indices[0])
                median_idx.append(sorted_indices[len(sorted_indices) // 2])
                worst_idx.append(sorted_indices[-1])
                
        # Constrói os vetores de itens
        best_items = [self.items[i] for i in best_idx]
        median_items = [self.items[i] for i in median_idx]
        worst_items = [self.items[i] for i in worst_idx]
        all_items = best_items + median_items + worst_items
        
        meta = self.loaded_metadata.copy()
        
        ds_total = EmbeddingDataset(name=f"{self.name}_Norm_Total", items=all_items, loaded_metadata=meta)
        ds_best = EmbeddingDataset(name=f"{self.name}_Norm_Best", items=best_items, loaded_metadata=meta)
        ds_median = EmbeddingDataset(name=f"{self.name}_Norm_Median", items=median_items, loaded_metadata=meta)
        ds_worst = EmbeddingDataset(name=f"{self.name}_Norm_Worst", items=worst_items, loaded_metadata=meta)
        
        return ds_total, ds_best, ds_median, ds_worst
    
    # =========================================================================
    # PERSISTÊNCIA (LEITURA - PICKLE LEGADO)
    # =========================================================================

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> 'EmbeddingDataset':
        """Carrega de .pkl (Legado/Rápido para objetos Python)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset não encontrado no caminho: {p}")

        with open(p, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Esperava EmbeddingDataset, recebeu {type(obj)}.")

        return obj

    # =========================================================================
    # PERSISTÊNCIA (ESCRITA - PARQUET COM METADADOS)
    # =========================================================================

    def to_parquet(self, path: Union[str, Path], extra_metadata: Dict[str, str] = None) -> None:
        """
        Salva o dataset em Parquet, incluindo metadados customizados no cabeçalho.

        Args:
            path: Caminho do arquivo.
            extra_metadata: Dicionário de strings (ex: {'step_type': 'CoralTransformation'})
                            que será gravado internamente no arquivo.
        """
        if not self.items:
            df = pd.DataFrame(columns=[
                'identity', 'embedding', 'image_path', 'file_name',
                'source', 'origin', 'face_image', 'face_hash',
                'quality_scores_json',
                COLUNA_COUNT, COLUNA_HASHES, COLUNA_PATHS,
            ])
        else:
            data_dicts = []
            for item in self.items:
                emb_list = item.embedding.tolist() if isinstance(item.embedding, np.ndarray) else item.embedding

                face_img_list = None
                if item.face_image is not None:
                    if isinstance(item.face_image, np.ndarray):
                        face_img_list = item.face_image.tolist()
                    else:
                        face_img_list = item.face_image

                q_scores_json = json.dumps(item.quality_scores)

                row = {
                    "embedding": emb_list,
                    "face_hash": item.face_hash,
                    "image_path": item.image_path,
                    "identity": item.identity,
                    "file_name": item.file_name,
                    "source": item.source,
                    "origin": item.origin,
                    "face_image": face_img_list,
                    "quality_scores_json": q_scores_json,
                    # Pose NÃO era gravada — só relida (o `from_parquet` já
                    # aceitava as colunas). Consequência: pose se perdia em todo
                    # save e voltava como 0.0 inventado, de modo que o
                    # HUBNESS_ABLATION BY_POSE operava sobre zeros. `None`
                    # quando a face não tem pose — ausente continua ausente.
                    "yaw": item.yaw,
                    "pitch": item.pitch,
                    # Proveniência de template agregado. Em face comum sai
                    # vazia; sem ela, saber de quais faces um template foi
                    # feito dependeria de decifrar o nome inventado pela
                    # agregação, de onde só se extrai o número.
                    COLUNA_COUNT: item.aggregation_source_count,
                    COLUNA_HASHES: (
                        list(item.aggregation_source_hashes)
                        if item.aggregation_source_hashes else None
                    ),
                    COLUNA_PATHS: (
                        list(item.aggregation_source_paths)
                        if item.aggregation_source_paths else None
                    ),
                }
                data_dicts.append(row)

            df = pd.DataFrame(data_dicts)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # --- METADADOS NO PARQUET ---
        table = pa.Table.from_pandas(df)

        # `loaded_metadata` era LIDO por `from_parquet` mas nunca GRAVADO —
        # metadado de dataset (proveniência de agregação, avisos de viés,
        # origem) vivia só em memória e evaporava no primeiro save. Escrevê-lo
        # fecha o round-trip. `pandas` é chave reservada do pyarrow: reescrevê-la
        # a partir de um dict relido corromperia o schema.
        _RESERVADAS = {"pandas"}
        combinado = {
            k: v for k, v in (self.loaded_metadata or {}).items()
            if k not in _RESERVADAS
        }
        # `extra_metadata` é override explícito no momento da chamada — vence.
        if extra_metadata:
            combinado.update(extra_metadata)

        if combinado:
            existing_meta = dict(table.schema.metadata or {})
            for k, v in combinado.items():
                existing_meta[str(k).encode('utf-8')] = str(v).encode('utf-8')
            table = table.replace_schema_metadata(existing_meta)

        pq.write_table(table, p)

    # =========================================================================
    # PERSISTÊNCIA (LEITURA - PARQUET COM METADADOS)
    # =========================================================================

    @classmethod
    def from_parquet(cls, path: Union[str, Path],
                     embedding_column: Optional[str] = None) -> 'EmbeddingDataset':
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Arquivo Parquet não encontrado: {p}")

        table = pq.read_table(p)
        df = table.to_pandas()

        raw_meta = table.schema.metadata or {}
        decoded_meta = {k.decode('utf-8'): v.decode('utf-8') for k, v in raw_meta.items()}

        # Fallback para DataFrame vazio
        if df.empty:
            return cls(
                name=p.stem.split('_')[0],
                items=[],
                description=f"Carregado de {p.name}",
                loaded_metadata=decoded_meta
            )

        items: List[EmbeddingInfo] = []

        # Resolve coluna de embedding em 3 etapas:
        # 1. Override explícito do caller (estratégia com embedding_type setado).
        # 2. Auto-detect via metadata do parquet — procura ColumnSpec com
        #    role=EMBEDDING. Cobre parquets migrados (embedding_v2/embedding_v4
        #    sem alias 'embedding' redundante).
        # 3. Nomes canônicos em ordem (embedding_v2, embedding_v4, embedding) —
        #    fallback pra parquets sem metadata estruturada (legado pré-migração
        #    ou gerados por scripts externos).
        _emb_col: Optional[str] = None
        if embedding_column and embedding_column in df.columns:
            _emb_col = embedding_column

        if _emb_col is None and decoded_meta.get('faceaudit:column_specs'):
            try:
                specs = json.loads(decoded_meta['faceaudit:column_specs'])
                for col_name, spec in specs.items():
                    if (
                        col_name in df.columns
                        and isinstance(spec, dict)
                        and str(spec.get('role', '')).lower() == 'embedding'
                    ):
                        _emb_col = col_name
                        break
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

        if _emb_col is None:
            for _cand in ('embedding_v2', 'embedding_v4', 'embedding'):
                if _cand in df.columns:
                    _emb_col = _cand
                    break

        if _emb_col is None:
            raise KeyError(
                f"Nenhuma coluna de embedding encontrada em '{p.name}'. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        for row in df.itertuples(index=False):
            row_dict = row._asdict()

            emb_array = np.array(row_dict[_emb_col], dtype=np.float32)

            face_img_array = None
            if row_dict.get('face_image') is not None:
                val = row_dict['face_image']
                if isinstance(val, list) and len(val) > 0:
                    face_img_array = np.array(val, dtype=np.uint8)
                elif isinstance(val, np.ndarray):
                    face_img_array = val

            q_scores: Dict[str, float] = {}
            if 'quality_scores_json' in row_dict and row_dict['quality_scores_json']:
                try:
                    q_scores = json.loads(row_dict['quality_scores_json'])
                except Exception:
                    q_scores = {}

            # Fallback para schema legado: colunas separadas fiqa_score /
            # ofiq_score / serfiq_score (formato gerado pelos scripts de
            # extração antigos). Só preenchemos quando o JSON está
            # ausente/vazio para não sobrescrever valores explícitos.
            if not q_scores:
                _legacy_map = {
                    'fiqa_score': 'fiqa',
                    'ofiq_score': 'ofiq',
                    'serfiq_score': 'serfiq',
                }
                for col_name, key in _legacy_map.items():
                    if col_name in row_dict:
                        val = row_dict[col_name]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            try:
                                q_scores[key] = float(val)
                            except (TypeError, ValueError):
                                pass

            # Pose também pode vir em colunas separadas (yaw/pitch) no
            # schema legado — preserva pra HUBNESS_ABLATION BY_POSE.
            # AUSENTE ≠ ZERO. Antes, coluna faltando ou nula virava 0.0 — e
            # 0° é uma pose fisicamente válida (rosto frontal), então o dado
            # inventado era indistinguível de medição real. Efeito prático: a
            # `HasPoseMetadataRule` lia cobertura 1.0 e LIBERAVA o
            # HUBNESS_ABLATION BY_POSE sobre um dataset inteiro de zeros
            # falsos. `None` deixa o campo fora do dict `pose`, e as regras de
            # cobertura passam a bloquear corretamente. Também é o que
            # preserva, após o round-trip, a ausência de pose nos datasets de
            # templates agregados — que não têm pose por definição.
            def _pose_ou_none(v):
                if v is None:
                    return None
                if isinstance(v, float) and np.isnan(v):
                    return None
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None

            yaw_f = _pose_ou_none(row_dict.get('yaw'))
            pitch_f = _pose_ou_none(row_dict.get('pitch'))

            # `hash` é o nome canônico no novo column_catalog; `face_hash`
            # é o alias legado (parquets antigos + atributo Python interno
            # de EmbeddingInfo). Aceita ambos pra portabilidade entre
            # versões — preferindo o canônico quando ambos existem.
            hash_value = (
                row_dict.get('hash')
                if row_dict.get('hash') is not None
                else row_dict.get('face_hash', '')
            )
            info = EmbeddingInfo(
                embedding=emb_array,
                face_hash=hash_value or '',
                image_path=row_dict.get('image_path', ''),
                identity=row_dict.get('identity'),
                file_name=row_dict.get('file_name', ''),
                source=row_dict.get('source'),
                origin=row_dict.get('origin'),
                face_image=face_img_array,
                yaw=yaw_f,
                pitch=pitch_f,
                aggregation_source_count=_prov_count(row_dict),
                aggregation_source_hashes=_prov_lista(
                    row_dict.get(COLUNA_HASHES)
                ),
                aggregation_source_paths=_prov_lista(
                    row_dict.get(COLUNA_PATHS)
                ),
                quality_scores=q_scores,
            )
            items.append(info)

        return cls(
            name=p.stem.split('_')[0],
            items=items,
            description=f"Carregado de {p.name}",
            loaded_metadata=decoded_meta
        )

    # =========================================================================
    # MÉTODOS MÁGICOS E OPERAÇÕES
    # =========================================================================

    def add(self, item: EmbeddingInfo) -> None:
        self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index) -> EmbeddingInfo:
        return self.items[index]

    def __add__(self, other: 'EmbeddingDataset') -> 'EmbeddingDataset':
        if not isinstance(other, EmbeddingDataset):
            raise TypeError("Soma permitida apenas entre EmbeddingDatasets.")

        new_items = self.items + other.items
        return EmbeddingDataset(
            name=f"{self.name}_{other.name}_Merged",
            description=f"União de {self.name} e {other.name}",
            items=new_items
        )

    # =========================================================================
    # OPERAÇÕES DE CONJUNTO (SPLIT)
    # =========================================================================

    def split_by_identities(self, identity_ids: Set[Optional[str]]) -> Tuple['EmbeddingDataset', 'EmbeddingDataset']:
        """
        Divide o dataset em dois novos datasets baseando-se em um conjunto de identidades.

        Regra:
        - Se item.identity is None -> cai no subset_out, a menos que None esteja explicitamente em identity_ids.
        - Comparação é feita sem coerção para str (evita o bug do "None").
        """
        items_in: List[EmbeddingInfo] = []
        items_out: List[EmbeddingInfo] = []

        for item in self.items:
            ident = item.identity  # Optional[str]
            if ident in identity_ids:
                items_in.append(item)
            else:
                items_out.append(item)

        subset_in = EmbeddingDataset(
            name=f"{self.name}_subset",
            description=f"Subset de {self.name} (In)",
            items=items_in,
            loaded_metadata=self.loaded_metadata.copy()
        )

        subset_out = EmbeddingDataset(
            name=f"{self.name}_complement",
            description=f"Complemento de {self.name} (Out)",
            items=items_out,
            loaded_metadata=self.loaded_metadata.copy()
        )

        return subset_in, subset_out

    # =========================================================================
    # PROPRIEDADES MATEMÁTICAS
    # =========================================================================

    @property
    def matrix(self) -> np.ndarray:
        if not self.items:
            return np.array([])
        return np.vstack([item.embedding for item in self.items])

    @property
    def embeddings(self) -> np.ndarray:
        return self.matrix

    @property
    def identities(self) -> np.ndarray:
        return np.array([item.identity for item in self.items])
