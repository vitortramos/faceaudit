from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import os

from faceaudit.enums.embedding_type import EmbeddingType

# --- IMPORTS ---
from .ff_info import FFInfo
from .ofiq_info import OFIQInfo
from .serfiq_info import SERFIQInfo
from .embedding_info import EmbeddingInfo

# Colunas de atributo do FFInfo (nomes ANTES do rename gender→gender_est /
# age→age_est em to_flat_dict). Quando None, não são emitidas como coluna —
# datasets enxutos (sem pose/demografia) e a seleção de colunas do fluxo de
# criação (DatasetIngestionConfig.ff_kept_attributes) não carregam colunas
# todas-nulas. Consumidores já toleram colunas ausentes (o schema varia entre
# datasets legados e novos).
_FF_ATTR_COLUMNS = (
    "ipd", "fiqa_score", "det_score", "gender", "age", "yaw", "pitch", "roll",
    "detection_rotation",
)


@dataclass
class FaceInfo:
    """
    Dataclass principal que agrega todas as informações de uma face auditada.
    Atua como o 'Ground Truth' e container de metadados biométricos.
    """
    image_path: str

    # --- Identificadores do Arquivo ---
    file_name: Optional[str] = None
    hash: Optional[str] = None

    # --- Sub-modelos ---
    ff_info: Optional[FFInfo] = None
    ofiq_info: Optional[OFIQInfo] = None
    serfiq_info: Optional[SERFIQInfo] = None

    # --- Metadados ---
    origin: Optional[str] = None
    ethnicity: Optional[str] = None
    identity: Optional[str] = None
    gender: Optional[str] = None

    # --- Rastreamento de erros (criação + enriquecimento) ---
    # Marcado quando a face falhou em algum estágio da última execução
    # de ingestion/enrich. False/None quando passou em tudo. Permite
    # ao perito filtrar `error=True` na tabela pra inspecionar falhas
    # sem precisar de um parquet separado. Re-run de enriquecimento
    # com sucesso limpa essas colunas.
    error: Optional[bool] = None
    # Estágio(s) que falharam, ex: "ff" ou "ff,ofiq" (vírgula-separado
    # quando múltiplos). Mantido como string pra facilitar filtro via
    # `str.contains("ff")` na UI da tabela.
    error_stage: Optional[str] = None
    # Mensagens concatenadas em formato legível, ex:
    # "[ff] nenhuma face detectada | [ofiq] timeout".
    error_message: Optional[str] = None

    # PR 21 (Tema 4.2) — captura colunas custom adicionadas via UI ou
    # presentes em parquets com `column_specs` que não pertencem ao
    # mapeamento canônico de FaceInfo. Mantém valores escalares (str,
    # int, float, bool, None). Round-trip:
    #   from_dataframe → captura leftovers da row aqui
    #   to_flat_dict   → re-emite as chaves no row (flatten)
    # Sem isso, custom columns ficavam só no `_df` do table model e
    # eram perdidas em save/load/filter.
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Preenche automaticamente o file_name se estiver vazio.

        PR L1c.1 — normaliza `image_path` pra POSIX (forward slashes)
        em qualquer FaceInfo construído. Garante que datasets gerados
        no Windows e abertos no macOS/Linux (e vice-versa) tenham
        `image_path` consistente. `pathlib.Path` aceita `/` em todos
        os SOs, então não há perda de funcionalidade.
        """
        if self.image_path:
            self.image_path = self.image_path.replace("\\", "/")
        if self.file_name is None and self.image_path:
            self.file_name = os.path.basename(self.image_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceInfo':
        if 'image_path' not in data:
            raise ValueError("O atributo 'image_path' é obrigatório.")

        return cls(
            image_path=data['image_path'],
            file_name=data.get('file_name'),
            hash=data.get('hash'),
            ff_info=data.get('ff_info'),
            ofiq_info=data.get('ofiq_info'),
            serfiq_info=data.get('serfiq_info'),
            origin=data.get('origin', data.get('source')),
            ethnicity=data.get('ethnicity'),
            identity=data.get('identity'),
            gender=data.get('gender')
        )

    def to_dict(self) -> Dict[str, Any]:
        base_dict = {
            "image_path": self.image_path,
            "file_name": self.file_name,
            "hash": self.hash,
            "origin": self.origin,
            "ethnicity": self.ethnicity,
            "identity": self.identity,
            "gender": self.gender
        }

        for key, value in [
            ("ff_info", self.ff_info), 
            ("ofiq_info", self.ofiq_info), 
            ("serfiq_info", self.serfiq_info)
        ]:
            if value is not None and hasattr(value, "to_dict"):
                base_dict[key] = value.to_dict()
            elif value is not None and hasattr(value, "__dict__"):
                base_dict[key] = asdict(value)
            else:
                base_dict[key] = value

        return base_dict

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])

    def to_embedding(self, embedding_type: EmbeddingType = EmbeddingType.SEPAEL_V2) -> Optional[EmbeddingInfo]:
        """
        Converte os dados desta FaceInfo para um EmbeddingInfo leve.
        
        A lógica de qual vetor usar (V2, V4, SER-FIQ) é delegada ao `embedding_type`.
        
        Args:
            embedding_type (EmbeddingType): Enum que define a estratégia de extração.
                                            Default: SEPAEL_V2 (Padrão/Estável).

        Returns:
            Optional[EmbeddingInfo]: Objeto contendo o vetor e metadados, ou None se não encontrado.
        """
        # 1. Extração Inteligente via Enum (Inversão de Controle)
        # O Enum sabe onde buscar o dado (ff_info.v2, ff_info.v4, serfiq.mean, etc)
        selected_embedding = embedding_type.extract_from(self)

        # Se a face não tem o embedding solicitado, não podemos criar o EmbeddingInfo
        if selected_embedding is None:
            return None

        # 2. Extração de Qualidade (Normalizada para 0.0 - 1.0)
        # Prioridade: OFIQ Scalar -> SERFIQ Score -> 0.0
        q_score = 0.0
        
        if self.ofiq_info and self.ofiq_info.unified_quality_score_scalar is not None:
            q_score = float(self.ofiq_info.unified_quality_score_scalar) / 100.0
            
        elif self.serfiq_info and self.serfiq_info.score is not None:
            q_score = float(self.serfiq_info.score) / 100.0
        
        q_score = max(0.0, min(1.0, q_score))

        # 3. Extração do Crop (Imagem Alinhada)
        face_img = None
        # Tenta pegar do FFInfo (onde geralmente fica o crop primário)
        if self.ff_info and self.ff_info.aligned_face is not None and isinstance(self.ff_info.aligned_face, np.ndarray):
            face_img = self.ff_info.aligned_face

        # 4. Construção do Objeto
        return EmbeddingInfo(
            embedding=selected_embedding,
            face_hash=self.hash,
            image_path=self.image_path,
            identity=self.identity,
            file_name=self.file_name,
            origin=self.origin,
            face_image=face_img,

            # ✅ NOVO: dicionário rico (mantém compatibilidade com sua nova arquitetura)
            quality_scores={"generic": q_score},

            # Usa o nome amigável do Enum (ex: "Sepael V2 (Estável)")
            source=embedding_type.display_name
        )
    
    
    def to_flat_dict(self, include_embeddings: bool = True) -> Dict[str, Any]:
        """
        Achata a hierarquia para salvar em Parquet.
        Atualizado para persistir múltiplos embeddings se existirem.

        `include_embeddings=False` pula a serialização dos vetores (o
        `emb.tolist()` de ~512 floats por face/modelo) — usado pelo caminho
        de EXIBIÇÃO da tabela, que nunca mostra o embedding e não deve pagar
        o custo de RAM/tempo em datasets de milhões. Persistência
        (`to_parquet`) usa o default `True`.
        """
        row = {}
        
        # 1. FFInfo (Dados Algorítmicos)
        if self.ff_info:
            # exclude_heavy=True removeu imagens e embeddings
            ff_data = self.ff_info.to_dict(exclude_heavy=True)

            # Não emite colunas de atributo FF vazias (None). Sem isso, todo
            # dataset com ff_info carregaria as 8 colunas mesmo todas-nulas
            # (asdict emite todos os campos), e a seleção de colunas do fluxo
            # de criação (que zera os atributos não pedidos) não removeria a
            # coluna do parquet.
            for _attr in _FF_ATTR_COLUMNS:
                if ff_data.get(_attr) is None:
                    ff_data.pop(_attr, None)

            # --- REINJEÇÃO DE EMBEDDINGS ---
            # Itera o SepaelModel enum em vez de hard-codear nomes — assim
            # v3/v5/v6 (e qualquer modelo futuro) entram automaticamente
            # quando adicionados ao enum.
            #
            # NÃO escrevemos a coluna `embedding` (sem sufixo) — ela era
            # alias histórico do v2 nos parquets pré-multi-modelo. Custava
            # ~2KB por face em duplicação inútil (512 floats × 4 bytes
            # × N faces). O loader em `face_dataset.py:344` continua
            # aceitando `embedding` como fallback pra carregar parquets
            # antigos, então a remoção da escrita é puramente economia
            # de disco — sem quebra de retrocompat.
            if include_embeddings:
                from faceaudit.enums.sepael_model import SepaelModel
                for member in SepaelModel:
                    emb = getattr(self.ff_info, member.ffinfo_slot, None)
                    if emb is None:
                        continue
                    ff_data[member.parquet_column] = emb.tolist()
            
            # Tratamento de conflitos de nome
            if 'gender' in ff_data:
                ff_data['gender_est'] = ff_data.pop('gender')
            if 'age' in ff_data:
                ff_data['age_est'] = ff_data.pop('age')
                
            row.update(ff_data)

        # 2. OFIQInfo
        if self.ofiq_info:
            row['ofiq_yaw'] = self.ofiq_info.head_pose_yaw_raw
            row['ofiq_pitch'] = self.ofiq_info.head_pose_pitch_raw
            row['ofiq_roll'] = self.ofiq_info.head_pose_roll_raw
            row['ofiq_score'] = self.ofiq_info.unified_quality_score_scalar
            
            for k, v in self.ofiq_info.to_dict().items():
                if k not in ['head_pose_yaw_raw', 'head_pose_pitch_raw', 'head_pose_roll_raw', 'unified_quality_score_scalar']:
                      row[f"ofiq_{k}"] = v

        # 3. SERFIQInfo
        if self.serfiq_info:
            serfiq_data = self.serfiq_info.to_dict()
            
            # Se tiver mean_embedding, salva também (pulado no caminho de
            # exibição — include_embeddings=False).
            if include_embeddings and getattr(self.serfiq_info, 'mean_embedding', None) is not None:
                 serfiq_data['embedding_serfiq'] = self.serfiq_info.mean_embedding.tolist()

            row.update({f"serfiq_{k}": v for k, v in serfiq_data.items()})

        # 4. Dados Manuais (Metadados)
        row.update({
            'image_path': self.image_path,
            'file_name': self.file_name,
            'hash': self.hash,
            'origin': self.origin,
            'ethnicity': self.ethnicity,
            'identity': self.identity,
            'gender': self.gender
        })

        # 4b. Erro (rastreamento de criação/enriquecimento). Só emite
        # as colunas quando algum dos 3 está populado — preserva
        # schema enxuto pra datasets que nunca passaram por essa
        # marcação. Cada coluna entra independente pra que a UI possa
        # filtrar individualmente.
        if (
            self.error is not None
            or self.error_stage is not None
            or self.error_message is not None
        ):
            row['error'] = self.error
            row['error_stage'] = self.error_stage
            row['error_message'] = self.error_message

        # 5. Custom columns capturadas via UI (PR 21). Aplicadas por último
        # para que sobrescrevam qualquer chave canônica eventual com mesmo
        # nome — política: usuário sempre vence quando explicitamente edita.
        if self.extra_data:
            row.update(self.extra_data)

        return row