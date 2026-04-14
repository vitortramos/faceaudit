from dataclasses import dataclass, asdict
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

    def __post_init__(self):
        """Preenche automaticamente o file_name se estiver vazio."""
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
    
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Achata a hierarquia para salvar em Parquet.
        Atualizado para persistir múltiplos embeddings se existirem.
        """
        row = {}
        
        # 1. FFInfo (Dados Algorítmicos)
        if self.ff_info:
            # exclude_heavy=True removeu imagens e embeddings
            ff_data = self.ff_info.to_dict(exclude_heavy=True)
            
            # --- REINJEÇÃO DE EMBEDDINGS (V2 e V4) ---
            # Salvamos todos que existirem para garantir backup completo no Parquet
            
            if self.ff_info.embedding_sepaelv2 is not None:
                # 'embedding' genérico mantém retrocompatibilidade
                ff_data['embedding'] = self.ff_info.embedding_sepaelv2.tolist()
                # 'embedding_v2' explícito para clareza futura
                ff_data['embedding_v2'] = self.ff_info.embedding_sepaelv2.tolist()
                
            if self.ff_info.embedding_sepaelv4 is not None:
                ff_data['embedding_v4'] = self.ff_info.embedding_sepaelv4.tolist()
            
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
            
            # Se tiver mean_embedding, salva também
            if getattr(self.serfiq_info, 'mean_embedding', None) is not None:
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

        return row