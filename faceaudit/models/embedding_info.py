from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

@dataclass
class EmbeddingInfo:
    """
    Modelo simplificado contendo as informações essenciais
    para indexação, busca vetorial e pipelines de transformação.
    """
    # --- Campos Obrigatórios ---
    embedding: np.ndarray          # Vetor de características (512D)
    face_hash: str                 # Identificador único do crop
    image_path: str                # Caminho da imagem original
    identity: Optional[str]        # Label da identidade (pode ser None)
    file_name: str                 # Nome do arquivo original

    # --- Campos Opcionais ---
    source: Optional[str] = None          # Ex: "FaceAudit", "Importado"
    origin: Optional[str] = None          # Ex: "cctv", "webcam", "documento"
    face_image: Optional[np.ndarray] = None # Crop da face (imagem alinhada)
    
    # --- Metadados de Pose (Essenciais para Auditoria de Hubness) ---
    yaw: Optional[float] = 0.0
    pitch: Optional[float] = 0.0
    
    # [DEFINITIVO] Dicionário para múltiplos scores (FIQA, OFIQ, SERFIQ)
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # --- Propriedade de Conveniência (Somente Leitura) ---
    @property
    def quality_score(self) -> Optional[float]:
        """
        Retorna o 'melhor' score disponível no dicionário para uso geral.
        Prioridade: FIQA > OFIQ > SERFIQ > Genérico.
        """
        if not self.quality_scores:
            return None
        
        # Ordem de prioridade definida pela lógica de negócio
        for key in ['fiqa', 'ofiq', 'serfiq', 'generic']:
            if key in self.quality_scores:
                return self.quality_scores[key]
        
        # Se não achar os conhecidos, retorna o primeiro que tiver
        return next(iter(self.quality_scores.values()))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingInfo':
        """
        Cria uma instância a partir de um dicionário, com tratamento para tipos NumPy.
        """
        # 1. Recupera Embedding e Imagem
        emb = data.get('embedding')
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
            
        img = data.get('face_image')
        if isinstance(img, list):
            img = np.array(img, dtype=np.uint8)

        # 2. Migração de Dados de Qualidade Antigos
        q_scores = data.get('quality_scores', {})
        legacy_val = data.get('quality_score')
        if legacy_val is not None and not q_scores:
            q_scores = {'generic': float(legacy_val)}

        return cls(
            embedding=emb,
            face_hash=data.get('face_hash', ""),
            image_path=data.get('image_path', ""),
            identity=data.get('identity'),
            file_name=data.get('file_name', ""),
            source=data.get('source'),
            origin=data.get('origin'),
            face_image=img,
            yaw=float(data.get('yaw', 0.0)),
            pitch=float(data.get('pitch', 0.0)),
            quality_scores=q_scores
        )

    @property
    def size(self) -> int:
        return self.embedding.size
    
    @property
    def ndim(self) -> int:
        return self.embedding.ndim
    
    @property
    def shape(self) -> tuple:
        return self.embedding.shape

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte para dicionário serializável, transformando arrays em listas.
        """
        emb_val = self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding
        img_val = self.face_image.tolist() if isinstance(self.face_image, np.ndarray) else self.face_image

        return {
            "embedding": emb_val,
            "face_hash": self.face_hash,
            "image_path": self.image_path,
            "identity": self.identity,
            "file_name": self.file_name,
            "source": self.source,
            "origin": self.origin,
            "face_image": img_val,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "quality_scores": self.quality_scores
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])