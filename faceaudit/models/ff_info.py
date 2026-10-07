from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any, List
import numpy as np

@dataclass
class FFInfo:
    """
    Dataclass que mapeia o output da biblioteca ForensicFace.
    Armazena métricas de qualidade, pose, demografia e embeddings.
    """
    # Métricas Físicas e Qualidade
    ipd: Optional[float] = None           # Inter-pupillary distance
    fiqa_score: Optional[float] = None    # Face Image Quality Assessment
    det_score: Optional[float] = None     # Detection confidence score
    
    # Demografia Estimada
    gender: Optional[Union[str, float]] = None
    age: Optional[float] = None
    
    # Pose (Head Pose Estimation)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    
    # Geometria (Arrays)
    # repr=False evita imprimir a matriz inteira no terminal ao debugar
    #
    # INVARIANTE: `bbox` e `keypoints` estão SEMPRE no sistema de
    # coordenadas da imagem que o dataset guarda, isto é, a de
    # `FaceInfo.image_path`. Quem muda a imagem muda a geometria junto, e
    # na CRIAÇÃO do dataset: o recorte com margem translada os dois para a
    # origem do recorte; a materialização alinhada os descarta, porque os
    # pontos no sistema do recorte 112×112 não estão disponíveis ali e
    # guardar os da foto original seria guardar outra régua.
    #
    # Sem o invariante, o mesmo par de campos significava coisas diferentes
    # por fluxo e nada no dado dizia qual. Desenhá-los sobre a imagem punha
    # os cinco pontos fora da face, e o consumidor do KPRPE, que lia este
    # slot assumindo 112×112, alimentava o modelo com a régua errada e
    # produzia embedding errado sem erro aparente.
    #
    # Os pontos no sistema 112×112, que os modelos KPRPE exigem, têm casa
    # própria: `FaceInfo.extra_data[ALIGNED_KEYPOINTS_EXTRA_KEY]`.
    bbox: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = field(default=None, repr=False)

    
    # Embeddings (Vetores de características) — um slot por modelo
    # ForensicFace. Nomeação canônica `embedding_<ff_name>`; novos modelos
    # do upstream (v6 KPRPE, ou versões futuras) ganham slot novo aqui +
    # entrada no `SepaelModel` enum em `faceaudit.enums.sepael_model`.
    embedding_sepaelv2: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_sepaelv3: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_sepaelv4: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_sepaelv5: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_sepaelv6: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Orientação em que a face foi encontrada — rotação HORÁRIA em graus
    # aplicada à imagem antes de detectar. None/ausente = a imagem como
    # está gravada (o caso normal). Existe porque foto tirada com a câmera
    # de lado só tem face detectável depois de girar, e o leitor precisa
    # saber que a pose e o recorte vieram do frame corrigido, não do
    # arquivo como ele é exibido.
    detection_rotation: Optional[int] = None

    # Resolução do detector que encontrou a face, quando diferente da
    # usada na ingestão. Vazio = encontrada na resolução configurada (o
    # caso normal). Preenchido indica face pequena no quadro, recuperada
    # numa segunda tentativa em resolução maior pedida na revisão.
    detection_det_size: Optional[int] = None

    # Imagem Processada
    aligned_face: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, serialize_numpy: bool = False, exclude_heavy: bool = True) -> dict:
        """
        Gera dicionário dos dados.
        exclude_heavy: Remove embeddings e arrays para exportação tabular leve.
        """
        data = asdict(self)
        
        # Remove campos pesados/técnicos se solicitado (para Pandas/Parquet)
        if exclude_heavy:
            keys_to_remove = [k for k in data.keys() if 'embedding' in k or k in ['keypoints', 'aligned_face', 'bbox']]
            for k in keys_to_remove:
                data.pop(k, None)
        
        if serialize_numpy:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                    
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'FFInfo':
        """
        Cria instância a partir do dicionário cru do ForensicFace.
        Garante que apenas as chaves esperadas sejam passadas.
        """
        # Filtra o dicionário de entrada para pegar apenas chaves que existem na dataclass
        # Isso evita erro de "unexpected keyword argument" se a lib atualizar no futuro
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)