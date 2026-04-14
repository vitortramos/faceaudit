# faceaudit/models/kde_config.py
from typing import Optional, Type, TypeVar, List
from dataclasses import dataclass, field

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.lr_model_type import LRModelType
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.parameters.kde_parameters import KDEParameters
from faceaudit.models.project_file import ProjectFile

# --- INJEÇÃO DO NOVO DTO ---
from faceaudit.models.aggregation_config import AggregationConfig

T = TypeVar("T", bound="KDEConfig")

@dataclass
class KDEConfig(JSONSerializableMixin):
    """
    Configuração completa de um modelo KDE específico.
    Une a estratégia de seleção de pares, os dados, agregação e os hiperparâmetros.
    """
    
    # Identificador Único Imutável (A "Chave Primária" da estratégia)
    id: str = field(default="")

    # Identificador do Tipo
    type: LRModelType = LRModelType.KDE_STRATEGY

    # --- FONTE DO VETOR ---
    embedding_type: EmbeddingType = field(
        default=EmbeddingType.SEPAEL_V2,
        metadata={"help": "Define qual extrator de características será utilizado."}
    )

    # --- AGREGAÇÃO DE EMBEDDINGS (Smart Enrollment) ---
    # Se None, o pipeline de preparação do KDE não fará agregação.
    aggregation: Optional[AggregationConfig] = None

    # --- PARÂMETROS MATEMÁTICOS ---
    parameters: KDEParameters = field(default_factory=KDEParameters)

    # --- REGRAS LÓGICAS (Treinamento/Calibração) ---
    probe_dataset_name: str = field(default="")
    reference_dataset_name: str = field(default="")
    
    # --- REGRAS DE VALIDAÇÃO (Teste de Generalização) ---
    probe_val_dataset_name: str = field(default="")
    reference_val_dataset_name: str = field(default="")
    
    # --- MANIFESTO FÍSICO (Auditoria e Cadeia de Custódia) ---
    project_files: List[ProjectFile] = field(default_factory=list)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o KDEConfig garantindo a tipagem dos Enums e dos DTOs.
        """
        if not data:
            return None
            
        # 1. Infla Enums de primeiro nível
        if 'type' in data:
            data['type'] = LRModelType.from_str(data['type'])
            
        if 'embedding_type' in data:
            data['embedding_type'] = EmbeddingType.from_str(data['embedding_type'])
            
        # 2. Delegamos a reconstrução do DTO de Agregação (Se Existir)
        if 'aggregation' in data and data['aggregation'] is not None:
            data['aggregation'] = AggregationConfig.from_dict(data['aggregation'])
                     
        # 3. Delegamos a reconstrução recursiva para o KDEParameters limpo
        if 'parameters' in data and isinstance(data['parameters'], dict):
            data['parameters'] = KDEParameters.from_dict(data['parameters'])
            
        # 4. Reconstrói a lista de arquivos rastreados
        if 'project_files' in data and isinstance(data['project_files'], list):
            data['project_files'] = [ProjectFile.from_dict(f) for f in data['project_files']]
            
        return cls(**data)