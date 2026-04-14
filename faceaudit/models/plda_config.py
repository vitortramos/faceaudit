from typing import Optional, Type, TypeVar, List
from dataclasses import dataclass, field
import uuid # Opcional, mas boa prática se o manager não gerar um
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.lr_model_type import LRModelType
from faceaudit.enums.plda_training_strategy import PLDATrainingStrategy
from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.parameters.plda_parameters import PLDAParameters
from faceaudit.models.project_file import ProjectFile

T = TypeVar("T", bound="PLDAConfig")

@dataclass
class PLDAConfig(JSONSerializableMixin):
    """
    Configuração completa de um modelo PLDA específico.
    Une a estratégia de treinamento, os dados e os hiperparâmetros.
    Herdar de JSONSerializableMixin permite a persistência recursiva via Workspace.
    """
    
    # Identificador Único Imutável (A "Chave Primária" da estratégia)
    id: str = field(default="")

    # Identificador do Tipo
    type: LRModelType = LRModelType.PLDA_STRATEGY

    # --- ESTRATÉGIA ---
    strategy: PLDATrainingStrategy = field(
        default=PLDATrainingStrategy.STANDARD,
        metadata={"help": "Define o modo de operação (Standard, Adaptation, etc)."}
    )
    
    # --- FONTE DO VETOR ---
    embedding_type: EmbeddingType = field(
        default=EmbeddingType.SEPAEL_V2,
        metadata={"help": "Define qual extrator de características será utilizado."}
    )

    # --- PARÂMETROS MATEMÁTICOS ---
    parameters: PLDAParameters = field(default_factory=PLDAParameters)

    # --- REGRAS LÓGICAS (Atalhos em String para os ComboBoxes da UI) ---
    source_dataset_name: str = field(default="")
    target_dataset_name: str = field(default="")
    source_pipeline_name: str = field(default="Whitening_L2")
    target_pipeline_name: str = field(default="Whitening_L2")
    
    # [NOVO] Referências de texto conectando a UI aos JSONs globais
    mining_config_name: str = field(default="")
    split_config_name: str = field(default="")

    # --- MANIFESTO FÍSICO (Auditoria e Cadeia de Custódia) ---
    # Guarda os DTOs ProjectFile com o caminho real e Hash (Gerados no momento do Treino)
    project_files: List[ProjectFile] = field(default_factory=list)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o PLDAConfig garantindo a tipagem dos Enums, do DTO de parâmetros
        e da lista de arquivos rastreados.
        """
        if not data:
            return None

        # Uso do from_str para resiliência no carregamento
        if 'type' in data:
            data['type'] = LRModelType.from_str(data['type'])
            
        if 'strategy' in data:
            data['strategy'] = PLDATrainingStrategy.from_str(data['strategy'])
            
        if 'embedding_type' in data:
            data['embedding_type'] = EmbeddingType.from_str(data['embedding_type'])
            
        if 'parameters' in data and isinstance(data['parameters'], dict):
            # Delegação para PLDAParameters que revisamos na pasta parameters
            data['parameters'] = PLDAParameters.from_dict(data['parameters'])
            
        # Reconstrói a lista de arquivos de auditoria (se existir no JSON)
        if 'project_files' in data and isinstance(data['project_files'], list):
            data['project_files'] = [ProjectFile.from_dict(f) for f in data['project_files']]
            
        return cls(**data)