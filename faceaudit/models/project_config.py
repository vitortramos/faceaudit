from dataclasses import dataclass, field
from typing import List, Type, TypeVar, Optional
from datetime import datetime

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.strategy_entry import StrategyEntry
from faceaudit.models.validation_config import ValidationConfig  # <--- NOVO IMPORT

T = TypeVar("T", bound="ProjectConfig")

@dataclass
class ProjectConfig(JSONSerializableMixin):
    """
    DTO Principal para persistência de projetos FaceAudit (Formato .fap).
    Atua como o nó raiz da árvore de configuração do projeto autocontido.
    
    Este arquivo fica na raiz da pasta do projeto e contém referências (hashes e 
    caminhos relativos) para os datasets, pipelines e modelos treinados armazenados 
    nas subpastas internas.
    """
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    strategies: List[StrategyEntry] = field(default_factory=list)
    validation_config: Optional[ValidationConfig] = None  # <--- NOVO ATRIBUTO
    version: str = "2.0.0" # Bump de versão refletindo a arquitetura autocontida

    def update_timestamp(self) -> None:
        """Atualiza o campo updated_at indicando a última modificação."""
        self.updated_at = datetime.now().isoformat()

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o ProjectConfig e todas as suas estratégias aninhadas.
        A delegação para StrategyEntry resolve o polimorfismo (PLDA/KDE).
        """
        if not data:
            return None
            
        # Processa a lista de estratégias com segurança
        if 'strategies' in data and isinstance(data['strategies'], list):
            data['strategies'] = [
                StrategyEntry.from_dict(s) if isinstance(s, dict) else s 
                for s in data['strategies']
            ]
            
        # Reconstrói o DTO da Arena de Validação (se existir no JSON)
        if 'validation_config' in data and isinstance(data['validation_config'], dict):
            data['validation_config'] = ValidationConfig.from_dict(data['validation_config'])
        
        return cls(**data)