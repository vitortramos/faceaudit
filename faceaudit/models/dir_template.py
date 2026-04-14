from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
import json

T = TypeVar("T", bound="DirTemplate")

@dataclass
class DirTemplate(JSONSerializableMixin):
    """
    Define as regras para extração de metadados a partir da estrutura de diretórios
    e nomenclatura de arquivos de um dataset.
    """
    name: str
    description: Optional[str] = None
    
    separator: str = "_"
    use_separator: bool = False
    
    # Mapeamento por nível de pasta. Nível 0 é a primeira subpasta.
    folder_map: Dict[int, List[str]] = field(default_factory=dict)
    filename_map: Dict[int, str] = field(default_factory=dict)
    identity_from_parent: bool = False

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Garante a tipagem correta das chaves (de str para int) independente do JSON."""
        if not data:
            return None
            
        data_copy = data.copy()
        
        if 'folder_map' in data_copy and isinstance(data_copy['folder_map'], dict):
            data_copy['folder_map'] = {int(k): v for k, v in data_copy['folder_map'].items()}
            
        if 'filename_map' in data_copy and isinstance(data_copy['filename_map'], dict):
            data_copy['filename_map'] = {int(k): v for k, v in data_copy['filename_map'].items()}
            
        return cls(**data_copy)
        
    @classmethod
    def from_json(cls: Type[T], file_path: str) -> T:
        """Sobrescreve o Mixin para forçar a passagem pelo nosso from_dict seguro."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)