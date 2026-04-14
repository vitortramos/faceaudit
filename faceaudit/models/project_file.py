from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Any
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.project_file_type import ProjectFileType

T = TypeVar("T", bound="ProjectFile")

@dataclass
class ProjectFile(JSONSerializableMixin):
    """
    DTO que representa um arquivo físico rastreado pelo sistema (O Lacre Digital).
    Mantém o caminho relativo e o hash de integridade forense.
    """
    file_type: ProjectFileType
    path: str
    hash: Optional[str] = None

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> Optional[T]:
        """
        Reconstrói o objeto, garantindo que a string do JSON 
        seja convertida de volta para a instância do Enum ProjectFileType.
        """
        if not data:
            return None

        # Fazemos uma cópia para não alterar o dicionário original em memória
        init_data = data.copy()

        if 'file_type' in init_data and isinstance(init_data['file_type'], str):
            init_data['file_type'] = ProjectFileType.from_str(init_data['file_type'])

        return cls(**init_data)