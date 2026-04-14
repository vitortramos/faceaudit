# faceaudit/models/project/face_audit_project_metadata.py

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="FaceAuditProjectMetadata")


@dataclass
class FaceAuditProjectMetadata(JSONSerializableMixin):
    """
    Metadados de um projeto FaceAudit — conteúdo do arquivo .fa.

    O arquivo .fa é a âncora do projeto: identifica a pasta como um projeto
    FaceAudit e carrega apenas os metadados descritivos. Estratégias, datasets
    e artefatos são descobertos pelos repositórios varrendo as subpastas.

    Formato em disco: JSON com indentação (UTF-8).
    """

    name: str = field(
        metadata={"help": "Nome legível do projeto, definido pelo usuário."}
    )
    project_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={"help": "Identificador único do projeto (UUID gerado na criação)."},
    )
    description: str = field(
        default="",
        metadata={"help": "Descrição opcional do projeto."},
    )
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        metadata={"help": "Data/hora de criação do projeto (ISO 8601)."},
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(),
        metadata={"help": "Data/hora da última modificação dos metadados (ISO 8601)."},
    )
    version: str = field(
        default="1.0",
        metadata={"help": "Versão do schema do arquivo .fa."},
    )

    # =========================================================================
    # MUTAÇÕES
    # =========================================================================

    def update_timestamp(self) -> None:
        """Atualiza updated_at para o momento atual."""
        self.updated_at = datetime.now().isoformat()

    def rename(self, new_name: str) -> None:
        """
        Altera o nome do projeto e atualiza o timestamp.

        Raises:
            ValueError: Se new_name for vazio.
        """
        if not new_name or not new_name.strip():
            raise ValueError("O nome do projeto não pode ser vazio.")
        self.name = new_name.strip()
        self.update_timestamp()

    def set_description(self, description: str) -> None:
        """Altera a descrição do projeto e atualiza o timestamp."""
        self.description = description
        self.update_timestamp()

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói os metadados a partir de um dicionário JSON.

        Campos desconhecidos são ignorados para garantir compatibilidade
        com versões futuras do schema.
        """
        if not data:
            return None

        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
