"""DTO raiz da configuração de filtros — uma lista de `ColumnFilter`.

Substitui o `FilterConfig` legado (50+ fields hardcoded). Combinação
entre `filters` é AND (todos precisam passar). Salvo como JSON na
biblioteca global (`GlobalLibraryFolder.FILTER_CONFIGS`) ou no projeto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.filter.column_filter import ColumnFilter


T = TypeVar("T", bound="FilterConfig")


@dataclass
class FilterConfig(JSONSerializableMixin):
    """Configuração de filtragem nomeada e salvável.

    `filters` é a lista de `ColumnFilter` aplicados em conjunção
    (AND). Cada `ColumnFilter` filtra UMA coluna; pra filtrar a mesma
    coluna por critérios alternativos (OR), use `values`/`ranges`
    múltiplos no mesmo filter.
    """
    name: str = "Novo Filtro"
    description: Optional[str] = None
    filters: List[ColumnFilter] = field(default_factory=list)

    def to_dict(self) -> dict:
        # Serialização explícita pra delegar aos `to_dict` polimórficos
        # dos filtros — `asdict` do JSONSerializableMixin não preserva
        # a discriminação de subclasse.
        return {
            "name": self.name,
            "description": self.description,
            "filters": [f.to_dict() for f in self.filters],
        }

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None
        filters_raw = data.get("filters") or []
        filters: List[ColumnFilter] = []
        for fd in filters_raw:
            if isinstance(fd, dict):
                cf = ColumnFilter.from_dict(fd)
                if cf is not None:
                    filters.append(cf)
            elif isinstance(fd, ColumnFilter):
                filters.append(fd)
        return cls(
            name=str(data.get("name") or "Novo Filtro"),
            description=data.get("description"),
            filters=filters,
        )
