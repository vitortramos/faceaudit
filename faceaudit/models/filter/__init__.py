"""DTOs de configuração de filtros sobre datasets faciais.

Substitui o `filter_config.py` legado (50+ fields hardcoded) por um
modelo polimórfico baseado em `ColumnFilter` (um por coluna a filtrar).
Cada filter declara o nome da coluna + tipo de dado + critério (valor,
lista, range, lista de ranges).

Uso:
    from faceaudit.models.filter import (
        FilterConfig, ColumnFilter,
        NumericColumnFilter, StringColumnFilter, BooleanColumnFilter,
        ColumnFilterDataType,
    )
"""

from faceaudit.models.filter.column_filter_data_type import (
    ColumnFilterDataType,
)
from faceaudit.models.filter.column_filter import (
    ColumnFilter,
    NumericColumnFilter,
    StringColumnFilter,
    BooleanColumnFilter,
)
from faceaudit.models.filter.filter_config import FilterConfig

__all__ = [
    "ColumnFilterDataType",
    "ColumnFilter",
    "NumericColumnFilter",
    "StringColumnFilter",
    "BooleanColumnFilter",
    "FilterConfig",
]
