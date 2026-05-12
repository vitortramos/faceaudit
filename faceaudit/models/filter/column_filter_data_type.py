"""Enum do tipo de dado de uma coluna filtrável.

Mapeia os 6 `ColumnDType` do schema (STR/INT/FLOAT/BOOL/VECTOR/LIST)
para os 3 tipos efetivamente filtráveis aqui:
  - NUMERIC: int + float (range, valor, lista de ranges, lista de valores)
  - STRING: str (valor, lista de valores)
  - BOOLEAN: bool (valor)

VECTOR e LIST não têm representação direta em filtro escalar e são
filtrados out do dropdown de coluna na UI.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class ColumnFilterDataType(str, Enum):
    """Tipo de dado da coluna alvo do filtro."""
    NUMERIC = "numeric"   # cobre INT e FLOAT
    STRING = "string"     # STR
    BOOLEAN = "boolean"   # BOOL

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "ColumnFilterDataType":
        try:
            return cls(value)
        except ValueError:
            return cls.STRING

    @classmethod
    def from_column_dtype(
        cls, dtype_value: str,
    ) -> Optional["ColumnFilterDataType"]:
        """Mapeia `ColumnDType` (do column_catalog) → este enum.

        Retorna None pra dtypes não-filtráveis (vector, list) — caller
        usa pra filtrar o dropdown de coluna na UI.
        """
        mapping = {
            "int": cls.NUMERIC,
            "float": cls.NUMERIC,
            "str": cls.STRING,
            "bool": cls.BOOLEAN,
        }
        return mapping.get(str(dtype_value).lower())
