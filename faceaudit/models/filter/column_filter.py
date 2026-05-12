"""DTOs polimórficos de filtro por coluna.

Cada `ColumnFilter` declara qual coluna filtrar + o critério. 3 variantes
concretas (numérico, string, booleano), cada uma com os campos que fazem
sentido pra seu tipo. `ColumnFilter.from_dict` é polimórfico — dispatcheia
pra a subclasse correta via `data_type`. Mesmo padrão usado no projeto
em `StrategyInstructionDefinition.from_dict`.

Combinação dentro de uma mesma `ColumnFilter` é OR (matched se qualquer
critério bate); combinação entre múltiplas `ColumnFilter` num
`FilterConfig` é AND (todos os filters precisam passar).

Ex numérico:
    NumericColumnFilter(
        column_name="fiqa_score",
        values=[2.50],                # exato 2.50
        ranges=[(1.0, 1.5), (2.0, 3.0)],  # OU 1.0–1.5 OU 2.0–3.0
    )
    → match se fiqa == 2.50 OR fiqa em [1.0, 1.5] OR fiqa em [2.0, 3.0]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.filter.column_filter_data_type import (
    ColumnFilterDataType,
)


T = TypeVar("T", bound="ColumnFilter")


@dataclass
class ColumnFilter(JSONSerializableMixin):
    """Base — não usar diretamente. Use Numeric/String/Boolean.

    `column_name` é o nome da coluna alvo no dataset (canônico, usado
    pra resolver via `column_specs` ou via `face.extra_data`).
    `data_type` permite o `from_dict` polimórfico saber qual subclasse
    instanciar.
    """
    column_name: str = ""
    data_type: ColumnFilterDataType = ColumnFilterDataType.STRING

    def to_dict(self) -> dict:
        # Override pra serializar o enum como string
        d = super().to_dict()
        if isinstance(d.get("data_type"), ColumnFilterDataType):
            d["data_type"] = d["data_type"].to_str()
        return d

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional["ColumnFilter"]:
        """Dispatcher polimórfico — instancia a subclasse correta com
        base em `data_type`. Defaults toleram dicts incompletos."""
        if not data:
            return None
        d = dict(data)
        raw_type = d.get("data_type")
        if isinstance(raw_type, str):
            dt = ColumnFilterDataType.from_str(raw_type)
        elif isinstance(raw_type, ColumnFilterDataType):
            dt = raw_type
        else:
            dt = ColumnFilterDataType.STRING
        d["data_type"] = dt

        if dt == ColumnFilterDataType.NUMERIC:
            return NumericColumnFilter(
                column_name=str(d.get("column_name", "")),
                data_type=dt,
                values=[float(v) for v in (d.get("values") or [])],
                ranges=[
                    (float(r[0]), float(r[1]))
                    for r in (d.get("ranges") or [])
                    if isinstance(r, (list, tuple)) and len(r) >= 2
                ],
            )
        if dt == ColumnFilterDataType.STRING:
            return StringColumnFilter(
                column_name=str(d.get("column_name", "")),
                data_type=dt,
                values=[str(v) for v in (d.get("values") or [])],
            )
        if dt == ColumnFilterDataType.BOOLEAN:
            return BooleanColumnFilter(
                column_name=str(d.get("column_name", "")),
                data_type=dt,
                value=bool(d.get("value", False)),
            )
        return None

    # ── API que cada subclasse implementa ────────────────────────────

    def is_empty(self) -> bool:
        """True quando o filter não tem critério algum (devia ser
        considerado no-op pelo service e tratado como inválido na UI)."""
        return True

    def matches(self, value: Any) -> bool:
        """True se `value` (lido da coluna do face) bate o critério.
        Subclasses sobrescrevem."""
        return False


@dataclass
class NumericColumnFilter(ColumnFilter):
    """Filtro pra colunas INT/FLOAT.

    Combina `values` (lista de igualdades) + `ranges` (lista de tuples
    (min, max), inclusivos). Match = qualquer um (OR interno).

    Ranges abertos: use float('-inf') ou float('inf') no boundary.
    Ex: `(float('-inf'), 30.0)` = "FIQA até 30".
    """
    values: List[float] = field(default_factory=list)
    ranges: List[Tuple[float, float]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.values and not self.ranges

    def matches(self, value: Any) -> bool:
        if value is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        # Igualdade exata (match com tolerância numérica)
        for target in self.values:
            if abs(v - target) < 1e-9:
                return True
        # Range inclusivo
        for lo, hi in self.ranges:
            if lo <= v <= hi:
                return True
        return False


@dataclass
class StringColumnFilter(ColumnFilter):
    """Filtro pra colunas STR.

    `values` é a lista de strings aceitas — match exato. Match = OR
    interno (any). Vazio = no-op (is_empty).

    Comparação case-sensitive intencional (identidade, hash, source são
    sensíveis); pra demográficos a UI deve normalizar antes.
    """
    values: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.values

    def matches(self, value: Any) -> bool:
        if value is None:
            return False
        v = str(value)
        return v in self.values


@dataclass
class BooleanColumnFilter(ColumnFilter):
    """Filtro pra colunas BOOL — match exato com `value`."""
    value: bool = False

    def is_empty(self) -> bool:
        # Boolean filter sempre tem valor (default False) — nunca empty
        # do ponto de vista de critério. Mas na UI o caller pode tratar
        # filtros booleanos como sempre válidos.
        return False

    def matches(self, value: Any) -> bool:
        if value is None:
            return False
        try:
            return bool(value) == self.value
        except Exception:
            return False
