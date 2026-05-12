"""Service de aplicação de `FilterConfig` sobre `FaceDataset`.

Stateless. Aceita o config + dataset, retorna FaceDataset filtrado
(novo objeto — não muta o original). Combina filters por AND.

Também expõe `validate(dataset, config)` que retorna a lista de
filtros inválidos (coluna não existe no dataset corrente) — consumido
pela UI pra marcar essas linhas em vermelho ao abrir um filtro salvo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.filter.column_filter import ColumnFilter
from faceaudit.models.filter.filter_config import FilterConfig


# Mapeamento de coluna canônica → atributo do FaceInfo. Usado quando
# `column_specs` está vazio (parquet legado) ou pra colunas que não
# vivem em `extra_data`. Mantém pequeno e focado nos canônicos — qualquer
# outra coisa cai no `face.extra_data` automaticamente.
_FACEINFO_ATTR_MAP = {
    "identity": "identity",
    "image_path": "image_path",
    "file_name": "file_name",
    "hash": "hash",
    "face_hash": "hash",
    "source": "source",
    "origin": "origin",
    "gender": "gender",
    "ethnicity": "ethnicity",
    "age_est": "age_est",
}

# Atributos aninhados em ff_info / ofiq_info / serfiq_info — keys do
# FaceInfo via getattr. Lookup hierárquico em `_resolve_value`.
_NESTED_PATHS = {
    "fiqa_score": ("ff_info", "fiqa_score"),
    "ipd": ("ff_info", "ipd"),
    "det_score": ("ff_info", "det_score"),
    "ff_yaw": ("ff_info", "yaw"),
    "ff_pitch": ("ff_info", "pitch"),
    "ff_roll": ("ff_info", "roll"),
    "yaw": ("ff_info", "yaw"),
    "pitch": ("ff_info", "pitch"),
    "roll": ("ff_info", "roll"),
    "ofiq_score": ("ofiq_info", "unified_quality_score_scalar"),
    "ofiq_yaw": ("ofiq_info", "yaw"),
    "ofiq_pitch": ("ofiq_info", "pitch"),
    "ofiq_roll": ("ofiq_info", "roll"),
    "serfiq_score": ("serfiq_info", "score"),
}


@dataclass(frozen=True)
class InvalidFilter:
    """Razão pela qual um filter individual é inválido contra o
    dataset corrente. Usado pela UI pra marcar visualmente."""
    filter_index: int
    column_name: str
    reason: str


class DatasetFilterService:
    """Stateless — métodos podem ser chamados sem instância."""

    @classmethod
    def apply(
        cls,
        dataset: FaceDataset,
        config: FilterConfig,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> FaceDataset:
        """Aplica o filter ao dataset, retorna novo `FaceDataset` com
        as faces que passam em TODOS os filters não-vazios e válidos.

        Filters inválidos (coluna ausente) são silenciosamente ignorados
        — caller deve usar `validate(...)` antes pra avisar o usuário.

        `progress_callback(pct)` é chamado a cada 5% de avanço.
        """
        # Preserva os filters não-empty + cuja coluna existe no dataset.
        invalid = {
            inv.filter_index for inv in cls.validate(dataset, config)
        }
        active: List[ColumnFilter] = []
        for i, f in enumerate(config.filters):
            if i in invalid:
                continue
            if f.is_empty():
                continue
            active.append(f)

        if not active:
            # Nada a filtrar — retorna cópia rasa do dataset original
            # pra consistência semântica (caller pode swap sem cuidado
            # de aliasing).
            return cls._clone_face_dataset(dataset, list(dataset.faces))

        kept: List = []
        n = len(dataset.faces)
        last_pct = -1
        for i, face in enumerate(dataset.faces):
            if all(cls._face_matches_filter(face, f, dataset) for f in active):
                kept.append(face)
            if progress_callback is not None and n > 0:
                pct = int(100 * (i + 1) / n)
                if pct != last_pct and pct % 5 == 0:
                    progress_callback(pct)
                    last_pct = pct

        return cls._clone_face_dataset(dataset, kept)

    @classmethod
    def validate(
        cls, dataset: FaceDataset, config: FilterConfig,
    ) -> List[InvalidFilter]:
        """Lista filters cuja coluna não existe no dataset — UI usa
        pra marcar em vermelho. Empty filters NÃO são reportados como
        inválidos (UI trata separadamente)."""
        out: List[InvalidFilter] = []
        # Coleta os nomes de coluna disponíveis: column_specs + atributos
        # canônicos do FaceInfo + paths aninhados conhecidos.
        available = cls._available_column_names(dataset)
        for i, f in enumerate(config.filters):
            if not f.column_name:
                out.append(InvalidFilter(
                    i, "", "Filtro sem coluna definida",
                ))
                continue
            if f.column_name not in available:
                out.append(InvalidFilter(
                    i, f.column_name,
                    f"Coluna '{f.column_name}' não existe no dataset atual",
                ))
        return out

    # ── Internals ────────────────────────────────────────────────────

    @staticmethod
    def _available_column_names(dataset: FaceDataset) -> set:
        """Conjunto de nomes de coluna válidos pra filtrar nesse
        dataset. Inclui column_specs + canônicos + aninhados conhecidos
        + custom columns do face.extra_data."""
        names = set()
        # column_specs (formato novo)
        if dataset.column_specs:
            names.update(dataset.column_specs.keys())
        # Canônicos sempre válidos quando há ao menos uma face
        if dataset.faces:
            names.update(_FACEINFO_ATTR_MAP.keys())
            names.update(_NESTED_PATHS.keys())
            # Custom columns vivem em face.extra_data
            for face in dataset.faces[:5]:  # samplea poucos pra performance
                extra = getattr(face, "extra_data", None)
                if isinstance(extra, dict):
                    names.update(extra.keys())
        return names

    @classmethod
    def _face_matches_filter(
        cls, face: Any, f: ColumnFilter, dataset: FaceDataset,
    ) -> bool:
        """Resolve o valor da coluna no face e delega ao
        `f.matches(value)`."""
        value = cls._resolve_value(face, f.column_name)
        return f.matches(value)

    @staticmethod
    def _resolve_value(face: Any, column_name: str) -> Any:
        """Pega o valor de `column_name` no face. Tenta nessa ordem:
          1. Atributo direto via `_FACEINFO_ATTR_MAP`
          2. Path aninhado via `_NESTED_PATHS` (ff_info.fiqa_score etc.)
          3. `face.extra_data[column_name]` (custom columns)
          4. getattr direto (último recurso)
        """
        attr = _FACEINFO_ATTR_MAP.get(column_name)
        if attr is not None:
            return getattr(face, attr, None)

        path = _NESTED_PATHS.get(column_name)
        if path is not None:
            obj = face
            for step in path:
                if obj is None:
                    return None
                obj = getattr(obj, step, None)
            return obj

        extra = getattr(face, "extra_data", None)
        if isinstance(extra, dict) and column_name in extra:
            return extra[column_name]

        # Fallback: atributo direto com mesmo nome
        return getattr(face, column_name, None)

    @staticmethod
    def _clone_face_dataset(
        dataset: FaceDataset, faces: list,
    ) -> FaceDataset:
        """Retorna FaceDataset com lista de faces nova preservando
        nome/descrição/specs/path_mode/is_aligned do original. Não copia
        faces (compartilha referências) — semântica de "view" filtrada."""
        new_ds = FaceDataset(
            name=dataset.name or "Filtrado",
            description=dataset.description,
        )
        new_ds.faces = faces
        # Preserva metadados de schema / paths / alinhamento. Sem
        # `is_aligned` aqui, qualquer filtro descartava silenciosamente
        # o flag de "dataset alinhado" — o status bar passava a mostrar
        # "fotos originais" mesmo num dataset que era alinhado.
        new_ds.column_specs = dict(dataset.column_specs or {})
        new_ds.path_mode = dataset.path_mode
        new_ds.path_root = dataset.path_root
        new_ds._parquet_dir = getattr(dataset, "_parquet_dir", None)
        new_ds.is_aligned = getattr(dataset, "is_aligned", False)
        return new_ds
