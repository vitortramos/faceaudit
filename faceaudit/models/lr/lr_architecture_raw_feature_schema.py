# faceaudit/models/lr/lr_architecture_raw_feature_schema.py
"""
Schema descritivo das colunas brutas (s, r_F, r_P) que alimentam a
construção da matriz de features.

Persistido junto ao `LRArchitectureRawInputPayload`. Permite que o
`ArchitectureVectorizerTransformer` da v2 (sklearn-compat) reconstrua o
contexto sem precisar consultar o `LRArchitectureSpec` original — basta
ler o schema e identificar quais colunas correspondem a base/rarity_probe/
rarity_reference dentro da matriz X_raw recebida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from faceaudit.models.lr._canonical_hash import hash_canonical_payload


@dataclass(frozen=True)
class LRArchitectureRawFeatureSchema:
    """
    Descreve as colunas de `LRArchitectureRawInputPayload.matrix`.

    `columns` é a tupla ordenada de identificadores. Para cada papel
    (base, rarity_probe, rarity_reference), `*_column` indica o índice
    correspondente em `columns`, ou `None` quando a arquitetura não
    consome aquele papel.

    `schema_version` é bumpada quando os identificadores das colunas ou
    a semântica do layout mudam.
    """

    schema_version: str
    columns: Tuple[str, ...]
    base_score_column: int
    rarity_probe_column: Optional[int]
    rarity_reference_column: Optional[int]

    def __post_init__(self) -> None:
        if self.base_score_column < 0 or self.base_score_column >= len(self.columns):
            raise ValueError(
                f"base_score_column={self.base_score_column} fora de range "
                f"[0, {len(self.columns)})"
            )
        if self.rarity_probe_column is not None:
            if self.rarity_probe_column < 0 or self.rarity_probe_column >= len(self.columns):
                raise ValueError(
                    f"rarity_probe_column={self.rarity_probe_column} fora de range"
                )
        if self.rarity_reference_column is not None:
            if self.rarity_reference_column < 0 or self.rarity_reference_column >= len(self.columns):
                raise ValueError(
                    f"rarity_reference_column={self.rarity_reference_column} fora de range"
                )
        # Detecta duplicatas de papel
        indices = [self.base_score_column]
        if self.rarity_probe_column is not None:
            indices.append(self.rarity_probe_column)
        if self.rarity_reference_column is not None:
            indices.append(self.rarity_reference_column)
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"Colunas de papéis duplicadas: {indices} — cada papel "
                "deve referenciar uma coluna distinta"
            )

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def fingerprint(self) -> str:
        """SHA-256 do schema. Bumpar `schema_version` muda o fingerprint."""
        payload = {
            "schema_version": self.schema_version,
            "columns": list(self.columns),
            "base_score_column": self.base_score_column,
            "rarity_probe_column": self.rarity_probe_column,
            "rarity_reference_column": self.rarity_reference_column,
        }
        return hash_canonical_payload(payload)


# Schemas canônicos por composição

def schema_for_base_only() -> LRArchitectureRawFeatureSchema:
    """Schema da arquitetura COSINE_ONLY/HINGE_ONLY/TWO_SIDED_HINGE_ONLY."""
    return LRArchitectureRawFeatureSchema(
        schema_version="1.0",
        columns=("base_score",),
        base_score_column=0,
        rarity_probe_column=None,
        rarity_reference_column=None,
    )


def schema_for_base_and_rarities() -> LRArchitectureRawFeatureSchema:
    """Schema das arquiteturas que consomem r_F e r_P."""
    return LRArchitectureRawFeatureSchema(
        schema_version="1.0",
        columns=("base_score", "rarity_probe", "rarity_reference"),
        base_score_column=0,
        rarity_probe_column=1,
        rarity_reference_column=2,
    )
