"""Categoria de recurso — eixo ortogonal ao `StrategyResourceType`.

Cinde os recursos do projeto por **provenância** (de onde vieram), não
pelo tipo (o que são):

  - PRIMARY: importados pelo usuário via "Adicionar Recurso"
    (datasets `.parquet`, pipeline/split/mining configs `.json`).
  - DERIVED: produzidos pela execução de uma instrução
    (pipelines treinados, splits, scores, validation results, etc.).

Necessário porque `FaceDataset`/`EmbeddingDataset` (StrategyResourceType
DATASET) pode ser qualquer um dos dois — depende se o usuário importou
ou se uma instrução o gerou. Antes existia `INTERMEDIATE_DATASET` como
tipo separado pra marcar a versão derivada, mas isso cindia
artificialmente um único conceito (dataset de embeddings).

Usado pra:
  - filtrar o painel lateral em abas Primários/Derivados;
  - decidir se um recurso precisa ser re-persistido pelo
    `persist_produced_resource` (primários já existem em disco);
  - mostrar/esconder o botão "Adicionar Recurso".
"""

from __future__ import annotations

from enum import Enum


class ResourceCategory(str, Enum):
    PRIMARY = "PRIMARY"
    DERIVED = "DERIVED"

    @property
    def label_pt(self) -> str:
        return {
            "PRIMARY": "Primários",
            "DERIVED": "Derivados",
        }.get(self.value, self.value)

    @classmethod
    def from_str(cls, value: str) -> "ResourceCategory":
        """Reconstrução robusta com fallback DERIVED.

        Default DERIVED porque a maioria absoluta dos recursos do projeto
        é produzida por execução; primários são exceção que sempre tem o
        campo explícito no JSON (ou migração detecta pelo `resource_type`
        legado).
        """
        if not value:
            return cls.DERIVED
        try:
            return cls(value.upper())
        except ValueError:
            return cls.DERIVED
