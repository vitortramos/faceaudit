"""De quais faces um template agregado foi feito.

`EMBEDDING_AGGREGATION` funde N faces da mesma identidade num template. O
resultado não é uma foto, e o `face_hash` dele é um digest de MÃO ÚNICA do
conjunto ordenado de faces contribuintes: dá para verificar que um
template veio daquelas faces, nunca para recuperá-las a partir dele.

Sem registrar a lista, a proveniência do template ficava só no nome que a
agregação inventava, `AGG_<id>_k3_<digest>`, de onde se extraía no máximo
o NÚMERO de faces. O que um exame precisa é ver quais foram.

Três campos, cada um com uma razão distinta:

- `count` responde "de quantas faces é este template" mesmo quando as
  imagens não estão alcançáveis;
- `hashes` é o rastro estável. Arquivo muda de lugar, o hash não, e é por
  ele que as faces são reencontradas no dataset de origem;
- `paths` é o atalho para exibir, resolvido no momento da agregação.

Os dois últimos coexistem de propósito: o caminho pode quebrar quando o
projeto é movido, e aí o hash ainda encontra a face; e o hash depende de o
dataset de origem estar por perto, enquanto o caminho não.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

# Nomes das colunas no parquet do dataset agregado. Ficam aqui para que
# quem escreve e quem lê não possam divergir por engano de digitação.
COLUNA_COUNT = "aggregation_source_count"
COLUNA_HASHES = "aggregation_source_hashes"
COLUNA_PATHS = "aggregation_source_paths"


@dataclass(frozen=True)
class AggregationProvenance:
    """As faces que formaram um template."""

    count: int
    hashes: Tuple[str, ...] = ()
    paths: Tuple[str, ...] = ()

    @property
    def tem_imagens(self) -> bool:
        return bool(self.paths)

    def rotulo(self) -> str:
        plural = "s" if self.count != 1 else ""
        return f"Template de {self.count} face{plural}"


def build(items: Sequence[Any]) -> AggregationProvenance:
    """Proveniência a partir das faces que entraram no template.

    Caminho vazio não entra na lista: template feito de faces que já não
    tinham imagem (um template de templates, por exemplo) declara o número
    e nenhuma imagem, em vez de uma lista de strings vazias.
    """
    hashes = tuple(
        str(getattr(it, "face_hash", "") or "") for it in items
    )
    paths = tuple(
        str(getattr(it, "image_path", "") or "")
        for it in items
        if getattr(it, "image_path", None)
    )
    return AggregationProvenance(
        count=len(items), hashes=hashes, paths=paths,
    )


def from_item(item: Any) -> Optional[AggregationProvenance]:
    """Proveniência de um `EmbeddingInfo`, ou `None` se não é template.

    `None` é o que faz quem chama tratar a linha como face comum.
    """
    count = getattr(item, "aggregation_source_count", None)
    if count is None:
        return None
    try:
        count = int(count)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return AggregationProvenance(
        count=count,
        hashes=_texto(getattr(item, "aggregation_source_hashes", None)),
        paths=_texto(getattr(item, "aggregation_source_paths", None)),
    )


def from_row(row: Any) -> Optional[AggregationProvenance]:
    """Proveniência a partir de uma linha de parquet (dict-like)."""
    if row is None:
        return None
    try:
        bruto = row.get(COLUNA_COUNT)
    except AttributeError:
        return None
    if bruto is None or (isinstance(bruto, float) and bruto != bruto):
        return None
    try:
        count = int(bruto)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    return AggregationProvenance(
        count=count,
        hashes=_texto(row.get(COLUNA_HASHES)),
        paths=_texto(row.get(COLUNA_PATHS)),
    )


def _texto(bruto: Any) -> Tuple[str, ...]:
    """Lista de strings tolerante ao que o parquet devolve.

    Coluna de lista volta como `numpy.ndarray`, e `None`/NaN aparecem em
    linha de face comum no mesmo parquet.
    """
    if bruto is None:
        return ()
    if isinstance(bruto, float) and bruto != bruto:
        return ()
    try:
        return tuple(str(v) for v in bruto if v is not None)
    except TypeError:
        return ()
