# faceaudit/models/lr/pair_identity_metadata.py
"""
Metadata de identidades por par usada pelos splitters do calibrador v2.

Para garantir CV estritamente identidade-disjunta nos dois lados de
cada par, o splitter precisa conhecer probe_identity_id E
reference_identity_id de cada linha do dataset. Não basta agrupar por
um único lado: pares (id_01, id_02) e (id_03, id_02) compartilham
id_02, e um agrupamento simples por probe permitiria id_02 vazar entre
folds.

A regra correta (v2):
  1. Particionar `{IDs únicas} = unique(probe ∪ reference)` em K folds.
  2. Train fold: pares cujas DUAS identidades estão no bloco train.
  3. Test fold: pares cujas DUAS identidades estão no bloco test.
  4. Pares cruzando blocos são DESCARTADOS daquele fold.

Esse contrato é compatível com o `IdentityDisjointPairKFold` existente
(V2.10) — `PairIdentityMetadata` é apenas a forma estruturada de passar
as duas listas de identidades quando o splitter for invocado fora do
contexto FaceAudit (e.g., dentro de um `sklearn.Pipeline` na v2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PairIdentityMetadata:
    """
    Identidades de probe e reference para cada linha de um dataset de pares.

    Tamanhos `probe_identity_ids` e `reference_identity_ids` devem ser
    iguais (= N_pairs). Identidades são strings; a igualdade é
    case-sensitive — a deduplicação ocorre via `set`/`unique`.

    Em casework sem identidades conhecidas (raríssimo, pois sempre temos
    pelo menos a identidade do material de comparação curado), os campos
    podem ser tuplas vazias — o splitter da v2 não é invocado em casework.
    """

    probe_identity_ids: Tuple[str, ...]
    reference_identity_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.probe_identity_ids) != len(self.reference_identity_ids):
            raise ValueError(
                f"probe_identity_ids ({len(self.probe_identity_ids)}) e "
                f"reference_identity_ids ({len(self.reference_identity_ids)}) "
                "devem ter o mesmo comprimento"
            )

    @property
    def n_pairs(self) -> int:
        return len(self.probe_identity_ids)

    @property
    def unique_identities(self) -> frozenset:
        """União das identidades dos dois lados (sem duplicação)."""
        return frozenset(self.probe_identity_ids) | frozenset(self.reference_identity_ids)
