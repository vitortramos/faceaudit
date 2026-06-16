# faceaudit/models/lr/lr_architecture_raw_input_payload.py
"""
Payload bruto persistido como recurso `LR_ARCHITECTURE_RAW_INPUT`.

Contém os arrays canônicos (s, r_F, r_P) que alimentam o builder, junto
com pair_keys, same_person opcional (None em casework), identidades por
lado para o splitter v2, e fingerprints de integridade.

Construir SOMENTE via `LRArchitectureRawInputPayloadFactory.create(...)`.
A factory:
  - Converte arrays para little-endian float64 + C-contiguous + read-only.
  - Calcula `pair_key_order_hash` (estrutura) e `pair_label_hash` (rótulos).
  - Calcula `raw_input_matrix_fingerprint` cobrindo schema + dtype +
    shape + bytes + pair_key_order_hash.
  - Valida obrigatoriedade dos campos por fase (PREPARATION,
    VALIDATION, CASEWORK).

A imutabilidade dos arrays é forçada via `np.ndarray.setflags(write=False)`
após `.copy()` — o `frozen=True` do dataclass não bloqueia mutação dos
arrays apontados por seus atributos; apenas a setflags faz isso.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.models.lr._canonical_hash import sha256_hex
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    LRArchitectureRawFeatureSchema,
)


@dataclass(frozen=True)
class LRArchitectureRawInputPayload:
    """
    Payload bruto + metadata. Construir via factory.

    Arrays são read-only após criação (write=False). Tentativas de
    mutação levantam `ValueError` do numpy.
    """

    matrix: np.ndarray
    schema: LRArchitectureRawFeatureSchema
    pair_keys: Tuple[Tuple[str, str], ...]
    same_person: Optional[np.ndarray]
    probe_identity_ids: Optional[Tuple[str, ...]]
    reference_identity_ids: Optional[Tuple[str, ...]]
    pair_role_orientation: PairRoleOrientation
    pair_key_mode: PairKeyMode

    pair_key_order_hash: str
    pair_label_hash: Optional[str]
    raw_input_matrix_fingerprint: str

    @property
    def n_pairs(self) -> int:
        return self.matrix.shape[0]


def _compute_pair_key_order_hash(
    pair_keys: Tuple[Tuple[str, str], ...],
) -> str:
    """
    SHA-256 sobre TODOS os pares + count.

    Formato dos bytes:
      <n_pairs>\\n
      <probe_hash>\\x00<reference_hash>\\n
      ... (uma linha por par, na ordem)
    """
    h = hashlib.sha256()
    h.update(f"{len(pair_keys)}\n".encode("utf-8"))
    for probe, reference in pair_keys:
        h.update(probe.encode("utf-8"))
        h.update(b"\x00")
        h.update(reference.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _compute_pair_label_hash(same_person: np.ndarray) -> str:
    """
    SHA-256 sobre os rótulos same_person (preserva ordem).

    Formato:
      <n>\\n<byte_por_par>...
      onde byte_por_par é '1' (True) ou '0' (False)
    """
    h = hashlib.sha256()
    h.update(f"{len(same_person)}\n".encode("utf-8"))
    h.update(b"".join(b"1" if v else b"0" for v in same_person))
    return h.hexdigest()


def _compute_raw_input_matrix_fingerprint(
    matrix: np.ndarray,
    schema: LRArchitectureRawFeatureSchema,
    pair_key_order_hash: str,
) -> str:
    """
    Fingerprint cobrindo schema_version, column_names, dtype, shape,
    bytes da matriz (em ordem C, dtype '<f8'), e pair_key_order_hash.

    Detecta qualquer alteração nos valores numéricos preservando os
    mesmos pares — proteção complementar ao pair_key_order_hash.
    """
    h = hashlib.sha256()
    h.update(schema.schema_version.encode("utf-8"))
    h.update(b"\n")
    h.update(",".join(schema.columns).encode("utf-8"))
    h.update(b"\n")
    h.update(str(matrix.dtype.str).encode("utf-8"))
    h.update(b"\n")
    h.update(",".join(str(d) for d in matrix.shape).encode("utf-8"))
    h.update(b"\n")
    h.update(matrix.tobytes(order="C"))
    h.update(b"\n")
    h.update(pair_key_order_hash.encode("utf-8"))
    return h.hexdigest()


class LRArchitectureRawInputPayloadFactory:
    """
    Construção segura de `LRArchitectureRawInputPayload`.

    Forçar uso da factory para garantir:
      - Arrays read-only.
      - Canonicalização de dtype/order.
      - Cálculo determinístico dos fingerprints.
      - Validações de obrigatoriedade por fase.
    """

    @classmethod
    def create(
        cls,
        *,
        matrix: np.ndarray,
        schema: LRArchitectureRawFeatureSchema,
        pair_keys: Tuple[Tuple[str, str], ...],
        same_person: Optional[np.ndarray] = None,
        probe_identity_ids: Optional[Tuple[str, ...]] = None,
        reference_identity_ids: Optional[Tuple[str, ...]] = None,
        pair_role_orientation: PairRoleOrientation = PairRoleOrientation.NOT_APPLICABLE,
        pair_key_mode: PairKeyMode = PairKeyMode.ORDERED,
        phase: Optional[StrategyPhase] = None,
    ) -> LRArchitectureRawInputPayload:
        """
        Cria o payload validando coerência.

        `phase` (opcional) ativa validação contextual de obrigatoriedade
        de `same_person` e `probe_identity_ids` / `reference_identity_ids`.
        Quando None, apenas validações estruturais são aplicadas.
        """
        # Canonicalização da matriz
        mat = np.ascontiguousarray(matrix, dtype="<f8").copy()
        mat.setflags(write=False)

        if mat.ndim != 2:
            raise ValueError(f"matrix deve ser 2D; recebido shape={mat.shape}")
        n_pairs = mat.shape[0]

        if mat.shape[1] != schema.n_columns:
            raise ValueError(
                f"matrix.shape[1]={mat.shape[1]} != schema.n_columns="
                f"{schema.n_columns}"
            )

        if len(pair_keys) != n_pairs:
            raise ValueError(
                f"len(pair_keys)={len(pair_keys)} != n_pairs={n_pairs}"
            )

        # same_person
        if same_person is not None:
            sp_arr = np.ascontiguousarray(same_person, dtype=np.bool_).copy()
            sp_arr.setflags(write=False)
            if sp_arr.shape != (n_pairs,):
                raise ValueError(
                    f"same_person.shape={sp_arr.shape} != ({n_pairs},)"
                )
        else:
            sp_arr = None

        # identidades
        if probe_identity_ids is not None and len(probe_identity_ids) != n_pairs:
            raise ValueError(
                f"len(probe_identity_ids)={len(probe_identity_ids)} != {n_pairs}"
            )
        if reference_identity_ids is not None and len(reference_identity_ids) != n_pairs:
            raise ValueError(
                f"len(reference_identity_ids)={len(reference_identity_ids)} != {n_pairs}"
            )

        # Hashes
        pair_key_order_hash = _compute_pair_key_order_hash(pair_keys)
        pair_label_hash = (
            _compute_pair_label_hash(sp_arr) if sp_arr is not None else None
        )
        raw_input_matrix_fingerprint = _compute_raw_input_matrix_fingerprint(
            mat, schema, pair_key_order_hash
        )

        # Validação contextual de obrigatoriedade por fase
        if phase is not None:
            cls._validate_phase_requirements(
                phase=phase,
                same_person=sp_arr,
                probe_identity_ids=probe_identity_ids,
                reference_identity_ids=reference_identity_ids,
            )

        return LRArchitectureRawInputPayload(
            matrix=mat,
            schema=schema,
            pair_keys=tuple(pair_keys),
            same_person=sp_arr,
            probe_identity_ids=(
                tuple(probe_identity_ids) if probe_identity_ids is not None else None
            ),
            reference_identity_ids=(
                tuple(reference_identity_ids) if reference_identity_ids is not None else None
            ),
            pair_role_orientation=pair_role_orientation,
            pair_key_mode=pair_key_mode,
            pair_key_order_hash=pair_key_order_hash,
            pair_label_hash=pair_label_hash,
            raw_input_matrix_fingerprint=raw_input_matrix_fingerprint,
        )

    @staticmethod
    def _validate_phase_requirements(
        *,
        phase: StrategyPhase,
        same_person: Optional[np.ndarray],
        probe_identity_ids: Optional[Tuple[str, ...]],
        reference_identity_ids: Optional[Tuple[str, ...]],
    ) -> None:
        """
        Aplica a tabela de obrigatoriedade do plano:

        - PREPARATION supervisionada: same_person + identidades obrigatórios.
        - VALIDATION supervisionada: same_person + identidades obrigatórios.
        - CASEWORK: ambos opcionais.

        Nota: A obrigatoriedade fina para PREPARATION + MANUAL (sem AUTO_LOGREG)
        é responsabilidade do executor, que conhece o `parameter_mode`. Esta
        factory aplica a regra padrão "supervisionada exige rótulos".
        """
        if phase == StrategyPhase.CASEWORK:
            return  # casework: tudo opcional

        if same_person is None:
            raise ValueError(
                f"same_person é obrigatório na fase {phase.value}. "
                "Disponível apenas casework sem ground truth."
            )
        if probe_identity_ids is None or reference_identity_ids is None:
            raise ValueError(
                f"probe_identity_ids e reference_identity_ids são obrigatórios "
                f"na fase {phase.value} para suportar CV identidade-disjunta."
            )
