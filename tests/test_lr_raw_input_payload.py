"""
Testes da `LRArchitectureRawInputPayloadFactory`.

Cobre:
- Canonicalização de dtype (`<f8`) e ordem (`C`).
- Arrays read-only após criação (write=False).
- `pair_key_order_hash` cobre TODOS os pares + same_person ausente
  produz `pair_label_hash=None`.
- `raw_input_matrix_fingerprint` muda quando qualquer valor numérico
  muda (proteção complementar ao pair_key_order_hash).
- Obrigatoriedade por fase: VALIDATION exige same_person + identidades;
  CASEWORK aceita ambos ausentes.
- Schema mismatch detectado.
- `PairIdentityMetadata` valida tamanhos.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.pair_key_mode import PairKeyMode
from faceaudit.enums.pair_role_orientation import PairRoleOrientation
from faceaudit.enums.strategy_phase import StrategyPhase
from faceaudit.models.lr.lr_architecture_raw_feature_schema import (
    LRArchitectureRawFeatureSchema,
    schema_for_base_and_rarities,
    schema_for_base_only,
)
from faceaudit.models.lr.lr_architecture_raw_input_payload import (
    LRArchitectureRawInputPayloadFactory,
)
from faceaudit.models.lr.pair_identity_metadata import PairIdentityMetadata


def _make_basic_inputs():
    matrix = np.array([
        [0.35, 0.4, 0.6],
        [0.25, 0.3, 0.5],
        [-0.1, 0.2, 0.4],
        [0.55, 0.5, 0.7],
    ], dtype=np.float64)
    pair_keys = (("a", "b"), ("a", "c"), ("d", "e"), ("d", "b"))
    same_person = np.array([True, False, False, True])
    return matrix, pair_keys, same_person


class TestCanonicalization:
    def test_dtype_normalized_to_little_endian_f8(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        # Entrada em float32 — factory deve canonicalizar para <f8
        matrix32 = matrix.astype(np.float32)
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix32, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        # Confirma dtype canônico
        assert payload.matrix.dtype.str == "<f8"


class TestImmutability:
    def test_matrix_is_read_only(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        with pytest.raises(ValueError):
            payload.matrix[0, 0] = 999.0

    def test_same_person_is_read_only(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        with pytest.raises(ValueError):
            payload.same_person[0] = False

    def test_input_arrays_can_be_modified_after_construction(self):
        """Mutação do array fonte NÃO afeta o payload (copy real)."""
        matrix, pair_keys, sp = _make_basic_inputs()
        original_matrix = matrix.copy()
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        matrix[0, 0] = 0.0  # mutação no array fonte
        # Payload preserva valor original
        assert payload.matrix[0, 0] == original_matrix[0, 0]


class TestHashes:
    def test_pair_key_order_hash_deterministic(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        p1 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        p2 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        assert p1.pair_key_order_hash == p2.pair_key_order_hash
        assert p1.pair_label_hash == p2.pair_label_hash
        assert p1.raw_input_matrix_fingerprint == p2.raw_input_matrix_fingerprint

    def test_pair_key_order_hash_changes_on_reorder(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        p1 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        # Reorder linha 0 ↔ 1 — order_hash deve mudar
        matrix2 = matrix[[1, 0, 2, 3]].copy()
        pair_keys2 = (pair_keys[1], pair_keys[0], pair_keys[2], pair_keys[3])
        sp2 = sp[[1, 0, 2, 3]].copy()
        p2 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix2, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys2, same_person=sp2,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        assert p1.pair_key_order_hash != p2.pair_key_order_hash

    def test_raw_input_matrix_fingerprint_changes_on_value_change(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        p1 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        matrix_v2 = matrix.copy()
        matrix_v2[0, 0] = 0.351  # mudança mínima
        p2 = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix_v2, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys, same_person=sp,
            probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
            phase=StrategyPhase.PREPARATION,
        )
        # pair_key_order_hash idêntico (mesmos pares), matrix_fp diferente
        assert p1.pair_key_order_hash == p2.pair_key_order_hash
        assert p1.raw_input_matrix_fingerprint != p2.raw_input_matrix_fingerprint


class TestPhaseRequirements:
    def test_casework_allows_no_labels(self):
        matrix, pair_keys, _ = _make_basic_inputs()
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys,
            # sem same_person, sem identidades
            phase=StrategyPhase.CASEWORK,
        )
        assert payload.same_person is None
        assert payload.pair_label_hash is None
        assert payload.probe_identity_ids is None

    def test_validation_requires_same_person(self):
        matrix, pair_keys, _ = _make_basic_inputs()
        with pytest.raises(ValueError, match="same_person é obrigatório"):
            LRArchitectureRawInputPayloadFactory.create(
                matrix=matrix, schema=schema_for_base_and_rarities(),
                pair_keys=pair_keys,
                # sem same_person
                probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
                phase=StrategyPhase.VALIDATION,
            )

    def test_preparation_requires_identities(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        with pytest.raises(ValueError, match="identity_ids são obrigatórios"):
            LRArchitectureRawInputPayloadFactory.create(
                matrix=matrix, schema=schema_for_base_and_rarities(),
                pair_keys=pair_keys, same_person=sp,
                # sem identidades
                phase=StrategyPhase.PREPARATION,
            )

    def test_no_phase_skips_phase_validation(self):
        """`phase=None` permite construir payload mínimo."""
        matrix, pair_keys, _ = _make_basic_inputs()
        payload = LRArchitectureRawInputPayloadFactory.create(
            matrix=matrix, schema=schema_for_base_and_rarities(),
            pair_keys=pair_keys,
        )
        assert payload.same_person is None


class TestSchemaMismatch:
    def test_wrong_n_columns_raises(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        with pytest.raises(ValueError, match="schema.n_columns"):
            LRArchitectureRawInputPayloadFactory.create(
                matrix=matrix,  # 3 cols
                schema=schema_for_base_only(),  # 1 col
                pair_keys=pair_keys, same_person=sp,
                probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
                phase=StrategyPhase.PREPARATION,
            )

    def test_wrong_pair_keys_length_raises(self):
        matrix, pair_keys, sp = _make_basic_inputs()
        with pytest.raises(ValueError, match="pair_keys"):
            LRArchitectureRawInputPayloadFactory.create(
                matrix=matrix, schema=schema_for_base_and_rarities(),
                pair_keys=pair_keys[:2],  # tamanho errado
                same_person=sp,
                probe_identity_ids=("A",) * 4, reference_identity_ids=("B",) * 4,
                phase=StrategyPhase.PREPARATION,
            )


class TestSchemaValidation:
    def test_duplicate_column_index_raises(self):
        with pytest.raises(ValueError, match="duplicadas"):
            LRArchitectureRawFeatureSchema(
                schema_version="1.0",
                columns=("a", "b", "c"),
                base_score_column=0,
                rarity_probe_column=0,  # duplicado!
                rarity_reference_column=2,
            )

    def test_out_of_range_column_raises(self):
        with pytest.raises(ValueError, match="fora de range"):
            LRArchitectureRawFeatureSchema(
                schema_version="1.0",
                columns=("a", "b"),
                base_score_column=5,  # fora
                rarity_probe_column=None,
                rarity_reference_column=None,
            )


class TestPairIdentityMetadata:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="mesmo comprimento"):
            PairIdentityMetadata(
                probe_identity_ids=("a",),
                reference_identity_ids=("b", "c"),
            )

    def test_unique_identities_union(self):
        pim = PairIdentityMetadata(
            probe_identity_ids=("id_01", "id_02", "id_03"),
            reference_identity_ids=("id_04", "id_02", "id_05"),
        )
        assert pim.n_pairs == 3
        assert pim.unique_identities == frozenset(
            {"id_01", "id_02", "id_03", "id_04", "id_05"}
        )
