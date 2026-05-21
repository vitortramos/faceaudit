"""Confirma que `FaceInfo.to_flat_dict` serializa embeddings de TODOS
os modelos do `SepaelModel` enum (não só v2 e v4 como antes do fix).
"""
from __future__ import annotations

import numpy as np

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo


def test_to_flat_dict_serializes_all_present_embeddings():
    """Caso real do bug: dataset criado com v2+v4+v6 só persistia v2/v4."""
    ff = FFInfo(
        embedding_sepaelv2=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        embedding_sepaelv4=np.array([4.0, 5.0, 6.0], dtype=np.float32),
        embedding_sepaelv6=np.array([7.0, 8.0, 9.0], dtype=np.float32),
        fiqa_score=0.85,
    )
    face = FaceInfo(image_path="x.jpg", identity="42", ff_info=ff)

    flat = face.to_flat_dict()

    # Todos os 3 modelos presentes têm coluna nomeada pela convenção
    # `parquet_column` do enum.
    assert flat["embedding_v2"] == [1.0, 2.0, 3.0]
    assert flat["embedding_v4"] == [4.0, 5.0, 6.0]
    assert flat["embedding_v6"] == [7.0, 8.0, 9.0]

    # Alias `embedding` (genérico) aponta pro v2 — retrocompat.
    assert flat["embedding"] == [1.0, 2.0, 3.0]


def test_to_flat_dict_skips_absent_embeddings():
    """Modelo sem embedding não cria coluna fantasma."""
    ff = FFInfo(
        embedding_sepaelv2=np.array([1.0, 2.0], dtype=np.float32),
    )
    face = FaceInfo(image_path="x.jpg", identity="1", ff_info=ff)

    flat = face.to_flat_dict()

    assert "embedding" in flat
    assert "embedding_v2" in flat
    assert "embedding_v3" not in flat
    assert "embedding_v4" not in flat
    assert "embedding_v5" not in flat
    assert "embedding_v6" not in flat


def test_to_flat_dict_handles_v6_only():
    """Edge case: face só com v6 (sem v2). Alias `embedding` ausente."""
    ff = FFInfo(
        embedding_sepaelv6=np.array([7.0, 8.0, 9.0], dtype=np.float32),
    )
    face = FaceInfo(image_path="x.jpg", identity="1", ff_info=ff)

    flat = face.to_flat_dict()

    assert flat["embedding_v6"] == [7.0, 8.0, 9.0]
    # Sem v2, sem alias genérico.
    assert "embedding" not in flat
    assert "embedding_v2" not in flat


def test_enum_membership_completeness():
    """Smoke check: o enum cobre todos os slots de FFInfo. Se um modelo
    novo for adicionado em FFInfo sem entrada no enum, esse teste falha
    e força a sincronização."""
    enum_slots = {member.ffinfo_slot for member in SepaelModel}
    ffinfo_slots = {
        f.name for f in FFInfo.__dataclass_fields__.values()
        if f.name.startswith("embedding_sepael")
    }
    assert enum_slots == ffinfo_slots, (
        f"SepaelModel enum tem {enum_slots} mas FFInfo tem {ffinfo_slots}. "
        "Adicione o membro faltante no enum ou o slot no FFInfo."
    )


def test_from_dataframe_loads_v6_without_extra_data_crash():
    """Regressão: parquet com `embedding_v6` causava crash em
    `from_dataframe` porque a coluna não estava em
    `_CANONICAL_FACE_COLUMNS` → caía no loop de extra_data → `pd.isna(array)`
    explodia com `truth value of an array is ambiguous`."""
    import pandas as pd
    from faceaudit.models.face_dataset import FaceDataset

    df = pd.DataFrame([{
        "image_path": "/fake/img.jpg",
        "identity": "1",
        "fiqa_score": 0.9,
        "embedding_v2": [1.0, 2.0, 3.0],
        "embedding_v6": [7.0, 8.0, 9.0],
    }])

    ds = FaceDataset.from_dataframe(df, name="test_v6_load")

    assert len(ds.faces) == 1
    face = ds.faces[0]
    assert face.ff_info is not None
    assert face.ff_info.embedding_sepaelv2 is not None
    assert face.ff_info.embedding_sepaelv2.tolist() == [1.0, 2.0, 3.0]
    assert face.ff_info.embedding_sepaelv6 is not None
    assert face.ff_info.embedding_sepaelv6.tolist() == [7.0, 8.0, 9.0]


def test_is_canonical_face_column_covers_all_sepael_versions():
    """Sanity: TODOS os parquet_columns do enum são classificados como
    canônicos, garantindo que extra_data loop não tenta processá-los."""
    from faceaudit.models.face_dataset import _is_canonical_face_column

    for member in SepaelModel:
        assert _is_canonical_face_column(member.parquet_column), (
            f"{member.parquet_column} não está reconhecida como canonical "
            f"— vai cair no extra_data loop e crashar com array."
        )


if __name__ == "__main__":
    test_to_flat_dict_serializes_all_present_embeddings()
    test_to_flat_dict_skips_absent_embeddings()
    test_to_flat_dict_handles_v6_only()
    test_enum_membership_completeness()
    test_from_dataframe_loads_v6_without_extra_data_crash()
    test_is_canonical_face_column_covers_all_sepael_versions()
    print("Todos os testes passaram.")
