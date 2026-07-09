"""Regressão da perf-de-milhões (Gerenciador de Datasets):

O df de EXIBIÇÃO da tabela é metadata-only (`to_pandas(include_embeddings=False)`)
para não segurar gigabytes de vetores na RAM nem congelar a GUI. Como o
`sync_changes_from_table` reconstrói o dataset a partir DESSE df antes de salvar,
os embeddings precisam ser RE-ENXERTADOS das faces atuais (fonte de verdade) por
`image_path` — senão salvar após uma edição gravaria um parquet SEM embeddings.

Estes testes travam:
  1. `to_flat_dict(include_embeddings=False)` não emite colunas de vetor.
  2. `to_pandas(include_embeddings=False)` é metadata-only; `True` mantém tudo.
  3. O round-trip editar → (sync) → salvar preserva a edição E os embeddings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo


def test_to_flat_dict_excludes_embeddings_when_requested():
    ff = FFInfo(
        embedding_sepaelv2=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        embedding_sepaelv6=np.array([7.0, 8.0, 9.0], dtype=np.float32),
        fiqa_score=0.85,
    )
    face = FaceInfo(image_path="x.jpg", identity="42", ff_info=ff)

    full = face.to_flat_dict()  # default include_embeddings=True
    lean = face.to_flat_dict(include_embeddings=False)

    assert "embedding_v2" in full and "embedding_v6" in full
    assert "embedding_v2" not in lean and "embedding_v6" not in lean
    # Metadados algorítmicos leves permanecem em ambos.
    assert lean["fiqa_score"] == 0.85
    assert lean["identity"] == "42"


def test_to_pandas_display_is_metadata_only():
    emb_col = SepaelModel.SEPAEL_V2.parquet_column
    df_full = pd.DataFrame([
        {"image_path": "a/0.png", "identity": "A", "fiqa_score": 0.9,
         emb_col: np.arange(8, dtype=np.float32).tolist()},
        {"image_path": "a/1.png", "identity": "B", "fiqa_score": 0.8,
         emb_col: (np.arange(8, dtype=np.float32) + 100).tolist()},
    ])
    ds = FaceDataset.from_dataframe(df_full, name="t")

    display_df = ds.to_pandas(include_embeddings=False)
    save_df = ds.to_pandas(include_embeddings=True)

    # Exibição: sem NENHUMA coluna de embedding do enum.
    for member in SepaelModel:
        assert member.parquet_column not in display_df.columns
    assert "identity" in display_df.columns
    assert "fiqa_score" in display_df.columns

    # Persistência (default): o embedding volta.
    assert emb_col in save_df.columns


def test_display_roundtrip_regrafts_embeddings_after_edit():
    """Reproduz a lógica de `sync_changes_from_table`: reconstrói do df
    metadata-only e re-enxerta o enriquecimento pesado por image_path."""
    m = SepaelModel.SEPAEL_V2
    emb0 = np.arange(512, dtype=np.float32)
    emb1 = np.arange(512, dtype=np.float32) + 1000.0
    df_full = pd.DataFrame([
        {"image_path": "a/0.png", "identity": "ID_A", "fiqa_score": 0.9,
         m.parquet_column: emb0.tolist()},
        {"image_path": "a/1.png", "identity": "ID_B", "fiqa_score": 0.8,
         m.parquet_column: emb1.tolist()},
    ])
    ds = FaceDataset.from_dataframe(df_full, name="t")
    orig_emb0 = getattr(ds.faces[0].ff_info, m.ffinfo_slot)

    # Usuário edita a identidade na tabela (muta o df metadata-only).
    display_df = ds.to_pandas(include_embeddings=False)
    display_df.loc[0, "identity"] = "ID_A_EDITADO"

    # sync: reconstrói (sem embedding) + re-enxerta por image_path.
    heavy_by_path = {f.image_path: f for f in ds.faces}
    rebuilt = FaceDataset.from_dataframe(display_df, name="t")
    assert getattr(rebuilt.faces[0].ff_info, m.ffinfo_slot, None) is None, \
        "pré-re-enxerto o embedding deve estar ausente (df metadata-only)"
    for face in rebuilt.faces:
        orig = heavy_by_path.get(face.image_path)
        if orig is not None:
            face.ff_info = orig.ff_info
            face.ofiq_info = orig.ofiq_info
            face.serfiq_info = orig.serfiq_info

    # Edição preservada e embedding restaurado idêntico.
    assert rebuilt.faces[0].identity == "ID_A_EDITADO"
    assert rebuilt.faces[1].identity == "ID_B"
    assert np.array_equal(getattr(rebuilt.faces[0].ff_info, m.ffinfo_slot), orig_emb0)

    # SAVE path reemite o embedding correto + persiste a edição.
    save_df = rebuilt.to_pandas(include_embeddings=True)
    row0 = save_df.loc[save_df["image_path"] == "a/0.png"].iloc[0]
    assert np.array_equal(np.array(row0[m.parquet_column], dtype=np.float32), orig_emb0)
    assert row0["identity"] == "ID_A_EDITADO"


def test_compute_from_dataset_preserves_embedding_coverage():
    """A otimização de cobertura (df metadata-only + reinject) deve produzir a
    MESMA cobertura de embedding que o caminho antigo (que materializava os
    vetores), inclusive `embedding_dim` — do qual regras como Fisher dependem."""
    from faceaudit.services.dataset_stats_service import DatasetStatsService

    m = SepaelModel.SEPAEL_V2
    D = 16
    df = pd.DataFrame([
        {"image_path": "a/0.png", "identity": "A", "fiqa_score": 0.9, m.parquet_column: list(range(D))},
        {"image_path": "a/1.png", "identity": "B", "fiqa_score": 0.7, m.parquet_column: list(range(D))},
        {"image_path": "a/2.png", "identity": "C", "fiqa_score": 0.5, m.parquet_column: None},  # sem vetor
    ])
    ds = FaceDataset.from_dataframe(df, name="t")

    stats = DatasetStatsService().compute_from_dataset(ds)

    # Cobertura da coluna de embedding reinjetada: 2 de 3 faces têm o vetor.
    cs = stats.column_coverage.get(m.parquet_column)
    assert cs is not None, "coverage da coluna de embedding não foi reinjetada"
    assert cs.dim == D
    assert cs.non_null_count == 2
    assert cs.total_count == 3

    # Derivados restaurados (Fisher & cia dependem de embedding_dim > 0).
    assert stats.embedding_dim == D
    assert m.parquet_column in stats.embedding_columns

    # Cobertura escalar segue idêntica (df metadata-only tem as colunas escalares).
    fq = stats.column_coverage.get("fiqa_score")
    assert fq is not None and fq.non_null_count == 3


if __name__ == "__main__":
    test_to_flat_dict_excludes_embeddings_when_requested()
    test_to_pandas_display_is_metadata_only()
    test_display_roundtrip_regrafts_embeddings_after_edit()
    test_compute_from_dataset_preserves_embedding_coverage()
    print("Todos os testes passaram.")
