"""Fix A — apply-in-place no `sync_changes_from_table`.

Editar célula / adicionar coluna custom na tabela deve aplicar direto nas
faces existentes (metadados + extra_data), SEM reconstruir o dataset via
`from_dataframe` (que fazia iterrows sobre N faces, ~35s em 40k), e SEM tocar
nos embeddings. Testa `_apply_table_edits_inplace` isoladamente (o método só
usa `self.dataset`, então chamamos como função não-ligada com um stub — evita
instanciar o QObject inteiro).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.face_dataset import FaceDataset
from interface.tabs.dataset_manager.dataset_manager_controller import (
    DatasetManagerController,
)


def _apply(ds: FaceDataset, df_edit: pd.DataFrame) -> None:
    class _Stub:
        pass
    stub = _Stub()
    stub.dataset = ds
    DatasetManagerController._apply_table_edits_inplace(stub, df_edit)


def test_apply_inplace_applies_edits_custom_column_and_keeps_embeddings():
    m = SepaelModel.SEPAEL_V2
    D = 32
    df_src = pd.DataFrame([
        {"image_path": "a/0.png", "identity": "ID_A", "gender": "M",
         "fiqa_score": 0.9, m.parquet_column: list(range(D))},
        {"image_path": "a/1.png", "identity": "ID_B", "gender": "F",
         "fiqa_score": 0.8, m.parquet_column: list(range(D))},
    ])
    ds = FaceDataset.from_dataframe(df_src, name="t")
    orig_emb0 = getattr(ds.faces[0].ff_info, m.ffinfo_slot).copy()

    # df de exibição (metadata-only) + edições do usuário na tabela:
    df_edit = ds.to_pandas(include_embeddings=False)
    df_edit.loc[df_edit["image_path"] == "a/0.png", "identity"] = "ID_A_EDIT"
    df_edit["minha_col"] = ["v0", "v1"]  # coluna custom recém-adicionada

    _apply(ds, df_edit)

    by_path = {f.image_path: f for f in ds.faces}
    f0, f1 = by_path["a/0.png"], by_path["a/1.png"]

    # Edição de identidade aplicada; a outra face intacta.
    assert f0.identity == "ID_A_EDIT"
    assert f1.identity == "ID_B"
    # Coluna custom → extra_data.
    assert f0.extra_data.get("minha_col") == "v0"
    assert f1.extra_data.get("minha_col") == "v1"
    # gender NÃO editado → preservado.
    assert f0.gender == "M"
    # Embedding preservado (apply-in-place nunca toca no enriquecimento pesado).
    assert getattr(f0.ff_info, m.ffinfo_slot) is not None
    assert np.array_equal(getattr(f0.ff_info, m.ffinfo_slot), orig_emb0)


def test_apply_inplace_is_robust_to_sorted_dataframe():
    """A junção é por image_path — deve estar correta mesmo se a tabela foi
    ordenada (ordem do df != ordem das faces)."""
    m = SepaelModel.SEPAEL_V2
    df_src = pd.DataFrame([
        {"image_path": "a/0.png", "identity": "AAA", "fiqa_score": 0.9, m.parquet_column: [1.0, 2.0]},
        {"image_path": "a/1.png", "identity": "BBB", "fiqa_score": 0.8, m.parquet_column: [3.0, 4.0]},
    ])
    ds = FaceDataset.from_dataframe(df_src, name="t")

    df_edit = ds.to_pandas(include_embeddings=False)
    # Inverte a ordem das linhas (simula sort) e edita a identidade de a/1.png.
    df_edit = df_edit.iloc[::-1].reset_index(drop=True)
    df_edit.loc[df_edit["image_path"] == "a/1.png", "identity"] = "BBB_EDIT"

    _apply(ds, df_edit)

    by_path = {f.image_path: f for f in ds.faces}
    assert by_path["a/1.png"].identity == "BBB_EDIT"
    assert by_path["a/0.png"].identity == "AAA"  # não editada, apesar da ordem trocada


if __name__ == "__main__":
    test_apply_inplace_applies_edits_custom_column_and_keeps_embeddings()
    test_apply_inplace_is_robust_to_sorted_dataframe()
    print("Todos os testes passaram.")
