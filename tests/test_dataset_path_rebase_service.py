"""Salvar Como noutra pasta não pode quebrar os caminhos das imagens.

Caso real: dataset em `datasets/` com `path_root=<pai>_images` salvo como
`datasets/Confronto 2/RGMA-9-14.parquet`. O `path_root` herdado passou a
apontar para `datasets/Confronto 2/<pai>_images/`, que não existe, e a
importação com cópia de imagens gravou um dataset vazio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.path_rebase_action import PathRebaseAction
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.dataset_path_rebase_service import rebase_dataset_paths


def _dataset(path_root, names, *, parquet_dir=None, mode=PathMode.RELATIVE):
    ds = FaceDataset(
        name="ds",
        faces=[FaceInfo(image_path=n, file_name=n) for n in names],
        path_mode=mode,
        path_root=path_root,
    )
    if parquet_dir is not None:
        ds._parquet_dir = parquet_dir
    return ds


@pytest.fixture
def tree(tmp_path):
    """`datasets/<pai>_images/a.png` + subpasta `datasets/Confronto 2/`."""
    source = tmp_path / "datasets"
    images = source / "pai_images"
    images.mkdir(parents=True)
    (images / "a.png").write_bytes(b"x")
    (images / "b.png").write_bytes(b"y")
    target = source / "Confronto 2"
    target.mkdir()
    return source, images, target


def test_rebases_path_root_when_saving_into_subfolder(tree):
    source, images, target = tree
    ds = _dataset("pai_images", ["a.png", "b.png"], parquet_dir=source)

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=target,
    )

    assert result.action is PathRebaseAction.REBASED
    assert ds.path_root == "../pai_images"
    assert ds.path_mode is PathMode.RELATIVE
    # O ponteiro novo resolve para os mesmos arquivos de antes.
    assert ds.resolve_image_path(ds.faces[0]) == images / "a.png"
    assert ds.resolve_image_path(ds.faces[0]).exists()


def test_rebases_when_saving_into_parent_folder(tree):
    source, images, target = tree
    ds = _dataset("pai_images", ["a.png"], parquet_dir=source)

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=source.parent,
    )

    assert result.action is PathRebaseAction.REBASED
    assert ds.path_root == "datasets/pai_images"
    assert ds.resolve_image_path(ds.faces[0]).exists()


def test_same_folder_is_a_noop(tree):
    source, _, _ = tree
    ds = _dataset("pai_images", ["a.png"], parquet_dir=source)

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=source,
    )

    assert result.action is PathRebaseAction.NOT_NEEDED
    assert not result.changed
    assert ds.path_root == "pai_images"


def test_absolute_dataset_is_untouched(tree):
    source, images, target = tree
    abs_path = str(images / "a.png")
    ds = _dataset(None, [abs_path], parquet_dir=source, mode=PathMode.ABSOLUTE)

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=target,
    )

    assert result.action is PathRebaseAction.NOT_NEEDED
    # FaceInfo normaliza "\" → "/"; o que importa é apontar pro mesmo arquivo.
    assert Path(ds.faces[0].image_path) == Path(abs_path)


def test_unknown_source_folder_keeps_paths(tree):
    _, _, target = tree
    ds = _dataset("pai_images", ["a.png"])

    result = rebase_dataset_paths(
        ds, source_parquet_dir=None, target_parquet_dir=target,
    )

    assert result.action is PathRebaseAction.NOT_NEEDED
    assert ds.path_root == "pai_images"


def test_already_broken_source_is_reported_as_unverified(tree):
    source, _, target = tree
    ds = _dataset("pasta_que_nao_existe", ["a.png"], parquet_dir=source)

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=target,
    )

    assert result.action is PathRebaseAction.REBASED_UNVERIFIED
    assert ds.path_root == "../pasta_que_nao_existe"
    assert "já não eram encontradas" in result.message


def test_absolutizes_when_no_relative_path_exists(tree, monkeypatch):
    source, images, target = tree
    ds = _dataset("pai_images", ["a.png", "b.png"], parquet_dir=source)

    # Simula destino noutro drive (Windows): relpath levanta ValueError.
    import faceaudit.services.dataset_path_rebase_service as mod
    monkeypatch.setattr(
        mod.os.path, "relpath",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("outro drive")),
    )

    result = rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=target,
    )

    assert result.action is PathRebaseAction.ABSOLUTIZED
    assert ds.path_mode is PathMode.ABSOLUTE
    assert ds.path_root is None
    assert Path(ds.faces[0].image_path) == images / "a.png"
    assert ds.resolve_image_path(ds.faces[0]).exists()


def test_round_trip_through_parquet_keeps_images_resolvable(tree):
    """Fim a fim: grava na subpasta, relê e as imagens continuam achadas."""
    source, images, target = tree
    ds = _dataset("pai_images", ["a.png", "b.png"], parquet_dir=source)
    for face in ds.faces:
        face.embedding = np.zeros(4, dtype=np.float32)

    rebase_dataset_paths(
        ds, source_parquet_dir=source, target_parquet_dir=target,
    )
    destino = target / "derivado.parquet"
    ds.to_parquet(destino)

    relido = FaceDataset.from_parquet(str(destino))
    assert relido.path_root == "../pai_images"
    assert relido.resolve_image_path(relido.faces[0]) == images / "a.png"
    assert all(relido.resolve_image_path(f).exists() for f in relido.faces)
