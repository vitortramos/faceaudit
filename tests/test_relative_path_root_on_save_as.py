"""Salvar um dataset RELATIVE em outra pasta não pode quebrar as imagens.

Caso real: um dataset em `datasets/X.parquet` com `path_root=X_images`
foi filtrado e salvo via "Salvar como" em `datasets/Confronto 2/Y.parquet`.
O `path_root` foi copiado literal, mas ele é relativo à pasta DO PARQUET
— então passou a apontar para `datasets/Confronto 2/X_images/`, que não
existe. As 300 faces continuaram listadas e só falharam ao abrir a
imagem, muito depois.

A regra agora: se a âncora não resolve a partir do destino mas resolve a
partir da origem, `to_parquet` absolutiza os caminhos e declara ABSOLUTE.
Quem prepara o destino (fluxo autocontido, que copia as imagens para
junto do parquet) passa direto e continua RELATIVE.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo


def _meta(path: Path) -> dict:
    raw = pq.ParquetFile(path).schema_arrow.metadata or {}
    return {k.decode(): v.decode("utf-8") for k, v in raw.items()}


def _dataset_with_images(tmp_path: Path, n: int = 3):
    """Dataset RELATIVE em `origem/ds.parquet` + `origem/ds_images/`."""
    source_dir = tmp_path / "origem"
    images_dir = source_dir / "ds_images"
    images_dir.mkdir(parents=True)

    faces = []
    for i in range(n):
        name = f"face_{i}.png"
        (images_dir / name).write_bytes(b"png")
        faces.append(FaceInfo(image_path=name, identity=f"P{i:03d}"))

    ds = FaceDataset(name="ds", faces=faces)
    ds.path_mode = PathMode.RELATIVE
    ds.path_root = "ds_images"
    ds._parquet_dir = source_dir.resolve()
    return ds, source_dir, images_dir


def test_save_to_other_folder_absolutizes_paths(tmp_path):
    """Destino sem a pasta de imagens → absolutiza contra a origem."""
    ds, source_dir, images_dir = _dataset_with_images(tmp_path)

    dest = source_dir / "Confronto 2" / "filtrado.parquet"
    ds.to_parquet(dest)

    meta = _meta(dest)
    assert meta["faceaudit:path_mode"] == PathMode.ABSOLUTE.value
    assert "faceaudit:path_root" not in meta, (
        "path_root herdado não pode sobreviver em modo ABSOLUTE"
    )

    reloaded = FaceDataset.from_parquet(dest)
    assert len(reloaded) == len(ds)
    for face in reloaded.faces:
        resolved = reloaded.resolve_image_path(face)
        assert resolved.exists(), f"imagem não resolve após salvar: {resolved}"
        assert Path(resolved).parent == images_dir.resolve()


def test_save_in_same_folder_keeps_relative(tmp_path):
    """Mesma pasta → a âncora continua válida; nada muda."""
    ds, source_dir, _ = _dataset_with_images(tmp_path)

    dest = source_dir / "outro_nome.parquet"
    ds.to_parquet(dest)

    meta = _meta(dest)
    assert meta["faceaudit:path_mode"] == PathMode.RELATIVE.value
    assert meta["faceaudit:path_root"] == "ds_images"

    reloaded = FaceDataset.from_parquet(dest)
    for face in reloaded.faces:
        assert reloaded.resolve_image_path(face).exists()


def test_self_contained_destination_stays_relative(tmp_path):
    """Quem copia as imagens para o destino mantém RELATIVE.

    Reproduz o fluxo autocontido: as imagens já estão em
    `<destino>/images/` e o dataset declara `path_root=images`.
    """
    ds, _, _ = _dataset_with_images(tmp_path)

    dest_dir = tmp_path / "autocontido"
    new_images = dest_dir / "images"
    new_images.mkdir(parents=True)
    for face in ds.faces:
        (new_images / face.image_path).write_bytes(b"png")
    ds.path_root = "images"

    dest = dest_dir / "ds.parquet"
    ds.to_parquet(dest)

    meta = _meta(dest)
    assert meta["faceaudit:path_mode"] == PathMode.RELATIVE.value
    assert meta["faceaudit:path_root"] == "images"

    reloaded = FaceDataset.from_parquet(dest)
    for face in reloaded.faces:
        assert reloaded.resolve_image_path(face).exists()


def test_absolute_dataset_is_untouched(tmp_path):
    """Dataset já ABSOLUTE não é reprocessado."""
    images_dir = tmp_path / "fotos"
    images_dir.mkdir()
    img = images_dir / "a.png"
    img.write_bytes(b"png")

    ds = FaceDataset(name="abs", faces=[FaceInfo(image_path=str(img), identity="P1")])
    dest = tmp_path / "saida" / "abs.parquet"
    ds.to_parquet(dest)

    meta = _meta(dest)
    assert meta["faceaudit:path_mode"] == PathMode.ABSOLUTE.value
    reloaded = FaceDataset.from_parquet(dest)
    assert reloaded.resolve_image_path(reloaded.faces[0]).exists()


def test_broken_source_anchor_is_preserved_not_rewritten(tmp_path):
    """Origem também quebrada → não inventa caminho, preserva o declarado."""
    ds = FaceDataset(name="ds", faces=[FaceInfo(image_path="x.png", identity="P1")])
    ds.path_mode = PathMode.RELATIVE
    ds.path_root = "sumida"
    ds._parquet_dir = (tmp_path / "origem").resolve()

    dest = tmp_path / "destino" / "ds.parquet"
    ds.to_parquet(dest)

    meta = _meta(dest)
    assert meta["faceaudit:path_mode"] == PathMode.RELATIVE.value
    assert meta["faceaudit:path_root"] == "sumida"
