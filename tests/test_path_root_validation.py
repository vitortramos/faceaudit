"""Validação de `path_root` via `resolve_image_path`.

Núcleo lógico da operação FIX_PATH_ROOT (janela "Modificar Caminhos"):
o candidato de subpasta é validado resolvendo uma face de exemplo e
checando existência em disco — sem mutar o dataset.
"""

from types import SimpleNamespace

from faceaudit.enums.path_mode import PathMode
from faceaudit.services.image_path_resolver import resolve_image_path


def test_resolve_relative_existing_subfolder(tmp_path):
    sub = tmp_path / "images"
    sub.mkdir()
    img = sub / "0001.png"
    img.write_bytes(b"x")

    face = SimpleNamespace(image_path="0001.png")
    resolved = resolve_image_path(
        face, parquet_dir=tmp_path, path_mode=PathMode.RELATIVE,
        path_root="images",
    )
    assert resolved.exists()
    assert resolved == img.resolve()


def test_resolve_relative_wrong_subfolder(tmp_path):
    sub = tmp_path / "images"
    sub.mkdir()
    (sub / "0001.png").write_bytes(b"x")

    face = SimpleNamespace(image_path="0001.png")
    resolved = resolve_image_path(
        face, parquet_dir=tmp_path, path_mode=PathMode.RELATIVE,
        path_root="errado",
    )
    assert not resolved.exists()


def test_resolve_relative_root_when_empty(tmp_path):
    (tmp_path / "0001.png").write_bytes(b"x")
    face = SimpleNamespace(image_path="0001.png")
    # "", ".", None são todos equivalentes à raiz da pasta do parquet.
    for root in ("", ".", None):
        resolved = resolve_image_path(
            face, parquet_dir=tmp_path, path_mode=PathMode.RELATIVE,
            path_root=root,
        )
        assert resolved.exists(), f"root={root!r} deveria resolver p/ raiz"
