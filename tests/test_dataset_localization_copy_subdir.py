"""`copy_subdir` do DatasetLocalizationService — subpasta de imagens POR
dataset (`<nome>_images`) para não misturar imagens quando vários datasets
autocontidos vão para a mesma pasta, preservando o default `"images"` (de que
selar/importar projeto dependem)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.dataset_localization_service import (
    DatasetLocalizationService,
    dataset_images_subdir,
)


def _ds_with_images(tmp_path, name, n=2):
    faces = []
    for i in range(n):
        p = tmp_path / "src" / name / f"img_{i}.png"
        imwrite_unicode_safe(p, np.full((8, 8, 3), 10 + i, np.uint8))
        faces.append(FaceInfo(image_path=str(p), identity="id"))
    return FaceDataset(name=name, faces=faces, path_mode=PathMode.ABSOLUTE)


def test_dataset_images_subdir_sanitizes():
    assert dataset_images_subdir("meu ds") == "meu_ds_images"
    assert dataset_images_subdir("a/b:c") == "a_b_c_images"
    assert dataset_images_subdir("   ") == "dataset_images"   # vazio → fallback


def test_copy_subdir_defaults_to_images(tmp_path):
    # Sem copy_subdir: comportamento legado (selar/importar projeto dependem).
    ds = _ds_with_images(tmp_path, "x")
    res = DatasetLocalizationService().localize(
        ds, tmp_path / "out", mode=LocalizationMode.COPY_ORIGINALS)
    assert res.dataset.path_root == "images"
    assert (tmp_path / "out" / "images").is_dir()


def test_copy_subdir_per_dataset_avoids_collision(tmp_path):
    target = tmp_path / "out"
    svc = DatasetLocalizationService()
    for nm in ("alpha", "beta"):
        ds = _ds_with_images(tmp_path, nm)
        res = svc.localize(ds, target, mode=LocalizationMode.COPY_ORIGINALS,
                           copy_subdir=dataset_images_subdir(nm))
        assert res.dataset.path_root == f"{nm}_images"
    assert not (target / "images").exists()                  # nada na pasta comum
    assert len(list((target / "alpha_images").glob("*.png"))) == 2
    assert len(list((target / "beta_images").glob("*.png"))) == 2
