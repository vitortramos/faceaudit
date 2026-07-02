"""Janela "Enriquecer Dataset" redesenhada (por tipo de coluna).

Cobre status por coluna (presente/parcial/ausente + auto-check), o mapeamento
de get_config para o backend (ff_kept_attributes/use_gpu/batch/is_aligned), o
"some se alinhada" e o fallback de ff_models quando só atributos são pedidos.
Offscreen, sem GPU/ForensicFace.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.models.ff_info import FFInfo  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _face(**ffkw):
    return FaceInfo(image_path="x.jpg", identity="1", ff_info=FFInfo(**ffkw))


def _ds(faces, *, aligned=False, name="d"):
    return SimpleNamespace(faces=faces, name=name, is_aligned=aligned)


def _ctrl(ds, *, use_gpu=False, batch_size=32):
    from interface.tabs.dataset_manager.enrich_dataset_dialog.enrich_dataset_controller import (
        EnrichDatasetController,
    )
    return EnrichDatasetController(ds, parent=None, ff_available=True, ofiq_available=True,
                                   default_use_gpu=use_gpu, default_batch_size=batch_size)


def test_status_per_column(app):
    emb = np.ones(4, np.float32)
    f1 = _face(embedding_sepaelv2=emb.copy(), yaw=1.0, det_score=0.9, fiqa_score=0.8)
    f2 = _face(embedding_sepaelv2=emb.copy(), det_score=0.9)   # sem yaw/fiqa/gênero; det presente
    v = _ctrl(_ds([f1, f2])).view
    assert v._status_labels["emb:sepaelv2"].text() == "presente"    # 2/2
    assert v._status_labels["attr:yaw"].text().startswith("parcial")   # yaw 1/2
    assert v._status_labels["fiqa"].text().startswith("parcial")    # 1/2
    assert v._status_labels["attr:det_score"].text() == "presente"  # det 2/2 (por item!)
    assert v._status_labels["attr:gender"].text() == "ausente"      # gênero 0/2
    # auto-check POR item: só PARCIAL é pré-marcado (completa); presente e
    # totalmente ausente ficam desmarcados.
    assert not v._ff_model_checks["sepaelv2"].isChecked()   # presente
    assert not v.chk_det_conf.isChecked()                    # presente
    assert v.chk_yaw.isChecked()                             # parcial → marca
    assert not v.chk_gender.isChecked()                      # ausente → NÃO marca (opt-in)


def test_get_config_maps_to_backend(app):
    f1 = _face(embedding_sepaelv2=np.ones(4, np.float32), yaw=1.0)
    c = _ctrl(_ds([f1]), use_gpu=True, batch_size=16)
    v = c.view
    for chk in (v.chk_yaw, v.chk_pitch, v.chk_roll, v.chk_det_conf, v.chk_ipd, v.chk_age):
        chk.setChecked(False)
    v.chk_gender.setChecked(True)
    v.chk_fiqa.setChecked(True)
    cfg = c.get_config()
    assert cfg.compute_ff is True
    assert set(cfg.ff_kept_attributes) == {"gender", "fiqa"}
    assert cfg.ff_extended is True                 # gender/fiqa fazem parte do bundle
    assert cfg.use_gpu is True and cfg.batch_size == 16
    assert cfg.compute_serfiq is False and cfg.is_aligned is False


def test_hide_if_aligned(app):
    f1 = _face(embedding_sepaelv2=np.ones(4, np.float32))
    c = _ctrl(_ds([f1], aligned=True))
    v = c.view
    assert v.get_ff_is_aligned() is True
    assert not v._pose_w.isVisibleTo(v)
    assert not v._outros_w.isVisibleTo(v)
    assert not v._ofiq_w.isVisibleTo(v)
    assert not v._ff_det_size_spin.isVisibleTo(v)
    v.chk_ofiq.setChecked(True)
    assert c.get_config().compute_ofiq is False     # OFIQ exige a original


def test_ff_models_uses_present_when_only_attrs(app):
    f1 = _face(embedding_sepaelv2=np.ones(4, np.float32), yaw=None)   # emb presente, pose ausente
    c = _ctrl(_ds([f1]))
    v = c.view
    v.chk_yaw.setChecked(True)                      # usuário pede pose manualmente
    assert not v.any_model_checked()                # sepaelv2 presente → desmarcado
    cfg = c.get_config()
    assert cfg.ff_models == ["sepaelv2"]            # cai nos modelos já presentes
    assert cfg.compute_ff is True                   # atributo pedido → roda FF
    assert "yaw" in cfg.ff_kept_attributes


def test_gpu_batch_defaults(app):
    v = _ctrl(_ds([_face(embedding_sepaelv2=np.ones(4, np.float32))]),
              use_gpu=True, batch_size=64).view
    assert v.get_use_gpu() is True and v.get_batch_size() == 64
