"""Enrich multi-modelo em faces alinhadas processa UM modelo por vez.

Garante o pico de VRAM limitado a ~1 pipeline (em vez de N simultâneos), para
rodar em GPUs menores sem OOM/thrashing. Testa só o BRANCH e o laço por modelo
(sem carregar modelos reais): os builders de FFService são monkeypatchados por
fakes que registram as chamadas.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline


def _make_pipeline(tmp_path):
    ff = MagicMock()
    ff.matches_config.return_value = False   # nunca reusa o singleton → sempre temporário
    ff._use_gpu = False
    ff._batch_size = 32
    return FeatureExtractionPipeline(MagicMock(), ff, MagicMock(), MagicMock(), tmp_path)


def _make_dataset(n=2):
    faces = [FaceInfo(image_path=f"_aligned/alvo/f{i}.png", identity="alvo") for i in range(n)]
    return FaceDataset(name="ds", faces=faces)


def _wire(pipe, monkeypatch):
    """Monkeypatcha os builders + a seleção de alvo; devolve o registrador."""
    rec = {"single": [], "resolve": [], "models_per_call": []}

    def fake_process(target, progress_callback=None, models=None, rotation_attempts=(),
                     realign_from_keypoints=False):
        rec["models_per_call"].append(list(models or []))
        for f in target.faces:
            f.ff_info = FFInfo()   # sem falhas

    def fake_single(config, model):
        rec["single"].append(model)
        m = MagicMock()
        m.process_dataset.side_effect = fake_process
        return m

    def fake_resolve(config):
        rec["resolve"].append("resolve")
        m = MagicMock()
        m.process_dataset.side_effect = fake_process
        return m

    monkeypatch.setattr(pipe, "_build_single_model_ff_service", fake_single)
    monkeypatch.setattr(pipe, "_resolve_ff_service", fake_resolve)
    # isola o teste do I/O de seleção de alvo — devolve o próprio dataset.
    monkeypatch.setattr(pipe, "_select_target_for_stage", lambda *a, **k: a[0])
    return rec


def _cfg(tmp_path, models, is_aligned):
    return DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="ds", ff_models=models, is_aligned=is_aligned,
        compute_ff=True, compute_ofiq=False, compute_serfiq=False)


def test_aligned_multimodel_one_model_at_a_time(tmp_path, monkeypatch):
    pipe = _make_pipeline(tmp_path)
    ds = _make_dataset(2)
    rec = _wire(pipe, monkeypatch)
    pipe.enrich_existing_dataset(ds, _cfg(tmp_path, ["sepaelv2", "sepaelv4", "sepaelv6"], True))
    # um build por modelo, cada process_dataset com UM modelo, caminho único não usado
    assert rec["single"] == ["sepaelv2", "sepaelv4", "sepaelv6"]
    assert rec["models_per_call"] == [["sepaelv2"], ["sepaelv4"], ["sepaelv6"]]
    assert rec["resolve"] == []


def test_single_model_uses_combined_path(tmp_path, monkeypatch):
    pipe = _make_pipeline(tmp_path)
    ds = _make_dataset(2)
    rec = _wire(pipe, monkeypatch)
    pipe.enrich_existing_dataset(ds, _cfg(tmp_path, ["sepaelv2"], True))
    # 1 modelo → sem laço sequencial; usa o caminho único
    assert rec["resolve"] == ["resolve"]
    assert rec["single"] == []


def test_non_aligned_multimodel_uses_combined_path(tmp_path, monkeypatch):
    pipe = _make_pipeline(tmp_path)
    ds = _make_dataset(2)
    rec = _wire(pipe, monkeypatch)
    # não-alinhado: laço por modelo refaria detecção → mantém caminho único
    pipe.enrich_existing_dataset(ds, _cfg(tmp_path, ["sepaelv2", "sepaelv4"], False))
    assert rec["resolve"] == ["resolve"]
    assert rec["single"] == []
    assert rec["models_per_call"] == [["sepaelv2", "sepaelv4"]]
