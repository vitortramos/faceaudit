"""Faces recuperadas na revisão passam pelos estágios que já haviam rodado.

Este é o furo silencioso do fluxo: a face recuperada chega DEPOIS da
qualidade (OFIQ, SER-FIQ) e da materialização do recorte, que rodaram sobre
o dataset de onde ela havia sido retirada. Sem completá-la, ela entra com
embedding e sem métrica de qualidade — e, num dataset de recortes, sem
arquivo de recorte. O dataset fica heterogêneo por dentro e nada na
interface mostra isso.

O que os testes travam:

- os estágios rodam sobre as faces recuperadas, e SÓ sobre elas;
- ordem: recorte antes da qualidade, porque a qualidade lê `image_path` e o
  recorte reescreve esse caminho;
- a conclusão vem ANTES da gravação — é a gravação que normaliza os
  caminhos, e inverter deixaria caminho absoluto num dataset relativo;
- a custódia distingue "recuperada numa segunda tentativa" de
  "incluída sem biometria": a primeira TEM embedding.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

SLOT = SepaelModel.default().ffinfo_slot


def _face(nome: str, *, det_size=None) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}")
    f.identity = "id1"
    f.ff_info = FFInfo(
        detection_det_size=det_size, **{SLOT: np.ones(4, dtype=np.float32)},
    )
    return f


def _pipeline(tmp_path, *, ofiq=None, serfiq=None):
    from unittest.mock import MagicMock

    return FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=MagicMock(),
        ofiq_service=ofiq or MagicMock(), serfiq_service=serfiq or MagicMock(),
        storage_path=Path(tmp_path),
    )


def _config(tmp_path, **kwargs):
    base = dict(
        input_path=tmp_path, dataset_name="ds", dataset_dir=tmp_path,
        compute_ofiq=False, compute_serfiq=False,
    )
    base.update(kwargs)
    return DatasetIngestionConfig(**base)


# ── Escopo ───────────────────────────────────────────────────────────────

def test_qualidade_roda_so_nas_recuperadas(tmp_path):
    from unittest.mock import MagicMock

    ofiq = MagicMock()
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    recuperada = _face("recuperada.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[_face("normal.jpg"), recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada], _config(tmp_path, compute_ofiq=True),
    )

    assert ofiq.process_dataset.call_count == 1
    sub = ofiq.process_dataset.call_args[0][0]
    assert [f.image_path for f in sub.faces] == ["C:/fotos/recuperada.jpg"], (
        "a face que já passou pela qualidade não pode passar de novo"
    )


def test_lista_vazia_nao_faz_nada(tmp_path):
    from unittest.mock import MagicMock

    ofiq = MagicMock()
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])

    pipe.complete_stages_for_recovered(ds, [], _config(tmp_path, compute_ofiq=True))

    ofiq.process_dataset.assert_not_called()


def test_estagio_desligado_nao_roda(tmp_path):
    from unittest.mock import MagicMock

    ofiq = MagicMock()
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada], _config(tmp_path, compute_ofiq=False),
    )

    ofiq.process_dataset.assert_not_called()


def test_subconjunto_carrega_os_mesmos_objetos(tmp_path):
    """Não pode haver cópia: o enriquecimento tem de aparecer no dataset."""
    from unittest.mock import MagicMock

    ofiq = MagicMock()
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada], _config(tmp_path, compute_ofiq=True),
    )

    sub = ofiq.process_dataset.call_args[0][0]
    assert sub.faces[0] is recuperada


def test_subconjunto_herda_a_resolucao_de_caminhos(tmp_path):
    """Sem herdar path_mode/path_root/_parquet_dir, o estágio não acha a
    imagem de um dataset de caminhos relativos."""
    from unittest.mock import MagicMock

    ofiq = MagicMock()
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])
    ds._parquet_dir = Path(tmp_path)
    ds.is_aligned = False

    pipe.complete_stages_for_recovered(
        ds, [recuperada], _config(tmp_path, compute_ofiq=True),
    )

    sub = ofiq.process_dataset.call_args[0][0]
    assert sub.path_mode is ds.path_mode
    assert sub.path_root == ds.path_root
    assert sub._parquet_dir == ds._parquet_dir


# ── Ordem ────────────────────────────────────────────────────────────────

def test_recorte_vem_antes_da_qualidade(tmp_path, monkeypatch):
    """A qualidade lê `image_path`, e o recorte reescreve esse caminho.

    Invertido, a qualidade mediria a foto inteira e o dataset guardaria
    métrica de uma imagem que não é a do arquivo.
    """
    from unittest.mock import MagicMock

    ordem: list = []
    ofiq = MagicMock()
    ofiq.process_dataset.side_effect = lambda *a, **k: ordem.append("qualidade")
    pipe = _pipeline(tmp_path, ofiq=ofiq)
    monkeypatch.setattr(
        FeatureExtractionPipeline, "_materialize_face_crops",
        staticmethod(lambda *a, **k: ordem.append("recorte")),
    )
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada],
        _config(
            tmp_path, compute_ofiq=True, self_contained=True,
            image_kind=DatasetImageKind.CROP,
        ),
    )

    assert ordem == ["recorte", "qualidade"]


def test_dataset_de_alinhadas_materializa_alinhado(tmp_path, monkeypatch):
    chamou: list = []
    monkeypatch.setattr(
        FeatureExtractionPipeline, "_materialize_aligned_crops",
        staticmethod(lambda *a, **k: chamou.append("alinhado")),
    )
    pipe = _pipeline(tmp_path)
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada],
        _config(
            tmp_path, self_contained=True, image_kind=DatasetImageKind.ALIGNED,
        ),
    )

    assert chamou == ["alinhado"]


def test_dataset_de_originais_nao_materializa_nada(tmp_path, monkeypatch):
    chamou: list = []
    for nome in ("_materialize_face_crops", "_materialize_aligned_crops"):
        monkeypatch.setattr(
            FeatureExtractionPipeline, nome,
            staticmethod(lambda *a, **k: chamou.append(nome)),
        )
    pipe = _pipeline(tmp_path)
    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])

    pipe.complete_stages_for_recovered(
        ds, [recuperada],
        _config(tmp_path, self_contained=True, image_kind=DatasetImageKind.ORIGINAL),
    )

    assert chamou == []


# ── O worker chama na ordem certa ────────────────────────────────────────

def test_worker_completa_antes_de_gravar(tmp_path):
    """Gravar antes deixaria caminho absoluto num dataset relativo — o
    dataset abriria, listaria as faces e só falharia ao abrir a imagem."""
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.workers.finalize_dataset_worker import (
        FinalizeDatasetWorker,
    )

    ordem: list = []

    class _PipelineFalso:
        def complete_stages_for_recovered(self, *a, **k):
            ordem.append("completar")

        def finalize_storage_and_save(self, dataset, config, progress_callback=None):
            ordem.append("gravar")
            return dataset

    recuperada = _face("a.jpg", det_size=640)
    ds = FaceDataset(name="ds", faces=[recuperada])
    worker = FinalizeDatasetWorker(
        _PipelineFalso(), ds, _config(tmp_path), recovered_faces=[recuperada],
    )
    worker.run()

    assert ordem == ["completar", "gravar"]


def test_worker_sem_recuperadas_apenas_grava(tmp_path):
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.workers.finalize_dataset_worker import (
        FinalizeDatasetWorker,
    )

    ordem: list = []

    class _PipelineFalso:
        def complete_stages_for_recovered(self, *a, **k):
            ordem.append("completar")

        def finalize_storage_and_save(self, dataset, config, progress_callback=None):
            ordem.append("gravar")
            return dataset

    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])
    FinalizeDatasetWorker(_PipelineFalso(), ds, _config(tmp_path)).run()

    assert ordem == ["gravar"]


# ── Custódia ─────────────────────────────────────────────────────────────

def _controller_duplo(tmp_path):
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        DatasetManagerController,
    )

    nomes = (
        "_descricao_da_criacao", "_metadados_da_deteccao", "_contar_giradas",
        "_nomes_de_arquivo",
    )
    corpo = {n: DatasetManagerController.__dict__[n] for n in nomes}
    ctrl = type("_Duplo", (), corpo)()
    ctrl._revisao_incluidas = []
    ctrl._revisao_descartadas = []
    ctrl._revisao_recuperadas = []
    return ctrl


def test_custodia_distingue_recuperada_de_incluida_sem_biometria(tmp_path):
    """Registrar as duas juntas apagaria a distinção que mais importa: uma
    tem embedding, a outra não."""
    pytest.importorskip("PyQt6")
    ctrl = _controller_duplo(tmp_path)
    recuperada = _face("recuperada.jpg", det_size=640)
    sem_biometria = FaceInfo(image_path="C:/fotos/sem.jpg")
    ctrl._revisao_incluidas = [recuperada, sem_biometria]
    ctrl._revisao_recuperadas = [recuperada]
    ds = FaceDataset(name="ds", faces=[recuperada])

    meta = ctrl._metadados_da_deteccao(ds)

    assert meta["undetected_included"] == 2
    assert meta["recovered_on_retry"] == 1
    assert meta["recovered_files"] == ["recuperada.jpg"]
    assert meta["recovered_det_sizes"] == [640]


def test_descricao_cita_a_resolucao_que_recuperou(tmp_path):
    pytest.importorskip("PyQt6")
    ctrl = _controller_duplo(tmp_path)
    recuperada = _face("a.jpg", det_size=640)
    ctrl._revisao_incluidas = [recuperada]
    ctrl._revisao_recuperadas = [recuperada]
    ds = FaceDataset(name="ds", faces=[recuperada])

    texto = ctrl._descricao_da_criacao(ds)

    assert "segunda tentativa" in texto
    assert "640 px" in texto


def test_sem_recuperadas_a_custodia_nao_inventa_campos(tmp_path):
    pytest.importorskip("PyQt6")
    ctrl = _controller_duplo(tmp_path)
    ds = FaceDataset(name="ds", faces=[_face("a.jpg")])

    meta = ctrl._metadados_da_deteccao(ds)

    assert "recovered_on_retry" not in meta
