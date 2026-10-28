"""O Processar (pipeline de imagens) cria um dataset AUTOCONTIDO.

`<pasta escolhida>/<nome>/` com o parquet, a cadeia de custódia, o
manifesto e `images/` dentro — a pasta pode ser movida inteira. O
derivado leva as colunas que descrevem a pessoa (template) e larga as
que foram medidas na imagem de origem, que deixaram de valer.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.hash_service import HashService
from faceaudit.services.image_pipeline_batch_service import (
    ImagePipelineBatchService,
)


def _pipeline(steps=None, **kw) -> ImageProcessingPipelineConfig:
    return ImageProcessingPipelineConfig.from_dict({
        "name": "oclusão boca",
        "steps": steps if steps is not None else [
            {"step_type": "blur", "kind": "gaussian", "kernel_size": 7, "sigma": 2.0},
        ],
        **kw,
    })


@pytest.fixture
def origem(tmp_path) -> tuple[FaceDataset, Path]:
    """Dataset autocontido de origem, gravado e relido (com specs)."""
    pasta = tmp_path / "Datasets" / "DB_References"
    imgs = pasta / "images"
    imgs.mkdir(parents=True)
    rng = np.random.default_rng(0)
    faces = []
    for i, idade in enumerate([45, None, 30]):
        nome = f"ID00{i}_RENACH2016_{idade or 'NC'}.jpg"
        cv2.imencode(".jpg", rng.integers(0, 256, (60, 50, 3), np.uint8))[1].tofile(
            str(imgs / nome))
        faces.append(FaceInfo(
            image_path=nome, identity=f"ID00{i}", origin="RENACH2016",
            extra_data={"idade": idade, "jpeg_quality_estimate": 75.0},
        ))
    ds = FaceDataset(name="DB_References", faces=faces,
                     path_mode=PathMode.RELATIVE, path_root="images")
    ds._parquet_dir = pasta
    parquet = pasta / "DB_References.parquet"
    ds.to_parquet(parquet)
    return FaceDataset.from_parquet(str(parquet)), parquet


def _rodar(origem, destino, **kw):
    ds, parquet = origem
    return ImagePipelineBatchService().run(
        ds, kw.pop("pipeline", _pipeline()), destino,
        dataset_name=kw.pop("dataset_name", "DB_References borrado"),
        source_parquet_path=parquet, **kw,
    )


def test_layout_autocontido_na_pasta_escolhida(origem, tmp_path):
    destino = tmp_path / "qualquer" / "lugar"
    r = _rodar(origem, destino)

    pasta = destino / "DB_References borrado"
    assert r.dataset_dir == pasta.resolve()
    assert r.parquet_path == pasta.resolve() / "DB_References borrado.parquet"
    assert r.images_dir == pasta.resolve() / "images"
    assert r.manifest_path == pasta.resolve() / "manifest.json"
    assert sorted(p.name for p in r.images_dir.iterdir()) == [
        "ID000_RENACH2016_45.png", "ID001_RENACH2016_NC.png", "ID002_RENACH2016_30.png",
    ]
    assert AuditLogService.sidecar_path_for(r.parquet_path).exists()
    assert r.n_failed == 0 and r.error_parquet_path is None


def test_pasta_movida_continua_achando_as_imagens(origem, tmp_path):
    r = _rodar(origem, tmp_path / "a")
    movida = tmp_path / "b" / "copia"
    shutil.move(str(r.dataset_dir), str(movida))

    ds = FaceDataset.from_parquet(str(movida / "DB_References borrado.parquet"))
    assert ds.path_mode == PathMode.RELATIVE
    for face in ds.faces:
        caminho = ds.resolve_image_path(face)
        assert caminho.exists(), caminho
        assert movida in caminho.parents
        assert HashService().calculate_file_hash(caminho) == face.hash


def test_colunas_de_template_acompanham_e_medidas_da_imagem_nao(origem, tmp_path):
    r = _rodar(origem, tmp_path)
    ds = FaceDataset.from_parquet(str(r.parquet_path))

    idades = {f.identity: f.extra_data.get("idade") for f in ds.faces}
    assert idades == {"ID000": 45, "ID001": None, "ID002": 30}
    assert all(type(v) is int for v in idades.values() if v is not None)
    assert "idade" in ds.column_specs
    assert all("jpeg_quality_estimate" not in f.extra_data for f in ds.faces)
    assert {f.origin for f in ds.faces} == {"RENACH2016"}

    manifesto = json.loads(r.manifest_path.read_text(encoding="utf-8"))
    assert manifesto["columns_carried"] == ["idade"]
    assert manifesto["columns_dropped"] == ["jpeg_quality_estimate"]


def test_cadeia_do_derivado_nasce_com_create(origem, tmp_path):
    _, parquet_origem = origem
    r = _rodar(origem, tmp_path)

    (entrada,) = AuditLogService.load(r.parquet_path)
    assert entrada.operation == AuditOperation.CREATE
    assert entrada.dataset_hash == HashService().calculate_file_hash(r.parquet_path)
    m = entrada.metadata
    assert m["source_dataset"] == "DB_References"
    assert m["source_parquet_sha256"] == HashService().calculate_file_hash(parquet_origem)
    assert m["manifest_sha256"] == HashService().calculate_file_hash(r.manifest_path)
    assert m["pipeline_config"]["steps"][0]["step_type"] == "blur"
    assert m["rows"] == 3 and m["n_failed"] == 0
    assert "oclusão boca" in entrada.description


def test_imagem_ausente_vai_para_o_parquet_de_erros_na_pasta_do_dataset(origem, tmp_path):
    ds, _ = origem
    perdida = ds.resolve_image_path(ds.faces[1])
    perdida.unlink()

    r = _rodar(origem, tmp_path)
    assert r.n_failed == 1
    assert r.error_parquet_path.parent == r.dataset_dir
    erros = FaceDataset.from_parquet(str(r.error_parquet_path))
    # O caminho aponta para a imagem-fonte, absoluto: a âncora relativa
    # da origem não vale a partir da pasta do derivado.
    assert Path(erros.faces[0].image_path) == perdida


def test_reprocessar_na_mesma_pasta_reaproveita_os_pngs(origem, tmp_path):
    r1 = _rodar(origem, tmp_path)
    antes = {p.name: p.stat().st_mtime_ns for p in r1.images_dir.iterdir()}
    r2 = _rodar(origem, tmp_path)
    assert r2.n_skipped == 3 and r2.n_succeeded == 0
    assert {p.name: p.stat().st_mtime_ns for p in r2.images_dir.iterdir()} == antes


def test_nome_com_caractere_proibido_vira_o_mesmo_nome_na_pasta_e_no_arquivo(origem, tmp_path):
    r = _rodar(origem, tmp_path, dataset_name="a:b")
    assert r.dataset_dir.name == "a_b"
    assert r.parquet_path.name == "a_b.parquet"


@pytest.mark.parametrize("origem_alinhada, steps, esperado", [
    (False, None, False),
    (True, None, True),  # borrar face alinhada continua face alinhada
    (True, [{"step_type": "crop", "mode": "centered",
             "target_width": 20, "target_height": 20}], False),
])
def test_derivado_de_dataset_alinhado(origem, tmp_path, origem_alinhada, steps, esperado):
    ds, _ = origem
    ds.is_aligned = origem_alinhada
    r = _rodar(origem, tmp_path, pipeline=_pipeline(steps))
    assert FaceDataset.from_parquet(str(r.parquet_path)).is_aligned is esperado
