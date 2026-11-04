"""O aviso "já existe, sobrescrever?" da criação olha o arquivo que será gravado.

O defeito: a interface montava `Destino/{nome}.parquet` por conta própria,
mas o modo autocontido grava `Destino/{nome}/{nome}.parquet` (convenção de
`self_contained_dataset_dir`). O aviso nunca disparava nesse modo e o dataset
anterior era substituído sem pergunta. Agora a interface pergunta ao
pipeline (`destination_parquet`), que usa a MESMA conta da gravação.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
from faceaudit.services.dataset_localization_service import (
    self_contained_dataset_dir,
)


def _pipeline(tmp_path) -> FeatureExtractionPipeline:
    return FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=MagicMock(),
        ofiq_service=MagicMock(), serfiq_service=MagicMock(),
        storage_path=Path(tmp_path) / "global",
    )


def _config(destino: Path, *, autocontido: bool) -> DatasetIngestionConfig:
    cfg = DatasetIngestionConfig(input_path=destino / "fotos", dataset_name="caso 01")
    cfg.dataset_dir = str(destino)
    cfg.self_contained = autocontido
    return cfg


def test_autocontido_aponta_para_a_pasta_propria(tmp_path):
    destino = tmp_path / "datasets"
    caminho = _pipeline(tmp_path).destination_parquet(
        _config(destino, autocontido=True)
    )
    assert caminho == self_contained_dataset_dir(destino, "caso 01") / "caso 01.parquet"
    assert caminho == destino / "caso 01" / "caso 01.parquet"


def test_absoluto_grava_direto_no_destino(tmp_path):
    destino = tmp_path / "datasets"
    caminho = _pipeline(tmp_path).destination_parquet(
        _config(destino, autocontido=False)
    )
    assert caminho == destino / "caso 01.parquet"


def test_sem_destino_vai_para_o_storage_do_pipeline(tmp_path):
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(input_path=tmp_path, dataset_name="x")
    cfg.dataset_dir = None
    assert pipe.destination_parquet(cfg) == pipe.storage_path / "x.parquet"


def test_coincide_com_o_destino_da_gravacao(tmp_path):
    """Uma só fonte de verdade: o aviso e a gravação não podem divergir."""
    pipe = _pipeline(tmp_path)
    for autocontido in (True, False):
        cfg = _config(tmp_path / "d", autocontido=autocontido)
        assert pipe.destination_parquet(cfg) == pipe._resolve_destination(cfg)[1]


def test_aviso_dispara_quando_o_autocontido_ja_existe(tmp_path):
    """O cenário do defeito: o dataset autocontido existe e o caminho
    antigo (`Destino/{nome}.parquet`) dizia que não."""
    destino = tmp_path / "datasets"
    existente = destino / "caso 01" / "caso 01.parquet"
    existente.parent.mkdir(parents=True)
    existente.write_bytes(b"parquet")

    cfg = _config(destino, autocontido=True)
    assert not (destino / "caso 01.parquet").exists()  # o que a UI olhava
    assert _pipeline(tmp_path).destination_parquet(cfg).exists()
