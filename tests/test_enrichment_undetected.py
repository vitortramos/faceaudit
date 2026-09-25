"""Giro e revisão no enriquecimento.

O enriquecimento é ADITIVO: a face já está no dataset e apenas não ganhou a
biometria pedida. Isso muda a pergunta — não há o que incluir, e remover
seria edição destrutiva com caminho próprio no aplicativo. A janela serve
para inspecionar e tentar numa resolução maior.

E havia uma regressão a fechar: um dataset criado COM a retentativa de giro
tem faces que só existem giradas. Enriquecê-lo tentando apenas a orientação
gravada falharia justamente nelas — em silêncio, porque o laço de
alinhamento simplesmente pula quem não alinhou. Por isso a orientação já
registrada na face é a PRIMEIRA a ser tentada.

O que os testes travam:

- a orientação conhecida vem primeiro, e é a que evita a regressão;
- nada é retirado do dataset na revisão do enriquecimento;
- a gravação é adiada até a revisão, e é UMA gravação (não duas);
- o enriquecimento grava por outro caminho que a criação: reescreve o
  parquet onde ele está, sem materializar modo de armazenamento;
- em dataset de faces já alinhadas a revisão não se aplica — não há
  detecção a refazer.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.enums.undetected_review_mode import UndetectedReviewMode
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.services.image_rotation_service import ROTACOES_PADRAO, girar_imagem
from faceaudit.services.oriented_alignment_service import (
    alinhar_tentando_rotacoes,
    rotacao_registrada,
)

SLOT = SepaelModel.default().ffinfo_slot
ALTURA, LARGURA = 30, 70
FONTE = (
    np.arange(ALTURA * LARGURA * 3, dtype=np.int64).reshape(ALTURA, LARGURA, 3) % 251
).astype(np.uint8)


class _EngineAlinhador:
    """Só alinha quando a imagem chega na orientação que ele "vê"."""

    def __init__(self, graus_que_ve: int):
        self.graus_que_ve = graus_que_ve
        self.orientacoes_recebidas: list = []

    def detect_and_align(self, img, single_face=True):
        for graus in (0, 90, 180, 270):
            if np.array_equal(img, girar_imagem(FONTE, graus)):
                self.orientacoes_recebidas.append(graus)
                break
        else:
            self.orientacoes_recebidas.append(None)
        if not np.array_equal(img, girar_imagem(FONTE, self.graus_que_ve)):
            return {}
        return {
            "aligned_face": np.zeros((112, 112, 3), dtype=np.uint8),
            "aligned_keypoints": np.zeros((5, 2), dtype=np.float32),
        }


def _face(nome: str, *, rotacao=None) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}")
    f.identity = "id1"
    f.ff_info = FFInfo(
        detection_rotation=rotacao, **{SLOT: np.ones(4, dtype=np.float32)},
    )
    return f


# ── A orientação conhecida vem primeiro ──────────────────────────────────

def test_orientacao_conhecida_e_tentada_primeiro():
    """Fecha a regressão: sem isso, a face que só existe girada nunca
    alinharia num enriquecimento — e o laço a pularia sem dizer nada."""
    engine = _EngineAlinhador(graus_que_ve=90)
    face = _face("de_lado.jpg", rotacao=90)

    item, graus = alinhar_tentando_rotacoes(
        engine, FONTE, ROTACOES_PADRAO, "C:/fotos/de_lado.jpg",
        rotacao_conhecida=rotacao_registrada(face),
    )

    assert item, "tinha de alinhar"
    assert graus == 90
    assert engine.orientacoes_recebidas == [90], (
        "a orientação gravada na face é a primeira tentativa — acertar de "
        "primeira é o ponto"
    )


def test_sem_orientacao_conhecida_comeca_pela_gravada():
    engine = _EngineAlinhador(graus_que_ve=0)
    face = _face("normal.jpg")

    item, graus = alinhar_tentando_rotacoes(
        engine, FONTE, ROTACOES_PADRAO, "C:/fotos/normal.jpg",
        rotacao_conhecida=rotacao_registrada(face),
    )

    assert item and graus == 0
    assert engine.orientacoes_recebidas == [0]


def test_orientacao_conhecida_errada_nao_impede_as_outras():
    """A marca pode estar desatualizada (imagem trocada, por exemplo)."""
    engine = _EngineAlinhador(graus_que_ve=270)
    face = _face("a.jpg", rotacao=90)

    item, graus = alinhar_tentando_rotacoes(
        engine, FONTE, ROTACOES_PADRAO, "C:/fotos/a.jpg",
        rotacao_conhecida=rotacao_registrada(face),
    )

    assert item and graus == 270
    assert engine.orientacoes_recebidas[0] == 90, "tentou a conhecida primeiro"


def test_sem_rotacoes_ainda_tenta_a_conhecida():
    """Mesmo com a retentativa desligada, a orientação já comprovada é
    tentada: ela não é uma busca, é um dado do registro."""
    engine = _EngineAlinhador(graus_que_ve=90)
    face = _face("a.jpg", rotacao=90)

    item, _ = alinhar_tentando_rotacoes(
        engine, FONTE, (), "C:/fotos/a.jpg",
        rotacao_conhecida=rotacao_registrada(face),
    )

    assert item


def test_nenhuma_orientacao_alinha_devolve_vazio():

    class _Cego(_EngineAlinhador):
        def detect_and_align(self, img, single_face=True):
            return {}

    item, graus = alinhar_tentando_rotacoes(
        _Cego(0), FONTE, ROTACOES_PADRAO, "C:/fotos/a.jpg",
        rotacao_conhecida=rotacao_registrada(_face("a.jpg")),
    )

    assert item is None and graus == 0


def test_falha_numa_orientacao_nao_condena_as_outras():

    class _Estoura(_EngineAlinhador):
        def detect_and_align(self, img, single_face=True):
            if np.array_equal(img, FONTE):
                raise RuntimeError("shape não suportado")
            return super().detect_and_align(img, single_face)

    erros: list = []
    item, graus = alinhar_tentando_rotacoes(
        _Estoura(graus_que_ve=90), FONTE, ROTACOES_PADRAO, "C:/fotos/a.jpg",
        rotacao_conhecida=rotacao_registrada(_face("a.jpg")),
        error_callback=erros.append,
    )

    assert item and graus == 90
    assert erros


# ── A revisão do enriquecimento não remove nada ──────────────────────────

def test_modo_enriquecimento_nao_permite_decidir_inclusao():
    assert UndetectedReviewMode.ENRICHMENT.permite_decidir_inclusao is False
    assert UndetectedReviewMode.CREATION.permite_decidir_inclusao is True


def test_politica_do_enriquecimento_aponta_o_caminho_correto_para_remover():
    """Remover face de dataset existente é destrutivo e já tem caminho com
    justificativa; duplicar aqui daria garantia mais fraca."""
    from faceaudit.services.undetected_review_policy import resolve_inclusion_policy

    politica = resolve_inclusion_policy(
        DatasetIngestionConfig(input_path=".", dataset_name="x"),
        UndetectedReviewMode.ENRICHMENT,
    )

    assert politica.blocked is True
    assert "Excluir selecionados" in politica.reason
    assert "justificativa" in politica.reason


# ── Gravação adiada ──────────────────────────────────────────────────────

def _pipeline(tmp_path):
    from unittest.mock import MagicMock

    return __import__(
        "faceaudit.pipelines.feature_extraction_pipeline", fromlist=["x"],
    ).FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=MagicMock(),
        ofiq_service=MagicMock(), serfiq_service=MagicMock(),
        storage_path=Path(tmp_path),
    )


def test_enriquecimento_grava_onde_o_parquet_ja_esta(tmp_path):
    """Não materializa modo de armazenamento: o dataset já existia em disco,
    nada foi copiado nem relativizado."""
    pipe = _pipeline(tmp_path)
    ds = FaceDataset(name="existente", faces=[_face("a.jpg")])
    ds._parquet_dir = Path(tmp_path)
    ds._save_deferred = True

    resultado = pipe.finalize_enrichment_and_save(
        ds, DatasetIngestionConfig(input_path=".", dataset_name="existente"),
    )

    assert (tmp_path / "existente.parquet").exists()
    assert resultado._save_deferred is False
    assert resultado._undetected_faces == []


def test_enriquecimento_nao_recusa_dataset_vazio(tmp_path):
    """Diferente da criação: aqui não há como esvaziar, porque a revisão do
    enriquecimento não remove face nenhuma."""
    pipe = _pipeline(tmp_path)
    ds = FaceDataset(name="vazio", faces=[])
    ds._parquet_dir = Path(tmp_path)

    pipe.finalize_enrichment_and_save(
        ds, DatasetIngestionConfig(input_path=".", dataset_name="vazio"),
    )  # não levanta


def test_worker_escolhe_o_caminho_do_enriquecimento(tmp_path):
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.workers.finalize_dataset_worker import (
        FinalizeDatasetWorker,
    )

    chamadas: list = []

    class _PipelineFalso:
        def finalize_storage_and_save(self, ds, cfg, progress_callback=None):
            chamadas.append("criacao")
            return ds

        def finalize_enrichment_and_save(self, ds, cfg, progress_callback=None):
            chamadas.append("enriquecimento")
            return ds

    ds = FaceDataset(name="x", faces=[_face("a.jpg")])
    cfg = DatasetIngestionConfig(input_path=".", dataset_name="x")

    FinalizeDatasetWorker(_PipelineFalso(), ds, cfg, enrichment=True).run()
    FinalizeDatasetWorker(_PipelineFalso(), ds, cfg).run()

    assert chamadas == ["enriquecimento", "criacao"]


# ── Quem liga a revisão no enriquecimento ────────────────────────────────

def test_enriquecimento_de_alinhadas_nao_liga_a_revisao():
    """Em faces já alinhadas o recorte 112×112 vai direto ao encoder — não
    há detecção a refazer, então não há o que revisar."""
    import inspect

    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.enrich_dataset_dialog.enrich_dataset_controller import (  # noqa: E501
        EnrichDatasetController,
    )

    fonte = inspect.getsource(EnrichDatasetController.get_config)
    assert "review_undetected=not aligned" in fonte
