"""Imagens sem face detectada saem do dataset e esperam decisão humana.

Antes, a imagem sem detecção entrava no dataset carregando a mensagem de
erro: uma linha sem embedding no meio dos dados, que só aparecia para quem
fosse olhar a coluna de erro. Agora ela é separada e devolvida a quem
chamou, e a gravação do parquet espera a decisão.

O que os testes travam:

- a separação acontece ANTES de copiar imagem e gravar parquet — decidir
  depois de gravar significaria criar um arquivo que a decisão invalida;
- as faces separadas saem com a marca de erro já aplicada (elas não passam
  mais pela consolidação do fim do fluxo);
- FFInfo existente sem embedding nenhum conta como falha: atributos de pose
  podem ter sido preenchidos por uma passada em que o encoder falhou;
- dataset vazio NÃO é gravado. Parquet de zero linhas é aceito pela
  interface, listado como recurso e desaparece dos seletores sem
  explicação — foi assim que datasets periciais viraram arquivos de 798
  bytes sem ninguém notar;
- gênero e etnia fixos são gravados em cada face (o caminho antigo
  escrevia num atributo inexistente e derrubava a ingestão).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.pipelines.feature_extraction_pipeline import (
    FeatureExtractionPipeline,
    _tem_algum_embedding,
)

SLOT = SepaelModel.default().ffinfo_slot


def _face(nome: str, *, com_embedding: bool) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}.jpg")
    f.identity = "id1"
    if com_embedding:
        f.ff_info = FFInfo(**{SLOT: np.ones(4, dtype=np.float32)})
    return f


def _dataset(*faces) -> FaceDataset:
    return FaceDataset(name="teste", faces=list(faces))


def _pipeline(tmp_path) -> FeatureExtractionPipeline:
    from unittest.mock import MagicMock

    return FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=MagicMock(),
        ofiq_service=MagicMock(), serfiq_service=MagicMock(),
        storage_path=Path(tmp_path),
    )


# ── Quem conta como "sem detecção" ───────────────────────────────────────

def test_ffinfo_ausente_conta_como_falha():
    assert _tem_algum_embedding(None) is False


def test_ffinfo_sem_embedding_nenhum_conta_como_falha():
    """Uma passada pode preencher pose e falhar no encoder.

    `ff_info is not None` não basta: o objeto existe e a face continua sem
    biometria nenhuma.
    """
    assert _tem_algum_embedding(FFInfo(yaw=10.0, pitch=2.0, det_score=0.8)) is False


def test_ffinfo_com_embedding_em_qualquer_slot_conta_como_sucesso():
    for member in SepaelModel:
        ff = FFInfo(**{member.ffinfo_slot: np.ones(4, dtype=np.float32)})
        assert _tem_algum_embedding(ff) is True, member.ffinfo_slot


# ── Separação ────────────────────────────────────────────────────────────

def test_separa_apenas_as_sem_deteccao():
    ds = _dataset(
        _face("ok1", com_embedding=True),
        _face("ruim", com_embedding=False),
        _face("ok2", com_embedding=True),
    )

    pendentes = FeatureExtractionPipeline._set_aside_undetected(ds, [])

    assert [f.image_path for f in pendentes] == ["C:/fotos/ruim.jpg"]
    assert len(ds) == 2
    assert all("ruim" not in f.image_path for f in ds.faces)


def test_sem_pendentes_nao_mexe_no_dataset():
    ds = _dataset(_face("ok", com_embedding=True))

    assert FeatureExtractionPipeline._set_aside_undetected(ds, []) == []
    assert len(ds) == 1


def test_faces_separadas_saem_marcadas():
    """Elas não passam mais pela consolidação do fim do fluxo — se voltarem
    (o usuário decidiu incluir), têm de chegar já marcadas."""
    ds = _dataset(_face("ruim", com_embedding=False))

    pendentes = FeatureExtractionPipeline._set_aside_undetected(ds, [])

    assert pendentes[0].error is True
    assert pendentes[0].error_stage == "ff"
    assert "orientações" in pendentes[0].error_message, (
        "a mensagem precisa dizer que as quatro orientações foram tentadas, "
        "senão parece que só houve uma tentativa"
    )


def test_entradas_de_falha_saem_da_lista():
    """A consolidação do fim só varre `dataset.faces`; entrada órfã ali daria
    a impressão de que a face foi tratada."""
    ruim = _face("ruim", com_embedding=False)
    ok = _face("ok", com_embedding=True)
    ds = _dataset(ok, ruim)
    falhas = [(ruim, "ff", "sem embedding"), (ok, "ofiq", "outro problema")]

    FeatureExtractionPipeline._set_aside_undetected(ds, falhas)

    assert [e[0] for e in falhas] == [ok], "só a entrada da face separada sai"


def test_progresso_avisa_quantas_foram():
    mensagens = []
    ds = _dataset(_face("a", com_embedding=False), _face("b", com_embedding=False))

    FeatureExtractionPipeline._set_aside_undetected(
        ds, [], lambda msg, pct: mensagens.append(msg),
    )

    assert any("2" in m for m in mensagens)


# ── Gravação adiada ──────────────────────────────────────────────────────

def test_destino_e_o_mesmo_nas_duas_chamadas(tmp_path):
    """A gravação acontece numa segunda chamada; recalcular o destino por
    fora arriscaria gravar noutro lugar."""
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="meu", dataset_dir=tmp_path,
    )

    assert pipe._resolve_destination(cfg) == pipe._resolve_destination(cfg)
    _, arquivo = pipe._resolve_destination(cfg)
    assert arquivo.name == "meu.parquet"


def test_autocontido_ganha_pasta_propria(tmp_path):
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="meu", dataset_dir=tmp_path,
        self_contained=True,
    )

    pasta, arquivo = pipe._resolve_destination(cfg)

    assert pasta.name == "meu"
    assert arquivo == pasta / "meu.parquet"


def test_recusa_gravar_dataset_vazio(tmp_path):
    """Todas descartadas na revisão: nada vai para o disco.

    Um parquet de zero linhas é aceito pela interface e some dos seletores
    sem explicação — é o jeito mais eficiente de perder um dataset em
    silêncio.
    """
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="vazio", dataset_dir=tmp_path,
    )
    ds = _dataset()

    with pytest.raises(ValueError, match="vazio"):
        pipe.finalize_storage_and_save(ds, cfg)

    assert not (tmp_path / "vazio.parquet").exists(), (
        "nem um arquivo de zero linhas pode sobrar"
    )


def test_finalizacao_grava_e_limpa_a_marca(tmp_path):
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="grava", dataset_dir=tmp_path,
    )
    ds = _dataset(_face("ok", com_embedding=True))
    ds._save_deferred = True
    ds._undetected_faces = [_face("ruim", com_embedding=False)]

    resultado = pipe.finalize_storage_and_save(ds, cfg)

    assert (tmp_path / "grava.parquet").exists()
    assert resultado._save_deferred is False
    assert resultado._undetected_faces == [], (
        "a lista tem de esvaziar: o dataset já está no disco e a decisão foi "
        "aplicada"
    )


def test_campos_de_runtime_ficam_fora_do_repr():
    ds = _dataset(_face("ok", com_embedding=True))
    ds._undetected_faces = [_face("ruim", com_embedding=False)]

    assert "_undetected_faces" not in repr(ds)
    assert "_save_deferred" not in repr(ds)


# ── Gênero/etnia fixos (defeito encontrado no caminho) ───────────────────

def test_genero_fixo_vai_para_cada_face(tmp_path, monkeypatch):
    """O caminho antigo escrevia em `dataset._df`, atributo que o
    FaceDataset não tem: escolher gênero fixo no diálogo derrubava a
    ingestão inteira com AttributeError."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    ds = _dataset(_face("a", com_embedding=True), _face("b", com_embedding=True))
    monkeypatch.setattr(
        FaceDatasetAdapter, "dataset_from_dir",
        staticmethod(lambda **kwargs: ds),
    )
    pipe = _pipeline(tmp_path)
    cfg = DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="gen", dataset_dir=tmp_path,
        default_gender="Male", default_ethnicity="Branca",
        compute_ff=False, compute_ofiq=False, compute_serfiq=False,
    )

    resultado = pipe.execute(cfg)

    assert [f.gender for f in resultado.faces] == ["Male", "Male"]
    assert [f.ethnicity for f in resultado.faces] == ["Branca", "Branca"]
