"""O terceiro tipo de imagem nos fluxos que gravam dataset.

`DatasetImageKind` decide o que o dataset autocontido guarda. O que estes
testes travam é a regra que não pode escorregar em nenhum fluxo:

- **`CROP` nunca marca `is_aligned`.** Esse flag faz o `FFService` pular
  detect+align e ir direto ao encoder; um recorte com margem não é face
  alinhada, e declarar que é produziria embeddings errados em silêncio.
- a margem é FATOR sobre a bbox, a mesma unidade da extração de vídeo;
- face cuja imagem não pôde ser produzida fica DE FORA do dataset e é
  registrada — nunca entra com a foto original no lugar.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.face_image_materialization_service import (
    FaceImageMaterializationService,
)


class _FaceFalsa:
    def __init__(self, bbox):
        self.bbox = np.asarray(bbox, dtype="float32")


class _GeometriaFalsa:
    def __init__(self, bbox=(40, 40, 80, 80)):
        self._bbox = bbox

    def detect_faces(self, image):
        if int(image[0, 0, 0]) == 7:
            return []
        return [_FaceFalsa(self._bbox)]

    def align_face(self, image):
        if int(image[0, 0, 0]) == 7:
            return None
        return np.full((112, 112, 3), 128, dtype="uint8")


def _imagem(caminho, *, com_face=True):
    import cv2

    arr = np.zeros((200, 200, 3), dtype="uint8")
    arr[:, :] = (30, 60, 90)
    arr[0, 0] = (0 if com_face else 7, 0, 0)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho), arr)


@pytest.fixture
def dataset(tmp_path):
    fotos = tmp_path / "fotos"
    faces = []
    for i in range(3):
        caminho = fotos / f"f{i}.png"
        _imagem(caminho)
        faces.append(FaceInfo(image_path=str(caminho), identity=f"P{i}"))
    ruim = fotos / "semface.png"
    _imagem(ruim, com_face=False)
    faces.append(FaceInfo(image_path=str(ruim), identity="P9"))
    return FaceDataset(name="base", faces=faces, path_mode=PathMode.ABSOLUTE)


# ── Config de ingestão ────────────────────────────────────────────────────

def test_config_converte_flag_antiga():
    """`align_during_ingestion=True` é a forma antiga de dizer ALIGNED."""
    cfg = DatasetIngestionConfig(
        input_path=".", dataset_name="x", align_during_ingestion=True,
    )
    assert cfg.image_kind is DatasetImageKind.ALIGNED
    assert cfg.align_during_ingestion is True


def test_config_crop_nao_liga_alinhamento():
    cfg = DatasetIngestionConfig(
        input_path=".", dataset_name="x", image_kind=DatasetImageKind.CROP,
    )
    assert cfg.align_during_ingestion is False, (
        "recorte não é alinhamento — ligar esse flag faria o pipeline "
        "materializar a face alinhada em vez do recorte"
    )
    assert cfg.crop_margin_factor == 2.0


# ── Materialização a partir de dataset ────────────────────────────────────

def test_dataset_recortado_nao_e_alinhado(dataset, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())

    novo, resultado = servico.materialize_dataset(
        dataset, tmp_path / "saida", kind=DatasetImageKind.CROP,
        margin_factor=2.0,
    )

    assert novo.is_aligned is False, (
        "um recorte com margem não é face alinhada — declarar que é faria "
        "o FFService pular detect+align e produzir embeddings errados"
    )
    assert novo.path_mode == PathMode.RELATIVE
    assert novo.path_root == "images"
    assert len(novo.faces) == 3
    assert resultado.n_failed == 1
    for face in novo.faces:
        assert novo.resolve_image_path(face).exists()


def test_dataset_alinhado_marca_is_aligned(dataset, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())

    novo, _ = servico.materialize_dataset(
        dataset, tmp_path / "saida", kind=DatasetImageKind.ALIGNED,
    )

    assert novo.is_aligned is True
    assert len(novo.faces) == 3


def test_face_sem_imagem_fica_de_fora_e_e_registrada(dataset, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())

    novo, resultado = servico.materialize_dataset(
        dataset, tmp_path / "saida", kind=DatasetImageKind.CROP,
    )

    assert "P9" not in [f.identity for f in novo.faces], (
        "sem face detectada, a linha não pode entrar apontando para a "
        "foto original — o dataset pareceria uniforme sem ser"
    )
    dados = json.loads(resultado.manifest_path.read_text(encoding="utf-8"))
    assert len(dados["failures"]) == 1


def test_margem_muda_o_tamanho_do_recorte(dataset, tmp_path):
    import cv2

    servico = FaceImageMaterializationService(_GeometriaFalsa((40, 40, 80, 80)))

    novo_1, _ = servico.materialize_dataset(
        dataset, tmp_path / "a", kind=DatasetImageKind.CROP, margin_factor=1.0,
    )
    novo_2, _ = servico.materialize_dataset(
        dataset, tmp_path / "b", kind=DatasetImageKind.CROP, margin_factor=2.0,
    )

    lado_1 = cv2.imread(str(novo_1.resolve_image_path(novo_1.faces[0]))).shape[0]
    lado_2 = cv2.imread(str(novo_2.resolve_image_path(novo_2.faces[0]))).shape[0]
    assert (lado_1, lado_2) == (40, 80)


def test_entrada_nao_e_mutada(dataset, tmp_path):
    caminhos_antes = [f.image_path for f in dataset.faces]
    servico = FaceImageMaterializationService(_GeometriaFalsa())

    servico.materialize_dataset(
        dataset, tmp_path / "saida", kind=DatasetImageKind.CROP,
    )

    assert [f.image_path for f in dataset.faces] == caminhos_antes
    assert dataset.path_mode == PathMode.ABSOLUTE
