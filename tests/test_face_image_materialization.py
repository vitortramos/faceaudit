"""Materialização de imagens de face: recorte com margem e alinhamento.

O serviço é o único lugar onde "produzir a imagem que será guardada"
acontece — antes cada fluxo resolvia por conta própria, com
comportamentos diferentes em caso de falha.

Os testes fixam o que não pode variar:

- a margem é FATOR sobre a bbox, a mesma unidade da extração de vídeo;
- imagem sem face detectada é PULADA, nunca copiada como está (uma pasta
  meio recortada e meio original parece uniforme e não é);
- toda saída tem manifesto, e ele registra também o que ficou de fora.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.services.face_image_materialization_service import (
    FaceImageMaterializationService,
    list_image_files,
)


class _FaceFalsa:
    def __init__(self, bbox):
        self.bbox = np.asarray(bbox, dtype="float32")


class _GeometriaFalsa:
    """Detector previsível: bbox fixa, e nomes que contenham 'semface' falham."""

    def __init__(self, bbox=(40, 40, 80, 80)):
        self._bbox = bbox
        self.imagens_vistas = 0

    def detect_faces(self, image):
        self.imagens_vistas += 1
        # A imagem "sem face" é marcada por um pixel sentinela no canto.
        if int(image[0, 0, 0]) == 7:
            return []
        return [_FaceFalsa(self._bbox)]

    def align_face(self, image):
        if int(image[0, 0, 0]) == 7:
            return None
        return np.full((112, 112, 3), 128, dtype="uint8")


def _imagem(caminho: Path, *, com_face: bool = True) -> None:
    import cv2

    arr = np.zeros((200, 200, 3), dtype="uint8")
    arr[:, :] = (30, 60, 90)
    # Sentinela lido pelo detector falso. PNG porque JPEG borraria o
    # pixel e o "sem face" viraria "com face".
    arr[0, 0] = (0 if com_face else 7, 0, 0)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(caminho), arr)


@pytest.fixture
def origem(tmp_path):
    pasta = tmp_path / "fotos"
    for i in range(3):
        _imagem(pasta / f"f{i}.png")
    _imagem(pasta / "semface.png", com_face=False)
    return pasta


def test_lista_apenas_imagens(tmp_path):
    pasta = tmp_path / "mix"
    _imagem(pasta / "a.png")
    (pasta / "notas.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in list_image_files(pasta)] == ["a.png"]


def test_recorte_usa_o_fator_sobre_a_bbox(origem, tmp_path):
    """bbox 40×40 com fator 2.0 → recorte de 80×80."""
    import cv2

    servico = FaceImageMaterializationService(_GeometriaFalsa((40, 40, 80, 80)))
    destino = tmp_path / "recortes"

    res = servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.CROP, margin_factor=2.0,
    )

    assert res.n_written == 3
    recorte = cv2.imread(str(destino / "f0.png"))
    assert recorte.shape[:2] == (80, 80)


def test_fator_um_devolve_a_bbox_exata(origem, tmp_path):
    import cv2

    servico = FaceImageMaterializationService(_GeometriaFalsa((40, 40, 80, 80)))
    destino = tmp_path / "recortes"

    servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.CROP, margin_factor=1.0,
    )

    assert cv2.imread(str(destino / "f0.png")).shape[:2] == (40, 40)


def test_imagem_sem_face_e_pulada_e_registrada(origem, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())
    destino = tmp_path / "recortes"

    res = servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.CROP, margin_factor=2.0,
    )

    assert res.n_written == 3
    assert res.n_failed == 1
    assert not (destino / "semface.png").exists(), (
        "sem face detectada, nada pode ser gravado — nem a original"
    )
    caminho, motivo = res.failures[0]
    assert "semface" in caminho and "face" in motivo


def test_manifesto_registra_origem_parametros_e_falhas(origem, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())
    destino = tmp_path / "recortes"

    res = servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.CROP, margin_factor=2.5,
    )

    dados = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    assert dados["image_kind"] == "crop"
    assert dados["margin_factor"] == 2.5
    assert dados["source_dir"] == str(origem)
    assert dados["totals"] == {
        "processed": 4, "written": 3, "skipped_existing": 0, "failed": 1,
    }
    assert len(dados["failures"]) == 1, (
        "o que ficou de fora precisa estar no manifesto — senão alguém "
        "conclui depois que a origem só tinha o que sobrou"
    )
    assert all(img["sha256"] for img in dados["images"])


def test_alinhado_produz_112(origem, tmp_path):
    import cv2

    servico = FaceImageMaterializationService(_GeometriaFalsa())
    destino = tmp_path / "alinhadas"

    res = servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.ALIGNED,
    )

    assert res.n_written == 3
    assert cv2.imread(str(destino / "f0.png")).shape[:2] == (112, 112)
    dados = json.loads(res.manifest_path.read_text(encoding="utf-8"))
    assert dados["margin_factor"] is None, "margem não se aplica ao alinhado"


def test_original_nao_e_materializavel(origem, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())
    with pytest.raises(ValueError):
        servico.materialize_directory(
            origem, tmp_path / "x", kind=DatasetImageKind.ORIGINAL,
        )


def test_reexecucao_pula_o_que_ja_existe(origem, tmp_path):
    servico = FaceImageMaterializationService(_GeometriaFalsa())
    destino = tmp_path / "recortes"

    servico.materialize_directory(origem, destino, kind=DatasetImageKind.CROP)
    segunda = servico.materialize_directory(
        origem, destino, kind=DatasetImageKind.CROP,
    )

    assert segunda.n_written == 0
    assert segunda.n_skipped == 3


def test_subpastas_sao_espelhadas(tmp_path):
    origem = tmp_path / "fotos"
    _imagem(origem / "sub" / "a.png")
    servico = FaceImageMaterializationService(_GeometriaFalsa())
    destino = tmp_path / "saida"

    servico.materialize_directory(origem, destino, kind=DatasetImageKind.CROP)

    assert (destino / "sub" / "a.png").exists()
