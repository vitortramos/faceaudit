"""Passada que preenche a geometria de um dataset que não a tem.

A caixa e os cinco pontos nunca chegaram ao parquet: existiam em memória
durante a ingestão e morriam na gravação. Recriar o dataset recuperaria,
mas é caro e refaz embeddings que já estão certos; e enriquecer não
resolve, porque o caminho de enriquecimento manda a face já alinhada ao
encoder e não produz geometria. Daí a passada que roda só o detector.

O invariante sai de graça, e é o ponto central: a detecção roda na imagem
que o dataset GUARDA, então o resultado já está no sistema de coordenadas
dela, seja foto original, recorte ou face alinhada. Nenhuma ramificação por
tipo de dataset.

O que os testes travam:

- a geometria é a da imagem lida, e nada toca em embedding, qualidade ou
  identidade;
- quadro com várias faces é contado como AMBÍGUO, porque a detecção é
  posterior e pode escolher outra face que não a que gerou o embedding;
- imagem ausente não vira exceção: a face fica sem geometria e é contada;
- a orientação já registrada na face é tentada primeiro, e a geometria
  volta ao sistema do arquivo quando a detecção precisou girar;
- o dataset registra que a geometria é de detecção posterior, não da
  extração original — num laudo essa diferença importa.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.services.face_geometry_backfill_service import (
    PROVENANCE_KEY,
    PROVENANCE_VALUE,
    STATUS_AMBIGUA,
    STATUS_JA_TINHA,
    STATUS_SEM_FACE,
    STATUS_SEM_IMAGEM,
    backfill_geometry,
)

_CAIXA = [10.0, 20.0, 110.0, 150.0]
_PONTOS = [[30, 50], [80, 50], [55, 80], [35, 110], [75, 110]]


class _DetectorFalso:
    """Devolve o que foi combinado, e conta as chamadas.

    Aceita dicionários porque o serviço tolera os dois formatos — o
    backend real entrega objetos com `bbox`/`kps`.
    """

    def __init__(self, resultado=None, por_forma=None):
        self._resultado = resultado if resultado is not None else [
            {"bbox": _CAIXA, "kps": _PONTOS},
        ]
        self._por_forma = por_forma or {}
        self.formas_vistas = []

    def detect_faces(self, imagem):
        self.formas_vistas.append(imagem.shape[:2])
        if self._por_forma:
            return self._por_forma.get(imagem.shape[:2], [])
        return self._resultado


def _png(caminho, largura=200, altura=300):
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (largura, altura), (120, 140, 160)).save(caminho)
    return caminho


def _dataset(tmp_path, n=1, *, com_geometria=False, com_imagem=True, **ff):
    faces = []
    for i in range(n):
        nome = f"f{i}.png"
        if com_imagem:
            _png(tmp_path / "imgs" / nome)
        face = FaceInfo(image_path=str(tmp_path / "imgs" / nome))
        face.identity = f"P{i}"
        face.file_name = nome
        base = dict(fiqa_score=1.1, embedding_sepaelv2=np.ones(4, dtype=np.float32))
        base.update(ff)
        if com_geometria:
            base.setdefault("bbox", np.asarray(_CAIXA, dtype=np.float32))
            base.setdefault("keypoints", np.asarray(_PONTOS, dtype=np.float32))
        face.ff_info = FFInfo(**base)
        faces.append(face)
    return FaceDataset(name="ds", faces=faces, path_mode=PathMode.ABSOLUTE)


# ── O caso simples ───────────────────────────────────────────────────────

def test_preenche_a_geometria(tmp_path):
    ds = _dataset(tmp_path, 2)

    r = backfill_geometry(ds, _DetectorFalso())

    assert r.total == 2 and r.preenchidas == 2
    for face in ds.faces:
        assert np.allclose(face.ff_info.bbox, _CAIXA)
        assert face.ff_info.keypoints.shape == (5, 2)
        assert np.allclose(face.ff_info.keypoints, _PONTOS)


def test_nao_toca_no_resto(tmp_path):
    """A passada é aditiva: embedding, qualidade e identidade ficam."""
    ds = _dataset(tmp_path, 1, fiqa_score=2.5)
    antes_emb = ds.faces[0].ff_info.embedding_sepaelv2.copy()

    backfill_geometry(ds, _DetectorFalso())

    ff = ds.faces[0].ff_info
    assert ff.fiqa_score == 2.5
    assert np.allclose(ff.embedding_sepaelv2, antes_emb)
    assert ds.faces[0].identity == "P0"


def test_detecta_na_imagem_que_o_dataset_guarda(tmp_path):
    """O invariante: a régua é a da imagem lida, qualquer que seja ela.
    É isso que dispensa ramificar por tipo de dataset."""
    ds = _dataset(tmp_path, 1)
    detector = _DetectorFalso()

    backfill_geometry(ds, detector)

    # 300 de altura e 200 de largura, a imagem gravada — e não 112×112
    # nem a foto de origem.
    assert detector.formas_vistas == [(300, 200)]


def test_quem_ja_tem_nao_e_mexido(tmp_path):
    ds = _dataset(tmp_path, 1, com_geometria=True)
    outro = _DetectorFalso([{"bbox": [1, 1, 2, 2], "kps": _PONTOS}])

    r = backfill_geometry(ds, outro)

    assert r.ja_tinham == 1 and r.preenchidas == 0
    assert np.allclose(ds.faces[0].ff_info.bbox, _CAIXA)


def test_sobrescrever_quando_pedido(tmp_path):
    ds = _dataset(tmp_path, 1, com_geometria=True)
    nova = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]

    r = backfill_geometry(
        ds, _DetectorFalso([{"bbox": [1, 1, 9, 9], "kps": nova}]),
        overwrite=True,
    )

    assert r.preenchidas == 1
    assert np.allclose(ds.faces[0].ff_info.keypoints, nova)


# ── Ambiguidade e falhas ─────────────────────────────────────────────────

def test_varias_faces_viram_ambiguas(tmp_path):
    """A detecção é POSTERIOR à que produziu o embedding. Num quadro com
    várias pessoas ela pode escolher outra, e a dúvida tem de aparecer no
    relatório em vez de se dissolver."""
    ds = _dataset(tmp_path, 1)
    duas = [
        {"bbox": [0, 0, 10, 10], "kps": _PONTOS},
        {"bbox": _CAIXA, "kps": _PONTOS},
    ]

    r = backfill_geometry(ds, _DetectorFalso(duas))

    assert r.ambiguas == 1 and r.preenchidas == 0
    assert [d.status for d in r.pedem_conferencia] == [STATUS_AMBIGUA]
    assert r.detalhes[0].n_faces_detected == 2


def test_ambiguidade_usa_a_confianca_gravada(tmp_path):
    """A face escolhida na extração original é a de `det_score` mais
    próximo do que o dataset guarda — é a melhor pista disponível."""
    ds = _dataset(tmp_path, 1, det_score=0.91)

    class _Face:
        def __init__(self, bbox, det):
            self.bbox, self.kps, self.det_score = bbox, _PONTOS, det

    candidatas = [_Face([0, 0, 300, 300], 0.50), _Face(_CAIXA, 0.90)]

    backfill_geometry(ds, _DetectorFalso(candidatas))

    # A de maior área tem confiança 0,50; a escolhida é a de 0,90.
    assert np.allclose(ds.faces[0].ff_info.bbox, _CAIXA)


def test_sem_confianca_gravada_fica_a_maior(tmp_path):
    ds = _dataset(tmp_path, 1)
    candidatas = [
        {"bbox": [0, 0, 5, 5], "kps": _PONTOS},
        {"bbox": [0, 0, 100, 100], "kps": _PONTOS},
    ]

    backfill_geometry(ds, _DetectorFalso(candidatas))

    assert np.allclose(ds.faces[0].ff_info.bbox, [0, 0, 100, 100])


def test_sem_face_detectada_e_contada(tmp_path):
    ds = _dataset(tmp_path, 1)

    r = backfill_geometry(ds, _DetectorFalso([]))

    assert r.sem_face == 1
    assert ds.faces[0].ff_info.keypoints is None
    assert [d.status for d in r.pedem_conferencia] == [STATUS_SEM_FACE]


def test_imagem_ausente_nao_quebra(tmp_path):
    ds = _dataset(tmp_path, 2, com_imagem=False)

    r = backfill_geometry(ds, _DetectorFalso())

    assert r.sem_imagem == 2 and r.preenchidas == 0
    assert all(d.status == STATUS_SEM_IMAGEM for d in r.detalhes)


def test_detector_que_explode_conta_como_sem_face(tmp_path):
    class _Explode:
        def detect_faces(self, imagem):
            raise RuntimeError("modelo não carregou")

    ds = _dataset(tmp_path, 1)

    r = backfill_geometry(ds, _Explode())

    assert r.sem_face == 1


# ── Orientação ───────────────────────────────────────────────────────────

def test_gira_quando_a_imagem_como_esta_nao_rende(tmp_path):
    """Foto gravada de lado só tem face detectável depois de girar."""
    ds = _dataset(tmp_path, 1)
    # 300×200 girada 90° fica 200×300.
    detector = _DetectorFalso(
        por_forma={(200, 300): [{"bbox": _CAIXA, "kps": _PONTOS}]},
    )

    r = backfill_geometry(ds, detector, rotations=(90, 270, 180))

    assert r.preenchidas == 1
    assert ds.faces[0].ff_info.detection_rotation == 90


def test_a_geometria_volta_ao_sistema_do_arquivo(tmp_path):
    """A detecção rodou num quadro girado, mas `image_path` aponta para o
    arquivo como ele está gravado. Guardar a caixa do quadro girado faria
    o recorte sair da região errada e o desenho cair fora da face."""
    ds = _dataset(tmp_path, 1)
    detector = _DetectorFalso(
        por_forma={(200, 300): [{"bbox": _CAIXA, "kps": _PONTOS}]},
    )

    backfill_geometry(ds, detector, rotations=(90,))

    caixa = ds.faces[0].ff_info.bbox
    # No quadro girado a caixa ia até x=110 numa largura de 300; de volta
    # ao arquivo (200 de largura) ela não pode exceder o limite dele.
    assert 0 <= caixa[0] <= 200 and 0 <= caixa[2] <= 200
    assert not np.allclose(caixa, _CAIXA), (
        "a caixa foi gravada no sistema do quadro girado"
    )


def test_a_orientacao_registrada_e_tentada_primeiro(tmp_path):
    """A face sabe em que giro foi encontrada; usar isso evita refazer a
    busca inteira e é o que garante o acerto em dataset criado com giro."""
    ds = _dataset(tmp_path, 1, detection_rotation=270)
    detector = _DetectorFalso(
        por_forma={(200, 300): [{"bbox": _CAIXA, "kps": _PONTOS}]},
    )

    backfill_geometry(ds, detector, rotations=(90, 180))

    # 270° é a primeira forma consultada, antes da imagem como está.
    assert detector.formas_vistas[0] == (200, 300)
    assert ds.faces[0].ff_info.detection_rotation == 270


# ── Proveniência e relatório ─────────────────────────────────────────────

def test_o_dataset_registra_que_a_geometria_e_posterior(tmp_path):
    """Num laudo, geometria de detecção posterior não é a mesma coisa que
    a geometria da extração que produziu o embedding."""
    ds = _dataset(tmp_path, 1)

    backfill_geometry(ds, _DetectorFalso())

    assert ds.provenance_metadata.get(PROVENANCE_KEY) == PROVENANCE_VALUE
    assert "geometry_backfill_at" in ds.provenance_metadata


def test_sem_mudanca_nao_marca_proveniencia(tmp_path):
    ds = _dataset(tmp_path, 1, com_geometria=True)

    r = backfill_geometry(ds, _DetectorFalso())

    assert not r.mudou
    assert PROVENANCE_KEY not in ds.provenance_metadata


def test_a_proveniencia_sobrevive_ao_parquet(tmp_path):
    ds = _dataset(tmp_path, 1)
    backfill_geometry(ds, _DetectorFalso())

    caminho = tmp_path / "ds.parquet"
    ds.to_parquet(caminho)
    de_volta = FaceDataset.from_parquet(caminho)

    assert de_volta.provenance_metadata.get(PROVENANCE_KEY) == PROVENANCE_VALUE
    assert np.allclose(de_volta.faces[0].ff_info.keypoints, _PONTOS)


def test_o_resumo_conta_tudo(tmp_path):
    ds = _dataset(tmp_path, 1)
    ds.faces.append(FaceInfo(image_path=str(tmp_path / "nao_existe.png")))
    ds.faces[-1].file_name = "nao_existe.png"

    r = backfill_geometry(ds, _DetectorFalso())

    assert "2 face(s)" in r.resumo()
    assert "1 preenchida" in r.resumo()
    assert "1 sem imagem" in r.resumo()


def test_dataset_vazio_nao_quebra():
    r = backfill_geometry(FaceDataset(name="v", faces=[]), _DetectorFalso())

    assert r.total == 0 and not r.mudou


def test_progresso_e_reportado(tmp_path):
    ds = _dataset(tmp_path, 3)
    mensagens = []

    backfill_geometry(
        ds, _DetectorFalso(),
        progress_callback=lambda m, p: mensagens.append((m, p)),
    )

    assert mensagens
    assert mensagens[-1][1] == 100
