"""A caixa e os cinco pontos da face sobrevivem ao parquet.

O que faltava: `FFInfo.to_dict(exclude_heavy=True)` tira `bbox`,
`keypoints` e `aligned_face` junto com os embeddings, e nada os reinjetava
na gravação. A caixa e os pontos existiam só em memória, durante a
ingestão, e morriam no `to_parquet`. Quem abria o parquet depois não tinha
como marcar a face na imagem, nem conferir QUAL face de um quadro com
várias produziu o embedding — num retrato de grupo é a diferença entre
conferir o exame e redetectar na esperança de cair na mesma face.

Dois vizinhos tinham o mesmo defeito, e pior: `detection_rotation` e
`detection_det_size` eram ESCRITOS (estão em `_FF_ATTR_COLUMNS`) e nunca
relidos, então um ciclo abrir → salvar apagava a coluna. A rotação é o que
explica por que a caixa está no sistema do arquivo enquanto a pose veio do
quadro corrigido; perdê-la em silêncio desmonta a leitura do registro.

E havia uma ambiguidade a resolver antes de gravar qualquer coisa: o slot
`keypoints` guardava DUAS réguas diferentes sem nada no dado dizendo qual.
O fluxo de fotos põe os pontos no sistema do arquivo; o importador de vídeo
põe os do recorte 112×112, que são o que os modelos KPRPE consomem.
Gravar as duas sob o mesmo nome de coluna produziria pontos empilhados no
canto da foto — e, pior, o consumidor do KPRPE lia o slot assumindo 112×112,
de modo que um dataset de recortes vindo de fotos alimentava o modelo com a
régua errada e produzia embedding errado, sem erro aparente.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo, geometry_from_row
from faceaudit.models.ff_info import FFInfo

_BBOX = np.array([10.0, 20.0, 110.0, 150.0], dtype=np.float32)
_KPS = np.array(
    [[30, 50], [80, 50], [55, 80], [35, 110], [75, 110]], dtype=np.float32,
)


def _face(**ff_kwargs):
    f = FaceInfo(image_path="/imgs/a.png")
    f.identity = "P1"
    f.hash = "h1"
    base = dict(
        fiqa_score=1.2, det_score=0.93,
        embedding_sepaelv2=np.ones(8, dtype=np.float32),
    )
    base.update(ff_kwargs)
    f.ff_info = FFInfo(**base)
    return f


def _ida_e_volta(tmp_path, face, nome="ds"):
    caminho = tmp_path / f"{nome}.parquet"
    FaceDataset(name=nome, faces=[face]).to_parquet(caminho)
    return caminho, FaceDataset.from_parquet(caminho).faces[0]


# ── A geometria sobrevive ────────────────────────────────────────────────

def test_caixa_e_pontos_sobrevivem(tmp_path):
    _, de_volta = _ida_e_volta(tmp_path, _face(bbox=_BBOX, keypoints=_KPS))

    assert np.allclose(de_volta.ff_info.bbox, _BBOX)
    assert de_volta.ff_info.keypoints.shape == (5, 2)
    assert np.allclose(de_volta.ff_info.keypoints, _KPS)


def test_as_colunas_aparecem_no_parquet(tmp_path):
    import pyarrow.parquet as pq

    caminho, _ = _ida_e_volta(tmp_path, _face(bbox=_BBOX, keypoints=_KPS))

    nomes = set(pq.ParquetFile(caminho).schema_arrow.names)
    assert {"bbox", "keypoints"} <= nomes


def test_geometria_ausente_nao_gera_coluna(tmp_path):
    """Coluna de nulos depois se confunde com dado perdido."""
    import pyarrow.parquet as pq

    caminho, de_volta = _ida_e_volta(tmp_path, _face())

    nomes = set(pq.ParquetFile(caminho).schema_arrow.names)
    assert "bbox" not in nomes and "keypoints" not in nomes
    assert de_volta.ff_info.bbox is None
    assert de_volta.ff_info.keypoints is None


def test_dois_ciclos_nao_degradam(tmp_path):
    """Abrir e salvar de novo é o caminho de todo trabalho real."""
    caminho, primeiro = _ida_e_volta(tmp_path, _face(bbox=_BBOX, keypoints=_KPS))
    segundo_caminho = tmp_path / "outra.parquet"
    FaceDataset(name="outra", faces=[primeiro]).to_parquet(segundo_caminho)
    segundo = FaceDataset.from_parquet(segundo_caminho).faces[0]

    assert np.allclose(segundo.ff_info.bbox, _BBOX)
    assert np.allclose(segundo.ff_info.keypoints, _KPS)


def test_parquet_antigo_sem_as_colunas_carrega(tmp_path):
    """Todos os datasets já existentes estão nessa situação."""
    _, de_volta = _ida_e_volta(tmp_path, _face())

    assert de_volta.ff_info is not None
    assert de_volta.ff_info.keypoints is None


# ── Os vizinhos que eram escritos e nunca relidos ───────────────────────

def test_rotacao_da_deteccao_sobrevive(tmp_path):
    _, de_volta = _ida_e_volta(
        tmp_path, _face(detection_rotation=90, detection_det_size=640),
    )

    assert de_volta.ff_info.detection_rotation == 90
    assert de_volta.ff_info.detection_det_size == 640


def test_a_rotacao_volta_como_inteiro(tmp_path):
    """O parquet devolve float quando a coluna tem nulos, e `90.0` não
    compara igual aos inteiros de `ROTACOES_PADRAO` — a orientação
    registrada deixaria de ser tentada primeiro no enriquecimento, que é
    exatamente o que ela existe para garantir."""
    _, de_volta = _ida_e_volta(tmp_path, _face(detection_rotation=90))

    assert isinstance(de_volta.ff_info.detection_rotation, int)


def test_a_rotacao_sobrevive_a_dois_ciclos(tmp_path):
    """Era aqui que ela sumia: escrita na primeira gravação, não relida, e
    apagada na segunda."""
    _, primeiro = _ida_e_volta(tmp_path, _face(detection_rotation=270))
    segundo_caminho = tmp_path / "outra.parquet"
    FaceDataset(name="outra", faces=[primeiro]).to_parquet(segundo_caminho)
    segundo = FaceDataset.from_parquet(segundo_caminho).faces[0]

    assert segundo.ff_info.detection_rotation == 270


# ── O invariante: a geometria é da imagem que o dataset guarda ──────────

def test_recorte_translada_a_geometria():
    """O recorte grava um PNG novo e reescreve `image_path`; a caixa e os
    pontos têm de ir junto, senão apontam para fora da imagem gravada."""
    from faceaudit.services.face_geometry_frame_service import (
        transladar_geometria,
    )

    ff = FFInfo(bbox=_BBOX.copy(), keypoints=_KPS.copy())

    transladar_geometria(ff, 10, 20, largura=100, altura=130)

    assert np.allclose(ff.bbox, [0.0, 0.0, 100.0, 130.0])
    assert np.allclose(ff.keypoints[:, 0], _KPS[:, 0] - 10)
    assert np.allclose(ff.keypoints[:, 1], _KPS[:, 1] - 20)


def test_a_caixa_e_presa_ao_recorte():
    """A margem é cortada nas bordas da foto, então a caixa transladada
    pode passar do limite por alguns pixels."""
    from faceaudit.services.face_geometry_frame_service import (
        transladar_geometria,
    )

    ff = FFInfo(bbox=np.array([5.0, 5.0, 500.0, 500.0], dtype=np.float32))

    transladar_geometria(ff, 10, 10, largura=50, altura=60)

    assert ff.bbox[0] == 0.0 and ff.bbox[1] == 0.0
    assert ff.bbox[2] == 50.0 and ff.bbox[3] == 60.0


def test_os_pontos_nao_sao_presos():
    """Ponto fora do recorte é informação verdadeira — olho fora do quadro
    em face muito de perfil. Prendê-lo inventaria uma posição."""
    from faceaudit.services.face_geometry_frame_service import (
        transladar_geometria,
    )

    ff = FFInfo(keypoints=np.array([[1.0, 1.0]] * 5, dtype=np.float32))

    transladar_geometria(ff, 10, 10, largura=50, altura=50)

    assert np.all(ff.keypoints < 0)


def test_alinhado_descarta_em_vez_de_traduzir_errado():
    """A transformação de similaridade fica dentro do `detect_and_align` e
    não está disponível a quem materializa; guardar a geometria da foto
    original sob o mesmo nome seria guardar outra régua."""
    from faceaudit.services.face_geometry_frame_service import (
        descartar_geometria,
    )

    ff = FFInfo(bbox=_BBOX.copy(), keypoints=_KPS.copy())

    descartar_geometria(ff)

    assert ff.bbox is None and ff.keypoints is None


def test_o_recorte_do_pipeline_converte():
    """Prova no ponto real: a materialização do recorte chama a conversão."""
    import inspect

    from faceaudit.pipelines.feature_extraction_pipeline import (
        FeatureExtractionPipeline,
    )

    fonte = inspect.getsource(
        FeatureExtractionPipeline._materialize_face_crops
    )
    assert "transladar_geometria" in fonte


def test_a_materializacao_alinhada_descarta():
    import inspect

    from faceaudit.pipelines.feature_extraction_pipeline import (
        FeatureExtractionPipeline,
    )

    fonte = inspect.getsource(
        FeatureExtractionPipeline._materialize_aligned_crops
    )
    assert "descartar_geometria" in fonte


# ── O conversor de leitura ──────────────────────────────────────────────

def test_leitura_descarta_tamanho_errado():
    """Meia caixa desenharia um retângulo errado sobre a face, o que é pior
    que não desenhar nada."""
    assert geometry_from_row([1.0, 2.0], None) == {}
    assert geometry_from_row(None, [1.0, 2.0, 3.0]) == {}


def test_leitura_tolera_vazio():
    assert geometry_from_row(None, None) == {}


def test_leitura_reconstroi_a_forma():
    saida = geometry_from_row(list(_BBOX), list(_KPS.reshape(-1)))

    assert saida["bbox"].shape == (4,)
    assert saida["keypoints"].shape == (5, 2)
    assert np.allclose(saida["keypoints"], _KPS)


# ── O catálogo descreve as colunas novas ────────────────────────────────

def test_as_colunas_estao_no_catalogo():
    from faceaudit.models.column_catalog import DatasetType, get_default_specs

    specs = get_default_specs(DatasetType.FACE)

    for nome in ("bbox", "keypoints"):
        assert nome in specs, nome
        assert not specs[nome].visible_default, (
            f"{nome} não é coluna de leitura humana"
        )


# ── O consumidor do KPRPE ───────────────────────────────────────────────

def test_kprpe_le_so_a_casa_propria():
    """`ff_info.keypoints` é o sistema da imagem do dataset. Num dataset de
    recortes vindo de fotos são as coordenadas da foto original, e alimentar
    o KPRPE com elas produzia embedding errado sem erro aparente. Os pontos
    112×112 têm casa própria."""
    import json

    from faceaudit.services.ff_service import (
        ALIGNED_KEYPOINTS_EXTRA_KEY, _aligned_keypoints_from_face,
    )

    do_dataset = FaceInfo(image_path="x.png", ff_info=FFInfo(keypoints=_KPS))
    assert _aligned_keypoints_from_face(do_dataset) is None

    com_casa = FaceInfo(image_path="x.png")
    com_casa.extra_data = {
        ALIGNED_KEYPOINTS_EXTRA_KEY: json.dumps(_KPS.tolist()),
    }
    assert _aligned_keypoints_from_face(com_casa) is not None
