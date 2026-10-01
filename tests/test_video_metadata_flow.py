"""Metadados de vídeo de ponta a ponta: rotação, manifesto e colunas.

Três coisas que este fluxo resolve, e que os testes travam.

**A rotação é aplicada num ponto único.** Três caminhos leem quadro — o
scrubber da interface, o quadro-âncora no worker e o laço da extração. Se
discordassem na orientação, o usuário escolheria o recorte olhando uma
imagem e o sistema extrairia de outra, sem nada no resultado denunciando a
diferença.

**Os metadados são lidos na EXTRAÇÃO, não na importação.** O manifesto não
guarda o caminho do vídeo (o run é portátil), então ler na importação
dependeria de o arquivo ainda estar no mesmo lugar. Capturar junto com o
hash é também a semântica de custódia correta.

**Cada face casa com o seu vídeo pelo SHA-256.** Um run pode ter vários
vídeos, e dois arquivos podem ter o mesmo nome em pastas diferentes.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.services.image_rotation_service import girar_imagem
from faceaudit.services.video_capture_metadata_service import (
    ORIGEM_INICIO_MAIS_QUADRO,
    VideoCaptureMetadata,
)
from faceaudit.services.video_frame_face_extraction_service import (
    VideoFrameFaceExtractionService,
)

ALTURA, LARGURA = 60, 100


class _MetadataFalso:
    def __init__(self, tags):
        self.tags = dict(tags)
        self.error = None

    def get_first(self, *nomes):
        for nome in nomes:
            for chave, valor in self.tags.items():
                if chave == nome or chave.split(":")[-1] == nome:
                    return valor
        return None


class _ExifFalso:
    def __init__(self, tags, chamadas=None):
        self._tags = tags
        self.chamadas = chamadas if chamadas is not None else []

    def extract(self, caminho):
        self.chamadas.append(str(caminho))
        return _MetadataFalso(self._tags)


def _servico(tags=None, com_exif=True):
    svc = object.__new__(VideoFrameFaceExtractionService)
    svc._exif = _ExifFalso(tags or {}) if com_exif else None
    svc._meta_container = {}
    return svc


def _quadro():
    return np.arange(ALTURA * LARGURA * 3, dtype=np.uint8).reshape(ALTURA, LARGURA, 3)


# ── Leitura única por vídeo ──────────────────────────────────────────────

def test_metadados_lidos_uma_vez_por_video():
    """É consultado a cada quadro: uma chamada por quadro inviabilizaria a
    extração de um vídeo de milhares de quadros."""
    svc = _servico({"Rotation": 90})

    for _ in range(50):
        svc.rotacao_do_container("C:/v/a.mp4")

    assert len(svc._exif.chamadas) == 1


def test_videos_distintos_sao_lidos_separadamente():
    svc = _servico({"Rotation": 90})

    svc.rotacao_do_container("C:/v/a.mp4")
    svc.rotacao_do_container("C:/v/b.mp4")

    assert svc._exif.chamadas == ["C:/v/a.mp4", "C:/v/b.mp4"]


def test_sem_leitor_de_metadados_nada_e_lido():
    """Comportamento histórico preservado: quadro como o OpenCV entrega."""
    svc = _servico(com_exif=False)

    assert svc.rotacao_do_container("C:/v/a.mp4") == 0
    assert svc.metadados_do_container("C:/v/a.mp4").tags == {}


# ── Aplicação da rotação ─────────────────────────────────────────────────

@pytest.mark.parametrize("graus", [90, 180, 270])
def test_quadro_e_girado_conforme_a_etiqueta(graus):
    svc = _servico({"Rotation": graus})
    quadro = _quadro()

    girado = svc._aplicar_rotacao(quadro, "C:/v/a.mp4")

    assert np.array_equal(girado, girar_imagem(quadro, graus))


def test_sem_etiqueta_o_quadro_passa_intacto():
    svc = _servico({})
    quadro = _quadro()

    assert svc._aplicar_rotacao(quadro, "C:/v/a.mp4") is quadro


def test_retrato_deitado_volta_a_ficar_de_pe():
    """O caso concreto: celular em retrato grava deitado com etiqueta 90°."""
    svc = _servico({"Rotation": 90})
    deitado = np.zeros((1080, 1920, 3), dtype=np.uint8)

    de_pe = svc._aplicar_rotacao(deitado, "C:/v/retrato.mp4")

    assert de_pe.shape[:2] == (1920, 1080)


def test_quadro_nulo_nao_estoura():
    assert _servico({"Rotation": 90})._aplicar_rotacao(None, "C:/v/a.mp4") is None


def test_os_tres_caminhos_usam_o_mesmo_ponto():
    """Garante que ninguém acrescentou um caminho de leitura sem rotação."""
    import inspect

    fonte = inspect.getsource(VideoFrameFaceExtractionService)
    # Uma chamada em cada: read_frame, _read_frame_uncached e o laço.
    assert fonte.count("self._aplicar_rotacao(") == 3, (
        "todo caminho que lê quadro tem de passar pelo ponto único — se a "
        "prévia e a extração discordarem na orientação, o usuário escolhe "
        "olhando uma imagem e o sistema extrai de outra"
    )


# ── Resolução efetiva ────────────────────────────────────────────────────

def test_resolucao_efetiva_troca_com_a_rotacao():
    from faceaudit.models.video.video_info import VideoInfo

    info = VideoInfo(
        path="x", filename="x.mp4", sha256="abc", file_size_bytes=1,
        fps_reported=30.0, frame_count_reported=10, duration_reported_s=1.0,
        resolution=(1920, 1080), opencv_version="4", container_rotation=90,
    )

    assert info.resolution == (1920, 1080), "o que o decodificador reporta"
    assert info.resolution_effective == (1080, 1920), "o que o sistema vê"


def test_manifesto_leva_as_duas_resolucoes():
    """Confundi-las é que geraria engano: uma descreve o arquivo, a outra o
    que foi efetivamente detectado."""
    from faceaudit.models.video.video_info import VideoInfo

    m = VideoInfo(
        path="x", filename="x.mp4", sha256="abc", file_size_bytes=1,
        fps_reported=30.0, frame_count_reported=10, duration_reported_s=1.0,
        resolution=(1920, 1080), opencv_version="4", container_rotation=270,
    ).to_manifest_dict()

    assert m["resolution"] == [1920, 1080]
    assert m["resolution_effective"] == [1080, 1920]
    assert m["container_rotation"] == 270


# ── Manifesto ────────────────────────────────────────────────────────────

class _InfoFalsa:
    def __init__(self, sha):
        self.sha256 = sha


def test_manifesto_indexa_pelo_sha_do_video():
    """Nome de arquivo não serviria: dois vídeos podem ter o mesmo nome em
    pastas diferentes."""
    svc = _servico({"Rotation": 90, "CreationDate": "2026:09:20 14:32:10-03:00"})

    saida = svc._capture_metadata_manifest(
        ["C:/v/a.mp4"], [_InfoFalsa("sha-a")], True,
    )

    assert list(saida) == ["sha-a"]
    assert saida["sha-a"]["container_rotation"] == 90
    assert saida["sha-a"]["timezone_declared"] is True


def test_manifesto_sem_a_opcao_guarda_so_o_que_a_extracao_usou():
    """A rotação é sempre lida; o dump completo só interessa a quem pediu."""
    svc = _servico({"Rotation": 90, "QuickTime:Make": "ACME"})

    saida = svc._capture_metadata_manifest(
        ["C:/v/a.mp4"], [_InfoFalsa("sha-a")], False,
    )

    assert "tags" not in saida["sha-a"]
    assert saida["sha-a"]["container_rotation"] == 90, (
        "a rotação entra de qualquer forma — foi aplicada aos quadros"
    )


def test_manifesto_com_a_opcao_guarda_o_dump():
    svc = _servico({"QuickTime:Make": "ACME"})

    saida = svc._capture_metadata_manifest(
        ["C:/v/a.mp4"], [_InfoFalsa("sha-a")], True,
    )

    assert saida["sha-a"]["tags"] == {"QuickTime:Make": "ACME"}


def test_varios_videos_no_mesmo_run():
    svc = object.__new__(VideoFrameFaceExtractionService)
    svc._meta_container = {}

    class _ExifPorVideo:
        chamadas: list = []

        def extract(self, caminho):
            self.chamadas.append(str(caminho))
            graus = 90 if "a.mp4" in str(caminho) else 0
            return _MetadataFalso({"Rotation": graus})

    svc._exif = _ExifPorVideo()
    saida = svc._capture_metadata_manifest(
        ["C:/v/a.mp4", "C:/v/b.mp4"],
        [_InfoFalsa("sha-a"), _InfoFalsa("sha-b")], False,
    )

    assert saida["sha-a"]["container_rotation"] == 90
    assert saida["sha-b"]["container_rotation"] == 0


# ── Importador: colunas por face ─────────────────────────────────────────

def _manifesto(sha="sha-a", com_tags=True, com_fuso=True):
    meta = VideoCaptureMetadata(
        tags={"QuickTime:Make": "ACME", "QuickTime:Model": "Cam 9"} if com_tags else {},
        start_datetime=(
            VideoCaptureMetadata.from_manifest_entry(
                {"start_datetime_raw": "2026:09:20 14:32:10-03:00"}
            ).start_datetime if com_fuso else None
        ),
        start_raw="2026:09:20 14:32:10-03:00" if com_fuso else "2026:09:20 14:32:10",
    )
    return {"capture_metadata": {sha: meta.to_manifest_entry(incluir_tags=com_tags)}}


def test_importador_resolve_as_colunas_por_sha():
    from faceaudit.services.video_run_importer import VideoRunImporter

    mapa = VideoRunImporter._capture_columns_by_video(_manifesto())

    assert list(mapa) == ["sha-a"]
    meta, colunas = mapa["sha-a"]
    assert meta.tem_inicio_confiavel is True
    assert colunas["camera_make"] == "ACME"


def test_importador_tolera_manifesto_sem_metadados():
    """Run antigo, gravado antes desta funcionalidade."""
    from faceaudit.services.video_run_importer import VideoRunImporter

    assert VideoRunImporter._capture_columns_by_video({}) == {}
    assert VideoRunImporter._capture_columns_by_video(
        {"capture_metadata": None},
    ) == {}


def test_importador_calcula_a_data_sem_o_dump():
    """Rotação gravada sem a opção: ainda assim o início está no manifesto,
    e a data por quadro é calculável."""
    from faceaudit.services.video_capture_metadata_service import columns_for_frame
    from faceaudit.services.video_run_importer import VideoRunImporter

    mapa = VideoRunImporter._capture_columns_by_video(_manifesto(com_tags=False))
    meta, colunas = mapa["sha-a"]

    assert colunas == {}, "sem dump não há colunas a promover"
    saida = columns_for_frame(meta, 45.0, colunas)
    assert saida["capture_datetime"].startswith("2026-09-20T14:32:55")


def test_faces_de_instantes_diferentes_recebem_datas_diferentes():
    from faceaudit.services.video_capture_metadata_service import columns_for_frame
    from faceaudit.services.video_run_importer import VideoRunImporter

    meta, colunas = VideoRunImporter._capture_columns_by_video(_manifesto())["sha-a"]

    primeira = columns_for_frame(meta, 0.0, colunas)["capture_datetime"]
    tardia = columns_for_frame(meta, 3600.0, colunas)["capture_datetime"]

    assert primeira != tardia
    assert "15:32:10" in tardia
    assert all(
        c["capture_datetime_origin"] == ORIGEM_INICIO_MAIS_QUADRO
        for c in (
            columns_for_frame(meta, 0.0, colunas),
            columns_for_frame(meta, 3600.0, colunas),
        )
    )


def test_sem_fuso_declarado_a_data_do_quadro_fica_vazia():
    from faceaudit.services.video_capture_metadata_service import columns_for_frame
    from faceaudit.services.video_run_importer import VideoRunImporter

    meta, colunas = VideoRunImporter._capture_columns_by_video(
        _manifesto(com_fuso=False),
    )["sha-a"]

    saida = columns_for_frame(meta, 45.0, colunas)

    assert "capture_datetime" not in saida
    assert saida["capture_datetime_raw"] == "2026:09:20 14:32:10"


# ── Especificações das colunas ───────────────────────────────────────────

def test_dataset_de_video_reusa_as_colunas_do_catalogo():
    """Redefini-las aqui daria rótulo e categoria diferentes das mesmas
    colunas num dataset de fotos — e as duas se encontram num merge."""
    from faceaudit.models.column_catalog import DatasetType, get_default_specs
    from faceaudit.services.video_run_importer import build_video_column_specs

    video = build_video_column_specs()
    foto = get_default_specs(DatasetType.FACE)

    # Igualdade, não identidade: `get_default_specs` constrói objetos novos
    # a cada chamada. O que precisa casar é a DEFINIÇÃO — rótulo, categoria,
    # papel —, que é o que um merge entre dataset de vídeo e de fotos exige.
    assert video["capture_datetime"] == foto["capture_datetime"]
    assert video["exif_json"] == foto["exif_json"]
    assert video["image_bits_per_pixel"] == foto["image_bits_per_pixel"]


def test_colunas_do_video_nao_sobrescrevem_as_da_extracao():
    from faceaudit.services.video_run_importer import build_video_column_specs

    specs = build_video_column_specs()

    assert specs["timestamp_s_from_fps"].label == "Timestamp (s)"


# ── Proveniência ─────────────────────────────────────────────────────────

def test_proveniencia_registra_a_rotacao_aplicada():
    """Sem isto ninguém reconstrói por que os recortes estão de pé e o vídeo,
    aberto num player, aparece deitado."""
    from faceaudit.services.video_run_importer import VideoRunImporter

    meta = VideoRunImporter._build_provenance_metadata({
        "capture_metadata": {
            "sha-a": {"container_rotation": 90},
            "sha-b": {"container_rotation": 0},
        },
    })

    assert meta["container_rotation_applied"] == "90"


def test_proveniencia_omite_quando_nao_houve_rotacao():
    from faceaudit.services.video_run_importer import VideoRunImporter

    meta = VideoRunImporter._build_provenance_metadata({
        "capture_metadata": {"sha-a": {"container_rotation": 0}},
    })

    assert "container_rotation_applied" not in meta


def test_proveniencia_lista_rotacoes_distintas():
    from faceaudit.services.video_run_importer import VideoRunImporter

    meta = VideoRunImporter._build_provenance_metadata({
        "capture_metadata": {
            "sha-a": {"container_rotation": 90},
            "sha-b": {"container_rotation": 270},
        },
    })

    assert meta["container_rotation_applied"] == "90,270"


# ── A opção na tela ──────────────────────────────────────────────────────

def test_opcao_desligada_por_padrao_no_config():
    from faceaudit.models.video.video_extraction_config import VideoExtractionConfig

    import dataclasses

    campo = next(
        f for f in dataclasses.fields(VideoExtractionConfig)
        if f.name == "extract_exif"
    )
    assert campo.default is False, (
        "custa uma leitura por vídeo; opt-in como nos outros fluxos"
    )
