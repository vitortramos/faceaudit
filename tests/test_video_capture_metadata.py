"""Metadados de captura de vídeo e taxa de compressão.

Duas coisas que a foto não tem.

**Data por quadro.** Todas as faces de um vídeo vêm do mesmo arquivo, mas
cada uma tem o seu instante. Com um início de gravação confiável, a data de
captura de cada face é `início + instante` — e isso responde ao que a data
do arquivo não responde: quando esta pessoa foi filmada. Num vídeo de duas
horas, a primeira e a última face distam duas horas.

**Fuso.** A regra é a mesma das fotos e é o que mais importa travar: sem
deslocamento EXPLÍCITO, não há data normalizada. O `CreateDate` do QuickTime
quase sempre está em UTC sem declarar, e tratá-lo como hora local erraria
por horas — em silêncio, numa coluna que depois aparece em laudo.

**Compressão.** Duas medidas que coexistem de propósito: a que a câmera
declarou na captura e a calculada do arquivo como ele está hoje. A
divergência entre elas é sinal de recompressão, não inconsistência.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from faceaudit.services.video_capture_metadata_service import (
    ORIGEM_INICIO_MAIS_QUADRO,
    VideoCaptureMetadata,
    columns_for_frame,
    datetime_do_quadro,
    probe_video_capture_metadata,
)


class _MetadataFalso:
    """Imita `ImageMetadata`: `get_first` resolve nome com prefixo de grupo."""

    def __init__(self, tags: dict, error=None):
        self.tags = dict(tags)
        self.error = error

    def get_first(self, *nomes):
        for nome in nomes:
            for chave, valor in self.tags.items():
                if chave == nome or chave.split(":")[-1] == nome:
                    return valor
        return None


class _ServicoFalso:
    def __init__(self, tags: dict, estoura=False):
        self._tags = tags
        self._estoura = estoura

    def extract(self, caminho):
        if self._estoura:
            raise RuntimeError("ExifTool não respondeu")
        return _MetadataFalso(self._tags)


def _probe(tags, **kw):
    return probe_video_capture_metadata(_ServicoFalso(tags, **kw), "C:/v/a.mp4")


# ── Início da gravação e fuso ────────────────────────────────────────────

def test_etiqueta_com_fuso_explicito_e_aceita():
    meta = _probe({"QuickTime:CreationDate": "2026:09:20 14:32:10-03:00"})

    assert meta.tem_inicio_confiavel is True
    assert meta.start_datetime == datetime(
        2026, 9, 20, 14, 32, 10, tzinfo=timezone(timedelta(hours=-3)),
    )
    assert meta.start_tag == "CreationDate"


def test_etiqueta_sem_fuso_nao_vira_data_normalizada():
    """O caso que mais importa: `CreateDate` do QuickTime quase sempre está
    em UTC sem declarar. Tratá-lo como hora local erraria por horas."""
    meta = _probe({"QuickTime:CreateDate": "2026:09:20 14:32:10"})

    assert meta.tem_inicio_confiavel is False
    assert meta.start_raw == "2026:09:20 14:32:10", (
        "o valor cru é preservado — é o que o arquivo declara"
    )


def test_etiqueta_com_fuso_vem_antes_da_sem_fuso():
    """Com as duas presentes, a que tem deslocamento ganha."""
    meta = _probe({
        "QuickTime:CreateDate": "2026:09:20 17:32:10",          # UTC, sem dizer
        "QuickTime:CreationDate": "2026:09:20 14:32:10-03:00",  # com fuso
    })

    assert meta.start_tag == "CreationDate"
    assert meta.start_datetime.utcoffset() == timedelta(hours=-3)


def test_deslocamento_sem_dois_pontos():
    meta = _probe({"CreationDate": "2026:09:20 14:32:10+0000"})

    assert meta.start_datetime.utcoffset() == timedelta(0)


def test_sufixo_z_e_utc():
    meta = _probe({"CreationDate": "2026:09:20 14:32:10Z"})

    assert meta.start_datetime.utcoffset() == timedelta(0)


def test_etiqueta_ilegivel_nao_estoura():
    meta = _probe({"CreationDate": "isto não é data"})

    assert meta.tem_inicio_confiavel is False
    assert meta.start_raw == "isto não é data"


def test_falha_do_exiftool_nao_interrompe():
    """Extração de vídeo não pode parar porque o metadado não foi lido."""
    meta = _probe({}, estoura=True)

    assert meta.error and "ExifTool" in meta.error
    assert meta.tags == {}
    assert meta.rotation_degrees == 0


# ── Rotação declarada ────────────────────────────────────────────────────

@pytest.mark.parametrize("valor,esperado", [
    (90, 90), ("90", 90), ("270", 270), (180, 180), (0, 0), (360, 0),
    ("90°", 90), (-90, 270),
])
def test_rotacao_declarada(valor, esperado):
    assert _probe({"QuickTime:Rotation": valor}).rotation_degrees == esperado


@pytest.mark.parametrize("valor", ["", None, "qualquer", 45, 33.3])
def test_rotacao_inesperada_vira_zero(valor):
    """Girar por palpite sairia pior: dataset torto sem explicação."""
    assert _probe({"Rotation": valor}).rotation_degrees == 0


def test_declara_rotacao_so_quando_ha():
    assert _probe({"Rotation": 90}).declara_rotacao is True
    assert _probe({"Rotation": 0}).declara_rotacao is False
    assert _probe({}).declara_rotacao is False


# ── Data por quadro ──────────────────────────────────────────────────────

def _meta_com_inicio():
    return VideoCaptureMetadata(
        start_datetime=datetime(
            2026, 9, 20, 14, 32, 10, tzinfo=timezone(timedelta(hours=-3)),
        ),
        start_raw="2026:09:20 14:32:10-03:00",
    )


def test_data_do_quadro_soma_o_instante():
    meta = _meta_com_inicio()

    assert datetime_do_quadro(meta, 0.0).startswith("2026-09-20T14:32:10")
    assert datetime_do_quadro(meta, 45.0).startswith("2026-09-20T14:32:55")
    assert datetime_do_quadro(meta, 120.0).startswith("2026-09-20T14:34:10")


def test_data_do_quadro_preserva_o_fuso():
    assert datetime_do_quadro(_meta_com_inicio(), 10.0).endswith("-03:00")


def test_duas_faces_distantes_no_video_tem_datas_distantes():
    """É o ponto do cálculo: a coluna distingue as faces entre si."""
    meta = _meta_com_inicio()

    primeira = datetime_do_quadro(meta, 0.0)
    ultima = datetime_do_quadro(meta, 7200.0)      # duas horas depois

    assert primeira != ultima
    assert "16:32:10" in ultima


def test_sem_inicio_confiavel_nao_calcula():
    meta = VideoCaptureMetadata(start_raw="2026:09:20 14:32:10")

    assert datetime_do_quadro(meta, 45.0) is None


def test_face_sem_instante_nao_calcula():
    """Falta dado; preencher seria afirmar o que não se sabe."""
    assert datetime_do_quadro(_meta_com_inicio(), None) is None


def test_instante_negativo_e_recusado():
    assert datetime_do_quadro(_meta_com_inicio(), -5.0) is None


def test_instante_nao_numerico_nao_estoura():
    assert datetime_do_quadro(_meta_com_inicio(), "abc") is None


# ── Colunas por face ─────────────────────────────────────────────────────

def test_colunas_trazem_a_data_calculada_e_a_origem():
    base = {"camera_make": "ACME", "capture_datetime_raw": "2026:09:20 14:32:10-03:00"}

    colunas = columns_for_frame(_meta_com_inicio(), 45.0, base)

    assert colunas["capture_datetime"].startswith("2026-09-20T14:32:55")
    assert colunas["capture_datetime_origin"] == ORIGEM_INICIO_MAIS_QUADRO
    assert colunas["camera_make"] == "ACME", "o resto do vídeo é preservado"


def test_data_do_arquivo_nao_vaza_para_a_face():
    """A data do ARQUIVO é o começo da gravação. Repeti-la em todas as faces
    afirmaria que todas foram capturadas no mesmo instante."""
    base = {
        "capture_datetime": "2026-09-20T14:32:10-03:00",   # do vídeo
        "capture_datetime_origin": "FILE_TAG",
    }

    colunas = columns_for_frame(_meta_com_inicio(), 45.0, base)

    assert colunas["capture_datetime"].startswith("2026-09-20T14:32:55")
    assert colunas["capture_datetime_origin"] == ORIGEM_INICIO_MAIS_QUADRO


def test_sem_inicio_confiavel_a_coluna_calculada_fica_vazia():
    base = {
        "capture_datetime": "2026-09-20T14:32:10-03:00",
        "capture_datetime_origin": "FILE_TAG",
        "capture_datetime_raw": "2026:09:20 14:32:10",
    }
    meta = VideoCaptureMetadata(start_raw="2026:09:20 14:32:10")

    colunas = columns_for_frame(meta, 45.0, base)

    assert "capture_datetime" not in colunas
    assert "capture_datetime_origin" not in colunas
    assert colunas["capture_datetime_raw"] == "2026:09:20 14:32:10", (
        "a crua do vídeo permanece — é o que o arquivo declara"
    )


def test_base_vazia_nao_estoura():
    colunas = columns_for_frame(_meta_com_inicio(), 10.0, None)

    assert colunas["capture_datetime_origin"] == ORIGEM_INICIO_MAIS_QUADRO


def test_base_nao_e_mutada():
    """Quem chama reusa a mesma base para todas as faces do vídeo."""
    base = {"camera_make": "ACME"}
    columns_for_frame(_meta_com_inicio(), 10.0, base)

    assert base == {"camera_make": "ACME"}


# ── Taxa de compressão ───────────────────────────────────────────────────

def test_compressao_calculada_do_arquivo_atual():
    from faceaudit.services.exif_extraction_service import _colunas_de_compressao

    # 12 MP em 3 MB → 2 bits/pixel → 12:1
    saida = _colunas_de_compressao(
        {"image_width": 4000, "image_height": 3000, "file_size_bytes": 3_000_000},
    )

    assert saida["image_bits_per_pixel"] == 2.0
    assert saida["image_compression_ratio"] == 12.0


def test_compressao_mais_agressiva_da_razao_maior():
    from faceaudit.services.exif_extraction_service import _colunas_de_compressao

    leve = _colunas_de_compressao(
        {"image_width": 4000, "image_height": 3000, "file_size_bytes": 6_000_000},
    )
    forte = _colunas_de_compressao(
        {"image_width": 4000, "image_height": 3000, "file_size_bytes": 500_000},
    )

    assert forte["image_compression_ratio"] > leve["image_compression_ratio"]


@pytest.mark.parametrize("faltando", ["image_width", "image_height", "file_size_bytes"])
def test_sem_um_dos_tres_nao_calcula(faltando):
    from faceaudit.services.exif_extraction_service import _colunas_de_compressao

    campos = {"image_width": 4000, "image_height": 3000, "file_size_bytes": 3_000_000}
    campos.pop(faltando)

    assert _colunas_de_compressao(campos) == {}


def test_valor_nao_numerico_nao_estoura():
    """ExifTool pode emitir "3.0 MB" se o `-n` falhar."""
    from faceaudit.services.exif_extraction_service import _colunas_de_compressao

    assert _colunas_de_compressao(
        {"image_width": 4000, "image_height": 3000, "file_size_bytes": "3.0 MB"},
    ) == {}


def test_declarada_e_calculada_coexistem():
    """A divergência entre as duas é informação: arquivo recomprimido depois
    da câmera tem o calculado abaixo do declarado."""
    from faceaudit.models.column_catalog import DatasetType, get_default_specs

    especificacoes = get_default_specs(DatasetType.FACE)
    declarada = especificacoes["declared_bits_per_pixel"]
    calculada = especificacoes["image_bits_per_pixel"]

    assert declarada.is_declared is True
    assert calculada.is_declared is False, (
        "a calculada não é declaração do arquivo — é medida nossa"
    )
    assert "recompressão" in declarada.description
