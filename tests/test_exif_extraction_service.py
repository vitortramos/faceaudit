"""Extração de metadados embutidos, nos dois backends.

O que os testes travam:

- **Caminho com acento funciona.** A árvore de trabalho inteira vive sob
  "Perícias"; passar os caminhos na linha de comando fazia o
  ExifTool responder "File not found" para todos eles, em silêncio. A
  correção é a lista de argumentos em arquivo UTF-8 (`-@`), e é isso que
  o teste do caminho acentuado protege.
- **Um arquivo ruim não contamina o lote.** O ExifTool reporta erro por
  arquivo; sem tratar isso, o ilegível passaria por "sem metadados".
- **A ordem de entrada é preservada** — quem chama casa o resultado com a
  face pela posição.
- **O dump não é filtrado pelo extrator.** Decidir o que vira coluna é de
  quem consome; descartar aqui perderia informação irrecuperável.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.services.exif_extraction_service import (
    ExifExtractionService,
    ImageMetadata,
)

_SEM_EXIFTOOL = not ExifExtractionService("exiftool").exiftool_available


def _imagem_com_exif(caminho: Path, *, make="ACME", model="Cam 9") -> Path:
    """JPEG com EXIF de câmera, escrito pelo próprio Pillow."""
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48), (120, 90, 60))
    exif = Image.Exif()
    exif[271] = make                      # Make
    exif[272] = model                     # Model
    exif[274] = 6                         # Orientation
    exif[305] = "FaceAudit Test"          # Software
    exif[36867] = "2026:09:24 10:30:00"   # DateTimeOriginal
    img.save(caminho, exif=exif)
    return caminho


# ── Resolução do executável ───────────────────────────────────────────────

def test_caminho_vazio_desliga_o_exiftool():
    svc = ExifExtractionService("")
    assert svc.exiftool_available is False
    assert svc.backend_name == ExifExtractionService.BACKEND_PILLOW


def test_caminho_inexistente_cai_no_pillow(tmp_path):
    svc = ExifExtractionService(str(tmp_path / "nao_existe.exe"))
    assert svc.exiftool_available is False


def test_caminho_absoluto_valido_e_aceito(tmp_path):
    """Binário fora do PATH — o caso da instalação por zip solto."""
    falso = tmp_path / "exiftool.exe"
    falso.write_text("", encoding="utf-8")
    svc = ExifExtractionService(str(falso))
    assert svc.exiftool_available is True


# ── Backend Pillow ────────────────────────────────────────────────────────

def test_pillow_le_exif_de_camera(tmp_path):
    arquivo = _imagem_com_exif(tmp_path / "foto.jpg")
    svc = ExifExtractionService(None)

    md = svc.extract(arquivo)

    assert md.backend == ExifExtractionService.BACKEND_PILLOW
    assert md.ok
    assert md.get_first("Make") == "ACME"
    assert md.get_first("Model") == "Cam 9"
    assert md.get_first("DateTimeOriginal") == "2026:09:24 10:30:00"


def test_get_first_resolve_nome_com_grupo():
    md = ImageMetadata(source_path="x", tags={"EXIF:Make": "ACME"})
    assert md.get_first("Make") == "ACME"
    assert md.get_first("EXIF:Make") == "ACME"
    assert md.get_first("Inexistente") is None


def test_get_first_respeita_a_ordem_dos_sinonimos():
    md = ImageMetadata(source_path="x", tags={"A": 1, "B": 2})
    assert md.get_first("B", "A") == 2


def test_arquivo_sem_exif_nao_e_erro(tmp_path):
    from PIL import Image

    caminho = tmp_path / "vazia.png"
    Image.new("RGB", (8, 8)).save(caminho)
    svc = ExifExtractionService(None)

    md = svc.extract(caminho)

    assert md.tags == {}
    assert md.error is not None, "ausência de metadados precisa ser dita"


def test_ordem_de_entrada_preservada(tmp_path):
    arquivos = [
        _imagem_com_exif(tmp_path / f"f{i}.jpg", model=f"M{i}") for i in range(3)
    ]
    svc = ExifExtractionService(None)

    resultados = svc.extract_many(arquivos)

    assert [m.get_first("Model") for m in resultados] == ["M0", "M1", "M2"]


# ── Backend ExifTool (pulado quando o binário não está disponível) ────────

@pytest.mark.skipif(_SEM_EXIFTOOL, reason="ExifTool não disponível nesta máquina")
def test_exiftool_le_caminho_com_acento(tmp_path):
    """O bug que motivou a lista de argumentos em arquivo."""
    pasta = tmp_path / "Perícias" / "Comparação"
    arquivo = _imagem_com_exif(pasta / "EMISSÃO 26_08_2021.jpg")
    svc = ExifExtractionService("exiftool")

    md = svc.extract(arquivo)

    assert md.backend == ExifExtractionService.BACKEND_EXIFTOOL
    assert md.ok, f"caminho acentuado falhou: {md.error}"
    assert md.get_first("Make") == "ACME"


@pytest.mark.skipif(_SEM_EXIFTOOL, reason="ExifTool não disponível nesta máquina")
def test_exiftool_traz_mais_que_o_pillow(tmp_path):
    """A razão de preferir o ExifTool: cobertura além do EXIF puro."""
    arquivo = _imagem_com_exif(tmp_path / "foto.jpg")

    com_tool = ExifExtractionService("exiftool").extract(arquivo)
    com_pillow = ExifExtractionService(None).extract(arquivo)

    assert len(com_tool.tags) > len(com_pillow.tags)
    grupos = {k.split(":", 1)[0] for k in com_tool.tags if ":" in k}
    assert "File" in grupos, "o ExifTool também descreve o arquivo, não só a foto"


@pytest.mark.skipif(_SEM_EXIFTOOL, reason="ExifTool não disponível nesta máquina")
def test_arquivo_ruim_nao_contamina_o_lote(tmp_path):
    bom = _imagem_com_exif(tmp_path / "bom.jpg")
    inexistente = tmp_path / "fantasma.jpg"
    svc = ExifExtractionService("exiftool")

    resultados = svc.extract_many([bom, inexistente])

    assert len(resultados) == 2
    assert resultados[0].ok
    assert not resultados[1].ok
    assert resultados[1].error


@pytest.mark.skipif(_SEM_EXIFTOOL, reason="ExifTool não disponível nesta máquina")
def test_lote_menor_que_o_conjunto(tmp_path):
    """Vários lotes têm de produzir o mesmo que um só."""
    arquivos = [
        _imagem_com_exif(tmp_path / f"f{i}.jpg", model=f"M{i}") for i in range(5)
    ]
    svc = ExifExtractionService("exiftool", batch_size=2)

    resultados = svc.extract_many(arquivos)

    assert [m.get_first("Model") for m in resultados] == ["M0", "M1", "M2", "M3", "M4"]


# ── Promoção para colunas do dataset ──────────────────────────────────────

def test_colunas_curadas_saem_das_tags():
    from faceaudit.services.exif_extraction_service import build_capture_columns

    md = ImageMetadata(source_path="x", tags={
        "EXIF:Make": "ACME", "EXIF:Model": "Cam 9",
        "EXIF:ISO": "400", "EXIF:FNumber": "2.8",
        "EXIF:Orientation": 6,
        "EXIF:DateTimeOriginal": "2026:09:24 10:30:00",
    })

    cols = build_capture_columns(md)

    assert cols["camera_make"] == "ACME"
    assert cols["iso"] == 400 and isinstance(cols["iso"], int)
    assert cols["f_number"] == 2.8
    assert cols["orientation"] == 6
    assert cols["capture_datetime_raw"] == "2026:09:24 10:30:00"


def test_datetime_normalizado_exige_offset_explicito():
    """Sem fuso declarado, atribuir um seria inventar informação — e data
    e hora costuma ser justamente o que está em disputa."""
    from faceaudit.services.exif_extraction_service import build_capture_columns

    sem_offset = ImageMetadata(source_path="x", tags={
        "EXIF:DateTimeOriginal": "2026:09:24 10:30:00",
    })
    com_offset = ImageMetadata(source_path="x", tags={
        "EXIF:DateTimeOriginal": "2026:09:24 10:30:00",
        "EXIF:OffsetTimeOriginal": "-03:00",
    })

    assert "capture_datetime" not in build_capture_columns(sem_offset)
    assert build_capture_columns(com_offset)["capture_datetime"] == (
        "2026-09-24T10:30:00-03:00"
    )


def test_offset_sem_dois_pontos_e_aceito():
    from faceaudit.services.exif_extraction_service import build_capture_columns

    md = ImageMetadata(source_path="x", tags={
        "EXIF:DateTimeOriginal": "2026:09:24 10:30:00",
        "EXIF:OffsetTimeOriginal": "-0300",
    })
    assert build_capture_columns(md)["capture_datetime"] == "2026-09-24T10:30:00-03:00"


def test_offset_ilegivel_nao_produz_data_normalizada():
    from faceaudit.services.exif_extraction_service import build_capture_columns

    md = ImageMetadata(source_path="x", tags={
        "EXIF:DateTimeOriginal": "2026:09:24 10:30:00",
        "EXIF:OffsetTimeOriginal": "horário de Brasília",
    })
    assert "capture_datetime" not in build_capture_columns(md)


def test_dump_completo_vai_para_exif_json():
    """Nada é descartado: as colunas são campos promovidos, não um filtro."""
    import json as _json
    from faceaudit.services.exif_extraction_service import build_capture_columns

    md = ImageMetadata(source_path="x", tags={
        "EXIF:Make": "ACME", "MakerNotes:CoisaExotica": "42",
    })

    cols = build_capture_columns(md)
    dump = _json.loads(cols["exif_json"])

    assert dump["MakerNotes:CoisaExotica"] == "42", (
        "tag fora da lista curada precisa sobreviver no dump"
    )


def test_valor_fora_do_formato_vira_texto_em_vez_de_sumir():
    from faceaudit.services.exif_extraction_service import build_capture_columns

    md = ImageMetadata(source_path="x", tags={"EXIF:ISO": "desconhecido"})
    assert build_capture_columns(md)["iso"] == "desconhecido"


def test_sem_tags_nao_produz_coluna_nenhuma():
    from faceaudit.services.exif_extraction_service import build_capture_columns

    assert build_capture_columns(ImageMetadata(source_path="x")) == {}
