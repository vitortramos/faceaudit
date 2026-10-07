"""Quais arquivos entram no dataset, e o aviso de quem ficou de fora.

A lista de extensões estava escrita três vezes, em lugares que não se
conheciam: a ingestão de pasta, a geração de imagens pasta a pasta e o
seletor de arquivo do fluxo de vídeo. Nenhuma trazia `.jfif`, que é como o
Windows nomeia imagem salva do navegador e que é JPEG com outro nome,
lido sem problema por quem decide pelo conteúdo.

Pior que a falta era o silêncio. O filtro descartava sem contar: uma pasta
só de `.jfif` rendia dataset de zero faces sem uma linha dizendo por quê,
e uma pasta mista rendia menos faces do que tinha arquivos, sem nada
apontando a diferença.

O que os testes travam:

- os formatos que são JPEG com outro nome entram;
- o que fica de fora é CONTADO e reportado, agrupado por extensão;
- o aviso chega a quem chamou, não só ao terminal;
- as três listas saem da mesma fonte, senão a próxima extensão entra em
  duas delas e some na terceira.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.models.image_file_extensions import (
    EXTENSOES_DE_IMAGEM,
    classificar,
    e_imagem,
    filtro_de_dialogo,
)


# ── A lista ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nome", [
    "a.jfif", "a.JFIF", "a.jfi", "a.jif", "a.jpe",
    "a.jpg", "a.jpeg", "a.png", "a.bmp", "a.webp", "a.tif", "a.tiff",
])
def test_formatos_aceitos(nome):
    assert e_imagem(nome), nome


@pytest.mark.parametrize("nome", ["a.txt", "a.db", "a.json", "a.mp4", "a"])
def test_formatos_recusados(nome):
    assert not e_imagem(nome)


def test_jfif_e_jpeg_com_outro_nome():
    """A razão de o descuido doer: o arquivo abre, só não era listado."""
    for ext in (".jfif", ".jfi", ".jif", ".jpe"):
        assert ext in EXTENSOES_DE_IMAGEM


# ── A triagem conta o que ficou de fora ─────────────────────────────────

def _pasta(tmp_path, nomes):
    for n in nomes:
        (tmp_path / n).write_bytes(b"x")
    return sorted(tmp_path.iterdir())


def test_separa_imagens_do_resto(tmp_path):
    arquivos = _pasta(tmp_path, ["a.jpg", "b.jfif", "c.txt", "Thumbs.db"])

    t = classificar(arquivos)

    assert sorted(p.name for p in t.imagens) == ["a.jpg", "b.jfif"]
    assert t.ignorados_por_extensao == {".txt": 1, ".db": 1}
    assert t.n_ignorados == 2


def test_pasta_nao_entra_na_contagem(tmp_path):
    """Diretório não é candidato a imagem e poluiria o aviso."""
    (tmp_path / "sub").mkdir()
    arquivos = _pasta(tmp_path, ["a.jpg"]) + [tmp_path / "sub"]

    t = classificar(arquivos)

    assert t.n_ignorados == 0


def test_arquivo_sem_extensao_e_contado(tmp_path):
    arquivos = _pasta(tmp_path, ["sem_extensao"])

    t = classificar(arquivos)

    assert t.ignorados_por_extensao == {"": 1}
    assert "(sem extensão)" in t.aviso()


def test_nada_de_fora_nao_gera_aviso(tmp_path):
    t = classificar(_pasta(tmp_path, ["a.jpg", "b.png"]))

    assert t.aviso() == ""


def test_o_aviso_diz_quantos_e_de_que_tipo(tmp_path):
    arquivos = _pasta(
        tmp_path, ["a.jpg", "b.txt", "c.txt", "d.txt", "e.db"],
    )

    aviso = classificar(arquivos).aviso()

    assert "4 arquivos na pasta" in aviso
    # Em ordem decrescente: quem olha quer saber primeiro o volume maior.
    assert aviso.index(".txt: 3") < aviso.index(".db: 1")
    assert ".jfif" in aviso, "o aviso tem de listar o que SERIA aceito"


def test_o_aviso_no_singular(tmp_path):
    aviso = classificar(_pasta(tmp_path, ["a.txt"])).aviso()

    assert "1 arquivo na pasta" in aviso
    assert "arquivos" not in aviso


# ── A ingestão reporta ───────────────────────────────────────────────────

def test_a_ingestao_avisa_quem_ficou_de_fora(tmp_path):
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for n in ("a.jpg", "b.jfif", "c.txt", "d.txt"):
        (pasta / n).write_bytes(b"x")

    mensagens = []
    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False,
        callback=lambda m, p: mensagens.append(m),
    )

    assert len(ds.faces) == 2, "o .jfif tem de entrar"
    avisos = [m for m in mensagens if "extensão" in m]
    assert avisos, mensagens
    assert ".txt: 2" in avisos[0]


def test_pasta_so_de_jfif_rende_dataset(tmp_path):
    """Era o caso que rendia zero faces em silêncio."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for i in range(3):
        (pasta / f"f{i}.jfif").write_bytes(b"x")

    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False,
    )

    assert len(ds.faces) == 3


def test_sem_nada_de_fora_a_ingestao_nao_avisa(tmp_path):
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    (pasta / "a.jpg").write_bytes(b"x")

    mensagens = []
    FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False,
        callback=lambda m, p: mensagens.append(m),
    )

    assert not [m for m in mensagens if "extensão" in m]


def test_lista_explicita_do_caller_continua_valendo(tmp_path):
    """O parâmetro existe e algum caller pode restringir de propósito."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for n in ("a.jpg", "b.png"):
        (pasta / n).write_bytes(b"x")

    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False, extensions=(".png",),
    )

    assert len(ds.faces) == 1


# ── Uma lista só ─────────────────────────────────────────────────────────

def test_a_materializacao_usa_a_mesma_lista():
    from faceaudit.services.face_image_materialization_service import (
        _IMAGE_SUFFIXES,
    )

    assert set(_IMAGE_SUFFIXES) == set(EXTENSOES_DE_IMAGEM)


def test_o_filtro_do_seletor_sai_da_mesma_lista():
    """Escrito à mão, o filtro divergia: o usuário escolhia um arquivo que
    o sistema depois descartava, ou o contrário."""
    filtro = filtro_de_dialogo()

    for ext in EXTENSOES_DE_IMAGEM:
        assert f"*{ext}" in filtro, ext
    assert "Todos (*.*)" in filtro


def test_o_dialogo_de_video_usa_o_filtro_compartilhado():
    import inspect

    from interface.tabs.dataset_manager.video_dataset_dialog import (
        video_dataset_setup_dialog as mod,
    )

    fonte = inspect.getsource(mod)
    assert "filtro_de_dialogo()" in fonte
    assert "*.png *.jpg *.jpeg" not in fonte, (
        "o filtro escrito à mão voltou e vai divergir da lista"
    )
