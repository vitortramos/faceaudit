"""Importar com cópia tem de deixar as imagens resolvíveis.

O defeito, reproduzido num projeto novo: adicionar um dataset ao projeto
respondendo SIM a "copiar imagens", calcular escores, e a tabela de
comparações não mostrar face nenhuma.

A causa não era a tabela. O `path_root` era escrito à mão como
`../images/<id>/images`, supondo o parquet em `<project>/datasets/`. Ele
vai para `datasets/primary/`, um nível mais fundo, então faltava um `..`:
a âncora resolvia para `datasets/images/...`, que não existe. O dataset
importava "com sucesso", abria, listava as faces e não encontrava imagem
nenhuma — silencioso até alguém procurar a imagem, muito depois.

O conserto é derivar a âncora do destino real do parquet em vez de supor o
layout, e é isso que estes testes travam. Medido no projeto que o expôs:
0 de 4 imagens resolvendo antes, 4 de 4 depois.
"""

from __future__ import annotations

import io
import contextlib
from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo


def _png(caminho: Path) -> Path:
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 40), (140, 110, 90)).save(caminho)
    return caminho


def _dataset_de_origem(tmp_path, n=3) -> Path:
    """Parquet fora do projeto, com as imagens ao lado — o que o usuário
    escolhe no seletor."""
    fora = tmp_path / "fora"
    faces = []
    for i in range(n):
        _png(fora / "imagens" / f"f{i}.png")
        face = FaceInfo(image_path=f"f{i}.png")
        face.identity = f"P{i}"
        face.file_name = f"f{i}.png"
        face.hash = f"h{i}"
        face.ff_info = FFInfo(
            embedding_sepaelv2=np.ones(8, dtype=np.float32),
        )
        faces.append(face)
    ds = FaceDataset(
        name="origem", faces=faces,
        path_mode=PathMode.RELATIVE, path_root="imagens",
    )
    caminho = fora / "origem.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        ds.to_parquet(caminho)
    return caminho


def _importar_com_copia(projeto: Path, origem: Path, resource_id: str) -> Path:
    """Reproduz a importação com cópia, sem a interface.

    Chama o mesmo método do controller, com um projeto de mentira que só
    expõe o que ele consulta — `storage.project_root`.
    """
    from interface.tabs.strategy_manager_v2.strategy_manager_v2_controller import (  # noqa: E501
        StrategyManagerV2Controller,
    )

    class _Storage:
        project_root = projeto

    class _Projeto:
        storage = _Storage()

    dublê = object.__new__(_Dublê)
    dublê._project = _Projeto()
    dublê._view = None
    with contextlib.redirect_stdout(io.StringIO()):
        ref = _Dublê._copiar(dublê, origem, resource_id)
    return projeto / ref.path


class _Dublê:
    """Empresta do controller só o método que importa com cópia.

    `StrategyManagerV2Controller` é `QObject`, e `object.__new__` sobre
    subclasse de QObject não é seguro — daí a classe de apoio que toma o
    método emprestado pelo `__dict__`, preservando staticmethods.
    """


def _preparar_duble():
    from interface.tabs.strategy_manager_v2.strategy_manager_v2_controller import (  # noqa: E501
        StrategyManagerV2Controller as C,
    )

    nome = next(
        n for n in C.__dict__
        if "copy" in n.lower() and "image" in n.lower() and "ask" not in n.lower()
    )
    _Dublê._copiar = C.__dict__[nome]
    return nome


_preparar_duble()


# ── O defeito ────────────────────────────────────────────────────────────

def test_as_imagens_resolvem_depois_de_importar(tmp_path):
    """Era 0 de N: o dataset importava, abria, listava as faces e não
    encontrava imagem nenhuma."""
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 3)

    destino = _importar_com_copia(projeto, origem, "meu dataset")

    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)
    resolvem = sum(1 for f in ds.faces if Path(ds.resolve_image_path(f)).is_file())
    assert resolvem == len(ds.faces) == 3


def test_a_ancora_aponta_para_a_pasta_de_imagens_do_projeto(tmp_path):
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 1)

    destino = _importar_com_copia(projeto, origem, "ds")

    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)
    ancora = (destino.parent / ds.path_root).resolve()
    assert ancora == (projeto / "images" / "ds" / "images").resolve()


def test_a_ancora_e_calculada_e_nao_suposta(tmp_path):
    """O número de `..` tem de sair da profundidade REAL do parquet. Era
    isto que estava escrito à mão, e por isso faltava um nível."""
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 1)

    destino = _importar_com_copia(projeto, origem, "ds")
    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)

    # `datasets/primary/` são dois níveis abaixo da raiz do projeto.
    profundidade = len(
        StrategyFileType.SOURCE_DATASET.build_path(dataset_id="ds").parts
    ) - 1
    assert ds.path_root.count("..") == profundidade, (
        f"path_root={ds.path_root!r} para parquet {profundidade} níveis "
        f"abaixo da raiz"
    )


def test_o_parquet_vai_para_onde_o_tipo_manda(tmp_path):
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 1)

    destino = _importar_com_copia(projeto, origem, "ds")

    assert destino.relative_to(projeto) == StrategyFileType.SOURCE_DATASET.build_path(
        dataset_id="ds"
    )


def test_a_ancora_e_posix(tmp_path):
    """Projeto criado no Windows tem de abrir no macOS; `pathlib` aceita
    `/` em todo SO, mas não aceita `\\` como separador fora dele."""
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 1)

    destino = _importar_com_copia(projeto, origem, "ds")
    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)

    assert "\\" not in ds.path_root


def test_nome_com_espaco_e_acento(tmp_path):
    """Os nomes reais dos datasets do caso: 'C3 - Padrão - Selecionadas'."""
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 2)

    destino = _importar_com_copia(projeto, origem, "C3 - Padrão - Selecionadas")

    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)
    assert all(Path(ds.resolve_image_path(f)).is_file() for f in ds.faces)


def test_o_dataset_continua_relativo(tmp_path):
    """Relativo é o que mantém o projeto portátil — copiar as imagens para
    dentro e depois absolutizar jogaria fora a razão de copiar."""
    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 1)

    destino = _importar_com_copia(projeto, origem, "ds")
    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(destino)

    assert ds.path_mode is PathMode.RELATIVE


def test_o_projeto_inteiro_pode_ser_movido(tmp_path):
    """A prova do Pendrive Paradigm: parquet e imagens dentro da mesma
    raiz, ancorados um ao outro."""
    import shutil

    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 2)
    _importar_com_copia(projeto, origem, "ds")

    mudado = tmp_path / "outro lugar" / "projeto"
    mudado.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(projeto, mudado)

    with contextlib.redirect_stdout(io.StringIO()):
        ds = FaceDataset.from_parquet(
            mudado / StrategyFileType.SOURCE_DATASET.build_path(dataset_id="ds")
        )
    assert all(Path(ds.resolve_image_path(f)).is_file() for f in ds.faces)


# ── A tabela de comparações volta a mostrar as faces ────────────────────

def test_a_tabela_acha_as_faces_do_dataset_importado(tmp_path):
    """O sintoma relatado, de ponta a ponta."""
    from faceaudit.models.embedding_info import EmbeddingInfo
    from faceaudit.models.similarity.similarity_score_dataset import (
        SimilarityScoreDataset,
    )
    from faceaudit.models.similarity.similarity_score_sample import (
        SimilarityScoreSample,
    )
    from faceaudit.services.score_comparison_table_service import (
        attach_face_images, build_from_parquet,
    )

    projeto = tmp_path / "projeto"
    origem = _dataset_de_origem(tmp_path, 2)
    destino = _importar_com_copia(projeto, origem, "ds")

    def _emb(h):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=h,
            image_path="", identity="P", file_name="",
        )

    escores = tmp_path / "escores.parquet"
    amostra = SimilarityScoreSample(
        probe=_emb("h0"), reference=_emb("h1"), raw_score=0.5,
        same_person=False,
    )
    amostra.same_person = False
    with contextlib.redirect_stdout(io.StringIO()):
        SimilarityScoreDataset(name="s", items=[amostra]).to_parquet(escores)

    t = attach_face_images(build_from_parquet(escores), [destino])

    assert t.has_images
    linha = t.rows[0]
    assert Path(linha.probe_image_path).is_file()
    assert Path(linha.reference_image_path).is_file()
