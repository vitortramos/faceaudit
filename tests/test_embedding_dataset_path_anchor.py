"""Carregar um dataset relativo não pode perder a âncora das imagens.

O sintoma: dataset importado com cópia das imagens para o projeto, escores
calculados, e a tabela de comparações sem as faces. Os caminhos do dataset
em disco estavam certos; o que quebrava era o CARREGAMENTO.

`EmbeddingDataset` é o dataset leve: não tem `path_mode` nem `path_root`,
por design. Mas `from_parquet` lia a coluna `image_path` crua e ignorava a
âncora que estava na metadata do parquet, então num dataset RELATIVE cada
item ficava com só o nome do arquivo. É esse o payload que o
`StrategyResourceLoaderService` entrega aos executores, de modo que os
escores, a proveniência da agregação e a tabela herdavam um caminho que não
abre — e nada no dado denunciava.

O segundo defeito era de confiança: a tabela aceitava qualquer
`image_path` preenchido como resolvido. Com isso ela se declarava com
imagens e abria no formato de miniatura, e `attach_face_images` não via a
face como pendente, portanto nunca recuperava o caminho real pelo hash.
Formato de imagem com célula sem imagem.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.services.score_comparison_table_service import (
    attach_face_images, build_from_dataset,
)


def _png(caminho: Path) -> Path:
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 40), (120, 130, 140)).save(caminho)
    return caminho


def _parquet_relativo(tmp_path, n=2, raiz="../images/ds/images") -> Path:
    """Parquet RELATIVE com a âncora apontando para fora da pasta dele —
    exatamente o que "copiar imagens para o projeto" produz."""
    pasta = tmp_path / "datasets" / "primary"
    pasta.mkdir(parents=True, exist_ok=True)
    imagens = (pasta / raiz).resolve()
    faces = []
    for i in range(n):
        _png(imagens / f"f{i}.png")
        face = FaceInfo(image_path=f"f{i}.png")
        face.identity = f"P{i}"
        face.file_name = f"f{i}.png"
        face.hash = f"h{i}"
        face.ff_info = FFInfo(embedding_sepaelv2=np.ones(8, dtype=np.float32))
        faces.append(face)
    ds = FaceDataset(
        name="ds", faces=faces, path_mode=PathMode.RELATIVE, path_root=raiz,
    )
    caminho = pasta / "ds.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        ds.to_parquet(caminho)
    return caminho


# ── A causa: o loader leve ignorava a âncora ────────────────────────────

def test_o_dataset_leve_resolve_o_caminho(tmp_path):
    """Era aqui que o caminho virava só o nome do arquivo."""
    caminho = _parquet_relativo(tmp_path, 2)

    ds = EmbeddingDataset.from_parquet(caminho, embedding_column="embedding_v2")

    assert len(ds.items) == 2
    for item in ds.items:
        assert Path(item.image_path).is_file(), item.image_path


def test_a_ancora_vem_da_metadata_do_parquet(tmp_path):
    """Não de uma suposição sobre o layout: a subpasta sai de
    `faceaudit:path_root`, que pode ser qualquer uma."""
    caminho = _parquet_relativo(tmp_path, 1, raiz="../../fotos/subpasta")

    ds = EmbeddingDataset.from_parquet(caminho, embedding_column="embedding_v2")

    assert Path(ds.items[0].image_path).is_file()
    assert "fotos" in ds.items[0].image_path


def test_parquet_absoluto_nao_muda(tmp_path):
    """Comportamento herdado preservado: sem `path_mode` relativo o
    caminho sai como está gravado."""
    imagens = tmp_path / "imgs"
    face = FaceInfo(image_path=str(_png(imagens / "a.png")))
    face.identity = "P"
    face.file_name = "a.png"
    face.hash = "h"
    face.ff_info = FFInfo(embedding_sepaelv2=np.ones(8, dtype=np.float32))
    caminho = tmp_path / "abs.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        FaceDataset(
            name="ds", faces=[face], path_mode=PathMode.ABSOLUTE,
        ).to_parquet(caminho)

    ds = EmbeddingDataset.from_parquet(caminho, embedding_column="embedding_v2")

    # Comparado como caminho, não como string: o projeto grava sempre em
    # POSIX para que parquet do Windows abra no macOS.
    assert Path(ds.items[0].image_path) == imagens / "a.png"
    assert Path(ds.items[0].image_path).is_file()


def test_parquet_de_embedding_dataset_continua_funcionando(tmp_path):
    """Parquet do dataset leve não tem metadata de âncora nenhuma."""
    caminho = tmp_path / "leve.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        EmbeddingDataset(
            name="leve",
            items=[EmbeddingInfo(
                embedding=np.ones(4, dtype=np.float32), face_hash="h",
                image_path="/qualquer/a.png", identity="P", file_name="a.png",
            )],
        ).to_parquet(caminho)

    ds = EmbeddingDataset.from_parquet(caminho)

    assert ds.items[0].image_path == "/qualquer/a.png"


def test_caminho_vazio_continua_vazio(tmp_path):
    """Template agregado não tem imagem; resolver o vazio devolveria a
    pasta do parquet e faria a linha passar por face com foto."""
    caminho = tmp_path / "t.parquet"
    with contextlib.redirect_stdout(io.StringIO()):
        EmbeddingDataset(
            name="t",
            items=[EmbeddingInfo(
                embedding=np.ones(4, dtype=np.float32), face_hash="SYNTH_x",
                image_path="", identity="P", file_name="AGG_P_k2_abc",
            )],
        ).to_parquet(caminho)

    ds = EmbeddingDataset.from_parquet(caminho)

    assert ds.items[0].image_path == ""


def test_o_loader_da_estrategia_entrega_resolvido(tmp_path):
    """O caminho real: é por aqui que o payload chega aos executores."""
    from faceaudit.services.strategy_resource_loader_service import (
        StrategyResourceLoaderService,
    )

    caminho = _parquet_relativo(tmp_path, 2)
    servico = object.__new__(StrategyResourceLoaderService)

    ds = StrategyResourceLoaderService._load_dataset(
        servico, caminho, embedding_column="embedding_v2",
    )

    assert all(Path(i.image_path).is_file() for i in ds.items)


# ── A confiança: caminho preenchido não é caminho resolvido ─────────────

def _par(hash_probe, caminho_probe, hash_ref, caminho_ref):
    def _f(h, c):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=h,
            image_path=c, identity="P", file_name=Path(c).name if c else "",
        )

    amostra = SimilarityScoreSample(
        probe=_f(hash_probe, caminho_probe),
        reference=_f(hash_ref, caminho_ref),
        raw_score=0.6, same_person=None,
    )
    amostra.same_person = None
    return amostra


def test_caminho_que_nao_abre_nao_conta_como_imagem(tmp_path):
    """Senão a tabela abre no formato de miniatura e mostra célula vazia."""
    ds = SimilarityScoreDataset(
        name="s", items=[_par("h0", "f0.png", "h1", "f1.png")],
        metadata={"is_casework": True},
    )

    t = build_from_dataset(ds)

    assert not t.has_images
    assert t.rows[0].probe_image_path is None


def test_caminho_quebrado_e_recuperado_pelo_hash(tmp_path):
    """É o que faz os escores JÁ calculados voltarem a mostrar as faces,
    sem precisar re-executar a instrução."""
    parquet = _parquet_relativo(tmp_path, 2)
    ds = SimilarityScoreDataset(
        name="s", items=[_par("h0", "f0.png", "h1", "f1.png")],
        metadata={"is_casework": True},
    )

    t = attach_face_images(build_from_dataset(ds), [parquet])

    assert t.has_images
    assert Path(t.rows[0].probe_image_path).is_file()
    assert Path(t.rows[0].reference_image_path).is_file()


def test_caminho_que_abre_e_mantido(tmp_path):
    """Sem dataset de origem alcançável, o caminho bom do payload é tudo
    o que existe — descartá-lo seria perder a imagem."""
    bom = _png(tmp_path / "imgs" / "a.png")
    ds = SimilarityScoreDataset(
        name="s", items=[_par("h0", str(bom), "h1", str(bom))],
        metadata={"is_casework": True},
    )

    t = build_from_dataset(ds)

    assert t.has_images
    assert t.rows[0].probe_image_path == str(bom)


def test_absoluto_de_outra_maquina_e_recuperado(tmp_path):
    """Projeto levado para outro computador: o caminho gravado existe como
    string e não como arquivo."""
    parquet = _parquet_relativo(tmp_path, 1)
    ds = SimilarityScoreDataset(
        name="s",
        items=[_par("h0", "/Volumes/outro/disco/f0.png", "h0", "X:/nada/f0.png")],
        metadata={"is_casework": True},
    )

    t = attach_face_images(build_from_dataset(ds), [parquet])

    assert Path(t.rows[0].probe_image_path).is_file()


# ── Ponta a ponta, como o app faz ───────────────────────────────────────

def test_do_parquet_relativo_ate_a_tabela(tmp_path):
    """Reproduz a sequência que o usuário fez: dataset com as imagens
    copiadas para dentro, escores calculados do payload carregado, tabela
    de comparações."""
    from faceaudit.services.strategy_resource_loader_service import (
        StrategyResourceLoaderService,
    )

    parquet = _parquet_relativo(tmp_path, 2)
    servico = object.__new__(StrategyResourceLoaderService)
    carregado = StrategyResourceLoaderService._load_dataset(
        servico, parquet, embedding_column="embedding_v2",
    )

    amostra = SimilarityScoreSample(
        probe=carregado.items[0], reference=carregado.items[1],
        raw_score=0.5, same_person=None,
    )
    amostra.same_person = None
    t = build_from_dataset(
        SimilarityScoreDataset(
            name="s", items=[amostra], metadata={"is_casework": True},
        )
    )

    assert t.has_images, "o payload carregado tem de trazer caminho que abre"
    assert Path(t.rows[0].probe_image_path).is_file()
