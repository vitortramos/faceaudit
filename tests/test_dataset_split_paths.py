"""Dividir um dataset não pode quebrar a resolução das imagens.

O defeito: `split_by_identity` e `split_by_identity_list` construíam
`FaceDataset(name, faces)` — mesmo defeito que o `concat` já teve. As
metades nasciam com `path_mode=ABSOLUTE` (default), sem `path_root` e sem
`_parquet_dir`, mas com os `image_path` RELATIVOS da origem dentro. E o
"Dividir" as gravava em `paths.datasets`, outra pasta.

Por que a rede de proteção do `to_parquet` (`_resolve_paths_for_destination`)
não salvou: ela só age sobre dataset declarado RELATIVE e que conhece a
pasta de origem. A metade já chegava declarando ABSOLUTE — a rede a deixava
passar como "nada a ajustar" — e gravava `images/a1.png` sob o rótulo de
caminho absoluto. Ao reabrir, o caminho era relativo ao diretório corrente
do processo: nenhuma imagem.

O que os testes travam:

- as metades herdam `path_mode`, `path_root`, `_parquet_dir`,
  `column_specs`, `is_aligned` e proveniência;
- gravadas noutra pasta e REABERTAS do disco, toda face resolve para a
  imagem original;
- metade vazia não é gravada.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.dataset_split_save_service import (
    EmptySplitHalfError,
    save_split_halves,
)


def _autocontido(tmp_path: Path) -> FaceDataset:
    """Dataset autocontido: `<base>/caso/caso.parquet` + `images/`."""
    pasta = tmp_path / "caso"
    imagens = pasta / "images"
    imagens.mkdir(parents=True)
    faces = []
    for ident in ("ana", "bia", "caio", "davi"):
        for k in range(2):
            nome = f"{ident}_{k}.png"
            (imagens / nome).write_bytes(b"x")
            f = FaceInfo(image_path=nome)
            f.identity = ident
            faces.append(f)
    ds = FaceDataset(
        name="caso", faces=faces,
        path_mode=PathMode.RELATIVE, path_root="images",
    )
    ds.is_aligned = True
    ds.provenance_metadata = {"embedding_source": "teste"}
    ds.to_parquet(pasta / "caso.parquet")
    return FaceDataset.from_parquet(pasta / "caso.parquet")


def _todas_resolvem(ds: FaceDataset) -> bool:
    return all(Path(ds.resolve_image_path(f)).is_file() for f in ds.faces)


@pytest.mark.parametrize("modo", ["ratio", "manual"])
def test_metades_herdam_a_ancoragem(tmp_path, modo):
    origem = _autocontido(tmp_path)
    if modo == "ratio":
        a, b = origem.split_by_identity(split_ratio=0.5)
    else:
        a, b = origem.split_by_identity_list({"ana"})

    for metade in (a, b):
        assert metade.path_mode is PathMode.RELATIVE
        assert metade.path_root == "images"
        assert metade._parquet_dir == origem._parquet_dir
        assert metade.is_aligned is True
        assert metade.provenance_metadata == {"embedding_source": "teste"}
        assert set(metade.column_specs) == set(origem.column_specs)
        # Antes de gravar, ainda na memória, as imagens já resolvem.
        assert _todas_resolvem(metade)


def test_metades_gravadas_noutra_pasta_continuam_achando_as_imagens(tmp_path):
    origem = _autocontido(tmp_path)
    a, b = origem.split_by_identity_list({"ana", "bia"})
    a.name, b.name = "caso_A", "caso_B"

    destino = tmp_path / "biblioteca_datasets"
    salvo_a, salvo_b = save_split_halves(a, b, destino)

    for salvo, n in ((salvo_a, 4), (salvo_b, 4)):
        assert salvo.path == destino / f"{salvo.dataset.name}.parquet"
        relido = FaceDataset.from_parquet(salvo.path)
        assert len(relido.faces) == n
        assert _todas_resolvem(relido), (
            "a metade gravada noutra pasta perdeu as imagens"
        )
        # E elas apontam para as imagens ORIGINAIS — nada foi copiado.
        for f in relido.faces:
            assert Path(relido.resolve_image_path(f)).resolve().parent == (
                tmp_path / "caso" / "images"
            ).resolve()
    assert not (destino / "images").exists()


def test_origem_nao_e_mutada(tmp_path):
    origem = _autocontido(tmp_path)
    antes = [f.image_path for f in origem.faces]
    a, b = origem.split_by_identity(split_ratio=0.5)
    save_split_halves(a, b, tmp_path / "out")
    assert [f.image_path for f in origem.faces] == antes
    assert origem.path_root == "images"
    assert a.path_root == "images" and b.path_root == "images"


def test_metade_vazia_nao_e_gravada(tmp_path):
    origem = _autocontido(tmp_path)
    # Proporção 1.0: todas as identidades vão para A, B fica vazio.
    a, b = origem.split_by_identity(split_ratio=1.0)
    assert len(b.faces) == 0
    destino = tmp_path / "out"
    with pytest.raises(EmptySplitHalfError):
        save_split_halves(a, b, destino)
    assert not destino.exists() or not list(destino.glob("*.parquet"))
