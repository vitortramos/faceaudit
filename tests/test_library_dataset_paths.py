"""📚 Biblioteca: dataset RELATIVE chega à biblioteca achando as imagens.

O defeito: o botão copiava o parquet para `<biblioteca>/datasets/` com
`shutil.copy2`, byte a byte. Num dataset RELATIVE (o autocontido grava
`path_root="images"`), as imagens são procuradas a partir da pasta DO
PARQUET; na biblioteca o mesmo `path_root` apontava para
`<biblioteca>/datasets/images/`, onde elas não estão. O "→ Estratégia" já
absolutizava (`LocalProjectArtifactStorage.import_dataset`); a biblioteca
passa a fazer o mesmo — referenciar as imagens onde estão, sem copiá-las.
"""

from __future__ import annotations

from pathlib import Path

from faceaudit.enums.path_mode import PathMode
from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.storage.global_library_storage import GlobalLibraryStorage


def _gravar(tmp_path: Path, modo: PathMode) -> Path:
    pasta = tmp_path / "caso"
    imagens = pasta / "images"
    imagens.mkdir(parents=True)
    faces = []
    for ident in ("ana", "bia"):
        nome = f"{ident}.png"
        (imagens / nome).write_bytes(b"x")
        f = FaceInfo(image_path=nome if modo is PathMode.RELATIVE else str(imagens / nome))
        f.identity = ident
        faces.append(f)
    ds = FaceDataset(
        name="caso", faces=faces, path_mode=modo,
        path_root="images" if modo is PathMode.RELATIVE else None,
    )
    destino = pasta / "caso.parquet"
    ds.to_parquet(destino)
    return destino


def _biblioteca(tmp_path: Path) -> GlobalLibraryStorage:
    lib = GlobalLibraryStorage(library_root=tmp_path / "biblioteca")
    lib.create_library_structure()
    return lib


def test_dataset_relativo_resolve_na_biblioteca(tmp_path):
    origem = _gravar(tmp_path, PathMode.RELATIVE)
    lib = _biblioteca(tmp_path)

    # Mesma chamada que o botão 📚 faz.
    ref = lib.copy_artifact_to_project(
        source_path=origem,
        artifact_type=StrategyArtifactType.GLOBAL_DATASET,
        dataset_id="caso_lib",
    )
    copia = lib.library_root / ref.path
    relido = FaceDataset.from_parquet(copia)

    assert relido.path_mode is PathMode.ABSOLUTE
    assert len(relido.faces) == 2
    for f in relido.faces:
        resolvido = Path(relido.resolve_image_path(f))
        assert resolvido.is_file()
        assert resolvido.resolve().parent == (tmp_path / "caso" / "images").resolve()
    # Nenhuma imagem foi copiada para a biblioteca.
    assert not (copia.parent / "images").exists()
    # O hash é o do arquivo que de fato está na biblioteca.
    from faceaudit.services.hash_service import HashService
    assert ref.hash == HashService().calculate_file_hash(copia)


def test_dataset_absoluto_e_copiado_como_esta(tmp_path):
    origem = _gravar(tmp_path, PathMode.ABSOLUTE)
    lib = _biblioteca(tmp_path)
    ref = lib.import_dataset(origem, dataset_id="caso_abs")
    copia = lib.library_root / ref.path
    assert copia.read_bytes() == origem.read_bytes()


def test_origem_nao_e_alterada(tmp_path):
    origem = _gravar(tmp_path, PathMode.RELATIVE)
    antes = origem.read_bytes()
    _biblioteca(tmp_path).import_dataset(origem, dataset_id="caso_lib")
    assert origem.read_bytes() == antes
    relido = FaceDataset.from_parquet(origem)
    assert relido.path_mode is PathMode.RELATIVE
    assert relido.path_root == "images"
