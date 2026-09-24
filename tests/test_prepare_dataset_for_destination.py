"""Os dois modos de levar um dataset para outra pasta.

`path_root` é relativo à pasta do parquet, então salvar noutra pasta
exige decidir o que fazer com as imagens. `prepare_dataset_for_destination`
materializa as duas intenções possíveis:

- portátil: copia as imagens para junto do destino (RELATIVE lá);
- referenciar: absolutiza os caminhos (ABSOLUTE), sem copiar nada.

A terceira possibilidade — herdar o `path_root` antigo — é justamente a
que quebrava as imagens em silêncio e não existe mais.
"""

from __future__ import annotations

from pathlib import Path

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.dataset_localization_service import (
    prepare_dataset_for_destination,
)


def _dataset(tmp_path: Path, n: int = 3):
    source_dir = tmp_path / "origem"
    images_dir = source_dir / "ds_images"
    images_dir.mkdir(parents=True)
    faces = []
    for i in range(n):
        name = f"face_{i}.png"
        (images_dir / name).write_bytes(b"png")
        faces.append(FaceInfo(image_path=name, identity=f"P{i:03d}"))
    ds = FaceDataset(name="ds", description="original", faces=faces)
    ds.path_mode = PathMode.RELATIVE
    ds.path_root = "ds_images"
    ds._parquet_dir = source_dir.resolve()
    return ds, images_dir


def test_referenciar_absolutiza_sem_copiar(tmp_path):
    ds, images_dir = _dataset(tmp_path)
    dest = tmp_path / "outra" / "ds.parquet"

    prepared, relatorio = prepare_dataset_for_destination(ds, dest, portable=False)

    assert relatorio.localization is None, "modo referenciar não copia nada"
    assert relatorio.rebase is not None and relatorio.rebase.changed
    # Reaponta o path_root para a MESMA pasta de imagens, em vez de
    # absolutizar: o dataset segue relativo e sobrevive a mover a árvore.
    for face in prepared.faces:
        assert prepared.resolve_image_path(face).exists()
        assert prepared.resolve_image_path(face).parent == images_dir.resolve()
    # entrada intocada
    assert ds.path_mode == PathMode.RELATIVE
    assert ds.faces[0].image_path == "face_0.png"


def test_portatil_copia_imagens_e_fica_relativo(tmp_path):
    ds, _ = _dataset(tmp_path)
    dest = tmp_path / "outra" / "ds.parquet"

    prepared, relatorio = prepare_dataset_for_destination(ds, dest, portable=True)
    result = relatorio.localization

    assert result is not None and result.n_failed == 0
    assert prepared.path_mode == PathMode.RELATIVE
    assert prepared.path_root == "images"
    for face in prepared.faces:
        assert (dest.parent / prepared.path_root / face.image_path).exists()

    # Gravado no destino, tudo resolve — e continua resolvendo se a pasta
    # inteira for movida.
    prepared.to_parquet(dest)
    movido = tmp_path / "levado_para_outro_lugar"
    dest.parent.rename(movido)
    reaberto = FaceDataset.from_parquet(movido / "ds.parquet")
    for face in reaberto.faces:
        assert reaberto.resolve_image_path(face).exists()


def test_portatil_preserva_identidade_do_dataset(tmp_path):
    """Nome, descrição e specs de coluna não podem se perder na cópia.

    É o que fazia o parquet chegar sem `column_specs` e, sem coluna de
    embedding declarada, sumir dos dropdowns do Strategy Manager.
    """
    ds, _ = _dataset(tmp_path)
    ds.column_specs = {"embedding_v2": object()}
    ds.provenance_metadata = {"source_run_id": "abc"}

    prepared, _ = prepare_dataset_for_destination(
        ds, tmp_path / "outra" / "ds.parquet", portable=True,
    )

    assert prepared.name == "ds"
    assert prepared.description == "original"
    assert set(prepared.column_specs) == {"embedding_v2"}
    assert prepared.provenance_metadata == {"source_run_id": "abc"}


def test_portatil_reporta_imagem_faltando(tmp_path):
    ds, images_dir = _dataset(tmp_path)
    (images_dir / "face_1.png").unlink()

    prepared, relatorio = prepare_dataset_for_destination(
        ds, tmp_path / "outra" / "ds.parquet", portable=True,
    )

    assert relatorio.localization.n_failed == 1
    assert len(prepared) == 2, "face sem imagem é descartada — e reportada"
