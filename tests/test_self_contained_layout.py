"""Layout do dataset autocontido: `<pasta>/<nome>/{<nome>.parquet, images/}`.

Um dataset autocontido é uma UNIDADE — parquet, imagens e sidecars na
mesma pasta, com `path_root` apontando para dentro dela. Assim a pasta
pode ser movida, compactada ou levada para outra máquina sem quebrar.

O layout anterior espalhava `<nome>.parquet` e `<nome>_images/` lado a
lado na pasta escolhida: mover só o parquet, ou renomeá-lo, quebrava a
relação implícita entre os dois — em silêncio.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.dataset_localization_service import (
    SELF_CONTAINED_IMAGES_SUBDIR,
    sanitize_dir_name,
    self_contained_dataset_dir,
)


def _img(path: Path, tom: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (tom, tom, tom)).save(path)


def test_sanitize_preserva_espacos_e_acentos():
    """O nome da pasta deve continuar parecido com o que o perito vê."""
    assert sanitize_dir_name("C1 - PAD") == "C1 - PAD"
    assert sanitize_dir_name("Questionadas nº 3") == "Questionadas nº 3"


def test_sanitize_remove_caracteres_ilegais():
    assert sanitize_dir_name("a/b:c") == "a_b_c"
    assert sanitize_dir_name('x<y>z|w?') == "x_y_z_w_"
    assert sanitize_dir_name("   ") == "dataset"      # vazio → fallback
    assert sanitize_dir_name("termina.") == "termina"  # ponto final é ilegal


def test_pasta_propria_do_dataset(tmp_path):
    assert self_contained_dataset_dir(tmp_path, "meu ds") == tmp_path / "meu ds"
    assert self_contained_dataset_dir(tmp_path, "a/b") == tmp_path / "a_b"


def test_modificar_caminhos_usa_pasta_propria(tmp_path):
    """COPY_ORIGINALS produz autocontido → pasta própria + `images/`."""
    from interface.tabs.dataset_manager.workers.modify_paths_worker import (
        ModifyPathsWorker,
    )

    origem = tmp_path / "fotos"
    caminhos = [origem / f"f{i}.png" for i in range(2)]
    for i, p in enumerate(caminhos):
        _img(p, i * 60)

    ds = FaceDataset(
        name="Confronto 2",
        faces=[FaceInfo(image_path=str(p), identity=f"P{i}") for i, p in enumerate(caminhos)],
        path_mode=PathMode.ABSOLUTE,
    )

    destino = tmp_path / "escolhida"
    worker = ModifyPathsWorker(
        dataset=ds,
        target_dir=destino,
        mode=LocalizationMode.COPY_ORIGINALS,
    )
    resultados = []
    worker.finished.connect(resultados.append)
    worker.run()

    assert len(resultados) == 1 and resultados[0] is not None
    result = resultados[0]
    propria = self_contained_dataset_dir(destino, "Confronto 2")
    assert Path(result.target_dir) == propria.resolve()
    assert result.dataset.path_root == SELF_CONTAINED_IMAGES_SUBDIR
    copiadas = sorted(p.name for p in (propria / SELF_CONTAINED_IMAGES_SUBDIR).iterdir())
    assert copiadas == ["f0.png", "f1.png"]
    # Nada solto na pasta que o usuário escolheu.
    assert sorted(p.name for p in destino.iterdir()) == ["Confronto 2"]


def test_pasta_movida_continua_resolvendo(tmp_path):
    """A propriedade que motiva o layout: a unidade é movível."""
    from interface.tabs.dataset_manager.workers.modify_paths_worker import (
        ModifyPathsWorker,
    )

    origem = tmp_path / "fotos"
    caminhos = [origem / f"f{i}.png" for i in range(2)]
    for i, p in enumerate(caminhos):
        _img(p, i * 60)
    ds = FaceDataset(
        name="ds",
        faces=[FaceInfo(image_path=str(p), identity=f"P{i}") for i, p in enumerate(caminhos)],
        path_mode=PathMode.ABSOLUTE,
    )

    worker = ModifyPathsWorker(
        dataset=ds,
        target_dir=tmp_path / "escolhida",
        mode=LocalizationMode.COPY_ORIGINALS,
    )
    resultados = []
    worker.finished.connect(resultados.append)
    worker.run()
    propria = Path(resultados[0].target_dir)
    resultados[0].dataset.to_parquet(propria / "ds.parquet")

    # Move a pasta inteira — inclusive para longe da origem das imagens.
    movida = tmp_path / "outro_lugar" / "ds"
    movida.parent.mkdir(parents=True)
    propria.rename(movida)

    relido = FaceDataset.from_parquet(movida / "ds.parquet")
    assert len(relido) == 2
    for face in relido.faces:
        assert relido.resolve_image_path(face).exists()
