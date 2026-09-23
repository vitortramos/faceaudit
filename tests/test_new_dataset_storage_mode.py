"""Destino + modo de armazenamento na criação de dataset (fotos).

Antes, a Etapa 2 do diálogo "Novo Dataset" era decorativa: o campo Destino só
virava `aligned_target_dir` (e apenas na combinação "Autocontido + Face
alinhada"), o parquet ia sempre pra biblioteca global, e "Autocontido + Face
original" era no-op — os getters `get_dataset_dir`/`get_self_contained`/
`get_image_type` do controller não eram chamados por ninguém.

Cobre (offscreen, sem GPU/ForensicFace):
- `_finalize_storage_mode` nos dois modos: autocontido (copia + RELATIVE) e
  caminho absoluto (não copia + ABSOLUTE).
- Crops já materializados na pasta final não são copiados de novo (idempotência
  por hash) e não estouram `SameFileError`.
- Face que falhou no alinhamento (imagem fora do destino) é PRESERVADA — a
  linha `error=True` é o que o perito usa pra inspecionar falhas.
- Round-trip: parquet gravado no destino relê e resolve as imagens.
- Plumbing diálogo → DTO: Destino e Autocontido chegam ao
  `DatasetIngestionConfig`.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from faceaudit.enums.path_mode import PathMode  # noqa: E402
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig  # noqa: E402
from faceaudit.models.face_dataset import FaceDataset  # noqa: E402
from faceaudit.models.face_info import FaceInfo  # noqa: E402
from faceaudit.models.ff_info import FFInfo  # noqa: E402
from faceaudit.pipelines.feature_extraction_pipeline import (  # noqa: E402
    FeatureExtractionPipeline,
)


def _img(path: Path, tom: int) -> None:
    Image.fromarray(np.full((32, 32, 3), tom, np.uint8)).save(path)


def _dataset(paths, *, name="meu_ds", is_aligned=False) -> FaceDataset:
    faces = [
        FaceInfo(
            image_path=str(p), identity=f"id{i}",
            ff_info=FFInfo(det_score=0.9, embedding_sepaelv2=np.ones(512)),
        )
        for i, p in enumerate(paths)
    ]
    return FaceDataset(name=name, faces=faces, path_mode=PathMode.ABSOLUTE,
                       is_aligned=is_aligned)


def _config(dest: Path, *, self_contained: bool, **kw) -> DatasetIngestionConfig:
    return DatasetIngestionConfig(
        input_path=dest, dataset_name="meu_ds",
        dataset_dir=dest, self_contained=self_contained, **kw,
    )


# ───────────────────────── modo autocontido ─────────────────────────

def test_autocontido_copia_imagens_e_relativiza(tmp_path):
    src = tmp_path / "fotos"; src.mkdir()
    paths = [src / f"id{i}.png" for i in range(3)]
    for i, p in enumerate(paths):
        _img(p, i * 40)
    dest = tmp_path / "destino"; dest.mkdir()

    out = FeatureExtractionPipeline._finalize_storage_mode(
        _dataset(paths), _config(dest, self_contained=True), dest,
    )

    assert out.path_mode == PathMode.RELATIVE
    assert out.path_root == "images"
    assert len(out.faces) == 3
    copiadas = sorted(p.name for p in (dest / "images").iterdir())
    assert copiadas == ["id0.png", "id1.png", "id2.png"]
    # Paths viram relativos à subpasta de imagens.
    assert all(not Path(f.image_path).is_absolute() for f in out.faces)
    # Shallow copy preserva os campos pesados — embedding não pode sumir.
    assert out.faces[0].ff_info.embedding_sepaelv2 is not None
    assert out.faces[0].ff_info.det_score == 0.9


def test_autocontido_nao_recopia_crops_ja_no_destino(tmp_path):
    """Crops materializados direto na pasta final: origem == destino.

    Sem o `skip_existing` por hash isto seria uma segunda cópia integral (ou um
    SameFileError). O ganho é justamente não pagar I/O duplicado no fluxo
    "alinhar durante a criação".
    """
    dest = tmp_path / "destino"
    imgs = dest / "images"; imgs.mkdir(parents=True)
    paths = [imgs / f"crop{i}.png" for i in range(3)]
    for i, p in enumerate(paths):
        _img(p, i * 40)
    mtimes = {p.name: p.stat().st_mtime_ns for p in paths}

    out = FeatureExtractionPipeline._finalize_storage_mode(
        _dataset(paths, is_aligned=True),
        _config(dest, self_contained=True, align_during_ingestion=True), dest,
    )

    assert len(out.faces) == 3
    assert out.path_mode == PathMode.RELATIVE
    assert out.is_aligned is True
    assert sorted(p.name for p in imgs.iterdir()) == [
        "crop0.png", "crop1.png", "crop2.png",
    ]
    # Arquivos intocados — não foram reescritos.
    assert {p.name: p.stat().st_mtime_ns for p in imgs.iterdir()} == mtimes


def test_autocontido_preserva_face_que_falhou_no_alinhamento(tmp_path):
    """Face sem crop alinhado aponta pro original, FORA do destino.

    `REWRITE_RELATIVE` descartaria essa face (o path relativo seria `../..`);
    `COPY_ORIGINALS` traz a imagem pra dentro e mantém a linha `error=True`.
    """
    dest = tmp_path / "destino"
    imgs = dest / "images"; imgs.mkdir(parents=True)
    ok = imgs / "crop0.png"; _img(ok, 10)
    fora = tmp_path / "originais"; fora.mkdir()
    falhou = fora / "falhou.png"; _img(falhou, 200)

    ds = _dataset([ok], is_aligned=True)
    ds.add(FaceInfo(image_path=str(falhou), identity="id_err", error=True))

    out = FeatureExtractionPipeline._finalize_storage_mode(
        ds, _config(dest, self_contained=True, align_during_ingestion=True), dest,
    )

    assert len(out.faces) == 2, "a face com erro não pode ser descartada"
    assert [f.identity for f in out.faces if f.error] == ["id_err"]
    assert (imgs / "falhou.png").exists(), "o original devia ser trazido pra dentro"


def test_autocontido_round_trip_resolve_imagens(tmp_path):
    src = tmp_path / "fotos"; src.mkdir()
    paths = [src / f"id{i}.png" for i in range(2)]
    for i, p in enumerate(paths):
        _img(p, i * 60)
    dest = tmp_path / "destino"; dest.mkdir()

    out = FeatureExtractionPipeline._finalize_storage_mode(
        _dataset(paths), _config(dest, self_contained=True), dest,
    )
    parquet = dest / "meu_ds.parquet"
    out.to_parquet(parquet)

    relido = FaceDataset.from_parquet(parquet)
    assert len(relido.faces) == 2
    assert all(Path(relido.resolve_image_path(f)).exists() for f in relido.faces)


# ─────────────────────── modo caminho absoluto ───────────────────────

def test_caminho_absoluto_nao_copia_nada(tmp_path):
    src = tmp_path / "fotos"; src.mkdir()
    paths = [src / f"id{i}.png" for i in range(3)]
    for i, p in enumerate(paths):
        _img(p, i * 40)
    dest = tmp_path / "destino"; dest.mkdir()

    out = FeatureExtractionPipeline._finalize_storage_mode(
        _dataset(paths), _config(dest, self_contained=False), dest,
    )

    assert out.path_mode == PathMode.ABSOLUTE
    assert out.path_root is None
    assert len(out.faces) == 3
    assert list(dest.iterdir()) == [], "modo absoluto não deve copiar imagem"
    assert all(Path(f.image_path).is_absolute() for f in out.faces)


# ─────────────────── nome sugerido a partir da pasta ───────────────────

@pytest.mark.parametrize("caminho, esperado", [
    # Pasta genérica → sobe um nível. Caso real: J:/FGNET/images virava um
    # dataset chamado "images" (e uma pasta de imagens "images_images").
    ("J:/FGNET/images", "FGNET"),
    ("J:/FGNET/Fotos", "FGNET"),
    ("/dados/CFP/aligned", "CFP"),
    ("/dados/RFW/dataset", "RFW"),
    # Nome específico → mantém.
    ("J:/FGNET", "FGNET"),
    ("/dados/BFW_Clean_2", "BFW_Clean_2"),
    ("/dados/images_do_caso", "images_do_caso"),
])
def test_nome_sugerido_ignora_pasta_generica(caminho, esperado):
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (
        suggest_dataset_name,
    )
    assert suggest_dataset_name(caminho) == esperado


def test_nome_sugerido_na_raiz_do_drive_nao_estoura():
    """Sem pai com nome, fica com o último componente — não vira string vazia."""
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (
        suggest_dataset_name,
    )
    assert suggest_dataset_name("J:/images") == "images"


# ───────────────────── plumbing diálogo → DTO ─────────────────────

def test_dialogo_propaga_destino_e_autocontido_para_o_dto(qapp, tmp_path):
    """Os getters da Etapa 2 precisam chegar ao DTO — eram código morto.

    Usa a fixture `qapp` do pytest-qt (instalado no projeto) em vez de criar o
    QApplication à mão: com dois donos do mesmo QApplication o teardown derruba
    o processo sem deixar saída. O diálogo também recebe um parent vivo durante
    todo o teste, pela mesma razão — QWidget órfão coletado fora de ordem
    derruba o Qt.
    """
    pytest.importorskip("PyQt6")
    from PyQt6 import QtWidgets

    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (
        NewDatasetController,
    )

    parent = QtWidgets.QWidget()
    dest = tmp_path / "meu_destino"; dest.mkdir()
    ctrl = NewDatasetController(parent, templates_path=tmp_path,
                                default_dataset_dir=str(dest))
    ctrl.view.txt_name.setText("ds_teste")
    ctrl.view.txt_path.setText(str(tmp_path))

    ctrl.view.rb_self_contained.setChecked(True)
    ctrl.view.rb_img_original.setChecked(True)
    cfg = ctrl.get_config()
    assert cfg.dataset_dir == dest
    assert cfg.self_contained is True
    # "Face original" não aciona alinhamento durante a ingestão…
    assert cfg.align_during_ingestion is False
    # …mas o Destino continua valendo pro parquet (antes era ignorado).

    ctrl.view.rb_img_aligned.setChecked(True)
    cfg = ctrl.get_config()
    assert cfg.align_during_ingestion is True
    assert cfg.aligned_target_dir == dest

    ctrl.view.rb_absolute.setChecked(True)
    cfg = ctrl.get_config()
    assert cfg.self_contained is False
    assert cfg.dataset_dir == dest

    parent.deleteLater()
