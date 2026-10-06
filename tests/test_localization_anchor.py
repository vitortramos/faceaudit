"""Dataset localizado tem de resolver as imagens IMEDIATAMENTE.

O defeito: "Modificar Caminhos" reescrevia os caminhos corretamente, como
relativos a `target_dir`, mas não registrava `target_dir` como âncora. O
dataset resultante só voltava a resolver depois de gravado dentro daquela
pasta. Entre a operação e o "Salvar Como", a galeria aparecia vazia e o
usuário concluía que a operação tinha falhado — quando ela havia
funcionado e só faltava a âncora.

Medido no caso real que expôs: dataset de 1817 faces, 0 imagens
encontradas depois de localizar, 1817 depois da correção.

`_parquet_dir` é o campo que `resolve_image_path` consulta, e `target_dir`
é por construção a pasta onde o parquet deve ser gravado.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.services.dataset_localization_service import (
    DatasetLocalizationService,
    LocalizationMode,
)


def _dataset_absoluto(tmp_path, nomes=("a.png", "b.png")):
    """Dataset ABSOLUTE com as imagens de fato no disco."""
    imagens = tmp_path / "originais"
    imagens.mkdir(parents=True, exist_ok=True)
    faces = []
    for n in nomes:
        (imagens / n).write_bytes(b"x")
        f = FaceInfo(image_path=str(imagens / n))
        f.identity = n.split(".")[0]
        faces.append(f)
    return FaceDataset(name="ds", faces=faces, path_mode=PathMode.ABSOLUTE)


def _todas_resolvem(ds) -> bool:
    return all(Path(ds.resolve_image_path(f)).is_file() for f in ds.faces)


# ── O defeito ────────────────────────────────────────────────────────────

def test_reescrever_relativo_resolve_sem_precisar_salvar(tmp_path):
    """Era 0 de N antes da correção: os caminhos ficavam relativos a uma
    pasta que o dataset não sabia qual era."""
    ds = _dataset_absoluto(tmp_path)

    r = DatasetLocalizationService().localize(
        ds, tmp_path / "originais", mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert r.dataset.path_mode is PathMode.RELATIVE
    assert r.dataset._parquet_dir is not None, "âncora não foi registrada"
    assert _todas_resolvem(r.dataset)


def test_copiar_originais_resolve_sem_precisar_salvar(tmp_path):
    """Mesmo problema no modo que copia: as imagens vão para o destino, mas
    o dataset não sabia onde era o destino."""
    ds = _dataset_absoluto(tmp_path)
    destino = tmp_path / "autocontido"

    r = DatasetLocalizationService().localize(
        ds, destino, mode=LocalizationMode.COPY_ORIGINALS,
    )

    assert r.dataset.path_mode is PathMode.RELATIVE
    assert _todas_resolvem(r.dataset)


def test_ancora_e_a_pasta_destino(tmp_path):
    """Não qualquer pasta: tem de ser exatamente aquela contra a qual os
    caminhos foram escritos, que é onde o parquet deve ser gravado."""
    ds = _dataset_absoluto(tmp_path)
    destino = (tmp_path / "originais").resolve()

    r = DatasetLocalizationService().localize(
        ds, destino, mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert Path(r.dataset._parquet_dir).resolve() == destino


def test_o_original_nao_e_mutado(tmp_path):
    """A documentação do serviço promete que o dataset de entrada não é
    mutado; a âncora do resultado não pode vazar para ele."""
    ds = _dataset_absoluto(tmp_path)
    antes_modo = ds.path_mode
    antes_caminho = ds.faces[0].image_path

    DatasetLocalizationService().localize(
        ds, tmp_path / "originais", mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert ds.path_mode is antes_modo
    assert ds.faces[0].image_path == antes_caminho


# ── Ida e volta ──────────────────────────────────────────────────────────

def test_absoluto_vira_relativo_e_continua_resolvendo(tmp_path):
    """É a resposta à pergunta "como transformo absoluto em relativo"."""
    ds = _dataset_absoluto(tmp_path)
    assert ds.path_mode is PathMode.ABSOLUTE
    assert _todas_resolvem(ds)

    r = DatasetLocalizationService().localize(
        ds, tmp_path / "originais", mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert r.dataset.path_mode is PathMode.RELATIVE
    assert _todas_resolvem(r.dataset)
    assert not Path(r.dataset.faces[0].image_path).is_absolute()


def test_unido_de_duas_pastas_vira_relativo_a_pasta_comum(tmp_path):
    """O caso que motivou tudo: dois datasets com pastas de imagens
    distintas, unidos, e depois relativizados contra a pasta que contém as
    duas. Sem cópia."""
    def _montar(nome, arquivos):
        pasta = tmp_path / nome
        pasta.mkdir(parents=True, exist_ok=True)
        faces = []
        for a in arquivos:
            (pasta / a).write_bytes(b"x")
            f = FaceInfo(image_path=a)
            f.identity = a.split("_")[0]
            faces.append(f)
        d = FaceDataset(
            name=nome, faces=faces, path_mode=PathMode.RELATIVE, path_root=nome,
        )
        d._parquet_dir = tmp_path
        return d

    a = _montar("imagens_a", ["P1_001.png"])
    b = _montar("imagens_b", ["P2_001.png"])
    unido = a.concat(b)
    assert unido.path_mode is PathMode.ABSOLUTE

    r = DatasetLocalizationService().localize(
        unido, tmp_path, mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert r.dataset.path_mode is PathMode.RELATIVE
    assert _todas_resolvem(r.dataset)
    # Os caminhos preservam a subpasta de cada origem.
    caminhos = sorted(f.image_path.replace("\\", "/") for f in r.dataset.faces)
    assert caminhos == ["imagens_a/P1_001.png", "imagens_b/P2_001.png"]


def test_imagem_fora_da_pasta_destino_vira_falha_declarada(tmp_path):
    """REWRITE_RELATIVE exige que as imagens já estejam sob o destino.
    Quem está fora tem de aparecer em `failures`, não sumir."""
    ds = _dataset_absoluto(tmp_path)
    fora = tmp_path / "outra"
    fora.mkdir()
    (fora / "c.png").write_bytes(b"x")
    f = FaceInfo(image_path=str(fora / "c.png"))
    f.identity = "c"
    ds.faces.append(f)

    r = DatasetLocalizationService().localize(
        ds, tmp_path / "originais", mode=LocalizationMode.REWRITE_RELATIVE,
    )

    assert r.n_failed >= 1
    assert any("c.png" in str(p) for p, _ in r.failures)
