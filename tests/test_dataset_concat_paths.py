"""Unir dois datasets não pode quebrar a resolução das imagens.

O defeito: `concat` construía `FaceDataset(name=..., faces=...)` e perdia
TODA a ancoragem — `path_mode`, `path_root`, `column_specs`, `is_aligned`
voltavam ao default. O dataset unido abria, listava as faces e não
encontrava imagem nenhuma.

O caso real que o expôs: dois datasets RELATIVE com `path_root` diferentes,
cujos `image_path` eram apenas nomes de arquivo. Um dataset tem UM
`path_root`, então nem preservar a âncora de um dos lados resolveria —
metade das faces apontaria para a pasta errada. A saída é resolver os dois
para absoluto antes de unir.

O que os testes travam:

- âncoras diferentes viram ABSOLUTE, e TODA face continua resolvível;
- âncoras iguais preservam o modo relativo, que é o que mantém o dataset
  portátil;
- as faces de origem NÃO são mutadas: elas seguem abertas na tela, e
  corrompê-las seria estragar dois datasets para produzir um terceiro;
- `is_aligned` só sobrevive se os DOIS forem alinhados;
- `column_specs` são unidas, senão as colunas do segundo dataset nascem
  como CUSTOM e perdem rótulo e papel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo


def _dataset(nome, pasta_parquet, raiz, arquivos, modo=PathMode.RELATIVE):
    faces = []
    for a in arquivos:
        f = FaceInfo(image_path=a)
        f.identity = a.split("_")[0]
        faces.append(f)
    ds = FaceDataset(name=nome, faces=faces, path_mode=modo, path_root=raiz)
    ds._parquet_dir = Path(pasta_parquet)
    return ds


def _montar_em_disco(tmp_path, nome, raiz, arquivos):
    """Cria a pasta de imagens e os arquivos, como um dataset real."""
    pasta = tmp_path / nome
    pasta.mkdir(parents=True, exist_ok=True)
    imagens = pasta / raiz
    imagens.mkdir(parents=True, exist_ok=True)
    for a in arquivos:
        (imagens / a).write_bytes(b"x")
    return _dataset(nome, pasta, raiz, arquivos)


# ── O caso que expôs o defeito ───────────────────────────────────────────

def test_ancoras_diferentes_continuam_resolviveis(tmp_path):
    """Dois relativos com raízes distintas: o resultado tem de achar TODAS
    as imagens, não metade."""
    a = _montar_em_disco(tmp_path, "ds_a", "imagens_a", ["P1_001.png", "P2_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "imagens_b", ["P3_001.png"])

    unido = a.concat(b)

    assert len(unido) == 3
    for face in unido.faces:
        assert Path(unido.resolve_image_path(face)).is_file(), face.image_path


def test_ancoras_diferentes_produzem_absoluto(tmp_path):
    a = _montar_em_disco(tmp_path, "ds_a", "imagens_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "imagens_b", ["P2_001.png"])

    unido = a.concat(b)

    assert unido.path_mode is PathMode.ABSOLUTE
    assert unido.path_root is None
    assert all(Path(f.image_path).is_absolute() for f in unido.faces)


def test_mesma_ancora_preserva_o_modo_relativo(tmp_path):
    """Preservar o relativo é o que mantém o dataset portátil; absolutizar
    sem necessidade cobraria esse preço à toa."""
    pasta = tmp_path / "comum"
    (pasta / "imagens").mkdir(parents=True)
    for a in ("P1_001.png", "P2_001.png"):
        (pasta / "imagens" / a).write_bytes(b"x")
    a = _dataset("a", pasta, "imagens", ["P1_001.png"])
    b = _dataset("b", pasta, "imagens", ["P2_001.png"])

    unido = a.concat(b)

    assert unido.path_mode is PathMode.RELATIVE
    assert unido.path_root == "imagens"
    for face in unido.faces:
        assert Path(unido.resolve_image_path(face)).is_file()


def test_origens_nao_sao_mutadas(tmp_path):
    """Elas continuam abertas na tela. Corrompê-las seria estragar dois
    datasets para produzir um terceiro."""
    a = _montar_em_disco(tmp_path, "ds_a", "imagens_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "imagens_b", ["P2_001.png"])
    antes_a = [f.image_path for f in a.faces]
    antes_b = [f.image_path for f in b.faces]

    a.concat(b)

    assert [f.image_path for f in a.faces] == antes_a
    assert [f.image_path for f in b.faces] == antes_b
    assert a.path_mode is PathMode.RELATIVE
    assert b.path_mode is PathMode.RELATIVE


def test_faces_do_resultado_sao_copias(tmp_path):
    """Mutar o resultado não pode alcançar a origem."""
    a = _montar_em_disco(tmp_path, "ds_a", "imagens_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "imagens_b", ["P2_001.png"])

    unido = a.concat(b)
    unido.faces[0].identity = "ALTERADA"

    assert a.faces[0].identity != "ALTERADA"


# ── Metadados que se perdiam junto ───────────────────────────────────────

def test_alinhado_so_sobrevive_se_os_dois_forem(tmp_path):
    """Recorte 112×112 misturado a foto original num dataset marcado como
    alinhado faz o enriquecimento seguinte mandar a foto ao encoder como se
    fosse face alinhada — embedding de lixo, sem erro aparente."""
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "im_b", ["P2_001.png"])

    a.is_aligned = b.is_aligned = True
    assert a.concat(b).is_aligned is True

    b.is_aligned = False
    assert a.concat(b).is_aligned is False
    assert b.concat(a).is_aligned is False


def test_especificacoes_de_coluna_sao_unidas(tmp_path):
    """Sem isso, as colunas do segundo dataset nascem CUSTOM no resultado e
    perdem rótulo, papel e categoria."""
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "im_b", ["P2_001.png"])
    a.column_specs = {"col_a": "spec_a"}
    b.column_specs = {"col_b": "spec_b"}

    unido = a.concat(b)

    assert set(unido.column_specs) == {"col_a", "col_b"}


def test_em_conflito_prevalece_o_dataset_atual(tmp_path):
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "im_b", ["P2_001.png"])
    a.column_specs = {"col": "do_atual"}
    b.column_specs = {"col": "do_outro"}

    assert a.concat(b).column_specs["col"] == "do_atual"


def test_nome_pode_ser_informado(tmp_path):
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])
    b = _montar_em_disco(tmp_path, "ds_b", "im_b", ["P2_001.png"])

    assert a.concat(b, new_name="Meu Dataset").name == "Meu Dataset"
    assert "Merged" in a.concat(b).name


def test_tipo_errado_continua_recusado(tmp_path):
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])

    with pytest.raises(TypeError):
        a.concat("não é um dataset")


def test_caminho_irresolvivel_nao_descarta_a_face(tmp_path):
    """Caminho quebrado já era quebrado antes da união; perder a face
    silenciosamente seria pior que preservá-la quebrada."""
    a = _montar_em_disco(tmp_path, "ds_a", "im_a", ["P1_001.png"])
    b = _dataset("ds_b", tmp_path / "inexistente", "im_b", ["P9_001.png"])

    unido = a.concat(b)

    assert len(unido) == 2


# ── A interface avisa ────────────────────────────────────────────────────

def test_amostragem_detecta_imagens_ausentes(tmp_path):
    """Mesmo com a união correta, um caminho quebrado na ORIGEM continua
    quebrado — e isso não pode passar em silêncio."""
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        DatasetManagerController,
    )

    ds = _montar_em_disco(tmp_path, "ds", "im", ["P1_001.png", "P2_001.png"])
    assert DatasetManagerController._amostrar_imagens_ausentes(ds) == (0, 2)

    (tmp_path / "ds" / "im" / "P1_001.png").unlink()
    ausentes, verificadas = DatasetManagerController._amostrar_imagens_ausentes(ds)
    assert (ausentes, verificadas) == (1, 2)


def test_amostragem_nao_varre_dataset_grande(tmp_path):
    """Conferir existência é I/O por face; em dataset de milhões, e em
    pasta sincronizada por nuvem, varrer tudo travaria a interface."""
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        DatasetManagerController as C,
    )

    ds = _dataset("grande", tmp_path, "im", [f"P{i}_001.png" for i in range(5000)])
    _, verificadas = C._amostrar_imagens_ausentes(ds)

    assert verificadas == C._AMOSTRA_VERIFICACAO_MERGE


def test_amostragem_tolera_dataset_vazio(tmp_path):
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.dataset_manager_controller import (
        DatasetManagerController as C,
    )

    assert C._amostrar_imagens_ausentes(FaceDataset(name="v", faces=[])) == (0, 0)
