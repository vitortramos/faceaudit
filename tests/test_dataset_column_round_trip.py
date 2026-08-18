"""Round-trip de colunas: nada com dado pode sumir, nada pode ficar sem spec.

Duas perdas silenciosas motivaram estes testes. Ambas só se manifestavam no
SEGUNDO save (load → save), que é o ciclo normal de quem abre um dataset no
Gerenciador e salva — por isso todo teste aqui faz o round-trip duas vezes.

1. `det_score` era emitido por `to_flat_dict` mas nunca relido em
   `from_dataframe`: o FFInfo era remontado sem ele e a coluna zerava. Passou
   meses despercebido; quando foi medido, só 9 datasets ainda tinham valor.
2. Coluna sem `ColumnSpec` é invisível na tabela (o model desenha exatamente
   `column_specs`). Custom column de template ia pro parquet com os dados
   certos e sem forma de exibir ou filtrar.
"""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from faceaudit.models.column_catalog import (
    ColumnRole, DatasetType, get_default_specs,
)
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo


def _dataset(n=4, *, com_extra=True) -> FaceDataset:
    faces = []
    for i in range(n):
        face = FaceInfo(
            image_path=f"/fotos/id{i}.png", identity=f"id{i % 2}",
            ff_info=FFInfo(
                det_score=0.90 + i / 100, fiqa_score=70.0 + i, ipd=30.0 + i,
                yaw=1.0 * i, pitch=2.0 * i, roll=3.0 * i,
                embedding_sepaelv2=np.full(512, 0.1 * (i + 1)),
            ),
        )
        if com_extra:
            # Equivalente ao que o template produz: custom column declarada em
            # `DirTemplate.custom_columns` cai em `FaceInfo.extra_data`.
            face.extra_data = {"age": f"{i:02d}", "camera": f"cam{i}"}
        faces.append(face)
    return FaceDataset(name="ds", faces=faces)


def _ida_e_volta(ds: FaceDataset, tmp_path, nome="ds") -> FaceDataset:
    caminho = tmp_path / f"{nome}.parquet"
    ds.to_parquet(caminho)
    return FaceDataset.from_parquet(caminho)


# ───────────────────────── det_score ─────────────────────────

def test_det_score_sobrevive_a_dois_ciclos(tmp_path):
    """O bug só aparecia no segundo save — o primeiro gravava certo."""
    original = [f.ff_info.det_score for f in _dataset().faces]

    volta1 = _ida_e_volta(_dataset(), tmp_path, "v1")
    assert [f.ff_info.det_score for f in volta1.faces] == original

    # Segundo ciclo: salva o que foi lido. Aqui a coluna sumia.
    volta2 = _ida_e_volta(volta1, tmp_path, "v2")
    assert [f.ff_info.det_score for f in volta2.faces] == original

    tabela = pq.ParquetFile(tmp_path / "v2.parquet")
    assert "det_score" in tabela.schema_arrow.names
    cobertura = json.loads(
        tabela.schema_arrow.metadata[b"faceaudit:column_coverage"]
    )
    assert cobertura["det_score"]["non_null_count"] == 4


@pytest.mark.parametrize("atributo", ["det_score", "fiqa_score", "ipd", "yaw", "pitch", "roll"])
def test_atributos_ff_escalares_sobrevivem_a_dois_ciclos(tmp_path, atributo):
    """Varre a família inteira — `det_score` era o único furo, mas o custo de
    verificar os vizinhos é zero e impede uma reincidência silenciosa."""
    esperado = [getattr(f.ff_info, atributo) for f in _dataset().faces]
    volta = _ida_e_volta(_ida_e_volta(_dataset(), tmp_path, "a"), tmp_path, "b")
    assert [getattr(f.ff_info, atributo) for f in volta.faces] == esperado


def test_embedding_sobrevive_a_dois_ciclos(tmp_path):
    volta = _ida_e_volta(_ida_e_volta(_dataset(), tmp_path, "a"), tmp_path, "b")
    for i, face in enumerate(volta.faces):
        assert face.ff_info.embedding_sepaelv2 is not None
        np.testing.assert_allclose(
            face.ff_info.embedding_sepaelv2, np.full(512, 0.1 * (i + 1)), rtol=1e-6,
        )


# ───────────────────────── column_specs ─────────────────────────

def test_toda_coluna_do_parquet_tem_spec(tmp_path):
    """Invariante central: coluna sem spec é coluna invisível na tabela."""
    caminho = tmp_path / "ds.parquet"
    _dataset().to_parquet(caminho)
    relido = FaceDataset.from_parquet(caminho)

    orfas = [c for c in relido.to_pandas().columns if c not in relido.column_specs]
    assert orfas == [], f"colunas sem spec ficariam invisíveis: {orfas}"


def test_custom_column_de_template_ganha_spec_e_valores(tmp_path):
    """Caso que originou a investigação: a coluna `age` do template."""
    relido = _ida_e_volta(_dataset(), tmp_path)

    df = relido.to_pandas()
    assert list(df["age"]) == ["00", "01", "02", "03"]
    assert "age" in relido.column_specs
    assert "camera" in relido.column_specs

    spec = relido.column_specs["age"]
    # role=CUSTOM é deliberado: um spec sintetizado NUNCA deve reivindicar
    # role semântica (EMBEDDING/POSE/QUALITY) sobre a qual o código branchea.
    assert spec.role == ColumnRole.CUSTOM
    assert spec.visible_default is True


def test_spec_sintetizado_e_persistido_no_parquet(tmp_path):
    """A cura precisa sobreviver ao disco — senão depende de rodar de novo."""
    caminho = tmp_path / "ds.parquet"
    _dataset().to_parquet(caminho)
    # Segundo save, agora a partir do que foi lido.
    FaceDataset.from_parquet(caminho).to_parquet(caminho)

    specs = json.loads(
        pq.ParquetFile(caminho).schema_arrow.metadata[b"faceaudit:column_specs"]
    )
    assert "age" in specs and "camera" in specs


def test_parquet_gravado_antes_da_correcao_e_curado_na_leitura(tmp_path):
    """Simula o parquet já em disco: coluna presente, spec ausente.

    Sem cura na leitura, o dataset esconderia a coluna pra sempre — os specs
    vêm da metadata do próprio arquivo, então regravar não resolveria.
    """
    caminho = tmp_path / "legado.parquet"
    _dataset().to_parquet(caminho)

    # Remove o spec de `age` da metadata, preservando a coluna de dados.
    tabela = pq.read_table(caminho)
    meta = dict(tabela.schema.metadata)
    specs = json.loads(meta[b"faceaudit:column_specs"])
    del specs["age"]
    meta[b"faceaudit:column_specs"] = json.dumps(specs).encode()
    pq.write_table(tabela.replace_schema_metadata(meta), caminho)
    assert "age" not in json.loads(
        pq.ParquetFile(caminho).schema_arrow.metadata[b"faceaudit:column_specs"]
    )

    relido = FaceDataset.from_parquet(caminho)
    assert "age" in relido.column_specs, "a coluna continuaria invisível"
    assert list(relido.to_pandas()["age"]) == ["00", "01", "02", "03"]


def test_spec_customizado_pelo_usuario_vence_o_catalogo(tmp_path):
    """A cura não pode atropelar customização — precedência é
    metadata > catálogo > sintetizado."""
    caminho = tmp_path / "ds.parquet"
    ds = _dataset()
    ds.to_parquet(caminho)
    relido = FaceDataset.from_parquet(caminho)

    # Usuário renomeia o rótulo e esconde a coluna.
    import dataclasses
    relido.column_specs["fiqa_score"] = dataclasses.replace(
        relido.column_specs["fiqa_score"], label="Qualidade (minha)",
        visible_default=False,
    )
    relido.to_parquet(caminho)

    final = FaceDataset.from_parquet(caminho)
    assert final.column_specs["fiqa_score"].label == "Qualidade (minha)"
    assert final.column_specs["fiqa_score"].visible_default is False


# ───────────────────── lacuna do catálogo (OFIQ) ─────────────────────

@pytest.mark.parametrize("eixo", ["yaw", "pitch", "roll"])
def test_catalogo_cobre_os_escalares_de_pose_do_ofiq(eixo):
    """Faltavam no catálogo: dados reais, invisíveis pelo mesmo motivo.

    Ficam com role=QUALITY, e não POSE, pra não concorrerem com
    `ofiq_yaw`/`_pitch`/`_roll` (que são os valores em graus) no índice por
    role/axis.
    """
    specs = get_default_specs(DatasetType.FACE)
    spec = specs[f"ofiq_head_pose_{eixo}_scalar"]
    assert spec.role == ColumnRole.QUALITY
    assert spec.visible_default is False
