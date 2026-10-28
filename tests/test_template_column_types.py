"""Coluna de template nasce com o tipo escolhido, não sempre texto.

O defeito: o criador de template pedia só o NOME da coluna custom. Toda
coluna nascia texto, então uma coluna "idade" extraída do nome do arquivo
virava "12" em vez de 12. Ordenar por ela saía em ordem alfabética, com 10
antes de 9, e o perito tinha de converter o dataset depois de cada
ingestão.

O que os testes travam:

- a conversão por tipo, incluindo os formatos que aparecem em nome de
  arquivo ("012", "1,75");
- valor que não casa com o tipo vira vazio e é CONTADO, não vira texto: uma
  coluna que mistura 12 e "doze" é tipada como texto pelo pandas, e o
  problema voltaria sem ninguém perceber;
- templates gravados antes deste campo continuam funcionando, como texto;
- de ponta a ponta: o valor chega tipado ao dataset e ao parquet.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from faceaudit.models.dir_template import DirTemplate


@pytest.fixture(scope="module")
def app():
    """QApplication com referência viva.

    Criá-lo numa expressão solta deixa o objeto sem referência: ele é
    coletado com widgets ainda vivos e o processo morre no encerramento,
    depois de todos os testes passarem.
    """
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _template(**kwargs) -> DirTemplate:
    base = dict(
        name="t",
        custom_columns=["idade", "peso", "ativo", "obs"],
        custom_column_types={"idade": "int", "peso": "float", "ativo": "bool"},
    )
    base.update(kwargs)
    return DirTemplate(**base)


# ── Conversão ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bruto,esperado", [
    ("12", 12), ("012", 12), ("  7 ", 7), ("-3", -3), ("12.0", 12),
])
def test_inteiro(bruto, esperado):
    """"012" aparece o tempo todo em nome de arquivo numerado."""
    assert _template().convert_custom_value("idade", bruto) == esperado


@pytest.mark.parametrize("bruto,esperado", [
    ("1.75", 1.75), ("1,75", 1.75), ("0", 0.0),
])
def test_decimal(bruto, esperado):
    """Vírgula decimal porque o arquivo pode vir nomeado em português."""
    assert _template().convert_custom_value("peso", bruto) == esperado


@pytest.mark.parametrize("bruto,esperado", [
    ("sim", True), ("S", True), ("1", True), ("true", True),
    ("nao", False), ("não", False), ("0", False), ("false", False),
])
def test_booleano(bruto, esperado):
    assert _template().convert_custom_value("ativo", bruto) is esperado


def test_texto_passa_intacto():
    assert _template().convert_custom_value("obs", "frontal") == "frontal"


def test_coluna_sem_tipo_declarado_e_texto():
    """Template gravado antes deste campo existir continua funcionando."""
    velho = DirTemplate(name="v", custom_columns=["idade"])

    assert velho.convert_custom_value("idade", "12") == "12"
    assert velho.dtype_for("idade") == "str"


# ── Valor que não casa ───────────────────────────────────────────────────

@pytest.mark.parametrize("coluna,bruto", [
    ("idade", "doze"), ("idade", "12,5"), ("peso", "leve"), ("ativo", "talvez"),
])
def test_valor_incompativel_vira_vazio(coluna, bruto):
    """Não cai de volta para texto de propósito: uma coluna misturando
    número e palavra é tipada como texto pelo pandas, e o defeito voltaria
    sem ninguém perceber."""
    assert _template().convert_custom_value(coluna, bruto) is None


def test_inteiro_recusa_decimal_com_virgula():
    """Truncar "12,5" para 12 em silêncio perderia informação; é melhor
    declarar a falha e deixar o perito corrigir o template."""
    assert _template().convert_custom_value("idade", "12,5") is None


def test_vazio_vira_vazio_e_nao_conta_como_falha():
    assert _template().convert_custom_value("idade", "") is None
    assert _template().convert_custom_value("idade", None) is None


# ── Persistência ─────────────────────────────────────────────────────────

def test_tipos_sobrevivem_ao_json():
    t = _template()
    de_volta = DirTemplate.from_dict(t.to_dict())

    assert de_volta.custom_column_types == t.custom_column_types
    assert de_volta.convert_custom_value("idade", "9") == 9


def test_json_antigo_sem_o_campo_carrega():
    dados = {"name": "antigo", "custom_columns": ["idade"]}

    t = DirTemplate.from_dict(dados)

    assert t.custom_columns == ["idade"]
    assert t.dtype_for("idade") == "str"


# ── Ponta a ponta ────────────────────────────────────────────────────────

def test_valor_chega_tipado_ao_dataset(tmp_path):
    """A prova: pasta de imagens, template declarando idade como inteiro, e
    a coluna sai numérica no dataset."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for nome in ("P001_12.png", "P002_9.png"):
        (pasta / nome).write_bytes(b"x")

    t = DirTemplate(
        name="t",
        custom_columns=["idade"],
        custom_column_types={"idade": "int"},
        use_filename_separator=True,
        filename_separator="_",
        filename_map={0: "identity", 1: "idade"},
    )
    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False, template=t,
    )

    idades = sorted(f.extra_data.get("idade") for f in ds.faces)
    assert idades == [9, 12], f"esperava inteiros, veio {idades}"
    assert all(isinstance(v, int) for v in idades)


def test_sem_tipo_declarado_continua_texto(tmp_path):
    """Comportamento anterior preservado para template antigo."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    (pasta / "P001_12.png").write_bytes(b"x")

    t = DirTemplate(
        name="t", custom_columns=["idade"],
        use_filename_separator=True, filename_separator="_",
        filename_map={0: "identity", 1: "idade"},
    )
    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False, template=t,
    )

    assert ds.faces[0].extra_data.get("idade") == "12"


def test_falhas_de_conversao_sao_reportadas(tmp_path):
    """Um valor que não casa com o tipo é sinal de que o template não
    descreve aqueles arquivos, e sumiria sem rastro se ninguém contasse."""
    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for nome in ("P001_12.png", "P002_doze.png"):
        (pasta / nome).write_bytes(b"x")

    t = DirTemplate(
        name="t", custom_columns=["idade"],
        custom_column_types={"idade": "int"},
        use_filename_separator=True, filename_separator="_",
        filename_map={0: "identity", 1: "idade"},
    )
    mensagens = []
    FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False, template=t,
        callback=lambda msg, pct: mensagens.append(msg),
    )

    avisos = [m for m in mensagens if "idade" in m and "convertido" in m]
    assert avisos, f"a falha tem de ser reportada: {mensagens}"
    assert "doze" in avisos[0], "o aviso precisa mostrar o valor que falhou"


# ── A interface oferece a escolha ────────────────────────────────────────

def test_criador_oferece_os_tipos():
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.new_dataset_dialog.template_creator_dialog.template_creator_view import (  # noqa: E501
        TIPOS_DE_COLUNA,
    )

    assert set(TIPOS_DE_COLUNA) == {"str", "int", "float", "bool"}
    assert "Vetor" not in TIPOS_DE_COLUNA.values(), (
        "vetor e lista não saem de um trecho de nome de arquivo"
    )


def test_tipo_aparece_na_lista_e_e_colhido(app):
    """Sem o tipo visível na lista, errá-lo só se descobre quando a coluna
    chega ao dataset como texto."""
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.new_dataset_dialog.template_creator_dialog.template_creator_view import (  # noqa: E501
        TemplateCreatorView,
    )

    v = TemplateCreatorView()
    try:
        v.add_custom_column_item("idade", "int")
        v.add_custom_column_item("obs", "str")

        textos = [
            v.list_columns.item(i).text() for i in range(v.list_columns.count())
        ]
        assert any("idade" in t and "inteiro" in t.lower() for t in textos), textos
        assert v.get_custom_column_types() == {"idade": "int", "obs": "str"}
    finally:
        v.deleteLater()


def test_sugestao_de_tipo_pelo_nome():
    """Poupa o clique no caso comum sem decidir pelo usuário: o diálogo
    apenas pré-seleciona."""
    pytest.importorskip("PyQt6")
    from interface.tabs.dataset_manager.new_dataset_dialog.template_creator_dialog.template_creator_controller import (  # noqa: E501
        TemplateCreatorController as C,
    )

    assert C._tipo_sugerido("idade") == "int"
    assert C._tipo_sugerido("ano_nascimento") == "int"
    assert C._tipo_sugerido("peso") == "float"
    assert C._tipo_sugerido("nome_da_pessoa") == "str"


# ── Inteiro com vazio sobrevive ao parquet ───────────────────────────────

def test_inteiro_com_vazio_continua_inteiro_no_parquet(tmp_path):
    """Idade "NC" (não cadastrado) fica vazia, e a coluna continua inteira.

    O pandas faz decimal toda coluna de inteiros com vazio (43 → 43.0), e o
    parquet a gravava como decimal; reabrir também a devolvia decimal,
    porque o FaceAudit troca a metadata do pandas pela própria.
    """
    import pyarrow.parquet as pq

    from faceaudit.adapters.face_dataset_adapter import FaceDatasetAdapter
    from faceaudit.models.column_catalog import ColumnDType
    from faceaudit.models.face_dataset import FaceDataset

    pasta = tmp_path / "fotos"
    pasta.mkdir()
    for nome in ("ID001_SINPA_43.png", "ID002_SINPA_NC.png", "ID003_SINPA_9.png"):
        (pasta / nome).write_bytes(b"x")
    t = DirTemplate(
        name="t", custom_columns=["idade"], custom_column_types={"idade": "int"},
        use_filename_separator=True, filename_separator="_",
        filename_map={0: "identity", 1: "origin", 2: "idade"},
    )
    ds = FaceDatasetAdapter.dataset_from_dir(
        dir_path=pasta, name="ds", recursive=False, template=t,
    )
    destino = tmp_path / "ds.parquet"
    ds.to_parquet(destino)

    assert str(pq.read_schema(destino).field("idade").type) == "int64"
    relido = FaceDataset.from_parquet(destino)
    assert relido.column_specs["idade"].dtype == ColumnDType.INT
    idades = {f.identity: f.extra_data.get("idade") for f in relido.faces}
    assert idades == {"ID001": 43, "ID002": None, "ID003": 9}
    assert all(type(v) is int for v in idades.values() if v is not None)

    # um segundo ciclo abrir → salvar não pode degradar a coluna
    relido.to_parquet(destino)
    assert str(pq.read_schema(destino).field("idade").type) == "int64"


def test_decimal_redondo_com_vazio_continua_decimal(tmp_path):
    """A decisão olha o valor original: 2.0 é decimal, não inteiro."""
    import pyarrow.parquet as pq

    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo

    faces = [FaceInfo(image_path=f"{i}.png", identity="x") for i in range(2)]
    faces[0].extra_data = {"medida": 2.0}
    faces[1].extra_data = {"medida": None}
    destino = tmp_path / "ds.parquet"
    FaceDataset(name="ds", faces=faces).to_parquet(destino)
    assert str(pq.read_schema(destino).field("medida").type) == "double"


def test_tabela_mostra_inteiro_com_vazio_sem_casas_decimais(tmp_path, app):
    """No arquivo a idade já era inteira; na tabela aparecia 45.00.

    A tabela é montada por `to_pandas`, e o pandas faz decimal a coluna de
    inteiros com vazio. Quem decide como exibir é o tipo do spec.
    """
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo
    from interface.tabs.dataset_manager.table.dataset_table_model import DatasetTableModel

    faces = [FaceInfo(image_path=f"{i}.png", identity=f"ID{i}") for i in range(3)]
    for f, v in zip(faces, (45, None, 9)):
        f.extra_data = {"idade": v, "medida": 2.5 if v else None}
    destino = tmp_path / "ds.parquet"
    FaceDataset(name="ds", faces=faces).to_parquet(destino)
    ds = FaceDataset.from_parquet(destino)

    modelo = DatasetTableModel(ds.to_pandas(include_embeddings=False))
    modelo.set_column_specs(ds.column_specs)
    chaves = [c.key for c in modelo.visible_columns]
    col_idade, col_medida = chaves.index("idade"), chaves.index("medida")
    exibido = [modelo.data(modelo.index(r, col_idade)) for r in range(3)]
    assert exibido == ["45", "", "9"]
    # coluna decimal continua com duas casas
    assert modelo.data(modelo.index(0, col_medida)) == "2.50"
