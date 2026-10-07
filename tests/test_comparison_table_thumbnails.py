"""A tabela de comparações mostra as duas faces, não só texto.

Num confronto a comparação É visual: duas miniaturas e o escore dizem mais
que quatro colunas de identidade e nome de arquivo, que obrigam o perito a
conferir nome contra pasta — o trabalho que a tabela existe para poupar.

O obstáculo é que o parquet de escores guarda identidade e `face_hash` e
**não** guarda caminho de imagem, o que é correto: a imagem pertence ao
dataset. A ponte é o `face_hash`, que no parquet do dataset se chama `hash`
e vem com `image_path`, `file_name` e `identity`.

O que os testes travam:

- a localização pelo hash, incluindo dataset RELATIVE, em que o caminho
  gravado só resolve ancorado na pasta do parquet mais o `path_root`;
- a varredura não materializa a linha que não interessa — um dataset de
  referência tem dezenas de milhares de faces e a tabela mostra cem;
- face não encontrada fica sem miniatura e NÃO sem linha: esconder a
  comparação por causa da moldura seria perder o resultado;
- o nome do arquivo real substitui o hash abreviado, e a ressalva sobre o
  hash sai da frase quando isso acontece;
- a coluna de classe desaparece em conjunto de caso, onde seria constante.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.services.face_image_locator_service import (
    find_dataset_parquet,
    locate_faces,
)
from faceaudit.services.score_comparison_table_service import (
    MODO_CASO,
    attach_face_images,
    build_from_dataset,
    build_from_parquet,
)


@pytest.fixture(scope="module")
def app():
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _png(caminho, cor=(200, 120, 60)):
    """Uma imagem de verdade, porque o leitor do Qt é quem vai abri-la."""
    from PIL import Image

    caminho.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 50), cor).save(caminho)
    return caminho


def _dataset_em_disco(
    tmp_path, nome, faces, *, modo=PathMode.ABSOLUTE, raiz=None,
):
    """Grava um parquet de dataset com as imagens no disco.

    `faces` é uma lista de (hash, nome_do_arquivo, identidade).
    """
    pasta = tmp_path / nome
    pasta.mkdir(parents=True, exist_ok=True)
    imagens = pasta / (raiz or ".")
    objetos = []
    for h, arquivo, identidade in faces:
        _png(imagens / arquivo)
        face = FaceInfo(
            image_path=(arquivo if modo is PathMode.RELATIVE
                        else str((imagens / arquivo).resolve())),
        )
        face.identity = identidade
        face.file_name = arquivo
        face.hash = h
        objetos.append(face)
    ds = FaceDataset(
        name=nome, faces=objetos, path_mode=modo, path_root=raiz,
    )
    caminho = pasta / f"{nome}.parquet"
    ds.to_parquet(caminho)
    return caminho


def _face_embutida(h, arquivo, identidade, caminho=None):
    return EmbeddingInfo(
        embedding=np.zeros(4, dtype=np.float32), face_hash=h,
        image_path=str(caminho or ""), identity=identidade, file_name=arquivo,
    )


def _escores(pares, *, casework=False):
    itens = []
    for probe, referencia, escore, mesma in pares:
        amostra = SimilarityScoreSample(
            probe=probe, reference=referencia, raw_score=escore,
            same_person=mesma,
        )
        amostra.same_person = mesma
        itens.append(amostra)
    ds = SimilarityScoreDataset(name="escores", items=itens)
    if casework:
        ds.metadata["is_casework"] = True
    return ds


# ── Localização pelo hash ────────────────────────────────────────────────

def test_encontra_a_imagem_pelo_hash(tmp_path):
    parquet = _dataset_em_disco(tmp_path, "padrao", [
        ("h1", "P1.png", "padrão"), ("h2", "P2.png", "padrão"),
    ])

    achados = locate_faces([parquet], {"h2"})

    assert set(achados) == {"h2"}
    assert achados["h2"].file_name == "P2.png"
    assert achados["h2"].identity == "padrão"
    assert achados["h2"].image_path.is_file()


def test_dataset_relativo_resolve_contra_a_ancora(tmp_path):
    """O caminho gravado num dataset relativo é só o nome do arquivo; sem a
    âncora do parquet mais o `path_root` ele não abre."""
    parquet = _dataset_em_disco(
        tmp_path, "portatil", [("h1", "Q1.png", "Questionadas")],
        modo=PathMode.RELATIVE, raiz="images",
    )

    achados = locate_faces([parquet], {"h1"})

    assert achados["h1"].image_path.is_file(), achados["h1"].image_path


def test_procura_em_varios_datasets(tmp_path):
    """Probe e referência vêm de datasets diferentes num confronto."""
    a = _dataset_em_disco(tmp_path, "padrao", [("h1", "P1.png", "padrão")])
    b = _dataset_em_disco(tmp_path, "quest", [("h9", "Q1.png", "Questionadas")])

    achados = locate_faces([a, b], {"h1", "h9"})

    assert set(achados) == {"h1", "h9"}


def test_hash_ausente_simplesmente_nao_aparece(tmp_path):
    parquet = _dataset_em_disco(tmp_path, "padrao", [("h1", "P1.png", "x")])

    assert locate_faces([parquet], {"inexistente"}) == {}


def test_parquet_inexistente_nao_quebra(tmp_path):
    assert locate_faces([tmp_path / "nao_existe.parquet"], {"h1"}) == {}


def test_parquet_de_outro_schema_e_ignorado(tmp_path):
    import pandas as pd

    caminho = tmp_path / "outro.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(caminho)

    assert locate_faces([caminho], {"h1"}) == {}


def test_varredura_nao_materializa_o_dataset(tmp_path):
    """A linha que não interessa é descartada antes de virar dicionário."""
    from faceaudit.services import face_image_locator_service as svc

    faces = [(f"h{i}", f"F{i}.png", "id") for i in range(60)]
    parquet = _dataset_em_disco(tmp_path, "grande", faces)

    materializadas = []
    original = svc._filtrar_lote

    def espiao(lote, procurados):
        linhas = original(lote, procurados)
        materializadas.append(len(linhas))
        return linhas

    svc._filtrar_lote = espiao
    try:
        achados = svc.locate_faces([parquet], {"h3", "h40"})
    finally:
        svc._filtrar_lote = original

    assert set(achados) == {"h3", "h40"}
    assert sum(materializadas) == 2, (
        f"materializou {sum(materializadas)} linhas de 60 para achar 2"
    )


def test_busca_do_parquet_pelo_nome_do_recurso(tmp_path):
    destino = tmp_path / "datasets" / "primary"
    destino.mkdir(parents=True)
    (destino / "RGMA - Masculino.parquet").write_bytes(b"x")

    achado = find_dataset_parquet(tmp_path, "RGMA - Masculino")

    assert achado is not None and achado.name == "RGMA - Masculino.parquet"
    assert find_dataset_parquet(tmp_path, "nao existe") is None


def test_busca_alcanca_subpasta_de_estrategia(tmp_path):
    """A convenção de onde cada dataset cai mudou ao longo do tempo."""
    destino = tmp_path / "datasets" / "Confronto_3_abc" / "intermediate"
    destino.mkdir(parents=True)
    (destino / "derivado.parquet").write_bytes(b"x")

    assert find_dataset_parquet(tmp_path, "derivado") is not None


# ── A tabela ganha as imagens ────────────────────────────────────────────

def _tabela_de_caso(tmp_path):
    padrao = _dataset_em_disco(tmp_path, "padrao", [
        ("hp1", "P3_CNH.png", "padrão"), ("hp2", "P3_Passaporte.png", "padrão"),
    ])
    quest = _dataset_em_disco(tmp_path, "quest", [
        ("hq1", "Q3_Carved.png", "Questionadas"),
    ])
    escores = _escores([
        (_face_embutida("hp1", "", "padrão"),
         _face_embutida("hq1", "", "Questionadas"), 0.63, None),
        (_face_embutida("hp2", "", "padrão"),
         _face_embutida("hq1", "", "Questionadas"), 0.62, None),
    ], casework=True)
    caminho = tmp_path / "escores.parquet"
    escores.to_parquet(caminho)
    return build_from_parquet(caminho), [padrao, quest]


def test_imagens_sao_anexadas_a_tabela(tmp_path):
    tabela, datasets = _tabela_de_caso(tmp_path)
    assert not tabela.has_images

    com_imagem = attach_face_images(tabela, datasets)

    assert com_imagem.has_images
    assert all(r.probe_image_path for r in com_imagem.rows)
    assert all(r.reference_image_path for r in com_imagem.rows)


def test_o_nome_do_arquivo_substitui_o_hash(tmp_path):
    """Lido do parquet de escores, a face aparecia pelo início do hash; o
    dataset tem o nome real, que é o que serve ao perito."""
    tabela, datasets = _tabela_de_caso(tmp_path)
    assert not tabela.has_file_names

    com_imagem = attach_face_images(tabela, datasets)

    assert com_imagem.has_file_names
    nomes = sorted(r.probe_face for r in com_imagem.rows)
    assert nomes == ["P3_CNH.png", "P3_Passaporte.png"], nomes


def test_a_ressalva_sobre_o_hash_sai_da_frase(tmp_path):
    tabela, datasets = _tabela_de_caso(tmp_path)
    assert "hash" in tabela.note.lower()

    com_imagem = attach_face_images(tabela, datasets)

    assert "hash" not in com_imagem.note.lower()


def test_face_sem_imagem_mantem_a_linha(tmp_path):
    """Esconder a comparação por causa da miniatura seria esconder o
    resultado por causa da moldura."""
    tabela, datasets = _tabela_de_caso(tmp_path)

    com_imagem = attach_face_images(tabela, [datasets[0]])  # falta o outro

    assert len(com_imagem.rows) == len(tabela.rows)
    assert all(r.probe_image_path for r in com_imagem.rows)
    assert all(r.reference_image_path is None for r in com_imagem.rows)
    assert com_imagem.has_images


def test_sem_dataset_nenhum_a_tabela_fica_como_estava(tmp_path):
    tabela, _ = _tabela_de_caso(tmp_path)

    assert attach_face_images(tabela, []) is tabela


def test_payload_hidratado_ja_traz_a_imagem(tmp_path):
    """É o único caminho em que a miniatura aparece sem consultar o
    dataset: o payload carrega `image_path` em cada face."""
    img = _png(tmp_path / "imgs" / "a.png")
    escores = _escores([
        (_face_embutida("h1", "a.png", "P1", img),
         _face_embutida("h2", "a.png", "P1", img), 0.9, True),
    ])

    tabela = build_from_dataset(escores)

    assert tabela.has_images
    assert tabela.rows[0].probe_image_path == str(img)


def test_anexar_nao_refaz_o_que_o_payload_ja_tinha(tmp_path):
    img = _png(tmp_path / "imgs" / "a.png")
    escores = _escores([
        (_face_embutida("h1", "a.png", "P1", img),
         _face_embutida("h2", "a.png", "P1", img), 0.9, True),
    ])
    tabela = build_from_dataset(escores)

    assert attach_face_images(tabela, [tmp_path / "qualquer.parquet"]) is tabela


# ── A interface ──────────────────────────────────────────────────────────

def _widget():
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    return ScoreComparisonTableWidget()


def test_com_imagem_a_tabela_mostra_duas_faces_e_o_escore(app, tmp_path):
    tabela, datasets = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(attach_face_images(tabela, datasets))

        titulos = [
            w._grid.horizontalHeaderItem(i).text()
            for i in range(w._grid.columnCount())
        ]
        assert titulos == ["Probe", "Referência", "Escore"], titulos
        assert w._grid.rowCount() == 2
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_a_classe_aparece_fora_do_modo_caso(app, tmp_path):
    """Em conjunto de referência é a classe que distingue o genuíno de
    menor escore do impostor de maior escore."""
    img = _png(tmp_path / "imgs" / "a.png")
    escores = _escores([
        (_face_embutida("h1", "a.png", "P1", img),
         _face_embutida("h2", "a.png", "P1", img), 0.9, True),
        (_face_embutida("h3", "a.png", "P2", img),
         _face_embutida("h4", "a.png", "P3", img), 0.2, False),
    ])
    tabela = build_from_dataset(escores)
    assert tabela.mode != MODO_CASO

    w = _widget()
    try:
        w.set_table(tabela)
        titulos = [
            w._grid.horizontalHeaderItem(i).text()
            for i in range(w._grid.columnCount())
        ]
        assert titulos == ["Probe", "Referência", "Escore", "Classe"], titulos
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_a_dica_do_mouse_traz_identidade_e_arquivo(app, tmp_path):
    """Identidade e nome de arquivo saíram das colunas; é na dica que o
    perito os encontra."""
    tabela, datasets = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(attach_face_images(tabela, datasets))

        dica = w._grid.item(0, 0).toolTip()
        assert "Identidade: padrão" in dica
        assert "Arquivo: P3_" in dica
        assert "Hash: " in dica
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_sem_imagem_a_tabela_volta_ao_texto(app, tmp_path):
    """Instrução cujo dataset de origem não é alcançável a partir dela
    (PAIR_MINING, SCORE_CALIBRATION) e dataset movido."""
    tabela, _ = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(tabela)

        titulos = [
            w._grid.horizontalHeaderItem(i).text()
            for i in range(w._grid.columnCount())
        ]
        assert titulos[0] == "Identidade (probe)"
        assert w._grid.rowCount() == 2
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_face_sem_imagem_mostra_o_texto_na_celula(app, tmp_path):
    """Célula vazia pareceria comparação sem aquele lado."""
    tabela, datasets = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(attach_face_images(tabela, [datasets[0]]))

        assert w._grid.item(0, 0).text() == ""          # probe tem imagem
        assert w._grid.item(0, 1).text() != ""          # referência não tem
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_a_miniatura_chega_na_celula(app, tmp_path):
    """A decodificação é feita em thread própria; o teste espera o fim e
    confere que o ícone foi aplicado na célula certa."""
    tabela, datasets = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(attach_face_images(tabela, datasets))
        assert w._thread is not None

        w._thread.quit()
        w._thread.wait(5000)
        from PyQt6 import QtCore
        QtCore.QCoreApplication.processEvents()

        com_icone = sum(
            1
            for linha in range(w._grid.rowCount())
            for coluna in (0, 1)
            if not w._grid.item(linha, coluna).icon().isNull()
        )
        assert com_icone > 0, "nenhuma miniatura chegou à tabela"
    finally:
        w._parar_miniaturas()
        w.deleteLater()


def test_remontar_a_tabela_cancela_a_carga_anterior(app, tmp_path):
    tabela, datasets = _tabela_de_caso(tmp_path)
    w = _widget()
    try:
        w.set_table(attach_face_images(tabela, datasets))
        w.show_placeholder()

        assert w._thread is None
        assert w._itens_de_miniatura == []
    finally:
        w._parar_miniaturas()
        w.deleteLater()


# ── O painel resolve os datasets de origem ───────────────────────────────

def test_o_painel_acha_os_datasets_da_instrucao(app, tmp_path):
    """O parquet de escores não diz onde estão as imagens; é a instrução
    que nomeia os datasets comparados."""
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    from faceaudit.models.instructions.strategy_instruction_definition import (
        StrategyInstructionDefinition,
    )
    from faceaudit.models.strategy.lr_strategy_definition import (
        LRStrategyDefinition,
    )
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )

    primary = tmp_path / "datasets" / "primary"
    primary.mkdir(parents=True)
    (primary / "C3 - Padrão.parquet").write_bytes(b"x")
    (primary / "C3 - Questionadas.parquet").write_bytes(b"x")

    instrucao = StrategyInstructionDefinition.from_dict({
        "instruction_type": StrategyInstructionType.SIMILARITY_SCORING.value,
        "instruction_id": "i1", "name": "escorar",
        "probe_dataset_resource_id": "C3 - Padrão",
        "reference_dataset_resource_id": "C3 - Questionadas",
        "output_similarity_scores_resource_id": "Escores C3",
    })
    definicao = LRStrategyDefinition(
        strategy_id="s1", name="Confronto 3", casework_instructions=[instrucao],
    )

    class _Loader:
        def __init__(self, raiz):
            class _S:
                project_root = raiz
            self._storage = _S()

    card = _ResultsCard(instrucao)
    try:
        achados = card._resolve_source_dataset_parquets(
            definicao, None, _Loader(tmp_path),
        )
        nomes = sorted(p.stem for p in achados)
        assert nomes == ["C3 - Padrão", "C3 - Questionadas"], nomes
    finally:
        card.deleteLater()


def test_sem_projeto_o_painel_nao_promete_imagem(app):
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    from faceaudit.models.instructions.strategy_instruction_definition import (
        StrategyInstructionDefinition,
    )
    from faceaudit.models.strategy.lr_strategy_definition import (
        LRStrategyDefinition,
    )
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )

    instrucao = StrategyInstructionDefinition.from_dict({
        "instruction_type": StrategyInstructionType.SIMILARITY_SCORING.value,
        "instruction_id": "i1", "name": "escorar",
        "probe_dataset_resource_id": "X",
    })
    definicao = LRStrategyDefinition(
        strategy_id="s1", name="e", casework_instructions=[instrucao],
    )

    card = _ResultsCard(instrucao)
    try:
        assert card._resolve_source_dataset_parquets(definicao, None, None) == []
    finally:
        card.deleteLater()
