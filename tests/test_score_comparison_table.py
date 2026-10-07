"""Tabela das comparações que produziram cada escore.

O que faltava: as instruções de scoring mostravam a distribuição agregada e
as contagens, mas não deixavam saber QUAL escore saiu de QUAL par de faces.
Num confronto de caso essa é a informação central do exame; num conjunto de
referência, são os extremos que revelam onde o sistema erra.

O que os testes travam:

- o regime depende de `is_casework`, lido da marca explícita e não inferido
  de `same_person` nulo: um conjunto de calibração tratado como caso
  tentaria listar milhões de linhas;
- fora de caso, a seleção são de fato os genuínos de MENOR escore e os
  impostores de MAIOR escore, limitados por classe;
- a frase que acompanha a tabela declara o recorte, senão uma amostra
  enviesada é lida como se fosse o conjunto;
- parquet e payload produzem a mesma seleção, e a leitura do parquet não
  materializa linha que não vai aparecer;
- o nome do arquivo não existe no parquet, e nesse caso a face aparece pelo
  hash abreviado, com o fato declarado em `has_file_names`.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.services.score_comparison_table_service import (
    CLASSE_CASO,
    CLASSE_GENUINO,
    CLASSE_IMPOSTOR,
    MODO_CASO,
    MODO_EXTREMOS,
    build_comparison_table,
    build_from_dataset,
    build_from_parquet,
)


@pytest.fixture(scope="module")
def app():
    """QApplication com referência viva — sem ela o processo morre no
    encerramento, depois de todos os testes passarem."""
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _face(identidade, nome, hash_=None):
    return EmbeddingInfo(
        embedding=np.zeros(4, dtype=np.float32),
        face_hash=hash_ or f"{identidade}_{nome}_hash_longo_demais",
        image_path=f"/imgs/{nome}",
        identity=identidade,
        file_name=nome,
    )


def _par(id_a, id_b, escore, *, same=None, log_lr=None):
    s = SimilarityScoreSample(
        probe=_face(id_a, f"{id_a}_probe.png"),
        reference=_face(id_b, f"{id_b}_ref.png"),
        raw_score=escore,
        same_person=same,
    )
    if log_lr is not None:
        s.metadata["log_lr"] = log_lr
    return s


def _referencia(n_gen=8, n_imp=8):
    """Conjunto com genuínos e impostores em escores previsíveis."""
    itens = []
    for i in range(n_gen):
        itens.append(_par(f"P{i}", f"P{i}", 0.90 - 0.05 * i, same=True))
    for i in range(n_imp):
        itens.append(_par(f"Q{i}", f"R{i}", 0.10 + 0.05 * i, same=False))
    return SimilarityScoreDataset(name="ref", items=itens)


def _caso(n=4):
    itens = [
        _par("questionada", "padrao", 0.30 + 0.10 * i, same=None)
        for i in range(n)
    ]
    ds = SimilarityScoreDataset(name="caso", items=itens)
    ds.metadata["is_casework"] = True
    return ds


# ── Regime de caso ───────────────────────────────────────────────────────

def test_caso_mostra_todas_as_comparacoes():
    t = build_from_dataset(_caso(4))

    assert t.mode == MODO_CASO
    assert len(t.rows) == 4
    assert t.total_pairs == 4
    assert not t.truncated


def test_caso_vem_do_maior_para_o_menor_escore():
    """É a ordem em que se lê um confronto."""
    t = build_from_dataset(_caso(4))

    escores = [r.score for r in t.rows]
    assert escores == sorted(escores, reverse=True)


def test_caso_nao_classifica_como_genuino_nem_impostor():
    """Ground truth desconhecido é a pergunta do exame, não um dado
    faltante que se possa preencher com um dos dois rótulos.

    E o resíduo é real: `SimilarityScoreSample` infere `same_person`
    comparando as strings de identidade, que em casework são convencionais
    ("questionada", "padrão"). Sem forçar a classe de caso, todo par de um
    confronto apareceria como "pessoas diferentes" — a conclusão do exame
    afirmada pela interface, no lugar do exame.
    """
    ds = _caso(2)
    assert any(i.same_person is False for i in ds.items), (
        "a fixture tem de reproduzir o resíduo que o modelo deixa"
    )

    t = build_from_dataset(ds)

    assert all(r.classe == CLASSE_CASO for r in t.rows)
    assert all(r.same_person is None for r in t.rows)
    assert t.n_genuines == 0 and t.n_impostors == 0


def test_caso_grande_demais_declara_o_corte():
    """Um dataset marcado como caso por engano não pode despejar tudo na
    interface sem avisar."""
    grande = _caso(50)
    grande.metadata["is_casework"] = True

    t = build_from_dataset(grande, limite_caso=10)

    assert len(t.rows) == 10
    assert t.truncated
    assert "50" in t.note


# ── Regime de extremos ───────────────────────────────────────────────────

def test_fora_de_caso_seleciona_os_extremos():
    t = build_from_dataset(_referencia(), extremos_por_classe=3)

    assert t.mode == MODO_EXTREMOS
    assert len(t.rows) == 6

    genuinos = [r for r in t.rows if r.classe == CLASSE_GENUINO]
    impostores = [r for r in t.rows if r.classe == CLASSE_IMPOSTOR]
    # Genuínos de MENOR escore: os três mais baixos do conjunto.
    assert [round(r.score, 2) for r in genuinos] == [0.55, 0.60, 0.65]
    # Impostores de MAIOR escore: os três mais altos.
    assert [round(r.score, 2) for r in impostores] == [0.45, 0.40, 0.35]


def test_contagem_total_e_do_conjunto_nao_da_tabela():
    """Sem isso o perito leria "3 genuínos" num conjunto de oito."""
    t = build_from_dataset(_referencia(8, 8), extremos_por_classe=3)

    assert t.total_pairs == 16
    assert t.n_genuines == 8
    assert t.n_impostors == 8
    assert t.truncated


def test_conjunto_menor_que_o_limite_nao_e_truncado():
    t = build_from_dataset(_referencia(2, 2), extremos_por_classe=50)

    assert len(t.rows) == 4
    assert not t.truncated


def test_classe_ausente_nao_inventa_linha():
    """Pares sem ground truth num conjunto que não é de caso ficam fora:
    não são extremo de classe nenhuma."""
    ds = _referencia(2, 2)
    sem_classe = _par("X", "Y", 0.99, same=None)
    sem_classe.same_person = None  # o modelo infere pela identidade
    ds.items.append(sem_classe)

    t = build_from_dataset(ds, extremos_por_classe=50)

    assert all(r.classe != CLASSE_CASO for r in t.rows)
    assert 0.99 not in [round(r.score, 2) for r in t.rows]


# ── A frase que acompanha ────────────────────────────────────────────────

def test_a_nota_declara_o_recorte():
    """Sem ela o regime de extremos passa por tabela completa."""
    t = build_from_dataset(_referencia(100, 100), extremos_por_classe=5)

    assert "extremos" in t.note.lower()
    assert "menor" in t.note and "maior" in t.note
    assert "200" in t.note


def test_a_nota_usa_separador_de_milhar_brasileiro():
    t = build_from_dataset(_referencia(600, 600), extremos_por_classe=5)

    assert "1.200" in t.note


def test_a_nota_do_caso_nao_fala_de_extremos():
    t = build_from_dataset(_caso(3))

    assert "extremos" not in t.note.lower()
    assert "caso" in t.note.lower()


def test_singular_quando_ha_uma_comparacao():
    t = build_from_dataset(_caso(1))

    assert "1 comparação" in t.note
    assert "comparações" not in t.note


# ── log-LR ───────────────────────────────────────────────────────────────

def test_log_lr_aparece_quando_existe():
    ds = _referencia(2, 2)
    for i, item in enumerate(ds.items):
        item.metadata["log_lr"] = float(i)

    t = build_from_dataset(ds)

    assert t.has_log_lr
    assert all(r.log_lr is not None for r in t.rows)


def test_sem_log_lr_a_coluna_nao_e_prometida():
    t = build_from_dataset(_referencia(2, 2))

    assert not t.has_log_lr


# ── Parquet ──────────────────────────────────────────────────────────────

def _gravar(ds, tmp_path, nome="scores.parquet"):
    caminho = tmp_path / nome
    ds.to_parquet(caminho)
    return caminho


def test_parquet_da_a_mesma_selecao_que_o_payload(tmp_path):
    ds = _referencia(8, 8)
    caminho = _gravar(ds, tmp_path)

    do_disco = build_from_parquet(caminho, extremos_por_classe=3)
    da_memoria = build_from_dataset(ds, extremos_por_classe=3)

    assert [round(r.score, 6) for r in do_disco.rows] == [
        round(r.score, 6) for r in da_memoria.rows
    ]
    assert do_disco.total_pairs == da_memoria.total_pairs


def test_parquet_de_caso_e_reconhecido_pela_marca(tmp_path):
    caminho = _gravar(_caso(3), tmp_path)

    t = build_from_parquet(caminho)

    assert t.mode == MODO_CASO
    assert len(t.rows) == 3


def test_parquet_sem_a_marca_fica_no_regime_seguro(tmp_path):
    """Inferir "é caso" de `same_person` nulo faria um conjunto de
    calibração antigo tentar listar tudo."""
    ds = _referencia(4, 4)
    caminho = _gravar(ds, tmp_path)

    t = build_from_parquet(caminho)

    assert t.mode == MODO_EXTREMOS


def test_parquet_nao_tem_nome_de_arquivo_e_declara(tmp_path):
    caminho = _gravar(_referencia(2, 2), tmp_path)

    t = build_from_parquet(caminho)

    assert not t.has_file_names
    assert "hash" in t.note.lower()
    assert all(len(r.probe_face) <= 10 for r in t.rows)


def test_payload_mostra_o_nome_do_arquivo():
    t = build_from_dataset(_referencia(1, 1))

    assert t.has_file_names
    assert any(r.probe_face.endswith(".png") for r in t.rows)


def test_arquivo_ausente_devolve_nada_em_vez_de_tabela_vazia(tmp_path):
    """"Arquivo ausente" e "nenhuma comparação" têm de ser distinguíveis
    por quem exibe."""
    assert build_from_parquet(tmp_path / "nao_existe.parquet") is None


def test_parquet_de_outro_schema_e_recusado(tmp_path):
    import pandas as pd

    caminho = tmp_path / "outro.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(caminho)

    assert build_from_parquet(caminho) is None


def test_log_lr_nulo_no_parquet_nao_vira_nan_na_tabela(tmp_path):
    """NaN é como o parquet guarda "este par não foi calibrado"; exibir
    "nan" na coluna seria pior que deixá-la vazia."""
    ds = _referencia(2, 2)
    ds.items[0].metadata["log_lr"] = 1.5  # só um par calibrado
    caminho = _gravar(ds, tmp_path)

    t = build_from_parquet(caminho)

    assert t.has_log_lr
    valores = [r.log_lr for r in t.rows]
    assert any(v is None for v in valores)
    assert all(v is None or v == v for v in valores)


def test_varredura_do_parquet_nao_materializa_o_conjunto(tmp_path):
    """A varredura é vetorizada de propósito: converter 5 milhões de
    linhas em dicionários para descartar quase todas travaria a interface.
    O teste conta as linhas que de fato viram dicionário."""
    from faceaudit.services import score_comparison_table_service as svc

    ds = _referencia(400, 400)
    caminho = _gravar(ds, tmp_path)

    materializadas = []
    original = svc._materializar

    def espiao(lote, indices):
        materializadas.append(len(indices))
        return original(lote, indices)

    svc._materializar = espiao
    try:
        t = svc.build_from_parquet(caminho, extremos_por_classe=10)
    finally:
        svc._materializar = original

    assert t.total_pairs == 800
    assert sum(materializadas) <= 40, (
        f"materializou {sum(materializadas)} linhas de 800 para mostrar 20"
    )


def test_conjunto_vazio_nao_quebra():
    vazio = SimilarityScoreDataset(name="v", items=[])

    t = build_from_dataset(vazio)

    assert t.is_empty
    assert "Nenhuma" in t.note


# ── Despacho ─────────────────────────────────────────────────────────────

def test_despacho_prefere_o_payload(tmp_path):
    """Só ele traz o nome do arquivo."""
    ds = _referencia(2, 2)
    caminho = _gravar(ds, tmp_path)

    t = build_comparison_table(dataset=ds, parquet_path=caminho)

    assert t.has_file_names


def test_despacho_cai_no_parquet_sem_payload(tmp_path):
    caminho = _gravar(_referencia(2, 2), tmp_path)

    t = build_comparison_table(dataset=None, parquet_path=caminho)

    assert t is not None
    assert not t.has_file_names


def test_despacho_sem_fonte_nenhuma():
    assert build_comparison_table() is None


# ── A interface exibe ────────────────────────────────────────────────────

def test_widget_mostra_uma_linha_por_comparacao(app):
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    t = build_from_dataset(_referencia(3, 3), extremos_por_classe=2)
    w = ScoreComparisonTableWidget()
    try:
        w.set_table(t)

        assert w._grid.rowCount() == 4
        assert w._note.text() == t.note
    finally:
        w.deleteLater()


def test_widget_so_promete_a_coluna_de_log_lr_quando_ha(app):
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    w = ScoreComparisonTableWidget()
    try:
        w.set_table(build_from_dataset(_referencia(2, 2)))
        assert w._grid.columnCount() == 6

        ds = _referencia(2, 2)
        for item in ds.items:
            item.metadata["log_lr"] = 1.0
        w.set_table(build_from_dataset(ds))
        assert w._grid.columnCount() == 7
    finally:
        w.deleteLater()


def test_escore_ordena_por_valor_e_nao_por_texto(app):
    """Com vírgula decimal, a ordenação alfabética põe "0,9" antes de
    "0,85" — erro que uma tabela de escores não pode cometer."""
    pytest.importorskip("PyQt6")
    from PyQt6 import QtCore
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    itens = [
        _par("A", "A", 0.9, same=True),
        _par("B", "B", 0.85, same=True),
        _par("C", "C", 0.1, same=True),
    ]
    ds = SimilarityScoreDataset(name="s", items=itens)

    w = ScoreComparisonTableWidget()
    try:
        w.set_table(build_from_dataset(ds))
        w._grid.sortItems(4, QtCore.Qt.SortOrder.AscendingOrder)

        textos = [w._grid.item(i, 4).text() for i in range(w._grid.rowCount())]
        assert textos == ["0,1000", "0,8500", "0,9000"], textos
    finally:
        w.deleteLater()


def test_placeholder_antes_de_executar(app):
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.result_widgets import (
        ScoreComparisonTableWidget,
    )

    w = ScoreComparisonTableWidget()
    try:
        assert w._grid.rowCount() == 0
        assert "Execute" in w._note.text()
    finally:
        w.deleteLater()


# ── As três abas no painel de resultados ─────────────────────────────────

def _card(tipo):
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )
    from faceaudit.models.instructions.strategy_instruction_definition import (
        StrategyInstructionDefinition,
    )

    instrucao = StrategyInstructionDefinition.from_dict(
        {"instruction_type": tipo.value, "instruction_id": "i1", "name": "i1"}
    )
    return _ResultsCard(instrucao), StrategyInstructionType


def test_instrucao_de_escore_ganha_tres_abas(app):
    pytest.importorskip("PyQt6")
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType

    card, _ = _card(StrategyInstructionType.SIMILARITY_SCORING)
    try:
        assert card._result_tabs is not None
        titulos = [
            card._result_tabs.tabText(i)
            for i in range(card._result_tabs.count())
        ]
        assert titulos == ["Gráfico", "Comparações", "Estatísticas"]
    finally:
        card.deleteLater()


def test_instrucao_sem_escore_continua_com_grafico_solto(app):
    """A mudança não pode vazar para instruções que não produzem pares."""
    pytest.importorskip("PyQt6")
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType

    card, _ = _card(StrategyInstructionType.STRATEGY_EVALUATION)
    try:
        assert card._result_tabs is None
    finally:
        card.deleteLater()


def test_contagens_saem_da_linha_de_saida_e_vao_para_a_aba(app):
    """Repetir as contagens nos dois lugares era o que o usuário pediu
    para evitar: a linha de saída diz o que o artefato É, a aba diz o que
    ele CONTÉM."""
    pytest.importorskip("PyQt6")
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType

    card, _ = _card(StrategyInstructionType.SIMILARITY_SCORING)
    try:
        for row in card._output_widgets.values():
            assert row._show_scores_details is False
    finally:
        card.deleteLater()


def test_linha_de_saida_de_outra_instrucao_mantem_as_contagens(app):
    pytest.importorskip("PyQt6")
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType

    card, _ = _card(StrategyInstructionType.STRATEGY_EVALUATION)
    try:
        for row in card._output_widgets.values():
            assert row._show_scores_details is True
    finally:
        card.deleteLater()


def test_aba_de_estatisticas_nao_repete_caminho_e_hash(app):
    pytest.importorskip("PyQt6")
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType

    card, _ = _card(StrategyInstructionType.SIMILARITY_SCORING)
    try:
        assert card._stats_widget is not None
        assert card._stats_widget._show_basic_info is False
        assert card._stats_widget._show_scores_details is True
    finally:
        card.deleteLater()


# ── Ponta a ponta no painel de resultados ────────────────────────────────

class _LoaderFalso:
    """Mínimo que o painel consulta: `_storage.project_root`."""

    def __init__(self, raiz):
        class _S:
            project_root = raiz
        self._storage = _S()


def _definicao_com_parquet(tmp_path, ds, *, nome="scores"):
    from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
    from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
    from faceaudit.enums.strategy_resource_type import StrategyResourceType
    from faceaudit.models.instructions.strategy_instruction_definition import (
        StrategyInstructionDefinition,
    )
    from faceaudit.models.storage.strategy_artifact_reference import (
        StrategyArtifactReference,
    )
    from faceaudit.models.strategy.lr_strategy_definition import (
        LRStrategyDefinition,
    )
    from faceaudit.models.strategy.strategy_resource import StrategyResource

    pasta = tmp_path / "scores"
    pasta.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(pasta / f"{nome}.parquet")

    instrucao = StrategyInstructionDefinition.from_dict({
        "instruction_type": StrategyInstructionType.SIMILARITY_SCORING.value,
        "instruction_id": "i1", "name": "escorar",
        "output_similarity_scores_resource_id": nome,
    })
    recurso = StrategyResource(
        resource_id=nome, name=nome,
        resource_type=StrategyResourceType.SIMILARITY_SCORES,
        artifact_ref=StrategyArtifactReference(
            artifact_type=StrategyArtifactType.SIMILARITY_SCORES,
            path=f"scores/{nome}.parquet", hash="h",
        ),
    )
    definicao = LRStrategyDefinition(
        strategy_id="s1", name="estrategia",
        resources=[recurso], preparation_instructions=[instrucao],
    )
    return definicao, instrucao


def test_painel_le_as_comparacoes_do_parquet(app, tmp_path):
    """O caminho normal: o conjunto está em disco, não hidratado — é dele
    que a tabela tem de sair, sem o usuário carregar 5 milhões de pares na
    memória só para ver cinquenta linhas."""
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )

    definicao, instrucao = _definicao_com_parquet(tmp_path, _referencia(6, 6))
    card = _ResultsCard(instrucao)
    try:
        card.update_state(
            definicao, {"i1"}, None, resource_loader=_LoaderFalso(tmp_path),
        )
        # A aba abre no gráfico: a varredura é adiada de propósito.
        assert card._comparison_signature is None

        card._result_tabs.setCurrentIndex(card._ABA_COMPARACOES)

        assert card._comparison_signature is not None
        assert card._comparison_widget._grid.rowCount() == 12
    finally:
        card.deleteLater()


def test_painel_nao_varre_o_parquet_na_aba_do_grafico(app, tmp_path):
    """Varrer a cada `update_state` seria trabalho jogado fora: a aba que
    abre por padrão é a do gráfico."""
    pytest.importorskip("PyQt6")
    from faceaudit.services import score_comparison_table_service as svc
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )

    definicao, instrucao = _definicao_com_parquet(tmp_path, _referencia(4, 4))
    chamadas = []
    original = svc.build_from_parquet
    svc.build_from_parquet = lambda *a, **k: (
        chamadas.append(1) or original(*a, **k)
    )
    card = _ResultsCard(instrucao)
    try:
        for _ in range(3):
            card.update_state(
                definicao, {"i1"}, None,
                resource_loader=_LoaderFalso(tmp_path),
            )
        assert chamadas == []

        card._result_tabs.setCurrentIndex(card._ABA_COMPARACOES)
        assert len(chamadas) == 1

        # Re-render do mesmo recurso não varre de novo.
        card.update_state(
            definicao, {"i1"}, None, resource_loader=_LoaderFalso(tmp_path),
        )
        assert len(chamadas) == 1
    finally:
        svc.build_from_parquet = original
        card.deleteLater()


def test_painel_sem_resultado_mostra_placeholder(app):
    pytest.importorskip("PyQt6")
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
    })
    definicao = LRStrategyDefinition(
        strategy_id="s1", name="e", preparation_instructions=[instrucao],
    )
    card = _ResultsCard(instrucao)
    try:
        card.update_state(definicao, set(), None, resource_loader=None)
        card._result_tabs.setCurrentIndex(card._ABA_COMPARACOES)

        assert card._comparison_widget._grid.rowCount() == 0
    finally:
        card.deleteLater()


def test_re_execucao_descarta_a_tabela_anterior(app, tmp_path):
    """Manter a tabela antiga visível seria mostrar o resultado da execução
    anterior como se fosse o atual."""
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        _ResultsCard,
    )

    definicao, instrucao = _definicao_com_parquet(tmp_path, _referencia(3, 3))
    card = _ResultsCard(instrucao)
    try:
        card.update_state(
            definicao, {"i1"}, None, resource_loader=_LoaderFalso(tmp_path),
        )
        card._result_tabs.setCurrentIndex(card._ABA_COMPARACOES)
        assert card._comparison_widget._grid.rowCount() == 6

        card.invalidate()

        assert card._comparison_widget._grid.rowCount() == 0
        assert card._comparison_signature is None
    finally:
        card.deleteLater()
