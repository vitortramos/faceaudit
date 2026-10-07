"""Estratégia duplicada não pode herdar os resultados da original.

O defeito, visto num projeto real: na estratégia "Confronto 3", duplicada
de "Confronto 2", a instrução de SIMILARITY_SCORING da Análise de Caso
mostrava "Execute para gerar visualização da distribuição" apesar de ter
executado sem erro, e o gráfico estava gravado no disco. Nas outras duas
estratégias do mesmo projeto funcionava.

A causa: `instruction_id` é único DENTRO da estratégia e só dentro dela — o
próprio DTO declara isso — e duplicar uma estratégia copia os ids das
instruções. O painel de resultados cacheava os cards por `instruction_id`
puro, então a cópia recebia o card construído para a original. O card
carrega a instrução que o construiu, e aquela instrução aponta para o
recurso de saída da OUTRA estratégia; o painel procurava o gráfico desse
recurso na pasta da estratégia atual, não encontrava, e caía no texto de
"ainda não executou". Silencioso de ponta a ponta: nenhum erro, nenhum
aviso, e o dado certo no disco a um nome de distância.

Agravante: `invalidate_all` descartava os cards chamando
`invalidate_instruction`, que RECONSTRÓI o card da instrução selecionada
(para que re-executar uma instrução apareça de imediato). Na troca de
estratégia a instrução selecionada é a da estratégia que está sendo
abandonada, então o card renascia com ela e voltava ao cache — o descarte
que devia impedir a colisão era justamente o que a instalava.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.similarity.similarity_score_sample import (
    SimilarityScoreSample,
)
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

# O id que as duas estratégias compartilham depois da duplicação. No caso
# real era "b08db0c8", a instrução de escore da Análise de Caso.
ID_COLIDENTE = "b08db0c8"


@pytest.fixture(scope="module")
def app():
    """QApplication com referência viva — sem ela o processo morre no
    encerramento, depois de todos os testes passarem."""
    from PyQt6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _instrucao(nome_da_saida: str) -> StrategyInstructionDefinition:
    return StrategyInstructionDefinition.from_dict({
        "instruction_type": StrategyInstructionType.SIMILARITY_SCORING.value,
        "instruction_id": ID_COLIDENTE,
        "name": "SIMILARITY_SCORING (2)",
        "output_similarity_scores_resource_id": nome_da_saida,
    })


def _estrategia(strategy_id, nome, nome_da_saida, *, ref_path=None):
    instrucao = _instrucao(nome_da_saida)
    recursos = []
    if ref_path is not None:
        recursos.append(StrategyResource(
            resource_id=nome_da_saida, name=nome_da_saida,
            resource_type=StrategyResourceType.SIMILARITY_SCORES,
            artifact_ref=StrategyArtifactReference(
                artifact_type=StrategyArtifactType.SIMILARITY_SCORES,
                path=ref_path, hash="h",
            ),
        ))
    definicao = LRStrategyDefinition(
        strategy_id=strategy_id, name=nome,
        resources=recursos, casework_instructions=[instrucao],
    )
    return definicao, instrucao


class _LoaderFalso:
    def __init__(self, raiz):
        class _S:
            project_root = raiz
        self._storage = _S()


def _painel(app):
    from interface.tabs.strategy_manager_v2.strategy_editor.instruction_results_panel import (  # noqa: E501
        InstructionResultsPanel,
    )

    return InstructionResultsPanel()


# ── O card servido é o da estratégia aberta ──────────────────────────────

def test_cada_estrategia_recebe_o_proprio_card(app):
    painel = _painel(app)
    try:
        defn_a, instr_a = _estrategia("sid-A", "Confronto 2", "Escores C2")
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn_a, {ID_COLIDENTE})
        painel.load_instruction(instr_a)
        card_a = painel._stack.currentWidget()
        assert card_a._instruction is instr_a

        defn_b, instr_b = _estrategia("sid-B", "Confronto 3", "Escores C3")
        painel.set_strategy_id("sid-B")
        painel.update_runtime_state(defn_b, {ID_COLIDENTE})
        painel.load_instruction(instr_b)
        card_b = painel._stack.currentWidget()

        assert card_b._instruction is instr_b, (
            "o painel serviu o card da estratégia anterior"
        )
        assert card_b is not card_a
    finally:
        painel.deleteLater()


def test_a_saida_exibida_e_a_da_estrategia_aberta(app):
    """A consequência visível: o card da outra estratégia aponta para o
    recurso de saída dela, e é por isso que o gráfico não aparece."""
    painel = _painel(app)
    try:
        defn_a, instr_a = _estrategia("sid-A", "Confronto 2", "Escores C2")
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn_a, {ID_COLIDENTE})
        painel.load_instruction(instr_a)

        defn_b, instr_b = _estrategia("sid-B", "Confronto 3", "Escores C3")
        painel.set_strategy_id("sid-B")
        painel.update_runtime_state(defn_b, {ID_COLIDENTE})
        painel.load_instruction(instr_b)

        card = painel._stack.currentWidget()
        nomes = [row.current_value() for row in card._output_widgets.values()]
        assert nomes == ["Escores C3"], nomes
    finally:
        painel.deleteLater()


def test_voltar_para_a_primeira_estrategia_nao_traz_a_segunda(app):
    """A colisão vale nos dois sentidos."""
    painel = _painel(app)
    try:
        defn_a, instr_a = _estrategia("sid-A", "Confronto 2", "Escores C2")
        defn_b, instr_b = _estrategia("sid-B", "Confronto 3", "Escores C3")

        for sid, defn, instr in (
            ("sid-A", defn_a, instr_a),
            ("sid-B", defn_b, instr_b),
            ("sid-A", defn_a, instr_a),
        ):
            painel.set_strategy_id(sid)
            painel.update_runtime_state(defn, {ID_COLIDENTE})
            painel.load_instruction(instr)

        assert painel._stack.currentWidget()._instruction is instr_a
    finally:
        painel.deleteLater()


# ── O descarte não pode reinstalar o card ────────────────────────────────

def test_descartar_tudo_nao_ressuscita_o_card_selecionado(app):
    """Era o agravante: `invalidate_all` reconstruía o card da instrução
    selecionada, que na troca de estratégia é a da estratégia abandonada."""
    painel = _painel(app)
    try:
        defn_a, instr_a = _estrategia("sid-A", "Confronto 2", "Escores C2")
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn_a, {ID_COLIDENTE})
        painel.load_instruction(instr_a)
        assert painel._cards

        painel.invalidate_all()

        assert painel._cards == {}, "o card voltou ao cache depois do descarte"
        assert painel._current_instruction is None
    finally:
        painel.deleteLater()


def test_re_execucao_continua_reconstruindo_de_imediato(app):
    """O comportamento que a reconstrução existe para dar: re-executar uma
    instrução tem de atualizar o card sem o usuário trocar de seleção."""
    painel = _painel(app)
    try:
        defn, instr = _estrategia("sid-A", "Confronto 2", "Escores C2")
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn, {ID_COLIDENTE})
        painel.load_instruction(instr)
        antigo = painel._stack.currentWidget()

        painel.invalidate_instruction(ID_COLIDENTE)

        novo = painel._stack.currentWidget()
        assert novo is not antigo
        assert novo._instruction is instr
    finally:
        painel.deleteLater()


def test_trocar_de_estrategia_pelo_mesmo_id_e_no_op(app):
    """Re-carregar a mesma estratégia não pode jogar os cards fora — é o
    caminho de todo refresh do painel."""
    painel = _painel(app)
    try:
        defn, instr = _estrategia("sid-A", "Confronto 2", "Escores C2")
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn, {ID_COLIDENTE})
        painel.load_instruction(instr)
        card = painel._stack.currentWidget()

        painel.set_strategy_id("sid-A")

        assert painel._cards, "o cache foi descartado sem troca de estratégia"
        assert painel._stack.currentWidget() is card
    finally:
        painel.deleteLater()


# ── O sintoma relatado, de ponta a ponta ─────────────────────────────────

def _gravar_png(caminho):
    from matplotlib.figure import Figure

    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(3, 2))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    fig.savefig(caminho, dpi=50)


def _gravar_parquet(caminho):
    caminho.parent.mkdir(parents=True, exist_ok=True)

    def _face(i):
        return EmbeddingInfo(
            embedding=np.zeros(4, dtype=np.float32), face_hash=f"h{i}",
            image_path=f"/{i}.png", identity=f"P{i}", file_name=f"{i}.png",
        )

    itens = [
        SimilarityScoreSample(probe=_face(1), reference=_face(1),
                              raw_score=0.8, same_person=True),
        SimilarityScoreSample(probe=_face(2), reference=_face(3),
                              raw_score=0.1, same_person=False),
    ]
    SimilarityScoreDataset(name=caminho.stem, items=itens).to_parquet(caminho)


def test_o_grafico_da_estrategia_duplicada_aparece(app, tmp_path):
    """A reprodução do caso real: duas estratégias com o mesmo
    `instruction_id`, cada uma com o seu PNG no disco, e o da segunda tem
    de aparecer depois da troca."""
    from faceaudit.models.similarity.distribution_plot import (
        score_distribution_png_path,
    )

    _gravar_png(score_distribution_png_path(
        tmp_path, "sid-A", "Escores C2", strategy_name="Confronto 2",
    ))
    _gravar_png(score_distribution_png_path(
        tmp_path, "sid-B", "Escores C3", strategy_name="Confronto 3",
    ))
    _gravar_parquet(tmp_path / "scores" / "c2.parquet")
    _gravar_parquet(tmp_path / "scores" / "c3.parquet")

    defn_a, instr_a = _estrategia(
        "sid-A", "Confronto 2", "Escores C2", ref_path="scores/c2.parquet",
    )
    defn_b, instr_b = _estrategia(
        "sid-B", "Confronto 3", "Escores C3", ref_path="scores/c3.parquet",
    )

    painel = _painel(app)
    painel.set_resource_loader(_LoaderFalso(tmp_path))
    try:
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn_a, {ID_COLIDENTE})
        painel.load_instruction(instr_a)
        assert painel._stack.currentWidget()._plot_widget._full_pixmap is not None

        painel.set_strategy_id("sid-B")
        painel.update_runtime_state(defn_b, {ID_COLIDENTE})
        painel.load_instruction(instr_b)

        card = painel._stack.currentWidget()
        assert card._plot_widget._full_pixmap is not None, (
            "o painel não achou o gráfico da estratégia duplicada"
        )
        assert card._plot_widget._placeholder_text == ""
    finally:
        painel.deleteLater()


def test_as_comparacoes_da_estrategia_duplicada_aparecem(app, tmp_path):
    """Mesma colisão, mesma consequência nas abas novas: a tabela saía do
    recurso da outra estratégia."""
    _gravar_parquet(tmp_path / "scores" / "c2.parquet")
    _gravar_parquet(tmp_path / "scores" / "c3.parquet")

    defn_a, instr_a = _estrategia(
        "sid-A", "Confronto 2", "Escores C2", ref_path="scores/c2.parquet",
    )
    defn_b, instr_b = _estrategia(
        "sid-B", "Confronto 3", "Escores C3", ref_path="scores/c3.parquet",
    )

    painel = _painel(app)
    painel.set_resource_loader(_LoaderFalso(tmp_path))
    try:
        painel.set_strategy_id("sid-A")
        painel.update_runtime_state(defn_a, {ID_COLIDENTE})
        painel.load_instruction(instr_a)

        painel.set_strategy_id("sid-B")
        painel.update_runtime_state(defn_b, {ID_COLIDENTE})
        painel.load_instruction(instr_b)

        card = painel._stack.currentWidget()
        card._result_tabs.setCurrentIndex(card._ABA_COMPARACOES)

        assert card._comparison_widget._grid.rowCount() == 2
    finally:
        painel.deleteLater()


# ── O editor declara a estratégia ao painel ──────────────────────────────

def test_o_editor_declara_a_estrategia_ao_carregar(app):
    """Se o editor não declarar, o painel chaveia tudo sob a estratégia
    vazia e a colisão volta."""
    pytest.importorskip("PyQt6")
    from interface.tabs.strategy_manager_v2.strategy_editor.strategy_editor_widget import (  # noqa: E501
        StrategyEditorWidget,
    )

    defn, _ = _estrategia("sid-A", "Confronto 2", "Escores C2")
    editor = StrategyEditorWidget()
    try:
        editor.load_strategy(defn)

        for _, (_, _, painel) in editor._phase_widgets.items():
            assert painel._strategy_id == "sid-A"
    finally:
        editor.deleteLater()
