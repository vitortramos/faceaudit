"""Aba de metadados do painel de detalhes.

Mostra TODAS as tags lidas do arquivo, agrupadas pela origem. A árvore —
em vez do formulário das outras abas — existe porque são dezenas a
centenas de tags: sem agrupamento e filtro, a aba seria ilegível.

O que os testes travam:

- estado vazio distingue "não há metadado" de "o campo existe mas não
  abriu" — a segunda mensagem é a que evita concluir errado sobre a foto;
- o aviso de que metadado é declaração do arquivo está sempre à vista;
- o filtro esconde grupo que ficou sem filho visível.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from interface.tabs.dataset_manager.detail_panel.detail_panel_view import (  # noqa: E402
    DetailPanelView,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def view(app):
    v = DetailPanelView()
    yield v
    v.deleteLater()


def _grupos(view):
    arvore = view.tree_exif
    return [arvore.topLevelItem(i).text(0) for i in range(arvore.topLevelItemCount())]


def _grupos_visiveis(view):
    arvore = view.tree_exif
    return [
        arvore.topLevelItem(i).text(0)
        for i in range(arvore.topLevelItemCount())
        if not arvore.topLevelItem(i).isHidden()
    ]


def test_aba_existe(view):
    assert [view.tabs.tabText(i) for i in range(view.tabs.count())] == [
        "Geral / Bio", "Qualidade ISO", "Metadados",
    ]


def test_agrupa_por_origem(view):
    view.set_exif_metadata({
        "EXIF:Make": "ACME", "EXIF:Model": "Cam 9",
        "XMP:CreatorTool": "Editor", "File:FileSize": 1234,
        "SemPrefixo": "x",
    })

    assert _grupos(view) == ["EXIF (2)", "File (1)", "Outros (1)", "XMP (1)"]


def test_mostra_tudo_sem_truncar(view):
    tags = {f"EXIF:Tag{i}": i for i in range(120)}
    view.set_exif_metadata(tags)

    grupo = view.tree_exif.topLevelItem(0)
    assert grupo.childCount() == 120, "a aba mostra todas as tags, não uma amostra"


def test_aviso_de_declaracao_sempre_visivel(view):
    view.set_exif_metadata({"EXIF:Make": "ACME"})
    texto = view.lbl_exif_aviso.text().lower()
    assert "declaraç" in texto or "forjad" in texto, (
        "o perito precisa ver que metadado é declaração do arquivo"
    )


def test_estado_vazio_explica_o_motivo(view):
    view.clear_exif_metadata()
    assert view.tree_exif.topLevelItemCount() == 0
    texto = view.lbl_exif_aviso.text().lower()
    assert "recorte" in texto and "origina" in texto, (
        "o vazio precisa dizer POR QUE não há metadado"
    )


def test_mensagem_propria_para_conteudo_ilegivel(view):
    view.clear_exif_metadata("A coluna existe mas não pôde ser lida.")
    assert "não pôde ser lida" in view.lbl_exif_aviso.text()


def test_filtro_esconde_grupo_sem_filho(view):
    view.set_exif_metadata({"EXIF:Make": "ACME", "File:FileSize": 1234})

    view._filtrar_exif("make")

    assert _grupos_visiveis(view) == ["EXIF (1)"]


def test_filtro_casa_tambem_pelo_valor(view):
    view.set_exif_metadata({"EXIF:Make": "ACME", "File:FileSize": 1234})

    view._filtrar_exif("1234")

    assert _grupos_visiveis(view) == ["File (1)"]


def test_filtro_vazio_restaura_tudo(view):
    view.set_exif_metadata({"EXIF:Make": "ACME", "File:FileSize": 1234})
    view._filtrar_exif("make")
    view._filtrar_exif("")

    assert len(_grupos_visiveis(view)) == 2


def test_troca_de_registro_nao_acumula(view):
    view.set_exif_metadata({"EXIF:Make": "ACME"})
    view.set_exif_metadata({"XMP:CreatorTool": "Editor"})

    assert _grupos(view) == ["XMP (1)"]


# ── Controller: leitura da coluna ────────────────────────────────────────

def _controller(app):
    """O controller constrói a própria view — usamos a dele."""
    from interface.tabs.dataset_manager.detail_panel.detail_panel_controller import (
        DetailPanelController,
    )

    return DetailPanelController()


def test_controller_le_a_coluna_exif_json(app):
    ctrl = _controller(app)
    linha = pd.Series({"exif_json": json.dumps({"EXIF:Make": "ACME"})})

    ctrl._carregar_metadados_embutidos(linha)

    assert _grupos(ctrl.view) == ["EXIF (1)"]


def test_controller_distingue_ausente_de_ilegivel(app):
    ctrl = _controller(app)
    view = ctrl.view

    ctrl._carregar_metadados_embutidos(pd.Series({"exif_json": None}))
    ausente = view.lbl_exif_aviso.text()

    ctrl._carregar_metadados_embutidos(pd.Series({"exif_json": "{isso não é json"}))
    ilegivel = view.lbl_exif_aviso.text()

    assert ausente != ilegivel, (
        "campo ilegível não pode passar por 'não há metadado' — o registro "
        "TEM o dado, e isso muda a conclusão"
    )
    assert "inválido" in ilegivel
