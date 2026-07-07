"""Seção Proveniência (read-only, condicional) no diálogo de Metadados.

Só aparece quando o dataset expõe `provenance_metadata` não-vazio
(getattr defensivo). `get_result()` continua ignorando Caminhos e
Proveniência (só emite {name, description, is_aligned}). Offscreen.
"""

from types import SimpleNamespace

from PyQt6 import QtWidgets

from faceaudit.enums.path_mode import PathMode
from interface.tabs.dataset_manager.column_dialogs.dataset_metadata_editor_dialog import (  # noqa: E501
    DatasetMetadataEditorDialog,
)


def _ds(**over):
    base = dict(
        name="ds",
        description=None,
        is_aligned=False,
        column_specs={},
        path_mode=PathMode.RELATIVE,
        path_root="images",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _find_groupbox(dlg, title):
    for gb in dlg.findChildren(QtWidgets.QGroupBox):
        if gb.title() == title:
            return gb
    return None


def test_provenance_shown_when_present(qtbot):
    ds = _ds(provenance_metadata={
        "dataset_origin_type": "video_frame_extraction",
        "selection_warning": "viés de seleção por âncora",
    })
    dlg = DatasetMetadataEditorDialog(ds)
    qtbot.addWidget(dlg)
    assert _find_groupbox(dlg, "Proveniência") is not None


def test_provenance_hidden_when_absent(qtbot):
    ds = _ds()  # sem atributo provenance_metadata
    dlg = DatasetMetadataEditorDialog(ds)
    qtbot.addWidget(dlg)
    assert _find_groupbox(dlg, "Proveniência") is None


def test_provenance_hidden_when_empty(qtbot):
    ds = _ds(provenance_metadata={})
    dlg = DatasetMetadataEditorDialog(ds)
    qtbot.addWidget(dlg)
    assert _find_groupbox(dlg, "Proveniência") is None


def test_paths_group_always_present(qtbot):
    ds = _ds()
    dlg = DatasetMetadataEditorDialog(ds)
    qtbot.addWidget(dlg)
    assert _find_groupbox(dlg, "Caminhos das imagens") is not None
    assert _find_groupbox(dlg, "Identificação") is not None


def test_get_result_ignores_paths_and_provenance(qtbot):
    ds = _ds(provenance_metadata={"dataset_origin_type": "x"})
    dlg = DatasetMetadataEditorDialog(ds)
    qtbot.addWidget(dlg)
    dlg.txt_name.setText("novo_nome")
    ds_changes, col_changes = dlg.get_result()
    assert ds_changes == {"name": "novo_nome"}
    assert col_changes == {}
