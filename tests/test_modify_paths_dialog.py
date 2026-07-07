"""Testes do `ModifyPathsDialog` (janela "Modificar Caminhos").

Cobre: filtragem do combo de operação por `path_mode`, visibilidade dos
sub-forms, `get_result()` (FIX_PATH_ROOT vs migrações), o mapeamento
PathOp→LocalizationMode, o helper de relativização/rejeição do picker e
a validação por amostragem. Offscreen (pytest-qt `qtbot`).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.path_mode import PathMode
from interface.tabs.dataset_manager.modify_paths_dialog import (
    ModifyPathsDialog,
    PathOp,
    _PATHOP_TO_LOCALIZATION_MODE,
)


def _make(qtbot, *, path_mode, path_root=None, parquet_dir=None, dataset=None):
    dlg = ModifyPathsDialog(
        dataset_name="ds",
        suggested_target_dir=Path("/tmp/ds__localized"),
        path_mode=path_mode,
        path_root=path_root,
        parquet_dir=parquet_dir,
        dataset=dataset,
    )
    qtbot.addWidget(dlg)
    return dlg


def _op_index(dlg, op):
    return next(
        i for i in range(dlg.cmb_op.count())
        if dlg.cmb_op.itemData(i) == op
    )


# ── Filtragem por path_mode + defaults ──────────────────────────────────

def test_relative_default_is_fix_path_root(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.RELATIVE, path_root="images")
    assert dlg.cmb_op.count() == 4
    assert dlg.cmb_op.currentData() == PathOp.FIX_PATH_ROOT
    # sub-form de path_root visível; widgets de migração ocultos
    assert dlg.pathroot_box.isVisibleTo(dlg)
    assert not dlg.target_box.isVisibleTo(dlg)
    assert not dlg.naming_box.isVisibleTo(dlg)
    assert not dlg.chk_skip_existing.isVisibleTo(dlg)


def test_absolute_has_no_fix_path_root(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.ABSOLUTE)
    assert dlg.cmb_op.count() == 3
    ops = {dlg.cmb_op.itemData(i) for i in range(dlg.cmb_op.count())}
    assert PathOp.FIX_PATH_ROOT not in ops
    assert dlg.cmb_op.currentData() == PathOp.COPY_ORIGINALS
    # migração visível por default (COPY); sub-form de path_root oculto
    assert dlg.target_box.isVisibleTo(dlg)
    assert not dlg.pathroot_box.isVisibleTo(dlg)


# ── get_result ──────────────────────────────────────────────────────────

def test_get_result_fix_path_root(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.RELATIVE, path_root="images")
    res = dlg.get_result()
    assert res == {"op": PathOp.FIX_PATH_ROOT, "path_root": "images"}


def test_get_result_fix_path_root_empty_is_root(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.RELATIVE, path_root="images")
    dlg.input_path_root.setText("")
    res = dlg.get_result()
    assert res == {"op": PathOp.FIX_PATH_ROOT, "path_root": ""}


def test_get_result_migration_copy(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.ABSOLUTE)
    dlg.input_target.setText("/data/out")
    res = dlg.get_result()
    assert res["op"] == PathOp.COPY_ORIGINALS
    assert res["mode"] == LocalizationMode.COPY_ORIGINALS
    assert res["target_dir"] == Path("/data/out")
    assert res["old_prefix"] is None


def test_get_result_rebase_requires_old_prefix(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.ABSOLUTE)
    dlg.cmb_op.setCurrentIndex(_op_index(dlg, PathOp.REBASE_PREFIX))
    dlg.input_target.setText("/data/out")
    dlg.input_old_prefix.setText("")
    with pytest.raises(ValueError):
        dlg.get_result()


def test_get_result_migration_empty_target_raises(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.ABSOLUTE)
    dlg.input_target.setText("")
    with pytest.raises(ValueError):
        dlg.get_result()


# ── Mapeamento PathOp → LocalizationMode ────────────────────────────────

def test_pathop_maps_to_localization_mode():
    assert (_PATHOP_TO_LOCALIZATION_MODE[PathOp.REWRITE_RELATIVE]
            == LocalizationMode.REWRITE_RELATIVE)
    assert (_PATHOP_TO_LOCALIZATION_MODE[PathOp.COPY_ORIGINALS]
            == LocalizationMode.COPY_ORIGINALS)
    assert (_PATHOP_TO_LOCALIZATION_MODE[PathOp.REBASE_PREFIX]
            == LocalizationMode.REBASE_PREFIX)
    assert PathOp.FIX_PATH_ROOT not in _PATHOP_TO_LOCALIZATION_MODE


# ── Helper de relativização do picker ───────────────────────────────────

def test_relativize_subfolder_inside(tmp_path):
    chosen = tmp_path / "images" / "sub"
    chosen.mkdir(parents=True)
    assert ModifyPathsDialog._relativize_subfolder(chosen, tmp_path) == "images/sub"


def test_relativize_subfolder_root(tmp_path):
    assert ModifyPathsDialog._relativize_subfolder(tmp_path, tmp_path) == ""


def test_relativize_subfolder_outside(tmp_path):
    outside = tmp_path.parent / "somewhere_else_xyz_123"
    assert ModifyPathsDialog._relativize_subfolder(outside, tmp_path) is None


# ── Validação por amostragem ────────────────────────────────────────────

def test_validate_candidate_existing_and_wrong(qtbot, tmp_path):
    sub = tmp_path / "images"
    sub.mkdir()
    (sub / "0001.png").write_bytes(b"x")
    ds = SimpleNamespace(faces=[SimpleNamespace(image_path="0001.png")])
    dlg = _make(
        qtbot, path_mode=PathMode.RELATIVE, path_root="images",
        parquet_dir=tmp_path, dataset=ds,
    )
    ok, _ = dlg._validate_candidate("images")
    assert ok is True
    bad, _ = dlg._validate_candidate("errado")
    assert bad is False


def test_validate_candidate_unavailable_without_dataset(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.RELATIVE, path_root="images")
    ok, msg = dlg._validate_candidate("images")
    assert ok is True  # indisponível ⇒ não bloqueia
    assert "indispon" in msg.lower()


# ── Header "Estado atual" ───────────────────────────────────────────────

def test_state_header_relative(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.RELATIVE, path_root="images")
    txt = dlg.lbl_state.text()
    assert "Relativo" in txt and "images" in txt


def test_state_header_absolute(qtbot):
    dlg = _make(qtbot, path_mode=PathMode.ABSOLUTE)
    assert "Absoluto" in dlg.lbl_state.text()
