"""Galeria do Dataset Manager — toggle do ribbon "Exibir" + menu de contexto.

Cobre (offscreen):
- Toggle Tabela↔Galeria: o rótulo reflete o DESTINO (tabela→"Galeria",
  galeria→"Tabela").
- Menu de contexto da galeria: Remover / Criar dataset da seleção / Definir
  valor de coluna emitem os índices certos e são gated por seleção.
- SetColumnValueDialog: campo livre vs vocabulário.
- Poda de cópias órfãs no autocontido (apaga só órfã + só RELATIVE).
"""

from __future__ import annotations

import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _ribbon():
    from interface.tabs.dataset_manager.toolbar.ribbons.exibir_ribbon import ExibirRibbon
    return ExibirRibbon()


def test_toggle_label_reflects_destination(app):
    r = _ribbon()
    btn = r.btn_toggle_gallery
    # default = modo tabela (unchecked) → rótulo aponta para o destino "Galeria"
    assert btn.isChecked() is False
    assert btn._label_text == "Galeria"
    # entrar na galeria → rótulo passa a apontar para "Tabela"
    btn.setChecked(True)
    assert btn._label_text == "Tabela"
    # voltar para tabela → rótulo volta a "Galeria"
    btn.setChecked(False)
    assert btn._label_text == "Galeria"


def test_toggle_label_survives_compact(app):
    # O rótulo lógico (`_label_text`) permanece correto mesmo em modo compact
    # (só ícone), garantindo o display certo ao voltar para full.
    r = _ribbon()
    btn = r.btn_toggle_gallery
    btn.setChecked(True)
    btn.set_compact(True)
    assert btn._label_text == "Tabela"
    btn.set_compact(False)
    assert "Tabela" in btn.text()


# ───────────────────────── menu de contexto da galeria ─────────────────────────
def _gallery_with_faces(n=3):
    from interface.tabs.dataset_manager.gallery.gallery_widget import GalleryWidget
    g = GalleryWidget()
    faces = [types.SimpleNamespace(identity=f"id{i}", image_path=f"/x/img{i}.png")
             for i in range(n)]
    g.populate(faces)
    g._load_timer.stop()          # evita I/O de thumbnail em paths falsos
    return g, faces


def _find_action(menu, contains):
    return next((a for a in menu.actions() if contains in (a.text() or "")), None)


def _populated_menu(g):
    # Parent ao widget para o QMenu (e suas QActions) não serem coletados
    # enquanto o teste ainda os inspeciona.
    menu = QtWidgets.QMenu(g)
    g._populate_context_menu(menu)
    return menu


def test_gallery_remove_emits_selected_indices(app):
    g, _ = _gallery_with_faces(3)
    g.list_widget.item(0).setSelected(True)
    g.list_widget.item(2).setSelected(True)
    got = []
    g.remove_faces_requested.connect(lambda idxs: got.append(list(idxs)))
    act = _find_action(_populated_menu(g), "Remover")
    assert act is not None and act.isEnabled()
    act.trigger()
    assert got == [[0, 2]]


def test_gallery_create_subset_emits_selected_indices(app):
    g, _ = _gallery_with_faces(3)
    g.list_widget.item(1).setSelected(True)
    got = []
    g.create_dataset_from_selection_requested.connect(lambda idxs: got.append(list(idxs)))
    act = _find_action(_populated_menu(g), "Criar dataset")
    assert act is not None and act.isEnabled()
    act.trigger()
    assert got == [[1]]


def test_gallery_set_column_value_emits_selected_indices(app):
    g, _ = _gallery_with_faces(2)
    g.list_widget.item(0).setSelected(True)
    g.list_widget.item(1).setSelected(True)
    got = []
    g.set_column_value_requested.connect(lambda idxs: got.append(list(idxs)))
    act = _find_action(_populated_menu(g), "Definir valor")
    assert act is not None and act.isEnabled()
    act.trigger()
    assert got == [[0, 1]]


def test_gallery_actions_disabled_without_selection(app):
    g, _ = _gallery_with_faces(3)              # nada selecionado
    menu = _populated_menu(g)
    for txt in ("Remover", "Criar dataset", "Definir valor", "Comparar par"):
        act = _find_action(menu, txt)
        assert act is not None and not act.isEnabled(), txt


def test_gallery_compare_enabled_exactly_at_two(app):
    g, _ = _gallery_with_faces(3)
    g.list_widget.item(0).setSelected(True)
    g.list_widget.item(1).setSelected(True)
    got = []
    g.compare_pair_requested.connect(lambda a, b: got.append((a, b)))
    act = _find_action(_populated_menu(g), "Comparar este par")
    assert act is not None and act.isEnabled()
    act.trigger()
    assert got == [(0, 1)]


# ───────────────────────── SetColumnValueDialog ─────────────────────────
def test_set_column_value_dialog_free_text(app):
    from interface.tabs.dataset_manager.set_column_value_dialog import SetColumnValueDialog
    d = SetColumnValueDialog([("Identidade", "identity", None)], 3)
    d.txt_value.setText("Fulano")
    d._on_accept()
    assert d.get_result() == ("identity", "Fulano")


def test_set_column_value_dialog_vocabulary(app):
    from interface.tabs.dataset_manager.set_column_value_dialog import SetColumnValueDialog
    d = SetColumnValueDialog([("Gênero", "gender_est", ["M", "F"])], 2)
    assert d.cmb_value.count() == 2            # combo populado com o vocabulário
    d.cmb_value.setCurrentText("F")
    d._on_accept()
    assert d.get_result() == ("gender_est", "F")


def _dlg(col):
    from interface.tabs.dataset_manager.set_column_value_dialog import SetColumnValueDialog
    return SetColumnValueDialog([col], 1)


def test_set_column_value_dialog_bool_combo(app):
    # Coluna bool → combo Verdadeiro/Falso, gravando Python bool (não string).
    d = _dlg(("Erro", "error", None, "bool"))
    assert d.cmb_value.count() == 2 and not d.txt_value.isVisibleTo(d)
    d.cmb_value.setCurrentText("Verdadeiro")
    d._on_accept()
    assert d.get_result() == ("error", True)
    d2 = _dlg(("Erro", "error", None, "bool"))
    d2.cmb_value.setCurrentText("Falso")
    d2._on_accept()
    assert d2.get_result() == ("error", False)


def test_set_column_value_dialog_int(app):
    d = _dlg(("T", "serfiq_t", None, "int"))
    d.txt_value.setText("42")
    d._on_accept()
    key, val = d.get_result()
    assert key == "serfiq_t" and val == 42 and isinstance(val, int)


def test_set_column_value_dialog_float_accepts_dot_and_comma(app):
    d = _dlg(("A", "serfiq_alpha", None, "float"))
    d.txt_value.setText("3.14")
    d._on_accept()
    val = d.get_result()[1]
    assert isinstance(val, float) and abs(val - 3.14) < 1e-9
    d2 = _dlg(("A", "serfiq_alpha", None, "float"))
    d2.txt_value.setText("2,5")                # vírgula normalizada para ponto
    d2._on_accept()
    assert abs(d2.get_result()[1] - 2.5) < 1e-9


def test_set_column_value_dialog_numeric_rejects_garbage(app, monkeypatch):
    warned = {}
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.setdefault("w", True)))
    d = _dlg(("T", "serfiq_t", None, "int"))
    d.txt_value.setText("abc")                 # setText burla o validador
    d._on_accept()
    assert d.get_result() is None              # não fecha com lixo
    assert warned.get("w") is True


# ───────────────────── poda de cópias órfãs (autocontido) ─────────────────────
def _relative_dataset(tmp_path, n=3):
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo
    from faceaudit.enums.path_mode import PathMode
    imgs = tmp_path / "ds_images"
    imgs.mkdir()
    faces = []
    for i in range(n):
        (imgs / f"img{i}.png").write_bytes(b"x")
        faces.append(FaceInfo(image_path=f"img{i}.png"))
    ds = FaceDataset(name="ds", faces=faces, path_mode=PathMode.RELATIVE, path_root="ds_images")
    ds._parquet_dir = tmp_path
    return ds, imgs


def _ctrl():
    from interface.tabs.dataset_manager.dataset_manager_controller import DatasetManagerController
    return DatasetManagerController


def test_target_paths_by_index(tmp_path):
    ds, _ = _relative_dataset(tmp_path)
    fake = types.SimpleNamespace(dataset=ds)
    assert _ctrl()._gallery_target_paths(fake, [2, 0]) == ["img0.png", "img2.png"]


# ── guarda anti-amplificação por image_path duplicado ──────────────────────────
# "Remover" é NÃO-DESTRUTIVO (nenhum arquivo é apagado — não há mais poda). O
# risco residual é image_path duplicado (ex.: localização HASH com conteúdo
# idêntico): operar por caminho afetaria faces não selecionadas. _gallery_match_
# after_sync aborta nesse caso.

def _dup_path_dataset():
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo
    from faceaudit.enums.path_mode import PathMode
    # A e B compartilham "dup.png"; C é único.
    faces = [FaceInfo(image_path="dup.png"), FaceInfo(image_path="dup.png"),
             FaceInfo(image_path="uniq.png")]
    return FaceDataset(name="d", faces=faces, path_mode=PathMode.RELATIVE, path_root=".")


def _match_fake(ds, monkeypatch):
    seen = {}
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.setdefault("warned", True)))
    fake = types.SimpleNamespace(dataset=ds, view=None,
                                 sync_changes_from_table=lambda: None)
    fake._gallery_target_paths = lambda idxs: _ctrl()._gallery_target_paths(fake, idxs)
    return fake, seen


def test_match_unique_paths_ok(app, monkeypatch):
    ds = _dup_path_dataset()
    fake, seen = _match_fake(ds, monkeypatch)
    matched = _ctrl()._gallery_match_after_sync(fake, [2], "X")   # só C (uniq)
    assert matched is not None and len(matched) == 1
    assert matched[0].image_path == "uniq.png"
    assert "warned" not in seen


def test_match_aborts_when_path_amplifies_to_unselected(app, monkeypatch):
    ds = _dup_path_dataset()
    fake, seen = _match_fake(ds, monkeypatch)
    # seleciona só A (idx 0); o caminho "dup.png" também pertence a B (não sel.)
    matched = _ctrl()._gallery_match_after_sync(fake, [0], "X")
    assert matched is None                       # abortado por segurança
    assert seen.get("warned") is True


def test_match_ok_when_all_duplicates_selected(app, monkeypatch):
    ds = _dup_path_dataset()
    fake, seen = _match_fake(ds, monkeypatch)
    # seleciona A e B (ambas as faces de "dup.png") → sem amplificação
    matched = _ctrl()._gallery_match_after_sync(fake, [0, 1], "X")
    assert matched is not None and len(matched) == 2
    assert "warned" not in seen
