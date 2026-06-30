"""Fase 2 — âncora multifonte (vários quadros/vídeos + imagem estática).

Modelo (`VideoTarget.add_anchor`/`anchor_source_counts`) + integração na janela
A (clique acrescenta âncora; botão "Imagem de referência" ancora por foto;
alvo só-foto habilita extrair). Offscreen, sem GPU — o serviço é um fake que
devolve um embedding fixo; a foto é um PNG sintético no disco.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets  # noqa: E402


def _add_video_to_list(dlg, path):
    it = QtWidgets.QListWidgetItem(path)
    it.setData(QtCore.Qt.ItemDataRole.UserRole, path)
    dlg.lst_videos.addItem(it)

from faceaudit.models.video.video_target import VideoTarget  # noqa: E402
from faceaudit.services.image_io import imwrite_unicode_safe  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


# ───────────────────────── modelo ─────────────────────────
def test_add_anchor_aggregated_and_counts():
    t = VideoTarget(label="x")
    t.add_anchor(np.ones(512, np.float32), {"type": "video", "frame_index": 10, "bbox": [0, 0, 5, 5]})
    assert t.anchor_policy == "SINGLE" and t.anchor_frame_index == 10
    t.add_anchor(np.full(512, 2.0, np.float32), {"type": "image", "filename": "f.png", "frame_index": None})
    assert t.anchor_policy == "AGGREGATED" and t.anchor_embedding_count == 2
    ref = t.reference_embedding()
    assert abs(float(np.linalg.norm(ref)) - 1.0) < 1e-6   # referência L2-normalizada
    assert t.anchor_source_counts() == (1, 1)


def test_photo_only_target_has_no_annotated_frame():
    t = VideoTarget(label="x")
    t.add_anchor(np.ones(512, np.float32), {"type": "image", "filename": "f.png", "frame_index": None})
    assert t.anchor_frame_index is None        # foto-only não define quadro anotado
    assert t.anchor_embedding_count == 1
    assert t.reference_embedding() is not None  # mas a referência existe (busca por foto)
    assert t.anchor_source_counts() == (0, 1)


# ───────────────────────── janela A (fake service) ─────────────────────────
class _FakeSvc:
    """Serviço fake: cada clique gera um embedding one-hot DISTINTO (faces
    diferentes). `force_duplicate=True` devolve o 1º embedding de novo (mesma
    face) e espelha a deduplicação do serviço real."""
    default_model = "m"

    def __init__(self):
        self._k = 0
        self.force_duplicate = False
        self._first = None

    def detect_in_frame(self, frame):
        return [{"bbox": [0, 0, 10, 10]}]

    def close_preview_cap(self):
        pass

    def _new_emb(self):
        v = np.zeros(512, np.float32)
        v[self._k % 512] = 1.0
        self._k += 1
        return v

    def build_target_from_click(self, frame, x, y, *, label, similarity_threshold,
                                model=None, frame_index=None):
        t = VideoTarget(label=label, similarity_threshold=similarity_threshold,
                        recognition_model=model)
        self._first = self._new_emb()
        t.add_anchor(self._first,
                     {"type": "video", "frame_index": frame_index, "bbox": [0, 0, 10, 10], "det_index": 0})
        return t, 0, []

    def add_anchor_from_click(self, target, frame, x, y, *, model=None, source=None):
        emb = self._first if self.force_duplicate else self._new_emb()
        if target.has_near_duplicate_anchor(emb):
            return 0, []          # mesma face — não duplica (espelha o serviço real)
        target.add_anchor(emb, dict(source or {"type": "video"}))
        return 0, []


def _setup_dialog_plain():
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        VideoDatasetSetupDialog,
    )
    return VideoDatasetSetupDialog(None, use_gpu=False, model="adaface",
                                   datasets_dir=".", output_dir="C:/tmp/runs")


def _dlg_with_fake():
    dlg = _setup_dialog_plain()
    dlg._service = _FakeSvc()
    # Marca o fake como compatível com o detector atual (como o código real faz
    # ao construir o serviço) — senão _ensure_service_current() o reconstruiria.
    dlg._service_sig = dlg._detector_signature()
    dlg._current_frame = np.zeros((20, 20, 3), np.uint8)
    dlg._current_dets = [{"bbox": [0, 0, 10, 10]}]
    dlg.scrubber.set_range(0, 100)
    dlg.scrubber.setEnabled(True)
    return dlg


def test_current_anchor_source_video_vs_image(app):
    dlg = _dlg_with_fake()
    dlg._video_path = "C:/v/clip.mp4"
    dlg._anchor_source_is_image = False
    s = dlg._current_anchor_source()
    assert s["type"] == "video" and s["filename"] == "clip.mp4"
    dlg._anchor_source_is_image = True
    dlg._anchor_image_path = "C:/p/foto.png"
    s2 = dlg._current_anchor_source()
    assert s2["type"] == "image" and s2["filename"] == "foto.png" and s2["frame_index"] is None


def test_second_click_adds_anchor_aggregated(app):
    dlg = _dlg_with_fake()
    row = dlg._active_row
    dlg._on_view_clicked(5, 5)
    assert row.has_anchor() and row.video_target.anchor_embedding_count == 1
    assert row.video_target.anchor_policy == "SINGLE"
    dlg._on_view_clicked(6, 6)
    assert row.video_target.anchor_embedding_count == 2
    assert row.video_target.anchor_policy == "AGGREGATED"
    assert row.video_target.anchor_source_counts() == (2, 0)
    assert "2 âncoras" in row.lbl_status.text()


def test_reference_image_adds_photo_anchor(app, monkeypatch, tmp_path):
    dlg = _dlg_with_fake()
    p = tmp_path / "foto.png"
    imwrite_unicode_safe(p, np.full((30, 30, 3), 100, np.uint8))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(p), "")))
    row = dlg._active_row
    dlg._on_add_reference_image()
    assert dlg._anchor_source_is_image is True
    dlg._on_view_clicked(5, 5)                  # clica na face da foto
    assert row.has_anchor()
    n_vid, n_img = row.video_target.anchor_source_counts()
    assert n_img == 1 and n_vid == 0
    assert dlg.btn_extract.isEnabled()          # alvo só-foto habilita extrair


def test_clicking_same_face_does_not_duplicate(app):
    dlg = _dlg_with_fake()
    row = dlg._active_row
    dlg._on_view_clicked(5, 5)                  # 1ª âncora
    assert row.video_target.anchor_embedding_count == 1
    dlg._service.force_duplicate = True
    dlg._on_view_clicked(5, 5)                  # MESMA face → não duplica
    assert row.video_target.anchor_embedding_count == 1
    assert "não duplicada" in dlg.lbl_frame.text()


def test_anchors_button_is_a_real_button(app):
    dlg = _dlg_with_fake()
    assert isinstance(dlg._active_row.btn_anchors, QtWidgets.QPushButton)


def test_video_anchor_then_photo_anchor_mix(app, monkeypatch, tmp_path):
    dlg = _dlg_with_fake()
    row = dlg._active_row
    dlg._on_view_clicked(5, 5)                  # 1ª âncora: vídeo
    p = tmp_path / "ref.png"
    imwrite_unicode_safe(p, np.full((30, 30, 3), 80, np.uint8))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(p), "")))
    dlg._on_add_reference_image()
    dlg._on_view_clicked(5, 5)                  # 2ª âncora: foto
    assert row.video_target.anchor_source_counts() == (1, 1)
    assert "1 vídeo" in row.lbl_status.text() and "1 foto" in row.lbl_status.text()


# ─────────────── não exigir carregar o vídeo / extrair vídeo inteiro ───────────────
def test_reference_image_without_loading_video(app, monkeypatch, tmp_path):
    dlg = _setup_dialog_plain()                 # NÃO carrega vídeo, sem _service
    fake = _FakeSvc()
    monkeypatch.setattr(dlg, "_ensure_service",
                        lambda *a, **k: setattr(dlg, "_service", fake))
    p = tmp_path / "ref.png"
    imwrite_unicode_safe(p, np.full((30, 30, 3), 90, np.uint8))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(p), "")))
    row = dlg._active_row
    dlg._on_add_reference_image()               # constrói serviço sob demanda
    assert dlg._service is fake
    dlg._on_view_clicked(5, 5)
    assert row.has_anchor() and row.video_target.anchor_source_counts() == (0, 1)
    assert dlg.btn_extract.isEnabled()


def test_get_config_whole_video_when_not_loaded(app):
    dlg = _setup_dialog_plain()
    _add_video_to_list(dlg, "C:/v/clip.mp4")
    cfg = dlg.get_config("C:/out")
    assert cfg.video_path == "C:/v/clip.mp4"
    assert cfg.video_paths == ["C:/v/clip.mp4"]
    assert cfg.start_s == 0.0 and cfg.end_s is None     # vídeo inteiro


def test_effective_video_path_from_list(app):
    dlg = _setup_dialog_plain()
    _add_video_to_list(dlg, "C:/v/clip.mp4")
    assert dlg._effective_video_path() == "C:/v/clip.mp4"


def test_get_config_multivideo_sets_paths_and_case(app, monkeypatch):
    dlg = _setup_dialog_plain()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (["C:/v/a.mp4", "C:/v/b.mp4"], "")))
    dlg._on_add_videos()
    dlg.txt_case.setText("caso_x")
    cfg = dlg.get_config("C:/out")
    assert cfg.video_paths == ["C:/v/a.mp4", "C:/v/b.mp4"]
    assert cfg.case_name == "caso_x"
    assert cfg.start_s == 0.0 and cfg.end_s is None     # multivídeo → vídeo inteiro


def test_video_list_add_dedup_and_remove(app, monkeypatch):
    dlg = _setup_dialog_plain()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (["C:/v/a.mp4", "C:/v/a.mp4", "C:/v/b.mp4"], "")))
    dlg._on_add_videos()
    assert dlg._video_paths() == ["C:/v/a.mp4", "C:/v/b.mp4"]     # sem duplicata
    dlg.lst_videos.setCurrentRow(0)
    dlg._on_remove_video()
    assert dlg._video_paths() == ["C:/v/b.mp4"]


# ─────────────────────────── janela por vídeo ───────────────────────────
def test_video_window_dialog_result(app):
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        _VideoWindowDialog,
    )
    d = _VideoWindowDialog(None, name="x", start_s=300.0, end_s=600.0)
    assert d.result_window() == (300.0, 600.0)
    d.chk_to_end.setChecked(True)
    assert d.result_window() == (300.0, None)
    d.te_start.setTime(QtCore.QTime(0, 0, 0))
    assert d.result_window() is None              # 0 + até o fim = vídeo inteiro


def test_get_config_per_video_windows(app, monkeypatch):
    dlg = _setup_dialog_plain()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: (["C:/v/a.mp4", "C:/v/b.mp4"], "")))
    dlg._on_add_videos()
    dlg._video_windows["C:/v/a.mp4"] = (300.0, 600.0)   # 5–10 min no vídeo A
    cfg = dlg.get_config("C:/out")
    assert cfg.video_windows == [[300.0, 600.0], None]
    assert cfg.window_for(0) == (300.0, 600.0)
    assert cfg.window_for(1) == (0.0, None)


def test_on_set_window_no_load(app, monkeypatch):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog as mod
    dlg = _setup_dialog_plain()
    _add_video_to_list(dlg, "C:/v/a.mp4")
    dlg.lst_videos.setCurrentRow(0)

    class FakeWin:                                 # diálogo "Definir trecho…" sem carregar
        def __init__(self, *a, **k): pass
        def exec(self): return 1
        def result_window(self): return (300.0, 600.0)
    monkeypatch.setattr(mod, "_VideoWindowDialog", FakeWin)
    dlg._on_set_window()
    assert dlg._video_windows["C:/v/a.mp4"] == (300.0, 600.0)
    assert "5:00" in dlg.lst_videos.item(0).text()   # trecho mostrado no item


def test_on_window_changed_saves_per_video_window(app):
    import types
    dlg = _setup_dialog_plain()
    _add_video_to_list(dlg, "C:/v/a.mp4")
    dlg._video_path = "C:/v/a.mp4"
    dlg._info = types.SimpleNamespace(fps_reported=10.0, frame_count_reported=100)
    dlg.scrubber.setEnabled(True)
    dlg.scrubber.set_range(0, 99)
    dlg.scrubber.set_window(50, 80)                # quadros 50–80 a 10 fps = 5–8 s
    w = dlg._video_windows["C:/v/a.mp4"]
    assert abs(w[0] - 5.0) < 1e-6 and abs(w[1] - 8.0) < 1e-6
    dlg.scrubber.set_window(0, 99)                 # faixa cheia → some (vídeo inteiro)
    assert "C:/v/a.mp4" not in dlg._video_windows


# ─────────────────────────── exclusão de âncora ───────────────────────────
def test_remove_anchor_model():
    t = VideoTarget(label="x")
    t.add_anchor(np.ones(512, np.float32), {"type": "video", "frame_index": 5, "bbox": [0, 0, 5, 5]})
    t.add_anchor(np.full(512, 2.0, np.float32), {"type": "image", "filename": "f.png", "frame_index": None})
    t.remove_anchor(0)                           # remove a âncora de vídeo
    assert t.anchor_embedding_count == 1
    assert t.anchor_policy == "SINGLE"
    assert t.anchor_source_counts() == (0, 1)
    assert t.anchor_frame_index is None          # sem âncora de vídeo → sem quadro anotado


def test_anchors_dialog_shows_thumbnail(app):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog as mod
    t = VideoTarget(label="x")
    t.anchor_embeddings.append(np.ones(512, np.float32))
    t.anchor_sources.append({"type": "video", "frame_index": 5,
                             "thumbnail_bgr": np.full((112, 112, 3), 120, np.uint8)})
    d = mod._AnchorsDialog(t, None)
    d.show()
    app.processEvents()
    with_pix = [l for l in d.findChildren(QtWidgets.QLabel)
                if l.pixmap() is not None and not l.pixmap().isNull()]
    assert len(with_pix) >= 1


def test_anchors_button_disabled_until_anchor(app):
    dlg = _dlg_with_fake()
    row = dlg._active_row
    assert not row.btn_anchors.isEnabled()
    dlg._on_view_clicked(5, 5)
    assert row.btn_anchors.isEnabled()


def test_edit_anchors_clears_row_when_all_removed(app, monkeypatch):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog as mod
    dlg = _dlg_with_fake()
    row = dlg._active_row
    dlg._on_view_clicked(5, 5)
    assert row.has_anchor() and dlg.btn_extract.isEnabled()

    def fake_exec(self):                         # simula remover todas no diálogo
        while self.target.anchor_embedding_count:
            self.target.remove_anchor(0)
        return 0
    monkeypatch.setattr(mod._AnchorsDialog, "exec", fake_exec)
    dlg._on_edit_anchors(row)
    assert not row.has_anchor()
    assert not dlg.btn_extract.isEnabled()
    assert not row.btn_anchors.isEnabled()
