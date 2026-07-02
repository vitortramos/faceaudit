"""Fase 1 — reorganização A→B + filtros de intervalo + dataset na revisão.

Cobre o que muda na Fase 1 (offscreen, sem GPU/ForensicFace):
- Janela A (setup) perde Modo/Dataset e seus getters; ganha barra de detecção.
- Widget `_RangeSlider` (dois cabos): clamp, sinal, alças não cruzam.
- Janela B (revisão) lê o árbitro do manifesto e o trava; alterna Modo;
  miniaturas carregam pose; "Aplicar filtros" respeita intervalos e ignora
  métrica ausente; filtro por vídeo escopa a grade.
- Controller: a config do dataset vem da janela B (Modo PRODUCER vs END_TO_END).
"""

from __future__ import annotations

import json
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtCore, QtWidgets  # noqa: E402

from faceaudit.enums.video_extraction_mode import VideoExtractionMode  # noqa: E402
from faceaudit.models.video.face_provenance import FaceProvenance  # noqa: E402
from faceaudit.services.image_io import imwrite_unicode_safe  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


# ───────────────────────── janela A (setup) ─────────────────────────
def _setup_dialog():
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        VideoDatasetSetupDialog,
    )
    return VideoDatasetSetupDialog(None, use_gpu=False, model="adaface",
                                   datasets_dir=".", output_dir="C:/tmp/runs")


def test_setup_drops_mode_and_dataset(app):
    dlg = _setup_dialog()
    for gone in ("get_mode", "get_dataset_embeddings", "get_dataset_output_dir",
                 "get_dataset_self_contained", "get_dataset_name", "rb_e2e",
                 "chk_ofiq", "txt_name", "_dataset_group", "_emb_checks"):
        assert not hasattr(dlg, gone), f"esperado removido da janela A: {gone}"


def test_setup_keeps_detection_bar_and_load(app):
    dlg = _setup_dialog()
    for kept in ("cmb_model", "spn_det_size", "spn_det_thresh", "btn_load",
                 "btn_extract", "view", "scrubber", "get_config", "get_selected_model"):
        assert hasattr(dlg, kept), f"esperado preservado na janela A: {kept}"


def test_redetect_applies_params_then_detects(app, monkeypatch):
    # "Re-detectar": reconstrói o detector com os params atuais ANTES de detectar
    # o quadro exibido (sem recarregar o vídeo).
    dlg = _setup_dialog()
    assert dlg.btn_detect.text() == "Re-detectar"
    # sem quadro carregado → no-op seguro
    dlg._current_frame = None
    seen = []
    monkeypatch.setattr(dlg, "_ensure_service_current", lambda: seen.append("ensure"))
    monkeypatch.setattr(dlg, "_detect_current", lambda: seen.append("detect"))
    dlg._on_redetect_clicked()
    assert seen == []                       # sem quadro, não faz nada
    dlg._current_frame = np.zeros((8, 8, 3), np.uint8)
    dlg._on_redetect_clicked()
    assert seen == ["ensure", "detect"]     # aplica params, depois detecta


def test_setup_columns_non_collapsible(app):
    # A coluna de controles (direita) não pode sumir quando a prévia cresce.
    dlg = _setup_dialog()
    assert hasattr(dlg, "_columns")
    assert dlg._columns.count() == 2
    assert dlg._columns.childrenCollapsible() is False
    assert dlg._columns.widget(1).minimumWidth() >= 300


def test_setup_video_buttons_consolidated_and_renamed(app):
    # Botões numa linha só; "Remover" vira "Remover selecionado"; a lista de
    # vídeos deixa de ter o teto de altura antigo (cresce para preencher).
    dlg = _setup_dialog()
    assert dlg.btn_load.text() == "Carregar selecionado"
    assert dlg.btn_rm_video.text() == "Remover selecionado"
    assert dlg.lst_videos.maximumHeight() <= 200     # lista contida (não sprawl)


def test_setup_load_button_emphasis_follows_state(app, monkeypatch, tmp_path):
    # Destaque azul segue o próximo passo: Adicionar (lista vazia) → Carregar
    # (há vídeo) → Adicionar (esvaziou). Sempre um único botão aceso.
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        _BTN_PRIMARY_QSS,
    )
    dlg = _setup_dialog()
    # lista vazia → Adicionar destacado, Carregar neutro
    assert dlg.btn_add_videos.styleSheet() == _BTN_PRIMARY_QSS
    assert dlg.btn_load.styleSheet() == ""

    vid = tmp_path / "clip.mp4"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(vid)], "")))
    dlg._on_add_videos()
    # há vídeo → destaque passa para Carregar
    assert dlg.btn_add_videos.styleSheet() == ""
    assert dlg.btn_load.styleSheet() == _BTN_PRIMARY_QSS

    dlg.lst_videos.setCurrentRow(0)
    dlg._on_remove_video()
    # esvaziou → destaque volta para Adicionar
    assert dlg.btn_add_videos.styleSheet() == _BTN_PRIMARY_QSS
    assert dlg.btn_load.styleSheet() == ""


def test_setup_section_titles_emphasized(app):
    # Títulos das seções (QGroupBox) ganham destaque (negrito + acento) via
    # stylesheet do diálogo — separa-os dos rótulos de campo.
    dlg = _setup_dialog()
    qss = dlg.styleSheet()
    assert "QGroupBox::title" in qss and "font-weight:bold" in qss


def test_setup_preview_empty_state(app):
    # Estado-vazio semeado na prévia; carregar um quadro esconde o overlay.
    from PyQt6 import QtGui
    dlg = _setup_dialog()
    assert dlg.view._empty.text()                    # mensagem-guia presente
    assert not dlg.view._empty.isHidden()            # visível quando nada carregado
    dlg.view.set_frame(QtGui.QPixmap(4, 4))
    assert dlg.view._empty.isHidden()


def test_setup_extraction_name_default_and_conflict_gate(app, monkeypatch, tmp_path):
    # Nome default livre no add; colisão de pasta bloqueia (aviso + flag); trocar
    # para um nome livre libera de novo.
    dlg = _setup_dialog()
    dlg.txt_output.setText(str(tmp_path))
    vid = tmp_path / "clip.mp4"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a, **k: ([str(vid)], "")))
    dlg._on_add_videos()
    assert dlg.txt_case.text() == "clip"             # default = stem, livre
    assert dlg._name_conflict is False

    (tmp_path / "clip").mkdir()                       # cria a pasta colidente
    dlg._revalidate_extraction_name()
    assert dlg._name_conflict is True
    assert dlg.lbl_name_warn.text()                  # aviso não-vazio

    dlg.txt_case.setText("clip_novo")                # nome livre → revalida no sinal
    assert dlg._name_conflict is False
    assert dlg.lbl_name_warn.text() == ""


# ──────────── scrubber: arraste estrangulado + libera capture (perf) ────────────
def test_scrub_drag_throttles_frame_read(app):
    # Arrastar a alça não lê o quadro na hora: arma o timer e guarda o índice
    # pendente. Sem isso o seek por movimento travava o arraste em vídeos longos.
    dlg = _setup_dialog()
    dlg._on_scrub_changed(7)
    assert dlg._pending_scrub_idx == 7
    assert dlg._scrub_timer.isActive()
    dlg._scrub_timer.stop()


def test_scrub_release_clears_pending(app):
    # Soltar a alça encerra o estrangulamento (mostra o quadro exato + detecta).
    dlg = _setup_dialog()
    dlg._on_scrub_changed(3)
    dlg._on_scrub_released(3)               # sem vídeo carregado: handlers são no-op seguros
    assert dlg._pending_scrub_idx is None
    assert not dlg._scrub_timer.isActive()


def test_scrub_timeout_reads_latest_pending_frame(app, monkeypatch):
    # A metade "disparo" do estrangulamento: o timeout lê o ÚLTIMO índice
    # pendente (coalescência latest-wins) e chama _show_frame com ele.
    dlg = _setup_dialog()
    seen = []
    monkeypatch.setattr(dlg, "_show_frame", lambda idx: seen.append(idx))
    dlg._on_scrub_changed(7)
    dlg._on_scrub_changed(12)               # arraste avançou antes de o timer disparar
    dlg._on_scrub_timeout()
    assert seen == [12]
    dlg._scrub_timer.stop()


def test_load_video_rebuilds_only_when_detector_changes(app, monkeypatch, tmp_path):
    # O ganho de perf de #1: trocar de vídeo com o MESMO detector NÃO reconstrói
    # a FFService; mudar Resolução/Confiança reconstrói. Exercita _load_video de
    # verdade (não só o helper de assinatura), gravando o `rebuild` por chamada.
    dlg = _setup_dialog()
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"fake")               # probe_video é fake; só precisa existir

    rebuilds = []

    def fake_ensure(*, rebuild=False):
        rebuilds.append(rebuild)
        if dlg._service is None or rebuild:
            dlg._service = types.SimpleNamespace(
                probe_video=lambda p, compute_hash=True: types.SimpleNamespace(
                    frame_count_reported=100, fps_reported=30.0),
                close_preview_cap=lambda: None,
            )
            dlg._service_sig = dlg._detector_signature()

    monkeypatch.setattr(dlg, "_ensure_service", fake_ensure)
    monkeypatch.setattr(dlg, "_show_frame", lambda *a, **k: None)
    monkeypatch.setattr(dlg, "_detect_current", lambda *a, **k: None)

    dlg._load_video(str(vid))              # 1º load → constrói
    dlg._load_video(str(vid))              # mesmo detector → reusa
    assert rebuilds == [True, False]
    dlg.spn_det_thresh.setValue(dlg.spn_det_thresh.value() + 0.1)
    dlg._load_video(str(vid))              # detector mudou → reconstrói
    assert rebuilds[-1] is True


def test_done_releases_preview_capture(app):
    # Fechar a janela (aceitar OU cancelar) libera o capture de prévia — a
    # extração abre os seus próprios; o de prévia não pode segurar o arquivo.
    dlg = _setup_dialog()

    class _FakeSvc:
        def __init__(self):
            self.closed = 0

        def close_preview_cap(self):
            self.closed += 1

    fake = _FakeSvc()
    dlg._service = fake
    dlg.done(0)
    assert fake.closed == 1


def test_detector_signature_tracks_params(app):
    # Teste unitário do helper de assinatura (complementa o teste comportamental
    # acima): a assinatura muda com os parâmetros do detector.
    dlg = _setup_dialog()
    assert dlg._service_sig is None
    sig0 = dlg._detector_signature()
    assert sig0 == dlg._detector_signature()          # estável quando nada muda
    dlg.spn_det_thresh.setValue(dlg.spn_det_thresh.value() + 0.1)
    assert sig0 != dlg._detector_signature()          # limiar mudou → assinatura nova


# ───────────────────────── _RangeSlider ─────────────────────────
def _slider(lo=0.0, hi=1.0):
    from interface.tabs.dataset_manager.video_dataset_dialog.range_slider import _RangeSlider
    return _RangeSlider(None, minimum=lo, maximum=hi)


def test_range_slider_clamps_and_emits(app):
    s = _slider(-90.0, 90.0)
    seen = []
    s.rangeChanged.connect(lambda a, b: seen.append((a, b)))
    s.setValues(-200.0, 200.0)            # fora dos limites → clampa
    assert s.values() == (-90.0, 90.0)
    assert seen and seen[-1] == (-90.0, 90.0)


def test_range_slider_handles_dont_cross(app):
    s = _slider(0.0, 1.0)
    s.setValues(0.8, 0.2)                 # lo>hi → troca, não cruza
    lo, hi = s.values()
    assert lo <= hi and (lo, hi) == (0.2, 0.8)


def test_range_slider_contains_none_passes(app):
    s = _slider(0.0, 1.0)
    s.setValues(0.3, 0.7)
    assert s.contains(None) is True       # ausente passa
    assert s.contains(0.5) is True
    assert s.contains(0.9) is False


def test_range_slider_is_compact(app):
    s = _slider()                          # altura fixa enxuta (~18px)
    assert s.maximumHeight() <= 20


# ───────────────────────── janela B (revisão) ─────────────────────────
def _make_run(tmp_path, faces, *, arbiter="sepaelv2"):
    """faces: lista de (idx, video, yaw, det, fiqa)."""
    run = tmp_path / "camA_run01"
    meta = run / "_meta"
    meta.mkdir(parents=True)
    (meta / "_run_x.json").write_text(json.dumps({
        "run_id": "x", "recognition_model": arbiter,
        "targets": [{"label": "joao", "similarity_threshold": 0.4}]}), encoding="utf-8")
    for idx, video, yaw, det, fiqa in faces:
        img = f"joao_f{idx:07d}.png"
        crop = run / "joao" / img
        imwrite_unicode_safe(crop, np.full((24, 24, 3), 90, np.uint8))
        prov = FaceProvenance(
            run_id="x", run_manifest="_run_x.json", identity_label="joao", image_file=img,
            source_video_filename=video, frame_native_index=idx,
            det_score=det, fiqa_score=fiqa, similarity_to_anchor=0.7,
            pose={"yaw": yaw, "pitch": 3.0, "roll": 0.0},
            auto_decision="accepted", human_review_status="pending")
        (meta / prov.sidecar_filename()).write_text(
            json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    return run


def _review(run, **kw):
    from interface.tabs.dataset_manager.video_dataset_dialog.video_review_dialog import (
        VideoReviewDialog,
    )
    return VideoReviewDialog(None, str(run), operator="perito",
                             dataset_output_dir="D:/datasets", **kw)


def test_review_arbiter_locked_from_manifest(app, tmp_path):
    run = _make_run(tmp_path, [(1, "camA.mp4", 2.0, 0.9, 1.3)])
    dlg = _review(run)
    assert dlg._arbiter == "sepaelv2"
    cb = dlg._emb_checks["sepaelv2"]
    assert cb.isChecked() and not cb.isEnabled()
    assert "sepaelv2" in dlg.get_dataset_embeddings()


def test_review_mode_toggles_dataset_panel(app, tmp_path):
    run = _make_run(tmp_path, [(1, "camA.mp4", 2.0, 0.9, 1.3)])
    dlg = _review(run)
    assert dlg.get_mode() == VideoExtractionMode.PRODUCER
    assert not dlg._dataset_group.isVisibleTo(dlg)
    dlg.rb_e2e.setChecked(True)
    assert dlg.get_mode() == VideoExtractionMode.END_TO_END
    assert dlg._dataset_group.isVisibleTo(dlg)
    assert dlg.get_image_type() == "aligned"          # default
    dlg.rb_img_crop.setChecked(True)
    assert dlg.get_image_type() == "crop"


def test_review_thumbs_carry_pose(app, tmp_path):
    run = _make_run(tmp_path, [(1, "camA.mp4", 12.0, 0.9, 1.3)])
    dlg = _review(run)
    assert dlg._thumbs[0].yaw == 12.0
    assert dlg._thumbs[0].pitch == 3.0


def test_review_apply_filters_respects_ranges_and_none(app, tmp_path):
    run = _make_run(tmp_path, [
        (1, "camA.mp4", 2.0, 0.95, 1.30),    # dentro
        (2, "camA.mp4", 40.0, 0.60, 1.10),   # yaw fora de [-20,20]
        (3, "camB.mp4", -5.0, 0.99, 1.40),   # dentro
    ])
    dlg = _review(run)
    dlg._filters["yaw"].setValues(-20.0, 20.0)
    # zera uma pose para confirmar que None passa no filtro
    dlg._thumbs[2].yaw = None
    dlg._apply_filters_to_selection()
    marked = {t.image_file: t.chk.isChecked() for t in dlg._thumbs}
    assert marked["joao_f0000001.png"] is True
    assert marked["joao_f0000002.png"] is False
    assert marked["joao_f0000003.png"] is True   # yaw None → passa


def test_review_video_filter_scopes_grid(app, tmp_path):
    run = _make_run(tmp_path, [
        (1, "camA.mp4", 2.0, 0.9, 1.3),
        (2, "camB.mp4", -5.0, 0.9, 1.4),
    ])
    dlg = _review(run)
    assert hasattr(dlg, "cmb_video_filter")
    idx = dlg.cmb_video_filter.findData("camB.mp4")
    dlg.cmb_video_filter.setCurrentIndex(idx)
    vis = [t.image_file for t in dlg._visible_thumbs()]
    assert vis == ["joao_f0000002.png"]


def test_review_apply_filters_button_present_and_wired(app, tmp_path):
    run = _make_run(tmp_path, [
        (1, "camA.mp4", 2.0, 0.95, 1.30),
        (2, "camA.mp4", 40.0, 0.60, 1.10),   # yaw fora de [-20,20]
    ])
    dlg = _review(run)
    dlg.show()
    app.processEvents()
    assert hasattr(dlg, "_btn_apply_filters")
    assert dlg._btn_apply_filters.text() == "Aplicar filtros"
    assert dlg._btn_apply_filters.isVisibleTo(dlg)
    dlg._filters["yaw"].setValues(-20.0, 20.0)
    dlg._btn_apply_filters.click()           # o clique deve aplicar os filtros
    marked = {t.image_file: t.chk.isChecked() for t in dlg._thumbs}
    assert marked["joao_f0000001.png"] is True
    assert marked["joao_f0000002.png"] is False


def test_review_has_window_controls_not_maximized(app, tmp_path):
    run = _make_run(tmp_path, [(1, "camA.mp4", 2.0, 0.9, 1.3)])
    dlg = _review(run)
    flags = dlg.windowFlags()
    assert flags & QtCore.Qt.WindowType.WindowMaximizeButtonHint
    assert flags & QtCore.Qt.WindowType.WindowMinimizeButtonHint
    assert not (dlg.windowState() & QtCore.Qt.WindowState.WindowMaximized)


def test_review_thumbs_compact_and_flow(app, tmp_path):
    run = _make_run(tmp_path, [(i, "camA.mp4", 2.0, 0.9, 1.3) for i in range(1, 6)])
    dlg = _review(run)
    assert hasattr(dlg, "_flow")                 # FlowLayout (empacota + reflui)
    assert dlg._thumbs[0].width() <= 130         # célula enxuta (largura ~ foto)


# ───────────────────────── controller ─────────────────────────
class _FakeReview:
    def __init__(self, *a, mode=VideoExtractionMode.END_TO_END, **k):
        self._mode = mode

    def exec(self):
        return 1

    def get_mode(self):
        return self._mode

    def get_dataset_name(self):
        return "ds1"

    def compute_ofiq(self):
        return True

    def compute_serfiq(self):
        return False

    def get_dataset_embeddings(self):
        return ["sepaelv2", "sepaelv4"]

    def get_dataset_output_dir(self):
        return "D:/out"

    def get_dataset_self_contained(self):
        return False

    def get_image_type(self):
        return "crop"


class _FakeWorker:
    def __init__(self, run_dir, **kw):
        self.run_dir = run_dir
        self.kw = kw


class _FakeMgr:
    def __init__(self):
        self.view = None
        self.pipeline = None
        self.config = types.SimpleNamespace(
            paths=types.SimpleNamespace(datasets="D:/datasets", library="D:/lib"))
        self.captured = None

    def _run_worker(self, worker, **kw):
        self.captured = (worker, kw)


def _controller(monkeypatch, mode):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_controller as cm
    monkeypatch.setattr(cm, "VideoReviewDialog",
                        lambda *a, **k: _FakeReview(*a, mode=mode, **k))
    monkeypatch.setattr(cm, "VideoImportWorker", _FakeWorker)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    mgr = _FakeMgr()
    return cm.VideoDatasetController(mgr), mgr


def test_controller_end_to_end_reads_review_config(app, monkeypatch):
    ctrl, mgr = _controller(monkeypatch, VideoExtractionMode.END_TO_END)
    ctrl._open_review_then_finish("D:/runs/run01")
    assert mgr.captured is not None
    worker, _ = mgr.captured
    ing = worker.kw["ingestion_config"]
    assert ing.compute_ofiq is True and ing.compute_serfiq is False
    assert ing.ff_models == ["sepaelv2", "sepaelv4"]
    assert worker.kw["dataset_name"] == "ds1"
    assert worker.kw["dataset_dir"] == "D:/out"
    assert worker.kw["self_contained"] is False
    assert worker.kw["image_kind"] == "crop"


def test_controller_producer_runs_no_worker(app, monkeypatch):
    ctrl, mgr = _controller(monkeypatch, VideoExtractionMode.PRODUCER)
    ctrl._open_review_then_finish("D:/runs/run01")
    assert mgr.captured is None


# ─────────────── cancelar a revisão: descartar vs manter o run ───────────────
class _FakeReviewCancel:
    def __init__(self, *a, **k):
        pass

    def exec(self):
        return 0          # usuário cancela a revisão


def _make_run_dir(tmp_path, name="run01"):
    run = tmp_path / name
    (run / "_meta").mkdir(parents=True)
    (run / "_meta" / "_run_x.json").write_text("{}", encoding="utf-8")
    (run / "alvo").mkdir()
    return run


def _cancel_controller(monkeypatch, *, confirm_delete):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_controller as cm
    monkeypatch.setattr(cm, "VideoReviewDialog", lambda *a, **k: _FakeReviewCancel())
    ans = (QtWidgets.QMessageBox.StandardButton.Yes if confirm_delete
           else QtWidgets.QMessageBox.StandardButton.No)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: ans))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    return cm.VideoDatasetController(_FakeMgr())


def test_cancel_owned_run_confirm_yes_deletes(app, monkeypatch, tmp_path):
    run = _make_run_dir(tmp_path)
    ctrl = _cancel_controller(monkeypatch, confirm_delete=True)
    ctrl._open_review_then_finish(str(run), owns_run=True)
    assert not run.exists()                       # run novo + confirmou → apagado


def test_cancel_owned_run_confirm_no_keeps(app, monkeypatch, tmp_path):
    run = _make_run_dir(tmp_path)
    ctrl = _cancel_controller(monkeypatch, confirm_delete=False)
    ctrl._open_review_then_finish(str(run), owns_run=True)
    assert run.exists()                           # run novo + recusou → mantido


def test_cancel_imported_run_never_deletes(app, monkeypatch, tmp_path):
    run = _make_run_dir(tmp_path)
    # 'question' devolveria Yes, mas em run importado nem é perguntado.
    ctrl = _cancel_controller(monkeypatch, confirm_delete=True)
    ctrl._open_review_then_finish(str(run), owns_run=False)
    assert run.exists()                           # run importado → nunca apaga


# ───── flag owns_run nos pontos de entrada reais (segurança contra perda) ─────
def test_entry_points_pass_correct_owns_run(app, monkeypatch, tmp_path):
    # A propriedade de segurança (run novo = descartável; run importado = nunca
    # apagado) depende do flag certo nos DOIS pontos de entrada. Captura-o.
    ctrl = cm_controller(monkeypatch)
    captured = {}
    monkeypatch.setattr(ctrl, "_open_review_then_finish",
                        lambda run_dir, *, owns_run=False: captured.update(
                            run_dir=run_dir, owns_run=owns_run))

    # 1) extração concluída → run recém-criado → owns_run=True (descartável)
    ctrl._on_extraction_done(types.SimpleNamespace(run_dir="D:/runs/new01", n_accepted_auto=3))
    assert captured["owns_run"] is True

    # 2) importar run pré-existente → owns_run=False (nunca apaga)
    captured.clear()
    run = _make_run_dir(tmp_path, "imported01")     # tem _meta/ → passa o gate de importação
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(run)))
    ctrl.run_import_video_run()
    assert captured["owns_run"] is False


def cm_controller(monkeypatch):
    import interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_controller as cm
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    return cm.VideoDatasetController(_FakeMgr())
