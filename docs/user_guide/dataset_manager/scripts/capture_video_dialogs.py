"""Captura offscreen dos diálogos de vídeo + refresh do Novo/Enriquecer.

Ao contrário de `capture_dataset_manager.py` (que depende de um display X
via Xvfb), este roda **headless com `QT_QPA_PLATFORM=offscreen`** — funciona
no Windows sem display. Gera:

    assets/44_video_setup.png     — Janela A (configurar extração)
    assets/45_video_review.png    — Janela B (revisão + dataset)
    assets/30_novo_dataset.png    — diálogo Novo dataset (redesenhado)
    assets/32_enriquecer.png      — diálogo Enriquecer (redesenhado)

Uso:
    QT_QPA_PLATFORM=offscreen python docs/user_guide/dataset_manager/scripts/capture_video_dialogs.py

Não precisa de GPU, modelos, parquet de teste nem forensicface stub (usa o
forensicface real só para importar; nada de inferência roda).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_THIS = Path(__file__).resolve()
# scripts -> dataset_manager -> user_guide -> docs -> <repo>
_REPO = _THIS.parents[4]
sys.path.insert(0, str(_REPO))
os.chdir(str(_REPO))

import numpy as np  # noqa: E402
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

OUT = _THIS.parents[1] / "assets"
OUT.mkdir(exist_ok=True)


def _light_palette(app: QtWidgets.QApplication) -> None:
    """Paleta clara — os stylesheets do app assumem fundo claro."""
    pal = QtGui.QPalette()
    white = QtGui.QColor("#ffffff")
    near = QtGui.QColor("#f4f5f7")
    text = QtGui.QColor("#1a1a1a")
    for role in (QtGui.QPalette.ColorRole.Window, QtGui.QPalette.ColorRole.Base,
                 QtGui.QPalette.ColorRole.Button, QtGui.QPalette.ColorRole.AlternateBase,
                 QtGui.QPalette.ColorRole.ToolTipBase):
        pal.setColor(role, white if role != QtGui.QPalette.ColorRole.Window else near)
    for role in (QtGui.QPalette.ColorRole.WindowText, QtGui.QPalette.ColorRole.Text,
                 QtGui.QPalette.ColorRole.ButtonText, QtGui.QPalette.ColorRole.ToolTipText):
        pal.setColor(role, text)
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#1a73e8"))
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, white)
    app.setPalette(pal)


def _grab(widget: QtWidgets.QWidget, name: str, size=None) -> None:
    if size is not None:
        widget.resize(*size)
    widget.show()
    widget.raise_()
    for _ in range(12):
        QtWidgets.QApplication.processEvents()
    pix = widget.grab()
    path = OUT / f"{name}.png"
    pix.save(str(path))
    print(f"  OK  {name}.png  {pix.width()}x{pix.height()}")
    widget.close()


# ─────────────────────────── fixtures ───────────────────────────

def _write_png(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = QtGui.QPixmap(100, 100)
    pm.fill(QtGui.QColor(*rgb))
    pm.save(str(path))


def _build_fake_video_run(root: Path) -> Path:
    """Materializa uma pasta de run sintética que a Janela B consegue ler
    (manifesto + sidecars por face + miniaturas)."""
    meta = root / "_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "_run_abc1234567.json").write_text(json.dumps({
        "schema": "faceaudit.video_extraction.run_manifest",
        "schema_version": 2,
        "run_id": "abc1234567",
        "recognition_model": "sepaelv2",
        "review_status": "not_started",
        "video_paths": ["cam1.mp4", "cam2.mp4"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    spec = [("investigado", "cam1.mp4", 7, (70, 110, 170)),
            ("investigado_2", "cam2.mp4", 5, (150, 90, 90))]
    for label, video, n, rgb in spec:
        for i in range(n):
            frame = 40 + i * 17
            stem = f"{label}_v0{spec.index((label, video, n, rgb))}_f{frame:07d}"
            fname = f"{stem}.png"
            _write_png(root / label / fname, rgb)
            sidecar = {
                "identity": {"label": label},
                "artifact": {"image_file": fname},
                "reidentification": {"similarity_to_anchor": round(0.62 + 0.03 * i, 2)},
                "quality_gates": {"det_score": round(0.80 + 0.02 * i, 2),
                                  "fiqa_score": round(1.10 + 0.04 * i, 2)},
                "frame": {"native_index": frame},
                "detection": {"pose": {"yaw": -18.0 + 5.0 * i, "pitch": 3.0 + i}},
                "source_video": {"filename": video},
                "human_review": {"status": "pending"},
                "auto_decision": "accepted",
                "review_reasons": [] if i % 4 else ["low_fiqa"],
            }
            (meta / f"{stem}.json").write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def _fake_enrich_dataset():
    """Dataset sintético com cobertura MISTA — para os selos mostrarem
    presente / parcial / ausente."""
    from types import SimpleNamespace
    from faceaudit.models.face_info import FaceInfo
    from faceaudit.models.ff_info import FFInfo
    faces = []
    for i in range(8):
        kw = dict(embedding_sepaelv2=np.ones(512, np.float32),  # V2 presente; V4+ ausente
                  fiqa_score=1.2, det_score=0.9, ipd=64.0)
        if i < 5:  # pose parcial (5/8)
            kw.update(yaw=-10.0 + i, pitch=2.0, roll=0.5)
        if i < 4:  # gênero/idade parcial (4/8)
            kw.update(gender="M", age=32.0)
        faces.append(FaceInfo(image_path=f"f{i}.png", identity=str(i % 3),
                              ff_info=FFInfo(**kw)))
    return SimpleNamespace(faces=faces, name="investigado_cam1", is_aligned=False)


# ─────────────────────────── captures ───────────────────────────

def cap_video_setup() -> None:
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        VideoDatasetSetupDialog,
    )
    dlg = VideoDatasetSetupDialog(None, use_gpu=False, model="sepaelv2",
                                  datasets_dir="C:/faceaudit/datasets",
                                  output_dir="C:/casos/investigado/extracoes")
    dlg._add_target(activate=True)  # um segundo alvo para a lista de Alvos
    _grab(dlg, "44_video_setup", size=(1180, 780))


def cap_video_review() -> None:
    from interface.tabs.dataset_manager.video_dataset_dialog.video_review_dialog import (
        VideoReviewDialog,
    )
    run = _build_fake_video_run(Path(tempfile.mkdtemp()) / "investigado_cam1")
    dlg = VideoReviewDialog(None, str(run), operator="perito",
                            dataset_output_dir="C:/faceaudit/datasets")
    if hasattr(dlg, "rb_e2e"):
        dlg.rb_e2e.setChecked(True)  # revela o grupo "Dataset"
    _grab(dlg, "45_video_review", size=(1400, 860))


def cap_novo_dataset() -> None:
    from interface.tabs.dataset_manager.new_dataset_dialog.new_dataset_controller import (
        NewDatasetController,
    )
    tdir = Path(tempfile.mkdtemp())
    c = NewDatasetController(None, templates_path=tdir, ff_available=True,
                             ofiq_available=True, default_dataset_dir="C:/faceaudit/datasets",
                             default_use_gpu=True, default_batch_size=32)
    c.view.txt_path.setText("C:/casos/investigado/fotos")
    c.view.txt_name.setText("investigado")
    _grab(c.view, "30_novo_dataset", size=(620, 720))


def cap_enriquecer() -> None:
    from interface.tabs.dataset_manager.enrich_dataset_dialog.enrich_dataset_controller import (
        EnrichDatasetController,
    )
    c = EnrichDatasetController(_fake_enrich_dataset(), parent=None, ff_available=True,
                                ofiq_available=True, default_use_gpu=True,
                                default_batch_size=32)
    _grab(c.view, "32_enriquecer", size=(860, 560))


def _load_fonts() -> None:
    """O QPA `offscreen` sobe com 0 fontes — carrega TTFs do Windows na base de
    fontes do Qt para o texto não virar tofu (□□□)."""
    for f in ("segoeui.ttf", "segoeuib.ttf", "seguisym.ttf", "seguiemj.ttf", "arial.ttf"):
        p = f"C:/Windows/Fonts/{f}"
        if os.path.exists(p):
            QtGui.QFontDatabase.addApplicationFont(p)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    _load_fonts()
    app.setFont(QtGui.QFont("Segoe UI", 9))
    _light_palette(app)
    print(f"Salvando em {OUT}")
    ok = 0
    for fn, name in [(cap_video_setup, "44_video_setup"),
                     (cap_video_review, "45_video_review"),
                     (cap_novo_dataset, "30_novo_dataset"),
                     (cap_enriquecer, "32_enriquecer")]:
        try:
            fn()
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name}: {e!r}")
            traceback.print_exc()
    print(f"Concluído: {ok}/4")
    return 0 if ok == 4 else 1


if __name__ == "__main__":
    sys.exit(main())
