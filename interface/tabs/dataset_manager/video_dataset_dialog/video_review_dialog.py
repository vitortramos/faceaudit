"""Grade de revisão/cull de um run de extração de vídeo (Fase 5).

Mostra as faces aceitas automaticamente como thumbnails; o perito
marca/desmarca cada uma. "Concluir revisão" grava as decisões humanas e
fecha o review (review_status=completed). "Pular revisão" é override
explícito e auditado (aceita todas + marca skipped no manifesto).
"""

from __future__ import annotations

import math
from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.services.image_io import imread_unicode_safe
from faceaudit.services.video_review_service import VideoReviewService, ReviewFace
from .video_dataset_setup_dialog import _bgr_to_pixmap

_THUMB = 110


class _Thumb(QtWidgets.QFrame):
    def __init__(self, rf: ReviewFace, parent=None):
        super().__init__(parent)
        self.image_file = rf.image_file
        self.det_score = rf.det_score
        self.fiqa_score = rf.fiqa_score
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        lbl_img = QtWidgets.QLabel(self)
        lbl_img.setFixedSize(_THUMB, _THUMB)
        lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet("background:#222;")
        bgr = imread_unicode_safe(rf.crop_path) if rf.crop_path else None
        if bgr is not None:
            pix = _bgr_to_pixmap(bgr).scaled(
                _THUMB, _THUMB, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation)
            lbl_img.setPixmap(pix)
        else:
            lbl_img.setText("sem\nimagem")
        v.addWidget(lbl_img)

        sim = rf.similarity_to_anchor
        sim_txt = f"{sim:.2f}" if sim is not None else "—"
        self.chk = QtWidgets.QCheckBox(f"f{rf.frame_index}  sim={sim_txt}", self)
        self.chk.setChecked(rf.human_review_status != "rejected")
        v.addWidget(self.chk)

        # Qualidade (triagem): ajuda a decidir aceitar/rejeitar. FIQA do
        # CR-FIQA NÃO é 0–1 (tipicamente ~1.0–1.4) — use comparativamente.
        det_txt = f"{rf.det_score:.2f}" if rf.det_score is not None else "—"
        fiqa_txt = f"{rf.fiqa_score:.2f}" if rf.fiqa_score is not None else "—"
        lbl_q = QtWidgets.QLabel(f"det {det_txt} · fiqa {fiqa_txt}", self)
        lbl_q.setToolTip("det_score (confiança da detecção) · fiqa_score (qualidade — CR-FIQA, não 0–1)")
        lbl_q.setStyleSheet("color:#555; font-size:10px;")
        v.addWidget(lbl_q)

        if rf.review_reasons:
            hint = QtWidgets.QLabel("⚠ " + ", ".join(rf.review_reasons), self)
            hint.setStyleSheet("color:#e65100; font-size:10px;")
            v.addWidget(hint)

    def is_accepted(self) -> bool:
        return self.chk.isChecked()


class VideoReviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, run_dir, *, operator: str = "perito"):
        super().__init__(parent)
        self.setWindowTitle("Revisão das faces extraídas")
        self.resize(820, 640)
        self._run_dir = run_dir
        self._operator = operator
        self._svc = VideoReviewService()
        self._outcome: Optional[str] = None  # completed | skipped | None(cancelado)
        self._thumbs: List[_Thumb] = []
        self._build_ui()

    def outcome(self) -> Optional[str]:
        return self._outcome

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        faces = [f for f in self._svc.list_faces(self._run_dir)
                 if f.human_review_status != "rejected"]
        root.addWidget(QtWidgets.QLabel(
            f"<b>{len(faces)} face(s)</b> extraídas. Desmarque as que NÃO são o alvo "
            "(rejeitados são preservados em <code>_rejected/</code>).", self))

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        grid_host = QtWidgets.QWidget(scroll)
        grid = QtWidgets.QGridLayout(grid_host)
        cols = 6
        for i, rf in enumerate(faces):
            t = _Thumb(rf, grid_host)
            self._thumbs.append(t)
            grid.addWidget(t, i // cols, i % cols)
        scroll.setWidget(grid_host)
        root.addWidget(scroll, 1)

        # ── Seleção automática por qualidade (sliders de det/fiqa) ──
        dets = [f.det_score for f in faces if f.det_score is not None]
        fiqas = [f.fiqa_score for f in faces if f.fiqa_score is not None]
        self.sld_det = self._make_quality_slider(dets, (0.0, 1.0))
        self.sld_fiqa = self._make_quality_slider(fiqas, (0.0, 2.0))
        self.lbl_det_val = QtWidgets.QLabel("", self)
        self.lbl_fiqa_val = QtWidgets.QLabel("", self)
        self.lbl_qcount = QtWidgets.QLabel("", self)
        self.sld_det.valueChanged.connect(self._update_quality_labels)
        self.sld_fiqa.valueChanged.connect(self._update_quality_labels)
        grp = QtWidgets.QGroupBox("Seleção automática por qualidade", self)
        g = QtWidgets.QGridLayout(grp)
        g.addWidget(QtWidgets.QLabel("Mín. det_score:", self), 0, 0)
        g.addWidget(self.sld_det, 0, 1)
        g.addWidget(self.lbl_det_val, 0, 2)
        g.addWidget(QtWidgets.QLabel("Mín. fiqa_score:", self), 1, 0)
        g.addWidget(self.sld_fiqa, 1, 1)
        g.addWidget(self.lbl_fiqa_val, 1, 2)
        btn_apply = QtWidgets.QPushButton("Selecionar faces ≥ limiares", self)
        btn_apply.clicked.connect(self._apply_quality_selection)
        g.addWidget(btn_apply, 2, 0)
        g.addWidget(self.lbl_qcount, 2, 1, 1, 2)
        root.addWidget(grp)
        self._update_quality_labels()

        sel_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Marcar todos", self)
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QtWidgets.QPushButton("Desmarcar todos", self)
        btn_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        root.addLayout(sel_row)

        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Cancelar", self)
        btn_cancel.clicked.connect(self.reject)
        btn_skip = QtWidgets.QPushButton("Pular revisão", self)
        btn_skip.clicked.connect(self._on_skip)
        btn_finish = QtWidgets.QPushButton("Concluir revisão ✔", self)
        btn_finish.setDefault(True)
        btn_finish.clicked.connect(self._on_finish)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_skip)
        btn_row.addWidget(btn_finish)
        root.addLayout(btn_row)

    def _set_all(self, checked: bool):
        for t in self._thumbs:
            t.chk.setChecked(checked)

    def _on_finish(self):
        for t in self._thumbs:
            status = "accepted" if t.is_accepted() else "rejected"
            try:
                self._svc.set_face_decision(self._run_dir, t.image_file,
                                            status=status, operator=self._operator)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Revisão", f"Falha em {t.image_file}: {e}")
        self._svc.commit_review(self._run_dir, operator=self._operator, skipped=False)
        self._outcome = "completed"
        self.accept()

    def _on_skip(self):
        res = QtWidgets.QMessageBox.warning(
            self, "Pular revisão",
            "Você está criando um dataset final SEM revisão humana dos candidatos.\n"
            "Use apenas em teste/desenvolvimento ou com justificativa.\n\nContinuar?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if res != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        # skip = aceita todas (sem inspeção) + marca skipped no manifesto.
        for t in self._thumbs:
            try:
                self._svc.set_face_decision(self._run_dir, t.image_file,
                                            status="accepted", operator=self._operator,
                                            reason="review_skipped")
            except Exception:
                pass
        self._svc.commit_review(self._run_dir, operator=self._operator, skipped=True)
        self._outcome = "skipped"
        self.accept()

    # ── seleção automática por qualidade ──
    def _make_quality_slider(self, vals, fallback):
        """Slider [min, max] dos dados observados (int = valor×100)."""
        lo_f = min(vals) if vals else fallback[0]
        hi_f = max(vals) if vals else fallback[1]
        lo, hi = int(math.floor(lo_f * 100)), int(math.ceil(hi_f * 100))
        if hi <= lo:
            hi = lo + 1
        s = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        s.setRange(lo, hi)
        s.setValue(lo)  # default = mínimo observado → todos passam
        return s

    def _min_det(self) -> float:
        return self.sld_det.value() / 100.0

    def _min_fiqa(self) -> float:
        return self.sld_fiqa.value() / 100.0

    def _passes(self, t: "_Thumb") -> bool:
        d, f = t.det_score, t.fiqa_score
        return (d is None or d >= self._min_det()) and (f is None or f >= self._min_fiqa())

    def _update_quality_labels(self):
        self.lbl_det_val.setText(f"≥ {self._min_det():.2f}")
        self.lbl_fiqa_val.setText(f"≥ {self._min_fiqa():.2f}")
        n = sum(1 for t in self._thumbs if self._passes(t))
        self.lbl_qcount.setText(f"{n} de {len(self._thumbs)} face(s) passam nos limiares")

    def _apply_quality_selection(self):
        """Marca quem passa nos dois limiares, desmarca o resto."""
        for t in self._thumbs:
            t.chk.setChecked(self._passes(t))
        self._update_quality_labels()
