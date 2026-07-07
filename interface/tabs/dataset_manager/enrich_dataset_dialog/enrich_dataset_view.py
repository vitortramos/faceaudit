"""View da janela "Enriquecer Dataset" — redesenho por TIPO de coluna.

Mesmo padrão da Novo Dataset: um grupo "Configurações ForensicFace" (GPU/Batch/
Det size + override de faces alinhadas) e um grupo "Conteúdo a adicionar" em grade
de 2 colunas (Embeddings · Qualidade · Pose · Outros), cada item com um selo de
status (presente / parcial / ausente) preenchido pelo controller. Quando o dataset
é de faces já alinhadas, o que depende da foto original (Pose, Outros, OFIQ, Det
size) desaparece. SER-FIQ foi removido da interface.

Esta classe é só disposição + visibilidade condicional + selos de status. A
análise do dataset, o get_config e o mapeamento para o backend ficam no controller.
"""

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from faceaudit.enums.sepael_model import SepaelModel
from interface.theme import apply_themed_stylesheet, current_token_fn


def _supported_ff_models() -> tuple:
    available = SepaelModel.available_on_disk()
    if available:
        return tuple(m.ff_name for m in available)
    return (SepaelModel.default().ff_name,)


# Selo de status: (token de fundo, token de texto). Resolvidos por tema em runtime.
_STATUS_TOKENS = {
    "present": ("SUCCESS_BG", "SUCCESS"),
    "partial": ("WARNING_BG", "WARNING"),
    "absent": ("DANGER_BG", "DANGER"),
    "neutral": ("SURFACE_ALT", "TEXT_MUTED"),
}


class EnrichDatasetView(QDialog):
    def __init__(self, parent=None, *, default_use_gpu: bool = False,
                 default_batch_size: int = 32):
        super().__init__(parent)
        self.setWindowTitle("Enriquecer Dataset")
        self._default_use_gpu = bool(default_use_gpu)
        self._default_batch_size = int(default_batch_size)
        # Largura folgada para caber os modelos de embedding e os selos por item
        # numa linha só; altura modesta, o resto vira respiro embaixo.
        self.resize(840, 500)

        # Selos de status por item (chave → QLabel) e mapeamento chave → checkboxes
        # (para o auto-check sugerido pelo controller).
        self._status_labels: dict = {}
        self._item_checks: dict = {}
        self._ff_model_checks: dict = {}

        self._init_ui()
        self._wire_conditionals()
        self._update_conditional()

    # ──────────────────────────────────────────────────────────────────
    # Construção
    # ──────────────────────────────────────────────────────────────────
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        self.lbl_info = QLabel("", self)
        self.lbl_info.setWordWrap(True)
        apply_themed_stylesheet(self.lbl_info, lambda t: "color:" + t("TEXT_MUTED") + ";")
        outer.addWidget(self.lbl_info)

        outer.addWidget(self._build_group_ff_settings())
        outer.addWidget(self._build_group_content())

        # Idempotência (PR 18).
        self.chk_force_reprocess = QCheckBox(
            "Reprocessar mesmo as faces já enriquecidas", self)
        self.chk_force_reprocess.setChecked(False)
        self.chk_force_reprocess.setToolTip(
            "Por padrão, cada item processa apenas as faces que ainda NÃO têm "
            "aquele dado. Marque para reprocessar todas (recalcula embeddings).")
        outer.addWidget(self.chk_force_reprocess)

        # Empacota o conteúdo no topo; a sobra vira respiro embaixo (sem isso a
        # label de info estica e cria um vazio no topo).
        outer.addStretch(1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Iniciar processamento")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        outer.addWidget(self.button_box)

    # ── Grupo: Configurações ForensicFace ──
    def _build_group_ff_settings(self) -> QGroupBox:
        grp = QGroupBox("1 · Configurações ForensicFace", self)
        v = QVBoxLayout(grp)
        v.setSpacing(6)
        row = QHBoxLayout()
        self.chk_use_gpu = QCheckBox("Usar GPU", grp)
        self.chk_use_gpu.setChecked(self._default_use_gpu)
        self.chk_use_gpu.setToolTip(
            "Default das Configurações do Sistema; alterar aqui vale só para este "
            "enriquecimento.")
        row.addWidget(self.chk_use_gpu)
        row.addSpacing(16)
        row.addWidget(QLabel("Batch size:", grp))
        self.spn_batch_size = QSpinBox(grp)
        self.spn_batch_size.setRange(1, 512)
        self.spn_batch_size.setValue(self._default_batch_size)
        row.addWidget(self.spn_batch_size)
        row.addSpacing(16)
        self._lbl_det_size = QLabel("Det size:", grp)
        row.addWidget(self._lbl_det_size)
        self._ff_det_size_spin = QSpinBox(grp)
        self._ff_det_size_spin.setRange(160, 1280)
        self._ff_det_size_spin.setSingleStep(32)
        self._ff_det_size_spin.setValue(320)
        self._ff_det_size_spin.setToolTip(
            "Resolução do detector. Só afeta fotos originais — ignorado em faces "
            "já alinhadas.")
        row.addWidget(self._ff_det_size_spin)
        row.addStretch()
        v.addLayout(row)

        # Override de "faces já alinhadas" — pré-preenchido do dataset pelo
        # controller; dirige o "some se alinhada" e corrige metadata errada.
        self._ff_is_aligned_chk = QCheckBox(
            "Imagens são faces já alinhadas 112×112", grp)
        self._ff_is_aligned_chk.setToolTip(
            "Pré-preenchido com o valor atual do dataset. Marque/desmarque só para "
            "corrigir metadata errada — controla se a detecção é pulada e quais "
            "colunas fazem sentido.")
        v.addWidget(self._ff_is_aligned_chk)

        self._det_size_hint = QLabel(
            "Det size não se aplica a faces já alinhadas (detecção é pulada).", grp)
        apply_themed_stylesheet(
            self._det_size_hint,
            lambda t: "color:" + t("TEXT_MUTED") + "; font-size:11px;",
        )
        self._det_size_hint.setVisible(False)
        v.addWidget(self._det_size_hint)
        return grp

    # ── Grupo: Conteúdo a adicionar ──
    def _build_group_content(self) -> QGroupBox:
        grp = QGroupBox("2 · Conteúdo a adicionar", self)
        v = QVBoxLayout(grp)
        v.setSpacing(7)

        cg = QGridLayout()
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(8)
        cg.setColumnStretch(1, 1)

        # Embeddings — um checkbox + selo por modelo.
        cg.addWidget(QLabel("Embeddings:", grp), 0, 0)
        emb_w = QWidget(grp)
        emb_h = QHBoxLayout(emb_w); emb_h.setContentsMargins(0, 0, 0, 0)
        for name in _supported_ff_models():
            member = SepaelModel.from_ff_name(name)
            chk = QCheckBox(member.label_pt if member is not None else name, emb_w)
            self._ff_model_checks[name] = chk
            self._item_checks["emb:" + name] = [chk]
            emb_h.addWidget(chk)
            emb_h.addWidget(self._status_badge("emb:" + name, emb_w))
        emb_h.addStretch()
        cg.addWidget(emb_w, 0, 1)

        # Qualidade — FIQA (sobrevive alinhadas) + OFIQ (só originais).
        cg.addWidget(QLabel("Qualidade:", grp), 1, 0)
        qual_w = QWidget(grp)
        qual_h = QHBoxLayout(qual_w); qual_h.setContentsMargins(0, 0, 0, 0)
        self.chk_fiqa = QCheckBox("FIQA", qual_w)
        self._item_checks["fiqa"] = [self.chk_fiqa]
        qual_h.addWidget(self.chk_fiqa)
        qual_h.addWidget(self._status_badge("fiqa", qual_w))
        qual_h.addSpacing(10)
        self._ofiq_w = QWidget(qual_w)
        ofiq_h = QHBoxLayout(self._ofiq_w); ofiq_h.setContentsMargins(0, 0, 0, 0)
        self.chk_ofiq = QCheckBox("OFIQ (qualidade ISO)", self._ofiq_w)
        self.chk_ofiq.setToolTip("Exige a foto original — indisponível em faces já alinhadas.")
        self._item_checks["ofiq"] = [self.chk_ofiq]
        ofiq_h.addWidget(self.chk_ofiq)
        ofiq_h.addWidget(self._status_badge("ofiq", self._ofiq_w))
        qual_h.addWidget(self._ofiq_w)
        qual_h.addStretch()
        cg.addWidget(qual_w, 1, 1)

        # Pose — yaw/pitch/roll (só originais), status POR item.
        self._pose_lbl = QLabel("Pose:", grp)
        cg.addWidget(self._pose_lbl, 2, 0)
        self._pose_w = QWidget(grp)
        pose_h = QHBoxLayout(self._pose_w); pose_h.setContentsMargins(0, 0, 0, 0)
        self.chk_yaw = QCheckBox("Yaw", self._pose_w)
        self.chk_pitch = QCheckBox("Pitch", self._pose_w)
        self.chk_roll = QCheckBox("Roll", self._pose_w)
        for key, chk in (("attr:yaw", self.chk_yaw), ("attr:pitch", self.chk_pitch),
                         ("attr:roll", self.chk_roll)):
            self._item_checks[key] = [chk]
            pose_h.addWidget(chk)
            pose_h.addWidget(self._status_badge(key, self._pose_w))
        pose_h.addStretch()
        cg.addWidget(self._pose_w, 2, 1)

        # Outros — det conf/ipd/gênero/idade (só originais), status POR item.
        self._outros_lbl = QLabel("Outros:", grp)
        cg.addWidget(self._outros_lbl, 3, 0)
        self._outros_w = QWidget(grp)
        outros_h = QHBoxLayout(self._outros_w); outros_h.setContentsMargins(0, 0, 0, 0)
        self.chk_det_conf = QCheckBox("Det Conf", self._outros_w)
        self.chk_ipd = QCheckBox("IPD", self._outros_w)
        self.chk_gender = QCheckBox("Gênero", self._outros_w)
        self.chk_age = QCheckBox("Idade", self._outros_w)
        for key, chk in (("attr:det_score", self.chk_det_conf), ("attr:ipd", self.chk_ipd),
                         ("attr:gender", self.chk_gender), ("attr:age", self.chk_age)):
            self._item_checks[key] = [chk]
            outros_h.addWidget(chk)
            outros_h.addWidget(self._status_badge(key, self._outros_w))
        outros_h.addStretch()
        cg.addWidget(self._outros_w, 3, 1)

        v.addLayout(cg)
        v.addWidget(self._hline())
        hint = QLabel(
            "Adicionar pose/gênero/idade a faces que já têm embedding roda só o "
            "detector (não recalcula o embedding). FIQA exige o passo de "
            "reconhecimento.", grp)
        hint.setWordWrap(True)
        apply_themed_stylesheet(hint, lambda t: "color:" + t("TEXT_MUTED") + "; font-size:11px;")
        v.addWidget(hint)
        return grp

    def _status_badge(self, key: str, parent) -> QLabel:
        lbl = QLabel("", parent)
        lbl.setStyleSheet("font-size:11px;")
        self._status_labels[key] = lbl
        return lbl

    def _hline(self) -> QFrame:
        ln = QFrame(self)
        ln.setFrameShape(QFrame.Shape.HLine)
        apply_themed_stylesheet(ln, lambda t: "color:" + t("BORDER") + "; max-height:1px;")
        return ln

    # ──────────────────────────────────────────────────────────────────
    # Condicional (some se alinhada)
    # ──────────────────────────────────────────────────────────────────
    def _wire_conditionals(self):
        self._ff_is_aligned_chk.toggled.connect(lambda _c: self._update_conditional())

    def _update_conditional(self):
        aligned = self._ff_is_aligned_chk.isChecked()
        # Pose, Outros, OFIQ e Det size dependem da foto original.
        self._pose_lbl.setVisible(not aligned)
        self._pose_w.setVisible(not aligned)
        self._outros_lbl.setVisible(not aligned)
        self._outros_w.setVisible(not aligned)
        self._ofiq_w.setVisible(not aligned)
        self._lbl_det_size.setVisible(not aligned)
        self._ff_det_size_spin.setVisible(not aligned)
        self._det_size_hint.setVisible(aligned)

    # ──────────────────────────────────────────────────────────────────
    # API consumida pelo controller
    # ──────────────────────────────────────────────────────────────────
    def set_info(self, text: str):
        self.lbl_info.setText(text)

    def set_item_status(self, key: str, kind: str, text: str):
        lbl = self._status_labels.get(key)
        if lbl is None:
            return
        bg_token, fg_token = _STATUS_TOKENS.get(kind, _STATUS_TOKENS["neutral"])
        t = current_token_fn()
        bg, fg = t(bg_token), t(fg_token)
        lbl.setText(text)
        lbl.setStyleSheet(
            f"background:{bg}; color:{fg}; border-radius:8px; padding:1px 7px; font-size:11px;")

    def set_item_checked(self, key: str, checked: bool):
        for chk in self._item_checks.get(key, []):
            if chk.isEnabled():
                chk.setChecked(checked)

    def set_ff_is_aligned(self, value: bool):
        self._ff_is_aligned_chk.setChecked(bool(value))
        self._update_conditional()

    def set_stage_availability(self, ff_available: bool, ofiq_available: bool):
        ff_tip = ("ForensicFace indisponível: nenhum modelo em "
                  "~/.forensicface/models/ ou o serviço falhou ao carregar.")
        ofiq_tip = "OFIQ indisponível: executável ou pasta de dados não encontrados."
        if not ff_available:
            ff_dependent = list(self._ff_model_checks.values()) + [
                self.chk_fiqa, self.chk_yaw, self.chk_pitch, self.chk_roll,
                self.chk_det_conf, self.chk_ipd, self.chk_gender, self.chk_age]
            for chk in ff_dependent:
                chk.setChecked(False); chk.setEnabled(False); chk.setToolTip(ff_tip)
        if not ofiq_available:
            self.chk_ofiq.setChecked(False); self.chk_ofiq.setEnabled(False)
            self.chk_ofiq.setToolTip(ofiq_tip)
        self._update_conditional()

    # ── Getters ──
    def get_ff_models(self) -> list:
        return [n for n, c in self._ff_model_checks.items() if c.isChecked()]

    def any_model_checked(self) -> bool:
        return any(c.isChecked() for c in self._ff_model_checks.values())

    def get_ff_det_size(self) -> int:
        return int(self._ff_det_size_spin.value())

    def get_use_gpu(self) -> bool:
        return bool(self.chk_use_gpu.isChecked())

    def get_batch_size(self) -> int:
        return int(self.spn_batch_size.value())

    def get_content_flags(self) -> dict:
        return {
            "fiqa": self.chk_fiqa.isChecked(),
            "yaw": self.chk_yaw.isChecked(),
            "pitch": self.chk_pitch.isChecked(),
            "roll": self.chk_roll.isChecked(),
            "gender": self.chk_gender.isChecked(),
            "age": self.chk_age.isChecked(),
            "det_score": self.chk_det_conf.isChecked(),
            "ipd": self.chk_ipd.isChecked(),
        }

    def get_ff_is_aligned(self) -> bool:
        return bool(self._ff_is_aligned_chk.isChecked())

    def is_input_aligned(self) -> bool:
        return bool(self._ff_is_aligned_chk.isChecked())
