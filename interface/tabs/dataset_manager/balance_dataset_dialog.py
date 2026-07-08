"""Diálogo de balanceamento por identidade (PR balance-toolbar).

Single-file (controller + view juntos, padrão do app:
schema_report_dialog.py / modify_paths_dialog.py /
apply_image_pipeline_dialog.py).

Replica as opções do `DatasetBalancingInstruction` do Strategy Manager:
modo (CAP_ONLY / FLOOR_AND_CAP / AUTO_MIN_BALANCE), target_n, critério
(RANDOM / BY_QUALITY) + métrica.

Após `exec() == Accepted`, chame os getters:
  - `mode()` → DatasetBalancingMode
  - `target_n()` → int
  - `selection()` → DatasetBalancingSelection
  - `random_seed()` → int
  - `quality_metric()` → QualityScoreType
  - `open_new_after()` → bool
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import (
    DatasetBalancingSelection,
)
from faceaudit.enums.quality_score_type import QualityScoreType


_MODE_OPTIONS = [
    (DatasetBalancingMode.CAP_ONLY, "Cap apenas",
     "Limita cada identidade a no máximo <code>target_n</code> "
     "amostras. Identidades com menos amostras permanecem "
     "intocadas (resultado ainda pode ficar desbalanceado, mas "
     "preserva todas as identidades)."),
    (DatasetBalancingMode.FLOOR_AND_CAP, "Floor + Cap",
     "Limita cada identidade a <code>target_n</code> E descarta "
     "identidades inteiras com menos de <code>target_n</code> "
     "amostras. Garante balanceamento perfeito ao custo de perder "
     "IDs subrrepresentadas."),
    (DatasetBalancingMode.AUTO_MIN_BALANCE, "Auto-min (todas iguais ao mínimo)",
     "Ignora target_n; usa N = mínimo observado entre as identidades. "
     "Todas as identidades ficam com a mesma contagem. Em datasets "
     "heterogêneos pode ser lossy demais — uma ID com poucas amostras "
     "força todo mundo a perder dados."),
]

_SELECTION_OPTIONS = [
    (DatasetBalancingSelection.RANDOM, "Aleatório (reprodutível)",
     "Sorteio uniforme com seed pra reprodutibilidade. Mais barato e "
     "neutro — não usa métrica de qualidade."),
    (DatasetBalancingSelection.BY_QUALITY, "Por qualidade (top-N)",
     "Ordena por <code>quality_metric</code> (maior = melhor) e mantém "
     "as N melhores por identidade. Empates desfeitos via hash "
     "determinístico do face_hash/file_name."),
]

_QUALITY_OPTIONS = [
    (QualityScoreType.FIQA, "FIQA (ForensicFace)"),
    (QualityScoreType.OFIQ, "OFIQ (ISO unified)"),
    (QualityScoreType.SERFIQ, "SER-FIQ (robustez)"),
    (QualityScoreType.EMBEDDING_NORM, "Norma do embedding (AdaFace)"),
]


class BalanceDatasetDialog(QtWidgets.QDialog):
    """Modal pra coletar parâmetros de balanceamento."""

    def __init__(
        self,
        dataset_name: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._dataset_name = dataset_name
        self.setWindowTitle(f"Balancear Dataset — {dataset_name}")
        self.setModal(True)
        self.resize(600, 540)
        self._build_ui()
        self._on_mode_changed(0)
        self._on_selection_changed(0)

    # ── UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Balanceamento por identidade</b> — gera um novo "
            "dataset com contagem controlada de amostras por ID. "
            "Original não é mutado."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Modo ──
        mode_box = QtWidgets.QGroupBox("Modo")
        mode_layout = QtWidgets.QVBoxLayout(mode_box)
        self.cmb_mode = QtWidgets.QComboBox()
        for mode, label, _ in _MODE_OPTIONS:
            self.cmb_mode.addItem(label, userData=mode)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.cmb_mode)
        self.lbl_mode_help = QtWidgets.QLabel("")
        self.lbl_mode_help.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_mode_help,
            lambda t: "color: " + t("TEXT") + "; font-size: 11px; padding: 4px;",
        )
        mode_layout.addWidget(self.lbl_mode_help)
        layout.addWidget(mode_box)

        # ── target_n ──
        self.row_target = QtWidgets.QWidget()
        target_layout = QtWidgets.QHBoxLayout(self.row_target)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(QtWidgets.QLabel("N alvo por identidade:"))
        self.spn_target = QtWidgets.QSpinBox()
        self.spn_target.setRange(1, 100000)
        self.spn_target.setValue(10)
        target_layout.addWidget(self.spn_target)
        target_layout.addStretch()
        layout.addWidget(self.row_target)

        # ── Critério de seleção ──
        sel_box = QtWidgets.QGroupBox("Critério de seleção")
        sel_layout = QtWidgets.QVBoxLayout(sel_box)
        self.cmb_selection = QtWidgets.QComboBox()
        for sel, label, _ in _SELECTION_OPTIONS:
            self.cmb_selection.addItem(label, userData=sel)
        self.cmb_selection.currentIndexChanged.connect(self._on_selection_changed)
        sel_layout.addWidget(self.cmb_selection)
        self.lbl_sel_help = QtWidgets.QLabel("")
        self.lbl_sel_help.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_sel_help,
            lambda t: "color: " + t("TEXT") + "; font-size: 11px; padding: 4px;",
        )
        sel_layout.addWidget(self.lbl_sel_help)

        # Subcampo: seed (random) ou métrica (quality)
        sub_form = QtWidgets.QFormLayout()
        self.spn_seed = QtWidgets.QSpinBox()
        self.spn_seed.setRange(0, 999_999_999)
        self.spn_seed.setValue(42)
        self.row_seed = QtWidgets.QWidget()
        seed_layout = QtWidgets.QHBoxLayout(self.row_seed)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(self.spn_seed)
        seed_layout.addStretch()
        sub_form.addRow("Seed:", self.row_seed)

        self.cmb_quality = QtWidgets.QComboBox()
        for metric, label in _QUALITY_OPTIONS:
            self.cmb_quality.addItem(label, userData=metric)
        sub_form.addRow("Métrica:", self.cmb_quality)
        sel_layout.addLayout(sub_form)
        layout.addWidget(sel_box)

        # ── Após balancear ──
        after_box = QtWidgets.QGroupBox("Após balancear")
        after_layout = QtWidgets.QVBoxLayout(after_box)
        self.rbn_keep = QtWidgets.QRadioButton(
            "Manter dataset atual aberto (balancear em memória; "
            "marca como dirty pra Salvar Como)"
        )
        self.rbn_replace = QtWidgets.QRadioButton(
            "Substituir o dataset corrente pelo balanceado"
        )
        self.rbn_keep.setChecked(True)
        after_layout.addWidget(self.rbn_keep)
        after_layout.addWidget(self.rbn_replace)
        layout.addWidget(after_box)

        layout.addStretch()

        # ── Botões ──
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("⚖ Balancear")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    # ── Slots ──────────────────────────────────────────────────────

    def _on_mode_changed(self, _idx: int):
        mode = self.cmb_mode.currentData()
        for m, _, h in _MODE_OPTIONS:
            if m == mode:
                self.lbl_mode_help.setText(h)
                break
        # target_n é irrelevante em AUTO_MIN — esconde.
        is_auto = mode == DatasetBalancingMode.AUTO_MIN_BALANCE
        self.row_target.setVisible(not is_auto)

    def _on_selection_changed(self, _idx: int):
        sel = self.cmb_selection.currentData()
        for s, _, h in _SELECTION_OPTIONS:
            if s == sel:
                self.lbl_sel_help.setText(h)
                break
        is_random = sel == DatasetBalancingSelection.RANDOM
        self.row_seed.setVisible(is_random)
        self.cmb_quality.setVisible(not is_random)
        # Esconde a label "Métrica:" também — busca o widget do form.

    # ── Accessors ──────────────────────────────────────────────────

    def mode(self) -> DatasetBalancingMode:
        return self.cmb_mode.currentData()

    def target_n(self) -> int:
        return int(self.spn_target.value())

    def selection(self) -> DatasetBalancingSelection:
        return self.cmb_selection.currentData()

    def random_seed(self) -> int:
        return int(self.spn_seed.value())

    def quality_metric(self) -> QualityScoreType:
        return self.cmb_quality.currentData()

    def replace_current(self) -> bool:
        return self.rbn_replace.isChecked()
