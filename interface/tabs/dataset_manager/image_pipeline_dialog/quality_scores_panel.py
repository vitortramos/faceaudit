"""QualityScoresPanel — tabela com FIQA + ‖emb‖ + cosine (PR I3).

Embaixo do preview no editor. 3 rows × 4 cols:
  - Métrica | Original | Processada | Δ

Estados:
  - `clear()` — vazio (—) em todas as células
  - `show_loading()` — "Calculando…" em todas as células
  - `show_results(QualityScoresResult)` — preenche com valores;
    cores sinalizam degradação (FIQA cair → vermelho)
  - `show_error(msg)` — mensagem na status line abaixo da tabela

Decisão #4 do plano: layout compacto, embaixo do preview.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet, current_token_fn
from faceaudit.models.image_processing.quality_scores_result import (
    QualityScoresResult,
)


_DASH = "—"
_LOADING = "…"


class QualityScoresPanel(QtWidgets.QWidget):
    """Tabela compacta de scores. Não emite sinais — só recebe."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("<b>Scores de qualidade</b>")
        title.setStyleSheet("font-size: 11px;")
        layout.addWidget(title)

        # Tabela 3 × 4
        self.table = QtWidgets.QTableWidget(3, 4)
        self.table.setHorizontalHeaderLabels([
            "Métrica", "Original", "Processada", "Δ",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection,
        )
        self.table.setShowGrid(True)
        self.table.setAlternatingRowColors(True)
        # Altura: 3 rows + header + buffer.
        # `horizontalHeader().height()` retorna a altura de layout — no
        # construtor (widget ainda não mostrado) isso era um valor mínimo
        # e o `setFixedHeight` resultante cabia só 2 linhas, forçando
        # scroll vertical pra ver a terceira (Cosine sim.). `sizeHint()`
        # é determinístico fora do paint cycle.
        row_h = 26
        self.table.verticalHeader().setDefaultSectionSize(row_h)
        header_h = self.table.horizontalHeader().sizeHint().height()
        # +2 px por row pra acomodar grid lines + 8 px de margem
        # geral (frame border + arredondamento de pixel).
        self.table.setFixedHeight(header_h + row_h * 3 + 8)
        # Larguras de coluna
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 100)
        self.table.setColumnWidth(3, 120)
        # Métricas fixas na coluna 0
        self._metric_labels = ["FIQA", "‖embedding‖", "Cosine sim."]
        for r, name in enumerate(self._metric_labels):
            item = QtWidgets.QTableWidgetItem(name)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self.table.setItem(r, 0, item)
        layout.addWidget(self.table)

        # Status line
        self.lbl_status = QtWidgets.QLabel("")
        apply_themed_stylesheet(
            self.lbl_status,
            lambda t: (
                "color: " + t("TEXT_MUTED")
                + "; font-size: 10px; padding: 2px 4px;"
            ),
        )
        layout.addWidget(self.lbl_status)

        self.clear()

    # ── API ────────────────────────────────────────────────────────────────

    def clear(self):
        """Esvazia a tabela (—). Estado inicial e quando muda imagem."""
        for r in range(3):
            for c in range(1, 4):
                self._set_cell(r, c, _DASH)
        self.lbl_status.setText("Clique ▶ Aplicar pra calcular.")

    def show_loading(self):
        """Indica cálculo em andamento."""
        for r in range(3):
            for c in range(1, 4):
                self._set_cell(r, c, _LOADING)
        self.lbl_status.setText("Calculando scores…")

    def show_results(self, result: QualityScoresResult):
        """Preenche tabela a partir do DTO."""
        # FIQA
        self._set_cell_float(0, 1, result.original_fiqa, fmt="{:.1f}")
        self._set_cell_float(0, 2, result.processed_fiqa, fmt="{:.1f}")
        self._set_cell_delta(
            0, 3, result.original_fiqa, result.processed_fiqa,
            fmt="{:+.1f}",
            arrow_lower_is_worse=True,  # FIQA cair = pior
        )

        # ‖embedding‖
        self._set_cell_float(1, 1, result.original_emb_norm, fmt="{:.3f}")
        self._set_cell_float(1, 2, result.processed_emb_norm, fmt="{:.3f}")
        self._set_cell_delta(
            1, 3, result.original_emb_norm, result.processed_emb_norm,
            fmt="{:+.3f}",
            arrow_lower_is_worse=False,  # informativo apenas
        )

        # Cosine
        # Cosine "original × original" é sempre 1.0 (auto-similaridade)
        original_self_cos = 1.0 if result.original_face_detected else None
        self._set_cell_float(2, 1, original_self_cos, fmt="{:.4f}")
        self._set_cell_float(2, 2, result.cosine_similarity, fmt="{:.4f}")
        # Δ no cosine = (proc - 1.0) — quanto se afastou da imagem
        # original. Convenção: mostrar valor direto pra facilitar
        # interpretação (1.0 = sem mudança; abaixo = degradação).
        self._set_cell_delta(
            2, 3, original_self_cos, result.cosine_similarity,
            fmt="{:+.4f}",
            arrow_lower_is_worse=False,  # informativo
        )

        # Status line: avisa se face não foi detectada na processada
        msgs = []
        if not result.original_face_detected:
            msgs.append("⚠ face não detectada na original")
        if not result.processed_face_detected:
            msgs.append("⚠ face destruída pelo pipeline (ou não detectada)")
        self.lbl_status.setText(
            " · ".join(msgs)
            if msgs
            else "Scores calculados."
        )

    def show_error(self, message: str):
        """Mostra mensagem de erro na status line e limpa tabela."""
        for r in range(3):
            for c in range(1, 4):
                self._set_cell(r, c, _DASH)
        self.lbl_status.setText(f"⚠ Erro: {message}")
        t = current_token_fn()
        self.lbl_status.setStyleSheet(
            "color: " + t("DANGER")
            + "; font-size: 10px; padding: 2px 4px; font-weight: 500;"
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _set_cell(self, row: int, col: int, text: str):
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table.setItem(row, col, item)

    def _set_cell_float(
        self, row: int, col: int,
        value: Optional[float],
        fmt: str = "{:.2f}",
    ):
        if value is None:
            self._set_cell(row, col, _DASH)
            return
        self._set_cell(row, col, fmt.format(value))

    def _set_cell_delta(
        self, row: int, col: int,
        before: Optional[float],
        after: Optional[float],
        fmt: str,
        arrow_lower_is_worse: bool,
    ):
        if before is None or after is None:
            self._set_cell(row, col, _DASH)
            return
        delta = after - before
        text = fmt.format(delta)
        # Arrow + cor
        if arrow_lower_is_worse:
            # FIQA: caiu = pior (vermelho); subiu = melhor (verde).
            # Cores de ESTADO da célula → tokens de tema (SUCCESS/DANGER).
            tk = current_token_fn()
            if delta < -0.1:
                text += " ↓"
                color = QtGui.QColor(tk("DANGER"))
            elif delta > 0.1:
                text += " ↑"
                color = QtGui.QColor(tk("SUCCESS"))
            else:
                color = None
        else:
            color = None
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        if color:
            item.setForeground(QtGui.QBrush(color))
            font = item.font()
            font.setBold(True)
            item.setFont(font)
        self.table.setItem(row, col, item)
