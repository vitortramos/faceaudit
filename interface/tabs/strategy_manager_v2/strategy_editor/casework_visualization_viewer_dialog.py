# interface/tabs/strategy_manager_v2/strategy_editor/casework_visualization_viewer_dialog.py

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"

_KDE_MIN_CASE_SAMPLES = 30


class CaseworkVisualizationViewerDialog(QtWidgets.QDialog):
    """
    Dialog single-page para visualizar um CaseworkVisualizationResult.

    Sobrepõe a distribuição dos scores do caso forense (questionada × padrão)
    sobre as distribuições de referência (genuínos verde, impostores
    vermelho). Quando há ≥30 pares de caso, usa KDE; abaixo disso, usa
    linhas verticais tracejadas com valor numérico anotado.

    Header âmbar mostra modo (RAW/CALIBRATED), contagens, sumário do caso
    e (em RAW) o threshold de EER.
    """

    def __init__(
        self,
        result,  # CaseworkVisualizationResult
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        title = (
            f"Visualização Forense de Caso — "
            f"{result.case_dataset_name or 'caso'} sobre "
            f"{result.reference_dataset_name or 'referência'}"
        )
        self.setWindowTitle(title)
        self.resize(950, 720)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ─── Header com sumário ──────────────────────────────────────
        layout.addWidget(self._build_header())

        # ─── Plot ────────────────────────────────────────────────────
        figure = Figure(figsize=(10, 6), dpi=100)
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)

        try:
            self._draw_visualization(figure)
        except Exception as exc:
            ax = figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Erro ao gerar plot:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                color="#a33", fontsize=10,
            )
        canvas.draw()

        # ─── Texto de ajuda contextual ───────────────────────────────
        layout.addWidget(self._build_help_box())

        # ─── Botão Fechar ─────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ─── UI helpers ────────────────────────────────────────────────────

    def _build_header(self) -> QtWidgets.QWidget:
        r = self._result
        cs = r.case_summary or {}
        case_part = (
            f"min={cs.get('min', 0):+.2f} | "
            f"mediana={cs.get('median', 0):+.2f} | "
            f"max={cs.get('max', 0):+.2f}"
            if cs else "sem amostras"
        )
        eer_part = (
            f" &nbsp;·&nbsp; <b>EER threshold</b>={r.eer_threshold:.4f}"
            if r.eer_threshold is not None else ""
        )

        text = (
            f"<b>Modo: {r.score_type}</b>"
            f" &nbsp;·&nbsp; Referência: {r.n_genuine:,} gen / {r.n_impostor:,} imp"
            f" &nbsp;·&nbsp; Caso: {r.n_case} pares ({case_part})"
            f"{eer_part}"
        ).replace(",", ".")

        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("background: transparent; color: #5a4500; font-size: 11px;")
        lbl.setWordWrap(True)
        h.addWidget(lbl, 1)
        return frame

    def _build_help_box(self) -> QtWidgets.QWidget:
        if self._result.score_type == "CALIBRATED":
            html = (
                "<b>Como ler o gráfico:</b> as duas curvas (verde gen, vermelho imp) "
                "mostram onde o calibrador posiciona pares conhecidos da população "
                "de referência (validação). A linha tracejada azul indica os escores "
                "do caso forense em log-LR. Se a marca do caso cai bem dentro da "
                "curva verde → forte evidência a favor de Hp (mesma pessoa). Bem "
                "dentro da curva vermelha → forte evidência a favor de Hd (pessoas "
                "diferentes). Na zona de sobreposição → evidência fraca/ambígua. "
                "A linha cinza tracejada em log-LR=0 marca o limiar neutro."
            )
        else:
            html = (
                "<b>Como ler o gráfico:</b> distribuição bruta (raw_score) dos pares "
                "de validação por classe (verde gen, vermelho imp). A linha cinza "
                "tracejada marca o threshold de EER calculado on-the-fly do dataset "
                "de referência — ponto onde FNR=FPR. Os escores do caso (azul) "
                "podem ser comparados visualmente: à direita do threshold sugere "
                "Hp, à esquerda sugere Hd. Para evidência probabilística "
                "(log-LR), use o modo CALIBRATED após passar os scores por "
                "SCORE_CALIBRATION."
            )

        container = QtWidgets.QFrame()
        container.setStyleSheet(
            "QFrame { background: #f4f6fa; border-left: 3px solid #2E5090;"
            " border-radius: 2px; }"
        )
        container.setMaximumHeight(110)
        inner = QtWidgets.QVBoxLayout(container)
        inner.setContentsMargins(10, 6, 10, 6)
        lbl = QtWidgets.QLabel(html)
        lbl.setStyleSheet("background: transparent; color: #1A2942; font-size: 10px;")
        lbl.setWordWrap(True)
        inner.addWidget(lbl)
        return container

    # ─── Plot ──────────────────────────────────────────────────────────

    def _draw_visualization(self, figure: Figure) -> None:
        from scipy.stats import gaussian_kde

        r = self._result
        ax = figure.add_subplot(111)

        gen = np.asarray(r.genuine_scores, dtype=np.float64)
        imp = np.asarray(r.impostor_scores, dtype=np.float64)
        case = np.asarray(r.case_scores, dtype=np.float64)

        # Determina range do eixo X cobrindo as três distribuições + folga.
        all_vals = []
        if gen.size > 0:
            all_vals.append(gen)
        if imp.size > 0:
            all_vals.append(imp)
        if case.size > 0:
            all_vals.append(case)
        if not all_vals:
            ax.text(0.5, 0.5, "Sem dados para plotar.",
                    ha="center", va="center", transform=ax.transAxes)
            return
        combined = np.concatenate(all_vals)
        x_min = float(np.min(combined))
        x_max = float(np.max(combined))
        pad = max(0.05 * (x_max - x_min), 0.5)
        x_min -= pad
        x_max += pad
        xs = np.linspace(x_min, x_max, 500)

        # ─── Histogramas + KDE de referência ────────────────────────
        if imp.size > 0:
            ax.hist(
                imp, bins=50, density=True, color="#d62728", alpha=0.35,
                label=f"Impostores (n={imp.size:,})".replace(",", "."),
            )
            if imp.size > 1:
                try:
                    ax.plot(xs, gaussian_kde(imp)(xs),
                            color="#8b0000", linewidth=1.4, alpha=0.9)
                except Exception:
                    pass

        if gen.size > 0:
            ax.hist(
                gen, bins=50, density=True, color="#2ca02c", alpha=0.35,
                label=f"Genuínos (n={gen.size:,})".replace(",", "."),
            )
            if gen.size > 1:
                try:
                    ax.plot(xs, gaussian_kde(gen)(xs),
                            color="#1d4d1d", linewidth=1.4, alpha=0.9)
                except Exception:
                    pass

        # ─── Sobreposição do caso ────────────────────────────────────
        if case.size >= _KDE_MIN_CASE_SAMPLES:
            ax.hist(
                case, bins=30, density=True, color="#1f77b4", alpha=0.35,
                label=f"Caso (n={case.size:,}, KDE)".replace(",", "."),
            )
            if case.size > 1:
                try:
                    ax.plot(xs, gaussian_kde(case)(xs),
                            color="#0d3b66", linewidth=1.6, alpha=0.95)
                except Exception:
                    pass
        else:
            # Linhas verticais tracejadas com valor numérico anotado.
            y_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
            for i, val in enumerate(case):
                label = (
                    f"Caso (n={case.size}, marcas)"
                    if i == 0 else None
                )
                ax.axvline(
                    val, color="#1f77b4", linestyle="--", linewidth=1.4,
                    alpha=0.9, label=label,
                )
                ax.annotate(
                    f"{val:+.2f}",
                    xy=(val, y_top * 0.95),
                    xytext=(0, -5),
                    textcoords="offset points",
                    rotation=90,
                    color="#0d3b66",
                    fontsize=8,
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="#1f77b4", lw=0.8, alpha=0.85),
                )

        # ─── Linha de referência ─────────────────────────────────────
        if r.score_type == "CALIBRATED":
            ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                       alpha=0.7, label="log-LR = 0 (limiar neutro)")
        elif r.score_type == "RAW" and r.eer_threshold is not None:
            ax.axvline(
                r.eer_threshold, color="#444", linestyle=":", linewidth=1.0,
                alpha=0.7,
                label=f"EER threshold = {r.eer_threshold:.4f}",
            )

        # ─── Estilo final ────────────────────────────────────────────
        x_label = (
            "log-LR calibrado"
            if r.score_type == "CALIBRATED" else "raw_score"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Densidade")
        ax.set_xlim(x_min, x_max)
        ax.set_title(
            "Distribuição forense — Caso sobreposto à referência"
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)

        figure.tight_layout()
