# interface/tabs/strategy_manager_v2/strategy_editor/hubness_report_viewer_dialog.py

"""
Dialog full-screen para visualização do HubnessReport.

Embute um `HubnessReportWidget` (5 abas com plots inline) + uma aba
extra "Galeria Completa" listando todos os 20 plots do
`HubnessPlotService` em um grid scrollável. Útil para o perito explorar
todos os ângulos do diagnóstico em uma sessão.
"""

from __future__ import annotations

import io
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtGui, QtWidgets


class HubnessReportViewerDialog(QtWidgets.QDialog):
    """
    Dialog em tela cheia para visualização completa de um HubnessReport.

    Aba 1: HubnessReportWidget com os 5 plots curados (mesmo widget inline
           usado na 3ª coluna do editor V2).
    Aba 2: Galeria com TODOS os 20 plots do HubnessPlotService em grid
           scrollável (PNGs estáticos — interativo só via "Salvar como…"
           usando a toolbar dos plots da aba 1).
    """

    def __init__(
        self,
        result: Any,  # HubnessReport
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle(
            f"Hubness Report — {getattr(result, 'dataset_name', 'dataset')}"
        )
        self.resize(1200, 850)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        outer_tabs = QtWidgets.QTabWidget()
        layout.addWidget(outer_tabs, 1)

        # ─── Aba 1: widget reutilizável (5 abas curadas) ──────────────────
        from .result_widgets.hubness_report_widget import HubnessReportWidget
        self.curated = HubnessReportWidget(
            parent=self,
            result=result,
            show_header=True,
            show_toolbar=True,
            show_help=True,
        )
        outer_tabs.addTab(self.curated, "Diagnóstico (5 plots curados)")

        # ─── Aba 2: galeria completa (20 plots) ───────────────────────────
        gallery = self._build_gallery_tab()
        outer_tabs.addTab(gallery, "Galeria Completa (todos os plots)")

        # ─── Botão Fechar ─────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    # ── Galeria ────────────────────────────────────────────────────────────

    def _build_gallery_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        container = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(container)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(12)

        from faceaudit.services.hubness_plot_service import HubnessPlotService
        ps = HubnessPlotService()

        analysis = self._result.analysis
        diag = self._result.quality_diagnostics
        opt = self._result.optimization_study

        # (título, callable, *args)
        plots: List[Tuple[str, Callable, tuple]] = [
            ("Hubness × Similaridade Impostora",
             ps.plot_hubness_vs_impostor_similarity, (analysis,)),
            ("Norma × Hubness — distribuição",
             ps.plot_norm_vs_hubness_distribution, (analysis,)),
            ("Norma × Pose — impacto",
             ps.plot_norm_vs_pose_impact, (analysis,)),
            ("Pose × Hubness — direto",
             ps.plot_pose_vs_hubness_direct, (analysis,)),
            ("Pose × Norma × Hubness — interação",
             ps.plot_pose_norm_interaction, (analysis,)),
            ("Estratificação por pose (yaw)",
             ps.plot_stratified_hubness, (analysis,)),
            ("Estudo pareado — slope chart",
             ps.plot_paired_slope_impact, (analysis,)),
            ("Estudo pareado — Δ risco impostor",
             ps.plot_paired_delta_impostor_risk, (analysis,)),
            ("Hubness × Δ pose dos vizinhos",
             ps.plot_hubness_vs_pose_delta, (analysis,)),
            ("Norma × qualidade estocástica (SERFIQ)",
             ps.plot_norm_vs_stochastic_quality, (analysis,)),
            ("Tendência de otimização (sweet spot)",
             ps.plot_optimization_trends, (opt,)),
            ("Norma — ridgeline por pose",
             ps.plot_norm_ridgeline, (analysis,)),
            ("Diagnóstico FIQA / Norma × pose",
             ps.plot_quality_diagnostics, (diag,)),
            ("Densidade de Hubness",
             ps.plot_hubness_density, (analysis,)),
            ("Hubness por estratos de norma",
             ps.plot_hubness_by_norm_strata, (analysis,)),
            ("Norma × Nk — LOWESS",
             ps.plot_norm_vs_nk_lowess, (analysis,)),
        ]

        # Renderiza cada plot como PNG e adiciona ao grid (3 colunas).
        rendered = self._render_all_to_pngs(plots)
        cols = 3
        for i, (title, png_bytes) in enumerate(rendered):
            row, col = divmod(i, cols)
            cell = self._build_gallery_cell(title, png_bytes)
            grid.addWidget(cell, row, col)

        # Stretch da última coluna pra layout não esticar células.
        for c in range(cols):
            grid.setColumnStretch(c, 1)

        scroll.setWidget(container)
        return scroll

    def _render_all_to_pngs(
        self, plots: List[Tuple[str, Callable, tuple]]
    ) -> List[Tuple[str, bytes]]:
        """
        Renderiza cada plot como PNG via monkey-patch de plt.show.
        Retorna lista de (título, png_bytes). Plots que falharem ficam
        com PNG vazio + título marcado (a célula mostra erro).
        """
        results: List[Tuple[str, bytes]] = []
        original_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            for title, fn, args in plots:
                try:
                    fn(*args)
                    fig = plt.gcf()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    results.append((title, buf.read()))
                except Exception as exc:
                    # Marca falha — célula vai mostrar texto de erro.
                    results.append((f"{title} — ERRO: {exc}", b""))
        finally:
            plt.show = original_show
        return results

    @staticmethod
    def _build_gallery_cell(title: str, png_bytes: bytes) -> QtWidgets.QWidget:
        cell = QtWidgets.QFrame()
        cell.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        cell.setStyleSheet(
            "QFrame { border: 1px solid #d0d8e0; border-radius: 4px;"
            " background: white; }"
        )
        v = QtWidgets.QVBoxLayout(cell)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        lbl_title = QtWidgets.QLabel(title)
        lbl_title.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #1a2942;"
            " background: transparent; border: none;"
        )
        lbl_title.setWordWrap(True)
        v.addWidget(lbl_title)

        if png_bytes:
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(png_bytes, "PNG")
            # Escala pra largura máxima ~360px mantendo aspect ratio.
            pixmap = pixmap.scaledToWidth(
                360, QtCore.Qt.TransformationMode.SmoothTransformation
            )
            lbl_img = QtWidgets.QLabel()
            lbl_img.setPixmap(pixmap)
            lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl_img.setStyleSheet("background: transparent; border: none;")
            v.addWidget(lbl_img)
        else:
            err_lbl = QtWidgets.QLabel("(falha ao gerar plot)")
            err_lbl.setStyleSheet(
                "color: #a33; font-style: italic; font-size: 10px;"
                " background: transparent; border: none;"
            )
            err_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            v.addWidget(err_lbl)

        return cell
