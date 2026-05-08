"""Widget Qt-puro pra exibir o card de métricas de balanceamento.

Consumido pela `DatasetAnalysisView` quando o usuário seleciona a
auditoria `balance_metrics`. Sem matplotlib — é um painel de informação
estruturada com verdict em destaque + grid de números, no mesmo padrão
visual que o `ResourceInfoWidget` usa pro mesmo dado.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtWidgets


class BalanceMetricsWidget(QtWidgets.QWidget):
    """Card com verdict + tabela de métricas de balanceamento."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)
        outer.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Header de verdict — fonte grande, cor depende do tier
        self._verdict_label = QtWidgets.QLabel("—")
        self._verdict_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        self._verdict_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 14px; "
            "border-radius: 6px; background: #f5f5f5; color: #333;"
        )
        outer.addWidget(self._verdict_label)

        # Linha explicativa — abaixo do verdict
        self._explain_label = QtWidgets.QLabel("")
        self._explain_label.setWordWrap(True)
        self._explain_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        self._explain_label.setStyleSheet(
            "color: #555; font-size: 11px; padding: 0 20px;"
        )
        outer.addWidget(self._explain_label)

        # Grid de métricas
        self._grid = QtWidgets.QGridLayout()
        self._grid.setHorizontalSpacing(24)
        self._grid.setVerticalSpacing(8)
        self._grid.setColumnStretch(0, 0)
        self._grid.setColumnStretch(1, 1)
        outer.addLayout(self._grid)

        outer.addStretch()

    def render_metrics(self, metrics: Dict[str, Any]) -> None:
        """Atualiza o card com novos números. `metrics` segue o shape
        do metadata produzido por `audit_balance` no service:
        verdict / n_identities / min_samples / max_samples /
        mean_samples / median_samples / std_samples / cv /
        imbalance_ratio / n_total_samples."""
        verdict = str(metrics.get("verdict", "—"))
        bg, fg = self._verdict_palette(verdict)
        self._verdict_label.setText(verdict)
        self._verdict_label.setStyleSheet(
            f"font-size: 22px; font-weight: bold; padding: 14px; "
            f"border-radius: 6px; background: {bg}; color: {fg};"
        )

        # Texto explicativo do tier (curto, contextual)
        self._explain_label.setText(self._verdict_explanation(verdict))

        # Limpa o grid e re-popula
        while self._grid.count() > 0:
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        rows = [
            ("Total de identidades", f"{int(metrics.get('n_identities', 0)):,}"),
            ("Total de amostras", f"{int(metrics.get('n_total_samples', 0)):,}"),
            ("Min. amostras / ID", f"{int(metrics.get('min_samples', 0)):,}"),
            ("Max. amostras / ID", f"{int(metrics.get('max_samples', 0)):,}"),
            ("Mediana de amostras / ID",
             f"{float(metrics.get('median_samples', 0.0)):.1f}"),
            ("Média ± desvio-padrão",
             f"{float(metrics.get('mean_samples', 0.0)):.2f} "
             f"± {float(metrics.get('std_samples', 0.0)):.2f}"),
            ("Coef. de variação (CV)",
             f"{float(metrics.get('cv', 0.0)):.3f}"),
            ("Razão max/min",
             f"{float(metrics.get('imbalance_ratio', 0.0)):.1f} : 1"),
        ]
        for row_idx, (label, value) in enumerate(rows):
            lbl = QtWidgets.QLabel(label + ":")
            lbl.setStyleSheet(
                "color: #555; font-size: 12px;"
            )
            lbl.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignRight
                | QtCore.Qt.AlignmentFlag.AlignVCenter,
            )
            val = QtWidgets.QLabel(str(value))
            val.setStyleSheet(
                "color: #1a3d6e; font-size: 13px; font-weight: bold;"
            )
            val.setTextInteractionFlags(
                QtCore.Qt.TextInteractionFlag.TextSelectableByMouse,
            )
            self._grid.addWidget(lbl, row_idx, 0)
            self._grid.addWidget(val, row_idx, 1)

    def clear(self) -> None:
        self._verdict_label.setText("—")
        self._verdict_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; padding: 14px; "
            "border-radius: 6px; background: #f5f5f5; color: #333;"
        )
        self._explain_label.setText("")
        while self._grid.count() > 0:
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    @staticmethod
    def _verdict_palette(verdict: str) -> tuple:
        """Retorna (bg, fg) baseado no emoji do verdict — alinhado com
        os 4 tiers do `BalanceMetrics`."""
        if verdict.startswith("✅"):
            return "#e8f5e9", "#1b5e20"  # verde
        if verdict.startswith("🟢"):
            return "#f1f8e9", "#2e7d32"  # verde-claro
        if verdict.startswith("🟡"):
            return "#fff8e1", "#ef6c00"  # âmbar
        if verdict.startswith("🔴"):
            return "#ffebee", "#c62828"  # vermelho
        return "#f5f5f5", "#333"

    @staticmethod
    def _verdict_explanation(verdict: str) -> str:
        """Texto curto contextualizando o verdict."""
        if verdict.startswith("✅"):
            return (
                "Todas as identidades têm a mesma quantidade de amostras "
                "— condição ideal pra treinamento e calibração."
            )
        if verdict.startswith("🟢"):
            return (
                "Variação leve entre identidades — uso forense sem "
                "ressalvas. CV ≤ 0.35 e razão max/min ≤ 4:1."
            )
        if verdict.startswith("🟡"):
            return (
                "Identidades com massas estatísticas bem diferentes "
                "podem dominar centroides e enviesar calibração. "
                "Considere balancear (DATASET_BALANCING)."
            )
        if verdict.startswith("🔴"):
            return (
                "Algumas identidades têm muito mais amostras que outras — "
                "alto risco de viés sistemático no treinamento. "
                "Balancear é fortemente recomendado."
            )
        return ""
