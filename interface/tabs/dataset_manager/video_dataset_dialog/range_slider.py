# interface/tabs/dataset_manager/video_dataset_dialog/range_slider.py
"""_RangeSlider — slider horizontal de DOIS cabos (intervalo [lo, hi]).

Usado nos filtros da janela de revisão de vídeo (det / fiqa / yaw / pitch):
o usuário seleciona um intervalo min–max e o botão "Aplicar filtros" marca/
desmarca os recortes cuja métrica cai dentro do intervalo. Sem dependência
externa (superqt) — QWidget pintado, no mesmo espírito do `_RangeScrubber`
da janela de extração. Trabalha em escala de ponto-flutuante (det/fiqa em
0..1; yaw/pitch em graus) — os limites são definidos por `setBounds`.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


class _RangeSlider(QtWidgets.QWidget):
    """Trilho com duas alças delimitando um intervalo contínuo em [min, max].
    Mantém lo ≤ hi (as alças não se cruzam). Emite `rangeChanged(lo, hi)` a
    cada arraste e `rangeReleased(lo, hi)` ao soltar."""

    rangeChanged = QtCore.pyqtSignal(float, float)
    rangeReleased = QtCore.pyqtSignal(float, float)

    _MARGIN = 9
    _HANDLE_R = 5.5

    def __init__(self, parent=None, *, minimum: float = 0.0, maximum: float = 1.0):
        super().__init__(parent)
        self._min = float(minimum)
        self._max = float(maximum) if maximum > minimum else float(minimum) + 1.0
        self._lo = self._min
        self._hi = self._max
        self._drag: Optional[str] = None
        # Compacto: ~18px de altura (não dominar o grupo de filtros).
        self.setFixedHeight(18)
        self.setMinimumWidth(80)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    # ----------------------------- API -----------------------------
    def setBounds(self, lo: float, hi: float, *, keep_values: bool = False) -> None:
        """Define os limites do trilho. Por padrão reseta as alças para os
        extremos (intervalo aberto = sem filtro). `keep_values=True` apenas
        re-clampa as alças atuais aos novos limites."""
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            hi = lo + 1.0
        self._min, self._max = lo, hi
        if keep_values:
            self._lo = self._clamp(self._lo)
            self._hi = self._clamp(self._hi)
            if self._lo > self._hi:
                self._lo = self._hi
        else:
            self._lo, self._hi = lo, hi
        self.update()

    def setValues(self, lo: float, hi: float) -> None:
        lo, hi = self._clamp(lo), self._clamp(hi)
        if lo > hi:
            lo, hi = hi, lo
        self._lo, self._hi = lo, hi
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)

    def values(self) -> Tuple[float, float]:
        return self._lo, self._hi

    def bounds(self) -> Tuple[float, float]:
        return self._min, self._max

    def contains(self, v: Optional[float]) -> bool:
        """True se `v` está no intervalo selecionado. `None` passa (métrica
        ausente não exclui a face — política do filtro)."""
        if v is None:
            return True
        return self._lo <= float(v) <= self._hi

    # --------------------------- mapping ---------------------------
    def _clamp(self, v: float) -> float:
        return max(self._min, min(self._max, float(v)))

    def _x(self, val: float) -> float:
        m = self._MARGIN
        w = max(1, self.width() - 2 * m)
        if self._max <= self._min:
            return float(m)
        return m + (val - self._min) / (self._max - self._min) * w

    def _val(self, x: float) -> float:
        m = self._MARGIN
        w = max(1, self.width() - 2 * m)
        return self._clamp(self._min + (x - m) / w * (self._max - self._min))

    # ---------------------------- paint ----------------------------
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        y = self.height() // 2
        m = self._MARGIN
        p.setPen(QtGui.QPen(QtGui.QColor("#cfcfcf"), 3))
        p.drawLine(m, y, self.width() - m, y)
        if not self.isEnabled():
            return
        xlo, xhi = self._x(self._lo), self._x(self._hi)
        p.setPen(QtGui.QPen(QtGui.QColor("#1976d2"), 3))
        p.drawLine(int(xlo), y, int(xhi), y)
        self._handle(p, xlo, y)
        self._handle(p, xhi, y)

    def _handle(self, p, x, y):
        p.setBrush(QtGui.QColor("#ffffff"))
        p.setPen(QtGui.QPen(QtGui.QColor("#1976d2"), 1.5))
        p.drawEllipse(QtCore.QPointF(x, y), self._HANDLE_R, self._HANDLE_R)

    # ---------------------------- mouse ----------------------------
    def mousePressEvent(self, ev):
        if not self.isEnabled():
            return
        x = ev.position().x()
        dlo, dhi = abs(self._x(self._lo) - x), abs(self._x(self._hi) - x)
        if dlo == dhi:
            self._drag = "lo" if x <= self._x(self._lo) else "hi"
        else:
            self._drag = "lo" if dlo < dhi else "hi"
        self._apply(x)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._apply(ev.position().x())

    def mouseReleaseEvent(self, ev):
        if self._drag:
            self._drag = None
            self.rangeReleased.emit(self._lo, self._hi)

    def _apply(self, x):
        v = self._val(x)
        if self._drag == "lo":
            self._lo = min(v, self._hi)
        elif self._drag == "hi":
            self._hi = max(v, self._lo)
        self.update()
        self.rangeChanged.emit(self._lo, self._hi)
