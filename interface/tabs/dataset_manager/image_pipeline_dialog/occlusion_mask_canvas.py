"""OcclusionMaskCanvas — dialog modal pra desenhar máscara livre
(PR I6).

UX:
  - Imagem do preview ao centro com aspect-ratio preservado.
  - Toolbar superior com 3 ferramentas + Limpar:
      🖊 Polígono — cliques sequenciais; double-click finaliza
      ⬛ Retângulo — drag (press → arrastar → release)
      ⭕ Elipse — drag idem retângulo (define bbox)
      🗑 Limpar — apaga todos os shapes desenhados
  - Shapes finalizados são pintados em preto semi-transparente.
  - Shape em construção em vermelho.
  - OK retorna `OcclusionMask` com coords em [0..1].
  - Cancel retorna None.

Coordenadas são armazenadas em pixels do widget durante o desenho;
convertidas pra [0..1] da IMAGEM (não do widget) ao retornar a mask.
Isso garante que a mesma máscara aplica em imagens de tamanhos
diferentes (decisão herdada — coords percentuais).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.enums.occlusion_mask_shape import OcclusionMaskShape
from faceaudit.models.image_processing.occlusion_mask import (
    OcclusionMask,
    OcclusionMaskShapeDTO,
)


# Cores de desenho (RGB pra Qt)
_FINALIZED_COLOR = QtGui.QColor(0, 0, 0, 160)       # preto semi-transparente
_IN_PROGRESS_COLOR = QtGui.QColor(220, 30, 30, 200)  # vermelho


class _ImageCanvas(QtWidgets.QWidget):
    """Widget custom — renderiza imagem + shapes + captura mouse."""

    def __init__(self, pixmap: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._tool = OcclusionMaskShape.POLYGON
        # Cada shape = (shape_type, list of (x, y) em pixels do widget)
        self._shapes: List[Tuple[OcclusionMaskShape, List[Tuple[int, int]]]] = []
        # Estado de construção
        self._poly_in_progress: List[Tuple[int, int]] = []
        self._rect_start: Optional[Tuple[int, int]] = None
        self._rect_current: Optional[Tuple[int, int]] = None

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        # Cursor crosshair pra indicar drawing
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    # ── API pública ────────────────────────────────────────────────────────

    def set_tool(self, tool: OcclusionMaskShape):
        """Reset de estado ao trocar ferramenta."""
        self._tool = tool
        self._poly_in_progress = []
        self._rect_start = None
        self._rect_current = None
        self.update()

    def clear_shapes(self):
        self._shapes = []
        self._poly_in_progress = []
        self._rect_start = None
        self._rect_current = None
        self.update()

    def get_mask_shapes(self) -> List[OcclusionMaskShapeDTO]:
        """Devolve shapes em coords [0..1] da IMAGEM (não do widget).

        Se há um polígono em construção (não finalizado via double-
        click), ele é descartado — usuário precisa finalizar pra
        contar.
        """
        img_x, img_y, img_w, img_h = self._image_render_rect()
        if img_w <= 0 or img_h <= 0:
            return []
        result: List[OcclusionMaskShapeDTO] = []
        for shape_type, pts_widget in self._shapes:
            pts_norm = []
            for x, y in pts_widget:
                # Pixel widget → percentual da imagem renderizada
                ix = (x - img_x) / img_w
                iy = (y - img_y) / img_h
                ix = max(0.0, min(1.0, ix))
                iy = max(0.0, min(1.0, iy))
                pts_norm.append((float(ix), float(iy)))
            result.append(OcclusionMaskShapeDTO(
                shape_type=shape_type,
                points=tuple(pts_norm),
            ))
        return result

    @property
    def has_shapes(self) -> bool:
        return len(self._shapes) > 0 or bool(self._poly_in_progress)

    # ── Layout / coords ────────────────────────────────────────────────────

    def _image_render_rect(self) -> Tuple[int, int, int, int]:
        """Devolve (x, y, w, h) em pixels do widget onde a imagem é
        renderizada (com aspect-ratio preservado)."""
        widget_w, widget_h = self.width(), self.height()
        if self._pixmap.isNull():
            return (0, 0, 0, 0)
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        # Aspect-fit
        scale = min(widget_w / pix_w, widget_h / pix_h) if pix_w and pix_h else 1
        rendered_w = int(pix_w * scale)
        rendered_h = int(pix_h * scale)
        x = (widget_w - rendered_w) // 2
        y = (widget_h - rendered_h) // 2
        return (x, y, rendered_w, rendered_h)

    def _is_inside_image(self, x: int, y: int) -> bool:
        ix, iy, iw, ih = self._image_render_rect()
        return ix <= x <= ix + iw and iy <= y <= iy + ih

    def _clamp_to_image(self, x: int, y: int) -> Tuple[int, int]:
        ix, iy, iw, ih = self._image_render_rect()
        return (
            max(ix, min(ix + iw, x)),
            max(iy, min(iy + ih, y)),
        )

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Fundo cinza escuro
        painter.fillRect(self.rect(), QtGui.QColor("#222"))

        # Imagem
        ix, iy, iw, ih = self._image_render_rect()
        if iw > 0 and ih > 0:
            scaled = self._pixmap.scaled(
                iw, ih,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(ix, iy, scaled)

        # Shapes finalizados (preto semi-transparente)
        painter.setBrush(QtGui.QBrush(_FINALIZED_COLOR))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 220), 2))
        for shape_type, pts in self._shapes:
            self._draw_shape(painter, shape_type, pts)

        # Shape em construção (vermelho)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 30, 30, 80)))
        painter.setPen(QtGui.QPen(_IN_PROGRESS_COLOR, 2))

        if self._poly_in_progress:
            # Linhas conectando + ponto final livre seguindo cursor
            self._draw_polygon_in_progress(painter, self._poly_in_progress)
        if (
            self._rect_start is not None
            and self._rect_current is not None
        ):
            self._draw_rect_or_ellipse_preview(
                painter, self._rect_start, self._rect_current, self._tool,
            )

    def _draw_shape(
        self,
        painter: QtGui.QPainter,
        shape_type: OcclusionMaskShape,
        pts: List[Tuple[int, int]],
    ):
        if shape_type == OcclusionMaskShape.POLYGON and len(pts) >= 3:
            polygon = QtGui.QPolygon([
                QtCore.QPoint(p[0], p[1]) for p in pts
            ])
            painter.drawPolygon(polygon)
        elif shape_type == OcclusionMaskShape.RECTANGLE and len(pts) >= 2:
            rect = self._rect_from_two_points(pts[0], pts[1])
            painter.drawRect(rect)
        elif shape_type == OcclusionMaskShape.ELLIPSE and len(pts) >= 2:
            rect = self._rect_from_two_points(pts[0], pts[1])
            painter.drawEllipse(rect)

    def _draw_polygon_in_progress(self, painter, pts):
        # Linhas entre pontos clicados + bolinhas nos vértices
        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                painter.drawLine(
                    pts[i][0], pts[i][1],
                    pts[i + 1][0], pts[i + 1][1],
                )
        for x, y in pts:
            painter.drawEllipse(QtCore.QPoint(x, y), 4, 4)

    def _draw_rect_or_ellipse_preview(
        self, painter, p1, p2, tool,
    ):
        rect = self._rect_from_two_points(p1, p2)
        if tool == OcclusionMaskShape.RECTANGLE:
            painter.drawRect(rect)
        else:
            painter.drawEllipse(rect)

    @staticmethod
    def _rect_from_two_points(
        p1: Tuple[int, int], p2: Tuple[int, int],
    ) -> QtCore.QRect:
        x = min(p1[0], p2[0])
        y = min(p1[1], p2[1])
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        return QtCore.QRect(x, y, w, h)

    # ── Mouse events ────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())
        if not self._is_inside_image(x, y):
            return
        x, y = self._clamp_to_image(x, y)

        if self._tool == OcclusionMaskShape.POLYGON:
            self._poly_in_progress.append((x, y))
        elif self._tool in (
            OcclusionMaskShape.RECTANGLE,
            OcclusionMaskShape.ELLIPSE,
        ):
            self._rect_start = (x, y)
            self._rect_current = (x, y)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._rect_start is not None:
            pos = event.position()
            x, y = self._clamp_to_image(int(pos.x()), int(pos.y()))
            self._rect_current = (x, y)
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._rect_start is None or self._rect_current is None:
            return
        # Só salva se há área significativa
        dx = abs(self._rect_current[0] - self._rect_start[0])
        dy = abs(self._rect_current[1] - self._rect_start[1])
        if dx > 4 and dy > 4:
            self._shapes.append((
                self._tool,
                [self._rect_start, self._rect_current],
            ))
        self._rect_start = None
        self._rect_current = None
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        # Finaliza polígono atual (precisa ≥ 3 vértices)
        if self._tool != OcclusionMaskShape.POLYGON:
            return
        if len(self._poly_in_progress) >= 3:
            self._shapes.append((
                OcclusionMaskShape.POLYGON,
                list(self._poly_in_progress),
            ))
        self._poly_in_progress = []
        self.update()


class OcclusionMaskCanvas(QtWidgets.QDialog):
    """Modal pra desenhar máscara. Construa com `pixmap` da imagem
    de preview + opcional `initial_mask` (não usado em I6 — sempre
    começa vazio). Após `exec() == Accepted`, chame `get_mask()`."""

    def __init__(
        self,
        pixmap: QtGui.QPixmap,
        initial_mask: Optional[OcclusionMask] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Desenhar Máscara de Oclusão")
        self.setModal(True)
        self.resize(900, 720)

        self._build_ui(pixmap)

        # I6 não suporta carregar máscara existente pra editar — sempre
        # começa vazio. Limitação intencional: edição visual de
        # polígonos existentes é UX complexa (drag de vértices).
        # Versões futuras podem aceitar `initial_mask` pra pre-popular.

    def _build_ui(self, pixmap: QtGui.QPixmap):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Toolbar topo: 3 tools + Limpar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_polygon = self._make_tool_button(
            "🖊 Polígono",
            "Cliques sequenciais; double-click pra fechar.",
        )
        self.btn_rectangle = self._make_tool_button(
            "⬛ Retângulo",
            "Drag (clique e arraste) pra desenhar.",
        )
        self.btn_ellipse = self._make_tool_button(
            "⭕ Elipse",
            "Drag pra desenhar (bbox define a elipse).",
        )

        self.btn_polygon.setCheckable(True)
        self.btn_rectangle.setCheckable(True)
        self.btn_ellipse.setCheckable(True)
        self.btn_polygon.setChecked(True)

        self._tool_buttons = {
            OcclusionMaskShape.POLYGON: self.btn_polygon,
            OcclusionMaskShape.RECTANGLE: self.btn_rectangle,
            OcclusionMaskShape.ELLIPSE: self.btn_ellipse,
        }

        self.btn_polygon.clicked.connect(
            lambda: self._on_tool_selected(OcclusionMaskShape.POLYGON)
        )
        self.btn_rectangle.clicked.connect(
            lambda: self._on_tool_selected(OcclusionMaskShape.RECTANGLE)
        )
        self.btn_ellipse.clicked.connect(
            lambda: self._on_tool_selected(OcclusionMaskShape.ELLIPSE)
        )

        toolbar.addWidget(self.btn_polygon)
        toolbar.addWidget(self.btn_rectangle)
        toolbar.addWidget(self.btn_ellipse)
        toolbar.addSpacing(20)

        self.btn_clear = QtWidgets.QPushButton("🗑 Limpar")
        self.btn_clear.setToolTip("Apagar todos os shapes desenhados")
        self.btn_clear.clicked.connect(self._on_clear)
        toolbar.addWidget(self.btn_clear)
        toolbar.addStretch()

        # Texto de ajuda
        help_lbl = QtWidgets.QLabel(
            "<i>Polígono</i>: cliques + double-click finaliza.   "
            "<i>Retângulo / Elipse</i>: arrastar."
        )
        help_lbl.setStyleSheet("color: #666; font-size: 11px;")
        toolbar.addWidget(help_lbl)

        layout.addLayout(toolbar)

        # Canvas
        self._canvas = _ImageCanvas(pixmap)
        layout.addWidget(self._canvas, 1)

        # Botões OK/Cancel
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("✓ Aplicar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _make_tool_button(self, text: str, tooltip: str):
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setStyleSheet(
            "QPushButton { padding: 6px 12px; border: 1px solid #ccc; "
            "border-radius: 4px; background: white; } "
            "QPushButton:checked { background: #cce4ff; "
            "border-color: #4a90d9; font-weight: bold; }"
        )
        return btn

    def _on_tool_selected(self, tool: OcclusionMaskShape):
        for t, btn in self._tool_buttons.items():
            btn.setChecked(t == tool)
        self._canvas.set_tool(tool)

    def _on_clear(self):
        self._canvas.clear_shapes()

    # ── API pública ────────────────────────────────────────────────────────

    def get_mask(self) -> OcclusionMask:
        """Devolve a máscara atual em coords [0..1]. Pode estar vazia."""
        shapes = self._canvas.get_mask_shapes()
        return OcclusionMask(shapes=tuple(shapes))
