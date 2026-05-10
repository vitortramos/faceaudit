"""OcclusionMaskCanvas — dialog modal pra desenhar/editar máscara
de oclusão livre.

Histórico:
  - PR I6: criação. Modo desenho com 3 ferramentas (polígono,
    retângulo, elipse). Coords em pixels do widget; conversão
    pra [0..1] no `get_mask_shapes()`.
  - Refactor pós-I7: suporte a edição (mover/redimensionar/apagar/
    duplicar shapes), painel lateral com lista, modo dedicado
    (Desenhar ↔ Editar), e load de máscara existente. Coords passam
    a ser armazenadas em [0..1] internamente — elimina inconsistência
    em resize do dialog e simplifica o load.

UX:
  - Toolbar topo: 2 botões de modo (Desenhar/Editar) + 3 ferramentas
    (só ativas em modo Desenhar) + Limpar.
  - Painel esquerdo: lista de shapes + botões 🗑 Apagar e 📋 Duplicar.
  - Canvas direito: imagem + shapes em preto semi-transparente; shape
    selecionado tem borda azul + handles azuis.
  - Modo Editar: clique numa forma seleciona; arrastar o corpo move;
    arrastar handle redimensiona (corner pra rect/elipse, vértice
    pra polígono).
  - Tecla Delete remove o shape selecionado.

Coordenadas internas em [0..1] da imagem, convertidas pra pixels
do widget só na hora de pintar/hit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.enums.occlusion_mask_shape import OcclusionMaskShape
from faceaudit.models.image_processing.occlusion_mask import (
    OcclusionMask,
    OcclusionMaskShapeDTO,
)


# Cores
_FINALIZED_FILL = QtGui.QColor(0, 0, 0, 160)
_FINALIZED_OUTLINE = QtGui.QColor(0, 0, 0, 220)
_SELECTED_OUTLINE = QtGui.QColor(74, 144, 217, 255)
_IN_PROGRESS_FILL = QtGui.QColor(220, 30, 30, 80)
_IN_PROGRESS_OUTLINE = QtGui.QColor(220, 30, 30, 200)
_HANDLE_FILL = QtGui.QColor(74, 144, 217, 255)
_HANDLE_BORDER = QtGui.QColor(255, 255, 255, 255)
_HANDLE_SIZE = 10  # px no widget

# Modos
_MODE_DRAW = "draw"
_MODE_EDIT = "edit"

# Tipo de manipulação ativa em modo EDIT
_DRAG_NONE = 0
_DRAG_BODY = 1
_DRAG_HANDLE = 2


@dataclass
class _Shape:
    """Shape mutável armazenado em coords [0..1] da imagem."""
    shape_type: OcclusionMaskShape
    points: List[Tuple[float, float]]


class _ImageCanvas(QtWidgets.QWidget):
    """Custom widget — renderiza imagem + shapes + captura mouse."""

    # Sinais pro dialog atualizar painel lateral
    shapes_changed = QtCore.pyqtSignal()       # add/remove/duplicate/clear
    selection_changed = QtCore.pyqtSignal(int)  # idx ou -1

    def __init__(self, pixmap: QtGui.QPixmap, parent=None):
        super().__init__(parent)
        self._pixmap = pixmap
        self._mode = _MODE_DRAW
        self._tool = OcclusionMaskShape.POLYGON

        # Shapes em coords [0..1] da imagem
        self._shapes: List[_Shape] = []
        self._selected_idx: int = -1

        # Estado de desenho (modo DRAW)
        self._poly_in_progress: List[Tuple[float, float]] = []
        self._rect_start: Optional[Tuple[float, float]] = None
        self._rect_current: Optional[Tuple[float, float]] = None

        # Estado de manipulação (modo EDIT)
        self._drag_kind: int = _DRAG_NONE
        self._drag_handle_idx: int = -1
        self._drag_last_norm: Optional[Tuple[float, float]] = None

        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    # ── API pública ────────────────────────────────────────────────────────

    def set_tool(self, tool: OcclusionMaskShape):
        self._tool = tool
        self._reset_in_progress()
        self.update()

    def set_mode(self, mode: str):
        if mode == self._mode:
            return
        self._mode = mode
        self._reset_in_progress()
        if mode == _MODE_DRAW:
            self._selected_idx = -1
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            self.selection_changed.emit(-1)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.update()

    def load_shapes(self, shapes: Tuple[OcclusionMaskShapeDTO, ...]):
        """Hidrata o canvas com shapes pré-existentes (em [0..1])."""
        self._shapes = [
            _Shape(
                shape_type=s.shape_type,
                points=[(float(x), float(y)) for x, y in s.points],
            )
            for s in shapes
        ]
        self._selected_idx = -1
        self._reset_in_progress()
        self.shapes_changed.emit()
        self.selection_changed.emit(-1)
        self.update()

    def get_mask_shapes(self) -> List[OcclusionMaskShapeDTO]:
        result: List[OcclusionMaskShapeDTO] = []
        for shape in self._shapes:
            pts = tuple(
                (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))
                for x, y in shape.points
            )
            result.append(OcclusionMaskShapeDTO(
                shape_type=shape.shape_type,
                points=pts,
            ))
        return result

    @property
    def has_shapes(self) -> bool:
        return len(self._shapes) > 0 or bool(self._poly_in_progress)

    @property
    def shape_types(self) -> List[OcclusionMaskShape]:
        return [s.shape_type for s in self._shapes]

    @property
    def selected_idx(self) -> int:
        return self._selected_idx

    def select_index(self, idx: int):
        new_idx = idx if 0 <= idx < len(self._shapes) else -1
        if new_idx == self._selected_idx:
            return
        self._selected_idx = new_idx
        self.selection_changed.emit(new_idx)
        self.update()

    def delete_selected(self):
        if not (0 <= self._selected_idx < len(self._shapes)):
            return
        del self._shapes[self._selected_idx]
        # Mantém seleção válida quando possível: o item logo abaixo
        # passa a ocupar o índice atual.
        if self._selected_idx >= len(self._shapes):
            self._selected_idx = len(self._shapes) - 1
        self.shapes_changed.emit()
        self.selection_changed.emit(self._selected_idx)
        self.update()

    def duplicate_selected(self):
        if not (0 <= self._selected_idx < len(self._shapes)):
            return
        original = self._shapes[self._selected_idx]
        offset = 0.03  # ~3% do lado da imagem
        new_pts = [
            (
                max(0.0, min(1.0, x + offset)),
                max(0.0, min(1.0, y + offset)),
            )
            for x, y in original.points
        ]
        copy = _Shape(shape_type=original.shape_type, points=new_pts)
        self._shapes.insert(self._selected_idx + 1, copy)
        self._selected_idx += 1
        self.shapes_changed.emit()
        self.selection_changed.emit(self._selected_idx)
        self.update()

    def clear_shapes(self):
        had_anything = bool(self._shapes) or bool(self._poly_in_progress)
        self._shapes = []
        self._selected_idx = -1
        self._reset_in_progress()
        if had_anything:
            self.shapes_changed.emit()
            self.selection_changed.emit(-1)
        self.update()

    # ── Conversões coord ────────────────────────────────────────────────────

    def _image_render_rect(self) -> Tuple[int, int, int, int]:
        widget_w, widget_h = self.width(), self.height()
        if self._pixmap.isNull():
            return (0, 0, 0, 0)
        pix_w = self._pixmap.width()
        pix_h = self._pixmap.height()
        scale = min(widget_w / pix_w, widget_h / pix_h) if pix_w and pix_h else 1
        rendered_w = int(pix_w * scale)
        rendered_h = int(pix_h * scale)
        x = (widget_w - rendered_w) // 2
        y = (widget_h - rendered_h) // 2
        return (x, y, rendered_w, rendered_h)

    def _widget_to_norm(self, x: int, y: int) -> Tuple[float, float]:
        ix, iy, iw, ih = self._image_render_rect()
        if iw <= 0 or ih <= 0:
            return (0.0, 0.0)
        return ((x - ix) / iw, (y - iy) / ih)

    def _norm_to_widget(self, nx: float, ny: float) -> Tuple[int, int]:
        ix, iy, iw, ih = self._image_render_rect()
        return (int(ix + nx * iw), int(iy + ny * ih))

    def _is_inside_image(self, x: int, y: int) -> bool:
        ix, iy, iw, ih = self._image_render_rect()
        return ix <= x <= ix + iw and iy <= y <= iy + ih

    @staticmethod
    def _clamp_norm(nx: float, ny: float) -> Tuple[float, float]:
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    # ── Paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#222"))

        ix, iy, iw, ih = self._image_render_rect()
        if iw > 0 and ih > 0:
            scaled = self._pixmap.scaled(
                iw, ih,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(ix, iy, scaled)

        # Shapes finalizados
        for idx, shape in enumerate(self._shapes):
            self._draw_shape(painter, shape, idx == self._selected_idx)

        # Em construção (apenas modo DRAW)
        if self._mode == _MODE_DRAW:
            painter.setBrush(QtGui.QBrush(_IN_PROGRESS_FILL))
            painter.setPen(QtGui.QPen(_IN_PROGRESS_OUTLINE, 2))
            if self._poly_in_progress:
                self._draw_polygon_in_progress(painter, self._poly_in_progress)
            if self._rect_start is not None and self._rect_current is not None:
                self._draw_rect_or_ellipse_preview(
                    painter, self._rect_start, self._rect_current, self._tool,
                )

    def _draw_shape(
        self,
        painter: QtGui.QPainter,
        shape: _Shape,
        is_selected: bool,
    ):
        pts_widget = [self._norm_to_widget(x, y) for x, y in shape.points]

        outline = (
            QtGui.QPen(_SELECTED_OUTLINE, 3)
            if is_selected
            else QtGui.QPen(_FINALIZED_OUTLINE, 2)
        )
        painter.setPen(outline)
        painter.setBrush(QtGui.QBrush(_FINALIZED_FILL))

        if shape.shape_type == OcclusionMaskShape.POLYGON and len(pts_widget) >= 3:
            polygon = QtGui.QPolygon([
                QtCore.QPoint(p[0], p[1]) for p in pts_widget
            ])
            painter.drawPolygon(polygon)
        elif shape.shape_type == OcclusionMaskShape.RECTANGLE and len(pts_widget) >= 2:
            rect = self._rect_from_two_points(pts_widget[0], pts_widget[1])
            painter.drawRect(rect)
        elif shape.shape_type == OcclusionMaskShape.ELLIPSE and len(pts_widget) >= 2:
            rect = self._rect_from_two_points(pts_widget[0], pts_widget[1])
            painter.drawEllipse(rect)

        # Handles do shape selecionado (apenas em modo EDIT)
        if is_selected and self._mode == _MODE_EDIT:
            handles = self._handle_positions(shape)
            painter.setPen(QtGui.QPen(_HANDLE_BORDER, 2))
            painter.setBrush(QtGui.QBrush(_HANDLE_FILL))
            half = _HANDLE_SIZE // 2
            for hx, hy in handles:
                painter.drawRect(
                    hx - half, hy - half, _HANDLE_SIZE, _HANDLE_SIZE,
                )

    def _draw_polygon_in_progress(self, painter, pts_norm):
        pts_widget = [self._norm_to_widget(x, y) for x, y in pts_norm]
        if len(pts_widget) >= 2:
            for i in range(len(pts_widget) - 1):
                painter.drawLine(
                    pts_widget[i][0], pts_widget[i][1],
                    pts_widget[i + 1][0], pts_widget[i + 1][1],
                )
        for x, y in pts_widget:
            painter.drawEllipse(QtCore.QPoint(x, y), 4, 4)

    def _draw_rect_or_ellipse_preview(self, painter, p1_norm, p2_norm, tool):
        p1 = self._norm_to_widget(*p1_norm)
        p2 = self._norm_to_widget(*p2_norm)
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

    def _handle_positions(self, shape: _Shape) -> List[Tuple[int, int]]:
        if shape.shape_type == OcclusionMaskShape.POLYGON:
            return [self._norm_to_widget(x, y) for x, y in shape.points]
        if len(shape.points) < 2:
            return []
        (x1, y1), (x2, y2) = shape.points[0], shape.points[1]
        nx_min, nx_max = (x1, x2) if x1 <= x2 else (x2, x1)
        ny_min, ny_max = (y1, y2) if y1 <= y2 else (y2, y1)
        # Ordem fixa: TL, TR, BR, BL
        corners_norm = [
            (nx_min, ny_min),
            (nx_max, ny_min),
            (nx_max, ny_max),
            (nx_min, ny_max),
        ]
        return [self._norm_to_widget(x, y) for x, y in corners_norm]

    # ── Hit testing ─────────────────────────────────────────────────────────

    def _hit_handle(self, shape: _Shape, x: int, y: int) -> int:
        handles = self._handle_positions(shape)
        # Tolerância um pouco maior que _HANDLE_SIZE pra clique fácil
        tol = _HANDLE_SIZE
        for i, (hx, hy) in enumerate(handles):
            if hx - tol <= x <= hx + tol and hy - tol <= y <= hy + tol:
                return i
        return -1

    def _hit_shape_body(self, x: int, y: int) -> int:
        # Itera de trás pra frente — última camada (topo) é prioritária
        for idx in range(len(self._shapes) - 1, -1, -1):
            if self._point_in_shape(self._shapes[idx], x, y):
                return idx
        return -1

    def _point_in_shape(self, shape: _Shape, x: int, y: int) -> bool:
        pts_widget = [self._norm_to_widget(px, py) for px, py in shape.points]
        if shape.shape_type == OcclusionMaskShape.POLYGON and len(pts_widget) >= 3:
            polygon = QtGui.QPolygon([
                QtCore.QPoint(p[0], p[1]) for p in pts_widget
            ])
            return polygon.containsPoint(
                QtCore.QPoint(x, y), QtCore.Qt.FillRule.OddEvenFill,
            )
        if len(pts_widget) >= 2:
            rect = self._rect_from_two_points(pts_widget[0], pts_widget[1])
            if shape.shape_type == OcclusionMaskShape.RECTANGLE:
                return rect.contains(x, y)
            if shape.shape_type == OcclusionMaskShape.ELLIPSE:
                cx, cy = rect.center().x(), rect.center().y()
                rx = rect.width() / 2 or 1
                ry = rect.height() / 2 or 1
                dx = (x - cx) / rx
                dy = (y - cy) / ry
                return (dx * dx + dy * dy) <= 1.0
        return False

    # ── Mouse events ────────────────────────────────────────────────────────

    def _reset_in_progress(self):
        self._poly_in_progress = []
        self._rect_start = None
        self._rect_current = None
        self._drag_kind = _DRAG_NONE
        self._drag_handle_idx = -1
        self._drag_last_norm = None

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())
        if not self._is_inside_image(x, y):
            return
        if self._mode == _MODE_EDIT:
            self._handle_press_edit(x, y)
        else:
            self._handle_press_draw(x, y)
        self.update()

    def _handle_press_draw(self, x: int, y: int):
        nx, ny = self._clamp_norm(*self._widget_to_norm(x, y))
        if self._tool == OcclusionMaskShape.POLYGON:
            self._poly_in_progress.append((nx, ny))
        elif self._tool in (
            OcclusionMaskShape.RECTANGLE,
            OcclusionMaskShape.ELLIPSE,
        ):
            self._rect_start = (nx, ny)
            self._rect_current = (nx, ny)

    def _handle_press_edit(self, x: int, y: int):
        # 1. Handle do shape selecionado (prioridade alta — handles
        # pequenos, hit-test estrito).
        if 0 <= self._selected_idx < len(self._shapes):
            sel = self._shapes[self._selected_idx]
            h_idx = self._hit_handle(sel, x, y)
            if h_idx >= 0:
                self._drag_kind = _DRAG_HANDLE
                self._drag_handle_idx = h_idx
                self._drag_last_norm = self._widget_to_norm(x, y)
                return
        # 2. Corpo de qualquer shape (de trás pra frente)
        body_idx = self._hit_shape_body(x, y)
        if body_idx >= 0:
            if body_idx != self._selected_idx:
                self._selected_idx = body_idx
                self.selection_changed.emit(body_idx)
            self._drag_kind = _DRAG_BODY
            self._drag_last_norm = self._widget_to_norm(x, y)
            return
        # 3. Vazio → desseleciona
        if self._selected_idx >= 0:
            self._selected_idx = -1
            self.selection_changed.emit(-1)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.position()
        x, y = int(pos.x()), int(pos.y())
        if self._mode == _MODE_DRAW:
            if self._rect_start is not None:
                self._rect_current = self._clamp_norm(*self._widget_to_norm(x, y))
                self.update()
            return
        # Modo EDIT
        if self._drag_kind == _DRAG_NONE:
            self._update_cursor_for_hover(x, y)
            return
        if not (0 <= self._selected_idx < len(self._shapes)):
            return
        nx, ny = self._clamp_norm(*self._widget_to_norm(x, y))
        if self._drag_last_norm is None:
            self._drag_last_norm = (nx, ny)
            return
        sel = self._shapes[self._selected_idx]
        if self._drag_kind == _DRAG_BODY:
            dx = nx - self._drag_last_norm[0]
            dy = ny - self._drag_last_norm[1]
            self._move_shape(sel, dx, dy)
        elif self._drag_kind == _DRAG_HANDLE:
            self._resize_shape(sel, self._drag_handle_idx, nx, ny)
        self._drag_last_norm = (nx, ny)
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._mode == _MODE_EDIT:
            self._drag_kind = _DRAG_NONE
            self._drag_handle_idx = -1
            self._drag_last_norm = None
            return
        # Modo DRAW: finaliza retângulo/elipse
        if self._rect_start is None or self._rect_current is None:
            return
        sx, sy = self._norm_to_widget(*self._rect_start)
        cx, cy = self._norm_to_widget(*self._rect_current)
        # Tolerância em pixels do widget
        if abs(cx - sx) > 4 and abs(cy - sy) > 4:
            self._shapes.append(_Shape(
                shape_type=self._tool,
                points=[self._rect_start, self._rect_current],
            ))
            self.shapes_changed.emit()
        self._rect_start = None
        self._rect_current = None
        self.update()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        if self._mode != _MODE_DRAW:
            return
        if self._tool != OcclusionMaskShape.POLYGON:
            return
        if len(self._poly_in_progress) >= 3:
            self._shapes.append(_Shape(
                shape_type=OcclusionMaskShape.POLYGON,
                points=list(self._poly_in_progress),
            ))
            self.shapes_changed.emit()
        self._poly_in_progress = []
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() in (QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace):
            if self._mode == _MODE_EDIT and self._selected_idx >= 0:
                self.delete_selected()
                return
        super().keyPressEvent(event)

    def _update_cursor_for_hover(self, x: int, y: int):
        if 0 <= self._selected_idx < len(self._shapes):
            sel = self._shapes[self._selected_idx]
            h_idx = self._hit_handle(sel, x, y)
            if h_idx >= 0:
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
                return
        body_idx = self._hit_shape_body(x, y)
        if body_idx >= 0:
            self.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    # ── Mutações ────────────────────────────────────────────────────────────

    @staticmethod
    def _move_shape(shape: _Shape, dx: float, dy: float):
        if not shape.points:
            return
        # Limita dx/dy pra que nenhum ponto saia de [0..1]
        min_x = min(p[0] for p in shape.points)
        max_x = max(p[0] for p in shape.points)
        min_y = min(p[1] for p in shape.points)
        max_y = max(p[1] for p in shape.points)
        if min_x + dx < 0:
            dx = -min_x
        if max_x + dx > 1:
            dx = 1 - max_x
        if min_y + dy < 0:
            dy = -min_y
        if max_y + dy > 1:
            dy = 1 - max_y
        shape.points = [(x + dx, y + dy) for x, y in shape.points]

    def _resize_shape(
        self, shape: _Shape, handle_idx: int, nx: float, ny: float,
    ):
        if shape.shape_type == OcclusionMaskShape.POLYGON:
            if 0 <= handle_idx < len(shape.points):
                shape.points[handle_idx] = (nx, ny)
            return
        if len(shape.points) < 2:
            return
        (x1, y1), (x2, y2) = shape.points[0], shape.points[1]
        nx_min, nx_max = (x1, x2) if x1 <= x2 else (x2, x1)
        ny_min, ny_max = (y1, y2) if y1 <= y2 else (y2, y1)
        if handle_idx == 0:    # TL
            nx_min, ny_min = nx, ny
        elif handle_idx == 1:  # TR
            nx_max, ny_min = nx, ny
        elif handle_idx == 2:  # BR
            nx_max, ny_max = nx, ny
        elif handle_idx == 3:  # BL
            nx_min, ny_max = nx, ny
        # Garante ordem normalizada (min, max) — evita inversão quando
        # o usuário arrasta o canto pra além do canto oposto.
        if nx_min > nx_max:
            nx_min, nx_max = nx_max, nx_min
        if ny_min > ny_max:
            ny_min, ny_max = ny_max, ny_min
        shape.points = [(nx_min, ny_min), (nx_max, ny_max)]


class OcclusionMaskCanvas(QtWidgets.QDialog):
    """Modal pra desenhar/editar máscara. Construa com `pixmap` da
    imagem de preview + opcional `initial_mask` (popula o canvas pra
    edição). Após `exec() == Accepted`, chame `get_mask()`."""

    def __init__(
        self,
        pixmap: QtGui.QPixmap,
        initial_mask: Optional[OcclusionMask] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Editar Máscara de Oclusão")
        self.setModal(True)
        self.resize(1100, 720)

        self._build_ui(pixmap)
        self._wire_signals()

        if initial_mask is not None and not initial_mask.is_empty:
            self._canvas.load_shapes(initial_mask.shapes)
        self._update_mode_ui()
        self._update_op_buttons()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self, pixmap: QtGui.QPixmap):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Toolbar superior
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_mode_draw = self._make_toggle_button(
            "✏️ Desenhar", "Adicionar novas formas à máscara.",
        )
        self.btn_mode_edit = self._make_toggle_button(
            "👆 Editar",
            "Selecionar formas existentes pra mover, redimensionar, "
            "duplicar ou apagar.",
        )
        self.btn_mode_draw.setChecked(True)
        toolbar.addWidget(self.btn_mode_draw)
        toolbar.addWidget(self.btn_mode_edit)

        # Separador visual
        sep1 = QtWidgets.QFrame()
        sep1.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep1.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        toolbar.addWidget(sep1)

        # Ferramentas (apenas modo Desenhar)
        self.btn_polygon = self._make_toggle_button(
            "🖊 Polígono",
            "Cliques sequenciais; double-click pra fechar.",
        )
        self.btn_rectangle = self._make_toggle_button(
            "⬛ Retângulo", "Drag (clique e arraste) pra desenhar.",
        )
        self.btn_ellipse = self._make_toggle_button(
            "⭕ Elipse", "Drag pra desenhar (bbox define a elipse).",
        )
        self.btn_polygon.setChecked(True)

        self._tool_buttons = {
            OcclusionMaskShape.POLYGON: self.btn_polygon,
            OcclusionMaskShape.RECTANGLE: self.btn_rectangle,
            OcclusionMaskShape.ELLIPSE: self.btn_ellipse,
        }
        toolbar.addWidget(self.btn_polygon)
        toolbar.addWidget(self.btn_rectangle)
        toolbar.addWidget(self.btn_ellipse)

        sep2 = QtWidgets.QFrame()
        sep2.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep2.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        toolbar.addWidget(sep2)

        self.btn_clear = QtWidgets.QPushButton("🗑 Limpar tudo")
        self.btn_clear.setToolTip("Apagar TODAS as formas desenhadas")
        toolbar.addWidget(self.btn_clear)

        toolbar.addStretch()

        self.help_lbl = QtWidgets.QLabel("")
        self.help_lbl.setStyleSheet("color: #666; font-size: 11px;")
        toolbar.addWidget(self.help_lbl)

        root.addLayout(toolbar)

        # Splitter horizontal: lista lateral + canvas
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(4)

        title = QtWidgets.QLabel("Formas")
        title.setStyleSheet("font-weight: bold; padding: 2px 0;")
        side_layout.addWidget(title)

        self._list = QtWidgets.QListWidget()
        self._list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        side_layout.addWidget(self._list, 1)

        ops_row = QtWidgets.QHBoxLayout()
        ops_row.setSpacing(4)
        self.btn_delete = QtWidgets.QPushButton("🗑 Apagar")
        self.btn_delete.setToolTip("Remove a forma selecionada (Delete)")
        self.btn_duplicate = QtWidgets.QPushButton("📋 Duplicar")
        self.btn_duplicate.setToolTip("Copia a forma selecionada com offset")
        ops_row.addWidget(self.btn_delete)
        ops_row.addWidget(self.btn_duplicate)
        side_layout.addLayout(ops_row)

        self._splitter.addWidget(side)

        self._canvas = _ImageCanvas(pixmap)
        self._splitter.addWidget(self._canvas)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([220, 880])

        root.addWidget(self._splitter, 1)

        # OK/Cancel
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("✓ Aplicar")
        root.addWidget(self.buttons)

    def _make_toggle_button(self, text: str, tooltip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setCheckable(True)
        btn.setStyleSheet(
            "QPushButton { padding: 6px 12px; border: 1px solid #ccc; "
            "border-radius: 4px; background: white; } "
            "QPushButton:checked { background: #cce4ff; "
            "border-color: #4a90d9; font-weight: bold; }"
        )
        return btn

    # ── Wiring ──────────────────────────────────────────────────────────────

    def _wire_signals(self):
        self.btn_mode_draw.clicked.connect(lambda: self._set_mode(_MODE_DRAW))
        self.btn_mode_edit.clicked.connect(lambda: self._set_mode(_MODE_EDIT))
        self.btn_polygon.clicked.connect(
            lambda: self._set_tool(OcclusionMaskShape.POLYGON)
        )
        self.btn_rectangle.clicked.connect(
            lambda: self._set_tool(OcclusionMaskShape.RECTANGLE)
        )
        self.btn_ellipse.clicked.connect(
            lambda: self._set_tool(OcclusionMaskShape.ELLIPSE)
        )
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_duplicate.clicked.connect(self._on_duplicate)

        self._list.currentRowChanged.connect(self._on_list_row_changed)
        self._canvas.shapes_changed.connect(self._refresh_list)
        self._canvas.selection_changed.connect(self._on_canvas_selection_changed)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    # ── Slots ───────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self._canvas.set_mode(mode)
        self.btn_mode_draw.setChecked(mode == _MODE_DRAW)
        self.btn_mode_edit.setChecked(mode == _MODE_EDIT)
        self._update_mode_ui()

    def _update_mode_ui(self):
        is_draw = self.btn_mode_draw.isChecked()
        for btn in (self.btn_polygon, self.btn_rectangle, self.btn_ellipse):
            btn.setEnabled(is_draw)
        if is_draw:
            self.help_lbl.setText(
                "<i>Polígono</i>: cliques + double-click finaliza.   "
                "<i>Retângulo / Elipse</i>: arrastar."
            )
        else:
            self.help_lbl.setText(
                "<i>Editar</i>: clique numa forma pra selecionar. "
                "Arraste o corpo pra mover, ou os handles azuis pra "
                "redimensionar. Tecla Delete apaga."
            )

    def _set_tool(self, tool: OcclusionMaskShape):
        self._canvas.set_tool(tool)
        for t, btn in self._tool_buttons.items():
            btn.setChecked(t == tool)

    def _on_clear(self):
        if not self._canvas.has_shapes:
            return
        res = QtWidgets.QMessageBox.question(
            self, "Limpar tudo",
            "Apagar todas as formas desenhadas?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if res == QtWidgets.QMessageBox.StandardButton.Yes:
            self._canvas.clear_shapes()

    def _on_delete(self):
        self._canvas.delete_selected()

    def _on_duplicate(self):
        self._canvas.duplicate_selected()

    def _refresh_list(self):
        types = self._canvas.shape_types
        self._list.blockSignals(True)
        self._list.clear()
        icon_map = {
            OcclusionMaskShape.POLYGON: "🖊",
            OcclusionMaskShape.RECTANGLE: "⬛",
            OcclusionMaskShape.ELLIPSE: "⭕",
        }
        for i, t in enumerate(types):
            self._list.addItem(
                f"{icon_map.get(t, '?')}  {t.label_pt} #{i + 1}"
            )
        cur = self._canvas.selected_idx
        if 0 <= cur < self._list.count():
            self._list.setCurrentRow(cur)
        self._list.blockSignals(False)
        self._update_op_buttons()

    def _on_list_row_changed(self, row: int):
        self._canvas.select_index(row)
        # Auto-switch pra modo Editar quando o usuário clica na lista
        # — sem isso, o shape selecionado não mostra handles.
        if row >= 0 and not self.btn_mode_edit.isChecked():
            self._set_mode(_MODE_EDIT)

    def _on_canvas_selection_changed(self, idx: int):
        self._list.blockSignals(True)
        if 0 <= idx < self._list.count():
            self._list.setCurrentRow(idx)
        else:
            self._list.setCurrentRow(-1)
        self._list.blockSignals(False)
        self._update_op_buttons()

    def _update_op_buttons(self):
        has_selection = self._canvas.selected_idx >= 0
        self.btn_delete.setEnabled(has_selection)
        self.btn_duplicate.setEnabled(has_selection)

    # ── API pública ────────────────────────────────────────────────────────

    def get_mask(self) -> OcclusionMask:
        """Devolve a máscara atual em coords [0..1]. Pode estar vazia."""
        shapes = self._canvas.get_mask_shapes()
        return OcclusionMask(shapes=tuple(shapes))
