"""Widget que mostra thumbnails das faces afetadas por um `AuditIssue`.

Layout: linha horizontal de thumbs (com scroll horizontal quando passa
da largura disponível). Cada thumb é uma célula 200×~250 com a imagem
em cima e caption (filename + identidade quando conhecida) embaixo.
QPixmap é cacheado por path no próprio widget — trocar de issue volta
pra cache em vez de reler do disco.

Casos cobertos:
  - 1 path → 1 thumb
  - 2 paths → 2 thumbs lado a lado (caso clássico: hash conflict, hard
    pair, identidade duplicada)
  - 3+ paths → todos numa scroll-row, capped em `_MAX_VISIBLE` com texto
    "+N ocultas" no header
  - path inexistente → placeholder "⚠️ arquivo não encontrado"

Reuso esperado em PR 13/14: dialog de identidades duplicadas/misturadas
+ preview pré-rename/merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PyQt6 import QtCore, QtGui, QtWidgets


_THUMB_SIZE = 200            # lado da imagem em pixels
_CAPTION_HEIGHT = 36         # altura reservada pro caption multi-linha
_CELL_PADDING = 8            # padding interno do frame
_MAX_VISIBLE = 8             # thumbs visíveis antes de truncar com +N


class IssuePreviewWidget(QtWidgets.QWidget):
    """Mostra thumbnails das faces afetadas por um issue selecionado."""

    # Emitido quando o usuário dá double-click em uma thumb (futura
    # expansão fullsize). Por enquanto só registramos o sinal.
    thumb_double_clicked = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        # Cache: path → QPixmap (já carregado). Pixmaps "null" também são
        # armazenados pra evitar re-tentativa de leitura de paths quebrados.
        self._cache: Dict[str, QtGui.QPixmap] = {}
        self._build_ui()
        self.set_paths([])

    # ── Construção ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(2)

        self.lbl_header = QtWidgets.QLabel()
        self.lbl_header.setStyleSheet("color: #666; font-size: 11px;")
        outer.addWidget(self.lbl_header)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.scroll_content = QtWidgets.QWidget()
        self.row = QtWidgets.QHBoxLayout(self.scroll_content)
        self.row.setSpacing(8)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

        self.scroll.setWidget(self.scroll_content)
        outer.addWidget(self.scroll, 1)

    # ── API pública ──────────────────────────────────────────────────────

    def set_paths(
        self,
        paths: Sequence[str],
        captions: Optional[Sequence[str]] = None,
    ) -> None:
        """Atualiza o widget com a lista de paths.

        `captions` opcional: lista paralela com texto curto pra mostrar
        abaixo de cada thumb (tipicamente identidade da face).
        """
        # Limpa células anteriores
        while self.row.count():
            item = self.row.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                w.deleteLater()

        if not paths:
            self.lbl_header.setText("(selecione uma issue na tabela acima)")
            return

        total = len(paths)
        truncated = total > _MAX_VISIBLE
        visible = list(paths[:_MAX_VISIBLE])
        cap_list = list(captions or [])

        header_text = f"{total} face(s) na issue"
        if truncated:
            header_text += (
                f"  —  exibindo {len(visible)}; +{total - _MAX_VISIBLE} ocultas"
            )
        self.lbl_header.setText(header_text)

        for i, path in enumerate(visible):
            cap = cap_list[i] if i < len(cap_list) else ""
            cell = self._make_thumb_cell(path, cap)
            self.row.addWidget(cell)

        # Stretch no fim pra empurrar tudo pra esquerda
        self.row.addStretch()

    def clear_cache(self) -> None:
        """Esvazia o cache de pixmaps. Chamado pelo controller quando o
        dataset muda (ex: várias faces foram excluídas)."""
        self._cache.clear()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_thumb_cell(self, path: str, caption: str) -> QtWidgets.QWidget:
        cell = QtWidgets.QFrame()
        cell.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        cell.setStyleSheet(
            "QFrame { background: #fafafa; border: 1px solid #ddd; "
            "border-radius: 4px; }"
        )
        cell.setFixedSize(
            _THUMB_SIZE + 2 * _CELL_PADDING,
            _THUMB_SIZE + _CAPTION_HEIGHT + 2 * _CELL_PADDING,
        )
        cell.setToolTip(path or "(sem path)")

        v = QtWidgets.QVBoxLayout(cell)
        v.setContentsMargins(_CELL_PADDING, _CELL_PADDING, _CELL_PADDING, _CELL_PADDING)
        v.setSpacing(4)

        lbl_img = QtWidgets.QLabel()
        lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_img.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        lbl_img.setStyleSheet(
            "background: #f0f0f0; border: 1px solid #e0e0e0; border-radius: 2px;"
        )

        pix = self._load_pixmap(path)
        if pix is not None:
            scaled = pix.scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            lbl_img.setPixmap(scaled)
        else:
            # Path quebrado ou sem decode — placeholder visual claro
            lbl_img.setText("⚠️ arquivo\nnão encontrado")
            lbl_img.setStyleSheet(
                "background: #fff3e0; border: 1px solid #ffb74d; "
                "color: #c62828; font-size: 11px; border-radius: 2px;"
            )
        v.addWidget(lbl_img)

        # Caption: filename + (identidade, se houver)
        path_short = Path(path).name if path else "(sem path)"
        text = path_short
        if caption:
            text += f"\n{caption}"
        lbl_cap = QtWidgets.QLabel(text)
        lbl_cap.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_cap.setStyleSheet(
            "color: #333; font-size: 10px; background: transparent; border: none;"
        )
        lbl_cap.setWordWrap(True)
        lbl_cap.setFixedHeight(_CAPTION_HEIGHT)
        v.addWidget(lbl_cap)

        # Double-click → emite sinal pra futuro fullsize
        cell.mouseDoubleClickEvent = (
            lambda _ev, p=path: self.thumb_double_clicked.emit(p)
        )

        return cell

    def _load_pixmap(self, path: str) -> Optional[QtGui.QPixmap]:
        """Carrega QPixmap com cache. Retorna None se path falhar."""
        if not path:
            return None
        cached = self._cache.get(path)
        if cached is not None:
            return cached if not cached.isNull() else None

        pix = QtGui.QPixmap(path)
        # Cacheia mesmo o null pra evitar re-tentativa
        self._cache[path] = pix
        return pix if not pix.isNull() else None
