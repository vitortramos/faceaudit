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
from typing import Any, Dict, List, Optional, Sequence

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet, current_token_fn


_THUMB_SIZE = 200            # lado da imagem em pixels
_CAPTION_HEIGHT = 36         # altura reservada pro caption multi-linha
_CELL_PADDING = 8            # padding interno do frame
_MAX_VISIBLE = 8             # thumbs visíveis antes de truncar com +N


# Estilos por estado da cell — combinação de selected/deleted/renamed.
# Cada combinação aplica fundo + borda. Um banner colorido extra (label
# sobre o thumb) reforça os estados deleted/renamed. Derivados de tokens
# do tema atual (lidos a cada `_restyle`), pra seguir claro/escuro.
def _thumb_cell_style_default(t) -> str:
    return (
        "QFrame { background: " + t("SURFACE_RAISED") + "; border: 1px solid "
        + t("BORDER") + "; border-radius: 4px; }"
    )


def _thumb_cell_style_selected(t) -> str:
    return (
        "QFrame { background: " + t("PRIMARY_BG") + "; border: 3px solid "
        + t("PRIMARY") + "; border-radius: 4px; }"
    )


def _thumb_cell_style_deleted(t) -> str:
    return (
        "QFrame { background: " + t("DANGER_BG") + "; border: 2px solid "
        + t("DANGER") + "; border-radius: 4px; }"
    )


def _thumb_cell_style_renamed(t) -> str:
    return (
        "QFrame { background: " + t("WARNING_BG") + "; border: 2px solid "
        + t("WARNING") + "; border-radius: 4px; }"
    )


class _ThumbCell(QtWidgets.QFrame):
    """Célula de thumbnail clicável com estado triplo (selected /
    deleted / renamed) + click multi-select.

    Click simples toggla a seleção da própria célula; Ctrl+Click toggla
    sem afetar as outras (multi-select aditivo); Shift+Click amplia
    desde a última âncora (range select). Click sem modificador limpa
    as outras e seleciona apenas esta.

    Cells `deleted` ficam visualmente marcadas em vermelho e ignoram
    cliques (não selecionáveis) — evita reaplicar uma ação sobre uma
    face que já foi excluída.
    """

    clicked = QtCore.pyqtSignal(object, QtCore.Qt.KeyboardModifier)
    double_clicked = QtCore.pyqtSignal(str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._selected = False
        self._deleted = False
        self._renamed: Optional[tuple] = None  # (old_id, new_id) quando aplicável
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(_thumb_cell_style_default(current_token_fn()))
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    @property
    def path(self) -> str:
        return self._path

    def is_selected(self) -> bool:
        return self._selected

    def is_deleted(self) -> bool:
        return self._deleted

    def set_selected(self, selected: bool) -> None:
        # Cells deletadas não são selecionáveis — feedback visual mantém
        # o estado "deleted" mesmo quando alguém tenta selecionar.
        if self._deleted:
            self._selected = False
            self._restyle()
            return
        if self._selected == selected:
            return
        self._selected = selected
        self._restyle()

    def set_deleted(self, deleted: bool) -> None:
        if self._deleted == deleted:
            return
        self._deleted = deleted
        if deleted:
            self._selected = False
            self.setCursor(QtCore.Qt.CursorShape.ForbiddenCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._restyle()

    def set_renamed(self, mapping: Optional[tuple]) -> None:
        """`mapping` é `(old_id, new_id)` quando o path foi renomeado;
        None desfaz a marcação."""
        self._renamed = mapping
        self._restyle()

    def _restyle(self) -> None:
        """Aplica o estilo de fundo conforme prioridade:
        deleted > selected > renamed > default. (deleted vence porque é
        terminal — selected/renamed não fazem sentido sobre algo que já
        foi excluído.)
        """
        t = current_token_fn()
        if self._deleted:
            self.setStyleSheet(_thumb_cell_style_deleted(t))
        elif self._selected:
            self.setStyleSheet(_thumb_cell_style_selected(t))
        elif self._renamed is not None:
            self.setStyleSheet(_thumb_cell_style_renamed(t))
        else:
            self.setStyleSheet(_thumb_cell_style_default(t))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(event)
        # Cells deletadas ignoram clique — evita ações redundantes.
        if self._deleted:
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(self, event.modifiers())

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self._path)


class IssuePreviewWidget(QtWidgets.QWidget):
    """Mostra thumbnails das faces afetadas por um issue selecionado."""

    # Emitido quando o usuário dá double-click em uma thumb (futura
    # expansão fullsize). Por enquanto só registramos o sinal.
    thumb_double_clicked = QtCore.pyqtSignal(str)
    # Emitido quando a seleção de thumbs muda (click, Ctrl+Click, etc).
    # Sem payload — caller chama `selected_paths()` pra ler a lista.
    selection_changed = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        # Cache: path → QPixmap (já carregado). Pixmaps "null" também são
        # armazenados pra evitar re-tentativa de leitura de paths quebrados.
        self._cache: Dict[str, QtGui.QPixmap] = {}
        # Dataset corrente — usado pra resolver paths em modo RELATIVE.
        # Setado por `set_dataset` antes de `set_paths`/`set_paired_groups`.
        self._dataset: Optional[Any] = None
        # Cells "vivas" no momento — usado pra seleção multi-cell e pra
        # limpar seleção ao trocar de issue. Reset a cada `set_paths` /
        # `set_paired_groups`.
        self._cells: List[_ThumbCell] = []
        # Última cell clicada — âncora pra Shift+Click range select.
        self._last_anchor: Optional[_ThumbCell] = None
        # Estado das ações pendentes — propagado pelo controller via
        # `set_active_marks`. Re-aplicado a cada repopulate de thumbs
        # pra que a marcação visual sobreviva à troca de issue.
        self._deleted_paths: set = set()
        self._renamed_paths: dict = {}  # path → (old_id, new_id)
        self._build_ui()
        self.set_paths([])

    # ── Marcações persistentes (deleted/renamed) ─────────────────────────

    def set_active_marks(self, deleted: set, renamed: dict) -> None:
        """Atualiza os sets/dicts globais de paths marcados como
        excluídos / renomeados. As cells visíveis recebem o estilo
        correspondente imediatamente; as cells criadas a partir daqui
        (ao trocar de issue) também herdam.

        Args:
            deleted: set de image_paths que foram removidos via
                "Excluir face(s)" e que ainda estão no histórico.
            renamed: dict `path → (old_identity, new_identity)` para
                paths cuja identidade mudou desde a abertura do dataset.
        """
        self._deleted_paths = set(deleted or [])
        self._renamed_paths = dict(renamed or {})
        # Re-aplica nas cells correntes
        for c in self._cells:
            self._apply_marks_to_cell(c)

    def _apply_marks_to_cell(self, cell: _ThumbCell) -> None:
        """Pinta a cell conforme o estado global. Chamado em
        `_make_thumb_cell` quando criamos thumbs novos e em
        `set_active_marks` pra cells já vivas."""
        path = cell.path
        cell.set_deleted(path in self._deleted_paths)
        cell.set_renamed(self._renamed_paths.get(path))
        # Atualiza o caption banner (se existir como child) sem rebuild.
        banner = cell.findChild(QtWidgets.QLabel, "_state_banner")
        if banner is not None:
            text = self._banner_text_for(path)
            banner.setText(text)
            banner.setVisible(bool(text))

    def _banner_text_for(self, path: str) -> str:
        """Texto que aparece sobre o thumb pra indicar deleted/renamed."""
        if path in self._deleted_paths:
            return "✗ EXCLUÍDA"
        mapping = self._renamed_paths.get(path)
        if mapping:
            old, new = mapping
            return f"✎ ID: {old} → {new}"
        return ""

    # ── Construção ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(2)

        self.lbl_header = QtWidgets.QLabel()
        apply_themed_stylesheet(
            self.lbl_header,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;",
        )
        outer.addWidget(self.lbl_header)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # set_paired_groups precisa de scroll vertical (2 linhas podem
        # extrapolar a altura disponível). set_paths usa só horizontal.
        # Default: AsNeeded — Qt decide.
        self.scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        # Container interno: VBox que pode acomodar uma única linha de
        # thumbs (set_paths) ou múltiplos grupos label+linha (set_paired_groups).
        self.scroll_content = QtWidgets.QWidget()
        self.outer_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.outer_layout.setSpacing(6)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # Linha de thumbs default (modo set_paths). Mantida pra retrocompat
        # — quando set_paired_groups é chamado, removemos esta linha.
        self.row = QtWidgets.QHBoxLayout()
        self.row.setSpacing(8)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._row_widget = QtWidgets.QWidget()
        self._row_widget.setLayout(self.row)
        self.outer_layout.addWidget(self._row_widget)

        self.scroll.setWidget(self.scroll_content)
        outer.addWidget(self.scroll, 1)

    # ── API pública ──────────────────────────────────────────────────────

    def set_paths(
        self,
        paths: Sequence[str],
        captions: Optional[Sequence[str]] = None,
    ) -> None:
        """Atualiza o widget com uma única linha de paths.

        `captions` opcional: lista paralela com texto curto pra mostrar
        abaixo de cada thumb (tipicamente identidade da face).
        """
        # Reseta o layout interno pro modo "linha única".
        self._reset_outer_layout(single_row_mode=True)

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

    def set_paired_groups(
        self,
        groups: Sequence[tuple],
        header_override: Optional[str] = None,
    ) -> None:
        """Mostra múltiplos grupos de thumbs, cada um com seu header.

        Args:
            groups: lista de tuplas `(label, paths, captions)`. Cada
                tupla vira: 1 QLabel header + 1 QHBoxLayout com os thumbs.
                Útil pra `audit_duplicate_identities`, onde cada issue
                cobre 2 identidades — uma linha por ID.
            header_override: texto custom pro header geral. Default
                conta `len(groups)` como "identidades", o que só está
                certo em duplicate_identities. Em mixed_identities os
                2 grupos pertencem à MESMA identidade (foto suspeita
                vs demais), então o caller passa o override correto.

        Os thumbs aqui NÃO são truncados em `_MAX_VISIBLE` — o usuário
        precisa ver todas as fotos pra decidir. Scroll horizontal por
        linha acomoda casos com muitas fotos.
        """
        self._reset_outer_layout(single_row_mode=False)

        if not groups:
            self.lbl_header.setText("(selecione uma issue na tabela acima)")
            return

        total = sum(len(paths) for (_lbl, paths, _caps) in groups)
        if header_override is not None:
            self.lbl_header.setText(header_override)
        else:
            self.lbl_header.setText(
                f"{total} face(s) em {len(groups)} identidade(s)"
            )

        for label, paths, captions in groups:
            # Header de grupo
            group_header = QtWidgets.QLabel(
                f"<b>{label}</b>  —  {len(paths)} foto(s)"
            )
            _gt = current_token_fn()
            group_header.setStyleSheet(
                "color: " + _gt("PRIMARY_STRONG") + "; font-size: 12px; "
                "padding: 4px 0 2px 0; background: transparent;"
            )
            self.outer_layout.addWidget(group_header)

            # Linha horizontal de thumbs com seu próprio scroll horizontal
            row_scroll = QtWidgets.QScrollArea()
            row_scroll.setWidgetResizable(True)
            row_scroll.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded
            )
            row_scroll.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            row_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            # Altura fixa pra UMA célula (thumb+caption+padding); o
            # scroll horizontal aparece quando passa.
            row_scroll.setFixedHeight(
                _THUMB_SIZE + _CAPTION_HEIGHT + 2 * _CELL_PADDING + 16
            )

            row_content = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_content)
            row_layout.setSpacing(8)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)

            cap_list = list(captions or [])
            for i, path in enumerate(paths):
                cap = cap_list[i] if i < len(cap_list) else ""
                cell = self._make_thumb_cell(path, cap)
                row_layout.addWidget(cell)
            row_layout.addStretch()

            row_scroll.setWidget(row_content)
            self.outer_layout.addWidget(row_scroll)

    def _reset_outer_layout(self, single_row_mode: bool) -> None:
        """Limpa o layout interno e prepara pro modo escolhido.

        - `single_row_mode=True`: deixa só `self._row_widget` no layout
          (limpa só os thumbs antigos dentro de `self.row`).
        - `single_row_mode=False`: remove TODOS os widgets do
          `outer_layout` (incluindo `self._row_widget`) — em seguida o
          caller adiciona seus próprios grupos.

        Em ambos os modos o tracker de cells (`self._cells`) é resetado
        — as referências antigas são pra cells que foram (ou serão)
        deletadas via `deleteLater()`.
        """
        self._cells.clear()
        self._last_anchor = None
        if single_row_mode:
            # Limpa células dentro de self.row mas preserva self._row_widget
            # como container — outros widgets (grupos antigos) somem.
            while self.row.count():
                item = self.row.takeAt(0)
                w = item.widget() if item else None
                if w is not None:
                    w.deleteLater()
            # Remove widgets adicionais de grupos anteriores
            for i in reversed(range(self.outer_layout.count())):
                item = self.outer_layout.itemAt(i)
                w = item.widget() if item else None
                if w is not None and w is not self._row_widget:
                    self.outer_layout.removeWidget(w)
                    w.deleteLater()
            # Garante que _row_widget está visível
            self._row_widget.show()
        else:
            # Modo agrupado — esconde a linha default e remove qualquer
            # widget de grupos anteriores.
            while self.row.count():
                item = self.row.takeAt(0)
                w = item.widget() if item else None
                if w is not None:
                    w.deleteLater()
            self._row_widget.hide()
            for i in reversed(range(self.outer_layout.count())):
                item = self.outer_layout.itemAt(i)
                w = item.widget() if item else None
                if w is not None and w is not self._row_widget:
                    self.outer_layout.removeWidget(w)
                    w.deleteLater()

    def clear_cache(self) -> None:
        """Esvazia o cache de pixmaps. Chamado pelo controller quando o
        dataset muda (ex: várias faces foram excluídas)."""
        self._cache.clear()

    # ── Helpers ──────────────────────────────────────────────────────────

    # ── Seleção pública ──────────────────────────────────────────────────

    def selected_paths(self) -> List[str]:
        """Lista de paths selecionados (em ordem visual)."""
        return [c.path for c in self._cells if c.is_selected() and c.path]

    def clear_selection(self) -> None:
        """Desseleciona todas as cells. Não emite sinal — caller decide."""
        for c in self._cells:
            c.set_selected(False)
        self._last_anchor = None

    def select_all(self) -> None:
        """Seleciona todas as cells exceto as marcadas como excluídas
        (não faz sentido selecioná-las pra ações que vão atuar de novo
        sobre elas). Emite `selection_changed`."""
        for c in self._cells:
            if c.is_deleted():
                c.set_selected(False)
            else:
                c.set_selected(True)
        # Âncora vira a primeira cell não-deletada (ou None).
        self._last_anchor = next(
            (c for c in self._cells if not c.is_deleted()), None
        )
        self.selection_changed.emit()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_thumb_cell(self, path: str, caption: str) -> _ThumbCell:
        cell = _ThumbCell(path)
        cell.setFixedSize(
            _THUMB_SIZE + 2 * _CELL_PADDING,
            _THUMB_SIZE + _CAPTION_HEIGHT + 2 * _CELL_PADDING,
        )
        cell.setToolTip(path or "(sem path)")

        v = QtWidgets.QVBoxLayout(cell)
        v.setContentsMargins(_CELL_PADDING, _CELL_PADDING, _CELL_PADDING, _CELL_PADDING)
        v.setSpacing(4)

        t = current_token_fn()
        lbl_img = QtWidgets.QLabel()
        lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_img.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        lbl_img.setStyleSheet(
            "background: " + t("SURFACE_ALT") + "; border: 1px solid "
            + t("BORDER") + "; border-radius: 2px;"
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
                "background: " + t("WARNING_BG") + "; border: 1px solid "
                + t("WARNING") + "; color: " + t("DANGER") + "; "
                "font-size: 11px; border-radius: 2px;"
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
            "color: " + t("TEXT") + "; font-size: 10px; "
            "background: transparent; border: none;"
        )
        lbl_cap.setWordWrap(True)
        lbl_cap.setFixedHeight(_CAPTION_HEIGHT)
        v.addWidget(lbl_cap)

        # Banner de estado — fica entre o thumb e o caption. Hidden quando
        # a face não foi excluída/renomeada; ganha texto + cor via
        # `_apply_marks_to_cell` baseado nos sets do widget.
        state_banner = QtWidgets.QLabel("")
        state_banner.setObjectName("_state_banner")
        state_banner.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        state_banner.setStyleSheet(
            "color: " + t("HIGHLIGHT_TEXT") + "; background: " + t("DANGER") + "; "
            "border-radius: 2px; font-size: 10px; font-weight: bold; "
            "padding: 2px;"
        )
        state_banner.setVisible(False)
        v.addWidget(state_banner)

        # Sinais: click toggla seleção; double-click reservado pra fullsize.
        cell.clicked.connect(self._on_cell_clicked)
        cell.double_clicked.connect(self.thumb_double_clicked)

        # Registra no tracker de cells vivas pra seleção/limpeza.
        self._cells.append(cell)
        # Aplica marca persistente (deleted/renamed) se este path tem
        # uma ação no histórico ativo.
        self._apply_marks_to_cell(cell)
        # Banner colorido conforme estado
        if path in self._renamed_paths and path not in self._deleted_paths:
            state_banner.setStyleSheet(
                "color: " + t("ACCENT_TEXT") + "; background: " + t("WARNING_BG") + "; "
                "border: 1px solid " + t("WARNING") + "; "
                "border-radius: 2px; font-size: 10px; font-weight: bold; "
                "padding: 2px;"
            )
        return cell

    def _on_cell_clicked(self, cell: _ThumbCell, modifiers) -> None:
        """Resolve o clique conforme modifiers:
        - sem modifier: substitui seleção (limpa outras, seleciona esta)
        - Ctrl: toggla a célula clicada (multi-select aditivo)
        - Shift: range desde a última âncora até a clicada (inclusive)
        """
        ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

        if shift and self._last_anchor is not None and self._last_anchor in self._cells:
            # Range select — não muda âncora.
            try:
                a = self._cells.index(self._last_anchor)
                b = self._cells.index(cell)
            except ValueError:
                a = b = self._cells.index(cell)
            lo, hi = min(a, b), max(a, b)
            for i, c in enumerate(self._cells):
                c.set_selected(lo <= i <= hi)
        elif ctrl:
            cell.set_selected(not cell.is_selected())
            self._last_anchor = cell
        else:
            # Click simples: limpa outras, seleciona só esta.
            for c in self._cells:
                c.set_selected(c is cell)
            self._last_anchor = cell

        self.selection_changed.emit()

    def set_dataset(self, dataset: Optional[Any]) -> None:
        """Recebe dataset corrente — usado pra resolver paths em modo
        RELATIVE antes de carregar QPixmap. Cache anterior é dropado
        porque chaves (paths absolutos) mudam entre datasets."""
        self._dataset = dataset
        self._cache.clear()

    def _load_pixmap(self, path: str) -> Optional[QtGui.QPixmap]:
        """Carrega QPixmap com cache. Retorna None se path falhar.

        `path` pode ser relativo (modo RELATIVE) — resolve via dataset
        antes da leitura. Cache key é o path RAW pra que `_deleted_paths`
        e `_renamed_paths` sigam usando o mesmo identificador lógico.
        """
        if not path:
            return None
        cached = self._cache.get(path)
        if cached is not None:
            return cached if not cached.isNull() else None

        resolved = (
            str(self._dataset.resolve_image_path(path))
            if self._dataset is not None
            else path
        )
        pix = QtGui.QPixmap(resolved)
        # Cacheia mesmo o null pra evitar re-tentativa
        self._cache[path] = pix
        return pix if not pix.isNull() else None
