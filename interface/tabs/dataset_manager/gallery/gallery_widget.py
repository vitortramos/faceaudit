"""Modo galeria do Dataset Manager — grid de thumbnails clicáveis.

Substitui a tabela quando o usuário aciona o toggle na toolbar. Cada
item exibe miniatura da face + caption (identidade + nome do arquivo).
Selecionar um item emite `face_selected(int)` com o índice da face no
dataset (mesma semântica do detail panel pra clicks na tabela).

Lazy loading via QTimer batched: ao popular, todos os items entram com
ícone placeholder e o widget agenda carregamentos em batches de
`_LOAD_BATCH_SIZE` items por tick. Cache `Dict[path, QPixmap]` evita
re-leitura quando o usuário re-popula o mesmo dataset (ex: re-render
após filtro). Cache é dropado em `populate(...)` quando o dataset muda.

Performance: pra datasets de algumas dezenas de milhares de faces, o
QListWidget em IconMode renderiza apenas os items visíveis na viewport
(virtualização nativa do Qt). O custo principal é o carregamento I/O
das thumbnails, mitigado pelo batching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet


_THUMB_SIZE = 96            # lado em pixels do ícone exibido
_LOAD_BATCH_SIZE = 12       # items carregados por tick do timer
_LOAD_TICK_INTERVAL_MS = 30 # ~33fps; suaviza UI sem travar


class GalleryWidget(QtWidgets.QWidget):
    """Visualização em grid de thumbnails das faces do dataset."""

    # face_idx (0-based) clicado pelo usuário. Mesma assinatura que o
    # `face_selected` da tabela usa internamente — facilita sync.
    face_selected = QtCore.pyqtSignal(int)
    # Double-click — reservado pra futuro "abrir comparador" ou similar.
    face_double_clicked = QtCore.pyqtSignal(int)
    # PR 15.2: usuário pediu comparação de par via context menu, com
    # exatamente 2 thumbs selecionadas.
    compare_pair_requested = QtCore.pyqtSignal(int, int)
    # Ações de seleção (N faces) via context menu. Emitem a lista de
    # face_idx (índices em dataset.faces) — o controller faz a mutação.
    remove_faces_requested = QtCore.pyqtSignal(list)
    create_dataset_from_selection_requested = QtCore.pyqtSignal(list)
    set_column_value_requested = QtCore.pyqtSignal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap_cache: Dict[str, QtGui.QPixmap] = {}
        self._faces_ref: List = []  # faces atuais (para lookup por idx)
        self._pending_load: List[int] = []  # face_idx pendentes
        # Dataset corrente — usado pra resolver paths em modo RELATIVE.
        # Setado por `set_dataset` antes de cada `populate`.
        self._dataset: Optional[Any] = None
        self._build_ui()

        # Timer batched pra carregar thumbs em background sem freezar UI.
        # Singleshot=False, intervalo curto, processa _LOAD_BATCH_SIZE items
        # por tick. Auto-stop quando _pending_load esvazia.
        self._load_timer = QtCore.QTimer(self)
        self._load_timer.setInterval(_LOAD_TICK_INTERVAL_MS)
        self._load_timer.timeout.connect(self._load_next_batch)

    # ── UI ────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        # Header com contador (cresce dinamicamente conforme dataset)
        self.lbl_header = QtWidgets.QLabel("(nenhum dataset carregado)")
        apply_themed_stylesheet(
            self.lbl_header,
            lambda t: "padding: 6px 8px; color: " + t("TEXT_MUTED") + "; font-size: 11px;",
        )
        outer.addWidget(self.lbl_header)

        # QListWidget em IconMode — virtualização nativa do Qt
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.list_widget.setIconSize(QtCore.QSize(_THUMB_SIZE, _THUMB_SIZE))
        self.list_widget.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QtWidgets.QListView.Movement.Static)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setSpacing(8)
        self.list_widget.setWordWrap(True)
        self.list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        # PR 15.2: context menu com "Comparar par" (gated por 2 selecionados)
        self.list_widget.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.list_widget.customContextMenuRequested.connect(self._open_context_menu)
        outer.addWidget(self.list_widget, 1)

    # ── API pública ───────────────────────────────────────────────────

    def populate(self, faces: Sequence) -> None:
        """Recarrega a galeria com a lista de faces fornecida.

        Cada face vira um QListWidgetItem com placeholder + caption
        (identidade + filename). O carregamento real das thumbnails é
        agendado pelo timer batched — items "vazios" aparecem
        instantaneamente e ganham imagem aos poucos.
        """
        self._load_timer.stop()
        self._pending_load.clear()
        self.list_widget.clear()
        self._faces_ref = list(faces)

        if not self._faces_ref:
            self.lbl_header.setText("(dataset vazio)")
            return

        n = len(self._faces_ref)
        self.lbl_header.setText(f"{n} face(s)")

        placeholder = self._make_placeholder_pixmap()
        for idx, face in enumerate(self._faces_ref):
            caption = _format_caption(face)
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(placeholder), caption)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, idx)
            tooltip = getattr(face, "image_path", "") or ""
            item.setToolTip(tooltip)
            item.setSizeHint(QtCore.QSize(_THUMB_SIZE + 32, _THUMB_SIZE + 48))
            self.list_widget.addItem(item)
            self._pending_load.append(idx)

        # Inicia o batched-load
        self._load_timer.start()

    def select_face_idx(self, idx: Optional[int]) -> None:
        """Seleciona o item correspondente ao `face_idx` (sem disparar sinal).

        Chamado pelo controller pra sincronizar seleção tabela ↔ galeria.
        Quando `idx is None`, limpa a seleção.
        """
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clearSelection()
            if idx is not None and 0 <= idx < self.list_widget.count():
                item = self.list_widget.item(idx)
                if item is not None:
                    item.setSelected(True)
                    self.list_widget.scrollToItem(
                        item, QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
                    )
        finally:
            self.list_widget.blockSignals(False)

    def clear_cache(self) -> None:
        """Esvazia o cache de pixmaps (útil quando muitas faces foram
        excluídas ou o dataset trocou)."""
        self._pixmap_cache.clear()

    def set_dataset(self, dataset: Optional[Any]) -> None:
        """Recebe o dataset corrente pra resolver paths em modo RELATIVE.
        Chame antes de `populate(...)` quando o dataset trocar.
        """
        self._dataset = dataset
        # Cache anterior pode ter chaves de outro dataset — invalida.
        self._pixmap_cache.clear()

    # ── Lazy load ─────────────────────────────────────────────────────

    def _load_next_batch(self) -> None:
        """Tick do timer: processa até `_LOAD_BATCH_SIZE` items pendentes."""
        if not self._pending_load:
            self._load_timer.stop()
            return

        # Pega os próximos N e tenta carregar
        batch = self._pending_load[:_LOAD_BATCH_SIZE]
        self._pending_load = self._pending_load[_LOAD_BATCH_SIZE:]

        for idx in batch:
            if idx >= self.list_widget.count():
                continue  # populate sobrescreveu durante o load
            face = self._faces_ref[idx] if idx < len(self._faces_ref) else None
            if face is None:
                continue
            raw = getattr(face, "image_path", None)
            if not raw:
                continue
            resolved = (
                str(self._dataset.resolve_image_path(face))
                if self._dataset is not None
                else raw
            )
            pix = self._get_or_load_pixmap(resolved)
            if pix is not None and not pix.isNull():
                item = self.list_widget.item(idx)
                if item is not None:
                    item.setIcon(QtGui.QIcon(pix))

        if not self._pending_load:
            self._load_timer.stop()

    def _get_or_load_pixmap(self, path: str) -> Optional[QtGui.QPixmap]:
        """Carrega QPixmap com cache. `path` deve ser absoluto (caller
        resolve via `dataset.resolve_image_path` em modo RELATIVE)."""
        cached = self._pixmap_cache.get(path)
        if cached is not None:
            return cached if not cached.isNull() else None

        pix = QtGui.QPixmap(path)
        # Cacheia mesmo o null pra evitar re-tentativa
        self._pixmap_cache[path] = pix
        return pix if not pix.isNull() else None

    @staticmethod
    def _make_placeholder_pixmap() -> QtGui.QPixmap:
        """Pixmap cinza com '⋯' que vira ícone enquanto a real não chega."""
        pix = QtGui.QPixmap(_THUMB_SIZE, _THUMB_SIZE)
        pix.fill(QtGui.QColor("#e0e0e0"))
        painter = QtGui.QPainter(pix)
        painter.setPen(QtGui.QColor("#888"))
        font = painter.font()
        font.setPointSize(20)
        painter.setFont(font)
        painter.drawText(
            pix.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "⋯",
        )
        painter.end()
        return pix

    # ── Slots ─────────────────────────────────────────────────────────

    def _on_selection_changed(self) -> None:
        """Emite face_idx do item selecionado (último, em multi-seleção)."""
        items = self.list_widget.selectedItems()
        if not items:
            return
        # Em multi-seleção emitimos o "current" pra detail panel mostrar
        # a face mais recentemente clicada — semântica idêntica à tabela.
        current = self.list_widget.currentItem() or items[-1]
        idx = current.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(idx, int):
            self.face_selected.emit(idx)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        idx = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(idx, int):
            self.face_double_clicked.emit(idx)

    # ── Context menu ──────────────────────────────────────────────────

    def _open_context_menu(self, position) -> None:
        menu = QtWidgets.QMenu(self.list_widget)
        self._populate_context_menu(menu)
        menu.exec(self.list_widget.viewport().mapToGlobal(position))

    def selected_indexes(self) -> list:
        """Índices (em `dataset.faces`) dos itens selecionados, ordenados.

        Público porque a toolbar também precisa da seleção — antes só o
        menu de contexto a conhecia, por dentro.
        """
        return sorted({
            it.data(QtCore.Qt.ItemDataRole.UserRole)
            for it in self.list_widget.selectedItems()
            if isinstance(it.data(QtCore.Qt.ItemDataRole.UserRole), int)
        })

    def _populate_context_menu(self, menu) -> None:
        """Popula o menu de contexto da galeria. Ações por seleção (N faces):
        comparar par (exatamente 2), remover, definir valor de coluna e criar
        dataset a partir da seleção. Cada ação emite um sinal; o controller faz
        a mutação sobre `dataset.faces` (cujo índice é o que cada item carrega).
        Separado do exec() para ser testável sem bloquear no event loop."""
        selected_idxs = self.selected_indexes()
        n = len(selected_idxs)

        # ── Comparar par (exatamente 2) ──
        if n == 2:
            a, b = selected_idxs
            act = menu.addAction("⚖️ Comparar este par lado a lado…")
            act.triggered.connect(
                lambda _checked=False, ia=a, ib=b:
                    self.compare_pair_requested.emit(ia, ib)
            )
        else:
            act = menu.addAction(f"⚖️ Comparar par (selecione 2 — {n})")
            act.setEnabled(False)

        menu.addSeparator()

        # ── Ações sobre a seleção (N ≥ 1) ──
        act_setval = menu.addAction(f"✏️ Definir valor de coluna… ({n})")
        act_setval.setEnabled(n >= 1)
        act_setval.triggered.connect(
            lambda _checked=False, idxs=list(selected_idxs):
                self.set_column_value_requested.emit(idxs)
        )

        act_subset = menu.addAction(f"🧩 Criar dataset a partir da seleção… ({n})")
        act_subset.setEnabled(n >= 1)
        act_subset.triggered.connect(
            lambda _checked=False, idxs=list(selected_idxs):
                self.create_dataset_from_selection_requested.emit(idxs)
        )

        menu.addSeparator()

        act_remove = menu.addAction(f"🗑️ Remover do dataset ({n})")
        act_remove.setEnabled(n >= 1)
        act_remove.triggered.connect(
            lambda _checked=False, idxs=list(selected_idxs):
                self.remove_faces_requested.emit(idxs)
        )


# ── Helpers livres ────────────────────────────────────────────────────

def _format_caption(face) -> str:
    """Caption multi-linha pro item da galeria.

    Linha 1: identity ou (sem ID); linha 2: filename. O QListWidget em
    IconMode quebra automaticamente baseado no setSizeHint.
    """
    identity = getattr(face, "identity", None) or "(sem ID)"
    img_path = getattr(face, "image_path", None) or ""
    file_name = Path(img_path).name if img_path else ""
    if file_name and len(file_name) > 22:
        file_name = file_name[:19] + "…"
    return f"{identity}\n{file_name}" if file_name else str(identity)
