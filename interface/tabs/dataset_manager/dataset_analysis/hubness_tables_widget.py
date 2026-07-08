"""Widget pra inspeção tabular dos extremos de hubness.

Renderiza duas tabelas QTableWidget empilhadas — top hubs (cauda
direita do Nk) e top anti-hubs (Nk=0 ou menor) — com colunas
Identidade / Foto / FIQA / Norma / Nk. Ao selecionar uma linha de
qualquer tabela, a imagem da face é carregada num preview embaixo.

Reusa o pattern de tabela+preview que o resto da auditoria já usa,
mas adaptado pra mostrar duas listagens lado a lado pro perito ler
hubs vs anti-hubs num só relance.
"""

from __future__ import annotations

from typing import Any, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet


class HubnessTablesWidget(QtWidgets.QWidget):
    """Painel com duas tabelas + preview de imagem único."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # Dataset corrente — usado pra resolver paths em modo RELATIVE
        # antes de armazenar no UserRole das células (consumido pelo
        # `_load_preview` ao clicar). Set por `set_dataset`.
        self._dataset = None

        # Splitter vertical externo: tabelas em cima (lado a lado),
        # preview embaixo. Splitter horizontal interno divide as duas
        # tabelas — usuário pode arrastar pra ajustar.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)

        v_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        outer.addWidget(v_splitter, 1)

        # ── Linha de tabelas (lado a lado) ──
        tables_h_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Tabela hubs (esquerda)
        hubs_frame = QtWidgets.QFrame()
        hubs_layout = QtWidgets.QVBoxLayout(hubs_frame)
        hubs_layout.setContentsMargins(0, 0, 0, 0)
        hubs_layout.setSpacing(2)
        lbl_hubs = QtWidgets.QLabel(
            "<b>Top hubs</b> — Nk alto (atraem vizinhança errada)"
        )
        apply_themed_stylesheet(
            lbl_hubs,
            lambda t: "color: " + t("DANGER") + "; font-size: 11px;",
        )
        hubs_layout.addWidget(lbl_hubs)
        self.table_hubs = self._make_table()
        hubs_layout.addWidget(self.table_hubs)
        tables_h_splitter.addWidget(hubs_frame)

        # Tabela anti-hubs (direita)
        anti_frame = QtWidgets.QFrame()
        anti_layout = QtWidgets.QVBoxLayout(anti_frame)
        anti_layout.setContentsMargins(0, 0, 0, 0)
        anti_layout.setSpacing(2)
        lbl_anti = QtWidgets.QLabel(
            "<b>Top anti-hubs</b> — Nk baixo (faces topologicamente isoladas)"
        )
        apply_themed_stylesheet(
            lbl_anti,
            lambda t: "color: " + t("SUCCESS") + "; font-size: 11px;",
        )
        anti_layout.addWidget(lbl_anti)
        self.table_anti = self._make_table()
        anti_layout.addWidget(self.table_anti)
        tables_h_splitter.addWidget(anti_frame)

        # Divisão equânime entre as duas tabelas
        tables_h_splitter.setSizes([500, 500])
        v_splitter.addWidget(tables_h_splitter)

        # ── Preview ──
        preview_frame = QtWidgets.QFrame()
        preview_layout = QtWidgets.QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(0, 4, 0, 0)
        preview_layout.setSpacing(2)
        self._caption_label = QtWidgets.QLabel(
            "<i>Selecione uma linha pra ver a face</i>"
        )
        self._caption_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self._caption_label,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;",
        )
        preview_layout.addWidget(self._caption_label)
        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self._image_label,
            lambda t: (
                "background: " + t("SURFACE_RAISED") + "; "
                "border: 1px solid " + t("BORDER") + ";"
            ),
        )
        self._image_label.setMinimumHeight(200)
        preview_layout.addWidget(self._image_label, 1)
        v_splitter.addWidget(preview_frame)

        # Tabelas (lado a lado) ~60% da altura, preview ~40%
        v_splitter.setSizes([360, 240])

        # Selection wiring — uma seleção numa tabela limpa a outra (radio
        # behavior implícito) e carrega a imagem no preview.
        self.table_hubs.itemSelectionChanged.connect(
            lambda: self._on_selection_changed(self.table_hubs, self.table_anti)
        )
        self.table_anti.itemSelectionChanged.connect(
            lambda: self._on_selection_changed(self.table_anti, self.table_hubs)
        )

    @staticmethod
    def _make_table() -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget()
        t.setColumnCount(6)
        t.setHorizontalHeaderLabels(
            ["Identidade", "Foto", "FIQA", "Norma", "Nk", "Fotos do ID"]
        )
        t.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        t.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        t.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        t.verticalHeader().setVisible(False)

        h = t.horizontalHeader()
        # Modo Interactive em todas as colunas + setStretchLastSection(False)
        # — o usuário pode arrastar bordas; sem nenhuma coluna engolindo
        # todo o espaço disponível como acontecia com `Stretch` em "Foto".
        # Larguras default escolhidas pra caber confortavelmente nas duas
        # tabelas lado a lado em ~500px cada.
        h.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        h.setStretchLastSection(False)
        for col, width in enumerate((100, 130, 60, 70, 50, 80)):
            t.setColumnWidth(col, width)
        return t

    def set_dataset(self, dataset) -> None:
        """Recebe dataset corrente — resolve paths antes de armazenar
        no UserRole das células das tabelas. Sem isso, o preview
        carrega `0005_01.png` direto e falha em modo RELATIVE."""
        self._dataset = dataset

    def render_data(
        self,
        hubs: List[dict],
        anti_hubs: List[dict],
    ) -> None:
        """Popula as duas tabelas. Cada item é um dict com chaves
        identity / image_path / fiqa / norm / nk."""
        self._populate_table(self.table_hubs, hubs)
        self._populate_table(self.table_anti, anti_hubs)
        # Pré-seleciona o primeiro hub pra que o preview já mostre algo
        if hubs:
            self.table_hubs.selectRow(0)
        elif anti_hubs:
            self.table_anti.selectRow(0)
        else:
            self._show_placeholder()

    def clear(self) -> None:
        self.table_hubs.setRowCount(0)
        self.table_anti.setRowCount(0)
        self._show_placeholder()

    # ── Internal helpers ─────────────────────────────────────────────

    def _populate_table(self, t: QtWidgets.QTableWidget, rows: List[dict]) -> None:
        t.setUpdatesEnabled(False)
        t.blockSignals(True)
        try:
            t.setRowCount(len(rows))
            for r, row in enumerate(rows):
                ident = str(row.get("identity") or "?")
                raw_path = str(row.get("image_path") or "")
                # Resolve via dataset em modo RELATIVE (Image Pipeline);
                # armazena no UserRole pra que `_load_preview` consuma
                # path absoluto.
                resolved_path = (
                    str(self._dataset.resolve_image_path(raw_path))
                    if self._dataset is not None and raw_path
                    else raw_path
                )
                # Foto = basename pra leitura rápida; tooltip tem path completo
                from os.path import basename
                file_name = basename(raw_path) if raw_path else "(sem path)"
                fiqa = row.get("fiqa")
                norm = row.get("norm")
                nk = row.get("nk")

                # Coluna 0 — Identidade (carrega o path no UserRole pro preview)
                cell0 = QtWidgets.QTableWidgetItem(ident)
                cell0.setData(QtCore.Qt.ItemDataRole.UserRole, resolved_path)
                t.setItem(r, 0, cell0)

                cell1 = QtWidgets.QTableWidgetItem(file_name)
                cell1.setToolTip(resolved_path)
                t.setItem(r, 1, cell1)

                cell2 = QtWidgets.QTableWidgetItem(
                    f"{float(fiqa):.3f}" if fiqa is not None else "—"
                )
                cell2.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                t.setItem(r, 2, cell2)

                cell3 = QtWidgets.QTableWidgetItem(
                    f"{float(norm):.3f}" if norm is not None else "—"
                )
                cell3.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                t.setItem(r, 3, cell3)

                cell4 = QtWidgets.QTableWidgetItem(
                    str(nk) if nk is not None else "—"
                )
                cell4.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                t.setItem(r, 4, cell4)

                n_id_photos = row.get("n_id_photos")
                cell5 = QtWidgets.QTableWidgetItem(
                    str(n_id_photos) if n_id_photos is not None else "—"
                )
                cell5.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight
                    | QtCore.Qt.AlignmentFlag.AlignVCenter,
                )
                # Cor de fundo leve sinaliza tamanho da identidade —
                # ajuda a triar viés de desbalanceamento visualmente:
                #   - Singleton/quase-singleton (≤2): cinza-azulado
                #   - Grande (≥50): âmbar (possível celebridade)
                if isinstance(n_id_photos, int):
                    if n_id_photos <= 2:
                        cell5.setBackground(QtGui.QColor("#eceff1"))
                    elif n_id_photos >= 50:
                        cell5.setBackground(QtGui.QColor("#fff8e1"))
                t.setItem(r, 5, cell5)
        finally:
            t.setUpdatesEnabled(True)
            t.blockSignals(False)

    def _on_selection_changed(
        self,
        active_table: QtWidgets.QTableWidget,
        other_table: QtWidgets.QTableWidget,
    ) -> None:
        """Quando o usuário clica numa linha, limpa a seleção da outra
        tabela (UX de "uma seleção global") e carrega a imagem no
        preview."""
        rows = sorted({i.row() for i in active_table.selectedIndexes()})
        if not rows:
            return
        # Limpa a outra tabela sem disparar novo selectionChanged
        other_table.blockSignals(True)
        other_table.clearSelection()
        other_table.blockSignals(False)

        cell0 = active_table.item(rows[0], 0)
        if cell0 is None:
            return
        path = cell0.data(QtCore.Qt.ItemDataRole.UserRole)
        ident = cell0.text()
        nk_cell = active_table.item(rows[0], 4)
        nk_str = nk_cell.text() if nk_cell else "?"
        self._load_preview(path or "", ident, nk_str)

    def _load_preview(self, path: str, identity: str, nk: str) -> None:
        """Carrega a imagem em `path` no QLabel de preview, escalada."""
        if not path:
            self._image_label.clear()
            self._image_label.setText("(sem path armazenado)")
            self._caption_label.setText(
                f"<b>{identity}</b> — Nk={nk}  <i>(sem path armazenado)</i>"
            )
            return

        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self._image_label.clear()
            self._image_label.setText("(imagem não encontrada)")
            self._caption_label.setText(
                f"<b>{identity}</b> — Nk={nk}  <i>(arquivo ausente)</i>"
            )
            return

        # Escala mantendo aspect ratio. Usa o tamanho atual do label
        # (o splitter pode ter sido redimensionado pelo usuário).
        target_size = self._image_label.size()
        scaled = pix.scaled(
            target_size,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._caption_label.setText(
            f"<b>{identity}</b> — Nk={nk}  ·  {path}"
        )

    def _show_placeholder(self) -> None:
        self._image_label.clear()
        self._image_label.setText("")
        self._caption_label.setText(
            "<i>Selecione uma linha pra ver a face</i>"
        )

    # Reescala quando o widget é redimensionado pelo splitter
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        # Re-load se houver imagem corrente — re-renderiza no novo tamanho
        # do label. Identifica via re-query da seleção ativa.
        for tbl in (self.table_hubs, self.table_anti):
            rows = sorted({i.row() for i in tbl.selectedIndexes()})
            if not rows:
                continue
            cell0 = tbl.item(rows[0], 0)
            if cell0 is None:
                continue
            path = cell0.data(QtCore.Qt.ItemDataRole.UserRole)
            if path:
                ident = cell0.text()
                nk_cell = tbl.item(rows[0], 4)
                nk_str = nk_cell.text() if nk_cell else "?"
                self._load_preview(path, ident, nk_str)
                break
