"""View do diálogo de filtro — lista dinâmica de `ColumnFilter`.

Substitui a versão antiga (form gigante hardcoded com 50+ fields).
Estrutura:
  - Header: nome + descrição + toolbar (Abrir/Salvar/Salvar Como/Limpar)
  - Lista vertical de rows; cada row é um `_FilterRowWidget` que
    combina dropdown de coluna + widget de critério (varia por tipo) +
    botão remover
  - Footer: ➕ Adicionar filtro + Aplicar / Cancelar

Stateless o quanto possível — popular columns + carregar/salvar é
responsabilidade do controller.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.filter.column_filter_data_type import (
    ColumnFilterDataType,
)


# Cor de fundo pra row inválida (coluna não existe no dataset corrente)
_INVALID_BG = "#ffebee"
_INVALID_BORDER = "#c62828"
_VALID_BG = "transparent"


class _MultiSelectPopup(QtWidgets.QDialog):
    """Diálogo modal pequeno com lista de valores checkable. Usado pelo
    botão ▾ do filtro de string pra escolher múltiplos valores únicos
    de uma coluna sem precisar digitar."""

    def __init__(
        self,
        values: List[str],
        preselected: Optional[set] = None,
        parent: Optional[QtWidgets.QWidget] = None,
        title: str = "Selecionar valores",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 420)
        preselected = preselected or set()

        layout = QtWidgets.QVBoxLayout(self)

        # Toolbar acima da lista — Marcar/Desmarcar tudo
        top = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("☑ Marcar tudo")
        btn_all.clicked.connect(self._mark_all)
        btn_none = QtWidgets.QPushButton("☐ Desmarcar tudo")
        btn_none.clicked.connect(self._mark_none)
        top.addWidget(btn_all)
        top.addWidget(btn_none)
        top.addStretch()
        layout.addLayout(top)

        # Lista checkable
        self._list = QtWidgets.QListWidget()
        for v in values:
            item = QtWidgets.QListWidgetItem(str(v))
            item.setFlags(
                item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
            )
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if str(v) in preselected
                else QtCore.Qt.CheckState.Unchecked
            )
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        # Botões
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _mark_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(
                QtCore.Qt.CheckState.Checked
            )

    def _mark_none(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(
                QtCore.Qt.CheckState.Unchecked
            )

    def get_selected(self) -> List[str]:
        out: List[str] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out


class _FilterRowWidget(QtWidgets.QFrame):
    """Uma linha = um ColumnFilter. Coluna (combo) + critério (varia
    por tipo, switched in-place) + remover."""

    removed = QtCore.pyqtSignal(object)  # emite self pra parent remover

    def __init__(
        self,
        available_columns: List[Tuple[str, ColumnFilterDataType]],
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        available_values_per_column: Optional[Dict[str, List[str]]] = None,
    ):
        """`available_columns` é lista de (column_name, data_type) — UI
        popula o dropdown. `available_values_per_column` é um dict
        {column_name → [valores únicos]} usado pelo popup multi-select
        do filtro de string."""
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setObjectName("FilterRow")
        # Vertical size policy = Fixed → row mantém altura natural
        # compacta. Sem isso, com poucos rows o stretch trailing do
        # parent não consegue empurrar pra cima e os rows expandem.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self._available_columns = list(available_columns)
        self._available_values_per_column: Dict[str, List[str]] = dict(
            available_values_per_column or {}
        )
        self._is_invalid: bool = False
        self._invalid_reason: str = ""

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Indicador de inválido (oculto por default)
        self._invalid_indicator = QtWidgets.QLabel("⚠")
        self._invalid_indicator.setStyleSheet(
            f"color: {_INVALID_BORDER}; font-size: 16px;"
        )
        self._invalid_indicator.hide()
        layout.addWidget(self._invalid_indicator)

        # Combo de coluna
        self._combo_column = QtWidgets.QComboBox()
        self._combo_column.setMinimumWidth(180)
        self._combo_column.addItem("(escolher coluna)", None)
        for col_name, dt in self._available_columns:
            display = f"{col_name}  ({dt.to_str()})"
            self._combo_column.addItem(display, (col_name, dt))
        self._combo_column.currentIndexChanged.connect(
            self._on_column_changed
        )
        layout.addWidget(self._combo_column)

        # Stack de critérios — substituído ao trocar tipo da coluna
        self._criterion_stack = QtWidgets.QStackedWidget()
        self._criterion_stack.setMinimumWidth(280)
        self._build_criterion_pages()
        layout.addWidget(self._criterion_stack, 1)

        # Botão remover
        self._btn_remove = QtWidgets.QToolButton()
        self._btn_remove.setText("✕")
        self._btn_remove.setToolTip("Remover este filtro")
        self._btn_remove.clicked.connect(
            lambda: self.removed.emit(self)
        )
        layout.addWidget(self._btn_remove)

    def _build_criterion_pages(self) -> None:
        """Constrói as 4 páginas do stack: empty (nenhuma coluna),
        numeric, string, boolean."""
        # Página 0 — placeholder
        empty = QtWidgets.QLabel(
            "<i>Selecione uma coluna pra configurar o filtro</i>"
        )
        empty.setStyleSheet("color: #999; padding: 4px;")
        self._criterion_stack.addWidget(empty)

        # Página 1 — numérico (values + ranges em linha de texto)
        numeric_page = QtWidgets.QWidget()
        numeric_layout = QtWidgets.QHBoxLayout(numeric_page)
        numeric_layout.setContentsMargins(0, 0, 0, 0)
        numeric_layout.setSpacing(4)

        self._numeric_values = QtWidgets.QLineEdit()
        self._numeric_values.setPlaceholderText(
            "Valores (ex: 2.5, 3.0)"
        )
        self._numeric_values.setToolTip(
            "Lista de valores exatos separados por vírgula. Match se a "
            "coluna == qualquer um deles. Ex: 2.5, 3.0, 4.7"
        )
        numeric_layout.addWidget(self._numeric_values, 1)

        self._numeric_ranges = QtWidgets.QLineEdit()
        self._numeric_ranges.setPlaceholderText(
            "Faixas (ex: 1.0:2.0, 3.0:5.0)"
        )
        self._numeric_ranges.setToolTip(
            "Lista de faixas inclusivas separadas por vírgula. Cada "
            "faixa usa ':' entre min e max. Match se a coluna estiver "
            "em qualquer faixa. Ex: 1.0:2.0, 3.0:5.0\n"
            "Use 'inf' / '-inf' pra abertos: ex '0:inf' = 'qualquer ≥ 0'."
        )
        numeric_layout.addWidget(self._numeric_ranges, 1)
        self._criterion_stack.addWidget(numeric_page)

        # Página 2 — string (lista de valores via texto OU popup multi-select)
        string_page = QtWidgets.QWidget()
        string_layout = QtWidgets.QHBoxLayout(string_page)
        string_layout.setContentsMargins(0, 0, 0, 0)
        string_layout.setSpacing(4)

        self._string_values = QtWidgets.QLineEdit()
        self._string_values.setPlaceholderText(
            "Valores (ex: M, F) ou clique ▾ pra escolher"
        )
        self._string_values.setToolTip(
            "Lista de strings exatas separadas por vírgula. Match se "
            "a coluna == qualquer uma. Ex: M, F  ou  WebFace4M, BFW.\n"
            "Use o botão ▾ pra escolher de uma lista de valores únicos "
            "encontrados no dataset."
        )
        string_layout.addWidget(self._string_values, 1)

        # Botão ▾ — abre popup com lista de valores únicos da coluna
        # corrente; usuário marca os que quiser e o LineEdit é preenchido.
        self._string_picker_btn = QtWidgets.QToolButton()
        self._string_picker_btn.setText("▾")
        self._string_picker_btn.setToolTip(
            "Escolher valores de uma lista (multi-select)"
        )
        self._string_picker_btn.clicked.connect(
            self._on_string_picker_clicked
        )
        string_layout.addWidget(self._string_picker_btn)
        self._criterion_stack.addWidget(string_page)

        # Página 3 — boolean
        bool_page = QtWidgets.QWidget()
        bool_layout = QtWidgets.QHBoxLayout(bool_page)
        bool_layout.setContentsMargins(0, 0, 0, 0)
        self._bool_check = QtWidgets.QCheckBox(
            "Match quando a coluna for True"
        )
        self._bool_check.setToolTip(
            "Quando marcado: passa só se a coluna for True. Quando "
            "desmarcado: passa só se for False."
        )
        bool_layout.addWidget(self._bool_check)
        bool_layout.addStretch()
        self._criterion_stack.addWidget(bool_page)

    def _on_column_changed(self) -> None:
        """Switch da página do stack conforme o data_type da coluna."""
        data = self._combo_column.currentData()
        if data is None:
            self._criterion_stack.setCurrentIndex(0)
            return
        _, dt = data
        if dt == ColumnFilterDataType.NUMERIC:
            self._criterion_stack.setCurrentIndex(1)
        elif dt == ColumnFilterDataType.STRING:
            self._criterion_stack.setCurrentIndex(2)
            # Habilita/desabilita o picker conforme há valores conhecidos
            cn = data[0]
            has_values = bool(
                self._available_values_per_column.get(cn)
            )
            self._string_picker_btn.setEnabled(has_values)
        elif dt == ColumnFilterDataType.BOOLEAN:
            self._criterion_stack.setCurrentIndex(3)
        else:
            self._criterion_stack.setCurrentIndex(0)

    def _on_string_picker_clicked(self) -> None:
        """Abre popup com lista de valores únicos da coluna corrente
        — usuário marca múltiplos via checkbox; ao aceitar, popula
        o LineEdit com os valores selecionados separados por vírgula."""
        data = self._combo_column.currentData()
        if data is None:
            return
        column_name, _ = data
        values = self._available_values_per_column.get(column_name) or []
        if not values:
            return

        # Pré-marca os já presentes no LineEdit
        already = set(
            self._parse_string_list(self._string_values.text())
        )

        dlg = _MultiSelectPopup(
            values, preselected=already, parent=self,
            title=f"Selecionar valores — {column_name}",
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        chosen = dlg.get_selected()
        self._string_values.setText(", ".join(chosen))

    # ── API pública ──────────────────────────────────────────────────

    def set_state(
        self,
        column_name: str,
        data_type: ColumnFilterDataType,
        *,
        numeric_values: Optional[List[float]] = None,
        numeric_ranges: Optional[List[Tuple[float, float]]] = None,
        string_values: Optional[List[str]] = None,
        bool_value: Optional[bool] = None,
    ) -> None:
        """Popula a row com um ColumnFilter já existente. Usado quando
        carregamos um filtro salvo do disco. Se a coluna não existe no
        dataset corrente, adiciona uma entry virtual no combo (a row
        ficará marcada como inválida pelo controller)."""
        target = None
        for i in range(1, self._combo_column.count()):
            data = self._combo_column.itemData(i)
            if data is None:
                continue
            cn, _ = data
            if cn == column_name:
                target = i
                break

        if target is None:
            display = f"{column_name}  ({data_type.to_str()})  ⚠ ausente"
            self._combo_column.addItem(display, (column_name, data_type))
            target = self._combo_column.count() - 1

        self._combo_column.setCurrentIndex(target)

        if data_type == ColumnFilterDataType.NUMERIC:
            if numeric_values:
                self._numeric_values.setText(
                    ", ".join(f"{v:g}" for v in numeric_values)
                )
            if numeric_ranges:
                self._numeric_ranges.setText(
                    ", ".join(
                        f"{lo:g}:{hi:g}" for lo, hi in numeric_ranges
                    )
                )
        elif data_type == ColumnFilterDataType.STRING:
            if string_values:
                self._string_values.setText(", ".join(string_values))
        elif data_type == ColumnFilterDataType.BOOLEAN:
            self._bool_check.setChecked(bool(bool_value))

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Retorna dict serializável com o estado atual da row, OU
        None se nenhuma coluna foi escolhida. Caller (controller) faz
        o build do DTO `ColumnFilter` correspondente."""
        data = self._combo_column.currentData()
        if data is None:
            return None
        column_name, dt = data
        out: Dict[str, Any] = {
            "column_name": column_name,
            "data_type": dt,
        }
        if dt == ColumnFilterDataType.NUMERIC:
            out["values"] = self._parse_numeric_list(
                self._numeric_values.text()
            )
            out["ranges"] = self._parse_range_list(
                self._numeric_ranges.text()
            )
        elif dt == ColumnFilterDataType.STRING:
            out["values"] = self._parse_string_list(
                self._string_values.text()
            )
        elif dt == ColumnFilterDataType.BOOLEAN:
            out["value"] = self._bool_check.isChecked()
        return out

    def set_invalid(self, invalid: bool, reason: str = "") -> None:
        """Marca a row em vermelho com tooltip do motivo."""
        self._is_invalid = bool(invalid)
        self._invalid_reason = reason or ""
        if self._is_invalid:
            self.setStyleSheet(
                f"#FilterRow {{ background: {_INVALID_BG}; "
                f"border: 1px solid {_INVALID_BORDER}; }}"
            )
            self._invalid_indicator.show()
            self._invalid_indicator.setToolTip(reason)
        else:
            self.setStyleSheet(
                f"#FilterRow {{ background: {_VALID_BG}; }}"
            )
            self._invalid_indicator.hide()
            self._invalid_indicator.setToolTip("")

    @property
    def is_invalid(self) -> bool:
        return self._is_invalid

    # ── Parsers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_numeric_list(text: str) -> List[float]:
        if not text or not text.strip():
            return []
        out: List[float] = []
        for tok in text.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                out.append(float(tok))
            except ValueError:
                continue
        return out

    @staticmethod
    def _parse_range_list(text: str) -> List[Tuple[float, float]]:
        if not text or not text.strip():
            return []
        out: List[Tuple[float, float]] = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if ":" not in chunk:
                continue
            lo_s, hi_s = chunk.split(":", 1)
            try:
                lo = float(lo_s.strip())
                hi = float(hi_s.strip())
                out.append((lo, hi))
            except ValueError:
                continue
        return out

    @staticmethod
    def _parse_string_list(text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return [
            tok.strip() for tok in text.split(",")
            if tok.strip()
        ]


class FilterDialogView(QtWidgets.QDialog):
    """Diálogo principal — header + lista de rows + footer."""

    # Sinais semânticos pra controller
    open_filter_requested = QtCore.pyqtSignal()
    save_filter_requested = QtCore.pyqtSignal()
    save_as_filter_requested = QtCore.pyqtSignal()
    clear_all_requested = QtCore.pyqtSignal()
    apply_requested = QtCore.pyqtSignal()
    add_filter_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Filtro de Dataset")
        self.resize(820, 580)

        self._rows: List[_FilterRowWidget] = []
        self._available_columns: List[Tuple[str, ColumnFilterDataType]] = []
        # Valores únicos por coluna string — usado pelo popup ▾ do
        # multi-select. Populado pelo controller via
        # `set_available_values_per_column`.
        self._available_values_per_column: Dict[str, List[str]] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Header
        header_form = QtWidgets.QFormLayout()
        header_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._edit_name = QtWidgets.QLineEdit()
        self._edit_name.setPlaceholderText("Nome do filtro")
        header_form.addRow("Nome:", self._edit_name)
        self._edit_description = QtWidgets.QLineEdit()
        self._edit_description.setPlaceholderText(
            "Descrição (opcional) — pra documentar o filtro"
        )
        header_form.addRow("Descrição:", self._edit_description)
        outer.addLayout(header_form)

        # Toolbar
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)
        self.btn_open = QtWidgets.QPushButton("📂 Abrir")
        self.btn_open.setToolTip("Carregar filtro salvo da biblioteca global")
        self.btn_open.clicked.connect(self.open_filter_requested.emit)
        toolbar.addWidget(self.btn_open)

        self.btn_save = QtWidgets.QPushButton("💾 Salvar")
        self.btn_save.setToolTip(
            "Salva sob o nome atual na biblioteca global "
            "(sobrescreve se já existir)"
        )
        self.btn_save.clicked.connect(self.save_filter_requested.emit)
        toolbar.addWidget(self.btn_save)

        self.btn_save_as = QtWidgets.QPushButton("💾 Salvar Como…")
        self.btn_save_as.setToolTip(
            "Pede um nome novo e salva na biblioteca global"
        )
        self.btn_save_as.clicked.connect(self.save_as_filter_requested.emit)
        toolbar.addWidget(self.btn_save_as)

        toolbar.addStretch()

        self.btn_clear = QtWidgets.QPushButton("🗑 Limpar tudo")
        self.btn_clear.setToolTip("Remove todos os filtros da lista")
        self.btn_clear.clicked.connect(self.clear_all_requested.emit)
        toolbar.addWidget(self.btn_clear)

        outer.addLayout(toolbar)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        outer.addWidget(sep)

        # Lista (scroll)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_container = QtWidgets.QWidget()
        self._rows_layout = QtWidgets.QVBoxLayout(self._scroll_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch()
        self._scroll.setWidget(self._scroll_container)
        outer.addWidget(self._scroll, 1)

        # Footer
        footer = QtWidgets.QHBoxLayout()
        self.btn_add_filter = QtWidgets.QPushButton("➕ Adicionar filtro")
        self.btn_add_filter.clicked.connect(self.add_filter_requested.emit)
        footer.addWidget(self.btn_add_filter)
        footer.addStretch()

        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.clicked.connect(self.reject)
        footer.addWidget(self.btn_cancel)

        self.btn_apply = QtWidgets.QPushButton("Aplicar")
        self.btn_apply.setStyleSheet(
            "QPushButton { background: #1976d2; color: white; "
            "font-weight: bold; padding: 5px 16px; border-radius: 4px; } "
            "QPushButton:hover { background: #115293; }"
        )
        self.btn_apply.clicked.connect(self.apply_requested.emit)
        footer.addWidget(self.btn_apply)

        outer.addLayout(footer)

    # ── API pública pro controller ──────────────────────────────────

    def set_available_columns(
        self,
        columns: List[Tuple[str, ColumnFilterDataType]],
    ) -> None:
        """Define quais colunas aparecem no dropdown de cada filter row."""
        self._available_columns = list(columns)

    def set_available_values_per_column(
        self,
        values: Dict[str, List[str]],
    ) -> None:
        """Define os valores únicos por coluna string — alimentam o
        popup ▾ do multi-select. Chamado pelo controller após
        inspecionar o dataset."""
        self._available_values_per_column = dict(values)

    def set_header(self, name: str, description: Optional[str]) -> None:
        self._edit_name.setText(str(name or ""))
        self._edit_description.setText(str(description or ""))

    def get_header(self) -> Tuple[str, Optional[str]]:
        return (
            self._edit_name.text().strip() or "Novo Filtro",
            self._edit_description.text().strip() or None,
        )

    def add_row(
        self,
        column_name: Optional[str] = None,
        data_type: Optional[ColumnFilterDataType] = None,
        **kwargs: Any,
    ) -> _FilterRowWidget:
        """Adiciona uma row vazia (ou pré-populada via kwargs)."""
        row = _FilterRowWidget(
            self._available_columns,
            parent=self,
            available_values_per_column=self._available_values_per_column,
        )
        row.removed.connect(self._on_row_removed)
        # Insere antes do stretch trailing
        self._rows_layout.insertWidget(
            self._rows_layout.count() - 1, row,
        )
        self._rows.append(row)

        if column_name and data_type is not None:
            row.set_state(column_name, data_type, **kwargs)
        return row

    def get_row_states(self) -> List[Dict[str, Any]]:
        """Retorna estados dos filtros (1 dict por row não-vazia)."""
        out = []
        for row in self._rows:
            state = row.get_state()
            if state is not None:
                out.append(state)
        return out

    def clear_rows(self) -> None:
        """Remove todas as rows."""
        for row in list(self._rows):
            self._remove_row(row)

    def mark_invalid_rows(
        self, invalid_indices_with_reasons: Dict[int, str],
    ) -> None:
        """Marca rows inválidas em vermelho. Dict[index, reason]."""
        for i, row in enumerate(self._rows):
            if i in invalid_indices_with_reasons:
                row.set_invalid(True, invalid_indices_with_reasons[i])
            else:
                row.set_invalid(False)

    def has_invalid_rows(self) -> bool:
        return any(row.is_invalid for row in self._rows)

    # ── Internal ─────────────────────────────────────────────────────

    def _on_row_removed(self, row: _FilterRowWidget) -> None:
        self._remove_row(row)

    def _remove_row(self, row: _FilterRowWidget) -> None:
        if row in self._rows:
            self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()
