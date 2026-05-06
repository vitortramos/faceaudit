"""Diálogo de propriedades de coluna existente.

Restrições por role:
  - **canônicas** (qualquer role != CUSTOM): só `label` e `vocabulary` (apenas
    para `demographic` que já têm vocab) podem ser editados. Role/source/
    dtype são imutáveis — alterar quebra os contratos de código (ex: trocar
    o role de `identity` para `custom` derruba todos os splits).
  - **custom** (role=CUSTOM, source=CUSTOM): permite editar label, vocabulary
    e default_value. Nome e dtype continuam fixos (mudar dtype invalidaria
    os valores já gravados).

O usuário pode adicionar/remover entradas do vocabulário (estendível por
dataset) — extensão respeita a decisão de design "vocabulários estendíveis".
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

from PyQt6 import QtCore, QtWidgets


class ColumnPropertiesDialog(QtWidgets.QDialog):
    """Edita atributos editáveis de um ColumnSpec já existente.

    Retorna (label, vocabulary_or_None, default_or_unset) via get_result.
    Quem pegar o resultado deve construir o novo spec via dataclasses.replace.
    """

    _UNSET = object()  # sentinela para "campo não foi mudado"

    def __init__(self, column_spec: Any, parent=None):
        super().__init__(parent)
        self._spec = column_spec
        self.setWindowTitle(f"Propriedades — {getattr(column_spec, 'name', '?')}")
        self.setModal(True)
        self.resize(480, 380)
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        info = self._build_info_section()
        layout.addLayout(info)

        layout.addWidget(self._horizontal_separator())

        editable = self._build_editable_section()
        layout.addLayout(editable)

        layout.addStretch()

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _build_info_section(self) -> QtWidgets.QFormLayout:
        """Linha de info read-only sobre name/role/source/dtype/category."""
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        spec = self._spec
        role_value = getattr(getattr(spec, "role", None), "value", "—")
        source_value = getattr(getattr(spec, "source", None), "value", "—")
        dtype_value = getattr(getattr(spec, "dtype", None), "value", "—")

        form.addRow("Nome:", _readonly_label(getattr(spec, "name", "—")))
        form.addRow("Role:", _readonly_label(role_value))
        form.addRow("Source:", _readonly_label(source_value))
        form.addRow("Tipo:", _readonly_label(dtype_value))
        form.addRow("Categoria:", _readonly_label(getattr(spec, "category", "—") or "—"))
        return form

    def _build_editable_section(self) -> QtWidgets.QVBoxLayout:
        """Campos editáveis dependentes do role."""
        layout = QtWidgets.QVBoxLayout()

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        # Label — sempre editável
        self.txt_label = QtWidgets.QLineEdit(getattr(self._spec, "label", ""))
        form.addRow("Rótulo:", self.txt_label)

        # Default value — só pra custom; outros roles têm semântica do enricher.
        self._is_custom = self._spec_is_custom()
        if self._is_custom:
            self.txt_default = QtWidgets.QLineEdit(
                "" if self._spec.default_value is None else str(self._spec.default_value)
            )
            self.txt_default.setPlaceholderText("(vazio = NaN)")
            form.addRow("Valor padrão:", self.txt_default)
        else:
            self.txt_default = None
            note = QtWidgets.QLabel("Coluna canônica — `default_value` não editável.")
            note.setStyleSheet("color: #888; font-style: italic;")
            form.addRow("Valor padrão:", note)

        layout.addLayout(form)

        # Vocabulary — editável se já existir vocab (demographic) ou se é custom
        self._supports_vocab = self._can_edit_vocabulary()
        if self._supports_vocab:
            layout.addWidget(QtWidgets.QLabel("<b>Vocabulário fechado</b>"))
            note = QtWidgets.QLabel(
                "Edite as opções permitidas (uma por linha). Vazio = sem vocabulário fixo."
            )
            note.setStyleSheet("color: #666; font-size: 11px;")
            note.setWordWrap(True)
            layout.addWidget(note)

            self.txt_vocab = QtWidgets.QPlainTextEdit()
            self.txt_vocab.setPlaceholderText("M\nF\nOutro\nDesconhecido")
            current = getattr(self._spec, "vocabulary", None) or ()
            self.txt_vocab.setPlainText("\n".join(current))
            self.txt_vocab.setMaximumHeight(120)
            layout.addWidget(self.txt_vocab)
        else:
            self.txt_vocab = None

        return layout

    def _horizontal_separator(self) -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        return line

    # ── Helpers ────────────────────────────────────────────────────────────

    def _spec_is_custom(self) -> bool:
        role = getattr(self._spec, "role", None)
        return getattr(role, "value", None) == "custom"

    def _can_edit_vocabulary(self) -> bool:
        """Permite editar vocab quando: (a) é custom (livre); ou (b) já tem
        vocab existente (canônica demographic). Não permite criar vocab onde
        nunca existiu numa coluna não-custom."""
        if self._is_custom:
            return True
        return getattr(self._spec, "vocabulary", None) is not None

    # ── API ────────────────────────────────────────────────────────────────

    def get_result(self) -> Tuple[str, Optional[Tuple[str, ...]], Any]:
        """Retorna (label, vocabulary_or_None, default_value_sentinel).

        - `vocabulary_or_None`: tupla das entradas (vazia → None — "sem vocab")
          ou `_UNSET` quando o widget não foi exibido (não muda).
        - `default_value`: o novo default ou `_UNSET` quando não exibido.
        """
        label = self.txt_label.text().strip() or self._spec.name

        if self.txt_vocab is not None:
            raw = self.txt_vocab.toPlainText().strip()
            if not raw:
                vocab: Any = None
            else:
                items = tuple(line.strip() for line in raw.splitlines() if line.strip())
                vocab = items if items else None
        else:
            vocab = self._UNSET

        if self.txt_default is not None:
            raw_def = self.txt_default.text()
            default: Any = raw_def if raw_def else None
        else:
            default = self._UNSET

        return label, vocab, default

    def apply_to_spec(self, spec: Any) -> Any:
        """Constrói novo ColumnSpec aplicando o resultado deste diálogo.

        Helper conveniente — o caller pode fazer manualmente também.
        """
        label, vocab, default = self.get_result()
        kwargs = {"label": label}
        if vocab is not self._UNSET:
            kwargs["vocabulary"] = vocab
        if default is not self._UNSET:
            kwargs["default_value"] = default
        return dataclasses.replace(spec, **kwargs)


def _readonly_label(text: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(str(text))
    lbl.setStyleSheet("color: #555;")
    lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return lbl
