"""DegradationMeasuresPanel — medidas da imagem original, da processada e
da questionada, lado a lado.

É o painel de calibração: o perito ajusta os passos até a coluna
"Processada" ficar próxima da "Questionada". A célula da processada fica
verde quando a medida está dentro da tolerância e âmbar quando não (a
tolerância de cada medida está na dica do mouse). As medidas e a regra de
"próximo" vêm do backend (`image_degradation_profile_service`); aqui só
se exibe.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet, current_token_fn
from faceaudit.services.image_degradation_profile_service import (
    MEASURES,
    DegradationMeasures,
)

_COL_ORIGINAL, _COL_PROCESSADA, _COL_QUESTIONADA = 1, 2, 3


class DegradationMeasuresPanel(QtWidgets.QWidget):
    """Tabela compacta. Não emite sinais — só recebe."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        titulo = QtWidgets.QLabel("<b>Medidas de degradação</b> (região da face)", parent=self)
        titulo.setStyleSheet("font-size: 11px;")
        layout.addWidget(titulo)

        self.table = QtWidgets.QTableWidget(len(MEASURES), 4, parent=self)
        self.table.setHorizontalHeaderLabels(["Medida", "Original", "Processada", "Questionada"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        altura_linha = 22
        self.table.verticalHeader().setDefaultSectionSize(altura_linha)
        cab = self.table.horizontalHeader().sizeHint().height()
        self.table.setFixedHeight(cab + altura_linha * len(MEASURES) + 8)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(
                c, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for r, m in enumerate(MEASURES):
            item = QtWidgets.QTableWidgetItem(m.label)
            item.setToolTip(f"{m.help}\nTolerância: {m.tolerance_text()}")
            self.table.setItem(r, 0, item)
        layout.addWidget(self.table)

        self.lbl_status = QtWidgets.QLabel("", parent=self)
        self.lbl_status.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_status,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 10px; padding: 2px 4px;",
        )
        layout.addWidget(self.lbl_status)

        self._original: Optional[DegradationMeasures] = None
        self._processada: Optional[DegradationMeasures] = None
        self._questionada: Optional[DegradationMeasures] = None
        self._render()

    # ── API ────────────────────────────────────────────────────────────────

    def set_original(self, m: Optional[DegradationMeasures]):
        self._original = m
        self._processada = None
        self._render()

    def set_processed(self, m: Optional[DegradationMeasures]):
        self._processada = m
        self._render()

    def set_questioned(self, m: Optional[DegradationMeasures]):
        self._questionada = m
        self._render()

    # ── Render ─────────────────────────────────────────────────────────────

    def _render(self):
        t = current_token_fn()
        perto = QtGui.QColor(t("SUCCESS"))
        longe = QtGui.QColor(t("WARNING"))
        n_perto = n_comparadas = 0
        for r, m in enumerate(MEASURES):
            for col, medidas in ((_COL_ORIGINAL, self._original),
                                 (_COL_PROCESSADA, self._processada),
                                 (_COL_QUESTIONADA, self._questionada)):
                valor = None if medidas is None else getattr(medidas, m.key)
                item = QtWidgets.QTableWidgetItem(m.format(valor))
                item.setTextAlignment(
                    QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if col == _COL_PROCESSADA and self._questionada is not None:
                    alvo = getattr(self._questionada, m.key)
                    proximo = m.is_close(alvo, valor)
                    if proximo is not None:
                        n_comparadas += 1
                        n_perto += int(proximo)
                        item.setForeground(QtGui.QBrush(perto if proximo else longe))
                        fonte = item.font()
                        fonte.setBold(True)
                        item.setFont(fonte)
                        item.setToolTip(
                            ("Dentro" if proximo else "Fora")
                            + f" da tolerância da questionada ({m.tolerance_text()})."
                        )
                self.table.setItem(r, col, item)

        partes = []
        if self._questionada is None:
            partes.append("Carregue a questionada para comparar.")
        elif self._processada is None:
            partes.append("Clique ▶ Aplicar para medir a imagem processada.")
        else:
            partes.append(f"{n_perto} de {n_comparadas} medidas dentro da tolerância.")
        regioes = {x.region for x in (self._original, self._processada, self._questionada)
                   if x is not None}
        if any(r != "face" for r in regioes):
            partes.append("⚠ Sem a posição da face em alguma imagem: medida na imagem "
                          "inteira, o que não é comparável com a região da face.")
        self.lbl_status.setText(" ".join(partes))
