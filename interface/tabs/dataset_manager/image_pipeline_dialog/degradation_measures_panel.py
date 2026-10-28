"""DegradationMeasuresPanel — medidas da imagem original, da processada e
da questionada, lado a lado.

É o painel de calibração: o perito ajusta os passos até a coluna
"Processada" ficar próxima da "Questionada". A célula da processada fica
verde quando a medida está dentro da tolerância e âmbar quando não (a
tolerância de cada medida está na dica do mouse).

Uma face só não diz se a BASE ficou parecida com a questionada. O botão
"Medir na base" roda o pipeline numa amostra e preenche a última coluna
com a mediana e a parcela das imagens que, uma a uma, ficam dentro da
tolerância — a mesma conta que o Processar grava no manifesto. As medidas e a regra de
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

_COL_ORIGINAL, _COL_PROCESSADA, _COL_QUESTIONADA, _COL_AMOSTRA = 1, 2, 3, 4


class DegradationMeasuresPanel(QtWidgets.QWidget):
    """Tabela compacta. Emite `sample_requested` pelo botão da amostra."""

    sample_requested = QtCore.pyqtSignal()
    fit_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        topo = QtWidgets.QHBoxLayout()
        titulo = QtWidgets.QLabel("<b>Medidas de degradação</b> (região da face)", parent=self)
        titulo.setStyleSheet("font-size: 11px;")
        topo.addWidget(titulo, 1)
        self.btn_sample = QtWidgets.QPushButton("📊 Medir na base", parent=self)
        self.btn_sample.setToolTip(
            "Roda o pipeline numa amostra da base (sem gravar nada) e mostra, "
            "por medida, a mediana e quantas imagens ficam dentro da "
            "tolerância da questionada, uma a uma."
        )
        self.btn_sample.clicked.connect(self.sample_requested.emit)
        topo.addWidget(self.btn_sample)
        self.btn_fit = QtWidgets.QPushButton("🎯 Ajustar à questionada", parent=self)
        self.btn_fit.setToolTip(
            "Procura os valores dos campos numéricos dos passos que deixam as "
            "imagens da base, uma a uma, mais parecidas com a questionada. "
            "Ajusta numa amostra e confere em outra, com imagens que o ajuste "
            "não viu. A ordem dos passos e as escolhas de técnica não mudam. "
            "Nada é aplicado sem confirmação."
        )
        self.btn_fit.clicked.connect(self.fit_requested.emit)
        topo.addWidget(self.btn_fit)
        layout.addLayout(topo)

        self.table = QtWidgets.QTableWidget(len(MEASURES), 5, parent=self)
        self.table.setHorizontalHeaderLabels(
            ["Medida", "Original", "Processada", "Questionada", "Base (amostra)"])
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
        for c in (1, 2, 3, 4):
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
        self._amostra = None  # SampleEvaluation
        self._amostra_desatualizada = False
        self._amostra_status: Optional[str] = None
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
        self._amostra = None
        self._render()

    def set_sample(self, avaliacao):
        """Resultado da amostra (`SampleEvaluation`) ou None."""
        self._amostra = avaliacao
        self._amostra_desatualizada = False
        self._amostra_status = None
        self.btn_sample.setEnabled(True)
        self._render()

    def set_sample_running(self, texto: str):
        self._amostra_status = texto
        self.btn_sample.setEnabled(False)
        self._render()

    def set_fit_running(self, rodando: bool):
        self.btn_fit.setText("⏹ Cancelar ajuste" if rodando else "🎯 Ajustar à questionada")
        self.btn_sample.setEnabled(not rodando)
        if not rodando:
            self._amostra_status = None
            self._render()

    def mark_sample_stale(self):
        """O pipeline mudou depois da medição da amostra."""
        if self._amostra is not None and not self._amostra_desatualizada:
            self._amostra_desatualizada = True
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
            self.table.setItem(r, _COL_AMOSTRA, self._celula_da_amostra(m, perto, longe))

        partes = []
        if self._questionada is None:
            partes.append("Carregue a questionada para comparar.")
        elif self._processada is None:
            partes.append("Clique ▶ Aplicar para medir a imagem processada.")
        else:
            partes.append(f"{n_perto} de {n_comparadas} medidas dentro da tolerância.")
        partes.extend(self._texto_da_amostra())
        regioes = {x.region for x in (self._original, self._processada, self._questionada)
                   if x is not None}
        if any(r != "face" for r in regioes):
            partes.append("⚠ Sem a posição da face em alguma imagem: medida na imagem "
                          "inteira, o que não é comparável com a região da face.")
        self.lbl_status.setText(" ".join(partes))

    def _celula_da_amostra(self, m, perto, longe) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem("—")
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if self._amostra is None:
            return item
        linha = next((x for x in self._amostra.comparison if x["measure"] == m.key), None)
        if linha is None:
            return item
        mediana = m.format(linha.get("set_median"))
        frac = linha.get("images_close_fraction")
        item.setText(mediana if frac is None else f"{mediana} · {100 * frac:.0f}%")
        if frac is not None:
            # A cor diz se a MAIORIA das imagens está dentro — não só a mediana.
            item.setForeground(QtGui.QBrush(perto if frac >= 0.5 else longe))
            item.setToolTip(
                f"Mediana da amostra: {mediana}. {100 * frac:.0f}% das imagens, "
                f"uma a uma, dentro da tolerância ({m.tolerance_text()})."
            )
        if self._amostra_desatualizada:
            item.setForeground(QtGui.QBrush(QtGui.QColor(current_token_fn()("TEXT_MUTED"))))
        return item

    def _texto_da_amostra(self):
        if self._amostra_status:
            return [self._amostra_status]
        if self._amostra is None:
            return []
        a = self._amostra
        partes = []
        if a.agreement:
            n_med = len(a.agreement["measures"])
            partes.append(
                f"Base: amostra de {a.n_measured} imagens — "
                f"{100 * a.agreement['all_close_fraction']:.0f}% com as {n_med} medidas "
                f"dentro, {100 * a.agreement['all_but_one_close_fraction']:.0f}% com "
                f"{n_med - 1} ou mais."
            )
        else:
            partes.append(f"Base: amostra de {a.n_measured} imagens medida.")
        if a.n_failed:
            partes.append(f"⚠ {a.n_failed} imagem(ns) da amostra falharam no pipeline.")
        if self._amostra_desatualizada:
            partes.append("(o pipeline mudou depois — meça de novo)")
        return partes
