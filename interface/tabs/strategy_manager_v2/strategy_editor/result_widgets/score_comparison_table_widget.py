"""Tabela das comparações que produziram os escores.

A distribuição mostra a forma do conjunto; esta tabela mostra de onde cada
ponto veio. Num confronto de caso é a informação central do exame, e fora
dele são os extremos que revelam onde o sistema erra.

A seleção das linhas é decidida no backend
(`faceaudit.services.score_comparison_table_service`). Este widget recebe
um `ComparisonTable` pronto e apenas o exibe, incluindo a frase que explica
por que são aquelas linhas — sem ela, o regime de extremos passa por tabela
completa e uma amostra enviesada é lida como se fosse o conjunto.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.plot_number_format import formatar_numero
from faceaudit.services.score_comparison_table_service import (
    CLASSE_GENUINO,
    CLASSE_IMPOSTOR,
    MODO_CASO,
    ComparisonRow,
    ComparisonTable,
)
from interface.theme import apply_themed_stylesheet, current_token_fn

from ..widget_helpers import WrappedLabel


# Lado da miniatura, em pixels. Grande o bastante para reconhecer a face e
# pequeno o bastante para caber cem linhas numa coluna de painel.
_LADO_MINIATURA = 72

# Papel do item que guarda o índice da linha NO BACKEND. A tabela é
# ordenável, então o índice visual deixa de corresponder ao da lista assim
# que o usuário clica num cabeçalho.
_PAPEL_INDICE_DA_LINHA = int(QtCore.Qt.ItemDataRole.UserRole) + 1
_ALTURA_DA_LINHA = _LADO_MINIATURA + 8


class _NumericItem(QtWidgets.QTableWidgetItem):
    """Célula que ordena pelo valor, não pelo texto.

    Com o texto em convenção brasileira, a ordenação alfabética do
    `QTableWidgetItem` põe "0,9" antes de "0,85" — exatamente o erro que
    uma tabela de escores não pode cometer.
    """

    def __init__(self, texto: str, valor: float):
        super().__init__(texto)
        self._valor = float(valor)
        self.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

    def __lt__(self, outro) -> bool:  # type: ignore[override]
        if isinstance(outro, _NumericItem):
            return self._valor < outro._valor
        return super().__lt__(outro)


class ScoreComparisonTableWidget(QtWidgets.QWidget):
    """Nota explicativa + tabela de comparações."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        table: Optional[ComparisonTable] = None,
    ):
        super().__init__(parent)
        self._table: Optional[ComparisonTable] = None
        # Itens das células de miniatura, na ordem em que foram pedidos ao
        # worker. Guardar o item (e não a coordenada) é o que mantém o
        # pareamento quando o usuário ordena a tabela durante a carga.
        self._itens_de_miniatura: list = []
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[QtCore.QObject] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self._note = WrappedLabel("", parent=self)
        apply_themed_stylesheet(
            self._note,
            lambda t: (
                "font-size: 11px; color: " + t("TEXT_MUTED") + ";"
                " background: " + t("SURFACE_RAISED") + ";"
                " border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                " padding: 6px 8px;"
            ),
        )
        layout.addWidget(self._note)

        self._grid = QtWidgets.QTableWidget(parent=self)
        self._grid.setAlternatingRowColors(True)
        self._grid.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._grid.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._grid.verticalHeader().setVisible(False)
        # Duplo clique abre a comparação ampliada. Clique simples continua
        # servindo para selecionar e copiar, que é o que se espera de uma
        # tabela.
        self._grid.cellDoubleClicked.connect(self._on_linha_ativada)
        self._grid.setToolTip(
            "Clique duas vezes numa linha para ver as duas faces lado a lado."
        )
        self._grid.setSortingEnabled(True)
        self._grid.setWordWrap(False)
        apply_themed_stylesheet(
            self._grid,
            lambda t: (
                "QTableWidget {"
                "  background: " + t("SURFACE") + "; color: " + t("TEXT") + ";"
                "  border: 1px solid " + t("BORDER") + "; border-radius: 4px;"
                "  gridline-color: " + t("BORDER") + "; font-size: 11px;"
                "}"
                "QTableWidget::item { padding: 2px 6px; }"
                "QTableWidget::item:selected {"
                "  background: " + t("HIGHLIGHT") + ";"
                # `color: inherit` resolveria para preto no tema escuro.
                "  color: " + t("HIGHLIGHT_TEXT") + ";"
                "}"
                "QHeaderView::section {"
                "  background: " + t("SURFACE_ALT") + "; color: " + t("TEXT") + ";"
                "  border: none; border-bottom: 1px solid " + t("BORDER") + ";"
                "  padding: 4px 6px; font-weight: bold; font-size: 11px;"
                "}"
            ),
        )
        layout.addWidget(self._grid, 1)

        if table is not None:
            self.set_table(table)
        else:
            self.show_placeholder()

    # ── API pública ──────────────────────────────────────────────────────

    def _on_linha_ativada(self, linha: int, _coluna: int) -> None:
        """Abre a comparação da linha clicada.

        A linha é encontrada pelo item, e não pelo índice: a tabela é
        ordenável, então o índice visual não é o índice do backend depois
        que o usuário clica num cabeçalho.
        """
        if self._table is None:
            return
        chave = self._grid.item(linha, 0)
        if chave is None:
            return
        indice = chave.data(_PAPEL_INDICE_DA_LINHA)
        if indice is None or not (0 <= int(indice) < len(self._table.rows)):
            return
        from ..face_comparison_dialog import FaceComparisonDialog

        dialogo = FaceComparisonDialog(self._table.rows[int(indice)], parent=self)
        dialogo.exec()

    def show_placeholder(self, mensagem: str = "") -> None:
        self._parar_miniaturas()
        self._table = None
        self._grid.setSortingEnabled(False)
        self._grid.clear()
        self._grid.setRowCount(0)
        self._grid.setColumnCount(0)
        self._grid.setSortingEnabled(True)
        self._note.setText(
            mensagem or "Execute a instrução para ver as comparações."
        )

    def set_table(self, table: Optional[ComparisonTable]) -> None:
        if table is None:
            self.show_placeholder()
            return
        self._parar_miniaturas()
        self._table = table
        self._note.setText(table.note)

        # Com as imagens localizadas a tabela mostra as duas faces e o
        # escore: num confronto é essa a comparação, e identidade e nome de
        # arquivo cabem na dica do mouse. Sem as imagens não há o que
        # mostrar além do texto, então o formato antigo continua valendo —
        # para instrução cujo dataset de origem não é alcançável a partir
        # dela (PAIR_MINING, SCORE_CALIBRATION) e para dataset movido.
        if table.has_images:
            self._montar_com_miniaturas(table)
        else:
            self._montar_em_texto(table)

        cabecalho = self._grid.horizontalHeader()
        cabecalho.setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        cabecalho.setStretchLastSection(True)

    # ── Formatos ─────────────────────────────────────────────────────────

    def _montar_em_texto(self, table: ComparisonTable) -> None:
        colunas = [
            "Identidade (probe)", "Face (probe)",
            "Identidade (referência)", "Face (referência)",
            "Escore", "Classe",
        ]
        if table.has_log_lr:
            colunas.append("log-LR")
        self._preparar_grade(colunas, len(table.rows))

        for linha, row in enumerate(table.rows):
            primeira = QtWidgets.QTableWidgetItem(row.probe_identity)
            primeira.setData(_PAPEL_INDICE_DA_LINHA, linha)
            self._grid.setItem(linha, 0, primeira)
            self._grid.setItem(linha, 1, QtWidgets.QTableWidgetItem(row.probe_face))
            self._grid.setItem(
                linha, 2, QtWidgets.QTableWidgetItem(row.reference_identity)
            )
            self._grid.setItem(
                linha, 3, QtWidgets.QTableWidgetItem(row.reference_face)
            )
            self._grid.setItem(
                linha, 4, _NumericItem(formatar_numero(row.score, 4), row.score)
            )
            self._grid.setItem(linha, 5, self._item_de_classe(row))
            if table.has_log_lr:
                self._grid.setItem(linha, 6, self._item_de_log_lr(row))
        self._grid.setSortingEnabled(True)

    def _montar_com_miniaturas(self, table: ComparisonTable) -> None:
        # Em conjunto de caso a classe é "Caso" em toda linha, e uma coluna
        # de valor constante só rouba largura das miniaturas.
        mostra_classe = table.mode != MODO_CASO
        colunas = ["Probe", "Referência", "Escore"]
        if mostra_classe:
            colunas.append("Classe")
        if table.has_log_lr:
            colunas.append("log-LR")
        self._preparar_grade(colunas, len(table.rows))
        self._grid.setIconSize(QtCore.QSize(_LADO_MINIATURA, _LADO_MINIATURA))

        caminhos: list = []
        for linha, row in enumerate(table.rows):
            for coluna, lado in ((0, "probe"), (1, "reference")):
                item = self._item_de_face(row, lado)
                if coluna == 0:
                    item.setData(_PAPEL_INDICE_DA_LINHA, linha)
                self._grid.setItem(linha, coluna, item)
                caminho = getattr(row, f"{lado}_image_path")
                if caminho:
                    # O índice na lista é o que casa a miniatura com a
                    # célula; guardar o ITEM e não (linha, coluna) é o que
                    # mantém o pareamento quando o usuário ordena a tabela
                    # enquanto as imagens ainda chegam.
                    self._itens_de_miniatura.append(item)
                    caminhos.append(str(caminho))
            self._grid.setItem(
                linha, 2, _NumericItem(formatar_numero(row.score, 4), row.score)
            )
            proxima = 3
            if mostra_classe:
                self._grid.setItem(linha, proxima, self._item_de_classe(row))
                proxima += 1
            if table.has_log_lr:
                self._grid.setItem(linha, proxima, self._item_de_log_lr(row))
            self._grid.setRowHeight(linha, _ALTURA_DA_LINHA)

        self._grid.setSortingEnabled(True)
        if caminhos:
            self._carregar_miniaturas(caminhos)

    # ── Células ──────────────────────────────────────────────────────────

    def _preparar_grade(self, colunas: list, n_linhas: int) -> None:
        # Preencher com a ordenação ligada reordena a cada inserção e
        # embaralha as linhas que o backend já ordenou.
        self._grid.setSortingEnabled(False)
        self._grid.clear()
        self._itens_de_miniatura = []
        self._grid.setColumnCount(len(colunas))
        self._grid.setHorizontalHeaderLabels(colunas)
        self._grid.setRowCount(n_linhas)

    def _item_de_face(self, row: ComparisonRow, lado: str) -> QtWidgets.QTableWidgetItem:
        identidade = getattr(row, f"{lado}_identity")
        arquivo = getattr(row, f"{lado}_face")
        caminho = getattr(row, f"{lado}_image_path")
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        if not caminho:
            # Sem imagem a célula mostra o texto, em vez de ficar vazia e
            # parecer que a comparação não tem aquele lado.
            item.setText(arquivo)
        linhas = [f"Identidade: {identidade}", f"Arquivo: {arquivo}"]
        hash_ = getattr(row, f"{lado}_hash")
        if hash_:
            linhas.append(f"Hash: {hash_[:16]}…")
        if caminho:
            linhas.append(str(caminho))
        item.setToolTip("\n".join(linhas))
        return item

    def _item_de_classe(self, row: ComparisonRow) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(row.classe_label)
        t = current_token_fn()
        cor = {
            CLASSE_GENUINO: t("SUCCESS"),
            CLASSE_IMPOSTOR: t("DANGER"),
        }.get(row.classe)
        if cor is not None:
            item.setForeground(QtGui.QColor(cor))
        return item

    def _item_de_log_lr(self, row: ComparisonRow) -> QtWidgets.QTableWidgetItem:
        if row.log_lr is None:
            return QtWidgets.QTableWidgetItem("—")
        return _NumericItem(formatar_numero(row.log_lr, 3), row.log_lr)

    # ── Miniaturas ───────────────────────────────────────────────────────

    def _carregar_miniaturas(self, caminhos: list) -> None:
        """Decodifica as miniaturas numa thread própria.

        Cem linhas são duzentas imagens, e várias delas são fotos de
        megapixels: decodificar na thread da interface congelaria a janela
        exatamente enquanto o usuário tenta olhar as faces.
        """
        from interface.tabs.dataset_manager.workers.thumbnail_load_worker import (
            ThumbnailLoadWorker,
        )

        self._thread = QtCore.QThread(self)
        self._worker = ThumbnailLoadWorker(caminhos, lado=_LADO_MINIATURA)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.thumbnail_ready.connect(self._on_miniatura_pronta)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    @QtCore.pyqtSlot(int, QtGui.QImage)
    def _on_miniatura_pronta(self, indice: int, imagem: QtGui.QImage) -> None:
        if imagem.isNull():
            return  # imagem ilegível — a célula fica com a dica do mouse
        if not (0 <= indice < len(self._itens_de_miniatura)):
            return
        item = self._itens_de_miniatura[indice]
        try:
            item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(imagem)))
        except RuntimeError:
            # A tabela foi remontada entre o pedido e a resposta; o item
            # desta célula já não existe.
            pass

    def _parar_miniaturas(self) -> None:
        """Cancela o carregamento em curso antes de remontar a tabela."""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
        self._worker = None
        self._thread = None
        self._itens_de_miniatura = []

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt API
        self._parar_miniaturas()
        super().closeEvent(event)
