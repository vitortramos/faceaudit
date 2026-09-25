"""Tela da revisão das imagens sem face detectada.

Grade de miniaturas à esquerda, imagem grande da selecionada à direita.
A grade serve para varrer o conjunto; o painel grande é onde a decisão de
fato é tomada, porque numa miniatura de 150 px não se distingue uma foto
sem face de uma foto com face de perfil que o detector perdeu.

Só desenha e coleta cliques. A regra de quem pode ser incluído vem do
backend (`undetected_review_policy`) e chega pronta, como texto e como
booleano.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet, current_token_fn

_LADO_MINIATURA = 150


class UndetectedReviewView(QtWidgets.QDialog):
    """Grade + painel grande. Emite intenção; não decide nada."""

    selecao_mudou = QtCore.pyqtSignal(int)
    marcacao_mudou = QtCore.pyqtSignal(int, bool)
    marcar_todas = QtCore.pyqtSignal()
    desmarcar_todas = QtCore.pyqtSignal()

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        n_imagens: int = 0,
        n_total: int = 0,
        inclusao_permitida: bool = True,
        motivo: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Imagens sem face detectada")
        self.resize(1040, 660)
        self._inclusao_permitida = bool(inclusao_permitida)

        raiz = QtWidgets.QVBoxLayout(self)
        raiz.setSpacing(8)

        raiz.addWidget(self._cabecalho(n_imagens, n_total))
        raiz.addWidget(self._faixa_motivo(motivo))

        divisor = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        divisor.addWidget(self._painel_grade())
        divisor.addWidget(self._painel_imagem())
        divisor.setStretchFactor(0, 0)
        divisor.setStretchFactor(1, 1)
        divisor.setSizes([420, 620])
        raiz.addWidget(divisor, 1)

        raiz.addLayout(self._rodape())

    # ── Cabeçalho e faixa ────────────────────────────────────────────────

    def _cabecalho(self, n_imagens: int, n_total: int) -> QtWidgets.QWidget:
        caixa = QtWidgets.QFrame(parent=self)
        apply_themed_stylesheet(
            caixa,
            lambda t: (
                "QFrame { background: " + t("ACCENT_BG") + "; border: 1px solid "
                + t("ACCENT") + "; border-radius: 4px; }"
            ),
        )
        layout = QtWidgets.QVBoxLayout(caixa)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        titulo = QtWidgets.QLabel(
            f"<b>{n_imagens} imagem(ns) sem face detectada</b>", parent=caixa,
        )
        apply_themed_stylesheet(
            titulo, lambda t: "color: " + t("ACCENT_TEXT") + "; font-size: 13px;",
        )
        layout.addWidget(titulo)

        detalhe = QtWidgets.QLabel(
            "O detector foi tentado nas quatro orientações (como está, 90°, "
            "270° e 180°) e não encontrou face em nenhuma delas. "
            f"As outras {max(n_total - n_imagens, 0)} imagens já estão no dataset.",
            parent=caixa,
        )
        detalhe.setWordWrap(True)
        apply_themed_stylesheet(
            detalhe, lambda t: "color: " + t("ACCENT_TEXT") + "; font-size: 11px;",
        )
        layout.addWidget(detalhe)
        return caixa

    def _faixa_motivo(self, motivo: str) -> QtWidgets.QWidget:
        caixa = QtWidgets.QFrame(parent=self)
        permitida = self._inclusao_permitida
        apply_themed_stylesheet(
            caixa,
            lambda t: (
                "QFrame { background: "
                + t("SURFACE_ALT" if permitida else "WARNING_BG")
                + "; border-left: 3px solid "
                + t("PRIMARY" if permitida else "WARNING")
                + "; border-radius: 3px; }"
            ),
        )
        layout = QtWidgets.QHBoxLayout(caixa)
        layout.setContentsMargins(10, 6, 10, 6)

        self.lbl_motivo = QtWidgets.QLabel(
            ("" if permitida else "<b>Inclusão indisponível.</b> ") + motivo,
            parent=caixa,
        )
        self.lbl_motivo.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_motivo,
            lambda t: "color: " + t("TEXT_MUTED" if permitida else "TEXT")
            + "; font-size: 11px;",
        )
        layout.addWidget(self.lbl_motivo, 1)
        return caixa

    # ── Grade ────────────────────────────────────────────────────────────

    def _painel_grade(self) -> QtWidgets.QWidget:
        caixa = QtWidgets.QWidget(parent=self)
        layout = QtWidgets.QVBoxLayout(caixa)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.lista = QtWidgets.QListWidget(parent=caixa)
        self.lista.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.lista.setIconSize(QtCore.QSize(_LADO_MINIATURA, _LADO_MINIATURA))
        self.lista.setGridSize(
            QtCore.QSize(_LADO_MINIATURA + 30, _LADO_MINIATURA + 46)
        )
        self.lista.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.lista.setMovement(QtWidgets.QListView.Movement.Static)
        self.lista.setWordWrap(True)
        self.lista.setUniformItemSizes(True)
        self.lista.setSpacing(2)
        # `color: inherit` em ::item:selected resolve para preto sobre o
        # fundo de seleção no tema escuro — daí o token explícito.
        apply_themed_stylesheet(
            self.lista,
            lambda t: (
                "QListWidget { background: " + t("SURFACE") + "; border: 1px solid "
                + t("BORDER") + "; }"
                "QListWidget::item { color: " + t("TEXT") + "; }"
                "QListWidget::item:selected { background: " + t("HIGHLIGHT")
                + "; color: " + t("HIGHLIGHT_TEXT") + "; }"
            ),
        )
        self.lista.currentRowChanged.connect(self._ao_trocar_linha)
        self.lista.itemChanged.connect(self._ao_marcar_item)
        layout.addWidget(self.lista, 1)
        return caixa

    # ── Imagem grande ────────────────────────────────────────────────────

    def _painel_imagem(self) -> QtWidgets.QWidget:
        caixa = QtWidgets.QWidget(parent=self)
        layout = QtWidgets.QVBoxLayout(caixa)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.lbl_imagem = QtWidgets.QLabel(parent=caixa)
        self.lbl_imagem.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_imagem.setMinimumSize(380, 340)
        apply_themed_stylesheet(
            self.lbl_imagem,
            lambda t: (
                "QLabel { background: " + t("SURFACE_ALT") + "; border: 1px solid "
                + t("BORDER") + "; color: " + t("TEXT_MUTED") + "; }"
            ),
        )
        layout.addWidget(self.lbl_imagem, 1)

        self.lbl_arquivo = QtWidgets.QLabel(parent=caixa)
        self.lbl_arquivo.setWordWrap(True)
        self.lbl_arquivo.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        apply_themed_stylesheet(
            self.lbl_arquivo, lambda t: "color: " + t("TEXT") + "; font-size: 11px;",
        )
        layout.addWidget(self.lbl_arquivo)

        self.chk_incluir = QtWidgets.QCheckBox(
            "Incluir esta imagem no dataset", parent=caixa,
        )
        self.chk_incluir.setEnabled(self._inclusao_permitida)
        if not self._inclusao_permitida:
            self.chk_incluir.setToolTip(self.lbl_motivo.text())
        self.chk_incluir.toggled.connect(self._ao_marcar_no_painel)
        layout.addWidget(self.chk_incluir)
        return caixa

    # ── Rodapé ───────────────────────────────────────────────────────────

    def _rodape(self) -> QtWidgets.QHBoxLayout:
        linha = QtWidgets.QHBoxLayout()

        self.btn_todas = QtWidgets.QPushButton("Incluir todas", parent=self)
        self.btn_todas.setEnabled(self._inclusao_permitida)
        self.btn_todas.clicked.connect(self.marcar_todas.emit)
        linha.addWidget(self.btn_todas)

        self.btn_nenhuma = QtWidgets.QPushButton("Descartar todas", parent=self)
        self.btn_nenhuma.clicked.connect(self.desmarcar_todas.emit)
        linha.addWidget(self.btn_nenhuma)

        linha.addStretch(1)

        self.lbl_contador = QtWidgets.QLabel(parent=self)
        apply_themed_stylesheet(
            self.lbl_contador, lambda t: "color: " + t("TEXT_MUTED") + ";",
        )
        linha.addWidget(self.lbl_contador)

        self.botoes = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok, parent=self,
        )
        self.botoes.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        ).setText("Concluir")
        self.botoes.accepted.connect(self.accept)
        linha.addWidget(self.botoes)
        return linha

    # ── População e estado ───────────────────────────────────────────────

    def popular(self, rotulos: List[str]) -> None:
        """Cria as linhas com placeholder; as miniaturas chegam depois."""
        self.lista.blockSignals(True)
        self.lista.clear()
        vazio = self._placeholder()
        for rotulo in rotulos:
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(vazio), rotulo)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            if not self._inclusao_permitida:
                # Visível e legível, mas não marcável — esconder a caixa
                # deixaria a impressão de que a escolha nem existe.
                item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            self.lista.addItem(item)
        self.lista.blockSignals(False)
        if rotulos:
            self.lista.setCurrentRow(0)

    def _placeholder(self) -> QtGui.QPixmap:
        pix = QtGui.QPixmap(_LADO_MINIATURA, _LADO_MINIATURA)
        t = current_token_fn()
        pix.fill(QtGui.QColor(t("SURFACE_ALT")))
        return pix

    def definir_miniatura(self, indice: int, imagem: QtGui.QImage) -> None:
        item = self.lista.item(indice)
        if item is None:
            return
        if imagem.isNull():
            item.setToolTip("A imagem não pôde ser lida.")
            return
        item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(imagem)))

    def mostrar_imagem(self, imagem: QtGui.QImage, descricao: str) -> None:
        self.lbl_arquivo.setText(descricao)
        if imagem.isNull():
            self.lbl_imagem.setPixmap(QtGui.QPixmap())
            self.lbl_imagem.setText(
                "A imagem não pôde ser lida.\nArquivo corrompido, formato não "
                "suportado ou caminho inacessível."
            )
            return
        self.lbl_imagem.setText("")
        self._imagem_atual = imagem
        self._redesenhar_imagem()

    def _redesenhar_imagem(self) -> None:
        imagem = getattr(self, "_imagem_atual", None)
        if imagem is None or imagem.isNull():
            return
        self.lbl_imagem.setPixmap(
            QtGui.QPixmap.fromImage(imagem).scaled(
                self.lbl_imagem.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):    # noqa: N802 (assinatura do Qt)
        super().resizeEvent(event)
        self._redesenhar_imagem()

    def definir_marcacao(self, indice: int, marcada: bool) -> None:
        item = self.lista.item(indice)
        if item is None:
            return
        self.lista.blockSignals(True)
        item.setCheckState(
            QtCore.Qt.CheckState.Checked if marcada
            else QtCore.Qt.CheckState.Unchecked
        )
        self.lista.blockSignals(False)
        if self.lista.currentRow() == indice:
            self._sincronizar_caixa_do_painel(marcada)

    def _sincronizar_caixa_do_painel(self, marcada: bool) -> None:
        self.chk_incluir.blockSignals(True)
        self.chk_incluir.setChecked(marcada)
        self.chk_incluir.blockSignals(False)

    def atualizar_contador(self, n_incluir: int, n_total: int) -> None:
        n_fora = n_total - n_incluir
        self.lbl_contador.setText(
            f"{n_incluir} a incluir · {n_fora} a descartar"
        )

    # ── Cliques ──────────────────────────────────────────────────────────

    def _ao_trocar_linha(self, indice: int) -> None:
        if indice < 0:
            return
        item = self.lista.item(indice)
        if item is not None:
            self._sincronizar_caixa_do_painel(
                item.checkState() == QtCore.Qt.CheckState.Checked
            )
        self.selecao_mudou.emit(indice)

    def _ao_marcar_item(self, item: QtWidgets.QListWidgetItem) -> None:
        self.marcacao_mudou.emit(
            self.lista.row(item),
            item.checkState() == QtCore.Qt.CheckState.Checked,
        )

    def _ao_marcar_no_painel(self, marcada: bool) -> None:
        indice = self.lista.currentRow()
        if indice >= 0:
            self.marcacao_mudou.emit(indice, marcada)
