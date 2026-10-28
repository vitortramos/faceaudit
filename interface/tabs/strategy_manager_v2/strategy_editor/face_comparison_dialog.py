"""As duas faces de uma comparação, lado a lado, com os pontos e o escore.

A tabela mostra de onde veio cada escore; esta janela é onde se confere a
comparação. Num confronto é a figura do exame: as duas faces no mesmo
tamanho na tela, os cinco pontos do detector marcados em cada uma, e o
número que o sistema produziu entre elas.

Os pontos vêm do dataset, não de uma detecção feita agora. A diferença
importa: são os pontos que definiram o alinhamento de onde saiu o
embedding, e portanto descrevem a face que o sistema de fato comparou.
Redetectar na hora poderia marcar outra face do quadro e ninguém notaria.
Quando o dataset não os guarda, a janela mostra as imagens sem marcação e
diz por quê, em vez de inventar.

O desenho é em `QPainter` sobre a imagem, que o projeto trata como desenho
e não como chrome, daí as cores fixas: a caixa e os pontos precisam ser
legíveis sobre pele clara e escura, sobre fundo claro e escuro, nos dois
temas da interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.plot_number_format import formatar_numero
from faceaudit.services.score_comparison_table_service import (
    CLASSE_GENUINO,
    CLASSE_IMPOSTOR,
    ComparisonRow,
)
from interface.theme import apply_themed_stylesheet, current_token_fn

from .widget_helpers import WrappedLabel
from interface.oriented_image import ler_pixmap_orientado

# Cores da marcação. Fixas de propósito: têm de ser legíveis sobre pele
# clara e escura e sobre fundo de qualquer cor, nos dois temas.
_COR_CAIXA = QtGui.QColor(255, 200, 0)
_COR_PONTO = QtGui.QColor(0, 140, 255)
_COR_CONTORNO_PONTO = QtGui.QColor(255, 255, 255)

# Quanto da caixa da face entra no enquadramento. 1,0 seria a caixa nua,
# sem cabelo nem queixo; 1,8 traz o contorno do rosto e um pouco de
# entorno, que é o que se olha numa comparação.
_FOLGA_DO_ENQUADRAMENTO = 1.8

# Lado da miniatura das faces agregadas, em pixels. Lidas já reduzidas:
# um template de vinte faces não deve trazer vinte fotos de megapixels à
# memória para desenhar um mosaico de miniaturas.
_LADO_DA_MINIATURA = 320


class _Montagem(QtWidgets.QWidget):
    """As faces que formaram um template agregado, em grade.

    Um template é a média de N faces e não corresponde a imagem nenhuma,
    mas as faces que o formaram existem — e num exame é isso que se quer
    ver. Elas ficam na coluna de proveniência que a agregação grava.

    A grade é quadrada o bastante para caber o número de faces, e as
    miniaturas são lidas já reduzidas: um template de vinte faces não
    deve trazer vinte fotos de megapixels à memória.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmaps: list = []
        self._ausentes = 0
        self.setMinimumSize(240, 240)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_faces(self, caminhos: Sequence[str]) -> None:
        self._pixmaps = []
        self._ausentes = 0
        for caminho in caminhos:
            pixmap = self._ler_reduzida(caminho)
            if pixmap is None:
                self._ausentes += 1
            else:
                self._pixmaps.append(pixmap)
        self.update()

    @staticmethod
    def _ler_reduzida(caminho: str) -> Optional[QtGui.QPixmap]:
        p = Path(str(caminho))
        if not p.is_file():
            return None
        leitor = QtGui.QImageReader(str(p))
        leitor.setAutoTransform(True)
        tamanho = leitor.size()
        if tamanho.isValid() and tamanho.width() > 0:
            escala = min(
                _LADO_DA_MINIATURA / tamanho.width(),
                _LADO_DA_MINIATURA / tamanho.height(),
                1.0,
            )
            leitor.setScaledSize(
                QtCore.QSize(
                    max(1, int(tamanho.width() * escala)),
                    max(1, int(tamanho.height() * escala)),
                )
            )
        imagem = leitor.read()
        if imagem.isNull():
            return None
        return QtGui.QPixmap.fromImage(imagem)

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        pintor = QtGui.QPainter(self)
        pintor.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        t = current_token_fn()
        pintor.fillRect(self.rect(), QtGui.QColor(t("SURFACE_RAISED")))

        if not self._pixmaps:
            pintor.setPen(QtGui.QColor(t("TEXT_MUTED")))
            fonte = pintor.font()
            fonte.setItalic(True)
            pintor.setFont(fonte)
            pintor.drawText(
                self.rect(),
                int(
                    QtCore.Qt.AlignmentFlag.AlignCenter
                    | QtCore.Qt.TextFlag.TextWordWrap
                ),
                "As imagens das faces agregadas não foram encontradas.",
            )
            return

        import math

        n = len(self._pixmaps)
        colunas = max(1, math.ceil(math.sqrt(n)))
        linhas = max(1, math.ceil(n / colunas))
        folga = 4
        largura = (self.width() - folga * (colunas + 1)) / colunas
        altura = (self.height() - folga * (linhas + 1)) / linhas
        lado = max(8.0, min(largura, altura))

        for indice, pixmap in enumerate(self._pixmaps):
            coluna = indice % colunas
            linha = indice // colunas
            x = folga + coluna * (lado + folga)
            y = folga + linha * (lado + folga)
            escala = min(lado / pixmap.width(), lado / pixmap.height())
            largura_f = pixmap.width() * escala
            altura_f = pixmap.height() * escala
            destino = QtCore.QRectF(
                x + (lado - largura_f) / 2, y + (lado - altura_f) / 2,
                largura_f, altura_f,
            )
            pintor.drawPixmap(destino, pixmap, QtCore.QRectF(pixmap.rect()))


class _FacePanel(QtWidgets.QWidget):
    """Uma face: a imagem, a marcação por cima e a legenda embaixo."""

    def __init__(self, titulo: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._bbox = None
        self._keypoints = None
        self._mensagem = ""
        self._marcar = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._titulo = QtWidgets.QLabel(titulo, parent=self)
        self._titulo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self._titulo,
            lambda t: (
                "font-weight: bold; font-size: 12px; color: " + t("TEXT")
                + "; padding: 2px;"
            ),
        )
        layout.addWidget(self._titulo)

        # Pilha: face única (índice 0) ou mosaico das faces de um template
        # (índice 1). Um template não tem imagem própria, mas as faces que
        # o formaram têm.
        self._pilha = QtWidgets.QStackedWidget(parent=self)
        self._tela = _Tela(parent=self._pilha)
        self._montagem = _Montagem(parent=self._pilha)
        self._pilha.addWidget(self._tela)
        self._pilha.addWidget(self._montagem)
        layout.addWidget(self._pilha, 1)

        self._legenda = WrappedLabel("", parent=self)
        self._legenda.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self._legenda,
            lambda t: "font-size: 11px; color: " + t("TEXT_MUTED") + ";",
        )
        layout.addWidget(self._legenda)

    def set_face(
        self,
        caminho: Optional[str],
        bbox,
        keypoints,
        identidade: str,
        arquivo: str,
        mensagem_sem_imagem: str = "",
        provenance=None,
    ) -> None:
        pixmap = None
        mensagem = mensagem_sem_imagem
        if provenance is not None:
            # Template agregado: não há foto própria a mostrar, mas as
            # faces que o formaram existem e é isso que se quer ver.
            self._pilha.setCurrentWidget(self._montagem)
            self._montagem.set_faces(provenance.paths)
            self._legenda.setText(f"{identidade}\n{arquivo}")
            return
        self._pilha.setCurrentWidget(self._tela)
        if mensagem:
            self._tela.set_conteudo(None, None, None, mensagem)
            self._legenda.setText(f"{identidade}\n{arquivo}")
            return
        if caminho and Path(caminho).is_file():
            carregado = ler_pixmap_orientado(caminho)
            if carregado.isNull():
                mensagem = "Imagem ilegível."
            else:
                pixmap = carregado
        elif caminho:
            mensagem = "Imagem não está no caminho que o dataset registra."
        else:
            mensagem = "O dataset de origem não foi localizado."

        self._tela.set_conteudo(pixmap, bbox, keypoints, mensagem)
        self._legenda.setText(f"{identidade}\n{arquivo}")

    def set_marcar(self, marcar: bool) -> None:
        self._tela.set_marcar(marcar)

    def set_enquadrar_na_face(self, enquadrar: bool) -> None:
        self._tela.set_enquadrar_na_face(enquadrar)


class _Tela(QtWidgets.QWidget):
    """A área de pintura: imagem ajustada e marcação na mesma escala."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._bbox = None
        self._keypoints = None
        self._mensagem = ""
        self._marcar = True
        self._enquadrar_na_face = False
        self.setMinimumSize(240, 240)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_conteudo(self, pixmap, bbox, keypoints, mensagem) -> None:
        self._pixmap = pixmap
        self._bbox = bbox
        self._keypoints = keypoints
        self._mensagem = mensagem
        self.update()

    def set_marcar(self, marcar: bool) -> None:
        self._marcar = bool(marcar)
        self.update()

    def set_enquadrar_na_face(self, enquadrar: bool) -> None:
        self._enquadrar_na_face = bool(enquadrar)
        self.update()

    def _recorte_de_origem(self) -> QtCore.QRectF:
        """A parte da imagem que vai preencher a tela.

        Sem enquadramento é a imagem inteira. Com enquadramento é a caixa
        da face com folga, e é isso que põe as duas faces no mesmo tamanho
        apesar de virem de fotos com resoluções e distâncias
        completamente diferentes. Sem isso, uma foto de documento e um
        quadro de vídeo aparecem lado a lado com rostos de tamanhos
        incomparáveis, justamente na janela que existe para compará-los.
        """
        inteira = QtCore.QRectF(self._pixmap.rect())
        if not self._enquadrar_na_face or self._bbox is None:
            return inteira
        try:
            x1, y1, x2, y2 = [float(v) for v in list(self._bbox)[:4]]
        except (TypeError, ValueError):
            return inteira
        largura, altura = x2 - x1, y2 - y1
        if largura <= 0 or altura <= 0:
            return inteira
        cx, cy = x1 + largura / 2.0, y1 + altura / 2.0
        meia_l = largura * _FOLGA_DO_ENQUADRAMENTO / 2.0
        meia_a = altura * _FOLGA_DO_ENQUADRAMENTO / 2.0
        recorte = QtCore.QRectF(
            cx - meia_l, cy - meia_a, meia_l * 2, meia_a * 2,
        ).intersected(inteira)
        # Preso à imagem: pedir pixel que não existe deixaria faixa vazia
        # de um lado e tiraria a face do centro.
        return recorte if recorte.width() > 0 and recorte.height() > 0 else inteira

    def _transformacao(self):
        """`(escala, dx, dy)` que leva coordenada de imagem para a tela.

        A marcação usa exatamente a mesma transformação da imagem, e é isso
        que mantém os pontos sobre a face em qualquer tamanho de janela e
        em qualquer enquadramento.
        """
        if self._pixmap is None or self._pixmap.width() <= 0:
            return 1.0, 0.0, 0.0
        area = self.rect()
        origem = self._recorte_de_origem()
        if origem.width() <= 0 or origem.height() <= 0:
            return 1.0, 0.0, 0.0
        escala = min(
            area.width() / origem.width(), area.height() / origem.height(),
        )
        dx = (area.width() - origem.width() * escala) / 2.0 - origem.x() * escala
        dy = (area.height() - origem.height() * escala) / 2.0 - origem.y() * escala
        return escala, dx, dy

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt API
        pintor = QtGui.QPainter(self)
        pintor.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pintor.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)

        if self._pixmap is None:
            t = current_token_fn()
            pintor.fillRect(self.rect(), QtGui.QColor(t("SURFACE_RAISED")))
            pintor.setPen(QtGui.QColor(t("TEXT_MUTED")))
            fonte = pintor.font()
            fonte.setItalic(True)
            pintor.setFont(fonte)
            pintor.drawText(
                self.rect(),
                int(
                    QtCore.Qt.AlignmentFlag.AlignCenter
                    | QtCore.Qt.TextFlag.TextWordWrap
                ),
                self._mensagem or "Sem imagem.",
            )
            return

        escala, dx, dy = self._transformacao()
        origem = self._recorte_de_origem()
        destino = QtCore.QRectF(
            origem.x() * escala + dx, origem.y() * escala + dy,
            origem.width() * escala, origem.height() * escala,
        )
        # Recorta ao painel: com enquadramento na face, o resto da foto
        # fica fora da tela em vez de vazar por cima do painel vizinho.
        pintor.setClipRect(self.rect())
        pintor.drawPixmap(destino, self._pixmap, origem)

        if not self._marcar:
            return

        if self._bbox is not None and len(self._bbox) >= 4:
            x1, y1, x2, y2 = [float(v) for v in list(self._bbox)[:4]]
            caneta = QtGui.QPen(_COR_CAIXA)
            caneta.setWidthF(max(1.5, 2.0))
            pintor.setPen(caneta)
            pintor.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            pintor.drawRect(
                QtCore.QRectF(
                    x1 * escala + dx, y1 * escala + dy,
                    (x2 - x1) * escala, (y2 - y1) * escala,
                )
            )

        if self._keypoints:
            raio = max(2.5, min(destino.width(), destino.height()) * 0.012)
            caneta = QtGui.QPen(_COR_CONTORNO_PONTO)
            caneta.setWidthF(1.2)
            pintor.setPen(caneta)
            pintor.setBrush(QtGui.QBrush(_COR_PONTO))
            for par in self._keypoints:
                px, py = float(par[0]), float(par[1])
                pintor.drawEllipse(
                    QtCore.QPointF(px * escala + dx, py * escala + dy),
                    raio, raio,
                )


class FaceComparisonDialog(QtWidgets.QDialog):
    """As duas faces de uma comparação, com o escore entre elas."""

    def __init__(
        self,
        row: ComparisonRow,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._row = row
        self.setWindowTitle("Comparação")
        self.resize(1000, 640)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        layout.addWidget(self._construir_cabecalho())

        faces = QtWidgets.QHBoxLayout()
        faces.setSpacing(12)
        self._probe = _FacePanel("Probe", parent=self)
        self._referencia = _FacePanel("Referência", parent=self)
        faces.addWidget(self._probe, 1)
        faces.addWidget(self._referencia, 1)
        layout.addLayout(faces, 1)

        layout.addWidget(self._construir_rodape())

        self._probe.set_face(
            row.probe_image_path, row.probe_bbox, row.probe_keypoints,
            row.probe_identity, row.probe_face,
            self._mensagem_de_template("probe"),
            provenance=row.probe_provenance,
        )
        self._referencia.set_face(
            row.reference_image_path, row.reference_bbox,
            row.reference_keypoints, row.reference_identity, row.reference_face,
            self._mensagem_de_template("reference"),
            provenance=row.reference_provenance,
        )
        self._on_enquadrar(self._check_enquadrar.isChecked())

    def _mensagem_de_template(self, lado: str) -> str:
        """Texto do painel quando aquele lado é um template agregado."""
        if not getattr(self._row, f"{lado}_is_template"):
            return ""
        from faceaudit.services.aggregated_template_identity import (
            template_tooltip,
        )

        return template_tooltip(
            getattr(self._row, f"{lado}_hash"),
            getattr(self._row, f"{lado}_face"),
        )

    # ── Montagem ─────────────────────────────────────────────────────────

    def _construir_cabecalho(self) -> QtWidgets.QWidget:
        row = self._row
        caixa = QtWidgets.QFrame(parent=self)
        apply_themed_stylesheet(
            caixa,
            lambda t: (
                "QFrame { background: " + t("SURFACE_RAISED")
                + "; border: 1px solid " + t("BORDER")
                + "; border-radius: 4px; }"
            ),
        )
        linha = QtWidgets.QHBoxLayout(caixa)
        linha.setContentsMargins(12, 8, 12, 8)
        linha.setSpacing(18)

        escore = QtWidgets.QLabel(
            f"Escore  {formatar_numero(row.score, 4)}", parent=caixa,
        )
        apply_themed_stylesheet(
            escore,
            lambda t: "font-size: 15px; font-weight: bold; color: " + t("TEXT") + ";",
        )
        linha.addWidget(escore)

        if row.log_lr is not None:
            lr = QtWidgets.QLabel(
                f"log-LR  {formatar_numero(row.log_lr, 3)}", parent=caixa,
            )
            apply_themed_stylesheet(
                lr,
                lambda t: "font-size: 13px; color: " + t("TEXT") + ";",
            )
            linha.addWidget(lr)

        classe = QtWidgets.QLabel(row.classe_label, parent=caixa)
        cor = {
            CLASSE_GENUINO: "SUCCESS",
            CLASSE_IMPOSTOR: "DANGER",
        }.get(row.classe, "TEXT_MUTED")
        apply_themed_stylesheet(
            classe,
            lambda t: "font-size: 13px; font-weight: bold; color: " + t(cor) + ";",
        )
        linha.addWidget(classe)
        linha.addStretch(1)

        self._check_enquadrar = QtWidgets.QCheckBox(
            "Enquadrar pela face", parent=caixa,
        )
        self._check_enquadrar.setToolTip(
            "Põe as duas faces no mesmo tamanho na tela. Sem isso, uma foto "
            "de documento e um quadro de vídeo aparecem com rostos de "
            "tamanhos incomparáveis."
        )
        tem_caixa = bool(row.probe_bbox and row.reference_bbox)
        self._check_enquadrar.setChecked(tem_caixa)
        self._check_enquadrar.setEnabled(tem_caixa)
        self._check_enquadrar.toggled.connect(self._on_enquadrar)
        linha.addWidget(self._check_enquadrar)

        self._check_marcar = QtWidgets.QCheckBox("Marcar pontos", parent=caixa)
        self._check_marcar.setChecked(True)
        self._check_marcar.setEnabled(self._row.tem_geometria)
        self._check_marcar.toggled.connect(self._on_marcar)
        linha.addWidget(self._check_marcar)

        salvar = QtWidgets.QPushButton("💾 Salvar figura…", parent=caixa)
        salvar.clicked.connect(self._on_salvar)
        linha.addWidget(salvar)
        return caixa

    def _construir_rodape(self) -> QtWidgets.QWidget:
        if self._row.tem_geometria:
            texto = (
                "Os pontos são os do dataset, isto é, os que definiram o "
                "alinhamento de onde saiu o embedding comparado."
            )
        else:
            texto = (
                "Este dataset não guarda os pontos da face. As imagens "
                "aparecem sem marcação, em vez de marcadas por uma detecção "
                "feita agora, que poderia apontar outra face do quadro."
            )
        nota = WrappedLabel(texto, parent=self)
        apply_themed_stylesheet(
            nota,
            lambda t: (
                "font-size: 11px; color: " + t("TEXT_MUTED")
                + "; font-style: italic;"
            ),
        )
        return nota

    # ── Ações ────────────────────────────────────────────────────────────

    def _on_marcar(self, marcar: bool) -> None:
        self._probe.set_marcar(marcar)
        self._referencia.set_marcar(marcar)

    def _on_enquadrar(self, enquadrar: bool) -> None:
        self._probe.set_enquadrar_na_face(enquadrar)
        self._referencia.set_enquadrar_na_face(enquadrar)

    def _on_salvar(self) -> None:
        """Grava a janela como figura, para ir ao laudo."""
        sugestao = (
            f"comparacao_{self._row.probe_face}_x_{self._row.reference_face}.png"
        )
        caminho, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Salvar figura da comparação", sugestao, "Imagem PNG (*.png)",
        )
        if not caminho:
            return
        imagem = self.grab()
        if not imagem.save(caminho, "PNG"):
            QtWidgets.QMessageBox.warning(
                self, "Não foi possível salvar",
                f"A figura não pôde ser gravada em:\n{caminho}",
            )
