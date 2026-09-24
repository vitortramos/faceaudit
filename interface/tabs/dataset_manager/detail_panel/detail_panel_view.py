from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTabWidget, QScrollArea, QProgressBar, QFormLayout
)
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet
from .quality_metric_widget import QualityMetricWidget


class DetailPanelView(QWidget):
    """
    View do Painel de Detalhes.
    Gerencia as abas (Geral, Qualidade) e a exibição da imagem.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Área da Imagem
        self.lbl_image = QLabel("Selecione uma face")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumHeight(150)
        self.lbl_image.setMaximumHeight(280)
        apply_themed_stylesheet(
            self.lbl_image,
            lambda t: "background-color: " + t("SURFACE_ALT")
            + "; border-bottom: 1px solid " + t("BORDER") + ";",
        )
        layout.addWidget(self.lbl_image)

        # Abas
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Aba 1: Geral
        self.tab_general = QWidget()
        self.scroll_general = QScrollArea() 
        self.scroll_general.setWidgetResizable(True)
        self.scroll_general.setWidget(self.tab_general)
        
        self.form_general = QFormLayout(self.tab_general)
        self.form_general.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_general.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        
        self.tabs.addTab(self.scroll_general, "Geral / Bio")
        
        # Aba 2: Qualidade ISO (OFIQ)
        self.scroll_ofiq = QScrollArea()
        self.scroll_ofiq.setWidgetResizable(True)
        self.container_ofiq = QWidget()
        
        self.layout_ofiq = QVBoxLayout(self.container_ofiq)
        self.layout_ofiq.setContentsMargins(5, 5, 5, 5)
        self.layout_ofiq.setSpacing(2)
        
        self.scroll_ofiq.setWidget(self.container_ofiq)
        self.tabs.addTab(self.scroll_ofiq, "Qualidade ISO")

        # Aba 3: Metadados embutidos (EXIF/XMP/IPTC) da foto original.
        # Árvore agrupada em vez de formulário: são dezenas a centenas de
        # tags, e o agrupamento por origem (EXIF, XMP, File, MakerNotes…)
        # é o que torna isso navegável.
        self.tab_exif = QWidget()
        layout_exif = QVBoxLayout(self.tab_exif)
        layout_exif.setContentsMargins(5, 5, 5, 5)
        layout_exif.setSpacing(4)

        self.txt_exif_filter = QtWidgets.QLineEdit(parent=self.tab_exif)
        self.txt_exif_filter.setPlaceholderText("Filtrar por tag ou valor…")
        self.txt_exif_filter.setClearButtonEnabled(True)
        self.txt_exif_filter.textChanged.connect(self._filtrar_exif)
        layout_exif.addWidget(self.txt_exif_filter)

        self.tree_exif = QtWidgets.QTreeWidget(parent=self.tab_exif)
        self.tree_exif.setColumnCount(2)
        self.tree_exif.setHeaderLabels(["Tag", "Valor"])
        self.tree_exif.setAlternatingRowColors(True)
        self.tree_exif.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.tree_exif.header().setStretchLastSection(True)
        layout_exif.addWidget(self.tree_exif, 1)

        self.lbl_exif_aviso = QLabel(parent=self.tab_exif)
        self.lbl_exif_aviso.setWordWrap(True)
        apply_themed_stylesheet(
            self.lbl_exif_aviso,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;",
        )
        layout_exif.addWidget(self.lbl_exif_aviso)

        self.tabs.addTab(self.tab_exif, "Metadados")
        self.clear_exif_metadata()

    # --- Metadados embutidos ---

    def set_exif_metadata(self, tags: dict):
        """Preenche a árvore com TODAS as tags lidas do arquivo.

        Agrupa por origem (o prefixo `GRUPO:` que o ExifTool emite). Tags
        sem prefixo caem em "Outros".
        """
        self.tree_exif.clear()
        self.txt_exif_filter.clear()
        if not tags:
            self.clear_exif_metadata()
            return

        grupos = {}
        for chave, valor in tags.items():
            texto = str(chave)
            grupo, _, nome = texto.partition(":")
            if not nome:
                grupo, nome = "Outros", texto
            grupos.setdefault(grupo, []).append((nome, valor))

        for grupo in sorted(grupos):
            itens = sorted(grupos[grupo], key=lambda kv: kv[0].lower())
            pai = QtWidgets.QTreeWidgetItem([f"{grupo} ({len(itens)})", ""])
            fonte = pai.font(0)
            fonte.setBold(True)
            pai.setFont(0, fonte)
            for nome, valor in itens:
                filho = QtWidgets.QTreeWidgetItem([nome, str(valor)])
                filho.setToolTip(1, str(valor))
                pai.addChild(filho)
            self.tree_exif.addTopLevelItem(pai)
            pai.setExpanded(True)

        self.tree_exif.resizeColumnToContents(0)
        total = sum(len(v) for v in grupos.values())
        self.lbl_exif_aviso.setText(
            f"{total} tag(s) lidas do arquivo. <b>São declarações do "
            "arquivo</b>, não medições: metadado embutido pode ser editado "
            "ou forjado."
        )

    def clear_exif_metadata(self, mensagem: str = ""):
        """Estado vazio — sem metadado, ou sem registro selecionado."""
        self.tree_exif.clear()
        self.lbl_exif_aviso.setText(
            mensagem
            or "Sem metadados embutidos para este registro. Eles existem "
               "apenas em datasets de fotos originais com a extração "
               "ligada — recorte e alinhamento gravam pixel novo, sem "
               "metadado de captura."
        )

    def _filtrar_exif(self, texto: str):
        """Esconde o que não casa; grupo sem filho visível some junto."""
        alvo = (texto or "").strip().lower()
        for i in range(self.tree_exif.topLevelItemCount()):
            grupo = self.tree_exif.topLevelItem(i)
            visiveis = 0
            for j in range(grupo.childCount()):
                filho = grupo.child(j)
                casa = (
                    not alvo
                    or alvo in filho.text(0).lower()
                    or alvo in filho.text(1).lower()
                )
                filho.setHidden(not casa)
                visiveis += int(casa)
            grupo.setHidden(visiveis == 0)
            if alvo:
                grupo.setExpanded(True)

    # --- API de Atualização Visual (Chamada pelo Controller) ---

    def set_image(self, pixmap: QtGui.QPixmap):
        """Atualiza o label com a imagem redimensionada."""
        if not pixmap.isNull():
            w = self.lbl_image.width()
            # Escala suave mantendo aspect ratio
            scaled = pixmap.scaled(w, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_image.setPixmap(scaled)
        else:
            self.reset_image_placeholder("Imagem inválida")

    def reset_image_placeholder(self, text="Selecione uma face"):
        """Reseta a área de imagem para o estado padrão (texto)."""
        self.lbl_image.clear()
        self.lbl_image.setText(text)

    def clear_general_info(self):
        """Remove todas as linhas do formulário geral."""
        while self.form_general.count():
            item = self.form_general.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_general_info(self, label: str, value: str):
        """Adiciona uma linha chave-valor na aba Geral."""
        lbl = QLabel(str(value))
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.form_general.addRow(f"<b>{label}:</b>", lbl)

    def clear_quality_metrics(self):
        """Remove todas as métricas da aba de qualidade."""
        while self.layout_ofiq.count():
            item = self.layout_ofiq.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_quality_metric(self, name: str, value: float):
        """Adiciona uma barra de qualidade."""
        self.layout_ofiq.addWidget(QualityMetricWidget(name, value))
        
    def add_quality_label(self, text: str):
        """Adiciona um texto simples na aba de qualidade (ex: avisos)."""
        self.layout_ofiq.addWidget(QLabel(text))
        
    def add_quality_stretch(self):
        """Empurra o conteúdo para o topo."""
        self.layout_ofiq.addStretch()