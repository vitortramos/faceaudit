# interface/tabs/dataset_manager/compare_dialog/compare_dialog_view.py
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QFrame, QPushButton, QTabWidget, QWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QTextEdit
)
from PyQt6.QtCore import Qt

from interface.theme import apply_themed_stylesheet, current_token_fn

class CompareDialogView(QDialog):
    """
    View do Diálogo de Comparação.
    Renderiza os gráficos de resumo (Interseção) e as tabelas detalhadas.
    Adaptada para receber dados do ComparisonResultDTO e CategoryDiff.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # O título será ajustado pelo Controller, mas definimos um padrão
        self.setWindowTitle("Comparativo de Datasets")
        self.resize(950, 650)
        self.setModal(True)
        
        # Nomes padrão (serão atualizados quando o controller chamar set_summary_data ou pelo init do controller)
        self.name_a = "Dataset A"
        self.name_b = "Dataset B"
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # TAB WIDGET
        self.tabs = QTabWidget()
        apply_themed_stylesheet(
            self.tabs,
            lambda t: (
                "QTabWidget::pane { border: 1px solid " + t("BORDER") + "; top: -1px; } "
                "QTabBar::tab { padding: 8px 12px; font-weight: bold; } "
                "QTabBar::tab:selected { background: " + t("SURFACE")
                + "; border-bottom: 2px solid " + t("PRIMARY") + "; }"
            ),
        )
        
        # ABA 1: RESUMO (Placeholder, será preenchida via código)
        self.tab_summary = QWidget()
        self.tab_summary.setLayout(QVBoxLayout()) 
        self.tabs.addTab(self.tab_summary, "📋 Resumo Geral")
        
        main_layout.addWidget(self.tabs)

        # BOTÃO FECHAR
        btn_close = QPushButton("Fechar")
        btn_close.setFixedWidth(120)
        btn_close.setStyleSheet("padding: 6px;")
        btn_close.clicked.connect(self.accept)
        main_layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignCenter)

    # --- API de Resumo (Aba 1) ---

    def set_summary_data(self, 
                         total_a, total_b, 
                         ids_a, ids_b, 
                         unique_ids_a, unique_ids_b, 
                         common_ids, 
                         # Parâmetros opcionais de imagem (caso o DTO forneça)
                         common_imgs=0, unique_imgs_a=0, unique_imgs_b=0,
                         # Parâmetros técnicos
                         common_cols=None, missing_cols=None):
        """
        Popula a aba de resumo com os dados estatísticos processados.
        """
        # Limpa layout anterior se houver
        layout = self.tab_summary.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            elif child.layout(): child.layout().deleteLater()

        # Header
        header = QLabel("Análise de Interseção (Identidades e Volumes)")
        apply_themed_stylesheet(
            header,
            lambda t: "font-size: 14px; font-weight: bold; margin: 10px 0; color: "
                      + t("TEXT") + ";",
        )
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Grid de Caixas (A | Interseção | B)
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(20)

        t = current_token_fn()

        # Box A
        box_a = self._create_dataset_box(
            self.name_a, t("PRIMARY_BG"), # Azul claro
            total_a, ids_a,
            unique_imgs_a, unique_ids_a, "Exclusivos aqui"
        )

        # Box Interseção
        box_common = self._create_intersection_box(
            common_imgs, common_ids
        )

        # Box B
        box_b = self._create_dataset_box(
            self.name_b, t("DANGER_BG"), # Vermelho claro
            total_b, ids_b,
            unique_imgs_b, unique_ids_b, "Exclusivos lá"
        )

        grid_layout.addWidget(box_a)
        grid_layout.addWidget(box_common)
        grid_layout.addWidget(box_b)
        
        layout.addLayout(grid_layout)

        # --- Seção Técnica (Colunas) ---
        if common_cols or missing_cols:
            layout.addSpacing(20)
            lbl_tech = QLabel("Detalhes Técnicos (Estrutura)")
            lbl_tech.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
            layout.addWidget(lbl_tech)
            
            tech_frame = QFrame()
            apply_themed_stylesheet(
                tech_frame,
                lambda t: "background-color: " + t("SURFACE_RAISED")
                          + "; border-radius: 4px; padding: 10px;",
            )
            tech_layout = QVBoxLayout(tech_frame)

            if common_cols:
                info_cols = ", ".join(common_cols)
                tech_layout.addWidget(QLabel(f"<b>Colunas em comum:</b> {info_cols}"))

            if missing_cols:
                missing_str = ", ".join(missing_cols)
                lbl_missing = QLabel(f"<b>⚠️ Ausentes em {self.name_b}:</b> {missing_str}")
                apply_themed_stylesheet(
                    lbl_missing, lambda t: "color: " + t("DANGER") + ";"
                )
                tech_layout.addWidget(lbl_missing)
            else:
                 tech_layout.addWidget(QLabel(f"✅ {self.name_b} possui todas as colunas de vetor de {self.name_a}."))

            layout.addWidget(tech_frame)

        layout.addStretch() # Empurra tudo para cima

    # --- API de Tabelas (Abas 2, 3...) ---

    def add_comparison_tab(self, tab_title: str, rows: list):
        """
        Adiciona uma nova aba contendo uma tabela comparativa baseada em CategoryDiff.
        
        Args:
            tab_title: Título da aba (ex: "Gênero")
            rows: Lista de objetos CategoryDiff
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Tabela
        table = QTableWidget()
        table.setColumnCount(4)
        
        # Cabeçalhos com nomes dos datasets
        headers = ["Categoria", self.name_a, self.name_b, "Diferença"]
        table.setHorizontalHeaderLabels(headers)
        
        # Configuração da Tabela
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents) # Categoria ajusta ao texto
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setRowCount(len(rows))
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

        t = current_token_fn()

        # Preenchimento
        for row_idx, diff_obj in enumerate(rows):
            # diff_obj é uma instância de CategoryDiff
            
            # 0. Categoria
            item_cat = QTableWidgetItem(str(diff_obj.category))
            item_cat.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            item_cat.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable) # Read-only
            table.setItem(row_idx, 0, item_cat)
            
            # 1. Valor A
            item_a = QTableWidgetItem(str(diff_obj.count_a))
            item_a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 1, item_a)
            
            # 2. Valor B
            item_b = QTableWidgetItem(str(diff_obj.count_b))
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row_idx, 2, item_b)
            
            # 3. Diferença (Formatada e Colorida)
            item_diff = QTableWidgetItem(diff_obj.diff_str) # Usa a property formatada
            item_diff.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            val_diff = diff_obj.diff
            if val_diff > 0:
                item_diff.setForeground(QtGui.QColor(t("SUCCESS"))) # Verde (Aumentou em B)
                item_diff.setBackground(QtGui.QColor(t("SUCCESS_BG")))
            elif val_diff < 0:
                item_diff.setForeground(QtGui.QColor(t("DANGER"))) # Vermelho (Diminuiu em B)
                item_diff.setBackground(QtGui.QColor(t("DANGER_BG")))
            else:
                item_diff.setForeground(QtGui.QColor(t("TEXT_MUTED"))) # Cinza (Igual)
            
            table.setItem(row_idx, 3, item_diff)

        layout.addWidget(table)
        
        # Ícone baseado no título (opcional)
        icon_map = {"Gênero": "⚥", "Etnia": "🌍", "Origem": "Source"}
        icon = icon_map.get(tab_title, "📊")
        
        self.tabs.addTab(tab, f"{icon} {tab_title}")

    # --- Helpers Visuais (Privados) ---

    def _create_dataset_box(self, name, color, total_imgs, total_ids, unique_imgs, unique_ids, unique_label):
        t = current_token_fn()
        group = QGroupBox(name)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {t("BORDER")};
                border-radius: 8px;
                background-color: {color};
                margin-top: 10px;
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        """)

        l = QVBoxLayout()
        l.setSpacing(10)

        # Seção Total
        lbl_total = QLabel("Total")
        lbl_total.setStyleSheet("font-size: 10pt; color: " + t("TEXT_MUTED") + ";")
        lbl_total.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info_total = QLabel(f"<span style='font-size: 12pt;'>🖼️ {total_imgs}</span> &nbsp;|&nbsp; <span style='font-size: 12pt;'>👤 {total_ids}</span>")
        info_total.setAlignment(Qt.AlignmentFlag.AlignCenter)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: " + t("BORDER") + "; height: 1px;")

        # Seção Exclusivos
        lbl_unique = QLabel(unique_label)
        lbl_unique.setStyleSheet("font-weight: bold; color: " + t("TEXT") + ";")
        lbl_unique.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        info_unique = QLabel(f"🖼️ {unique_imgs} <br> 👤 {unique_ids}")
        info_unique.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_unique.setStyleSheet("font-size: 11pt;")

        l.addWidget(lbl_total)
        l.addWidget(info_total)
        l.addWidget(line)
        l.addWidget(lbl_unique)
        l.addWidget(info_unique)
        l.addStretch()
        
        group.setLayout(l)
        return group

    def _create_intersection_box(self, n_common_imgs, n_common_ids):
        t = current_token_fn()
        group = QGroupBox("Interseção (Comum)")
        apply_themed_stylesheet(
            group,
            lambda t: (
                "QGroupBox { "
                "    font-weight: bold; "
                "    border: 1px solid " + t("ACCENT") + "; "
                "    border-radius: 8px; "
                "    background-color: " + t("ACCENT_BG") + "; "
                "    margin-top: 10px;"
                "}"
            ),
        )

        l = QVBoxLayout()

        lbl_icon = QLabel("🔗")
        lbl_icon.setStyleSheet("font-size: 32px; margin-bottom: 5px;")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Destaque para IDs
        txt_ids = f"<span style='font-size: 18pt; font-weight: bold; color: {t('PRIMARY')};'>{n_common_ids}</span>"
        lbl_ids = QLabel(f"{txt_ids}<br><span style='font-size: 10pt; color: {t('TEXT_MUTED')};'>Identidades Iguais</span>")
        lbl_ids.setAlignment(Qt.AlignmentFlag.AlignCenter)

        l.addStretch()
        l.addWidget(lbl_icon)
        l.addWidget(lbl_ids)

        # Se houver imagens idênticas (hash), mostra. Se for 0, mostra menor.
        if n_common_imgs > 0:
            l.addSpacing(10)
            txt_imgs = f"<span style='font-size: 12pt; font-weight: bold; color: {t('DANGER')};'>{n_common_imgs}</span>"
            lbl_imgs = QLabel(f"{txt_imgs} imagens duplicadas (Hash)")
            lbl_imgs.setAlignment(Qt.AlignmentFlag.AlignCenter)
            l.addWidget(lbl_imgs)
            
        l.addStretch()
        group.setLayout(l)
        return group