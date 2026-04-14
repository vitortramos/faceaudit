from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QPushButton, QHBoxLayout
)

class DatasetAnalysisView(QDialog):
    """
    View do Diálogo de Análise e Auditoria de Dataset.
    """
    def __init__(self, dataset_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Auditoria Forense - {dataset_name}")
        self.resize(850, 650)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        # Abas
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Aba 1: Integridade (Hashes)
        self.tab_hashes = QWidget()
        self.layout_hashes = QVBoxLayout(self.tab_hashes)
        self.tabs.addTab(self.tab_hashes, "📋 Integridade (Hashes)")

        # Aba 2: Piores Genuínos (Falsas Rejeições)
        self.tab_genuines = QWidget()
        self.layout_genuines = QVBoxLayout(self.tab_genuines)
        self.tabs.addTab(self.tab_genuines, "🔴 Piores Genuínos (FN)")

        # Aba 3: Piores Impostores (Falsos Aceites)
        self.tab_impostors = QWidget()
        self.layout_impostors = QVBoxLayout(self.tab_impostors)
        self.tabs.addTab(self.tab_impostors, "🟠 Piores Impostores (FP)")

        # Botões Inferiores
        btn_layout = QHBoxLayout()
        
        self.btn_run_mining = QPushButton("🔍 Iniciar Mineração Biométrica (Pesado)")
        self.btn_run_mining.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 6px;")
        
        self.btn_close = QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)

        btn_layout.addWidget(self.btn_run_mining)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)

        main_layout.addLayout(btn_layout)

    def build_hash_table(self, consistent_data: list, inconsistent_data: list):
        """
        Constrói as tabelas de auditoria de hashes na primeira aba.
        """
        # Limpa o layout caso seja chamado novamente
        while self.layout_hashes.count():
            child = self.layout_hashes.takeAt(0)
            if child.widget(): child.widget().deleteLater()

        # Secção Crítica (Inconsistentes)
        lbl_crit = QLabel(f"⚠️ Alertas Críticos (Múltiplas identidades para a mesma foto): {len(inconsistent_data)}")
        lbl_crit.setStyleSheet("color: #c62828; font-weight: bold; font-size: 14px;")
        self.layout_hashes.addWidget(lbl_crit)

        table_crit = QTableWidget()
        table_crit.setColumnCount(3)
        table_crit.setHorizontalHeaderLabels(["Hash do Arquivo", "Repetições", "Identidades Conflitantes"])
        table_crit.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table_crit.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table_crit.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.layout_hashes.addWidget(table_crit)

        # Preenche Inconsistentes
        table_crit.setRowCount(len(inconsistent_data))
        for i, item in enumerate(inconsistent_data):
            # Formata o hash para não ocupar a tela toda
            hsh_display = str(item['hash'])
            if len(hsh_display) > 16:
                hsh_display = hsh_display[:12] + "..."
                
            table_crit.setItem(i, 0, QTableWidgetItem(hsh_display))
            table_crit.setItem(i, 1, QTableWidgetItem(str(item['count'])))
            
            ids_str = " | ".join(item['identities'])
            item_ids = QTableWidgetItem(ids_str)
            item_ids.setForeground(QtGui.QColor("#c62828")) # Vermelho para destacar o erro
            table_crit.setItem(i, 2, item_ids)

        self.layout_hashes.addSpacing(20)

        # Secção Benigna (Consistentes)
        lbl_ok = QLabel(f"ℹ️ Duplicatas Benignas (Mesma foto, mesma identidade): {len(consistent_data)}")
        lbl_ok.setStyleSheet("color: #1565c0; font-weight: bold;")
        self.layout_hashes.addWidget(lbl_ok)

    def build_mining_tables(self, df_genuines, df_impostors):
        """
        Renderiza os DataFrames e adiciona um visualizador de imagens interativo.
        """
        def create_interactive_table(df, parent_layout, is_genuine):
            # Limpa qualquer layout anterior
            while parent_layout.count():
                child = parent_layout.takeAt(0)
                if child.widget(): child.widget().deleteLater()

            if df is None or df.empty:
                lbl = QLabel("✅ Nenhuma anomalia encontrada para esta categoria.")
                lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #2e7d32; padding: 10px;")
                parent_layout.addWidget(lbl)
                return

            if is_genuine:
                headers = ["Identidade", "Arquivo 1", "Arquivo 2", "Similaridade (Baixa)"]
                keys = ["identity_1", "file_name_1", "file_name_2", "cosine_similarity"]
            else:
                headers = ["ID 1", "ID 2", "Arquivo 1", "Arquivo 2", "Similaridade (Alta)"]
                keys = ["identity_1", "identity_2", "file_name_1", "file_name_2", "cosine_similarity"]

            # --- 1. A Tabela ---
            table = QTableWidget()
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
            
            header = table.horizontalHeader()
            for i in range(len(headers) - 1):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(len(headers) - 1, QHeaderView.ResizeMode.Stretch)
            
            table.setRowCount(len(df))
            
            for visual_row_idx, (orig_idx, row) in enumerate(df.iterrows()):
                for col_idx, key in enumerate(keys):
                    val = row.get(key, "N/A")
                    
                    if key == 'cosine_similarity':
                        try:
                            val_float = float(val)
                            item = QTableWidgetItem(f"{val_float:.4f}")
                            item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                            if is_genuine and val_float < 0.35: 
                                item.setForeground(QtGui.QColor("#d32f2f"))
                            elif not is_genuine and val_float > 0.65:
                                item.setForeground(QtGui.QColor("#ef6c00"))
                        except ValueError:
                            item = QTableWidgetItem(str(val))
                    else:
                        item = QTableWidgetItem(str(val))
                    
                    # MAGIA AQUI: Guardamos os caminhos reais da imagem secretamente na COLUNA 0
                    if col_idx == 0:
                        img1_path = row.get("image_path_1", "")
                        img2_path = row.get("image_path_2", "")
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, (img1_path, img2_path))
                        
                    table.setItem(visual_row_idx, col_idx, item)

            # --- 2. O Painel de Visualização das Imagens ---
            preview_widget = QWidget()
            preview_layout = QHBoxLayout(preview_widget)
            
            lbl_img1 = QLabel("Selecione uma linha acima")
            lbl_img2 = QLabel("para comparar as faces.")
            lbl_img1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl_img2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            
            # Estilo de "porta-retratos"
            style = "border: 1px solid #bdbdbd; background-color: #eeeeee; border-radius: 4px;"
            lbl_img1.setStyleSheet(style)
            lbl_img2.setStyleSheet(style)
            lbl_img1.setMinimumHeight(200)
            lbl_img2.setMinimumHeight(200)

            preview_layout.addWidget(lbl_img1)
            preview_layout.addWidget(lbl_img2)

            # --- 3. A Lógica do Clique ---
            def on_row_selected():
                selected = table.selectedItems()
                if not selected: return
                
                # Pegamos o item da Coluna 0 (onde escondemos a informação)
                row_idx = selected[0].row()
                item_col0 = table.item(row_idx, 0)
                
                paths = item_col0.data(QtCore.Qt.ItemDataRole.UserRole)
                if paths:
                    path1, path2 = paths
                    
                    # Carrega as imagens e redimensiona mantendo a proporção
                    if path1:
                        pix1 = QtGui.QPixmap(path1).scaled(250, 250, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                        lbl_img1.setPixmap(pix1)
                    if path2:
                        pix2 = QtGui.QPixmap(path2).scaled(250, 250, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
                        lbl_img2.setPixmap(pix2)

            # Conecta o sinal de clique à nossa função
            table.itemSelectionChanged.connect(on_row_selected)

            # --- 4. O Splitter (Permite redimensionar Tabela vs Imagem) ---
            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            splitter.addWidget(table)
            splitter.addWidget(preview_widget)
            
            # Dá mais espaço inicial para a tabela (proporção 70/30)
            splitter.setSizes([400, 200])

            parent_layout.addWidget(splitter)

        # Chama a função interativa para as duas abas
        create_interactive_table(df_genuines, self.layout_genuines, is_genuine=True)
        create_interactive_table(df_impostors, self.layout_impostors, is_genuine=False)