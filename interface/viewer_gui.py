import sys
import os
import pandas as pd
from pathlib import Path
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QApplication, QWidget, QTableView, QSplitter, 
    QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QScrollArea, 
    QProgressBar, QHeaderView, QFileDialog, QMessageBox, QFrame,
    QFormLayout, QInputDialog
)
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex

# --- IMPORTAÇÕES DO SEU PROJETO ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from faceaudit.models import FaceDataset
except ImportError:
    pass 

# =============================================================================
# 1. MODELO DA TABELA 
# =============================================================================
class FaceTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self.columns_map = {
            "file_name": "Arquivo",
            "identity": "Identidade",
            "fiqa_score": "FIQA (Bio)",
            "ofiq_score": "OFIQ (ISO)",
            "serfiq_score": "SER-FIQ",
            "age_est": "Idade Est.",
            "gender_est": "Gênero Est."
        }
        self.columns = [c for c in self.columns_map.keys() if c in df.columns]

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role):
        if not index.isValid(): return None
        row, col = index.row(), index.column()
        col_name = self.columns[col]
        val = self._df.iloc[row][col_name]

        if role == Qt.ItemDataRole.DisplayRole:
            if pd.isna(val): return "-"
            if isinstance(val, float): return f"{val:.2f}"
            return str(val)

        if role == Qt.ItemDataRole.BackgroundRole:
            if col_name in ["fiqa_score", "ofiq_score", "serfiq_score"]:
                if pd.notna(val):
                    if val < 40: return QtGui.QColor("#ffcccc") 
                    if val < 60: return QtGui.QColor("#fff4cc")
                    return QtGui.QColor("#ccffcc")

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(val, (int, float)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            col = self.columns[section]
            return self.columns_map.get(col, col)
        return None

    def sort(self, column, order):
        col_name = self.columns[column]
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort_values(col_name, ascending=(order == Qt.SortOrder.AscendingOrder))
        self.layoutChanged.emit()

    def get_row_data(self, row_idx):
        return self._df.iloc[row_idx]

# =============================================================================
# 2. WIDGETS DE DETALHE
# =============================================================================
class MetricWidget(QWidget):
    def __init__(self, label, value):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        lbl_name = QLabel(label)
        lbl_name.setFixedWidth(140)
        
        pbar = QProgressBar()
        pbar.setRange(0, 100)
        pbar.setTextVisible(False)
        pbar.setFixedHeight(10)
        
        lbl_val = QLabel(f"{value:.2f}" if value is not None else "N/A")
        lbl_val.setFixedWidth(50)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight)

        if value is not None:
            display_val = value if value > 1.0 else value * 100
            pbar.setValue(int(display_val))
            if display_val < 40: pbar.setStyleSheet("QProgressBar::chunk { background-color: #d9534f; }")
            elif display_val < 70: pbar.setStyleSheet("QProgressBar::chunk { background-color: #f0ad4e; }")
            else: pbar.setStyleSheet("QProgressBar::chunk { background-color: #5cb85c; }")
        else:
            pbar.setValue(0)

        layout.addWidget(lbl_name)
        layout.addWidget(pbar)
        layout.addWidget(lbl_val)

class FaceDetailPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_image = QLabel("Selecione uma face")
        self.lbl_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image.setMinimumHeight(150)
        self.lbl_image.setMaximumHeight(280)
        self.lbl_image.setStyleSheet("background-color: #e6e6e6; border-bottom: 1px solid #ccc;")
        layout.addWidget(self.lbl_image)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Aba Geral
        self.tab_general = QWidget()
        self.scroll_general = QScrollArea() 
        self.scroll_general.setWidgetResizable(True)
        self.scroll_general.setWidget(self.tab_general)
        self.form_general = QFormLayout(self.tab_general)
        self.form_general.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.form_general.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(self.scroll_general, "Geral / Bio")
        
        # Aba OFIQ
        self.scroll_ofiq = QScrollArea()
        self.scroll_ofiq.setWidgetResizable(True)
        self.container_ofiq = QWidget()
        self.layout_ofiq = QVBoxLayout(self.container_ofiq)
        self.layout_ofiq.setContentsMargins(5, 5, 5, 5)
        self.layout_ofiq.setSpacing(2)
        self.scroll_ofiq.setWidget(self.container_ofiq)
        self.tabs.addTab(self.scroll_ofiq, "Qualidade ISO (OFIQ)")

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_row_to_general(self, label, value):
        lbl = QLabel(str(value))
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.form_general.addRow(f"<b>{label}:</b>", lbl)

    def set_dataset(self, dataset):
        """Recebe dataset corrente pra resolver paths em modo RELATIVE."""
        self._dataset = dataset

    def load_face(self, face_data: pd.Series):
        raw = face_data.get('image_path')
        path = (
            str(self._dataset.resolve_image_path(raw))
            if getattr(self, "_dataset", None) is not None and raw
            else raw
        )
        if path and os.path.exists(path):
            pixmap = QtGui.QPixmap(path)
            if not pixmap.isNull():
                w = self.lbl_image.width()
                scaled = pixmap.scaled(w, 280, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.lbl_image.setPixmap(scaled)
        else:
            self.lbl_image.clear()
            self.lbl_image.setText("Imagem não encontrada")

        self.clear_layout(self.form_general)
        self.add_row_to_general("Arquivo", face_data.get('file_name'))
        self.add_row_to_general("Identidade", face_data.get('identity'))
        h = face_data.get('hash')
        if h: self.add_row_to_general("Hash", h)
        self.add_row_to_general("Idade Est.", face_data.get('age_est'))
        self.add_row_to_general("Gênero Est.", face_data.get('gender_est'))
        
        yaw = face_data.get('yaw')
        if pd.notna(yaw):
            self.add_row_to_general("Pose (Y/P/R)", f"{yaw:.1f}° / {face_data.get('pitch'):.1f}° / {face_data.get('roll'):.1f}°")
        
        if pd.notna(face_data.get('fiqa_score')):
            self.add_row_to_general("FIQA Score", f"{face_data.get('fiqa_score'):.2f}")
        if pd.notna(face_data.get('det_score')):
            self.add_row_to_general("Detecção Score", f"{face_data.get('det_score'):.4f}")

        self.clear_layout(self.layout_ofiq)
        ofiq_cols = [c for c in face_data.index if c.startswith('ofiq_') and c.endswith('_scalar')]
        
        if not ofiq_cols:
            self.layout_ofiq.addWidget(QLabel("Dados OFIQ não calculados."))
        else:
            for col in sorted(ofiq_cols):
                pretty_name = col.replace("ofiq_", "").replace("_scalar", "").replace("_", " ").title()
                val = face_data[col]
                if pd.isna(val): val = 0.0
                self.layout_ofiq.addWidget(MetricWidget(pretty_name, val))
            self.layout_ofiq.addStretch()

# =============================================================================
# 3. WIDGET PRINCIPAL (VIEWER & MANAGER)
# =============================================================================
class DatasetViewerWidget(QWidget):
    # Sinais para solicitar ações que envolvem a janela principal (como criar novo)
    request_new = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # --- 1. BARRA DE FERRAMENTAS ---
        self.top_container = QFrame()
        self.top_container.setFrameShape(QFrame.Shape.StyledPanel)
        self.top_container.setMaximumHeight(60)
        self.top_container.setStyleSheet("background-color: #f5f5f5; border-radius: 5px;")
        
        top_bar = QHBoxLayout(self.top_container)
        top_bar.setContentsMargins(10, 5, 10, 5)
        top_bar.setSpacing(10)
        
        # Estilos
        btn_action_style = """
            QPushButton { background-color: #ffffff; border: 1px solid #ccc; padding: 6px 12px; border-radius: 4px; color: #333; }
            QPushButton:hover { background-color: #e6e6e6; }
        """
        btn_primary_style = "QPushButton { background-color: #28a745; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; } QPushButton:hover { background-color: #218838; }"
        btn_blue_style = "QPushButton { background-color: #0078d7; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; } QPushButton:hover { background-color: #005a9e; }"

        # Botões
        self.btn_new = QtWidgets.QPushButton("➕ Novo")
        self.btn_new.setStyleSheet(btn_primary_style)
        self.btn_new.clicked.connect(self.request_new.emit)
        
        self.btn_load = QtWidgets.QPushButton("📂 Abrir")
        self.btn_load.setStyleSheet(btn_blue_style)
        self.btn_load.clicked.connect(self.abrir_dataset_dialog)

        self.btn_save = QtWidgets.QPushButton("💾 Salvar")
        self.btn_save.setStyleSheet(btn_action_style)
        self.btn_save.clicked.connect(self.salvar_dataset)
        self.btn_save.setEnabled(False)

        self.btn_merge = QtWidgets.QPushButton("🔗 Juntar")
        self.btn_merge.setStyleSheet(btn_action_style)
        self.btn_merge.clicked.connect(self.realizar_merge)
        self.btn_merge.setEnabled(False)

        self.btn_split = QtWidgets.QPushButton("✂️ Dividir")
        self.btn_split.setStyleSheet(btn_action_style)
        self.btn_split.clicked.connect(self.realizar_split)
        self.btn_split.setEnabled(False)

        self.btn_filter = QtWidgets.QPushButton("🔍 Filtrar")
        self.btn_filter.setStyleSheet(btn_action_style)
        self.btn_filter.clicked.connect(self.realizar_filtro)
        self.btn_filter.setEnabled(False)

        self.lbl_status = QLabel("Nenhum dataset.")
        self.lbl_status.setStyleSheet("color: #666; margin-left: 10px;")

        # Adiciona ao layout
        top_bar.addWidget(self.btn_new)
        top_bar.addWidget(self.btn_load)
        top_bar.addWidget(self.btn_save) # NOVO
        
        # Separador
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        top_bar.addWidget(line)
        
        top_bar.addWidget(self.btn_merge)
        top_bar.addWidget(self.btn_split)
        top_bar.addWidget(self.btn_filter)
        
        top_bar.addWidget(self.lbl_status)
        top_bar.addStretch()
        
        main_layout.addWidget(self.top_container, 0)

        # --- 2. ÁREA CENTRAL ---
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(2)
        main_layout.addWidget(self.splitter, 1)

        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(True)
        self.table_view.alternatingRowColors = True
        self.table_view.setStyleSheet("QTableView { border: 1px solid #ccc; }")
        self.splitter.addWidget(self.table_view)

        self.detail_panel = FaceDetailPanel()
        self.splitter.addWidget(self.detail_panel)
        self.splitter.setSizes([800, 400])
        
        self.dataset = None

    def _update_ui_state(self):
        """Habilita botões se houver dataset"""
        has_data = self.dataset is not None
        self.btn_save.setEnabled(has_data)
        self.btn_merge.setEnabled(has_data)
        self.btn_split.setEnabled(has_data)
        self.btn_filter.setEnabled(has_data)

    def carregar_dataset_objeto(self, dataset):
        """Carrega um objeto FaceDataset diretamente"""
        self.dataset = dataset
        # Propaga pro detail panel resolver paths em modo RELATIVE.
        if hasattr(self.detail_panel, "set_dataset"):
            self.detail_panel.set_dataset(dataset)
        self.model = FaceTableModel(self.dataset.to_pandas())
        self.table_view.setModel(self.model)
        
        try: self.table_view.selectionModel().selectionChanged.disconnect()
        except: pass
        self.table_view.selectionModel().selectionChanged.connect(self.on_selection_changed)
        
        self.table_view.resizeColumnsToContents()
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.lbl_status.setText(f"<b>{self.dataset.name}</b> | {len(self.dataset)} imagens")
        self._update_ui_state()

    def abrir_dataset_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir Dataset", "", "Parquet Files (*.parquet)")
        if path:
            try:
                self.lbl_status.setText("Carregando...")
                QApplication.processEvents()
                ds = FaceDataset.from_parquet(path)
                self.carregar_dataset_objeto(ds)
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao abrir:\n{e}")

    def salvar_dataset(self):
        """Salva o dataset atual (modificado ou não)"""
        if not self.dataset: return
        
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Dataset", f"{self.dataset.name}.parquet", "Parquet Files (*.parquet)")
        if path:
            try:
                self.dataset.name = Path(path).stem
                self.dataset.to_parquet(path)
                QMessageBox.information(self, "Sucesso", "Dataset salvo com sucesso!")
                self.lbl_status.setText(f"<b>{self.dataset.name}</b> (Salvo)")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao salvar:\n{e}")

    def realizar_merge(self):
        """Junta o dataset atual com outro arquivo"""
        path, _ = QFileDialog.getOpenFileName(self, "Selecionar Dataset para Juntar", "", "Parquet Files (*.parquet)")
        if not path: return
        
        try:
            other_ds = FaceDataset.from_parquet(path)
            
            # Usa o método concat do seu modelo
            new_ds = self.dataset.concat(other_ds, new_name=f"{self.dataset.name}_Merged")
            
            reply = QMessageBox.question(self, "Merge", 
                                         f"Juntado com sucesso!\nTotal de imagens: {len(new_ds)}\n\nDeseja carregar o novo dataset agora?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.carregar_dataset_objeto(new_ds)
                
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao fazer merge:\n{e}")

    def realizar_split(self):
        """Divide o dataset em dois e salva ambos"""
        ratio, ok = QInputDialog.getDouble(self, "Dividir Dataset", "Porcentagem para o primeiro split (0.0 - 1.0):", 0.8, 0.1, 0.9, 2)
        if not ok: return

        try:
            # Usa o método split do seu modelo
            ds_a, ds_b = self.dataset.split_by_identity(split_ratio=ratio)
            
            # Salva ambos automaticamente na mesma pasta do original (ou documents se novo)
            base_dir = Path("data/processed") # Ajuste se necessário ou use file dialog
            base_dir.mkdir(parents=True, exist_ok=True)
            
            path_a = base_dir / f"{ds_a.name}.parquet"
            path_b = base_dir / f"{ds_b.name}.parquet"
            
            ds_a.to_parquet(path_a)
            ds_b.to_parquet(path_b)
            
            # Pergunta qual carregar
            msg = (f"Divisão Concluída!\n\n"
                   f"Parte A ({len(ds_a)} imgs): {path_a}\n"
                   f"Parte B ({len(ds_b)} imgs): {path_b}\n\n"
                   f"Qual deseja visualizar agora?")
            
            box = QMessageBox()
            box.setWindowTitle("Split Sucesso")
            box.setText(msg)
            btn_a = box.addButton("Carregar Parte A", QMessageBox.ButtonRole.ActionRole)
            btn_b = box.addButton("Carregar Parte B", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            
            if box.clickedButton() == btn_a:
                self.carregar_dataset_objeto(ds_a)
            elif box.clickedButton() == btn_b:
                self.carregar_dataset_objeto(ds_b)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Erro", f"Erro no Split:\n{e}")

    def realizar_filtro(self):
        """Exemplo de filtro (Qualidade)"""
        # Exemplo simples: Filtra qualidade > X
        min_score, ok = QInputDialog.getInt(self, "Filtrar por Qualidade", "Score Mínimo (FIQA/OFIQ/SERFIQ):", 50, 0, 100)
        if not ok: return
        
        # Pergunta qual métrica
        items = ["fiqa", "ofiq", "serfiq"]
        metric, ok = QInputDialog.getItem(self, "Métrica", "Escolha a métrica:", items, 0, False)
        if not ok: return

        try:
            filtered_ds = self.dataset.filter_by_quality(min_score=min_score, metric=metric)
            
            if len(filtered_ds) == 0:
                QMessageBox.warning(self, "Filtro", "Nenhuma imagem encontrada com esses critérios.")
                return

            reply = QMessageBox.question(self, "Filtro", 
                                         f"Filtro aplicado!\nImagens restantes: {len(filtered_ds)}\n\nDeseja carregar este resultado?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                self.carregar_dataset_objeto(filtered_ds)
                
        except Exception as e:
             QMessageBox.critical(self, "Erro", f"Erro no Filtro:\n{e}")

    def on_selection_changed(self, selected, deselected):
        indexes = self.table_view.selectionModel().selectedRows()
        if indexes:
            self.detail_panel.load_face(self.model.get_row_data(indexes[0].row()))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = DatasetViewerWidget()
    w.showMaximized()
    sys.exit(app.exec())