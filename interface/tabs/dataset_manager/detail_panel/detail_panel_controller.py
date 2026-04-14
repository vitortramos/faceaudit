import os
import pandas as pd
from PyQt6 import QtCore, QtGui
from .detail_panel_view import DetailPanelView

class DetailPanelController(QtCore.QObject):
    """
    Controlador do Painel de Detalhes.
    Faz a ponte entre os dados brutos (Pandas Series) e a View.
    """
    
    def __init__(self, parent_layout=None):
        super().__init__()
        # O Controller é dono da View
        self.view = DetailPanelView()
        
        # Facilita a integração: se passar um layout, já se anexa
        if parent_layout:
            parent_layout.addWidget(self.view)

    def get_widget(self):
        """Retorna a view para ser adicionada a layouts externos manualmente."""
        return self.view

    def load_face_data(self, face_data: pd.Series):
        """
        Método principal: Recebe os dados de uma face e preenche a UI.
        """
        # Se não houver dados (seleção vazia), limpa tudo
        if face_data is None or face_data.empty:
            self.view.reset_image_placeholder()
            self.view.clear_general_info()
            self.view.clear_quality_metrics()
            return

        # 1. Lógica de Imagem (Verificação de Disco)
        path = face_data.get('image_path')
        if path and os.path.exists(path):
            pixmap = QtGui.QPixmap(path)
            self.view.set_image(pixmap)
        else:
            self.view.reset_image_placeholder("Imagem não encontrada")

        # 2. Lógica de Dados Gerais (Formatação)
        self.view.clear_general_info()
        self.view.add_general_info("Arquivo", face_data.get('file_name', 'N/A'))
        self.view.add_general_info("Identidade", face_data.get('identity', 'N/A'))
        
        h = face_data.get('hash')
        if h: self.view.add_general_info("Hash", h)
        
        self.view.add_general_info("Idade", face_data.get('age_est', '-'))
        self.view.add_general_info("Gênero", face_data.get('gender_est', '-'))

        # Formatação de Pose
        yaw = face_data.get('yaw')
        if pd.notna(yaw):
            p = face_data.get('pitch', 0)
            r = face_data.get('roll', 0)
            self.view.add_general_info("Pose (Y/P/R)", f"{yaw:.1f}° / {p:.1f}° / {r:.1f}°")

        # Scores
        if pd.notna(face_data.get('fiqa_score')):
            self.view.add_general_info("FIQA Score", f"{face_data.get('fiqa_score'):.2f}")
        
        if pd.notna(face_data.get('det_score')):
            self.view.add_general_info("Detecção Score", f"{face_data.get('det_score'):.4f}")

        # 3. Lógica de Qualidade (Filtro de Colunas OFIQ)
        self.view.clear_quality_metrics()
        
        # Filtra dinamicamente todas as colunas que começam com 'ofiq_' e terminam com '_scalar'
        ofiq_cols = [c for c in face_data.index if str(c).startswith('ofiq_') and str(c).endswith('_scalar')]
        
        if not ofiq_cols:
            self.view.add_quality_label("Dados OFIQ não calculados.")
        else:
            for col in sorted(ofiq_cols):
                # Limpa o nome da coluna para ficar legível na UI
                # Ex: 'ofiq_illumination_uniformity_scalar' -> 'Illumination Uniformity'
                pretty_name = col.replace("ofiq_", "").replace("_scalar", "").replace("_", " ").title()
                
                val = face_data[col]
                if pd.isna(val): val = 0.0
                
                self.view.add_quality_metric(pretty_name, float(val))
            
            self.view.add_quality_stretch()