import os
from typing import Optional
import pandas as pd
from PyQt6 import QtCore, QtGui
from faceaudit.models.face_dataset import FaceDataset
from .detail_panel_view import DetailPanelView
from interface.oriented_image import ler_pixmap_orientado

class DetailPanelController(QtCore.QObject):
    """
    Controlador do Painel de Detalhes.
    Faz a ponte entre os dados brutos (Pandas Series) e a View.
    """

    def __init__(self, parent_layout=None):
        super().__init__()
        # O Controller é dono da View
        self.view = DetailPanelView()
        # Dataset corrente — usado pra resolver paths em modo RELATIVE.
        # Setado por `set_dataset` quando o dataset_manager_controller
        # carrega/troca de dataset.
        self._dataset: Optional[FaceDataset] = None

        # Facilita a integração: se passar um layout, já se anexa
        if parent_layout:
            parent_layout.addWidget(self.view)

    def get_widget(self):
        """Retorna a view para ser adicionada a layouts externos manualmente."""
        return self.view

    def set_dataset(self, dataset: Optional[FaceDataset]) -> None:
        """Recebe referência ao dataset corrente pra que paths em modo
        RELATIVE possam ser resolvidos contra `_parquet_dir/path_root`.
        Sem isso, datasets do Image Pipeline (que usam `image_path` tipo
        `0005_01.png`) caem em "Imagem não encontrada" porque
        `os.path.exists` checa contra o CWD do processo.
        """
        self._dataset = dataset

    def _carregar_metadados_embutidos(self, face_data) -> None:
        """Preenche a aba de metadados a partir da coluna `exif_json`."""
        import json

        bruto = face_data.get("exif_json") if hasattr(face_data, "get") else None
        if bruto is None or (isinstance(bruto, float) and pd.isna(bruto)):
            self.view.clear_exif_metadata()
            return
        try:
            tags = json.loads(bruto) if isinstance(bruto, str) else dict(bruto)
        except (ValueError, TypeError):
            # Conteúdo ilegível não pode virar "sem metadados": o registro
            # TEM o campo, e o perito precisa saber que ele não abriu.
            self.view.clear_exif_metadata(
                "A coluna de metadados existe mas não pôde ser lida "
                "(conteúdo inválido)."
            )
            return
        self.view.set_exif_metadata(tags)

    def load_face_data(self, face_data: pd.Series):
        """
        Método principal: Recebe os dados de uma face e preenche a UI.
        """
        # Se não houver dados (seleção vazia), limpa tudo
        if face_data is None or face_data.empty:
            self.view.reset_image_placeholder()
            self.view.clear_general_info()
            self.view.clear_quality_metrics()
            self.view.clear_exif_metadata("Nenhum registro selecionado.")
            return

        # 1. Lógica de Imagem (Verificação de Disco). Resolve via dataset
        # quando disponível (suporta path_mode=RELATIVE); fallback pro
        # path raw quando o dataset não foi setado (compat com chamadas
        # diretas).
        raw_path = face_data.get('image_path')
        resolved = (
            str(self._dataset.resolve_image_path(raw_path))
            if self._dataset is not None and raw_path
            else raw_path
        )
        if resolved and os.path.exists(resolved):
            pixmap = ler_pixmap_orientado(resolved)
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

        # 2.b Metadados embutidos — o dump completo vive numa coluna só
        # (`exif_json`); as colunas de captura promovidas são atalhos para
        # consulta, não a fonte.
        self._carregar_metadados_embutidos(face_data)

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