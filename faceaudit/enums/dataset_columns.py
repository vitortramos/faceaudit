from __future__ import annotations
from enum import Enum
from typing import List, Optional, Set, Union

class DatasetColumns(Enum):
    """
    Schema Central de Colunas do Dataset.
    
    Estrutura da Tupla: 
    (key_df, label_ui, show_in_table, editable, ui_order)
    """

    # --- 1. IDENTIFICAÇÃO (0 - 90) ---
    FILE_NAME   = ("file_name", "Arquivo", True, False, 1)
    IDENTITY    = ("identity", "Identidade", True, True, 2)
    IMAGE_PATH  = ("image_path", "Caminho", False, False, 999)
    HASH        = ("hash", "Hash", False, False, 999)
    ORIGIN      = ("origin", "Origem", True, False, 3)

    # --- 2. DEMOGRAFIA (100 - 190) ---
    GENDER_EST  = ("gender_est", "Gênero (Est.)", False, False, 100)
    AGE_EST     = ("age_est", "Idade (Est.)", False, False, 110)
    GENDER_GT   = ("gender", "Gênero (Real)", True, True, 4)
    ETHNICITY   = ("ethnicity", "Etnia", True, True, 5)
    
    # --- 3. QUALIDADE E SCORES (200 - 290) ---
    FIQA_SCORE  = ("fiqa_score", "FF FIQA", True, False, 13)
    DET_SCORE   = ("det_score", "Detecção", False, False, 210)
    SERFIQ_SCORE = ("serfiq_score", "SER-FIQ", True, False, 15)
    OFIQ_SCORE  = ("ofiq_score", "OFIQ (ISO)", True, False, 14)

    # --- 4. POSE (300 - 390) ---
    YAW         = ("yaw", "FF Yaw", True, False, 7)
    PITCH       = ("pitch", "FF Pitch", True, False, 8)
    ROLL        = ("roll", "FF Roll", True, False, 9)

    # --- 5. TÉCNICO / INVISÍVEL (900+) ---
    EMB_V2      = ("embedding_v2", "Vetor V2", False, False, 900)
    EMB_V4      = ("embedding_v4", "Vetor V4", False, False, 901)
    BBOX        = ("bbox", "BBox", False, False, 902)
    KPS         = ("keypoints", "Keypoints", False, False, 903)
    
    # --- 6. OFIQ DETALHADO (ISO) ---
    OFIQ_YAW          = ("ofiq_yaw", "Yaw (ISO)", True, False, 10)
    OFIQ_PITCH        = ("ofiq_pitch", "Pitch (ISO)", True, False, 11)
    OFIQ_ROLL         = ("ofiq_roll", "Roll (ISO)", True, False, 12)
    OFIQ_LUM_MEAN     = ("ofiq_luminance_mean_scalar", "Luminância", False, False, 910)
    OFIQ_LUM_VAR      = ("ofiq_luminance_variance_scalar", "Var. Luminância", False, False, 911)
    OFIQ_SHARPNESS    = ("ofiq_sharpness_scalar", "Nitidez", False, False, 912)
    OFIQ_BG_UNIF      = ("ofiq_background_uniformity_scalar", "Unif. Fundo", False, False, 913)
    OFIQ_ILLUM_UNIF   = ("ofiq_illumination_uniformity_scalar", "Unif. Iluminação", False, False, 914)
    OFIQ_EYES_OPEN    = ("ofiq_eyes_open_scalar", "Olhos Abertos", False, False, 915)
    OFIQ_MOUTH_CLOSED = ("ofiq_mouth_closed_scalar", "Boca Fechada", False, False, 916)
    OFIQ_EXPRESSION   = ("ofiq_expression_neutrality_scalar", "Neutralidade", False, False, 917)
    OFIQ_FACE_OCC     = ("ofiq_face_occlusion_scalar", "Oclusão Face", False, False, 918)
    OFIQ_EYE_OCC      = ("ofiq_eye_occlusion_scalar", "Oclusão Olhos", False, False, 919)
    OFIQ_MOUTH_OCC    = ("ofiq_mouth_occlusion_scalar", "Oclusão Boca", False, False, 920)
    OFIQ_NATURAL_COL  = ("ofiq_natural_colour_scalar", "Cor Natural", False, False, 921)
    OFIQ_COMPRESSION  = ("ofiq_compression_artifacts_scalar", "Artefatos Comp.", False, False, 922)
    OFIQ_UNDER_EXP    = ("ofiq_under_exposure_scalar", "Subexposição", False, False, 923)
    OFIQ_OVER_EXP     = ("ofiq_over_exposure_scalar", "Superexposição", False, False, 924)

    def __init__(self, key: str, label: str, show_in_table: bool, editable: bool, order: int):
        self._value_ = key # A key do DataFrame é o valor real do Enum
        self.key = key
        self.label = label
        # label_pt alias — `label` já está em PT-BR. Mantido por consistência
        # de API com outros enums do projeto.
        self.label_pt = label
        self.show_in_table = show_in_table
        self.editable = editable
        self.order = order

    def __str__(self) -> str:
        return self.label_pt

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna a chave string da coluna (ex: 'file_name')."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> Optional[DatasetColumns]:
        """Reconstrói o Enum a partir da chave string."""
        try:
            return cls(value)
        except ValueError:
            return None

    # --- Helpers de Lógica de Negócio ---

    @classmethod
    def get_visible_columns(cls) -> List[DatasetColumns]:
        """Retorna lista das colunas visíveis, ordenadas para a UI."""
        visible = [c for c in cls if c.show_in_table]
        return sorted(visible, key=lambda x: x.order)

    @classmethod
    def get_editable_keys(cls) -> Set[str]:
        """Retorna o conjunto de strings das colunas que podem ser editadas."""
        return {c.value for c in cls if c.editable}