from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class OFIQInfo:
    """
    Mapeamento Total das métricas ISO/IEC 29794-5 do OFIQ.
    Agora inclui pares (Raw/Scalar) para todas as métricas identificadas.
    """

    # --- 1. Score Unificado ---
    unified_quality_score_scalar: Optional[float] = None # 0-100
    unified_quality_score_raw: Optional[float] = None    # Valor bruto

    # --- 2. Metadados ---
    filename: Optional[str] = None
    assessment_time_ms: Optional[float] = None

    # --- 3. Pose da Cabeça (Graus vs Nota) ---
    head_pose_yaw_scalar: Optional[float] = None
    head_pose_yaw_raw: Optional[float] = None
    
    head_pose_pitch_scalar: Optional[float] = None
    head_pose_pitch_raw: Optional[float] = None
    
    head_pose_roll_scalar: Optional[float] = None
    head_pose_roll_raw: Optional[float] = None

    # --- 4. Geometria e Dimensões ---
    head_size_scalar: Optional[float] = None
    head_size_raw: Optional[float] = None
    
    inter_eye_distance_scalar: Optional[float] = None
    inter_eye_distance_raw: Optional[float] = None # Pixels

    # --- 5. Margens e Recortes ---
    margin_above_scalar: Optional[float] = None
    margin_above_raw: Optional[float] = None
    
    margin_below_scalar: Optional[float] = None
    margin_below_raw: Optional[float] = None
    
    crop_left_scalar: Optional[float] = None
    crop_left_raw: Optional[float] = None
    
    crop_right_scalar: Optional[float] = None
    crop_right_raw: Optional[float] = None

    # --- 6. Iluminação e Exposição ---
    luminance_mean_scalar: Optional[float] = None
    luminance_mean_raw: Optional[float] = None
    
    luminance_variance_scalar: Optional[float] = None
    luminance_variance_raw: Optional[float] = None
    
    under_exposure_scalar: Optional[float] = None
    under_exposure_raw: Optional[float] = None
    
    over_exposure_scalar: Optional[float] = None
    over_exposure_raw: Optional[float] = None
    
    dynamic_range_scalar: Optional[float] = None
    dynamic_range_raw: Optional[float] = None

    # --- 7. Uniformidade e Cor ---
    background_uniformity_scalar: Optional[float] = None
    background_uniformity_raw: Optional[float] = None
    
    illumination_uniformity_scalar: Optional[float] = None
    illumination_uniformity_raw: Optional[float] = None
    
    natural_colour_scalar: Optional[float] = None
    natural_colour_raw: Optional[float] = None

    # --- 8. Nitidez e Artefatos ---
    sharpness_scalar: Optional[float] = None
    sharpness_raw: Optional[float] = None
    
    compression_artifacts_scalar: Optional[float] = None
    compression_artifacts_raw: Optional[float] = None

    # --- 9. Características Faciais e Oclusões ---
    single_face_present_scalar: Optional[float] = None
    single_face_present_raw: Optional[float] = None
    
    eyes_open_scalar: Optional[float] = None
    eyes_open_raw: Optional[float] = None
    
    eyes_visible_scalar: Optional[float] = None
    eyes_visible_raw: Optional[float] = None
    
    mouth_closed_scalar: Optional[float] = None
    mouth_closed_raw: Optional[float] = None
    
    mouth_occlusion_scalar: Optional[float] = None
    mouth_occlusion_raw: Optional[float] = None
    
    face_occlusion_scalar: Optional[float] = None
    face_occlusion_raw: Optional[float] = None
    
    expression_neutrality_scalar: Optional[float] = None
    expression_neutrality_raw: Optional[float] = None
    
    no_head_coverings_scalar: Optional[float] = None
    no_head_coverings_raw: Optional[float] = None

    # --- 10. Container de Sobra ---
    _extra_metrics: Dict[str, Any] = None

    @property
    def score(self) -> float:
        # Prioriza o scalar (0-100). Se não tiver, usa 0.0
        return self.unified_quality_score_scalar if self.unified_quality_score_scalar is not None else 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OFIQInfo':
        normalized_data = {}
        extra_metrics = {}
        
        # Mapeamento Completo (CSV Key -> Class Attribute)
        key_mapping = {
            # Metadata
            'Filename': 'filename',
            'assessment_time_in_ms': 'assessment_time_ms',

            # Score Principal
            'UnifiedQualityScore.scalar': 'unified_quality_score_scalar',
            'UnifiedQualityScore': 'unified_quality_score_raw', # Agora mapeado corretamente!
            
            # Poses
            'HeadPoseYaw.scalar': 'head_pose_yaw_scalar',
            'HeadPoseYaw': 'head_pose_yaw_raw',
            'HeadPosePitch.scalar': 'head_pose_pitch_scalar',
            'HeadPosePitch': 'head_pose_pitch_raw',
            'HeadPoseRoll.scalar': 'head_pose_roll_scalar',
            'HeadPoseRoll': 'head_pose_roll_raw',

            # Geometria
            'InterEyeDistance.scalar': 'inter_eye_distance_scalar',
            'InterEyeDistance': 'inter_eye_distance_raw', # Encontrado nas sobras!
            'HeadSize.scalar': 'head_size_scalar',
            'HeadSize': 'head_size_raw',

            # Uniformidade / Cor / Artefatos
            'BackgroundUniformity.scalar': 'background_uniformity_scalar',
            'BackgroundUniformity': 'background_uniformity_raw',
            'IlluminationUniformity.scalar': 'illumination_uniformity_scalar',
            'IlluminationUniformity': 'illumination_uniformity_raw',
            'NaturalColour.scalar': 'natural_colour_scalar',
            'NaturalColour': 'natural_colour_raw',
            'DynamicRange.scalar': 'dynamic_range_scalar',
            'DynamicRange': 'dynamic_range_raw',
            'Sharpness.scalar': 'sharpness_scalar',
            'Sharpness': 'sharpness_raw',
            'CompressionArtifacts.scalar': 'compression_artifacts_scalar',
            'CompressionArtifacts': 'compression_artifacts_raw',

            # Luminância / Exposição
            'LuminanceMean.scalar': 'luminance_mean_scalar',
            'LuminanceMean': 'luminance_mean_raw',
            'LuminanceVariance.scalar': 'luminance_variance_scalar',
            'LuminanceVariance': 'luminance_variance_raw',
            'UnderExposurePrevention.scalar': 'under_exposure_scalar',
            'UnderExposurePrevention': 'under_exposure_raw',
            'OverExposurePrevention.scalar': 'over_exposure_scalar',
            'OverExposurePrevention': 'over_exposure_raw',

            # Features / Oclusões / Expressão
            'SingleFacePresent.scalar': 'single_face_present_scalar',
            'SingleFacePresent': 'single_face_present_raw',
            'EyesOpen.scalar': 'eyes_open_scalar',
            'EyesOpen': 'eyes_open_raw',
            'EyesVisible.scalar': 'eyes_visible_scalar',
            'EyesVisible': 'eyes_visible_raw',
            'MouthClosed.scalar': 'mouth_closed_scalar',
            'MouthClosed': 'mouth_closed_raw',
            'ExpressionNeutrality.scalar': 'expression_neutrality_scalar',
            'ExpressionNeutrality': 'expression_neutrality_raw',
            'NoHeadCoverings.scalar': 'no_head_coverings_scalar',
            'NoHeadCoverings': 'no_head_coverings_raw',
            
            'MouthOcclusionPrevention.scalar': 'mouth_occlusion_scalar',
            'MouthOcclusionPrevention': 'mouth_occlusion_raw',
            'FaceOcclusionPrevention.scalar': 'face_occlusion_scalar',
            'FaceOcclusionPrevention': 'face_occlusion_raw',

            # Margens (Crops)
            'MarginAboveOfTheFaceImage.scalar': 'margin_above_scalar',
            'MarginAboveOfTheFaceImage': 'margin_above_raw',
            'MarginBelowOfTheFaceImage.scalar': 'margin_below_scalar',
            'MarginBelowOfTheFaceImage': 'margin_below_raw',
            'LeftwardCropOfTheFaceImage.scalar': 'crop_left_scalar',
            'LeftwardCropOfTheFaceImage': 'crop_left_raw',
            'RightwardCropOfTheFaceImage.scalar': 'crop_right_scalar',
            'RightwardCropOfTheFaceImage': 'crop_right_raw',
        }

        valid_class_keys = cls.__annotations__.keys()
        
        for key, value in data.items():
            clean_key = key.strip()
            
            # Ignora chaves vazias que podem aparecer por erro de CSV
            if not clean_key:
                continue

            target_key = key_mapping.get(clean_key)
            
            # Fallback
            if not target_key:
                target_key = clean_key.replace('.', '_').lower()

            final_value = value
            # Tenta converter para float
            if target_key != 'filename':
                try:
                    if value is not None and value != "":
                        final_value = float(value)
                except ValueError:
                    pass 

            if target_key in valid_class_keys:
                normalized_data[target_key] = final_value
            else:
                extra_metrics[clean_key] = final_value

        instance = cls(**normalized_data)
        instance._extra_metrics = extra_metrics
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        # Usamos dict(self.__dict__) para evitar o deepcopy do asdict se não for necessário
        base = {k: v for k, v in asdict(self).items() if k != '_extra_metrics'}
        if self._extra_metrics:
            base.update(self._extra_metrics)
        return base