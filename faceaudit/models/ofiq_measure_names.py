"""Nome de coluna do OFIQ → campo do `OFIQInfo`, em qualquer versão.

O OFIQ mudou os nomes entre versões, e a tradução antiga só conhecia os da
v1.0.3. A partir da v1.1.1 o valor bruto ganhou o sufixo `.native` (antes não
tinha sufixo nenhum), e oito medidas foram renomeadas. Com a tabela velha, um
CSV da v1.2.0 perdia TODOS os valores brutos e oito medidas inteiras —
incluindo os ângulos da cabeça, que alimentam `ofiq_yaw/pitch/roll` — sem
erro: o nome desconhecido caía na sobra (`_extra_metrics`) e ninguém via.

Uma tabela só atende às três fontes que o FaceAudit lê: o CSV do
`OFIQSampleApp` da v1.0.3, o da v1.2.x, e os nomes que o wrapper Python
(`OFIQ.quality_measure_name`) devolve — que são os da v1.2.x sem sufixo.
"""

from typing import Dict, Optional

# Nome da medida na v1.0.3 (a implementação de referência) → prefixo do campo
# no `OFIQInfo`. Cada medida vira `<prefixo>_raw` e `<prefixo>_scalar`.
MEASURE_FIELD_PREFIX: Dict[str, str] = {
    "UnifiedQualityScore": "unified_quality_score",
    "BackgroundUniformity": "background_uniformity",
    "IlluminationUniformity": "illumination_uniformity",
    "LuminanceMean": "luminance_mean",
    "LuminanceVariance": "luminance_variance",
    "UnderExposurePrevention": "under_exposure",
    "OverExposurePrevention": "over_exposure",
    "DynamicRange": "dynamic_range",
    "Sharpness": "sharpness",
    "CompressionArtifacts": "compression_artifacts",
    "NaturalColour": "natural_colour",
    "SingleFacePresent": "single_face_present",
    "EyesOpen": "eyes_open",
    "MouthClosed": "mouth_closed",
    "EyesVisible": "eyes_visible",
    "MouthOcclusionPrevention": "mouth_occlusion",
    "FaceOcclusionPrevention": "face_occlusion",
    "InterEyeDistance": "inter_eye_distance",
    "HeadSize": "head_size",
    "LeftwardCropOfTheFaceImage": "crop_left",
    "RightwardCropOfTheFaceImage": "crop_right",
    "MarginAboveOfTheFaceImage": "margin_above",
    "MarginBelowOfTheFaceImage": "margin_below",
    "HeadPoseYaw": "head_pose_yaw",
    "HeadPosePitch": "head_pose_pitch",
    "HeadPoseRoll": "head_pose_roll",
    "ExpressionNeutrality": "expression_neutrality",
    "NoHeadCoverings": "no_head_coverings",
}

# Renomeações da v1.1.1 em diante: nome novo → nome da v1.0.3. Conferido
# contra o cabeçalho real do `OFIQSampleApp` das duas versões.
RENAMED_MEASURES: Dict[str, str] = {
    "NoCompressionArtifacts": "CompressionArtifacts",
    "LeftwardCropOfFace": "LeftwardCropOfTheFaceImage",
    "RightwardCropOfFace": "RightwardCropOfTheFaceImage",
    "MarginAboveTheFace": "MarginAboveOfTheFaceImage",
    "MarginBelowTheFace": "MarginBelowOfTheFaceImage",
    "HeadPoseYawFrontal": "HeadPoseYaw",
    "HeadPosePitchFrontal": "HeadPosePitch",
    "HeadPoseRollFrontal": "HeadPoseRoll",
}

# Colunas que não são medida.
METADATA_COLUMNS: Dict[str, str] = {
    "Filename": "filename",
    "assessment_time_in_ms": "assessment_time_ms",
}

# Sufixo do valor bruto: nenhum na v1.0.3, `.native` da v1.1.1 em diante.
_RAW_SUFFIXES = ("", "native")
_SCALAR_SUFFIX = "scalar"


def field_for_column(column: str) -> Optional[str]:
    """Campo do `OFIQInfo` para uma coluna do OFIQ, ou None se desconhecida.

    Aceita `Medida`, `Medida.native` e `Medida.scalar`, com o nome de
    qualquer versão.
    """
    name = column.strip()
    if name in METADATA_COLUMNS:
        return METADATA_COLUMNS[name]
    base, _, suffix = name.partition(".")
    base = RENAMED_MEASURES.get(base, base)
    prefix = MEASURE_FIELD_PREFIX.get(base)
    if prefix is None:
        return None
    if suffix in _RAW_SUFFIXES:
        return f"{prefix}_raw"
    if suffix == _SCALAR_SUFFIX:
        return f"{prefix}_scalar"
    return None
