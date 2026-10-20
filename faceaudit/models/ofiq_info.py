import logging
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Dict, FrozenSet, Optional, Set

from .ofiq_measure_names import field_for_column

logger = logging.getLogger(__name__)

# Uma vez por nome por sessão: um lote de mil imagens com uma coluna nova
# não pode virar mil avisos iguais.
_colunas_ja_avisadas: Set[str] = set()


def _avisar_coluna_desconhecida(nome: str) -> None:
    if nome in _colunas_ja_avisadas:
        return
    _colunas_ja_avisadas.add(nome)
    logger.warning(
        "Coluna do OFIQ sem campo correspondente no FaceAudit: %r — "
        "guardada como métrica extra. Versão nova do OFIQ com nome diferente?",
        nome,
    )

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
    # Qual OFIQ produziu ESTA linha. Por face, e não só por dataset, porque
    # um enriquecimento posterior pode usar outra versão nas faces que
    # faltavam — e o laudo precisa saber de onde veio cada número.
    # None = linha gravada antes deste registro existir (versão desconhecida).
    version: Optional[str] = None      # ex.: "1.0.3"
    backend: Optional[str] = None      # "executable" ou "library"

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

    # Campos de texto: não passam pela conversão para float. "1.2.0" falha
    # na conversão e sobreviveria, mas uma versão "1.2" viraria o número 1.2.
    TEXT_FIELDS: ClassVar[FrozenSet[str]] = frozenset(
        {"filename", "version", "backend"}
    )

    @property
    def score(self) -> float:
        # Prioriza o scalar (0-100). Se não tiver, usa 0.0
        return self.unified_quality_score_scalar if self.unified_quality_score_scalar is not None else 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OFIQInfo':
        """Monta a partir de uma linha do OFIQ (CSV ou wrapper), de qualquer versão.

        Os nomes passam por `field_for_column`, que conhece os da v1.0.3 e os
        da v1.2.x. Chave que já é nome de campo (ex.: `version`) entra direto.
        O que sobrar vai para `_extra_metrics` — e é avisado no log, porque
        sobra silenciosa foi justamente como a troca de nomes da v1.1.1
        apagava medidas sem ninguém perceber.
        """
        normalized_data = {}
        extra_metrics = {}
        valid_class_keys = {f.name for f in fields(cls)} - {"_extra_metrics"}

        for key, value in data.items():
            clean_key = (key or "").strip()

            # Ignora chaves vazias que podem aparecer por erro de CSV
            # (o OFIQSampleApp termina o cabeçalho com `;`).
            if not clean_key:
                continue

            target_key = field_for_column(clean_key)
            if target_key is None and clean_key in valid_class_keys:
                target_key = clean_key

            final_value = value
            if target_key not in cls.TEXT_FIELDS:
                try:
                    if value is not None and value != "":
                        final_value = float(value)
                except (TypeError, ValueError):
                    pass

            if target_key in valid_class_keys:
                normalized_data[target_key] = final_value
            else:
                extra_metrics[clean_key] = final_value
                _avisar_coluna_desconhecida(clean_key)

        instance = cls(**normalized_data)
        instance._extra_metrics = extra_metrics
        return instance
    
    def to_dict(self) -> Dict[str, Any]:
        # Usamos dict(self.__dict__) para evitar o deepcopy do asdict se não for necessário
        base = {k: v for k, v in asdict(self).items() if k != '_extra_metrics'}
        if self._extra_metrics:
            base.update(self._extra_metrics)
        return base