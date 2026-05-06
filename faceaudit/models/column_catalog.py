"""Catálogo de schemas de colunas para os datasets do FaceAudit.

Este módulo é a fonte de verdade para:
  - Os enums `ColumnRole`, `ColumnSource`, `ColumnDType`, etc.
  - O dataclass `ColumnSpec` que descreve cada coluna individualmente.
  - O catálogo default de specs por `DatasetType` (hoje só `face`).
  - Vocabulários default para colunas demográficas.
  - Helpers de validação e indexação por role.

A intenção é substituir o enum estático `DatasetColumns` por um schema
declarado por dataset, persistido na metadata do parquet e estendível
pelo usuário (planilha-like). Veja CLAUDE.md para o desenho completo.

Não importado por nenhum caminho do código ainda — este é o passo 1 de uma
migração maior. Tem comportamento auto-contido e é puramente aditivo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


# =============================================================================
# Enums
# =============================================================================

class DatasetType(str, Enum):
    """Tipo do dataset — hoje só `face`; futuramente `voice`, `iris`, ..."""
    FACE = "face"


class ColumnRole(str, Enum):
    """Papel semântico que uma coluna desempenha no dataset.

    O código branchea por role (ex: "todos os embeddings") em vez de por nome
    de coluna específico. Adicionar um modelo novo de embedding é só registrar
    a coluna com role=EMBEDDING — nenhum código precisa mudar.
    """
    IDENTITY = "identity"          # singleton, obrigatória
    IMAGE_PATH = "image_path"      # singleton, obrigatória para dataset_type=face
    FILE_HASH = "file_hash"        # singleton, opcional
    EMBEDDING = "embedding"        # multi
    QUALITY = "quality"            # multi (com subtype: primary | aux | dimension)
    POSE = "pose"                  # multi (com axis: yaw | pitch | roll)
    GEOMETRY = "geometry"          # multi (ipd, head_size, ...)
    DEMOGRAPHIC = "demographic"    # multi (gender, ethnicity); is_estimate=True para FF
    DISPLAY = "display"            # multi — informativas, sem semântica de código
    METADATA = "metadata"          # multi — parâmetros de algoritmo, telemetria
    CUSTOM = "custom"              # multi — colunas adicionadas pelo usuário


class ColumnSource(str, Enum):
    """De onde a coluna vem."""
    MANUAL = "manual"      # anotação humana
    FF = "ff"              # ForensicFace
    OFIQ = "ofiq"          # OFIQ
    SERFIQ = "serfiq"      # SER-FIQ
    DERIVED = "derived"    # recomputado em runtime (ex: file_name de image_path)
    CUSTOM = "custom"      # criado pelo usuário


class ColumnDType(str, Enum):
    """Tipo do valor armazenado."""
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    VECTOR = "vector"  # list[float] de tamanho fixo (embeddings)
    LIST = "list"      # list de tamanho variável


class QualityDirection(str, Enum):
    """Para colunas de quality: maior é melhor ou menor é melhor."""
    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


class QualitySubtype(str, Enum):
    """Granularidade da métrica de qualidade.

    PRIMARY    — score agregado (ex: fiqa_score, ofiq_score, serfiq_score).
    AUX        — auxiliar (ex: det_score). Não aparece no dropdown de balanceamento por default.
    DIMENSION  — dimensão fina (ex: ofiq_sharpness_scalar). Útil para análise forense detalhada.
    """
    PRIMARY = "primary"
    AUX = "aux"
    DIMENSION = "dimension"


class PoseAxis(str, Enum):
    """Eixo de pose."""
    YAW = "yaw"
    PITCH = "pitch"
    ROLL = "roll"


# =============================================================================
# ColumnSpec
# =============================================================================

@dataclass(frozen=True)
class ColumnSpec:
    """Especificação completa de uma coluna do dataset.

    Cada coluna persistida no parquet (ou recomputada via persisted=False) tem
    um ColumnSpec correspondente registrado na metadata. O dict
    `column_specs: Dict[name, ColumnSpec]` vive em `schema.metadata`.

    Campos universais (toda coluna tem) + campos role-specific (Optional, só
    fazem sentido pro role correspondente).
    """
    # --- Universal ---
    name: str
    label: str
    role: ColumnRole
    source: ColumnSource
    dtype: ColumnDType
    editable: bool = False
    visible_default: bool = True
    persisted: bool = True
    category: Optional[str] = None
    description: Optional[str] = None

    # --- role=embedding ---
    dim: Optional[int] = None
    model_name: Optional[str] = None

    # --- role=quality ---
    metric_name: Optional[str] = None
    direction: Optional[QualityDirection] = None
    scale: Optional[Tuple[float, float]] = None
    quality_subtype: Optional[QualitySubtype] = None

    # --- role=pose ---
    axis: Optional[PoseAxis] = None
    unit: Optional[str] = None

    # --- role=demographic ---
    is_estimate: bool = False
    vocabulary: Optional[Tuple[str, ...]] = None

    # --- role=custom ---
    default_value: Optional[Any] = None

    # ---------------- Serialização ----------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dict JSON-friendly (enums → str, tuple → list)."""
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, Enum):
                out[f.name] = value.value
            elif isinstance(value, tuple):
                out[f.name] = list(value)
            else:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSpec":
        """Reconstrói a partir de dict (inverso de to_dict)."""
        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            raw = data[f.name]
            kwargs[f.name] = _coerce_field(f.name, raw)
        return cls(**kwargs)


# Mapa nome-do-campo → enum/tupla esperado, usado em `from_dict`.
_FIELD_COERCERS = {
    "role": ColumnRole,
    "source": ColumnSource,
    "dtype": ColumnDType,
    "direction": QualityDirection,
    "quality_subtype": QualitySubtype,
    "axis": PoseAxis,
}

_TUPLE_FIELDS = {"scale", "vocabulary"}


def _coerce_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _FIELD_COERCERS:
        enum_cls = _FIELD_COERCERS[name]
        return enum_cls(value) if not isinstance(value, enum_cls) else value
    if name in _TUPLE_FIELDS and isinstance(value, list):
        return tuple(value)
    return value


# =============================================================================
# Constantes — vocabulários e categorias default
# =============================================================================

#: Vocabulários default para colunas demográficas. Estendíveis por dataset
#: (o spec armazenado na metadata pode ter vocabulary expandido).
DEFAULT_VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "gender":    ("M", "F", "Outro", "Desconhecido"),
    "ethnicity": ("Branca", "Preta", "Parda", "Amarela", "Indígena", "Outra", "Desconhecida"),
}


# Categorias canônicas (livres no model — estas são apenas defaults usadas
# pelos specs do catálogo built-in). UI agrupa colunas pela categoria.
CATEGORY_MANUAL = "Identificação Manual"
CATEGORY_FF_IDENT = "FF — Identificadores"
CATEGORY_FF_POSE = "FF — Pose"
CATEGORY_FF_QUALITY = "FF — Qualidade"
CATEGORY_FF_EMBEDDING = "FF — Embeddings"
CATEGORY_OFIQ_PRIMARY = "OFIQ — Score Principal"
CATEGORY_OFIQ_POSE = "OFIQ — Pose"
CATEGORY_OFIQ_DIMENSIONS = "OFIQ — Dimensões"
CATEGORY_OFIQ_DIMENSIONS_RAW = "OFIQ — Dimensões (raw)"
CATEGORY_OFIQ_METADATA = "OFIQ — Metadados"
CATEGORY_SERFIQ = "SER-FIQ"
CATEGORY_CUSTOM = "Customizado"


# =============================================================================
# Roles permitidas e requeridas por DatasetType
# =============================================================================

#: Roles que podem aparecer no schema de um dataset do tipo dado. Roles fora
#: dessa lista produzem erro de validação no load.
ALLOWED_ROLES_BY_TYPE: Dict[DatasetType, FrozenSet[ColumnRole]] = {
    DatasetType.FACE: frozenset({
        ColumnRole.IDENTITY, ColumnRole.IMAGE_PATH, ColumnRole.FILE_HASH,
        ColumnRole.EMBEDDING, ColumnRole.QUALITY, ColumnRole.POSE,
        ColumnRole.GEOMETRY, ColumnRole.DEMOGRAPHIC, ColumnRole.DISPLAY,
        ColumnRole.METADATA, ColumnRole.CUSTOM,
    }),
}


#: Roles que TÊM que estar presentes (ao menos uma coluna com aquele role)
#: para o dataset do tipo dado ser considerado válido.
REQUIRED_ROLES_BY_TYPE: Dict[DatasetType, FrozenSet[ColumnRole]] = {
    DatasetType.FACE: frozenset({ColumnRole.IDENTITY, ColumnRole.IMAGE_PATH}),
}


# =============================================================================
# Catálogo built-in — FACE
# =============================================================================

# As 24 dimensões do OFIQ. Cada uma vira 2 colunas: <name>_scalar e <name>_raw.
# Direção é "higher = melhor" exceto para artefatos (compressão, exposição,
# oclusão, crop) onde "menor é melhor".
_OFIQ_DIMENSIONS: List[Tuple[str, QualityDirection]] = [
    ("head_size",                QualityDirection.HIGHER_IS_BETTER),
    ("inter_eye_distance",       QualityDirection.HIGHER_IS_BETTER),
    ("margin_above",             QualityDirection.HIGHER_IS_BETTER),
    ("margin_below",             QualityDirection.HIGHER_IS_BETTER),
    ("crop_left",                QualityDirection.LOWER_IS_BETTER),
    ("crop_right",               QualityDirection.LOWER_IS_BETTER),
    ("luminance_mean",           QualityDirection.HIGHER_IS_BETTER),
    ("luminance_variance",       QualityDirection.HIGHER_IS_BETTER),
    ("under_exposure",           QualityDirection.LOWER_IS_BETTER),
    ("over_exposure",            QualityDirection.LOWER_IS_BETTER),
    ("dynamic_range",            QualityDirection.HIGHER_IS_BETTER),
    ("background_uniformity",    QualityDirection.HIGHER_IS_BETTER),
    ("illumination_uniformity",  QualityDirection.HIGHER_IS_BETTER),
    ("natural_colour",           QualityDirection.HIGHER_IS_BETTER),
    ("sharpness",                QualityDirection.HIGHER_IS_BETTER),
    ("compression_artifacts",    QualityDirection.LOWER_IS_BETTER),
    ("single_face_present",      QualityDirection.HIGHER_IS_BETTER),
    ("eyes_open",                QualityDirection.HIGHER_IS_BETTER),
    ("eyes_visible",             QualityDirection.HIGHER_IS_BETTER),
    ("mouth_closed",             QualityDirection.HIGHER_IS_BETTER),
    ("mouth_occlusion",          QualityDirection.LOWER_IS_BETTER),
    ("face_occlusion",           QualityDirection.LOWER_IS_BETTER),
    ("expression_neutrality",    QualityDirection.HIGHER_IS_BETTER),
    ("no_head_coverings",        QualityDirection.HIGHER_IS_BETTER),
]


def _gen_ofiq_extra_specs() -> List[ColumnSpec]:
    """Gera os 48 specs (24 dimensões × 2 versões scalar/raw) do OFIQ extra."""
    specs: List[ColumnSpec] = []
    for name, direction in _OFIQ_DIMENSIONS:
        pretty = name.replace("_", " ").title()
        specs.append(ColumnSpec(
            name=f"ofiq_{name}_scalar",
            label=f"OFIQ {pretty} (scalar)",
            role=ColumnRole.QUALITY,
            source=ColumnSource.OFIQ,
            dtype=ColumnDType.FLOAT,
            editable=False,
            visible_default=False,
            category=CATEGORY_OFIQ_DIMENSIONS,
            metric_name=f"ofiq_{name}",
            direction=direction,
            scale=(0.0, 100.0),
            quality_subtype=QualitySubtype.DIMENSION,
        ))
        specs.append(ColumnSpec(
            name=f"ofiq_{name}_raw",
            label=f"OFIQ {pretty} (raw)",
            role=ColumnRole.QUALITY,
            source=ColumnSource.OFIQ,
            dtype=ColumnDType.FLOAT,
            editable=False,
            visible_default=False,
            category=CATEGORY_OFIQ_DIMENSIONS_RAW,
            metric_name=f"ofiq_{name}_raw",
            direction=direction,
            scale=None,
            quality_subtype=QualitySubtype.DIMENSION,
        ))
    return specs


def _face_default_specs() -> Dict[str, ColumnSpec]:
    """Catálogo completo das colunas conhecidas para `dataset_type=face`."""
    specs: Dict[str, ColumnSpec] = {}

    # ---------- Manual / identificação ----------
    specs["image_path"] = ColumnSpec(
        name="image_path", label="Caminho da Imagem",
        role=ColumnRole.IMAGE_PATH, source=ColumnSource.MANUAL,
        dtype=ColumnDType.STR,
        editable=False, visible_default=True, category=CATEGORY_MANUAL,
    )
    specs["file_name"] = ColumnSpec(
        name="file_name", label="Nome do Arquivo",
        role=ColumnRole.DISPLAY, source=ColumnSource.DERIVED,
        dtype=ColumnDType.STR,
        editable=False, visible_default=True, persisted=False,
        category=CATEGORY_MANUAL,
        description="Recomputado a partir de image_path no load.",
    )
    specs["hash"] = ColumnSpec(
        name="hash", label="Hash do Arquivo",
        role=ColumnRole.FILE_HASH, source=ColumnSource.DERIVED,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False, persisted=False,
        category=CATEGORY_MANUAL,
        description="Recomputado no load (verificação de integridade).",
    )
    specs["identity"] = ColumnSpec(
        name="identity", label="Identidade",
        role=ColumnRole.IDENTITY, source=ColumnSource.MANUAL,
        dtype=ColumnDType.STR,
        editable=True, visible_default=True, category=CATEGORY_MANUAL,
    )
    specs["gender"] = ColumnSpec(
        name="gender", label="Gênero",
        role=ColumnRole.DEMOGRAPHIC, source=ColumnSource.MANUAL,
        dtype=ColumnDType.STR,
        editable=True, visible_default=True, category=CATEGORY_MANUAL,
        is_estimate=False, vocabulary=DEFAULT_VOCABULARIES["gender"],
    )
    specs["ethnicity"] = ColumnSpec(
        name="ethnicity", label="Etnia",
        role=ColumnRole.DEMOGRAPHIC, source=ColumnSource.MANUAL,
        dtype=ColumnDType.STR,
        editable=True, visible_default=True, category=CATEGORY_MANUAL,
        is_estimate=False, vocabulary=DEFAULT_VOCABULARIES["ethnicity"],
    )
    specs["origin"] = ColumnSpec(
        name="origin", label="Origem",
        role=ColumnRole.METADATA, source=ColumnSource.MANUAL,
        dtype=ColumnDType.STR,
        editable=True, visible_default=True, category=CATEGORY_MANUAL,
        description="Procedência do arquivo (texto livre).",
    )

    # ---------- ForensicFace ----------
    specs["ipd"] = ColumnSpec(
        name="ipd", label="IPD",
        role=ColumnRole.GEOMETRY, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        unit="pixels",
        description="Inter-pupillary distance.",
    )
    specs["fiqa_score"] = ColumnSpec(
        name="fiqa_score", label="FIQA",
        role=ColumnRole.QUALITY, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=True, category=CATEGORY_FF_QUALITY,
        metric_name="fiqa", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=(0.0, 100.0), quality_subtype=QualitySubtype.PRIMARY,
    )
    specs["det_score"] = ColumnSpec(
        name="det_score", label="Det. Confidence",
        role=ColumnRole.QUALITY, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_FF_QUALITY,
        metric_name="detection", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=(0.0, 1.0), quality_subtype=QualitySubtype.AUX,
    )
    specs["gender_est"] = ColumnSpec(
        name="gender_est", label="Gênero (FF)",
        role=ColumnRole.DEMOGRAPHIC, source=ColumnSource.FF,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        is_estimate=True,
    )
    specs["age_est"] = ColumnSpec(
        name="age_est", label="Idade (FF)",
        role=ColumnRole.DEMOGRAPHIC, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        is_estimate=True,
    )
    specs["yaw"] = ColumnSpec(
        name="yaw", label="Yaw (FF)",
        role=ColumnRole.POSE, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=True, category=CATEGORY_FF_POSE,
        axis=PoseAxis.YAW, unit="degrees",
    )
    specs["pitch"] = ColumnSpec(
        name="pitch", label="Pitch (FF)",
        role=ColumnRole.POSE, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=True, category=CATEGORY_FF_POSE,
        axis=PoseAxis.PITCH, unit="degrees",
    )
    specs["roll"] = ColumnSpec(
        name="roll", label="Roll (FF)",
        role=ColumnRole.POSE, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_FF_POSE,
        axis=PoseAxis.ROLL, unit="degrees",
    )
    specs["embedding_v2"] = ColumnSpec(
        name="embedding_v2", label="Sepael V2",
        role=ColumnRole.EMBEDDING, source=ColumnSource.FF,
        dtype=ColumnDType.VECTOR,
        editable=False, visible_default=False, category=CATEGORY_FF_EMBEDDING,
        dim=256, model_name="sepaelv2",
    )
    specs["embedding_v4"] = ColumnSpec(
        name="embedding_v4", label="Sepael V4",
        role=ColumnRole.EMBEDDING, source=ColumnSource.FF,
        dtype=ColumnDType.VECTOR,
        editable=False, visible_default=False, category=CATEGORY_FF_EMBEDDING,
        dim=512, model_name="sepaelv4",
    )

    # ---------- OFIQ — atalhos canônicos ----------
    specs["ofiq_score"] = ColumnSpec(
        name="ofiq_score", label="OFIQ Score",
        role=ColumnRole.QUALITY, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=True, category=CATEGORY_OFIQ_PRIMARY,
        metric_name="ofiq", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=(0.0, 100.0), quality_subtype=QualitySubtype.PRIMARY,
    )
    specs["ofiq_yaw"] = ColumnSpec(
        name="ofiq_yaw", label="Yaw (OFIQ)",
        role=ColumnRole.POSE, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_POSE,
        axis=PoseAxis.YAW, unit="degrees",
    )
    specs["ofiq_pitch"] = ColumnSpec(
        name="ofiq_pitch", label="Pitch (OFIQ)",
        role=ColumnRole.POSE, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_POSE,
        axis=PoseAxis.PITCH, unit="degrees",
    )
    specs["ofiq_roll"] = ColumnSpec(
        name="ofiq_roll", label="Roll (OFIQ)",
        role=ColumnRole.POSE, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_POSE,
        axis=PoseAxis.ROLL, unit="degrees",
    )

    # ---------- OFIQ — metadados ----------
    specs["ofiq_filename"] = ColumnSpec(
        name="ofiq_filename", label="OFIQ Filename",
        role=ColumnRole.METADATA, source=ColumnSource.OFIQ,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_METADATA,
    )
    specs["ofiq_assessment_time_ms"] = ColumnSpec(
        name="ofiq_assessment_time_ms", label="OFIQ Time (ms)",
        role=ColumnRole.METADATA, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_METADATA,
    )
    specs["ofiq_unified_quality_score_raw"] = ColumnSpec(
        name="ofiq_unified_quality_score_raw", label="OFIQ Unified Score (raw)",
        role=ColumnRole.QUALITY, source=ColumnSource.OFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_METADATA,
        metric_name="ofiq_raw", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=None, quality_subtype=QualitySubtype.AUX,
    )

    # ---------- OFIQ — dimensões (48 colunas: 24 × scalar/raw) ----------
    for spec in _gen_ofiq_extra_specs():
        specs[spec.name] = spec

    # ---------- SER-FIQ ----------
    specs["serfiq_score"] = ColumnSpec(
        name="serfiq_score", label="SER-FIQ Score",
        role=ColumnRole.QUALITY, source=ColumnSource.SERFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=True, category=CATEGORY_SERFIQ,
        metric_name="serfiq", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=(0.0, 100.0), quality_subtype=QualitySubtype.PRIMARY,
    )
    specs["serfiq_t"] = ColumnSpec(
        name="serfiq_t", label="SER-FIQ T",
        role=ColumnRole.METADATA, source=ColumnSource.SERFIQ,
        dtype=ColumnDType.INT,
        editable=False, visible_default=False, category=CATEGORY_SERFIQ,
        description="Número de variações estocásticas usadas.",
    )
    specs["serfiq_alpha"] = ColumnSpec(
        name="serfiq_alpha", label="SER-FIQ α",
        role=ColumnRole.METADATA, source=ColumnSource.SERFIQ,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_SERFIQ,
        description="Parâmetro logístico de calibração.",
    )
    specs["serfiq_embedding_serfiq"] = ColumnSpec(
        name="serfiq_embedding_serfiq", label="SER-FIQ Mean Embedding",
        role=ColumnRole.EMBEDDING, source=ColumnSource.SERFIQ,
        dtype=ColumnDType.VECTOR,
        editable=False, visible_default=False, category=CATEGORY_SERFIQ,
        dim=None,  # depende da arquitetura SER-FIQ subjacente
        model_name="serfiq_mean",
    )

    return specs


# =============================================================================
# API pública
# =============================================================================

def get_default_specs(dataset_type: DatasetType) -> Dict[str, ColumnSpec]:
    """Retorna o catálogo default de ColumnSpecs para o `dataset_type` dado.

    Reconstrói a cada chamada — segura para o consumidor mutar (ex: adicionar
    uma coluna custom). Custo é trivial (~80 specs imutáveis).
    """
    if dataset_type == DatasetType.FACE:
        return _face_default_specs()
    raise ValueError(f"Dataset type sem catálogo registrado: {dataset_type}")


def build_role_index(specs: Dict[str, ColumnSpec]) -> Dict[ColumnRole, List[str]]:
    """Indexa specs por role: `role → [nomes de colunas com esse role]`.

    A ordem em cada lista preserva a ordem de inserção dos specs (insertion
    order do dict). Esse índice é o caminho recomendado para o código pegar
    "todos os embeddings", "todas as quality scores", etc.
    """
    index: Dict[ColumnRole, List[str]] = {}
    for name, spec in specs.items():
        index.setdefault(spec.role, []).append(name)
    return index


def validate_required_roles(
    specs: Dict[str, ColumnSpec],
    dataset_type: DatasetType,
) -> List[str]:
    """Retorna lista de mensagens de erro: roles requeridas pelo type ausentes."""
    errors: List[str] = []
    required = REQUIRED_ROLES_BY_TYPE.get(dataset_type, frozenset())
    role_index = build_role_index(specs)
    for role in required:
        if not role_index.get(role):
            errors.append(f"Role obrigatória ausente: {role.value}")
    return errors


def validate_allowed_roles(
    specs: Dict[str, ColumnSpec],
    dataset_type: DatasetType,
) -> List[str]:
    """Retorna lista de mensagens de erro: colunas com roles não permitidas pelo type."""
    errors: List[str] = []
    allowed = ALLOWED_ROLES_BY_TYPE.get(dataset_type, frozenset())
    for name, spec in specs.items():
        if spec.role not in allowed:
            errors.append(
                f"Coluna '{name}' tem role '{spec.role.value}' não permitida "
                f"para dataset_type={dataset_type.value}"
            )
    return errors


def specs_to_json(specs: Dict[str, ColumnSpec]) -> str:
    """Serializa o catálogo de specs para string JSON (persistência em parquet metadata)."""
    return json.dumps({name: spec.to_dict() for name, spec in specs.items()},
                      ensure_ascii=False, separators=(",", ":"))


def specs_from_json(payload: str) -> Dict[str, ColumnSpec]:
    """Inverso de specs_to_json."""
    raw = json.loads(payload)
    return {name: ColumnSpec.from_dict(d) for name, d in raw.items()}
