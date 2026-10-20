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
    CAPTURE = "capture"            # multi (com capture_subtype) — ver abaixo
    CUSTOM = "custom"              # multi — colunas adicionadas pelo usuário


class ColumnSource(str, Enum):
    """De onde a coluna vem."""
    MANUAL = "manual"      # anotação humana
    FF = "ff"              # ForensicFace
    OFIQ = "ofiq"          # OFIQ
    SERFIQ = "serfiq"      # SER-FIQ
    DERIVED = "derived"    # recomputado em runtime (ex: file_name de image_path)
    EXIF = "exif"          # metadados embutidos no arquivo (EXIF/XMP/IPTC)
    CUSTOM = "custom"      # criado pelo usuário


class ColumnDType(str, Enum):
    """Tipo do valor armazenado."""
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    VECTOR = "vector"  # list[float] de tamanho fixo (embeddings)
    LIST = "list"      # list de tamanho variável


class CaptureSubtype(str, Enum):
    """Que aspecto da CAPTURA a coluna descreve.

    Mesmo padrão de `quality_subtype` e `axis`: um role só (`CAPTURE`) com
    subtipo, em vez de um role por grupo. Multiplicar roles multiplicaria o
    branching do código sem ganho — o que o código precisa distinguir é
    "descreve a captura da foto original" contra "descreve a face".
    """

    DATETIME = "datetime"        # quando a foto foi tirada
    DEVICE = "device"            # câmera/celular que a produziu
    EXPOSURE = "exposure"        # parâmetros óticos da tomada
    GEOLOCATION = "geolocation"  # onde — dado pessoal, oculto por padrão
    SOFTWARE = "software"        # o que gravou/editou o arquivo
    RAW = "raw"                  # o dump completo, em JSON


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
    # Nomes legados aceitos como sinônimos do `name` canônico ao ler
    # parquets/dicts antigos. Útil em casos de renomeação como
    # `face_hash` → `hash` (nome canônico no catálogo) em que parquets
    # já em disco continuam tendo a coluna `face_hash`. NÃO usado na
    # escrita — apenas pra resolução tolerante na leitura.
    aliases: Optional[Tuple[str, ...]] = None

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

    # --- role=capture ---
    capture_subtype: Optional["CaptureSubtype"] = None
    # O valor é DECLARADO pelo arquivo, não medido pelo sistema. Metadado
    # embutido pode ser editado ou forjado — e num exame às vezes é
    # exatamente isso que está sob suspeita. A UI mostra a ressalva, do
    # mesmo modo que `is_estimate` marca a demografia estimada pelo FF.
    is_declared: bool = False

    # --- role=custom ---
    default_value: Optional[Any] = None

    # PR 23 (Tema 4.3) — coluna computada (read-only) com fórmula
    # simbólica. Quando `computed_expr` é não-nulo, o valor exibido é
    # avaliado por linha via `expression_evaluator.evaluate_expression`,
    # com o row do DataFrame como namespace de variáveis. A coluna NÃO
    # é editável (UI rejeita edição) e o valor NÃO é gravado no DataFrame
    # — só a string `computed_expr` round-trippa via `column_specs` na
    # metadata do parquet. Exemplos: "abs(yaw)", "fiqa_score * 0.5",
    # "1 if gender == 'F' else 0". Apenas válido com role=CUSTOM.
    computed_expr: Optional[str] = None

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

_TUPLE_FIELDS = {"scale", "vocabulary", "aliases"}


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
CATEGORY_CAPTURE_DATETIME = "Captura — Data e hora"
CATEGORY_CAPTURE_DEVICE = "Captura — Equipamento"
CATEGORY_CAPTURE_EXPOSURE = "Captura — Tomada"
CATEGORY_CAPTURE_GEO = "Captura — Localização"
CATEGORY_CAPTURE_SOFTWARE = "Captura — Software"
CATEGORY_CAPTURE_RAW = "Captura — Metadados completos"
CATEGORY_ERROR = "Erros (criação/enriquecimento)"
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
        # Parquets antigos (pré-refatoração schema-by-role) gravavam essa
        # coluna como `face_hash`. EmbeddingInfo (atributo Python) também
        # ainda se chama `face_hash` por compatibilidade — renomear o
        # atributo afetaria ~40 arquivos. O alias permite que o loader
        # encontre a coluna por qualquer um dos nomes.
        aliases=("face_hash",),
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
        # Faixa do CR-FIQA, que é o que o ForensicFace usa (`cr_fiqa_l`).
        # Estava declarada como (0, 100), faixa que o modelo nunca produz:
        # medido em 599 faces reais, os valores vão de 1,46 a 2,56. A
        # declaração errada não causava dano porque nada consome `scale`
        # hoje — mas qualquer normalização futura mostraria todas as faces
        # comprimidas em 2% da barra. O teto é generoso de propósito: o
        # CR-FIQA não tem limite superior formal.
        scale=(0.0, 3.0), quality_subtype=QualitySubtype.PRIMARY,
    )
    specs["det_score"] = ColumnSpec(
        name="det_score", label="Det. Confidence",
        role=ColumnRole.QUALITY, source=ColumnSource.FF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False, category=CATEGORY_FF_QUALITY,
        metric_name="detection", direction=QualityDirection.HIGHER_IS_BETTER,
        scale=(0.0, 1.0), quality_subtype=QualitySubtype.AUX,
    )
    # Geometria da detecção. Sem ela não há como marcar a face na imagem
    # nem saber QUAL face de um quadro com várias produziu o embedding —
    # redetectar depois pode cair noutra.
    specs["bbox"] = ColumnSpec(
        name="bbox", label="Caixa da face",
        role=ColumnRole.GEOMETRY, source=ColumnSource.FF,
        dtype=ColumnDType.LIST,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        description=(
            "Caixa da face no sistema de coordenadas do ARQUIVO, em "
            "[x1, y1, x2, y2]. Quando a detecção precisou girar a imagem, a "
            "caixa já volta convertida para o arquivo como ele está gravado "
            "— a rotação aplicada fica em 'Rotação p/ detectar'."
        ),
    )
    specs["keypoints"] = ColumnSpec(
        name="keypoints", label="Pontos da face",
        role=ColumnRole.GEOMETRY, source=ColumnSource.FF,
        dtype=ColumnDType.LIST,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        description=(
            "Os cinco pontos do detector, achatados em "
            "[x1, y1, … x5, y5]: olho esquerdo, olho direito, nariz, canto "
            "esquerdo da boca, canto direito. São os pontos que definiram o "
            "alinhamento de onde saiu o embedding, no sistema do arquivo."
        ),
    )
    # Compressão: a declarada pela câmera e a calculada do arquivo atual.
    # Coexistem de propósito — a divergência entre elas é sinal de
    # recompressão depois da captura.
    specs["declared_bits_per_pixel"] = ColumnSpec(
        name="declared_bits_per_pixel", label="Bits/pixel (declarado)",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE, is_declared=True,
        unit="bits/pixel",
        description=(
            "`CompressedBitsPerPixel` do EXIF — a taxa que a CÂMERA gravou "
            "no momento da captura. NÃO acompanha recompressão posterior: "
            "se a foto foi reencodada depois, este valor continua o da "
            "origem. Compare com `image_bits_per_pixel` (calculado do "
            "arquivo atual) — divergência indica recompressão."
        ),
    )
    specs["image_bits_per_pixel"] = ColumnSpec(
        name="image_bits_per_pixel", label="Bits/pixel (calculado)",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        unit="bits/pixel",
        description=(
            "Bits por pixel efetivamente gastos pelo arquivo como ele está "
            "HOJE: tamanho × 8 ÷ (largura × altura). Reflete a ÚLTIMA "
            "compressão que o arquivo sofreu. Calculado, não declarado — "
            "não depende de a câmera ter gravado a etiqueta. Em formato sem "
            "perdas (PNG, TIFF) continua válido como ocupação, mas ali a "
            "compressão não descarta informação e o valor não indica "
            "qualidade."
        ),
    )
    specs["image_compression_ratio"] = ColumnSpec(
        name="image_compression_ratio", label="Razão de compressão",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        description=(
            "Quantas vezes o arquivo é menor que o RGB sem compressão "
            "(24 ÷ bits por pixel calculado). Um valor de 12 significa "
            "\"12:1\". Mesma base de `image_bits_per_pixel`, em unidade mais "
            "usual."
        ),
    )
    specs["jpeg_quality_estimate"] = ColumnSpec(
        name="jpeg_quality_estimate", label="Qualidade JPEG (estimada)",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.FLOAT,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        scale=(1.0, 100.0),
        description=(
            "Fator de qualidade da ÚLTIMA gravação JPEG, obtido invertendo "
            "a fórmula do IJG sobre a tabela de quantização do arquivo. "
            "Diferente de `image_bits_per_pixel`, NÃO depende do conteúdo: "
            "o mesmo ajuste rende bits/pixel muito diferentes numa imagem "
            "lisa e numa ruidosa. É uma ESTIMATIVA — codificador com "
            "tabelas próprias (alguns celulares, Photoshop na escala 0–12) "
            "não casa exatamente com a escala do IJG, e aí o número indica "
            "a ordem de grandeza. Vazio em formato sem perdas, que não tem "
            "tabela de quantização."
        ),
    )
    specs["jpeg_quality_declared"] = ColumnSpec(
        name="jpeg_quality_declared", label="Qualidade JPEG (declarada)",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE, is_declared=True,
        description=(
            "Valor de qualidade que o PROGRAMA registrou em metadado, "
            "quando registrou. Quase sempre vazio: o JPEG não guarda o "
            "número escolhido no momento de salvar — guarda a tabela de "
            "quantização que esse número produziu. Preenchido, vem do "
            "Photoshop (escala 1–12) ou de makernote de câmera (\"Fine\", "
            "\"Normal\"), escalas que NÃO são comparáveis entre si nem com "
            "`jpeg_quality_estimate` (1–100, recuperada da tabela). Texto, "
            "e não número, por isso."
        ),
    )
    specs["jpeg_quality_is_saturated"] = ColumnSpec(
        name="jpeg_quality_is_saturated", label="Qualidade JPEG saturada",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.BOOL,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        description=(
            "Verdadeiro quando a tabela de quantização tem coeficientes no "
            "teto (255), o que acontece no fundo da escala de qualidade. "
            "Ali a inversão perde resolução e a estimativa é menos precisa "
            "— sinalizar isso evita que o número seja lido como exato."
        ),
    )
    specs["jpeg_chroma_subsampling"] = ColumnSpec(
        name="jpeg_chroma_subsampling", label="Subamostragem de croma",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        vocabulary=("4:4:4", "4:2:2", "4:2:0"),
        description=(
            "Quanto de crominância o JPEG descartou. Informação de "
            "procedência: foto de câmera costuma vir 4:2:0, e um reencode "
            "em 4:4:4 denuncia passagem por outro programa."
        ),
    )
    specs["jpeg_progressive"] = ColumnSpec(
        name="jpeg_progressive", label="JPEG progressivo",
        role=ColumnRole.CAPTURE, source=ColumnSource.DERIVED,
        dtype=ColumnDType.BOOL,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE,
        description=(
            "Se o JPEG é progressivo em vez de sequencial. Câmera grava "
            "sequencial; progressivo costuma indicar reencode por "
            "ferramenta de otimização ou plataforma web."
        ),
    )
    specs["file_size_bytes"] = ColumnSpec(
        name="file_size_bytes", label="Tamanho do arquivo",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.INT,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_DEVICE,
        capture_subtype=CaptureSubtype.DEVICE, is_declared=True,
        unit="bytes",
        description=(
            "Tamanho do arquivo de imagem original em bytes. Base do "
            "cálculo de bits por pixel."
        ),
    )
    specs["image_format"] = ColumnSpec(
        name="image_format", label="Formato do arquivo",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_DEVICE,
        capture_subtype=CaptureSubtype.DEVICE, is_declared=True,
        description=(
            "Formato do arquivo original (JPEG, PNG, HEIC…). Necessário "
            "para ler a taxa de compressão: em formato sem perdas ela mede "
            "ocupação, não qualidade."
        ),
    )
    specs["video_avg_bitrate"] = ColumnSpec(
        name="video_avg_bitrate", label="Taxa de bits do vídeo",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_EXPOSURE,
        capture_subtype=CaptureSubtype.EXPOSURE, is_declared=True,
        description=(
            "Taxa de bits média do vídeo de origem — o equivalente da taxa "
            "de compressão para dataset extraído de vídeo. Texto porque o "
            "ExifTool emite com unidade (ex.: \"4.5 Mbps\")."
        ),
    )
    specs["capture_datetime_origin"] = ColumnSpec(
        name="capture_datetime_origin", label="Origem da data",
        role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False,
        category=CATEGORY_CAPTURE_DATETIME,
        capture_subtype=CaptureSubtype.DATETIME, is_declared=True,
        description=(
            "Como a data de captura foi obtida. `FILE_TAG` = lida da "
            "etiqueta do próprio arquivo. `VIDEO_START_PLUS_FRAME` = "
            "calculada a partir do início da gravação somado ao instante do "
            "quadro de onde a face saiu. Sem esta coluna, dois valores de "
            "aparência idêntica significariam coisas diferentes."
        ),
    )
    specs["detection_rotation"] = ColumnSpec(
        name="detection_rotation", label="Rotação p/ detectar",
        role=ColumnRole.METADATA, source=ColumnSource.FF,
        dtype=ColumnDType.INT,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        unit="degrees",
        description=(
            "Rotação horária aplicada à imagem para que a face fosse "
            "detectada. Vazio significa que a face foi encontrada na "
            "orientação em que o arquivo está gravado — o caso normal. "
            "Preenchido (90, 180 ou 270) indica foto gravada de lado: a "
            "caixa e os pontos da face estão em coordenadas do arquivo "
            "original, mas a pose e o recorte alinhado vêm do quadro "
            "corrigido."
        ),
    )
    specs["detection_det_size"] = ColumnSpec(
        name="detection_det_size", label="Resolução p/ detectar",
        role=ColumnRole.METADATA, source=ColumnSource.FF,
        dtype=ColumnDType.INT,
        editable=False, visible_default=False, category=CATEGORY_FF_IDENT,
        unit="pixels",
        description=(
            "Resolução do detector que encontrou a face, quando diferente "
            "da usada na ingestão. Vazio significa que a face apareceu na "
            "resolução configurada — o caso normal. Preenchido indica face "
            "pequena no quadro, recuperada numa segunda tentativa em "
            "resolução maior pedida durante a revisão."
        ),
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
    # Embeddings dos modelos Sepael — gerados a partir do enum em vez
    # de hard-codear v2/v4. Qualquer modelo novo adicionado ao
    # `SepaelModel` ganha ColumnSpec automaticamente. Sem isso, colunas
    # como `embedding_v6` aparecem no parquet mas a UI não consegue
    # togglar visibilidade (sem spec → sem entrada no menu Mostrar/Ocultar).
    from faceaudit.enums.sepael_model import SepaelModel
    for member in SepaelModel:
        specs[member.parquet_column] = ColumnSpec(
            name=member.parquet_column,
            label=member.label_pt,
            role=ColumnRole.EMBEDDING,
            source=ColumnSource.FF,
            dtype=ColumnDType.VECTOR,
            editable=False,
            visible_default=False,
            category=CATEGORY_FF_EMBEDDING,
            dim=member.embedding_dim,
            model_name=member.ff_name,
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

    # Escalares 0–100 de pose do OFIQ. Diferente das 24 dimensões geradas
    # por `_gen_ofiq_extra_specs`, o OFIQ emite só a versão `_scalar` para
    # pose (o valor bruto em graus vai para `ofiq_yaw`/`_pitch`/`_roll`
    # acima). Ficam com role=QUALITY, e não POSE, justamente para não
    # concorrerem com as colunas em graus no índice por role/axis.
    for _axis_name, _axis_label in (("yaw", "Yaw"), ("pitch", "Pitch"), ("roll", "Roll")):
        specs[f"ofiq_head_pose_{_axis_name}_scalar"] = ColumnSpec(
            name=f"ofiq_head_pose_{_axis_name}_scalar",
            label=f"OFIQ Head Pose {_axis_label} (scalar)",
            role=ColumnRole.QUALITY, source=ColumnSource.OFIQ,
            dtype=ColumnDType.FLOAT,
            editable=False, visible_default=False,
            category=CATEGORY_OFIQ_DIMENSIONS,
            metric_name=f"ofiq_head_pose_{_axis_name}",
            direction=QualityDirection.HIGHER_IS_BETTER,
            scale=(0.0, 100.0), quality_subtype=QualitySubtype.DIMENSION,
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
    specs["ofiq_version"] = ColumnSpec(
        name="ofiq_version", label="OFIQ Versão",
        role=ColumnRole.METADATA, source=ColumnSource.OFIQ,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_METADATA,
        description=(
            "Versão do OFIQ que calculou as métricas desta face. A v1.0.3 é a "
            "implementação de referência da ISO/IEC 29794-5; da v1.1.0 em "
            "diante, implementação conforme. Vazio = calculada antes de o "
            "FaceAudit registrar a versão."
        ),
    )
    specs["ofiq_backend"] = ColumnSpec(
        name="ofiq_backend", label="OFIQ Modo",
        role=ColumnRole.METADATA, source=ColumnSource.OFIQ,
        dtype=ColumnDType.STR,
        editable=False, visible_default=False, category=CATEGORY_OFIQ_METADATA,
        description=(
            "Como o OFIQ foi chamado: 'executable' (programa OFIQSampleApp) "
            "ou 'library' (biblioteca carregada pelo FaceAudit)."
        ),
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

    # ---------- Rastreamento de erros ----------
    # Populadas por `process_dataset` (criação) e `enrich_existing_dataset`
    # (enriquecimento) quando alguma face falha em algum estágio. Permite
    # ao perito filtrar `error=True` na tabela pra ver as falhas e
    # planejar reanálise sem precisar de parquet separado. As 3 colunas
    # vêm visible_default=True só quando algum face tem error setado —
    # mas o spec aqui marca True (caso default não filtra "vazias").
    # ── Captura (EXIF/XMP/IPTC da foto ORIGINAL) ──
    # Todas nascem ocultas menos a data/hora crua: são ~20 colunas, e
    # despejá-las na tabela afogaria o resto. Quem precisa de uma delas
    # liga pelo menu Mostrar/Ocultar.
    #
    # `is_declared=True` em todas: o valor é declaração do ARQUIVO, não
    # medida do sistema. Metadado embutido pode ser editado ou forjado — e
    # num exame às vezes é exatamente isso que está sob suspeita.
    #
    # Só existem em dataset de fotos ORIGINAIS: recorte e alinhamento
    # gravam pixel novo, sem metadado de captura, e não há de onde
    # recuperar depois.
    def _captura(
        nome, rotulo, subtipo, dtype, categoria, *, visivel=False, descricao=None,
    ):
        specs[nome] = ColumnSpec(
            name=nome, label=rotulo,
            role=ColumnRole.CAPTURE, source=ColumnSource.EXIF,
            dtype=dtype, editable=False, visible_default=visivel,
            category=categoria, capture_subtype=subtipo, is_declared=True,
            description=descricao,
        )

    _captura(
        "capture_datetime_raw", "Data/hora da captura (como gravada)",
        CaptureSubtype.DATETIME, ColumnDType.STR, CATEGORY_CAPTURE_DATETIME,
        visivel=True,
        descricao=(
            "Valor de DateTimeOriginal exatamente como está no arquivo. A "
            "maioria das câmeras grava sem fuso horário."
        ),
    )
    _captura(
        "capture_datetime", "Data/hora da captura (normalizada)",
        CaptureSubtype.DATETIME, ColumnDType.STR, CATEGORY_CAPTURE_DATETIME,
        descricao=(
            "ISO-8601 com fuso. Preenchida SÓ quando o arquivo declara o "
            "offset explicitamente — sem isso, atribuir um fuso seria "
            "inventar informação."
        ),
    )
    _captura(
        "capture_timezone_offset", "Fuso declarado",
        CaptureSubtype.DATETIME, ColumnDType.STR, CATEGORY_CAPTURE_DATETIME,
    )
    _captura(
        "camera_make", "Fabricante",
        CaptureSubtype.DEVICE, ColumnDType.STR, CATEGORY_CAPTURE_DEVICE,
    )
    _captura(
        "camera_model", "Modelo",
        CaptureSubtype.DEVICE, ColumnDType.STR, CATEGORY_CAPTURE_DEVICE,
    )
    _captura(
        "camera_serial", "Número de série do corpo",
        CaptureSubtype.DEVICE, ColumnDType.STR, CATEGORY_CAPTURE_DEVICE,
        descricao=(
            "Quando presente, liga a imagem a um aparelho específico — não "
            "só a um modelo."
        ),
    )
    _captura(
        "lens_model", "Lente",
        CaptureSubtype.DEVICE, ColumnDType.STR, CATEGORY_CAPTURE_DEVICE,
    )
    _captura(
        "iso", "ISO",
        CaptureSubtype.EXPOSURE, ColumnDType.INT, CATEGORY_CAPTURE_EXPOSURE,
    )
    _captura(
        "f_number", "Abertura (f)",
        CaptureSubtype.EXPOSURE, ColumnDType.FLOAT, CATEGORY_CAPTURE_EXPOSURE,
    )
    _captura(
        "exposure_time", "Tempo de exposição",
        CaptureSubtype.EXPOSURE, ColumnDType.STR, CATEGORY_CAPTURE_EXPOSURE,
        descricao="Como gravado (ex.: 1/125).",
    )
    _captura(
        "focal_length_mm", "Distância focal (mm)",
        CaptureSubtype.EXPOSURE, ColumnDType.FLOAT, CATEGORY_CAPTURE_EXPOSURE,
    )
    _captura(
        "focal_length_35mm", "Distância focal equiv. 35mm",
        CaptureSubtype.EXPOSURE, ColumnDType.FLOAT, CATEGORY_CAPTURE_EXPOSURE,
        descricao="Comparável entre sensores de tamanhos diferentes.",
    )
    _captura(
        "orientation", "Orientação",
        CaptureSubtype.EXPOSURE, ColumnDType.INT, CATEGORY_CAPTURE_EXPOSURE,
        descricao="Explica rotação aparente do rosto sem que a foto tenha sido girada.",
    )
    _captura(
        "image_width", "Largura original (px)",
        CaptureSubtype.EXPOSURE, ColumnDType.INT, CATEGORY_CAPTURE_EXPOSURE,
        descricao="Dimensão do arquivo de origem, antes de qualquer recorte.",
    )
    _captura(
        "image_height", "Altura original (px)",
        CaptureSubtype.EXPOSURE, ColumnDType.INT, CATEGORY_CAPTURE_EXPOSURE,
    )
    _captura(
        "software", "Software",
        CaptureSubtype.SOFTWARE, ColumnDType.STR, CATEGORY_CAPTURE_SOFTWARE,
        descricao=(
            "O que gravou ou editou o arquivo. Sinal de reprocessamento: "
            "uma imagem que passou por editor não é a captura original."
        ),
    )
    _captura(
        "gps_latitude", "Latitude",
        CaptureSubtype.GEOLOCATION, ColumnDType.FLOAT, CATEGORY_CAPTURE_GEO,
        descricao="Dado pessoal — atenção ao compartilhar o dataset.",
    )
    _captura(
        "gps_longitude", "Longitude",
        CaptureSubtype.GEOLOCATION, ColumnDType.FLOAT, CATEGORY_CAPTURE_GEO,
        descricao="Dado pessoal — atenção ao compartilhar o dataset.",
    )
    _captura(
        "gps_altitude", "Altitude (m)",
        CaptureSubtype.GEOLOCATION, ColumnDType.FLOAT, CATEGORY_CAPTURE_GEO,
    )
    _captura(
        "exif_json", "Metadados completos (JSON)",
        CaptureSubtype.RAW, ColumnDType.STR, CATEGORY_CAPTURE_RAW,
        descricao=(
            "Dump íntegro do que foi lido do arquivo — EXIF, XMP, IPTC, "
            "maker notes. É daqui que a aba de metadados lê; as colunas "
            "acima são só os campos promovidos para consulta rápida."
        ),
    )

    specs["error"] = ColumnSpec(
        name="error", label="Erro",
        role=ColumnRole.METADATA, source=ColumnSource.DERIVED,
        dtype=ColumnDType.BOOL,
        editable=False, visible_default=True, category=CATEGORY_ERROR,
        description=(
            "True quando a face falhou em algum estágio da última "
            "execução de criação/enriquecimento. Veja error_stage e "
            "error_message para detalhes."
        ),
    )
    specs["error_stage"] = ColumnSpec(
        name="error_stage", label="Estágio do Erro",
        role=ColumnRole.METADATA, source=ColumnSource.DERIVED,
        dtype=ColumnDType.STR,
        editable=False, visible_default=True, category=CATEGORY_ERROR,
        description=(
            "Estágio(s) onde a falha ocorreu: 'scan' (load), 'ff', "
            "'ofiq', 'serfiq'. Múltiplos estágios separados por vírgula."
        ),
    )
    specs["error_message"] = ColumnSpec(
        name="error_message", label="Mensagem do Erro",
        role=ColumnRole.METADATA, source=ColumnSource.DERIVED,
        dtype=ColumnDType.STR,
        editable=False, visible_default=True, category=CATEGORY_ERROR,
        description=(
            "Mensagem(ns) detalhada(s) do que falhou em cada estágio. "
            "Formato: '[estágio] mensagem | [estágio] mensagem'."
        ),
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


def resolve_column_name(
    available_columns,
    spec: ColumnSpec,
) -> Optional[str]:
    """Encontra o nome real da coluna em `available_columns` que corresponde
    ao `spec`, considerando o nome canônico E os aliases legados.

    Args:
        available_columns: Iterável (set, list, dict.keys()) com os nomes
            das colunas presentes no parquet/DataFrame que está sendo lido.
        spec: ColumnSpec a procurar.

    Returns:
        Nome da coluna em `available_columns` (canônico se disponível;
        senão o primeiro alias encontrado). None se não há match nenhum.

    Exemplo:
        spec = specs["hash"]  # canônico "hash", aliases=("face_hash",)
        col = resolve_column_name({"face_hash", "identity"}, spec)
        # col == "face_hash"  (canônico não está, alias está)
    """
    cols = set(available_columns)
    if spec.name in cols:
        return spec.name
    if spec.aliases:
        for alias in spec.aliases:
            if alias in cols:
                return alias
    return None


def dtype_from_pandas_kind(kind: str, sample: Any = None) -> ColumnDType:
    """Mapeia o `dtype.kind` de uma Series pandas para um `ColumnDType`.

    Recebe a letra do kind (não a Series) para manter este módulo puro —
    sem dependência de pandas/numpy. Para `kind == "O"` (object), inspeciona
    `sample` (primeiro valor não-nulo da coluna) para distinguir
    str/int/float/bool/list.
    """
    simple = {"b": ColumnDType.BOOL, "i": ColumnDType.INT,
              "u": ColumnDType.INT, "f": ColumnDType.FLOAT}
    if kind in simple:
        return simple[kind]
    if kind != "O":
        # 'U'/'S' (string), 'M'/'m' (datetime/timedelta) → texto.
        return ColumnDType.STR
    if isinstance(sample, bool):
        return ColumnDType.BOOL
    if isinstance(sample, int):
        return ColumnDType.INT
    if isinstance(sample, float):
        return ColumnDType.FLOAT
    if isinstance(sample, (list, tuple)):
        return ColumnDType.LIST
    # ndarray e afins: sequência sem ser string/bytes.
    if sample is not None and not isinstance(sample, (str, bytes)) and hasattr(sample, "__len__"):
        return ColumnDType.LIST
    return ColumnDType.STR


def complete_missing_specs(
    specs: Dict[str, ColumnSpec],
    columns,
    dtypes: Optional[Dict[str, ColumnDType]] = None,
) -> Dict[str, ColumnSpec]:
    """Cria specs `role=custom` para colunas reais que não têm spec nenhum.

    Uma coluna sem `ColumnSpec` é invisível na UI: o table model renderiza
    exatamente `column_specs` (`_columns_from_specs`), e o catálogo default
    só cobre as colunas conhecidas. Colunas vindas de custom columns de
    template (`DirTemplate.custom_columns` → `FaceInfo.extra_data`) ou de
    enriquecimentos ainda não catalogados caíam nesse buraco: persistiam no
    parquet com os dados corretos, mas sem forma de exibi-las ou filtrá-las.

    Esta função é a rede de segurança: toda coluna presente no DataFrame
    ganha spec. `role=CUSTOM` é deliberado — um spec sintetizado NUNCA deve
    reivindicar um role semântico (EMBEDDING/POSE/QUALITY) sobre o qual o
    código faz branch, sob risco de a coluna ser consumida como se fosse
    canônica. Colunas realmente conhecidas devem ser declaradas no catálogo.

    Args:
        specs: specs já resolvidos (catálogo + metadata do parquet).
        columns: nomes das colunas presentes no DataFrame.
        dtypes: mapa opcional `nome → ColumnDType` (ver `dtype_from_pandas_kind`).
            Ausente → assume `STR`.

    Returns:
        Apenas os specs NOVOS (`nome → ColumnSpec`); o caller decide como
        mesclar. Vazio quando toda coluna já tem spec.
    """
    known = set(specs)
    for spec in specs.values():
        if spec.aliases:
            known.update(spec.aliases)

    minted: Dict[str, ColumnSpec] = {}
    for name in columns:
        if name in known or name in minted:
            continue
        dtype = (dtypes or {}).get(name, ColumnDType.STR)
        # Vetores/listas não são editáveis nem exibíveis de forma útil numa
        # célula de tabela — registra o spec (para não sumirem do schema),
        # mas fora da visão default.
        is_sequence = dtype in (ColumnDType.LIST, ColumnDType.VECTOR)
        minted[name] = ColumnSpec(
            name=name,
            label=name,
            role=ColumnRole.CUSTOM,
            source=ColumnSource.CUSTOM,
            dtype=dtype,
            editable=not is_sequence,
            visible_default=not is_sequence,
            category=CATEGORY_CUSTOM,
            description=(
                "Coluna sem spec no catálogo — registrada automaticamente "
                "para permanecer visível e editável na tabela."
            ),
        )
    return minted


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
