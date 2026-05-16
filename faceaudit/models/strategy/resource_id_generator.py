# faceaudit/models/strategy/resource_id_generator.py

"""Geração de IDs únicos pra StrategyResource.

Formato: `{PREFIXO}-{8hex}` (ex: `DS-8f3a1b2c`).

- Prefixo curto por tipo de recurso — torna o id auto-documentado
  ("se vejo DS-... é um Dataset").
- 8 chars hex = 4 bilhões de IDs por tipo, colisão astronômica.
- Lex-comparable e estável a serialização JSON.

Duas modalidades:

1. **`generate_resource_id(resource_type)`** — id aleatório novo, usado
   quando o usuário adiciona um recurso pela primeira vez ou quando o
   executor produz um recurso derivado.

2. **`migrate_resource_id(legacy_id, resource_type)`** — id determinístico
   baseado em hash do legacy_id + tipo. Garante que projetos antigos
   abertos em máquinas diferentes recebam o MESMO id no boot (sem isso,
   o "pendrive paradigm" quebraria — IDs divergiriam entre máquinas
   se geração fosse aleatória).
"""

from __future__ import annotations

import hashlib
import uuid

from faceaudit.enums.strategy_resource_type import StrategyResourceType


# Prefixo curto por tipo de recurso. Em commits anteriores propus
# DS/PL/MC/SC/TP/TS/TC/SS/CS/VR/CV/HR/HA. Mapeamento canônico:
_TYPE_PREFIXES: dict[StrategyResourceType, str] = {
    StrategyResourceType.DATASET:                   "DS",
    StrategyResourceType.PIPELINE_CONFIG:           "PL",
    StrategyResourceType.SPLIT_CONFIG:              "SC",
    StrategyResourceType.MINING_CONFIG:             "MC",
    StrategyResourceType.MERGE_CONFIG:              "MG",
    StrategyResourceType.SIMILARITY_MODEL_CONFIG:   "SM",
    StrategyResourceType.CALIBRATION_MODEL_CONFIG:  "CC",
    StrategyResourceType.TRAINED_PIPELINE:          "TP",
    StrategyResourceType.TRAINED_SIMILARITY_MODEL:  "TS",
    StrategyResourceType.TRAINED_CALIBRATION_MODEL: "TC",
    StrategyResourceType.SIMILARITY_SCORES:         "SS",
    StrategyResourceType.CALIBRATED_SCORES:         "CS",
    StrategyResourceType.TRAJECTORY_DATA:           "TJ",
    StrategyResourceType.VALIDATION_RESULT:         "VR",
    StrategyResourceType.CASEWORK_VISUALIZATION:    "CV",
    StrategyResourceType.HUBNESS_REPORT:            "HR",
    StrategyResourceType.HUBNESS_ABLATION_RESULT:   "HA",
    StrategyResourceType.DATASET_STATS:             "DT",
    StrategyResourceType.ARTIFACT:                  "AR",
    StrategyResourceType.UNKNOWN:                   "UN",
}

# Fallback quando o tipo é UNKNOWN ou não mapeado.
_DEFAULT_PREFIX = "UN"

# Comprimento da parte hex do id (após o prefixo + hífen).
_HEX_LEN = 8


def prefix_for(resource_type: StrategyResourceType) -> str:
    """Retorna o prefixo de 2 chars pro tipo informado."""
    return _TYPE_PREFIXES.get(resource_type, _DEFAULT_PREFIX)


def generate_resource_id(resource_type: StrategyResourceType) -> str:
    """Gera um id novo aleatório no formato `{PREFIXO}-{8hex}`.

    Usado quando:
    - Usuário adiciona um arquivo da biblioteca como recurso novo do projeto.
    - Executor produz um recurso derivado (instrução roda e cria
      TRAINED_PIPELINE, SIMILARITY_SCORES, etc.).
    """
    prefix = prefix_for(resource_type)
    rnd = uuid.uuid4().hex[:_HEX_LEN]
    return f"{prefix}-{rnd}"


def migrate_resource_id(legacy_id: str, resource_type: StrategyResourceType) -> str:
    """Gera id determinístico pra migração de projetos antigos.

    Hash SHA-256 do `legacy_id` (8 primeiros hex chars) garante que o
    mesmo recurso recebe o MESMO id em qualquer máquina/abertura — crítico
    pro pendrive paradigm (mover projeto entre Windows/macOS gera ids
    iguais).

    Args:
        legacy_id: o `resource_id` original do JSON antigo (pre-Fase 2).
        resource_type: usado pra escolher o prefixo correto.

    Returns:
        Id no formato `{PREFIXO}-{8hex}` baseado em hash.
    """
    prefix = prefix_for(resource_type)
    digest = hashlib.sha256(legacy_id.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:_HEX_LEN]}"


def is_valid_resource_id(value: str) -> bool:
    """Validação rasa do formato `XX-12345678`.

    Não verifica se o prefixo é conhecido — apenas formato genérico.
    """
    if not isinstance(value, str) or len(value) != 3 + _HEX_LEN:
        return False
    if value[2] != "-":
        return False
    prefix = value[:2]
    if not (prefix[0].isalpha() and prefix[1].isalpha()):
        return False
    hex_part = value[3:]
    try:
        int(hex_part, 16)
    except ValueError:
        return False
    return True
