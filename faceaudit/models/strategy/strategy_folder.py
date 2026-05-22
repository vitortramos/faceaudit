# faceaudit/models/strategy/strategy_folder.py
"""Helpers de naming da subpasta hierárquica de estratégia.

A nova arquitetura agrupa todos os artefatos derivados de uma estratégia
dentro de subpastas semânticas (`models/`, `scores/`, `results/`,
`graphics/`). Cada estratégia ocupa UMA subpasta dentro de cada uma delas,
nomeada pelo padrão canônico `<sanitize(strategy_name)>_<strategy_id>`.

Combinar `strategy_name` (legível) + `strategy_id` (UUID único) no nome
da pasta serve dois propósitos:
- Listagem filesystem amigável (perito identifica a estratégia sem
  precisar abrir nenhum JSON).
- Disambiguação garantida — duas estratégias com o mesmo nome (após
  duplicação, antes do rename) ainda têm folders distintos via UUID.

NÃO usar em pastas FLAT (datasets/, pipelines/, definitions/, etc.) —
nelas o `strategy_id` continua como prefixo do filename, não como pasta.
"""

from __future__ import annotations


# Caracteres inválidos pra nomes de arquivo/pasta em Windows + POSIX.
# Mantemos a mesma lista do `_sanitize_name` legado em distribution_plot.py
# (espaço incluído pra evitar trailing-space issues em alguns FS).
_INVALID_CHARS = '\\/:*?"<>| '


def sanitize_name(raw: str) -> str:
    """Substitui caracteres inválidos por `_`. Retorna `'unknown'` se vazio.

    Idempotente — sanitizar duas vezes produz o mesmo resultado.
    """
    out = raw or "unknown"
    for ch in _INVALID_CHARS:
        out = out.replace(ch, "_")
    return out


def strategy_folder_name(strategy_id: str, strategy_name: str = "") -> str:
    """Retorna o nome canônico da subpasta de uma estratégia.

    Formato: `<sanitize(strategy_name)>_<strategy_id>`.

    Se `strategy_name` for vazio/None (caso fallback raro — chamada antes
    da estratégia ser nomeada), retorna apenas `<strategy_id>` pra não
    gerar `unknown_<id>` ruído. Compatível com layout pré-rename de
    pastas: rglob pega ambos os formatos.
    """
    safe_id = sanitize_name(strategy_id)
    if not strategy_name:
        return safe_id
    safe_name = sanitize_name(strategy_name)
    return f"{safe_name}_{safe_id}"
