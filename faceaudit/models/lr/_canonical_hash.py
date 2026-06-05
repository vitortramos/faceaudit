# faceaudit/models/lr/_canonical_hash.py
"""
Helpers internos para cálculo de fingerprints canônicos do subsistema LR.

Convenções fixas para garantir estabilidade cross-platform dos hashes:
- JSON: `sort_keys=True, separators=(",", ":"), ensure_ascii=False`.
- Floats: `float.hex(value)` em vez de repr decimal — bit-exact.
- Numpy arrays: convertidos para `dtype='<f8'` (little-endian) + ordem C.
- Bytes/strings sempre UTF-8.

Estes helpers não dependem do código numérico do builder e podem ser
reutilizados em qualquer outro fingerprint que precise ser determinístico
e cross-platform (state, raw input, score vector).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    """
    Serialização canônica de um dict/list/scalar para uso em hashes.

    - Chaves ordenadas em todos os níveis.
    - Sem espaços.
    - Sem escape de unicode (UTF-8 nativo).
    - Floats são convertidos para `float.hex(...)` antes da serialização
      pelo chamador (esta função não faz a conversão automaticamente
      — quem chama é responsável por normalizar floats em strings).
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_hex(data: bytes) -> str:
    """SHA-256 em hex (lowercase). Conveniência."""
    return hashlib.sha256(data).hexdigest()


def hash_canonical_payload(payload: Any) -> str:
    """SHA-256 da serialização canônica de `payload` (UTF-8)."""
    return sha256_hex(canonical_json(payload).encode("utf-8"))


def hexify_float(v: float | None) -> str | None:
    """Conversão bit-exact de float para string hex. None passa por."""
    if v is None:
        return None
    return float(v).hex()
