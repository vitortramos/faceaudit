"""Qual FaceAudit produziu um artefato: versão + commit do código.

Vai para TODA entrada da cadeia de custódia (`AuditLogService.append`),
para o manifesto do Processar e da extração de vídeo e para os metadados
da execução de estratégia. A versão sozinha não basta: ela ficou em 0.1.0
por quase 600 commits, e dois artefatos com a mesma versão podiam ter
saído de códigos diferentes. O commit identifica o código; `git_dirty`
avisa quando o código em execução tinha alterações não commitadas em
`faceaudit/` ou `interface/` — aí o commit sozinho não o reproduz.
`config.json` e dados versionados ficam fora de propósito: não são
código e estão quase sempre modificados localmente.

O commit não fixa as bibliotecas: o `pyproject.toml` deixa a maioria sem
versão, e `forensicface`/`onnxruntime` decidem o embedding. Por isso
`dependencies` registra a versão instalada das que mudam o que o programa
produz (detecção, alinhamento, inferência, leitura de imagem, estatística),
além de Python e plataforma — o provedor de execução do ONNX (CUDA, CoreML,
CPU) depende do sistema e mexe nas últimas casas do embedding.

Lido UMA vez por processo: o que roda é o código carregado na abertura,
e editar um arquivo depois não muda o que está em memória. Fora de um
checkout git (ou sem git no PATH) o commit fica None — o registro diz
que não se sabe, em vez de inventar.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from faceaudit._version import __version__

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIRS = ("faceaudit", "interface")

# Distribuições (nomes do pip) cuja versão entra na proveniência. Variantes
# do mesmo pacote (onnxruntime-gpu × onnxruntime, opencv headless × GUI)
# aparecem só se instaladas — saber QUAL foi usada faz parte do registro.
_TRACKED_DISTRIBUTIONS = (
    "forensicface",
    "onnxruntime-gpu", "onnxruntime", "onnxruntime-silicon",
    "opencv-python-headless", "opencv-python",
    "numpy", "scipy", "scikit-learn", "speechbrain", "Pillow",
)


def _git(*args: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", *args], cwd=str(_REPO_ROOT),
                           capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


@lru_cache(maxsize=1)
def _git_state() -> Tuple[Optional[str], Optional[bool]]:
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no", "--", *_CODE_DIRS)
    return commit, (None if status is None else bool(status))


@lru_cache(maxsize=1)
def _dependency_versions() -> Tuple[Tuple[str, str], ...]:
    found = []
    for dist in _TRACKED_DISTRIBUTIONS:
        try:
            found.append((dist, metadata.version(dist)))
        except metadata.PackageNotFoundError:
            continue
    return tuple(found)


def dependency_versions() -> Dict[str, str]:
    """Versão instalada de cada biblioteca rastreada (ausentes ficam de fora)."""
    return dict(_dependency_versions())


def software_provenance() -> Dict[str, Any]:
    """`{"faceaudit_version", "git_commit", "git_dirty", "python", "platform",
    "dependencies"}` — JSON puro."""
    commit, dirty = _git_state()
    return {
        "faceaudit_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "dependencies": dependency_versions(),
    }


def commit_label() -> Optional[str]:
    """Commit curto com `-dirty` quando há alterações — formato do manifesto
    de vídeo."""
    commit, dirty = _git_state()
    if not commit:
        return None
    return commit[:10] + ("-dirty" if dirty else "")
