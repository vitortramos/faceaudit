"""Resolve `image_path` de um FaceInfo/EmbeddingInfo em path absoluto
pronto pra leitura (PR L1a, normalização cross-platform L1c.1).

Free-function — recebe parâmetros crus, sem dependência de
`FaceDataset`. Mora em `services/` por convenção do projeto, mas é
stateless.

A motivação é desacoplar os services (ff, hash, ofiq, serfiq) do
modelo `FaceDataset`: eles passam a receber apenas os 3 parâmetros
de resolução (parquet_dir, path_mode, path_root) sem precisar
conhecer a classe inteira do dataset.

Atalho de conveniência: `FaceDataset.resolve_image_path(face)` delega
a essa função preenchendo os parâmetros a partir do próprio dataset.

**Convenção cross-platform (L1c.1)** — todos os paths armazenados em
disco (parquet metadata, `FaceInfo.image_path`, `FaceDataset.path_root`)
usam `/` como separador, independente do SO. Em runtime, `pathlib.Path`
aceita `/` em todos os SOs (até Windows com `C:/Users/...`). Padrão
herdado de `StrategyArtifactReference.from_dict` (mesma motivação:
parquets criados no Windows abrirem no macOS sem normalização manual).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from faceaudit.enums.path_mode import PathMode


def normalize_stored_path(raw: str) -> str:
    """Normaliza path-string pra POSIX (forward slashes).

    Aplicar a qualquer string de path **antes de armazenar em disco**
    e **defensivamente ao ler** parquets/JSON gerados em outras
    plataformas. `pathlib.Path` aceita `/` em todos os SOs.

    No-op pra strings sem `\\`. Não toca em paths absolutos do macOS/
    Linux (já em POSIX). Em paths absolutos do Windows (`C:\\Users\\...`)
    converte pra `C:/Users/...` — `Path` no Windows aceita ambos.

    Args:
        raw: string de path possivelmente com mistura de separadores.

    Returns:
        String com apenas `/` como separador.
    """
    return raw.replace("\\", "/")


def resolve_image_path(
    face_or_path: Any,
    *,
    parquet_dir: Optional[Union[str, Path]] = None,
    path_mode: PathMode = PathMode.ABSOLUTE,
    path_root: Optional[str] = None,
) -> Path:
    """Resolve um `image_path` em path absoluto pronto pra leitura.

    Args:
        face_or_path: pode ser um FaceInfo, EmbeddingInfo, str ou Path —
            qualquer objeto com atributo `image_path` ou um path direto.
        parquet_dir: diretório do parquet do dataset (set por
            `from_parquet` em runtime). Necessário quando
            `path_mode=RELATIVE`.
        path_mode: modo declarado pelo dataset.
        path_root: subpasta dentro de `parquet_dir` onde estão as
            imagens (vem da metadata do parquet). Default `"."` quando
            ausente em modo RELATIVE.

    Returns:
        Path absoluto. Caller pode chamar `PIL.open` / `cv2.imread`
        / `os.path.exists` direto no resultado.

    Comportamento:
        - ABSOLUTE: retorna o path como veio (sem normalizar — preserva
          comportamento atual exato).
        - RELATIVE: ancora em `(parquet_dir / path_root) / image_path`
          e resolve. Se `parquet_dir` não foi setado (dataset em
          memória sem origem em disco), faz fallback pra `cwd` —
          consistente com o comportamento herdado do path absoluto.
    """
    raw = _extract_path_str(face_or_path)
    p = Path(raw)

    if path_mode == PathMode.ABSOLUTE:
        # Preserva o comportamento atual byte-exato — não normaliza,
        # não resolve symlinks; apenas retorna como Path.
        return p

    # path_mode == RELATIVE
    if p.is_absolute():
        # Defensivo: dataset declarou RELATIVE mas algum face tem path
        # absoluto. Honra o absoluto (provável legado de migração).
        return p

    base = Path(parquet_dir) if parquet_dir else Path.cwd()
    root = Path(path_root) if path_root else Path(".")
    return (base / root / p).resolve()


def _extract_path_str(face_or_path: Any) -> str:
    """Extrai a string de path de um FaceInfo/EmbeddingInfo/str/Path.

    Aceita qualquer um dos quatro pra que callers (services) não
    precisem desambiguar. **Normaliza para POSIX (L1c.1)** — protege
    contra paths gravados no Windows com `\\` lidos no macOS/Linux.
    """
    if hasattr(face_or_path, "image_path"):
        raw = str(face_or_path.image_path)
    else:
        raw = str(face_or_path)
    return normalize_stored_path(raw)
