# faceaudit/services/image_io.py
"""I/O de imagem seguro para caminhos não-ASCII no Windows.

`cv2.imwrite`/`cv2.imread` falham **silenciosamente** em caminhos com
caracteres não-ASCII no Windows (a API C usa `fopen` com a codepage
ANSI; caminhos com acento, como `Perícias/...`, quebram sem erro).
O workaround canônico — já usado em
`FeatureExtractionPipeline._materialize_aligned_crops` — é codificar em
memória (`cv2.imencode`) e gravar/ler os bytes via `pathlib`, que usa a
API Unicode do Windows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imwrite_unicode_safe(
    path: Union[str, Path],
    bgr_image: np.ndarray,
    *,
    default_ext: str = ".png",
) -> bool:
    """Grava `bgr_image` (BGR, como o OpenCV espera) em `path` de forma
    robusta a caminhos não-ASCII no Windows.

    Usa a extensão do próprio `path` quando presente; senão `default_ext`.
    Cria os diretórios pais. Retorna True em sucesso, False em falha
    (imencode falhou ou gravação no disco falhou) — **não** levanta, para
    permitir que o caller registre a falha por face e siga.
    """
    if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
        return False
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or default_ext
    success, buf = cv2.imencode(ext, bgr_image)
    if not success:
        return False
    try:
        p.write_bytes(buf.tobytes())
        return True
    except OSError:
        return False


def imread_unicode_safe(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:
    """Lê uma imagem de `path` (robusto a não-ASCII no Windows) via
    `Path.read_bytes` + `cv2.imdecode`. Retorna o array BGR ou None."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    try:
        data = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img if img is not None and img.size > 0 else None
