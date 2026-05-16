"""Modo de armazenamento de paths de imagem em FaceDataset (PR L1a).

Define a semântica do campo `FaceDataset.path_mode`. Ver também
`faceaudit/services/image_path_resolver.py` para a free-function que
faz a resolução de paths relativos.
"""

from __future__ import annotations

from enum import Enum


class PathMode(str, Enum):
    """Modo de path do `FaceInfo.image_path` em um dataset.

    - `ABSOLUTE`: caminhos absolutos (compatibilidade com tudo gerado
      antes do PR L1a). `from_parquet` carrega e usa direto;
      `resolve_image_path` retorna o path como veio.
    - `RELATIVE`: caminhos relativos a `(parquet_dir / path_root)`.
      Habilita portabilidade — mover o parquet leva as imagens junto
      sem precisar reescrever paths. `path_root` na metadata aponta
      pra subpasta dentro do diretório do parquet (ex: `"images"`,
      `"."` quando flat).

    Persistência: serializado como `faceaudit:path_mode` na metadata
    do schema do parquet (slot já reservado desde PR 3).
    """
    ABSOLUTE = "absolute"
    RELATIVE = "relative"

    @property
    def label_pt(self) -> str:
        return {
            "absolute": "Absoluto",
            "relative": "Relativo (portátil)",
        }.get(self.value, self.value)
