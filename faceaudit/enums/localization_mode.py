"""Modos de operação do `DatasetLocalizationService` (PR L1c).

Define qual transformação será aplicada ao dataset:
  - REWRITE_RELATIVE: não copia; reescreve `image_path` como relativo
    a `target_dir`. Requer que todas as imagens já estejam dentro de
    `target_dir`.
  - COPY_ORIGINALS: copia imagens pra `target_dir/images/`; paths
    viram relativos a essa subpasta. Caminho mais portátil.
  - COPY_ALIGNED: detecta + alinha + salva PNG (requer FFService).
  - REBASE_PREFIX: substitui um prefixo de path por outro (sem cópia).
    Caso de uso: dataset criado em outra máquina com paths absolutos
    do tipo `/home/colega/...` ou `C:\\Users\\...`, e a mesma estrutura
    de pasta existe nesta máquina em outro caminho.
"""

from __future__ import annotations

from enum import Enum


class LocalizationMode(str, Enum):
    REWRITE_RELATIVE = "rewrite_relative"
    COPY_ORIGINALS = "copy_originals"
    COPY_ALIGNED = "copy_aligned"  # PR L3 — exige FFService injetado no service
    REBASE_PREFIX = "rebase_prefix"
