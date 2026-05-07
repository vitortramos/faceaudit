"""Modos de operação do `DatasetLocalizationService` (PR L1c).

Define qual transformação será aplicada ao dataset:
  - REWRITE_RELATIVE: não copia; reescreve `image_path` como relativo
    a `target_dir`. Requer que todas as imagens já estejam dentro de
    `target_dir`.
  - COPY_ORIGINALS: copia imagens pra `target_dir/images/`; paths
    viram relativos a essa subpasta. Caminho mais portátil.

Modo COPY_ALIGNED entra em L4 (depende de aligned_face em RAM ou Enrich
prévio).
"""

from __future__ import annotations

from enum import Enum


class LocalizationMode(str, Enum):
    REWRITE_RELATIVE = "rewrite_relative"
    COPY_ORIGINALS = "copy_originals"
    COPY_ALIGNED = "copy_aligned"  # PR L3 — exige FFService injetado no service
