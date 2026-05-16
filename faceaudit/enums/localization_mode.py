"""Modos de operação do `DatasetLocalizationService` (PR L1c).

Define qual transformação será aplicada ao dataset:
  - REWRITE_RELATIVE: não copia; reescreve `image_path` como relativo
    a `target_dir`. Requer que todas as imagens já estejam dentro de
    `target_dir`.
  - COPY_ORIGINALS: copia imagens pra `target_dir/images/`; paths
    viram relativos a essa subpasta. Caminho mais portátil.
  - REBASE_PREFIX: substitui um prefixo de path por outro (sem cópia).
    Caso de uso: dataset criado em outra máquina com paths absolutos
    do tipo `/home/colega/...` ou `C:\\Users\\...`, e a mesma estrutura
    de pasta existe nesta máquina em outro caminho.

Histórico: COPY_ALIGNED foi removido — não era path management,
é transformação de conteúdo. Pra alinhar faces, use o Editor de
Pipeline de Imagens com `Origem = Face alinhada (112×112)` e
zero steps.
"""

from __future__ import annotations

from enum import Enum


class LocalizationMode(str, Enum):
    REWRITE_RELATIVE = "rewrite_relative"
    COPY_ORIGINALS = "copy_originals"
    REBASE_PREFIX = "rebase_prefix"

    @property
    def label_pt(self) -> str:
        return {
            "rewrite_relative": "Reescrever como relativo (sem copiar)",
            "copy_originals":   "Copiar originais (portátil)",
            "rebase_prefix":    "Substituir prefixo de path",
        }.get(self.value, self.value)
