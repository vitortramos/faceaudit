"""Diálogo de edição interativa do pipeline de processamento de
imagens (PR I2 — primeira fatia da UI do tema Imagens).

Estrutura:
  - `ImagePipelineDialogController` — entry point. Estado: pipeline
    config em RAM, índice da imagem corrente, step selecionado.
  - `ImagePipelineDialogView` — QDialog modal 1200×800 com QSplitter
    horizontal: lista esquerda + preview direita.
  - `StepsListWidget` — coluna esquerda. Lista drag-reorder + botão
    Adicionar (menu dos 5 step types) + form de config dinâmico.
  - `StepConfigPanel` — form dinâmico que renderiza widgets
    específicos por step_type.
  - `ImageCompareWidget` — preview lado-a-lado (original × processada)
    + setas ◀▶ + 🎲 + botão Aplicar.

I2 NÃO inclui: scores de qualidade ao vivo (I3), persistência (I4),
batch processing (I5).
"""

from .image_pipeline_dialog_controller import ImagePipelineDialogController

__all__ = ["ImagePipelineDialogController"]
