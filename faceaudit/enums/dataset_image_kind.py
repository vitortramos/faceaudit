"""Que imagem o dataset guarda para cada face.

Três formas da MESMA face, com usos distintos e não intercambiáveis:

- `ORIGINAL` — a foto como veio. Carrega a cena inteira (fundo, outras
  pessoas) e é a única que preserva o registro íntegro da captura.
- `CROP` — recorte da face detectada com margem, **sem rotação**.
  Preserva geometria e proporção reais; corta o irrelevante. É a forma
  útil para ilustrar laudo e para comparação visual, onde o alinhamento
  atrapalha.
- `ALIGNED` — face 112×112 alinhada pelo ForensicFace. Aplica
  transformação de similaridade (rotaciona, escala e reamostra para levar
  os olhos a posições canônicas): é o que o encoder vê, e por isso a
  forma certa para degradar/estudar o que o modelo enxerga — e a errada
  para avaliar visualmente semelhança, porque deforma o rosto.

O vocabulário vem do importador de runs de vídeo, que já distinguia
`aligned` de `crop`; aqui ele vale para o app inteiro.

**`CROP` não é face alinhada.** Só `ALIGNED` autoriza
`FaceDataset.is_aligned=True`, que faz o `FFService` pular detect+align e
ir direto ao encoder. Declarar um recorte como alinhado produz embeddings
errados em silêncio.
"""

from __future__ import annotations

from enum import Enum


class DatasetImageKind(str, Enum):
    ORIGINAL = "original"
    CROP = "crop"
    ALIGNED = "aligned"

    @property
    def label_pt(self) -> str:
        return {
            "original": "Foto original",
            "crop": "Face recortada (com margem)",
            "aligned": "Face alinhada (112×112)",
        }[self.value]

    @property
    def description_pt(self) -> str:
        return {
            "original": (
                "A foto como veio, com a cena inteira. Preserva o registro "
                "íntegro da captura."
            ),
            "crop": (
                "Recorte da face com margem, sem rotação — preserva a "
                "geometria e as proporções reais. Indicado para ilustrar "
                "laudo e para comparação visual."
            ),
            "aligned": (
                "Face 112×112 alinhada, que é o que o modelo enxerga. "
                "Deforma o rosto (rotação, escala e reamostragem), então "
                "não serve para avaliar semelhança a olho nu."
            ),
        }[self.value]

    @property
    def implies_aligned_dataset(self) -> bool:
        """Só esta forma autoriza `FaceDataset.is_aligned=True`."""
        return self is DatasetImageKind.ALIGNED

    @property
    def requires_face_detection(self) -> bool:
        """Formas que precisam localizar a face antes de gravar."""
        return self is not DatasetImageKind.ORIGINAL

    @classmethod
    def from_str(cls, raw, default: "DatasetImageKind" = None) -> "DatasetImageKind":
        """Tolerante a valor ausente/desconhecido — devolve `default`."""
        if isinstance(raw, cls):
            return raw
        try:
            return cls(str(raw).strip().lower())
        except (ValueError, AttributeError):
            return default if default is not None else cls.ORIGINAL
