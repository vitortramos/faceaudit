"""Alinhamento percorrendo orientações, com a já conhecida primeiro.

Função livre, e não método de serviço, porque dois chamadores precisam dela
por caminhos diferentes — o `FFService` no laço de alinhar-e-embutir-em-lote,
e o pipeline no caminho "atributos-só" — e nenhum dos dois deve depender do
interior do outro. Quem chama fornece o motor de detecção; aqui só se decide
em que orientação a imagem é apresentada a ele.

Diferente do caminho de extração crua, aqui **não há conversão de
coordenadas a fazer**: `detect_and_align` devolve o recorte alinhado (de pé,
112×112) e os keypoints no sistema desse recorte, não no da imagem. A
rotação afeta apenas qual orientação foi alimentada ao detector.

A orientação já registrada na face é tentada PRIMEIRO, e esse é o ponto
principal: um dataset criado com a retentativa de giro tem faces que só
existem giradas. Enriquecê-lo tentando apenas a orientação gravada falharia
exatamente nelas — em silêncio, porque o laço de alinhamento simplesmente
pula quem não alinhou.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from faceaudit.services.image_rotation_service import (
    descrever_rotacao,
    girar_imagem,
    normalizar_graus,
)


def rotacao_registrada(face) -> int:
    """Orientação que já funcionou para esta face, ou 0.

    Lida do `ff_info` por `getattr` para tolerar face sem biometria e
    pickles antigos, em que o campo não existe.
    """
    return normalizar_graus(
        getattr(getattr(face, "ff_info", None), "detection_rotation", None)
    )


def alinhar_tentando_rotacoes(
    align_engine,
    img_array: np.ndarray,
    rotacoes: Sequence[int] = (),
    image_path: str = "",
    *,
    rotacao_conhecida: int = 0,
    error_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[dict], int]:
    """Chama `detect_and_align` orientação por orientação.

    Devolve `(item, graus)` — `graus` é a rotação horária que funcionou
    (0 = a imagem como está). `(None, 0)` quando nenhuma orientação alinhou.

    A ordem é: a conhecida, a imagem como está, e depois as demais. A
    conhecida entra mesmo com `rotacoes` vazio: ela não é uma busca, é um
    dado do registro da face.
    """
    tentativas: List[int] = []
    for graus in (rotacao_conhecida, 0, *(rotacoes or ())):
        g = normalizar_graus(graus)
        if g not in tentativas:
            tentativas.append(g)

    for graus in tentativas:
        imagem = girar_imagem(img_array, graus) if graus else img_array
        try:
            item = align_engine.detect_and_align(imagem, single_face=True)
        except Exception as e:
            # Falha numa orientação não condena as outras — o detector pode
            # estourar num shape específico e ir bem no girado.
            (error_callback or print)(
                f"detect_and_align falhou pra {image_path} "
                f"({descrever_rotacao(graus)}): {e}"
            )
            continue
        if item:
            if graus:
                print(
                    f"🔄 [Alinhamento] Alinhou só com a imagem "
                    f"{descrever_rotacao(graus)}: {image_path}"
                )
            return item, graus

    return None, 0
