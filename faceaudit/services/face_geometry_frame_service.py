"""Conversão da geometria da face entre quadros de imagem.

`FFInfo.bbox` e `FFInfo.keypoints` estão, por invariante, no sistema de
coordenadas da imagem que o dataset guarda. Quando um fluxo troca essa
imagem — e o recorte com margem troca, porque grava um PNG novo e reescreve
`image_path` —, a geometria tem de ir junto, senão a caixa e os cinco
pontos passam a apontar para fora da imagem gravada e quem for marcar a
face desenha no vazio.

A conversão do recorte é uma translação: subtrair a origem do recorte. É
exata, diferente do caso alinhado, onde a transformação de similaridade
fica dentro do `detect_and_align` e não está disponível a quem materializa
— lá a geometria é descartada em vez de traduzida errado.

Recorte com margem nunca gira a imagem, então não há ângulo a compor aqui.
Rotação em múltiplos de 90° é assunto do `image_rotation_service`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def transladar_geometria(
    ff_info,
    origem_x: float,
    origem_y: float,
    largura: Optional[int] = None,
    altura: Optional[int] = None,
) -> None:
    """Move `bbox` e `keypoints` do quadro antigo para o do recorte.

    Muta o `ff_info` no lugar, que é o que os fluxos de materialização
    esperam. `largura`/`altura` do recorte, quando informadas, servem para
    prender a caixa dentro dele: a margem do recorte é cortada nas bordas
    da foto, então a caixa transladada pode passar do limite por alguns
    pixels.

    Os pontos NÃO são presos ao retângulo. Um ponto fora do recorte é
    informação verdadeira — olho fora do quadro em face muito de perfil, por
    exemplo — e prendê-lo à borda inventaria uma posição que o detector não
    viu.
    """
    if ff_info is None:
        return

    caixa = getattr(ff_info, "bbox", None)
    if caixa is not None:
        try:
            nova = np.asarray(caixa, dtype=np.float32).reshape(-1).copy()
            if nova.size >= 4:
                nova[0] -= float(origem_x)
                nova[1] -= float(origem_y)
                nova[2] -= float(origem_x)
                nova[3] -= float(origem_y)
                if largura is not None and altura is not None:
                    nova[0] = float(np.clip(nova[0], 0, largura))
                    nova[2] = float(np.clip(nova[2], 0, largura))
                    nova[1] = float(np.clip(nova[1], 0, altura))
                    nova[3] = float(np.clip(nova[3], 0, altura))
                ff_info.bbox = nova[:4]
        except (TypeError, ValueError):
            pass

    pontos = getattr(ff_info, "keypoints", None)
    if pontos is not None:
        try:
            novos = np.asarray(pontos, dtype=np.float32).copy()
            if novos.ndim == 2 and novos.shape[1] == 2:
                novos[:, 0] -= float(origem_x)
                novos[:, 1] -= float(origem_y)
                ff_info.keypoints = novos
        except (TypeError, ValueError):
            pass


def descartar_geometria(ff_info) -> None:
    """Apaga `bbox` e `keypoints`.

    Para quando a imagem do dataset passou a ser outra e a conversão não
    está disponível — o recorte alinhado é o caso. Geometria ausente é
    resposta correta; geometria na régua errada não é, e não há nada no
    dado que denuncie a troca.
    """
    if ff_info is None:
        return
    ff_info.bbox = None
    ff_info.keypoints = None
