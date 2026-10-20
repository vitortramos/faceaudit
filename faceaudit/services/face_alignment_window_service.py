"""Janela de alinhamento: que pedaço da imagem a face 112×112 usa.

O alinhamento estima, a partir dos 5 pontos, uma transformação de
semelhança para o gabarito do ArcFace e reamostra a imagem por ela
(`forensicface.insightface.face_align.norm_crop`). O quadrado 112×112 de
saída corresponde, na imagem de origem, a um quadrado GIRADO — e é isso
que este módulo calcula, para dois usos:

1. Na extração de vídeo, o recorte com margem passa a cobrir sempre a
   janela inteira. Sem isso, em cabeça inclinada a janela escapava do
   retângulo do recorte, e refazer o alinhamento a partir do recorte punha
   preto onde a extração tinha pixel (medido num caso real: 232 de 1024
   faces; pior cosseno 0,788 com o embedding da extração).
2. No enriquecimento de um dataset de recortes, realinha-se a partir dos
   pontos gravados em vez de detectar a face de novo, e mede-se quanto da
   janela ficou sem pixel — o que diz se o realinhamento reproduz o
   alinhamento original.

A matriz vem da MESMA função que o `norm_crop` usa; outra estimativa
(`cv2.estimateAffinePartial2D`, por exemplo) daria uma janela ligeiramente
diferente da que o alinhamento de fato amostra.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from forensicface.insightface import face_align

TAMANHO_ALINHADA = 112

# Folga em pixels em torno da janela: a interpolação bilinear lê o vizinho
# seguinte de cada ponto amostrado, e o arredondamento das coordenadas
# inteiras do recorte come mais meio pixel.
FOLGA_PADRAO = 2

Retangulo = Tuple[float, float, float, float]

# Coluna (em `FaceInfo.extra_data`) com a `fracao_sem_pixel` de cada face.
ALIGNMENT_MISSING_PIXELS_COLUMN = "alignment_missing_pixel_fraction"

# `geometry_source` cujos pontos são os da detecção que gerou o embedding
# original — os únicos de que se pode realinhar sem trocar de face. Pontos de
# detecção posterior (`backfill_detector`) não entram: num quadro com várias
# pessoas a detecção nova pode ter escolhido outra face.
ORIGENS_DE_GEOMETRIA_REALINHAVEIS = frozenset({"video_extraction_detection"})


def _pontos(keypoints) -> np.ndarray:
    pts = np.asarray(keypoints, dtype=np.float64).reshape(-1)
    if pts.size != 10 or not np.all(np.isfinite(pts)):
        raise ValueError("esperados 5 pontos (x, y) finitos")
    return pts.reshape(5, 2)


def matriz_de_alinhamento(keypoints) -> np.ndarray:
    """Matriz 2×3 imagem → face 112×112, a mesma do `norm_crop`."""
    return np.asarray(face_align.estimate_norm(_pontos(keypoints), TAMANHO_ALINHADA))


def _origem_das_amostras(keypoints) -> np.ndarray:
    """Coordenada na imagem de cada pixel da face 112×112, `(112·112, 2)`."""
    inversa = cv2.invertAffineTransform(matriz_de_alinhamento(keypoints))
    yy, xx = np.mgrid[0:TAMANHO_ALINHADA, 0:TAMANHO_ALINHADA]
    destino = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    return destino @ inversa[:, :2].T + inversa[:, 2]


def janela_de_alinhamento(keypoints) -> Retangulo:
    """Retângulo `(x1, y1, x2, y2)` que contém a janela, na imagem de origem.

    Transformação afim leva quadrado em paralelogramo; os extremos estão
    nos quatro cantos, então basta transformá-los.
    """
    inversa = cv2.invertAffineTransform(matriz_de_alinhamento(keypoints))
    lado = TAMANHO_ALINHADA - 1
    cantos = np.array([[0, 0], [lado, 0], [0, lado], [lado, lado]], dtype=np.float64)
    origem = cantos @ inversa[:, :2].T + inversa[:, 2]
    return (float(origem[:, 0].min()), float(origem[:, 1].min()),
            float(origem[:, 0].max()), float(origem[:, 1].max()))


def caixa_do_recorte(
    caixa_com_margem: Sequence[float],
    keypoints,
    largura_quadro: int,
    altura_quadro: int,
    folga: int = FOLGA_PADRAO,
) -> Tuple[int, int, int, int]:
    """Recorte que cobre a margem pedida E a janela de alinhamento.

    O que for maior em cada lado, preso às bordas do quadro. Com isso,
    realinhar a partir do recorte nunca precisa de pixel que o recorte não
    tem — a não ser fora do quadro, onde a extração também não tinha.
    """
    x1, y1, x2, y2 = janela_de_alinhamento(keypoints)
    jx1 = int(np.floor(x1)) - folga
    jy1 = int(np.floor(y1)) - folga
    jx2 = int(np.ceil(x2)) + 1 + folga   # exclusivo, como no fatiamento
    jy2 = int(np.ceil(y2)) + 1 + folga
    mx1, my1, mx2, my2 = (int(v) for v in caixa_com_margem)
    return (
        max(min(mx1, jx1), 0),
        max(min(my1, jy1), 0),
        min(max(mx2, jx2), int(largura_quadro)),
        min(max(my2, jy2), int(altura_quadro)),
    )


def fracao_sem_pixel(
    keypoints,
    largura: int,
    altura: int,
    existente: Optional[Retangulo] = None,
) -> float:
    """Fração da face 112×112 que, realinhada a partir desta imagem, sairia
    sem o pixel que o alinhamento original teve.

    `keypoints`, `largura` e `altura` são da imagem que será realinhada (o
    recorte). `existente` é o retângulo, no MESMO sistema, onde a imagem
    original (o quadro) tinha pixel: amostra fora dele também não tinha
    pixel na extração e não conta. Sem `existente`, conta toda amostra fora
    da imagem — um limite superior, porque parte dela pode estar além da
    borda do quadro.

    0 significa que o realinhamento reproduz o alinhamento original,
    exceto pelo arredondamento da interpolação.
    """
    origem = _origem_das_amostras(keypoints)
    x, y = origem[:, 0], origem[:, 1]
    fora_da_imagem = (x < 0) | (y < 0) | (x > largura - 1) | (y > altura - 1)
    if existente is not None:
        ex1, ey1, ex2, ey2 = existente
        dentro_do_original = (x >= ex1) & (y >= ey1) & (x <= ex2 - 1) & (y <= ey2 - 1)
        fora_da_imagem &= dentro_do_original
    return float(np.mean(fora_da_imagem))


def realinhar(imagem_bgr: np.ndarray, keypoints) -> Tuple[np.ndarray, np.ndarray]:
    """Face 112×112 (BGR) e os 5 pontos no sistema dela, sem detecção."""
    pts = _pontos(keypoints)
    matriz = matriz_de_alinhamento(pts)
    alinhada = cv2.warpAffine(
        imagem_bgr, matriz, (TAMANHO_ALINHADA, TAMANHO_ALINHADA), borderValue=0.0,
    )
    pontos_alinhados = pts @ matriz[:, :2].T + matriz[:, 2]
    return alinhada, pontos_alinhados.astype(np.float32)
