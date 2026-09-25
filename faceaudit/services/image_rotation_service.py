"""Rotação em múltiplos de 90° e conversão de coordenadas entre os frames.

Existe por causa de um caso concreto: foto tirada com a câmera de lado
não tem face detectável na orientação em que está gravada, mas tem depois
de girar 90°. Tentar as quatro orientações recupera essas imagens.

O preço de girar é que **a detecção passa a viver noutro sistema de
coordenadas**. A caixa e os pontos devolvidos pelo detector estão no frame
girado, enquanto `image_path` continua apontando para o arquivo original.
Guardar a caixa girada como se fosse do original faz o recorte sair da
região errada e o desenho da caixa cair fora da face — em silêncio, porque
nada no dado denuncia a troca de frame.

Por isso o contrato do módulo: **a caixa persistida está SEMPRE em
coordenadas da imagem original**, e a rotação que permitiu a detecção é
gravada ao lado. Quem precisa do frame girado (recortar a face de pé, por
exemplo) pede a conversão de volta com `bbox_original_para_girada`.

Rotação em múltiplo de 90° é exata: a caixa alinhada aos eixos continua
alinhada aos eixos, sem interpolação e sem perda. Nada disso valeria para
um ângulo arbitrário.

Convenção única, usada em todo o módulo: `graus` é a rotação **horária**
aplicada à imagem original para produzir a imagem que foi ao detector.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

# Ordem das tentativas depois da imagem como está. Retrato girado para um
# lado ou para o outro é o caso comum (câmera de celular na horizontal);
# 180° é raro mas não custa nada e cobre foto escaneada de cabeça para baixo.
ROTACOES_PADRAO: Tuple[int, ...] = (90, 270, 180)

# Ângulos aceitos. Fora daqui a rotação deixaria de ser exata.
GRAUS_VALIDOS: Tuple[int, ...] = (0, 90, 180, 270)


class RotacaoInvalidaError(ValueError):
    """Ângulo que não é múltiplo de 90° no intervalo [0, 360)."""


def normalizar_graus(graus: Optional[int]) -> int:
    """Devolve o ângulo em {0, 90, 180, 270}.

    `None` e 360 viram 0 — os dois significam "não girei".
    """
    if graus is None:
        return 0
    try:
        valor = int(graus) % 360
    except (TypeError, ValueError) as exc:
        raise RotacaoInvalidaError(f"Rotação inválida: {graus!r}") from exc
    if valor not in GRAUS_VALIDOS:
        raise RotacaoInvalidaError(
            f"Rotação deve ser múltiplo de 90° (recebido {graus!r}). "
            "Ângulo arbitrário não preserva a caixa alinhada aos eixos."
        )
    return valor


def girar_imagem(imagem: np.ndarray, graus: int) -> np.ndarray:
    """Gira no sentido HORÁRIO. 0° devolve a própria imagem.

    `np.rot90` gira no sentido anti-horário, daí o `k` negativo.
    """
    graus = normalizar_graus(graus)
    if graus == 0:
        return imagem
    return np.ascontiguousarray(np.rot90(imagem, k=-(graus // 90)))


def forma_girada(
    forma_original: Tuple[int, int], graus: int,
) -> Tuple[int, int]:
    """(altura, largura) depois de girar. 90° e 270° trocam os dois."""
    altura, largura = int(forma_original[0]), int(forma_original[1])
    return (largura, altura) if normalizar_graus(graus) in (90, 270) else (altura, largura)


def ponto_girado_para_original(
    x: float, y: float, forma_girada_hw: Tuple[int, int], graus: int,
) -> Tuple[float, float]:
    """Ponto do frame GIRADO → frame ORIGINAL.

    `forma_girada_hw` é (altura, largura) da imagem **girada** — a que o
    detector recebeu, e portanto a única cujas dimensões quem chama tem em
    mãos sem refazer conta.
    """
    graus = normalizar_graus(graus)
    altura_g, largura_g = float(forma_girada_hw[0]), float(forma_girada_hw[1])
    if graus == 0:
        return x, y
    if graus == 90:
        # Horário 90°: (x, y)_orig → (H_orig - 1 - y, x). Invertendo, e
        # notando que H_orig == largura_g:
        return y, (largura_g - 1.0) - x
    if graus == 180:
        return (largura_g - 1.0) - x, (altura_g - 1.0) - y
    # 270° (equivale a 90° anti-horário)
    return (altura_g - 1.0) - y, x


def ponto_original_para_girado(
    x: float, y: float, forma_original_hw: Tuple[int, int], graus: int,
) -> Tuple[float, float]:
    """Ponto do frame ORIGINAL → frame GIRADO. Inversa da função acima."""
    graus = normalizar_graus(graus)
    altura_o, largura_o = float(forma_original_hw[0]), float(forma_original_hw[1])
    if graus == 0:
        return x, y
    if graus == 90:
        return (altura_o - 1.0) - y, x
    if graus == 180:
        return (largura_o - 1.0) - x, (altura_o - 1.0) - y
    # 270°
    return y, (largura_o - 1.0) - x


def _caixa_por_cantos(
    caixa: Sequence[float], converter,
) -> np.ndarray:
    """Converte os dois cantos e reordena para (x1, y1, x2, y2).

    A reordenação é o ponto delicado: girar troca qual canto é o de cima à
    esquerda, e uma caixa com x1 > x2 é lida como vazia por quem recorta.
    """
    x1, y1, x2, y2 = (float(v) for v in caixa[:4])
    ax, ay = converter(x1, y1)
    bx, by = converter(x2, y2)
    return np.array(
        [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)], dtype=np.float32,
    )


def caixa_girada_para_original(
    caixa: Sequence[float], forma_girada_hw: Tuple[int, int], graus: int,
) -> np.ndarray:
    """Caixa (x1, y1, x2, y2) do frame GIRADO → frame ORIGINAL."""
    return _caixa_por_cantos(
        caixa,
        lambda x, y: ponto_girado_para_original(x, y, forma_girada_hw, graus),
    )


def caixa_original_para_girada(
    caixa: Sequence[float], forma_original_hw: Tuple[int, int], graus: int,
) -> np.ndarray:
    """Caixa (x1, y1, x2, y2) do frame ORIGINAL → frame GIRADO."""
    return _caixa_por_cantos(
        caixa,
        lambda x, y: ponto_original_para_girado(x, y, forma_original_hw, graus),
    )


def pontos_girados_para_originais(
    pontos: Optional[np.ndarray], forma_girada_hw: Tuple[int, int], graus: int,
) -> Optional[np.ndarray]:
    """Landmarks `(N, 2)` do frame GIRADO → frame ORIGINAL.

    Diferente da caixa, aqui não há reordenação: cada ponto é independente
    e a ordem da lista (olho esquerdo, olho direito, nariz…) é semântica.
    """
    if pontos is None:
        return None
    arr = np.asarray(pontos, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return arr
    convertidos = np.array(
        [
            ponto_girado_para_original(float(p[0]), float(p[1]), forma_girada_hw, graus)
            for p in arr
        ],
        dtype=np.float32,
    )
    if arr.shape[1] == 2:
        return convertidos
    # Preserva colunas extras (algumas libs anexam confiança por ponto).
    saida = arr.copy()
    saida[:, :2] = convertidos
    return saida


def descrever_rotacao(graus: Optional[int]) -> str:
    """Texto curto para log, tooltip e cadeia de custódia."""
    graus = normalizar_graus(graus)
    return {
        0: "sem rotação",
        90: "girada 90° no sentido horário",
        180: "girada 180°",
        270: "girada 90° no sentido anti-horário",
    }[graus]
