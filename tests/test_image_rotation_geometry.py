"""Geometria da rotação: a conta é conferida contra os pixels de verdade.

A álgebra da conversão entre frames é fácil de errar por um sinal ou por
um off-by-one, e o erro é silencioso: a caixa continua sendo uma caixa
plausível, só aponta para o lugar errado da foto. Então nenhum teste aqui
confia na fórmula — todos giram uma imagem com `np.rot90`, localizam o
pixel marcado e exigem que a conversão caia exatamente nele.

Imagem com altura diferente da largura de propósito: com H == W, trocar
altura por largura na fórmula não muda nada e o erro passa.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.services.image_rotation_service import (
    GRAUS_VALIDOS,
    ROTACOES_PADRAO,
    RotacaoInvalidaError,
    caixa_girada_para_original,
    caixa_original_para_girada,
    descrever_rotacao,
    forma_girada,
    girar_imagem,
    normalizar_graus,
    ponto_girado_para_original,
    ponto_original_para_girado,
    pontos_girados_para_originais,
)

ALTURA, LARGURA = 7, 11  # propositalmente diferentes
MARCA = 255


def _imagem_com_marca(linha: int, coluna: int) -> np.ndarray:
    img = np.zeros((ALTURA, LARGURA), dtype=np.uint8)
    img[linha, coluna] = MARCA
    return img


def _onde_esta_a_marca(img: np.ndarray) -> tuple:
    linhas, colunas = np.where(img == MARCA)
    assert len(linhas) == 1, "a marca tem de sobreviver à rotação, uma só vez"
    return int(linhas[0]), int(colunas[0])


POSICOES = [(0, 0), (0, LARGURA - 1), (ALTURA - 1, 0), (ALTURA - 1, LARGURA - 1), (3, 5)]


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
@pytest.mark.parametrize("linha,coluna", POSICOES)
def test_girado_para_original_cai_no_pixel_certo(graus, linha, coluna):
    original = _imagem_com_marca(linha, coluna)
    girada = girar_imagem(original, graus)
    linha_g, coluna_g = _onde_esta_a_marca(girada)

    x, y = ponto_girado_para_original(coluna_g, linha_g, girada.shape[:2], graus)

    assert (round(x), round(y)) == (coluna, linha)


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
@pytest.mark.parametrize("linha,coluna", POSICOES)
def test_original_para_girado_cai_no_pixel_certo(graus, linha, coluna):
    original = _imagem_com_marca(linha, coluna)
    girada = girar_imagem(original, graus)
    linha_g, coluna_g = _onde_esta_a_marca(girada)

    x, y = ponto_original_para_girado(coluna, linha, original.shape[:2], graus)

    assert (round(x), round(y)) == (coluna_g, linha_g)


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_ida_e_volta_do_ponto(graus):
    original = _imagem_com_marca(2, 8)
    girada = girar_imagem(original, graus)
    xg, yg = ponto_original_para_girado(8, 2, original.shape[:2], graus)
    x, y = ponto_girado_para_original(xg, yg, girada.shape[:2], graus)

    assert (round(x), round(y)) == (8, 2)


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_forma_girada_bate_com_a_imagem(graus):
    original = np.zeros((ALTURA, LARGURA), dtype=np.uint8)
    girada = girar_imagem(original, graus)

    assert forma_girada((ALTURA, LARGURA), graus) == girada.shape[:2]


# ── Caixa ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_caixa_convertida_cobre_a_regiao_girada(graus):
    """A caixa convertida tem de cobrir exatamente os pixels da região."""
    original = np.zeros((ALTURA, LARGURA), dtype=np.uint8)
    original[1:4, 2:9] = MARCA          # y de 1 a 3, x de 2 a 8
    caixa_original = (2, 1, 8, 3)       # x1, y1, x2, y2 (cantos inclusivos)

    girada = girar_imagem(original, graus)
    linhas, colunas = np.where(girada == MARCA)
    esperada = (colunas.min(), linhas.min(), colunas.max(), linhas.max())

    obtida = caixa_original_para_girada(caixa_original, original.shape[:2], graus)

    assert tuple(round(float(v)) for v in obtida) == esperada


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_caixa_volta_ao_original(graus):
    original = np.zeros((ALTURA, LARGURA), dtype=np.uint8)
    caixa_original = (2, 1, 8, 3)
    girada_forma = forma_girada((ALTURA, LARGURA), graus)

    no_frame_girado = caixa_original_para_girada(
        caixa_original, (ALTURA, LARGURA), graus,
    )
    de_volta = caixa_girada_para_original(no_frame_girado, girada_forma, graus)

    assert tuple(round(float(v)) for v in de_volta) == caixa_original


@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_caixa_sai_ordenada(graus):
    """x1 <= x2 e y1 <= y2 sempre.

    Girar troca qual canto é o de cima à esquerda; caixa invertida é lida
    como vazia por quem recorta, e o recorte sai em branco sem erro algum.
    """
    obtida = caixa_original_para_girada((2, 1, 8, 3), (ALTURA, LARGURA), graus)

    assert obtida[0] <= obtida[2] and obtida[1] <= obtida[3]


def test_caixa_nunca_fica_negativa_nos_cantos():
    """Canto na origem não pode virar coordenada negativa."""
    for graus in GRAUS_VALIDOS:
        obtida = caixa_original_para_girada((0, 0, 4, 2), (ALTURA, LARGURA), graus)
        assert (obtida >= 0).all(), f"{graus}° produziu coordenada negativa: {obtida}"


# ── Landmarks ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("graus", GRAUS_VALIDOS)
def test_landmarks_convertidos_um_a_um(graus):
    pontos = np.array([[2.0, 1.0], [8.0, 3.0], [5.0, 2.0]], dtype=np.float32)
    girada_forma = forma_girada((ALTURA, LARGURA), graus)
    no_girado = np.array(
        [ponto_original_para_girado(p[0], p[1], (ALTURA, LARGURA), graus) for p in pontos],
        dtype=np.float32,
    )

    de_volta = pontos_girados_para_originais(no_girado, girada_forma, graus)

    assert np.allclose(de_volta, pontos, atol=1e-4)


def test_landmarks_preservam_a_ordem():
    """A ordem da lista é semântica (olho esquerdo, direito, nariz…)."""
    pontos = np.array([[0.0, 0.0], [10.0, 6.0]], dtype=np.float32)
    convertidos = pontos_girados_para_originais(pontos, (ALTURA, LARGURA), 90)

    assert convertidos.shape == pontos.shape
    # O primeiro continua sendo o primeiro — não foi reordenado.
    assert not np.allclose(convertidos[0], convertidos[1])


def test_landmarks_preservam_colunas_extras():
    """Algumas libs anexam confiança por ponto; não podemos descartá-la."""
    pontos = np.array([[2.0, 1.0, 0.9], [8.0, 3.0, 0.7]], dtype=np.float32)
    convertidos = pontos_girados_para_originais(pontos, (ALTURA, LARGURA), 90)

    assert convertidos.shape == (2, 3)
    assert np.allclose(convertidos[:, 2], [0.9, 0.7])


def test_landmarks_ausentes_passam_como_none():
    assert pontos_girados_para_originais(None, (ALTURA, LARGURA), 90) is None


# ── Contrato do ângulo ───────────────────────────────────────────────────

def test_zero_graus_nao_copia_nem_altera():
    original = _imagem_com_marca(2, 3)
    assert girar_imagem(original, 0) is original


def test_none_e_360_sao_sem_rotacao():
    assert normalizar_graus(None) == 0
    assert normalizar_graus(360) == 0
    assert normalizar_graus(-90) == 270


def test_angulo_arbitrario_e_recusado():
    """45° não preserva a caixa alinhada aos eixos — recusar é o certo."""
    with pytest.raises(RotacaoInvalidaError):
        normalizar_graus(45)
    with pytest.raises(RotacaoInvalidaError):
        girar_imagem(np.zeros((4, 4), dtype=np.uint8), 45)


def test_rotacoes_padrao_nao_incluem_zero():
    """0° é a primeira tentativa, feita fora da lista de retentativas."""
    assert 0 not in ROTACOES_PADRAO
    assert set(ROTACOES_PADRAO) <= set(GRAUS_VALIDOS)


def test_descricao_distingue_os_dois_lados():
    """No log e na custódia, 90° e 270° não podem ler igual."""
    assert descrever_rotacao(90) != descrever_rotacao(270)
    assert "horário" in descrever_rotacao(90)
    assert "anti-horário" in descrever_rotacao(270)
    assert descrever_rotacao(None) == descrever_rotacao(0)


def test_imagem_colorida_tambem_gira():
    colorida = np.zeros((ALTURA, LARGURA, 3), dtype=np.uint8)
    colorida[1, 2] = (10, 20, 30)
    girada = girar_imagem(colorida, 90)

    assert girada.shape == (LARGURA, ALTURA, 3)
    x, y = ponto_original_para_girado(2, 1, (ALTURA, LARGURA), 90)
    assert tuple(int(v) for v in girada[round(y), round(x)]) == (10, 20, 30)
