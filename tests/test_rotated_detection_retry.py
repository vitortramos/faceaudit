"""Retentativa de detecção com a imagem girada.

Motivo: foto tirada com a câmera de lado não tem face detectável na
orientação em que foi gravada, e tem depois de girar 90°. Antes disso, a
imagem era classificada como "sem face" quando o problema era a orientação.

Os testes usam um detector falso que só "vê" face numa orientação
escolhida, então exercitam o laço de tentativas e a conversão de
coordenadas sem carregar modelo nenhum.

O que travam:

- a imagem como está é sempre a PRIMEIRA tentativa — o custo extra só é
  pago por quem falhou;
- a caixa gravada fica no sistema de coordenadas do ARQUIVO, não do frame
  girado. Guardá-la girada faz o recorte sair da região errada em silêncio;
- todos os modelos da mesma chamada usam a MESMA orientação, senão os
  embeddings deixam de ser comparáveis entre si;
- sem a lista de rotações, o comportamento é o antigo: uma única tentativa.
"""

from __future__ import annotations

import numpy as np
import pytest

from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.services.ff_service import FFService
from faceaudit.services.image_rotation_service import (
    ROTACOES_PADRAO,
    caixa_original_para_girada,
    girar_imagem,
)

ALTURA, LARGURA = 40, 90          # retrato gravado de lado
CAIXA_NO_FRAME_DE_PE = (10.0, 4.0, 30.0, 24.0)


# Conteúdo assimétrico: as quatro orientações são imagens DIFERENTES.
# Um marcador simétrico (ou a mera forma) confundiria 90° com 270° e 0°
# com 180°, e o teste passaria sem exercitar nada.
FONTE = (
    np.arange(ALTURA * LARGURA * 3, dtype=np.int64).reshape(ALTURA, LARGURA, 3) % 251
).astype(np.uint8)


class _EngineFalso:
    """Só devolve face quando a imagem chega na orientação `graus_que_ve`.

    Reconhece a orientação pelo CONTEÚDO (compara com a fonte girada), não
    pela forma. A caixa devolvida é sempre no frame que ele recebeu — é
    assim que um detector de verdade se comporta, e é justamente o que
    obriga a conversão de volta.
    """

    def __init__(self, graus_que_ve: int, etiqueta: str = "falso"):
        self.graus_que_ve = graus_que_ve
        self.etiqueta = etiqueta
        self.formas_recebidas: list = []

    def process_image(self, img):
        self.formas_recebidas.append(img.shape[:2])
        if not np.array_equal(img, girar_imagem(FONTE, self.graus_que_ve)):
            return {}
        caixa = caixa_original_para_girada(
            CAIXA_NO_FRAME_DE_PE, (ALTURA, LARGURA), self.graus_que_ve,
        )
        return {
            "bbox": np.asarray(caixa, dtype=np.float32),
            "keypoints": np.array([[caixa[0], caixa[1]], [caixa[2], caixa[3]]],
                                  dtype=np.float32),
            "embedding": np.ones(4, dtype=np.float32),
            "det_score": 0.9,
            "aligned_face": np.zeros((112, 112, 3), dtype=np.uint8),
        }


def _servico(engines: dict) -> FFService:
    """FFService sem carregar modelo — só o suficiente para process_face."""
    svc = object.__new__(FFService)
    svc._default_model = next(iter(engines))
    svc._engines = dict(engines)
    svc._resolve_engine = lambda m: (
        (svc._default_model, engines[svc._default_model])
        if m is None else (m, engines[m])
    )
    svc._safe_read_image = lambda _p: FONTE.copy()
    return svc


def _face() -> FaceInfo:
    return FaceInfo(image_path="C:/fotos/de_lado.jpg")


# ── Tentativas ───────────────────────────────────────────────────────────

def test_sem_rotacoes_e_uma_unica_tentativa():
    """Comportamento antigo preservado quando a lista vem vazia."""
    engine = _EngineFalso(graus_que_ve=90)
    svc = _servico({"m": engine})

    resultado = svc.process_face(_face(), model="m")

    assert len(engine.formas_recebidas) == 1
    assert resultado.ff_info is None, "sem retentativa, a face fica sem detecção"


def test_imagem_como_esta_e_a_primeira_tentativa():
    """Quem detecta de primeira não paga nada a mais."""
    engine = _EngineFalso(graus_que_ve=0)
    svc = _servico({"m": engine})

    svc.process_face(_face(), model="m", rotation_attempts=ROTACOES_PADRAO)

    assert engine.formas_recebidas == [(ALTURA, LARGURA)], (
        "detectou de primeira: não pode ter girado nada"
    )


@pytest.mark.parametrize("graus", [90, 270, 180])
def test_encontra_girando(graus):
    engine = _EngineFalso(graus_que_ve=graus)
    svc = _servico({"m": engine})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info is not None, f"devia achar a face com {graus}°"
    assert resultado.ff_info.detection_rotation == graus


def test_rotacao_fica_registrada_para_leitura_humana():
    """Sem esse registro, pose e recorte vêm de um frame que ninguém sabe
    qual é — e a foto na tela aparece deitada, contradizendo o dado."""
    svc = _servico({"m": _EngineFalso(graus_que_ve=270)})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info.detection_rotation == 270


def test_detectou_de_primeira_nao_marca_rotacao():
    svc = _servico({"m": _EngineFalso(graus_que_ve=0)})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info.detection_rotation is None, (
        "None é o caso normal; 0 na coluna seria ruído em todo dataset"
    )


def test_nenhuma_orientacao_serve_deixa_sem_deteccao():
    """Aqui é falha de verdade — é esta face que vai para a revisão."""
    class _EngineCego(_EngineFalso):
        def process_image(self, img):   # não vê face em orientação alguma
            self.formas_recebidas.append(img.shape[:2])
            return {}

    engine = _EngineCego(graus_que_ve=0)
    svc = _servico({"m": engine})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info is None
    assert len(engine.formas_recebidas) == 1 + len(ROTACOES_PADRAO)


def test_marca_de_rotacao_antiga_e_limpa_em_nova_passada():
    """Run anterior girou; este detecta de primeira. A marca velha não pode
    sobreviver ao merge e mentir sobre esta extração."""
    face = _face()
    face.ff_info = FFInfo(detection_rotation=90)
    svc = _servico({"m": _EngineFalso(graus_que_ve=0)})

    resultado = svc.process_face(
        face, model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info.detection_rotation is None


def test_falha_numa_orientacao_nao_condena_as_outras():
    """O detector pode estourar num shape e ir bem no girado."""

    class _EngineQueEstoura(_EngineFalso):
        def process_image(self, img):
            if np.array_equal(img, FONTE):     # falha só na orientação gravada
                raise RuntimeError("shape não suportado")
            return super().process_image(img)

    svc = _servico({"m": _EngineQueEstoura(graus_que_ve=90)})
    erros: list = []

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
        error_callback=erros.append,
    )

    assert resultado.ff_info is not None
    assert resultado.ff_info.detection_rotation == 90
    assert erros, "a falha da primeira orientação tem de aparecer no log"


# ── Coordenadas ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("graus", [90, 270, 180])
def test_caixa_gravada_esta_no_frame_do_arquivo(graus):
    """A caixa tem de casar com o arquivo que `image_path` aponta.

    Guardá-la no frame girado faz o recorte sair da região errada e o
    desenho cair fora da face — sem nada no dado denunciando a troca.
    """
    svc = _servico({"m": _EngineFalso(graus_que_ve=graus)})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    obtida = tuple(round(float(v)) for v in resultado.ff_info.bbox)
    assert obtida == tuple(round(v) for v in CAIXA_NO_FRAME_DE_PE)


@pytest.mark.parametrize("graus", [90, 270, 180])
def test_landmarks_tambem_voltam_ao_frame_do_arquivo(graus):
    svc = _servico({"m": _EngineFalso(graus_que_ve=graus)})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    pontos = resultado.ff_info.keypoints
    esperados = np.array(
        [[CAIXA_NO_FRAME_DE_PE[0], CAIXA_NO_FRAME_DE_PE[1]],
         [CAIXA_NO_FRAME_DE_PE[2], CAIXA_NO_FRAME_DE_PE[3]]],
        dtype=np.float32,
    )
    assert np.allclose(np.sort(pontos, axis=0), np.sort(esperados, axis=0), atol=1.0)


def test_recorte_alinhado_fica_como_veio():
    """`aligned_face` é o recorte de pé — é o que o encoder deve ver, e
    girá-lo de volta seria destruir justamente o ganho da retentativa."""
    svc = _servico({"m": _EngineFalso(graus_que_ve=90)})

    resultado = svc.process_face(
        _face(), model="m", rotation_attempts=ROTACOES_PADRAO,
    )

    assert resultado.ff_info.aligned_face.shape == (112, 112, 3)


# ── Vários modelos ───────────────────────────────────────────────────────

def test_todos_os_modelos_usam_a_mesma_orientacao():
    """Embeddings de modelos diferentes têm de sair do mesmo recorte.

    Se cada modelo procurasse sua própria orientação, dois embeddings da
    mesma face poderiam vir de recortes distintos — e aí não são
    comparáveis entre si nem com o resto do dataset.
    """
    a = _EngineFalso(graus_que_ve=90, etiqueta="a")
    b = _EngineFalso(graus_que_ve=90, etiqueta="b")
    svc = _servico({"a": a, "b": b})

    svc.process_face(_face(), models=["a", "b"], rotation_attempts=ROTACOES_PADRAO)

    forma_de_pe = girar_imagem(FONTE, 90).shape[:2]
    assert b.formas_recebidas == [forma_de_pe], (
        "o segundo modelo recebe a imagem já na orientação decidida, "
        "sem repetir a busca"
    )


def test_segundo_modelo_nao_repete_a_busca():
    """A busca custa uma passada de detector por tentativa; repeti-la por
    modelo multiplicaria o custo sem mudar a resposta."""
    a = _EngineFalso(graus_que_ve=180)
    b = _EngineFalso(graus_que_ve=180)
    svc = _servico({"a": a, "b": b})

    svc.process_face(_face(), models=["a", "b"], rotation_attempts=ROTACOES_PADRAO)

    # Ordem das tentativas: como está, 90°, 270°, 180° — a face está a 180°,
    # então o primeiro modelo paga as quatro passadas e o segundo, uma só.
    assert len(a.formas_recebidas) == 1 + len(ROTACOES_PADRAO)
    assert len(b.formas_recebidas) == 1
