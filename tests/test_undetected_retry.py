"""Nova tentativa de detecção em resolução maior, pedida na revisão.

O `det_size` continua 320 na ingestão — rápido. A escalada acontece na
janela de revisão, por decisão de quem está olhando a imagem: foto de
paisagem sem ninguém não merece a tentativa, foto em que claramente há um
rosto merece.

O que os testes travam:

- o FFService é construído UMA vez por resolução. Ele compila os modelos
  ONNX para a resolução pedida (~5–10 s); reconstruir por imagem tornaria o
  botão inutilizável;
- a resolução que recuperou a face fica gravada, e só quando difere da
  resolução da ingestão — proveniência, igual à rotação aplicada;
- a face recuperada perde a marca de erro: ela passou a ter biometria;
- cancelar no meio não transforma as não tentadas em falhas — elas não
  foram avaliadas, e afirmar o contrário seria dizer o que não se sabe;
- em dataset de recortes, o bloqueio de inclusão LEVANTA para a face
  recuperada: agora existe caixa, logo existe recorte a gerar;
- as recuperadas passam pelos estágios de qualidade/recorte antes da
  gravação, senão o dataset fica heterogêneo por dentro.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.services.undetected_retry_service import (
    RESOLUCOES_DE_RETENTATIVA,
    UndetectedRetryService,
)

SLOT = SepaelModel.default().ffinfo_slot


def _face(nome: str) -> FaceInfo:
    f = FaceInfo(image_path=f"C:/fotos/{nome}")
    f.error = True
    f.error_stage = "ff"
    f.error_message = "[ff] nenhuma face detectada"
    return f


class _ServicoFalso:
    """Detecta a partir de uma resolução mínima; abaixo dela, nada.

    Reproduz o comportamento real: face pequena no quadro não tem pixels
    suficientes em resolução baixa e aparece quando ela sobe.
    """

    def __init__(self, det_size: int, minimo_para_ver: int = 640):
        self.det_size = det_size
        self._minimo = minimo_para_ver
        self.chamadas: list = []

    def process_face(self, face, **kwargs):
        self.chamadas.append(
            (getattr(face, "image_path", ""), kwargs.get("rotation_attempts")),
        )
        if self.det_size >= self._minimo:
            face.ff_info = FFInfo(**{SLOT: np.ones(4, dtype=np.float32)})
        return face


def _servico(minimo=640):
    construidos: list = []

    def fabrica(det_size: int):
        construidos.append(det_size)
        return _ServicoFalso(det_size, minimo_para_ver=minimo)

    return UndetectedRetryService(fabrica), construidos


# ── Cache do serviço ─────────────────────────────────────────────────────

def test_servico_construido_uma_vez_por_resolucao():
    """Compilar os modelos custa ~5–10 s; por imagem seria inutilizável."""
    svc, construidos = _servico()

    svc.retry([_face("a.jpg"), _face("b.jpg")], det_size=640, baseline_det_size=320)
    svc.retry([_face("c.jpg")], det_size=640, baseline_det_size=320)

    assert construidos == [640], "uma construção, três imagens"


def test_resolucoes_diferentes_constroem_servicos_diferentes():
    svc, construidos = _servico()

    svc.retry([_face("a.jpg")], det_size=640, baseline_det_size=320)
    svc.retry([_face("b.jpg")], det_size=960, baseline_det_size=320)

    assert construidos == [640, 960]


def test_sabe_dizer_se_a_resolucao_ja_esta_carregada():
    """A interface usa isso para avisar que haverá espera."""
    svc, _ = _servico()

    assert svc.has_service_for(640) is False
    svc.service_for(640)
    assert svc.has_service_for(640) is True


def test_liberar_solta_os_modelos():
    svc, construidos = _servico()
    svc.service_for(640)
    svc.release()
    svc.service_for(640)

    assert construidos == [640, 640], "depois de liberar, reconstrói"


# ── Recuperação ──────────────────────────────────────────────────────────

def test_recupera_em_resolucao_maior():
    svc, _ = _servico(minimo=640)
    faces = [_face("a.jpg"), _face("b.jpg")]

    resultado = svc.retry(faces, det_size=640, baseline_det_size=320)

    assert len(resultado.recovered) == 2
    assert resultado.still_failing == []
    assert resultado.det_size == 640


def test_continua_falhando_quando_a_resolucao_nao_basta():
    svc, _ = _servico(minimo=1280)
    faces = [_face("a.jpg")]

    resultado = svc.retry(faces, det_size=640, baseline_det_size=320)

    assert resultado.recovered == []
    assert len(resultado.still_failing) == 1


def test_grava_a_resolucao_que_funcionou():
    """Se a face só apareceu a 640, isso é proveniência."""
    svc, _ = _servico(minimo=640)
    face = _face("a.jpg")

    svc.retry([face], det_size=640, baseline_det_size=320)

    assert face.ff_info.detection_det_size == 640


def test_nao_grava_quando_a_resolucao_e_a_mesma_da_ingestao():
    """Coluna preenchida em todo dataset seria ruído; o valor informa
    justamente por ser diferente do configurado."""
    svc, _ = _servico(minimo=320)
    face = _face("a.jpg")

    svc.retry([face], det_size=320, baseline_det_size=320)

    assert face.ff_info.detection_det_size is None


def test_recuperada_perde_a_marca_de_erro():
    """Ela passou a ter biometria — manter a marca diria o contrário."""
    svc, _ = _servico(minimo=640)
    face = _face("a.jpg")
    assert face.error is True

    svc.retry([face], det_size=640, baseline_det_size=320)

    assert face.error is False
    assert face.error_stage is None
    assert face.error_message is None


def test_a_que_continua_falhando_mantem_a_marca():
    svc, _ = _servico(minimo=1280)
    face = _face("a.jpg")

    svc.retry([face], det_size=640, baseline_det_size=320)

    assert face.error is True


def test_as_orientacoes_sao_refeitas_na_resolucao_nova():
    """Girar sozinho não serviria de nada — a imagem já falhou nas quatro
    orientações a 320. O que muda é a resolução; as orientações voltam a ser
    percorridas dentro dela."""
    from faceaudit.services.image_rotation_service import ROTACOES_PADRAO

    svc, _ = _servico()
    svc.retry([_face("a.jpg")], det_size=640, baseline_det_size=320)

    servico_interno = svc.service_for(640)
    _caminho, rotacoes = servico_interno.chamadas[0]
    assert tuple(rotacoes) == tuple(ROTACOES_PADRAO)


# ── Cancelamento ─────────────────────────────────────────────────────────

def test_cancelar_nao_condena_as_nao_tentadas():
    """Elas não foram avaliadas; chamá-las de falha afirmaria o que não se
    sabe. Continuam pendentes."""
    svc, _ = _servico(minimo=640)
    faces = [_face(f"{i}.jpg") for i in range(5)]
    tentadas = {"n": 0}

    def _cancelar_na_segunda():
        tentadas["n"] += 1
        return tentadas["n"] > 2

    resultado = svc.retry(
        faces, det_size=640, baseline_det_size=320,
        should_cancel=_cancelar_na_segunda,
    )

    assert resultado.cancelled is True
    assert len(resultado.recovered) + len(resultado.still_failing) == 5, (
        "nenhuma face pode desaparecer da contabilidade"
    )


def test_progresso_reporta_o_total():
    svc, _ = _servico()
    marcas: list = []

    svc.retry(
        [_face("a.jpg"), _face("b.jpg")], det_size=640, baseline_det_size=320,
        progress_callback=lambda i, t, c: marcas.append((i, t)),
    )

    assert marcas[0] == (0, 2)
    assert marcas[-1] == (2, 2), "o último tique fecha a barra"


# ── Resoluções oferecidas ────────────────────────────────────────────────

def test_resolucoes_oferecidas_sao_maiores_que_o_padrao():
    assert all(r > 320 for r in RESOLUCOES_DE_RETENTATIVA)
    assert 320 not in RESOLUCOES_DE_RETENTATIVA, (
        "320 já foi tentada quando a imagem chega à revisão"
    )


def test_resolucoes_estao_em_ordem_crescente():
    """A janela mostra a lista como está; fora de ordem confundiria."""
    assert list(RESOLUCOES_DE_RETENTATIVA) == sorted(RESOLUCOES_DE_RETENTATIVA)
