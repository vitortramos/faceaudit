"""Nova tentativa de detecção em resolução maior, sob comando humano.

O detector recebe a imagem reduzida a `det_size` pixels no lado maior.
A 320 px — o padrão, escolhido por velocidade — uma face pequena no quadro
pode simplesmente não ter pixels suficientes para ser encontrada. A 640 ela
aparece. O custo é real: quatro vezes a área, e as quatro orientações são
refeitas, então não é algo a ligar para toda a pasta às cegas.

Daí a forma: a escalada acontece **na revisão**, por decisão de quem está
olhando a imagem. Uma foto de paisagem sem ninguém não merece a tentativa;
uma foto em que claramente há um rosto merece. Escalada automática não
distingue as duas.

Trocar `det_size` exige **outro FFService** — os modelos ONNX são compilados
para a resolução pedida. São ~5–10 s de carga por resolução, uma vez: o
serviço fica em cache aqui para que a segunda imagem em diante seja rápida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from faceaudit.services.image_rotation_service import ROTACOES_PADRAO

# Resoluções oferecidas na revisão. 320 é o padrão da ingestão e já foi
# tentada quando a imagem chega à janela, então não entra na lista.
RESOLUCOES_DE_RETENTATIVA: tuple = (640, 960, 1280)


@dataclass(frozen=True)
class RetryOutcome:
    """Resultado de uma rodada de retentativa."""

    recovered: List = field(default_factory=list)
    still_failing: List = field(default_factory=list)
    det_size: int = 0
    cancelled: bool = False

    @property
    def n_tentadas(self) -> int:
        return len(self.recovered) + len(self.still_failing)


class UndetectedRetryService:
    """Repete a detecção em outra resolução e diz o que foi recuperado.

    Recebe uma fábrica de `FFService` por injeção: o serviço não sabe (nem
    deve saber) como construir um, e nos testes a fábrica devolve um duplo.
    """

    def __init__(self, ff_service_factory: Callable[[int], object]) -> None:
        self._factory = ff_service_factory
        self._cache: Dict[int, object] = {}

    def service_for(self, det_size: int):
        """FFService na resolução pedida, construído uma única vez."""
        chave = int(det_size)
        if chave not in self._cache:
            self._cache[chave] = self._factory(chave)
        return self._cache[chave]

    def has_service_for(self, det_size: int) -> bool:
        """True quando a resolução já está carregada (não custa espera)."""
        return int(det_size) in self._cache

    def release(self) -> None:
        """Solta os serviços em cache. Chamar ao fechar a revisão."""
        self._cache.clear()

    def retry(
        self,
        faces: Sequence,
        *,
        det_size: int,
        baseline_det_size: int,
        models: Optional[List[str]] = None,
        dataset=None,
        rotations: Sequence[int] = ROTACOES_PADRAO,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        error_callback: Optional[Callable[[str], None]] = None,
    ) -> RetryOutcome:
        """Tenta detectar cada face na resolução pedida.

        Reaproveita `FFService.process_face`, que já sabe percorrer as
        orientações — o que muda é a resolução do serviço por trás.

        `baseline_det_size` é a resolução da ingestão. Quando a face é
        recuperada em resolução diferente dela, isso fica gravado em
        `ff_info.detection_det_size`: se a face só apareceu a 640, essa é
        informação de proveniência, do mesmo tipo que a rotação aplicada.
        """
        servico = self.service_for(det_size)
        recuperadas: List = []
        falharam: List = []
        cancelada = False
        total = len(faces)

        for indice, face in enumerate(faces):
            if should_cancel is not None and should_cancel():
                cancelada = True
                # As não tentadas continuam pendentes — não foram avaliadas,
                # e tratá-las como falha seria afirmar o que não se sabe.
                falharam.extend(faces[indice:])
                break

            if progress_callback is not None:
                progress_callback(indice, total, str(getattr(face, "image_path", "")))

            servico.process_face(
                face,
                models=models,
                dataset=dataset,
                error_callback=error_callback,
                rotation_attempts=rotations,
            )

            if _tem_embedding(face):
                if int(det_size) != int(baseline_det_size):
                    face.ff_info.detection_det_size = int(det_size)
                # A marca de erro sai: esta face passou a ter biometria.
                face.error = False
                face.error_stage = None
                face.error_message = None
                recuperadas.append(face)
            else:
                falharam.append(face)

        if progress_callback is not None:
            progress_callback(total, total, "")

        return RetryOutcome(
            recovered=recuperadas,
            still_failing=falharam,
            det_size=int(det_size),
            cancelled=cancelada,
        )


def _tem_embedding(face) -> bool:
    """Mesma pergunta do pipeline: existe vetor em algum slot de modelo?

    `ff_info is not None` não serve — uma passada pode preencher pose e
    falhar no encoder, deixando um objeto sem vetor nenhum.
    """
    from faceaudit.enums.sepael_model import SepaelModel

    ff = getattr(face, "ff_info", None)
    if ff is None:
        return False
    return any(
        getattr(ff, member.ffinfo_slot, None) is not None for member in SepaelModel
    )
