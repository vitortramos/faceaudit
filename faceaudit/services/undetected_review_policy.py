"""Quando uma imagem sem face detectada PODE entrar no dataset.

A pergunta não é de interface: depende do que o dataset guarda. Num dataset
de fotos originais, incluir a imagem mesmo sem detecção é decisão legítima
do perito — a foto continua sendo a foto, só não rendeu biometria.

Num dataset de faces recortadas ou alinhadas, não é. Ali cada arquivo é um
recorte produzido a partir da caixa da face, e sem detecção não há caixa,
logo não há recorte a gerar. O que entraria é a foto inteira, e o estrago é
silencioso e tardio: o dataset fica marcado como alinhado, o enriquecimento
seguinte confia nessa marca e manda a foto inteira direto ao encoder como
se fosse uma face 112×112 — sai embedding, sem erro nenhum, e ele não
representa face alguma.

Por isso a regra mora aqui e não no widget: é uma propriedade do dataset
sendo criado, e vale para qualquer interface que venha a fazer a pergunta.
"""

from __future__ import annotations

from dataclasses import dataclass

from faceaudit.enums.dataset_image_kind import DatasetImageKind


@dataclass(frozen=True)
class UndetectedInclusionPolicy:
    """Pode incluir? Se não, `reason` é o texto a mostrar a quem decide.

    `reason` vem preenchido também no caso permitido — serve de legenda
    explicando o que a inclusão significa, que é informação que o perito
    precisa ter antes de marcar.
    """

    allowed: bool
    reason: str

    @property
    def blocked(self) -> bool:
        return not self.allowed


_MOTIVO_ORIGINAIS = (
    "A imagem entra no dataset apontando para a foto original, sem "
    "embedding e marcada com erro de detecção. Ela não participa de "
    "comparação biométrica — fica no dataset como registro."
)

_MOTIVO_ALINHADAS = (
    "Este dataset guarda faces alinhadas (112×112). Sem face detectada não "
    "há alinhamento a gerar, e incluir a foto inteira aqui faria com que o "
    "próximo enriquecimento a tratasse como face já alinhada — produzindo "
    "embedding que não representa face alguma, sem nenhum erro aparente."
)

_MOTIVO_RECORTES = (
    "Este dataset guarda recortes da face. Sem face detectada não há caixa "
    "e portanto não há recorte a gerar; o que entraria é a foto inteira, "
    "quebrando a premissa de que todo arquivo do dataset é um recorte."
)

_MOTIVO_FONTE_ALINHADA = (
    "Este dataset foi declarado como contendo faces já alinhadas. A imagem "
    "que falhou não é uma face alinhada legível, e incluí-la contaminaria o "
    "dataset com um arquivo que não é do tipo declarado."
)


def resolve_inclusion_policy(config) -> UndetectedInclusionPolicy:
    """Decide a partir do `DatasetIngestionConfig` da criação.

    Recebe o config por duck-typing (via `getattr`) para não obrigar quem
    chama a montar o DTO inteiro — a política depende de quatro campos.
    """
    if getattr(config, "is_aligned", False):
        return UndetectedInclusionPolicy(False, _MOTIVO_FONTE_ALINHADA)

    if getattr(config, "align_during_ingestion", False):
        return UndetectedInclusionPolicy(False, _MOTIVO_ALINHADAS)

    kind = getattr(config, "image_kind", DatasetImageKind.ORIGINAL)
    if getattr(config, "self_contained", False) and kind is not DatasetImageKind.ORIGINAL:
        motivo = (
            _MOTIVO_ALINHADAS if kind is DatasetImageKind.ALIGNED else _MOTIVO_RECORTES
        )
        return UndetectedInclusionPolicy(False, motivo)

    return UndetectedInclusionPolicy(True, _MOTIVO_ORIGINAIS)
