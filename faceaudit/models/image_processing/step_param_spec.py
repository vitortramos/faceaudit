"""Declaração dos parâmetros de um passo do pipeline de imagens.

Cada config de passo diz QUAIS parâmetros tem, de que tipo, em que faixa,
com que ajuda e quando aparecem. O painel do editor monta o formulário a
partir disso, sem código por tipo de passo. Antes cada passo tinha um
formulário escrito à mão no widget — acrescentar um passo exigia mexer na
interface, e o formulário podia divergir do config sem ninguém notar (o
campo "Padding face" do Recorte editava um atributo que o processador já
não lia).

Módulo puro, sem Qt: a interface consome; o backend declara.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple, Union


class StepParamKind(str, Enum):
    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"
    BOOL = "bool"
    # Botão que pede à interface uma ação que não é um valor simples
    # (ex.: abrir o canvas da máscara de oclusão).
    ACTION = "action"


Choices = Union[
    Sequence[Tuple[Any, str]],
    Callable[[Any], Sequence[Tuple[Any, str]]],
]


@dataclass(frozen=True)
class StepParamSpec:
    """Um campo do formulário de um passo.

    Attributes:
        name: atributo do config que o campo edita (ou id da ação).
        label: rótulo em português.
        kind: tipo do campo.
        help: texto de ajuda (tooltip); aceita HTML.
        minimum / maximum / step / decimals / suffix: para INT e FLOAT.
        choices: pares (valor, rótulo) para CHOICE — ou função do config,
            quando as opções dependem de outro campo.
        visible: função do config; o campo só aparece quando devolve True.
            Reavaliada a cada mudança de qualquer campo.
        action_label: função do config que dá o texto do botão (ACTION).
    """

    name: str
    label: str
    kind: StepParamKind
    help: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    decimals: int = 2
    suffix: str = ""
    choices: Optional[Choices] = None
    visible: Optional[Callable[[Any], bool]] = None
    action_label: Optional[Callable[[Any], str]] = None

    def is_visible(self, config: Any) -> bool:
        return True if self.visible is None else bool(self.visible(config))

    def choices_for(self, config: Any) -> Sequence[Tuple[Any, str]]:
        if self.choices is None:
            return ()
        return self.choices(config) if callable(self.choices) else self.choices

    def clamp(self, value: Any) -> Any:
        """Prende o valor à faixa declarada (INT/FLOAT)."""
        if self.kind not in (StepParamKind.INT, StepParamKind.FLOAT):
            return value
        v = float(value)
        if self.minimum is not None:
            v = max(float(self.minimum), v)
        if self.maximum is not None:
            v = min(float(self.maximum), v)
        return int(round(v)) if self.kind == StepParamKind.INT else v


def enum_choices(enum_values: Sequence[Any]) -> Tuple[Tuple[Any, str], ...]:
    """Pares (membro, `label_pt`) de uma lista de membros de enum."""
    return tuple((m, getattr(m, "label_pt", str(m))) for m in enum_values)
