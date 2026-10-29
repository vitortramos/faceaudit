"""Base abstrata pros configs de step (PR I0).

Cada step concreto (BlurConfig, JpegConfig, ...) herda dessa base
e adiciona seus próprios campos. O `step_type` discrimina o tipo
durante deserialização polimórfica via
`ImageProcessorFactory.config_from_dict`.

Além da serialização, cada config declara para o editor:
  - `param_specs()` — os campos do formulário (ver `step_param_spec`);
  - `editor_default()` — o config de um passo recém-adicionado;
  - `summary_pt()` — resumo curto exibido na lista de passos;
  - `requires_face` — se o passo precisa saber onde está a face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.step_param_spec import StepParamSpec


@dataclass
class ImageProcessingStepConfig:
    """Base — não usar diretamente. Concrete configs herdam e
    sobrescrevem `step_type` com o valor correto."""

    step_type: ImageProcessingStepType

    def to_dict(self) -> Dict[str, Any]:
        """Serializa pra dict JSON-safe.

        Concrete configs devem chamar `super().to_dict()` e adicionar
        seus campos específicos.
        """
        return {"step_type": self.step_type.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageProcessingStepConfig":
        """Não chamar direto na base — usar
        `ImageProcessorFactory.config_from_dict(data)` que despacha
        polimorficamente baseado em `data["step_type"]`."""
        raise NotImplementedError(
            "Use ImageProcessorFactory.config_from_dict(data) — não "
            "chame ImageProcessingStepConfig.from_dict diretamente."
        )

    # ── Editor ─────────────────────────────────────────────────────────────

    def param_specs(self) -> List[StepParamSpec]:
        """Campos do formulário. Passo sem parâmetros devolve []."""
        return []

    @classmethod
    def editor_default(cls) -> "ImageProcessingStepConfig":
        """Config de um passo recém-adicionado no editor. Pode diferir
        do default de esquema para que o efeito apareça de imediato."""
        return cls()

    def summary_pt(self) -> str:
        """Resumo curto dos parâmetros, para a lista de passos."""
        return ""

    @property
    def requires_face(self) -> bool:
        """True quando o passo precisa da geometria da face."""
        return False

    def start_from_questioned(self, measures: Any) -> bool:
        """Leva os campos ajustáveis a um ponto de partida calculado das
        medidas da questionada (`DegradationMeasures`), para o ajuste
        automático. Devolve True se mudou algo. Padrão: nada a fazer."""
        return False

    def normalize(self) -> None:
        """Reaplica as correções de faixa após edição de um campo."""
        post = getattr(self, "__post_init__", None)
        if callable(post):
            post()
