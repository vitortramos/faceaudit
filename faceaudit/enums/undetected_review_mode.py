"""De onde a revisão de imagens sem face detectada foi chamada.

A distinção não é cosmética: ela muda a pergunta que a janela faz.

Na **criação**, a imagem ainda não está em dataset nenhum. Incluí-la ou
descartá-la é escolha legítima, e descartar não destrói nada — a foto
continua na pasta de origem.

No **enriquecimento**, a face JÁ está no dataset; ela apenas não ganhou a
biometria pedida. Não há o que incluir. E descartar significaria remover uma
face de um artefato existente, que é edição destrutiva e tem caminho próprio
no aplicativo (Excluir selecionados, com justificativa na cadeia de
custódia). Oferecer isso aqui duplicaria a operação com garantia mais fraca.
"""

from enum import Enum


class UndetectedReviewMode(str, Enum):
    """Contexto da revisão."""

    CREATION = "creation"
    ENRICHMENT = "enrichment"

    @property
    def permite_decidir_inclusao(self) -> bool:
        """Só na criação a inclusão é uma decisão a tomar."""
        return self is UndetectedReviewMode.CREATION

    @property
    def label_pt(self) -> str:
        return {
            UndetectedReviewMode.CREATION: "criação de dataset",
            UndetectedReviewMode.ENRICHMENT: "enriquecimento",
        }[self]
