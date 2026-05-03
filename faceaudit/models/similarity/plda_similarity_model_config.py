# faceaudit/models/similarity/plda_similarity_model_config.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.enums.plda_variant import PLDAVariant


T = TypeVar("T", bound="PLDASimilarityModelConfig")


@dataclass
class PLDASimilarityModelConfig(JSONSerializableMixin):
    """
    DTO de configuração do modelo de similaridade PLDA na nova arquitetura.

    Esta configuração representa apenas parâmetros do modelo de similaridade,
    sem misturar:
    - calibração
    - pipeline de transformação
    - estratégias antigas de treino
    - decisões de execução da strategy engine

    Regras semânticas:
    - plda_variant=SIMPLE (padrão): formulação SpeechBrain, usa em_iterations,
      plda_mode, latent_dim. FULL_RANK usa o maior rank permitido pelos dados;
      LOW_RANK usa latent_dim como limite do espaço latente.
    - plda_variant=TWO_COVARIANCE: formulação Brümmer 2010, sem EM (forma
      fechada). Os campos em_iterations / plda_mode / latent_dim são
      IGNORADOS — covariâncias B e W são sempre full-rank D×D. Usa
      `regularization` opcional para estabilizar W em alta dimensionalidade.
    """

    plda_variant: PLDAVariant = field(
        default=PLDAVariant.SIMPLE,
        metadata={"help": (
            "<html><b>Variante matemática do PLDA.</b><br/><br/>"
            "<b>SIMPLE</b> — formulação Ioffe simplificada via SpeechBrain. "
            "Trabalha num <i>subspace</i> entre-classes de baixo rank "
            "(<code>F ∈ R^{D × latent_dim}</code>) treinado por EM iterativo. "
            "Direções fora do subspace são truncadas, o que regulariza "
            "implicitamente mas perde informação. Usa "
            "<code>em_iterations</code>, <code>plda_mode</code>, "
            "<code>latent_dim</code>.<br/><br/>"
            "<b>TWO_COVARIANCE</b> — formulação Brümmer &amp; de Villiers "
            "(2010), padrão na literatura forense via FoCal toolkit. "
            "Estima covariâncias <b>B</b> (entre-classes) e <b>W</b> "
            "(intra-classe) completas <code>D × D</code> em forma fechada "
            "(sem EM, via scatter matrices). Sem perda de informação por "
            "truncamento — só use se tiver dados suficientes para estimar "
            "matrizes cheias. Aceita <code>regularization</code> para "
            "estabilizar numericamente.</html>"
        )}
    )

    em_iterations: int = field(
        default=20,
        metadata={"help": (
            "Número de iterações do algoritmo EM no treino do PLDA. "
            "Usado apenas em plda_variant=SIMPLE."
        )}
    )

    plda_mode: PLDAMode = field(
        default=PLDAMode.LOW_RANK,
        metadata={"help": (
            "Modo de dimensionalidade latente do PLDA. Usado apenas em "
            "plda_variant=SIMPLE; ignorado em TWO_COVARIANCE."
        )}
    )

    latent_dim: int = field(
        default=128,
        metadata={"help": (
            "Dimensão latente usada quando plda_mode=LOW_RANK. Ignorada "
            "em plda_variant=TWO_COVARIANCE."
        )}
    )

    regularization: float = field(
        default=0.0,
        metadata={"help": (
            "<html><b>Regularização Tikhonov (ε ≥ 0).</b> "
            "Soma <code>ε · I</code> em B e W antes das inversões: "
            "<code>K=inv(W+B)</code>, <code>M=inv(W+2B)</code>, "
            "<code>N=inv(W)</code>. Empurra um valor positivo na diagonal, "
            "garantindo que todas as direções tenham variância mínima "
            "<code>ε</code>. Usado apenas em "
            "<b>plda_variant=TWO_COVARIANCE</b>.<br/><br/>"
            "<b>Quando aumentar:</b><ul>"
            "<li>D &gt; N efetivo (ex: 512D com poucas identidades × poucas amostras)</li>"
            "<li>Embeddings colineares (após L2 + transformações fortes)</li>"
            "<li>Poucas classes com ≥ 2 amostras (muitos singletons descartados)</li>"
            "<li>Pipeline reduziu dimensionalidade abaixo do n.º de classes</li>"
            "<li>Erro &quot;W não é positiva definida&quot; no treino</li></ul>"
            "<b>Valores típicos:</b><ul>"
            "<li><code>0.0</code> (default) — só use com dataset farto "
            "(~1000+ identidades × 10+ amostras, D=128–256)</li>"
            "<li><code>1e-6</code> a <code>1e-4</code> — &quot;leve&quot;, "
            "evita instabilidade numérica sem alterar log-LRs. Default seguro.</li>"
            "<li><code>1e-3</code> a <code>1e-2</code> — amacia decisão "
            "(log-LRs menos extremos). Pode mascarar problemas reais do "
            "pipeline.</li>"
            "<li><code>&gt; 1e-1</code> — evite. A identidade domina a "
            "covariância real.</li></ul>"
            "<b>Antes de subir:</b> verifique duplicatas exatas no dataset, "
            "passos de pipeline excessivos, dimensões coerentes com nº de "
            "classes (LDA com 30 classes não deve sair em 256D).</html>"
        )}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente aleatória para reprodutibilidade, quando aplicável."}
    )

    def __post_init__(self) -> None:
        if self.em_iterations <= 0:
            raise ValueError("em_iterations deve ser maior que zero.")

        if self.latent_dim <= 0:
            raise ValueError("latent_dim deve ser maior que zero.")

        if self.regularization < 0:
            raise ValueError("regularization deve ser ≥ 0.")

    @property
    def uses_full_rank(self) -> bool:
        return self.plda_mode == PLDAMode.FULL_RANK

    @property
    def uses_low_rank(self) -> bool:
        return self.plda_mode == PLDAMode.LOW_RANK

    @property
    def uses_simple_variant(self) -> bool:
        return self.plda_variant == PLDAVariant.SIMPLE

    @property
    def uses_two_covariance_variant(self) -> bool:
        return self.plda_variant == PLDAVariant.TWO_COVARIANCE

    # ── Visibilidade de campos na UI ─────────────────────────────────

    # Campo cuja mudança deve disparar rebuild do nested form (porque
    # afeta a visibilidade de outros fields). Lido pelo nested_dto_widget.
    REACTIVE_FIELDS = ("plda_variant",)

    def get_hidden_fields(self):
        """
        Retorna o conjunto de nomes de campos que devem ser OCULTADOS pela
        UI dado o estado atual do config. Usado pelo `nested_dto_widget`
        para esconder dinamicamente campos que não se aplicam à variante
        selecionada.

        - SIMPLE: oculta `regularization` (só faz sentido em TWO_COVARIANCE).
        - TWO_COVARIANCE: oculta `em_iterations`, `plda_mode`, `latent_dim`
          (formulação Brümmer não usa EM nem subspace de baixo rank).

        Returns:
            Set[str] com nomes de campos a esconder.
        """
        if self.uses_simple_variant:
            return {"regularization"}
        return {"em_iterations", "plda_mode", "latent_dim"}

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, convertendo corretamente os enums PLDAMode e
        PLDAVariant. Retrocompat: configs antigos sem `plda_variant` defaultam
        para SIMPLE — preserva comportamento de estratégias já persistidas.
        """
        if not data:
            return cls()

        processed_data = data.copy()

        if "plda_mode" in processed_data and processed_data["plda_mode"]:
            processed_data["plda_mode"] = PLDAMode.from_str(
                processed_data["plda_mode"]
            )

        if "plda_variant" in processed_data and processed_data["plda_variant"]:
            processed_data["plda_variant"] = PLDAVariant.from_str(
                processed_data["plda_variant"]
            )

        return cls(**processed_data)