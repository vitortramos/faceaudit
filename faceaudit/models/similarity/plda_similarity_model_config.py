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
            "Variante matemática do PLDA: SIMPLE (SpeechBrain, com subspace F) "
            "ou TWO_COVARIANCE (Brümmer 2010, covariâncias completas)."
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
            "Termo de regularização ε ≥ 0 somado a B e W como ε·I antes "
            "da inversão no scoring. Usado apenas em plda_variant=TWO_COVARIANCE; "
            "estabiliza numericamente quando D é alto comparado ao número de "
            "classes ou amostras. Padrão 0 = sem regularização."
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