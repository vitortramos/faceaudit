from __future__ import annotations
from enum import Enum
from typing import Optional

class PLDATrainingStrategy(Enum):
    """
    Define a estratégia de treinamento e adaptação de domínio para o modelo PLDA.
    
    Centraliza as regras de negócio sobre:
    1. Fluxo de Dados (Data Flow): Como os dados alimentam os algoritmos matemáticos.
    2. Validação de Pipeline: Restrições sobre a composição dos pipelines Source/Target.
    3. Validação de Dados: Requisitos sobre a qualidade dos datasets (ex: labels).
    """

    STANDARD = (
        "standard",
        "Sem Adaptação de Domínio",
        "Modo padrão (In-Domain). Assume que Source e Target pertencem ao mesmo domínio. "
        "Não realiza adaptação específica; aplica a mesma transformação em ambos."
    )

    UNSUPERVISED_ADAPTATION = (
        "unsupervised_adaptation",
        "Adaptação Não-Supervisionada",
        "Adaptação Não-Supervisionada. Usa alinhamento global (CORAL) para mover o Source "
        "até o Target. Não requer identidades conhecidas no Target."
    )

    SUPERVISED_ADAPTATION = (
        "supervised_adaptation",
        "Adaptação Supervisionada",
        "Adaptação Supervisionada. Usa labels confiáveis do Target para refinar "
        "a separação de classes (LDA) no novo domínio."
    )

    def __new__(cls, value: str, label: str, description: str) -> PLDATrainingStrategy:
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o valor identificador para o JSON (ex: 'standard')."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> PLDATrainingStrategy:
        """Reconstrói o Enum a partir da string com fallback seguro."""
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.STANDARD

    # =========================================================================
    # 1. REGRAS DE NEGÓCIO (DATA FLOW)
    # =========================================================================

    @property
    def use_target_for_coral(self) -> bool:
        """True se o Target deve ser usado para calcular a matriz de covariância do CORAL."""
        return self is not PLDATrainingStrategy.STANDARD

    @property
    def use_target_for_whitening(self) -> bool:
        """True se o Target deve entrar no cálculo do PCA (espaço comum)."""
        return self is not PLDATrainingStrategy.STANDARD

    @property
    def use_target_for_lda(self) -> bool:
        """True se os labels do Target devem ser usados no cálculo do LDA."""
        return self is PLDATrainingStrategy.SUPERVISED_ADAPTATION

    @property
    def use_target_for_plda(self) -> bool:
        """True se os dados do Target entram no treinamento do classificador PLDA."""
        return self is PLDATrainingStrategy.SUPERVISED_ADAPTATION

    # =========================================================================
    # 2. REGRAS DE VALIDAÇÃO DE PIPELINE (UI e Validador)
    # =========================================================================

    @property
    def requires_source_adaptation(self) -> bool:
        """O Pipeline Source DEVE conter uma etapa de Adaptação (ex: CORAL)."""
        return self is not PLDATrainingStrategy.STANDARD

    @property
    def allows_source_adaptation(self) -> bool:
        """Falso se o Pipeline Source NÃO PODE ter CORAL (Standard)."""
        return self is not PLDATrainingStrategy.STANDARD

    @property
    def allows_target_adaptation(self) -> bool:
        """O Target é a 'Âncora', não deve sofrer distorção global."""
        return False
    
    @property
    def requires_symmetric_pipelines(self) -> bool:
        """Source e Target devem ter exatamente a mesma estrutura."""
        return self is PLDATrainingStrategy.STANDARD

    @property
    def allows_target_supervised_steps(self) -> bool:
        """Verifica se o pipeline do Target pode conter passos supervisionados (ex: LDA)."""
        return self is PLDATrainingStrategy.SUPERVISED_ADAPTATION

    # =========================================================================
    # 3. REGRAS DE VALIDAÇÃO DE DADOS (Requirements)
    # =========================================================================

    @property
    def requires_target_labels(self) -> bool:
        """O Dataset Target precisa ter a coluna de identidade (label) válida."""
        return self is PLDATrainingStrategy.SUPERVISED_ADAPTATION
    
    # =========================================================================
    # 4. REGRAS DE NOMENCLATURA (I/O e Artefatos)
    # =========================================================================

    @property
    def artifact_prefixes(self) -> tuple[str, str]:
        """
        Retorna os prefixos corretos para a nomenclatura dos artefatos no disco,
        respeitando a semântica da estratégia selecionada.
        Retorna: (prefixo_pipeline_1, prefixo_pipeline_2)
        """
        if self is PLDATrainingStrategy.STANDARD:
            return "train", "calib"
        return "source", "target"