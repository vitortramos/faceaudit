from __future__ import annotations
from enum import Enum
from typing import Dict, Any, Optional, Callable, Type
from dataclasses import asdict

from faceaudit.enums.step_type import StepType

# Imports dos DTOs (Seguros contra import circular, pois DTOs não importam este Enum)
from faceaudit.models.aggregation_config import AggregationConfig
from faceaudit.models.whitening_config import WhiteningConfig
from faceaudit.models.lda_config import LDAConfig
from faceaudit.models.coral_config import CoralConfig
from faceaudit.models.radial_gaussianization_config import RadialGaussianizationConfig

class TransformationType(Enum):
    """
    Define os tipos de transformadores de embeddings disponíveis no sistema.
    Cada membro encapsula metadados de UI, lógica de sumário, vínculo com StepType,
    regras de renderização visual (recalculo de eixos) e a classe de DTO correspondente.
    """

    CORAL_TRANSFORMATION = (
        "CORAL_TRANSFORMATION", 
        "CORAL Alignment", 
        "CORAL",
        "Alinhamento de covariância de domínio (Domain Adaptation).",
        "🎨",
        False,
        StepType.CORAL,
        True,  
        lambda p: (
            str(p.get('estimator', '')).replace('_', '-').upper()
            if str(p.get('estimator', 'standard')) != 'standard'
            else (
                "Standard (Auto)" 
                if p.get('use_grid_search', True) 
                else f"Reg:{p.get('regularization', 0.0):.1e}"
            )
        ),
        CoralConfig # <--- INJEÇÃO DO DTO
    )

    WHITENING_TRANSFORMATION = (
        "WHITENING_TRANSFORMATION", 
        "Whitening (PCA)", 
        "Whitening",
        "Descorrelaciona os dados e normaliza a variância (PCA Whitening).",
        "⚪",
        True,
        StepType.WHITENING,
        False, 
        lambda p: f"{p.get('n_components') or 'Full'}D [{p.get('style', 'PCA')}]",
        WhiteningConfig # <--- INJEÇÃO DO DTO
    )

    LDA_TRANSFORMATION = (
        "LDA_TRANSFORMATION", 
        "LDA Projection", 
        "LDA",
        "Linear Discriminant Analysis. Maximiza a separação entre classes.",
        "📊",
        True,
        StepType.LDA,
        True,  
        lambda p: f"{p.get('n_components') or 'Max'}D, Reg:{p.get('regularization', 0.0):.1e}",
        LDAConfig # <--- INJEÇÃO DO DTO
    )

    CENTERING_TRANSFORMATION = (
        "CENTERING_TRANSFORMATION", 
        "Mean Centering", 
        "Center",
        "Remove a média dos dados (Centralização).",
        "🎯",
        False,
        StepType.CENTERING,
        False, 
        lambda p: "",
        None # <--- STATELESS (SEM DTO)
    ) 

    L2_NORMALIZATION = (
        "L2_NORMALIZATION", 
        "L2 Normalization", 
        "L2",
        "Normaliza a magnitude dos vetores para 1 (Esfera Unitária).",
        "📏",
        False,
        StepType.L2_NORM,
        False, 
        lambda p: "",
        None # <--- STATELESS (SEM DTO)
    )

    IDENTITY_AGGREGATION = (
        "IDENTITY_AGGREGATION",
        "Identity Aggregation",
        "IdAvg",
        "Calcula o vetor médio (Template) por identidade.",
        "👥",
        False,
        StepType.AGGREGATION,
        False,
        lambda p: "",
        AggregationConfig # <--- INJEÇÃO DO DTO
    )

    RADIAL_GAUSSIANIZATION = (
        "RADIAL_GAUSSIANIZATION",
        "Radial Gaussianization",
        "RadialGauss",
        "Mapeia a marginal radial dos embeddings para a chi-density "
        "de uma gaussiana esférica. Remove dependências não-lineares "
        "em fontes elipticamente simétricas (Lyu & Simoncelli 2009). "
        "Use após Whitening.",
        "🌀",
        False,
        StepType.RADIAL_GAUSSIANIZATION,
        False,
        lambda p: f"clip={p.get('eps_clip', 1e-6):.0e}",
        RadialGaussianizationConfig # <--- INJEÇÃO DO DTO
    )

    def __new__(cls, value: str, label: str, summary_name: str, description: str, 
                icon: str, supports_dimension_reduction: bool, 
                step_type: StepType, requires_axis_recalculation: bool, 
                param_formatter: Callable[[Dict[str, Any]], str],
                dto_class: Optional[Type]): # <--- NOVO PARÂMETRO NA FÁBRICA
        """
        Construtor para membros com múltiplos atributos.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.summary_name = summary_name
        obj.description = description
        obj.icon = icon
        obj.supports_dimension_reduction = supports_dimension_reduction
        obj.step_type = step_type
        obj.requires_axis_recalculation = requires_axis_recalculation
        obj.param_formatter = param_formatter
        obj.dto_class = dto_class # <--- ATRIBUTO SALVO
        return obj

    def format_summary(self, config_obj: Any) -> str:
        """Gera um texto curto descrevendo a etapa. Suporta DTOs ou Dicionários antigos."""
        # Se for um DTO (dataclass), converte para dict para a lambda antiga funcionar
        if config_obj and hasattr(config_obj, '__dataclass_fields__'):
            p = asdict(config_obj)
        else:
            p = config_obj if config_obj is not None else {}
            
        details = self.param_formatter(p)
        if details:
            return f"{self.summary_name} ({details})"
        return self.summary_name

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> Optional[TransformationType]:
        try:
            return cls(value)
        except ValueError:
            return None