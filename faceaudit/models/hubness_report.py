# faceaudit/models/hubness_report.py

"""
DTO compósito agregando os 5 estudos produzidos pela instrução HUBNESS_ANALYSIS:

- analysis              : HubnessAnalysisInfo (resultado base de compute_analysis)
- stratification        : List[HubnessStratificationInfo] (5 faixas de yaw)
- paired_study          : HubnessPairedStudyInfo (frontal × não-frontal por ID)
- quality_diagnostics   : HubnessQualityDiagnosticsInfo (FIQA × pose, Norma × pose)
- optimization_study    : HubnessOptimizationStudyInfo (sweet spot lab/wild/perfil)

Persistido como pickle (preserva ndarrays internos das listas de samples).
Carregado via StrategyResourceLoaderService quando o usuário seleciona o
recurso na UI da nova arquitetura.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.hubness_analysis_info import HubnessAnalysisInfo
from faceaudit.models.hubness_stratification_info import HubnessStratificationInfo
from faceaudit.models.hubness_paired_study_info import HubnessPairedStudyInfo
from faceaudit.models.hubness_quality_diagnostics_info import (
    HubnessQualityDiagnosticsInfo,
)
from faceaudit.models.hubness_optimization_study_info import (
    HubnessOptimizationStudyInfo,
)


T = TypeVar("T", bound="HubnessReport")


@dataclass
class HubnessReport(JSONSerializableMixin):
    """
    Relatório compósito do diagnóstico de hubness.

    Atributos:
        analysis: análise topológica base (Nk, impostor scores, correlações).
        stratification: 5 faixas de yaw (Frontal/Leve/Moderada/Forte/Extrema).
        paired_study: estudo pareado frontal × não-frontal por identidade.
        quality_diagnostics: covariável FIQA / Norma × pose.
        optimization_study: sweet spot lab (<8°) × wild (15-30°) × perfil (>45°).
        dataset_name: nome do dataset analisado (rastreabilidade).
        k_neighbors: K usado no KNN exaustivo.
        n_samples: contagem de amostras analisadas.
    """

    analysis: HubnessAnalysisInfo
    stratification: List[HubnessStratificationInfo] = field(default_factory=list)
    paired_study: Optional[HubnessPairedStudyInfo] = None
    quality_diagnostics: Optional[HubnessQualityDiagnosticsInfo] = None
    optimization_study: Optional[HubnessOptimizationStudyInfo] = None

    dataset_name: str = ""
    k_neighbors: int = 0
    n_samples: int = 0

    def to_dict(self) -> dict:
        """
        Serializa para JSON delegando aos to_dict() dos filhos.
        Atenção: o relatório é tipicamente persistido em PKL (não JSON) por
        conter ndarrays nos samples — este método existe para inspeção e
        roundtrip de testes.
        """
        def _child(obj):
            return obj.to_dict() if (obj is not None and hasattr(obj, "to_dict")) else obj

        return {
            "analysis": _child(self.analysis),
            "stratification": [_child(s) for s in self.stratification],
            "paired_study": _child(self.paired_study),
            "quality_diagnostics": _child(self.quality_diagnostics),
            "optimization_study": _child(self.optimization_study),
            "dataset_name": self.dataset_name,
            "k_neighbors": int(self.k_neighbors),
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        d = dict(data)

        if isinstance(d.get("analysis"), dict):
            d["analysis"] = HubnessAnalysisInfo.from_dict(d["analysis"])

        if isinstance(d.get("stratification"), list):
            d["stratification"] = [
                HubnessStratificationInfo.from_dict(s) if isinstance(s, dict) else s
                for s in d["stratification"]
            ]

        if isinstance(d.get("paired_study"), dict):
            d["paired_study"] = HubnessPairedStudyInfo.from_dict(d["paired_study"])

        if isinstance(d.get("quality_diagnostics"), dict):
            d["quality_diagnostics"] = HubnessQualityDiagnosticsInfo.from_dict(
                d["quality_diagnostics"]
            )

        if isinstance(d.get("optimization_study"), dict):
            d["optimization_study"] = HubnessOptimizationStudyInfo.from_dict(
                d["optimization_study"]
            )

        return cls(**d)
