from .comparison_service import ComparisonService
from .comparison_mining_service import ComparisonMiningService
from .config_loader_service import ConfigLoaderService
from .dataset_audit_service import DatasetAuditService
from .dataset_comparison_service import DatasetComparisonService
from .dataset_loader_service import DatasetLoaderService
from .dataset_stats_service import DatasetStatsService
from .dimensionality_reduction_service import DimensionalityReductionService
from .embedding_math_service import EmbeddingMathService
from .ff_service import FFService
from .hash_service import HashService
from .ofiq_service import OFIQService
from .serfiq_service import SERFIQService
from .trajectory_plotter_service import TrajectoryPlotterService
from .validation_math_service import ValidationMathService
from .plda_split_service import PLDASplitService
from .movmf_typicality_model import MoVMFTypicalityModel
from .knn_typicality_model import KNNTypicalityModel

# Compat: nome antigo continua importável até V2.16
TypicalityModel = MoVMFTypicalityModel
from .hubness_analysis_service import HubnessAnalysisService
from .hubness_plot_service import HubnessPlotService


__all__ = [
    "FFService",
    "OFIQService",
    "SERFIQService",
    "HashService",
    "ComparisonService",
    "ComparisonMiningService",
    "EmbeddingMathService",
    "DatasetStatsService",
    "DatasetLoaderService",
    "DimensionalityReductionService",
    "ConfigLoaderService",
    "DatasetAuditService",
    "DatasetComparisonService",
    "TrajectoryPlotterService",
    "ValidationMathService",
    "PLDASplitService",
    "MoVMFTypicalityModel",
    "TypicalityModel",  # alias deprecado
    "KNNTypicalityModel",
    "HubnessAnalysisService",
    "HubnessPlotService",
]
