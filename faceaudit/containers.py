import logging.config
from dependency_injector import containers, providers

# Services
from faceaudit.services import (
    HashService,
    FFService,
    OFIQService,
    SERFIQService,
    EmbeddingMathService,
    ComparisonService,
    DatasetComparisonService, 
    #EvaluationService,
    DatasetLoaderService,
    ConfigLoaderService,
    DatasetStatsService,
    PLDAValidationService,
    KDEValidationService,
    KDEVisualizationService,
    FileSystemService,
    KDEPlotterService,
    ComparisonMiningService,
    ValidationMathService,
    ArenaValidationService,
    PLDASplitService,  # <--- ADICIONADO
)

# Managers
from faceaudit.manager.strategies_manager import StrategiesManager
from faceaudit.manager.project_manager import ProjectManager
from faceaudit.manager.validation_manager import ValidationManager  # <--- ADICIONADO

# Pipelines
from faceaudit.pipelines import (
    CaseworkPipeline,
    FeatureExtractionPipeline,
    TransformationPipeline,
)


class ApplicationContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # =========================================================================
    # CORE SERVICES
    # =========================================================================

    hash_service = providers.Singleton(HashService)

    math_service = providers.Singleton(EmbeddingMathService)

    dataset_loader = providers.Singleton(DatasetLoaderService)

    # [CORREÇÃO] O nome do provider DEVE bater exatamente com o que está no Provide do Manager
    plda_split_service = providers.Singleton(PLDASplitService)  

    kde_plotter_service = providers.Factory(KDEPlotterService)
    kde_validation_service = providers.Singleton(KDEValidationService)
    kde_visualization_service = providers.Factory(KDEVisualizationService)
    validation_math_service = providers.Singleton(ValidationMathService)
    
    # Este provider pode ser usado com Provide[ApplicationContainer.comparison_mining_service]
    mining_service = providers.Singleton(ComparisonMiningService)

    _resolved_config_file = providers.Callable(
        lambda p: p or "config.json",
        config.system.config_file,
    )

    config_loader = providers.Singleton(
        ConfigLoaderService,
        config_file=_resolved_config_file,
    )

    file_system_service = providers.Singleton(
        FileSystemService,
        hash_service=hash_service,
        config_loader=config_loader
    )

    stats_service = providers.Singleton(
        DatasetStatsService,
    )

    plda_validation_service = providers.Singleton(PLDAValidationService)

    ff_service = providers.Singleton(
        FFService,
        model=config.ff.model,
        det_size=config.ff.det_size.as_int(),
        use_gpu=config.ff.use_gpu.as_bool()
    )

    ofiq_service = providers.Singleton(
        OFIQService,
        ofiq_exe_path=config.ofiq.ofiq_exe_path,
        ofiq_data_path=config.ofiq.ofiq_data_path
    )

    serfiq_service = providers.Singleton(
        SERFIQService,
        ff_service=ff_service,
        T=config.serfiq.t.as_int(),
        alpha=config.serfiq.alpha.as_float()
    )

    comparison_service = providers.Singleton(ComparisonService)
    
    dataset_comparison_service = providers.Singleton(DatasetComparisonService)
    
    #evaluation_service = providers.Singleton(EvaluationService)
    
    # <--- NOVO: O GATEKEEPER DA ARENA --->
    arena_validation_service = providers.Singleton(ArenaValidationService)

    # =========================================================================
    # MANAGERS
    # =========================================================================

    strategies_manager = providers.Singleton(StrategiesManager)
    
    # <--- NOVO: MANAGER DA VALIDAÇÃO --->
    # 1. PRIMEIRO o ValidationManager
    validation_manager = providers.Singleton(
        ValidationManager,
        strategies_manager=strategies_manager,
        math_service=validation_math_service,
        fs_service=file_system_service,
        dataset_loader=dataset_loader,
        stats_service=stats_service,
        arena_validator=arena_validation_service,
        hash_service=hash_service # <--- ADICIONADO AQUI
    )

    # 2. DEPOIS o ProjectManager (agora recebendo o validation_manager)
    project_manager = providers.Singleton(
        ProjectManager,
        strategies_manager=strategies_manager,
        validation_manager=validation_manager, 
        hash_service=hash_service,
        config_loader=config_loader,
        file_system_service=file_system_service
    )

    # =========================================================================
    # PIPELINES
    # =========================================================================

    feature_extraction_pipeline = providers.Factory(
        FeatureExtractionPipeline,
        hash_service=hash_service,
        ff_service=ff_service,
        ofiq_service=ofiq_service,
        serfiq_service=serfiq_service,
        storage_path=config.paths.datasets
    )

    transformation_pipeline = providers.Factory(TransformationPipeline)

    #validation_pipeline = providers.Factory(
    #    ValidationPipeline,
    #    strategies=[],
    #    evaluation_service=evaluation_service
    #)

    casework_pipeline = providers.Factory(CaseworkPipeline)