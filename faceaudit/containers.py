import logging.config
from dependency_injector import containers, providers

# Services (apenas os usados pela arch nova + Dataset Manager)
from faceaudit.services import (
    HashService,
    FFService,
    OFIQService,
    SERFIQService,
    EmbeddingMathService,
    ComparisonService,
    DatasetComparisonService,
    DatasetLoaderService,
    ConfigLoaderService,
    DatasetStatsService,
    DatasetAuditService,
    ComparisonMiningService,
    ValidationMathService,
    PLDASplitService,
)

# Pipelines
from faceaudit.pipelines import FeatureExtractionPipeline


class ApplicationContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # =========================================================================
    # CORE SERVICES
    # =========================================================================

    hash_service = providers.Singleton(HashService)
    math_service = providers.Singleton(EmbeddingMathService)
    dataset_loader = providers.Singleton(DatasetLoaderService)

    # PLDASplitService — usado pelo executor DATASET_SPLIT da nova arch.
    plda_split_service = providers.Singleton(PLDASplitService)

    validation_math_service = providers.Singleton(ValidationMathService)

    # ComparisonMiningService — consumido via
    # `Provide[ApplicationContainer.mining_service]` no executor PAIR_MINING.
    mining_service = providers.Singleton(ComparisonMiningService)

    _resolved_config_file = providers.Callable(
        lambda p: p or "config.json",
        config.system.config_file,
    )

    config_loader = providers.Singleton(
        ConfigLoaderService,
        config_file=_resolved_config_file,
    )

    stats_service = providers.Singleton(DatasetStatsService)

    # FFService Singleton — modelo principal vem da preferência do usuário
    # via `config.ff.default_model` (None cai no `SepaelModel.default()`).
    # `det_size` e `extended` continuam fixos em defaults; vêm do
    # `DatasetIngestionConfig` por execução quando o usuário pede outros
    # valores (FeatureExtractionPipeline cria FFService temporário).
    #
    # `use_gpu` / `batch_size` / `default_model` passados DIRETO (sem
    # `.as_bool()` / `.as_int()`). O dependency-injector usa esses helpers
    # pra converter strings de .env — quando o valor vem de JSON e já é
    # bool/int nativo, `.as_bool()` levanta AttributeError silenciado e
    # `.as_int()` retorna um Provider object não-resolvido.
    ff_service = providers.Singleton(
        FFService,
        use_gpu=config.ff.use_gpu,
        batch_size=config.ff.batch_size,
        model=config.ff.default_model,
    )

    ofiq_service = providers.Singleton(
        OFIQService,
        ofiq_exe_path=config.ofiq.ofiq_exe_path,
        ofiq_data_path=config.ofiq.ofiq_data_path,
    )

    # SERFIQService com defaults (T=10, alpha=1.5). Override por execução
    # via DatasetIngestionConfig.serfiq_t / serfiq_alpha.
    serfiq_service = providers.Singleton(
        SERFIQService,
        ff_service=ff_service,
    )

    comparison_service = providers.Singleton(ComparisonService)
    dataset_comparison_service = providers.Singleton(DatasetComparisonService)

    # DatasetAuditService — auditoria de FaceDataset (paths órfãos, hash
    # conflitante, identidades singleton, qualidade baixa, etc.). Stateless;
    # consumido pelo Dataset Manager (UI) e (futuramente) por validation
    # rules da V2 antes de treinar estratégia.
    dataset_audit_service = providers.Singleton(
        DatasetAuditService,
        comparison_service=comparison_service,
        stats_service=stats_service,
    )

    # =========================================================================
    # PIPELINES
    # =========================================================================

    # storage_path = library/datasets — derivado, já que paths.datasets virou
    # property de PathsConfig e não mais chave própria no config tree.
    _datasets_dir = providers.Callable(
        lambda lib: str(__import__("pathlib").Path(lib) / "datasets"),
        config.paths.library,
    )

    feature_extraction_pipeline = providers.Factory(
        FeatureExtractionPipeline,
        hash_service=hash_service,
        ff_service=ff_service,
        ofiq_service=ofiq_service,
        serfiq_service=serfiq_service,
        storage_path=_datasets_dir,
    )
