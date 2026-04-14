# faceaudit/controllers/kde_manager.py
from pathlib import Path
import shutil
import json
import pickle
import concurrent.futures
from typing import Optional, Tuple, Dict, Any, Callable, List
from datetime import datetime

# --- DEPENDENCY INJECTION ---
from dependency_injector.wiring import inject, Provide
from faceaudit.containers import ApplicationContainer

# PROTOCOLO
from faceaudit.protocols.strategy_manager_protocol import StrategyManagerProtocol

# ENUMS DE DOMÍNIO E DTOs
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.models.project_file import ProjectFile

# MODELS
from faceaudit.models.kde_config import KDEConfig
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.kde_validation_input import KDEValidationInput
from faceaudit.models.kde_validation_result import KDEValidationResult, ValidationSeverity
from faceaudit.models.kde_training_artifacts import KDETrainingArtifacts
from faceaudit.models.kde_visualization_data import KDEVisualizationData
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig

# STRATEGY
from faceaudit.strategies.kde_strategy import KDEStrategy

# SERVICES
from faceaudit.services.config_loader_service import ConfigLoaderService
from faceaudit.services.dataset_loader_service import DatasetLoaderService
from faceaudit.services.dataset_stats_service import DatasetStatsService
from faceaudit.services.kde_validation_service import KDEValidationService
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.services.comparison_service import ComparisonService
from faceaudit.services.comparison_mining_service import ComparisonMiningService
from faceaudit.services.kde_visualization_service import KDEVisualizationService
from faceaudit.services.kde_plotter_service import KDEPlotterService
from faceaudit.services.hash_service import HashService
from faceaudit.services.file_system_service import FileSystemService

class KDEManager(StrategyManagerProtocol):
    """
    Gerenciador de Sessão para estratégia KDE.
    Refatorado para o "Pendrive Paradigm", suportando leitura Lazy
    de arquivos através do FileSystemService e uso do Manifesto Forense.
    """

    @inject
    def __init__(self, 
                 config_loader: ConfigLoaderService = Provide[ApplicationContainer.config_loader],
                 dataset_loader: DatasetLoaderService = Provide[ApplicationContainer.dataset_loader],
                 stats_service: DatasetStatsService = Provide[ApplicationContainer.stats_service],
                 validation_service: KDEValidationService = Provide[ApplicationContainer.kde_validation_service],
                 math_service: EmbeddingMathService = Provide[ApplicationContainer.math_service],
                 comparison_service: ComparisonService = Provide[ApplicationContainer.comparison_service],
                 mining_service: ComparisonMiningService = Provide[ApplicationContainer.mining_service],
                 visualization_service: KDEVisualizationService = Provide[ApplicationContainer.kde_visualization_service],
                 plotter_service: KDEPlotterService = Provide[ApplicationContainer.kde_plotter_service],
                 hash_service: HashService = Provide[ApplicationContainer.hash_service],
                 file_system_service: FileSystemService = Provide[ApplicationContainer.file_system_service]):
        
        self.config_loader = config_loader
        self.dataset_loader = dataset_loader
        self.stats_service = stats_service
        self.validation_service = validation_service
        self.math_service = math_service
        self.comparison_service = comparison_service
        self.mining_service = mining_service 
        self.visualization_service = visualization_service
        self.plotter_service = plotter_service
        self.hash_service = hash_service
        self.fs_service = file_system_service

        # --- Estado da Configuração ---
        self._card_config: Optional[KDEConfig] = None
        
        # --- Observer de Modificação (Dirty Flag Callback) ---
        self._dirty_callback: Optional[Callable[[], None]] = None
        
        # --- Cache de Dados (Volátil) ---
        self._probe_raw: Optional[EmbeddingDataset] = None
        self._reference_raw: Optional[EmbeddingDataset] = None
        self._probe_ready: Optional[EmbeddingDataset] = None
        self._reference_ready: Optional[EmbeddingDataset] = None
        self._post_prep_stats: Optional[Dict[str, Any]] = None

        # --- Ciclo de Vida do Modelo (Persistência) ---
        self._fitted_strategy: Optional[KDEStrategy] = None
        self._last_model_path: Optional[Path] = None

    # =========================================================================
    # 0. NOTIFICAÇÕES DE ESTADO (DIRTY FLAG)
    # =========================================================================

    def set_dirty_callback(self, callback: Callable[[], None]) -> None:
        """Injeta a função que será chamada quando o estado do manager mudar."""
        self._dirty_callback = callback

    def _trigger_dirty(self) -> None:
        """Avisa o orquestrador (ProjectManager) que o projeto foi alterado."""
        if self._dirty_callback:
            self._dirty_callback()

    # =========================================================================
    # 1. GESTÃO DE CONFIGURAÇÃO E ESTADO
    # =========================================================================

    def update_config(self, new_config: KDEConfig, logger_callback: Callable[[str], None] = print) -> None:
        if self._card_config is None:
            self._card_config = new_config
            self._trigger_dirty() # Notifica carga inicial
            return

        old_config = self._card_config
        
        datasets_changed = (
            new_config.probe_dataset_name != old_config.probe_dataset_name or
            new_config.reference_dataset_name != old_config.reference_dataset_name or
            new_config.embedding_type != old_config.embedding_type
        )
        
        # [MODIFICADO]: Agora comparamos o DTO inteiro em vez de parâmetros soltos
        prep_changed = new_config.aggregation != old_config.aggregation
        
        # Dataclasses em Python suportam checagem de igualdade profunda nativa. 
        math_params_changed = new_config.parameters != old_config.parameters
        
        if datasets_changed:
            self.clear_cache(logger_callback)
            self.invalidate_math_artifacts(logger_callback)
        elif prep_changed:
            self._clear_prepared_data(logger_callback)
            self.invalidate_math_artifacts(logger_callback)
        elif math_params_changed:
            # OTIMIZAÇÃO: Se só a matemática mudou, não apaga os dados preparados da RAM!
            # Apenas invalida o modelo antigo para forçar uma nova calibração rápida.
            self.invalidate_math_artifacts(logger_callback)

        if datasets_changed or math_params_changed or prep_changed:
            self._trigger_dirty()

        # Preserva a lista de rastreamento forense
        new_config.project_files = old_config.project_files.copy()
        self._card_config = new_config

    def get_config(self) -> Optional[KDEConfig]:
        return self._card_config
    
    def get_training_datasets(self) -> List[str]:
        """Retorna os datasets usados na calibração do KDE para auditoria de Leakage."""
        if not self._card_config:
            return []
        
        datasets = []
        if self._card_config.probe_dataset_name and self._card_config.probe_dataset_name != "Selecione...":
            datasets.append(self._card_config.probe_dataset_name)
            
        if self._card_config.reference_dataset_name and self._card_config.reference_dataset_name != "Selecione...":
            datasets.append(self._card_config.reference_dataset_name)
            
        return datasets

    # =========================================================================
    # 2. VALIDAÇÃO (Check de Sanidade)
    # =========================================================================

    def validate_all(self) -> KDEValidationResult:
        if not self._card_config:
            return KDEValidationResult(ValidationSeverity.NEUTRAL)

        inp = self._build_validation_input()
        res = self.validation_service.validate(inp)

        if self._post_prep_stats:
            p_vecs = self._post_prep_stats.get("probe_vectors")
            r_vecs = self._post_prep_stats.get("ref_vectors")
            res.messages.append(f"\n📊 [Pós-Prep] Dados Prontos: Probe={p_vecs}, Ref={r_vecs}")
            
            if p_vecs < 10 or r_vecs < 10:
                res.severity = ValidationSeverity.WARNING
                res.messages.append("⚠️ Quantidade de vetores muito baixa após agregação.")
        return res

    def _build_validation_input(self) -> KDEValidationInput:
        if not self._card_config: 
            raise ValueError("Config não inicializada.")

        # Usa a ponte do Manager para buscar os stats corretamente
        st_probe = self.get_dataset_stats(self._card_config.probe_dataset_name)
        st_ref = self.get_dataset_stats(self._card_config.reference_dataset_name)

        # Retorna ESTRITAMENTE o que a dataclass KDEValidationInput pede
        return KDEValidationInput(
            config=self._card_config, 
            probe_stats=st_probe, 
            reference_stats=st_ref
        )

    def validate_validation_setup(self, probe_val_name: str, ref_val_name: str) -> KDEValidationResult:
        """
        Salva as escolhas de validação no DTO e aciona a auditoria pericial (Data Leakage)
        via KDEValidationService.
        """
        if not self._card_config:
            raise ValueError("Config não inicializada.")

        # 1. Limpa nomes e atualiza o DTO (Para salvar no pendrive depois)
        def _clean_name(name):
            return name.split(" (")[0] if name and name != "Selecione..." else ""

        p_val_clean = _clean_name(probe_val_name)
        r_val_clean = _clean_name(ref_val_name)

        # Atualiza a configuração e avisa que o projeto foi alterado
        if self._card_config.probe_val_dataset_name != p_val_clean or \
           self._card_config.reference_val_dataset_name != r_val_clean:
            self._card_config.probe_val_dataset_name = p_val_clean
            self._card_config.reference_val_dataset_name = r_val_clean
            self._trigger_dirty()

        # 2. Busca os 4 blocos de estatísticas necessários para a checagem cruzada
        # Treino:
        st_probe_train = self.get_dataset_stats(self._card_config.probe_dataset_name)
        st_ref_train = self.get_dataset_stats(self._card_config.reference_dataset_name)
        
        # Validação:
        st_probe_val = self.get_dataset_stats(p_val_clean)
        st_ref_val = self.get_dataset_stats(r_val_clean)

        # 3. Delega para o Gatekeeper (KDEValidationService)
        return self.validation_service.validate_validation_setup(
            probe_train_stats=st_probe_train,
            ref_train_stats=st_ref_train,
            probe_val_stats=st_probe_val,
            ref_val_stats=st_ref_val
        )
        
    # =========================================================================
    # 3. I/O DE DADOS E PREPARAÇÃO
    # =========================================================================

    def load_input_datasets(self) -> None:
        if not self._card_config: raise ValueError("Configuração não definida.")

        def _load_one(name: str) -> Optional[EmbeddingDataset]:
            if not name or name == "Selecione...": return None
            path = self.fs_service.resolve_file(name, ProjectFileType.DATASET_METADATA)
            if not path: raise FileNotFoundError(f"Dataset '{name}' não encontrado.")
            
            raw_data = self.dataset_loader.load(path)
            if isinstance(raw_data, FaceDataset):
                return raw_data.to_embedding_dataset(self._card_config.embedding_type)
            return raw_data

        self._probe_raw = _load_one(self._card_config.probe_dataset_name)
        self._reference_raw = _load_one(self._card_config.reference_dataset_name)

    def run_preparation(self, logger_callback: Callable[[str], None] = print) -> Dict[str, Any]:
        """
        Orquestra a preparação (Smart Enrollment) delegando para o KDEPreparationController,
        que é uma dependência instanciada localmente.
        """
        if not self._card_config: raise ValueError("Configuração não definida.")
        
        # --- [TRAVA DE SEGURANÇA FORENSE] ---
        logger_callback("🔒 Auditando integridade dos dados de entrada...")
        self._lock_local_inputs(logger_callback)
        
        logger_callback("📥 Carregando datasets do disco para a memória...")
        if not self._probe_raw or not self._reference_raw: self.load_input_datasets()

        # [MODIFICADO]: Delegamos o trabalho pesado para o Controller Refatorado.
        from faceaudit.controllers.kde_preparation_controller import KDEPreparationController
        prep_controller = KDEPreparationController(math_service=self.math_service)

        logger_callback("⚡ Iniciando processamento do Pipeline (Single-Thread por Design)...")
        # Nota: KDEPreparationController já lida com o paralelismo se necessário,
        # ou aplica a lógica serial. Nós só esperamos o resultado final.
        
        # Para emular o comportamento anterior e manter o log preciso, fazemos a chamada
        # direta para a nova função prepare_dataset do controller
        try:
            # Como prepare_dataset retorna o dataset combinado e nós precisamos das metades
            # para o get_ready_datasets e pro fit do modelo KDE, usamos o _process_single_dataset
            
            if self._card_config.aggregation:
                 logger_callback("⚡ Iniciando agregação paralela de vetores (Multi-threading)...")
                 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                     future_probe = executor.submit(prep_controller._process_single_dataset, self._card_config, self._probe_raw, "Probe")
                     future_ref = executor.submit(prep_controller._process_single_dataset, self._card_config, self._reference_raw, "Reference")
                     
                     self._probe_ready = future_probe.result()
                     self._reference_ready = future_ref.result()
            else:
                 logger_callback("ℹ️ Agregação desabilitada. Usando vetores originais.")
                 self._probe_ready = self._probe_raw
                 self._reference_ready = self._reference_raw
                 
        except Exception as e:
             logger_callback(f"❌ Erro durante a preparação: {e}")
             raise e

        p_count, r_count = len(self._probe_ready), len(self._reference_ready)
        self._post_prep_stats = {"probe_vectors": p_count, "ref_vectors": r_count}
        
        logger_callback("✅ Preparação concluída. Dados alocados na RAM.")
        return self._post_prep_stats

    def is_prepared(self) -> bool:
        return self._probe_ready is not None and self._reference_ready is not None

    # =========================================================================
    # 4. CICLO DE VIDA DO MODELO E GERENCIAMENTO DE MANIFESTO
    # =========================================================================

    def add_project_file(self, file_type: ProjectFileType, abs_path: Path, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root: return

        if not abs_path.exists():
            logger_callback(f"⚠️ Aviso: Arquivo não encontrado para rastreio: {abs_path.name}")
            return

        try:
            file_hash = self.hash_service.calculate_file_hash(abs_path)
            rel_path = abs_path.relative_to(self.fs_service.current_project_root).as_posix()
            
            existing_file = next((pf for pf in self._card_config.project_files if pf.path == rel_path), None)
            if existing_file:
                existing_file.hash = file_hash
                existing_file.file_type = file_type
            else:
                new_pf = ProjectFile(file_type=file_type, path=rel_path, hash=file_hash)
                self._card_config.project_files.append(new_pf)
                
            logger_callback(f"  ↳ Manifesto: [{file_type.name}] {Path(rel_path).name} assinado.")
            self._trigger_dirty() 
        except Exception as e:
            logger_callback(f"⚠️ Erro ao registrar rastreio de {abs_path.name}: {e}")

    def delete_project_file(self, project_file: ProjectFile, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root: return

        if project_file in self._card_config.project_files:
            self._card_config.project_files.remove(project_file)
            self._trigger_dirty() 
            
        abs_path = self.fs_service.current_project_root / project_file.path
        if abs_path.exists():
            try:
                abs_path.unlink()
                logger_callback(f"🗑️ Arquivo apagado do disco: {abs_path.name}")
            except Exception as e:
                logger_callback(f"⚠️ Não foi possível apagar o arquivo {abs_path.name}: {e}")

    def invalidate_math_artifacts(self, logger_callback: Callable[[str], None] = print) -> None:
        self.unload_model()
        if not self._card_config or not self.fs_service.current_project_root: return
        
        dynamic_types = {ProjectFileType.TRAINED_STRATEGY, ProjectFileType.KDE_TRAINING_ARTIFACT, ProjectFileType.DISTRIBUTION_GRAPH}
        
        for pf in list(self._card_config.project_files):
            if pf.file_type in dynamic_types:
                self.delete_project_file(pf, logger_callback)

    def _cleanup_local_files(self, logger_callback: Callable[[str], None] = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root: return
            
        proj_root = self.fs_service.current_project_root
        strategy_id = self._card_config.id
        valid_paths = {pf.path for pf in self._card_config.project_files}
        
        folders_to_check = [ProjectFolder.ARTIFACTS, ProjectFolder.STRATEGIES, ProjectFolder.GRAPHICS]
        
        has_deleted = False
        for folder in folders_to_check:
            folder_path = proj_root / folder.value
            if not folder_path.exists(): continue
                
            for file_path in folder_path.glob(f"{strategy_id}_*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(proj_root).as_posix()
                    if rel_path not in valid_paths:
                        try:
                            file_path.unlink()
                            has_deleted = True
                            logger_callback(f"🧹 Faxina Local: Lixo órfão removido -> {file_path.name}")
                        except Exception as e:
                            logger_callback(f"⚠️ Aviso: Não foi possível apagar {file_path.name}: {e}")
        
        if has_deleted:
            self._trigger_dirty()

    def run_training(self, logger_callback: Callable[[str], None] = print) -> str:
        if not self.is_prepared(): 
            logger_callback("⚠️ Preparação não encontrada. Executando preparação forçada...")
            self.run_preparation(logger_callback=logger_callback)

        if not self.fs_service.current_project_root:
            raise RuntimeError("Raiz de projeto não inicializada. Crie ou abra um projeto antes de treinar.")

        mining_cfg_name = self._card_config.parameters.mining_config_name
        if not mining_cfg_name or mining_cfg_name == "Selecione...":
            raise ValueError("Uma regra de mineração válida deve ser selecionada para o treinamento.")

        path = self.fs_service.resolve_file(mining_cfg_name, ProjectFileType.MINING_CONFIG)
        if not path or not path.exists():
            raise FileNotFoundError(f"Arquivo JSON da Regra de Mineração '{mining_cfg_name}' não encontrado.")
        
        proj_root = self.fs_service.current_project_root
        if proj_root and proj_root not in path.parents:
            logger_callback(f"📥 Importando regra global '{mining_cfg_name}' para o projeto local...")
            path, _ = self.fs_service.copy_file_to_project(
                src_path=path,
                file_type=ProjectFileType.MINING_CONFIG,
                logger_callback=logger_callback,
                config_name=mining_cfg_name
            )

        mining_cfg = ComparisonMiningConfig.from_json(path)

        logger_callback("🧠 Inicializando otimização da densidade probabilística (KDE)...")
        strategy = KDEStrategy(
            comparison_service=self.comparison_service, 
            mining_service=self.mining_service,
            parameters=self._card_config.parameters, 
            name="KDE_Session"
        )
        
        strategy.fit(
            train_dataset=self._probe_ready, 
            calibration_dataset=self._reference_ready, 
            mining_config=mining_cfg,
            logger_callback=logger_callback
        )

        proj_root = self.fs_service.current_project_root
        
        rel_model_path = ProjectFileType.TRAINED_STRATEGY.build_path(strategy_id=self._card_config.id)
        abs_model_path = proj_root / rel_model_path
        abs_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        rel_artifacts_path = ProjectFileType.KDE_TRAINING_ARTIFACT.build_path(strategy_id=self._card_config.id)
        abs_artifacts_path = proj_root / rel_artifacts_path
        abs_artifacts_path.parent.mkdir(parents=True, exist_ok=True)

        rel_plot_path = ProjectFileType.DISTRIBUTION_GRAPH.build_path(strategy_id=self._card_config.id)
        abs_plot_path = proj_root / rel_plot_path
        abs_plot_path.parent.mkdir(parents=True, exist_ok=True)

        logger_callback("💾 Persistindo modelo e artefatos de visualização...")
        strategy.save(str(abs_model_path))

        try:
            artifacts: KDETrainingArtifacts = strategy.export_training_artifacts()
            artifacts.to_pickle(abs_artifacts_path)

            logger_callback("📊 Gerando gráfico de distribuição (Headless)...")
            viz_data = self.visualization_service.build_visualization_data(
                artifacts=artifacts,
                strategy=strategy,
                x_min=-0.25,
                x_max=1.0
            )

            clean_probe = self._card_config.probe_dataset_name.split(" (")[0]
            clean_ref = self._card_config.reference_dataset_name.split(" (")[0]
            
            self.plotter_service.save_distribution_plot(
                viz_data=viz_data, 
                save_path=abs_plot_path, 
                title_suffix=f"({clean_probe} vs {clean_ref})"
            )

        except Exception as e:
            logger_callback(f"⚠️ Falha ao salvar artefatos extras/gráficos: {e}")

        # Atualiza o Estado
        self._fitted_strategy = strategy
        self._last_model_path = abs_model_path
        
        logger_callback("🔐 Assinando binários gerados para cadeia de custódia...")
        self.add_project_file(ProjectFileType.TRAINED_STRATEGY, abs_model_path, logger_callback)
        self.add_project_file(ProjectFileType.KDE_TRAINING_ARTIFACT, abs_artifacts_path, logger_callback)
        
        if abs_plot_path.exists():
            self.add_project_file(ProjectFileType.DISTRIBUTION_GRAPH, abs_plot_path, logger_callback)

        if hasattr(self._card_config.parameters, 'mining_config_name'):
            mining_name = self._card_config.parameters.mining_config_name
            if mining_name and mining_name != "Selecione...":
                mining_path = self.fs_service.resolve_file(mining_name, ProjectFileType.MINING_CONFIG)
                if mining_path and mining_path.exists() and self.fs_service.current_project_root in mining_path.parents:
                    self.add_project_file(ProjectFileType.MINING_CONFIG, mining_path, lambda x: None)

        logger_callback("✅ Treinamento concluído e artefatos salvos com sucesso!")
        return str(abs_model_path)

    def get_model_path(self) -> Optional[Path]:
        return self._last_model_path

    def get_available_datasets(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.DATASET_METADATA)

    def get_available_mining_configs(self) -> Tuple[List[str], List[str]]:
        return self.fs_service.list_available_files(ProjectFileType.MINING_CONFIG)

    def get_dataset_stats(self, dataset_name: str):
        if not dataset_name or dataset_name == "Selecione...": 
            return None
            
        clean_name = dataset_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
        
        if path:
            return self.stats_service.get_stats(clean_name, target_directory=path.parent)
            
        return None

    def set_model_path(self, path: Path) -> None:
        self._last_model_path = path

    def has_model_on_disk(self) -> bool:
        return self._last_model_path is not None and self._last_model_path.exists()

    def is_model_loaded(self) -> bool:
        return self._fitted_strategy is not None

    def ensure_model_loaded(self) -> None:
        if self.is_model_loaded(): return
        
        if self.has_model_on_disk():
            success = self.load_model(self._last_model_path)
            if not success:
                raise RuntimeError(f"Falha ao carregar modelo KDE em: {self._last_model_path}")
        else:
            raise FileNotFoundError("Modelo KDE não encontrado no disco para esta estratégia.")

    def load_model(self, path: Optional[Path] = None) -> bool:
        target_path = path or self._last_model_path
        if not target_path or not target_path.exists(): return False

        try:
            strategy = KDEStrategy(
                comparison_service=self.comparison_service,
                mining_service=self.mining_service,
                parameters=self._card_config.parameters if self._card_config else None,
                name="KDE_Loaded"
            )
            strategy.load(str(target_path))
            self._fitted_strategy = strategy
            self._last_model_path = target_path
            return True
        except Exception:
            return False

    def unload_model(self) -> None:
        self._fitted_strategy = None
        self._last_model_path = None
    

    # =========================================================================
    # 5. VALIDAÇÃO DA ESTRATÉGIA E EXECUÇÃO
    # =========================================================================

    def run_strategy_validation(self, logger_callback: Any = print) -> Any:
        self.ensure_model_loaded()
        pass

    def run_inference(self, logger_callback: Any = print) -> Any:
        self.ensure_model_loaded()
        pass

    def run_validation_inference(self, 
                                 probe_dataset: FaceDataset, 
                                 reference_dataset: FaceDataset, 
                                 logger_callback: Callable[[str], None] = print) -> Tuple[List[float], List[float]]:
        self.ensure_model_loaded()

        # 1. Converte FaceDataset bruto para o EmbeddingDataset da estratégia
        emb_type = self._card_config.embedding_type
        logger_callback(f"     [KDE] Convertendo faces para embeddings {emb_type.name}...")
        probe_emb = probe_dataset.to_embedding_dataset(emb_type)
        ref_emb = reference_dataset.to_embedding_dataset(emb_type)

        # 2. Agregação (Smart Enrollment)
        from faceaudit.controllers.kde_preparation_controller import KDEPreparationController
        prep_controller = KDEPreparationController(math_service=self.math_service)

        if self._card_config.aggregation:
            val_probe_ready = prep_controller._process_single_dataset(self._card_config, probe_emb, "ValProbe")
            val_ref_ready = prep_controller._process_single_dataset(self._card_config, ref_emb, "ValRef")
        else:
            val_probe_ready = probe_emb
            val_ref_ready = ref_emb

        # 3. Confronto (Gera a matriz de similaridade normalizada)
        logger_callback(f"     [KDE] Computando matriz de similaridade...")
        comp_ds = self.comparison_service.process_comparison(val_probe_ready, val_ref_ready)

        # 4. Inferência Vetorizada (Usando o novo método .infer)
        logger_callback(f"     [KDE] Convertendo similaridades em Log-Likelihood Ratios...")
        lrs = self._fitted_strategy.infer(comp_ds)

        # 5. Tabulação Genuínos vs Impostores
        return self._split_scores_by_ground_truth(comp_ds, lrs)

    def _split_scores_by_ground_truth(self, 
                                      comparisons_dataset, 
                                      log_lrs: List[float]) -> Tuple[List[float], List[float]]:
        """
        Método auxiliar que mapeia a lista de LRs com a lista de comparações 
        para separar quem é quem baseado no campo 'same_person'.
        """
        genuine_lrs = []
        impostor_lrs = []

        for comp, lr in zip(comparisons_dataset.items, log_lrs):
            # Validação extra de segurança: só usa se a identidade for conhecida
            if comp.embedding_1.identity is None or comp.embedding_2.identity is None:
                continue
                
            if comp.same_person:
                genuine_lrs.append(lr)
            else:
                impostor_lrs.append(lr)

        return genuine_lrs, impostor_lrs

    # =========================================================================
    # 6. LIMPEZA E CACHE (RAM vs DISCO)
    # =========================================================================

    def clear_cache(self, logger_callback: Callable[[str], None] = print) -> None:
        self._probe_raw = None
        self._reference_raw = None
        self._clear_prepared_data(logger_callback)

    def _clear_prepared_data(self, logger_callback: Callable[[str], None] = print) -> None:
        self._probe_ready = None
        self._reference_ready = None
        self.unload_model()
        self._post_prep_stats = None

    def clear_state(self, wipe_disk: bool = False, logger_callback: Callable[[str], None] = print) -> None:
        self.clear_cache(logger_callback)
        self._last_model_path = None
        
        if wipe_disk and self._card_config:
            logger_callback(f"🗑️ Excluindo rastros da estratégia {self._card_config.id} no disco...")
            self.invalidate_math_artifacts(logger_callback)
            self._cleanup_local_files(logger_callback)

        self._card_config = None

    # =========================================================================
    # 7. MÉTODOS DE APOIO E VISUALIZAÇÃO
    # =========================================================================

    def get_ready_datasets(self) -> Tuple[EmbeddingDataset, Optional[EmbeddingDataset]]:
        if not self._probe_raw or not self._reference_raw: self.load_input_datasets()
        return self._probe_raw, self._reference_raw

    def get_trained_artifacts(self) -> Dict[str, Any]:
        if not self.is_model_loaded(): raise ValueError("Modelo não carregado.")
        return {
            'strategy': self._fitted_strategy,
            'probe_used': self._probe_ready,
            'reference_used': self._reference_ready
        }
    
    def build_visualization_data_from_model(self, model_path: Path, x_min: float = -0.25, x_max: float = 1.0) -> KDEVisualizationData:
        self.ensure_model_loaded()
        
        if not self._card_config or not self.fs_service.current_project_root:
            raise ValueError("O projeto ou a configuração da estratégia não foram inicializados.")
            
        artifact_pf = next((pf for pf in self._card_config.project_files if pf.file_type == ProjectFileType.KDE_TRAINING_ARTIFACT), None)
        
        if not artifact_pf:
            raise ValueError("O arquivo de projeto (.fap) não possui registro dos artefatos de visualização KDE.")
            
        artifacts_full_path = self.fs_service.current_project_root / artifact_pf.path
        
        if not artifacts_full_path.exists():
            raise FileNotFoundError(f"Arquivo físico de artefatos ausente: {artifacts_full_path.name}")
            
        artifacts = KDETrainingArtifacts.from_pickle(artifacts_full_path)
        
        return self.visualization_service.build_visualization_data(
            artifacts=artifacts,
            strategy=self._fitted_strategy,
            x_min=x_min,
            x_max=x_max
        )

    def get_mining_config(self, config_name: str) -> Optional[ComparisonMiningConfig]:
        if not config_name or config_name == "Selecione...": return None
        
        clean_name = config_name.split(" (")[0]
        path = self.fs_service.resolve_file(clean_name, ProjectFileType.MINING_CONFIG)
        
        if path and path.exists():
            cfg = ComparisonMiningConfig.from_json(path)
            cfg.name = clean_name 
            return cfg
        return None

    def save_or_update_mining_config(self, config: ComparisonMiningConfig, logger_callback: Callable = print) -> None:
        if not self.fs_service.current_project_root:
            raise RuntimeError("Nenhum projeto ativo para salvar a regra de mineração.")

        config.name = "".join(x for x in config.name if x.isalnum() or x in ('_', '-'))

        rel_path = ProjectFileType.MINING_CONFIG.build_path(config_name=config.name)
        file_path = self.fs_service.current_project_root / rel_path
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config.to_json(file_path)
        logger_callback(f"💾 Regra de mineração '{config.name}' salva no projeto local.")

        self.add_project_file(ProjectFileType.MINING_CONFIG, file_path, logger_callback)
        self.invalidate_math_artifacts(logger_callback)

    # =========================================================================
    # 8. TRAVAS DE CADEIA DE CUSTÓDIA
    # =========================================================================

    def _lock_local_inputs(self, logger_callback: Callable = print) -> None:
        if not self._card_config or not self.fs_service.current_project_root:
            return
            
        inputs = [
            (self._card_config.probe_dataset_name, ProjectFileType.DATASET_METADATA),
            (self._card_config.reference_dataset_name, ProjectFileType.DATASET_METADATA)
        ]
        
        if hasattr(self._card_config.parameters, 'mining_config_name') and self._card_config.parameters.mining_config_name:
             inputs.append((self._card_config.parameters.mining_config_name, ProjectFileType.MINING_CONFIG))
        
        for name, f_type in inputs:
            if not name or name == "Selecione...": continue
            clean_name = name.split(" (")[0]
            
            path = self.fs_service.resolve_file(clean_name, f_type)
            if path and path.exists() and self.fs_service.current_project_root in path.parents:
                self.add_project_file(f_type, path, lambda x: None)