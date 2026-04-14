import json
import shutil
from pathlib import Path
from typing import Callable, Optional, Dict, Any, Union

# --- MANAGERS E SERVIÇOS ---
from faceaudit.manager.strategies_manager import StrategiesManager
from faceaudit.manager.validation_manager import ValidationManager  # <--- NOVO IMPORT
from faceaudit.services.hash_service import HashService
from faceaudit.services.config_loader_service import ConfigLoaderService
from faceaudit.services.file_system_service import FileSystemService

# --- MODELS E ENUMS ---
from faceaudit.enums.strategy_type import StrategyType
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.models.project_config import ProjectConfig
from faceaudit.models.strategy_entry import StrategyEntry
from faceaudit.models.plda_config import PLDAConfig
from faceaudit.models.kde_config import KDEConfig
from faceaudit.models.validation_config import ValidationConfig # <--- NOVO IMPORT
from faceaudit.models.project_file import ProjectFile

class ProjectManager:
    """
    Orquestrador de Projetos Autocontidos (Pendrive Paradigm).
    Salva e restaura o ambiente lendo e escrevendo o Manifesto Forense (project_files).
    """

    def __init__(self,
                 strategies_manager: StrategiesManager,
                 validation_manager: ValidationManager, # <--- NOVA INJEÇÃO AQUI
                 hash_service: HashService,
                 config_loader: ConfigLoaderService,
                 file_system_service: FileSystemService):
        
        self.strategies_manager = strategies_manager
        self.validation_manager = validation_manager
        self.hash_service = hash_service
        self.config_loader = config_loader
        self.fs_service = file_system_service
        
        self.current_project_path: Optional[Path] = None
        
        # --- CONTROLE DE ESTADO (DIRTY FLAG) ---
        self._is_dirty: bool = False 
        self.strategies_manager.set_global_dirty_callback(self.mark_as_dirty)

    # =========================================================================
    # 0. CONTROLE DE MODIFICAÇÕES (DIRTY STATE)
    # =========================================================================
    
    def is_dirty(self) -> bool:
        """Verifica se há alterações não salvas no projeto (Estratégias ou Validação)."""
        return self._is_dirty or self.validation_manager.is_dirty()

    def mark_as_dirty(self):
        """Marca o projeto como modificado (requer salvamento)."""
        self._is_dirty = True

    def reset_dirty_state(self):
        """Limpa a flag de modificação (usado após salvar ou carregar)."""
        self._is_dirty = False
        self.validation_manager.set_dirty(False)

    # =========================================================================
    # 1. FLUXO DE SALVAR (SAVE)
    # =========================================================================

    def save_project(self, 
                     filepath: Union[str, Path], 
                     copy_dependencies: bool = True, 
                     copy_images: bool = False, 
                     cleanup_orphans: bool = False, 
                     logger_callback: Callable[[str], None] = print) -> bool:
        
        logger_callback(f"<b>Iniciando salvamento do projeto autocontido...</b>")
        
        path_obj = Path(filepath)
        project_root = path_obj.parent
        project_name = path_obj.stem
        
        # 1. Cria a estrutura física
        self.fs_service.create_project_structure(project_root, logger_callback)
        self.current_project_path = project_root

        # 2. Inicializa o DTO Raiz
        project_config = ProjectConfig(name=project_name)
        active_managers = self.strategies_manager.get_all_managers()

        if not active_managers:
            logger_callback("⚠️ Nenhum modelo ativo para salvar.")

        # 3. Consolida as Estratégias e Executa Deep Copy de Dependências
        for strategy_id, manager in active_managers.items():
            config = manager.get_config()
            if not config: continue

            strat_type = StrategyType.PLDA if isinstance(config, PLDAConfig) else StrategyType.KDE
            logger_callback(f"📦 Consolidando estratégia {strat_type.value}: {config.id}")

            self._ensure_dependencies_tracked(config, strat_type, logger_callback)

            if copy_dependencies:
                self._import_external_dependencies(config, logger_callback)

            entry = StrategyEntry(
                strategy_id=strategy_id,
                strategy_type=strat_type,
                model_config=config,
                trained_model_path=getattr(config, 'model_path', None) 
            )
            project_config.strategies.append(entry)
            
        # 4. CONSOLIDA A ARENA DE VALIDAÇÃO (LIMPO!) <---
        val_config = self.validation_manager.get_config()
        if val_config:
            logger_callback(f"📦 Consolidando ambiente da Arena de Validação")
            
            # Deixa o ValidationManager atualizar os hashes por segurança e puxar dependências globais se precisar
            self.validation_manager._lock_local_inputs(logger_callback)
            
            if copy_dependencies:
                self._import_external_dependencies(val_config, logger_callback)
            
            project_config.validation_config = val_config

        # 5. Grava o Manifesto, Assina e Empacota
        try:
            fap_path, fap_hash = self.fs_service.save_project_config(project_config, logger_callback)
            
            hash_file_path = project_root / f"{project_name}.fap.sha256"
            with open(hash_file_path, "w", encoding="utf-8") as hf:
                hf.write(f"{fap_hash} *{fap_path.name}\n")
            
            logger_callback(f"🔐 <b>Lacre digital gerado:</b> {hash_file_path.name}")
            logger_callback(f"✅ <b>Projeto consolidado em disco:</b> {project_root.name}")

            # 5. Tratamento de Imagens e Lixo
            if copy_images:
                self._import_images(project_config, logger_callback)

            if cleanup_orphans:
                self._run_garbage_collector(project_config, project_root, logger_callback)

            self.reset_dirty_state()
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger_callback(f"❌ Erro crítico ao gravar manifesto .fap: {e}")
            return False

    # =========================================================================
    # 2. FLUXO DE ABRIR E AUDITORIA (LOAD)
    # =========================================================================

    def load_project(self, filepath: Union[str, Path], logger_callback: Callable[[str], None] = print) -> bool:
        path_obj = Path(filepath)
        project_root = path_obj.parent
        
        if not path_obj.exists():
            logger_callback(f"❌ Arquivo não encontrado: {filepath}")
            return False

        logger_callback(f"<b>📂 Abrindo projeto:</b> {project_root.name}")
        self.current_project_path = project_root

        try:
            project_config = self.fs_service.load_project_config(path_obj, logger_callback)
        except Exception as e:
            logger_callback(f"❌ Falha ao decodificar arquivo de projeto: {e}")
            return False

        logger_callback("♻️ Limpando área de trabalho atual...")
        for strat_id in list(self.strategies_manager.get_all_managers().keys()):
            self.strategies_manager.remove_strategy(strat_id)

        # Restaurar Estratégias
        for entry in project_config.strategies:
            config = entry.model_config
            strat_type = entry.strategy_type
            
            logger_callback(f"⚙️ Restaurando {strat_type.value}: {config.id}")
            self._verify_manifest_integrity(config, logger_callback)
                
            manager_id = self.strategies_manager.create_strategy(strat_type, force_id=config.id)
            manager = self.strategies_manager.get_manager(manager_id)
            manager.update_config(config)
            
            model_file = next((pf for pf in config.project_files if pf.file_type == ProjectFileType.TRAINED_STRATEGY), None)
            
            if model_file:
                model_full_path = project_root / model_file.path
                if model_full_path.exists():
                    logger_callback(f"  📥 Carregando binário matemático para a RAM...")
                    manager.set_model_path(model_full_path)
                    manager.load_model()
                else:
                    logger_callback(f"  ❌ Estratégia física ausente: {model_file.path}")

        # ---> NOVO: RESTAURA A ARENA DE VALIDAÇÃO <---
        if project_config.validation_config:
            logger_callback("⚙️ Restaurando cenário da Arena de Validação")
            self._verify_manifest_integrity(project_config.validation_config, logger_callback)
            self.validation_manager.load_session(project_config.validation_config)
        else:
            # Garante que inicie vazio se não havia config salvo
            self.validation_manager.load_session(ValidationConfig())

        logger_callback("✅ <b>Sincronização concluída com sucesso.</b>")
        self.reset_dirty_state()
        return True

    def _verify_manifest_integrity(self, config: Union[PLDAConfig, KDEConfig, ValidationConfig], logger: Callable):
        if not config.project_files:
            return

        proj_root = self.fs_service.current_project_root
        for pf in list(config.project_files):
            logger(f"  🔍 Auditando {pf.file_type.name}...")
            abs_path = proj_root / pf.path
            
            if not abs_path.exists():
                name_hint = Path(pf.path).stem
                abs_path = self.fs_service.current_project_root / pf.path

            if not abs_path or not abs_path.exists():
                logger(f"  ❌ ARQUIVO AUSENTE (Local/Global): {Path(pf.path).name}")
                config.project_files.remove(pf)
                continue

            current_hash = self.hash_service.calculate_file_hash(abs_path)
            if current_hash != pf.hash:
                logger(f"  🚨 CORRUPÇÃO DETECTADA! Hash divergiu: {abs_path.name}")
                logger(f"  🗑️ Apagando evidência não-confiável para proteger o laudo...")
                try:
                    abs_path.unlink()
                    config.project_files.remove(pf)
                except Exception as e:
                    logger(f"  ⚠️ Falha ao tentar excluir arquivo corrompido: {e}")
            else:
                logger(f"  ✅ Hash OK: {current_hash[:12]}...")

    # =========================================================================
    # 3. GESTÃO DE ATIVOS GLOBAIS (REUSÁVEIS)
    # =========================================================================
    
    def save_global_asset(self, file_type: ProjectFileType, data: Any, logger_callback: Optional[Callable[[str], None]] = print, **kwargs) -> Path:
        try:
            abs_path = self.fs_service.save_global_file(
                file_type=file_type, 
                data=data, 
                logger_callback=logger_callback, 
                **kwargs
            )
            return abs_path
        except Exception as e:
            if logger_callback:
                logger_callback(f"❌ Falha ao tentar salvar o ativo global '{file_type.name}': {e}")
            raise e

    # =========================================================================
    # 4. TRATAMENTO DE DEPENDÊNCIAS 
    # =========================================================================

    def _ensure_dependencies_tracked(self, config: Union[PLDAConfig, KDEConfig], strat_type: StrategyType, logger: Callable):
        def track_if_needed(name: str, f_type: ProjectFileType):
            if not name or name == "Selecione...": return
            clean_name = name.split(" (")[0]
            exists = any(pf.file_type == f_type and clean_name in pf.path for pf in config.project_files)
            if exists: return
            path = self.fs_service.resolve_file(clean_name, f_type)
            if path:
                rel_path = path.relative_to(self.fs_service.current_project_root).as_posix() if self.fs_service.current_project_root and str(self.fs_service.current_project_root) in str(path) else f"global/{clean_name}{f_type.extension}"
                f_hash = self.hash_service.calculate_file_hash(path)
                config.project_files.append(ProjectFile(file_type=f_type, path=rel_path, hash=f_hash))
                logger(f"  ↳ Vinculando dependência: {clean_name}")

        if strat_type == StrategyType.PLDA:
            track_if_needed(config.source_dataset_name, ProjectFileType.DATASET_METADATA)
            track_if_needed(config.target_dataset_name, ProjectFileType.DATASET_METADATA)
            track_if_needed(config.source_pipeline_name, ProjectFileType.PIPELINE_CONFIG)
            track_if_needed(config.target_pipeline_name, ProjectFileType.PIPELINE_CONFIG)
        else:
            track_if_needed(config.probe_dataset_name, ProjectFileType.DATASET_METADATA)
            track_if_needed(config.reference_dataset_name, ProjectFileType.DATASET_METADATA)
            mining_name = getattr(config.parameters, 'mining_config_name', None)
            if mining_name:
                track_if_needed(mining_name, ProjectFileType.MINING_CONFIG)

    def _import_external_dependencies(self, config: Union[PLDAConfig, KDEConfig, ValidationConfig], logger: Callable):
        if not self.fs_service.current_project_root: return

        for pf in config.project_files:
            if "global/" in pf.path:
                clean_name = Path(pf.path).stem
                actual_path = self.fs_service.resolve_file(clean_name, pf.file_type)
                
                if actual_path:
                    logger(f"  🔄 Deep Copy: Importando {clean_name} para o pendrive...")
                    new_path, new_hash = self.fs_service.copy_file_to_project(
                        src_path=actual_path, 
                        file_type=pf.file_type, 
                        logger_callback=logger,
                        dataset_name=clean_name,
                        pipeline_name=clean_name,
                        config_name=clean_name
                    )
                    pf.path = new_path.relative_to(self.fs_service.current_project_root).as_posix()
                    pf.hash = new_hash

    def _import_images(self, project_config: ProjectConfig, logger: Callable):
        logger(f"📸 Iniciando isolamento de fotografias (Deep Copy)...")
        dataset_names = set()
        
        # 1. Pega das estratégias
        for strat in project_config.strategies:
            for pf in strat.model_config.project_files:
                if pf.file_type == ProjectFileType.DATASET_METADATA:
                    dataset_names.add(Path(pf.path).stem)
                    
        # 2. Pega da Validação
        if project_config.validation_config:
            for pf in project_config.validation_config.project_files:
                if pf.file_type == ProjectFileType.DATASET_METADATA:
                    dataset_names.add(Path(pf.path).stem)
        
        for ds_name in dataset_names:
            self.fs_service.pack_dataset_images(ds_name, logger)
        logger("✅ Fotografias empacotadas com sucesso no projeto.")

    # =========================================================================
    # 5. GARBAGE COLLECTOR (FAXINA)
    # =========================================================================

    def _run_garbage_collector(self, project_config: ProjectConfig, project_root: Path, logger: Callable):
        logger("🧹 Iniciando Limpeza de Disco (Garbage Collector)...")
        
        authorized_paths = set()
        for strat in project_config.strategies:
            for pf in strat.model_config.project_files:
                abs_path = (project_root / pf.path).resolve()
                authorized_paths.add(str(abs_path))
                
        # ---> NOVO: Lista Branca respeita a Arena de Validação
        if project_config.validation_config:
            for pf in project_config.validation_config.project_files:
                abs_path = (project_root / pf.path).resolve()
                authorized_paths.add(str(abs_path))
        
        folders_to_sweep = [
            ProjectFolder.DATASETS, ProjectFolder.PIPELINES, ProjectFolder.MINING_CONFIGS,
            ProjectFolder.STRATEGIES, ProjectFolder.ARTIFACTS, ProjectFolder.GRAPHICS
        ]
        
        for folder in folders_to_sweep:
            folder_path = project_root / folder.value
            if not folder_path.exists(): continue
                
            for file_path in folder_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix != ".fap":
                    if str(file_path.resolve()) not in authorized_paths:
                        try:
                            file_path.unlink()
                            logger(f"  🗑️ Lixo removido: {file_path.name}")
                        except Exception as e:
                            logger(f"  ⚠️ Aviso: Não foi possível apagar {file_path.name}: {e}")
                            
        self._cleanup_orphan_image_folders(authorized_paths, project_root, logger)
        logger("✨ Sistema de arquivos limpo com sucesso.")

    def _cleanup_orphan_image_folders(self, authorized_paths: set, project_root: Path, logger: Callable):
        images_dir = project_root / "images"
        if not images_dir.exists(): return
        
        authorized_ds_names = {Path(p).stem for p in authorized_paths if Path(p).suffix == ".parquet"}
        
        for folder_path in images_dir.iterdir():
            if folder_path.is_dir() and folder_path.name not in authorized_ds_names:
                try:
                    shutil.rmtree(folder_path)
                    logger(f"  🗑️ Removido diretório de imagens órfão: {folder_path.name}")
                except Exception:
                    pass