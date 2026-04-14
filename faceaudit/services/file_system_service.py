import os
import shutil
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict, Callable, Union

from faceaudit.services.hash_service import HashService
from faceaudit.services.config_loader_service import ConfigLoaderService
from faceaudit.enums.project_folder import ProjectFolder
from faceaudit.enums.project_file_type import ProjectFileType
from faceaudit.models.project_config import ProjectConfig

class FileSystemService:
    """
    Serviço de Infraestrutura e Orquestração de I/O.
    
    Responsável por gerenciar a estrutura física de um "Projeto Autocontido" (Pendrive Paradigm).
    Garante que os arquivos sejam gravados nos locais corretos e atua em conjunto 
    com o HashService para manter a cadeia de custódia.
    """

    def __init__(self, hash_service: HashService, config_loader: ConfigLoaderService):
        self.hash_service = hash_service
        self.config_loader = config_loader
        
        # Guarda o estado do projeto atual ativo no sistema
        self.current_project_root: Optional[Path] = None

    def set_project_root(self, root_path: Union[str, Path]) -> None:
        """Define a raiz do projeto atual para todas as operações subsequentes."""
        self.current_project_root = Path(root_path)

    # =========================================================================
    # 1. ESTRUTURA DO PROJETO
    # =========================================================================

    def create_project_structure(self, root_path: Path, logger_callback: Optional[Callable[[str], None]] = None) -> None:
        """Gera a árvore de diretórios do projeto baseada no Enum ProjectFolder."""
        self.set_project_root(root_path)
        
        if logger_callback: 
            logger_callback(f"📁 Construindo árvore de diretórios em: {root_path.name}...")
            
        root_path.mkdir(parents=True, exist_ok=True)
        
        for folder in ProjectFolder:
            if folder.value:  # Ignora a raiz ("")
                folder_path = root_path / folder.value
                folder_path.mkdir(parents=True, exist_ok=True)
                
        # Garante a pasta de imagens para o Deep Copy
        (root_path / "images").mkdir(exist_ok=True)

    # =========================================================================
    # 2. PROJETO (O ARQUIVO .FAP)
    # =========================================================================

    def save_project_config(self, config: ProjectConfig, logger_callback: Optional[Callable[[str], None]] = None) -> Tuple[Path, str]:
        """Salva o manifesto do projeto (.fap) na raiz atual e retorna seu Path e Hash."""
        if not self.current_project_root:
            raise RuntimeError("Raiz do projeto não definida no FileSystemService.")
            
        file_name = f"{config.name}.fap"
        file_path = self.current_project_root / file_name
        
        if logger_callback: 
            logger_callback(f"📝 Salvando manifesto do projeto: {file_name}")
        
        config.to_json(file_path)
        
        if logger_callback: 
            logger_callback("🔐 Calculando assinatura do projeto (Hash)...")
            
        file_hash = self.hash_service.calculate_file_hash(file_path)
        return file_path, file_hash

    def load_project_config(self, file_path: Path, logger_callback: Optional[Callable[[str], None]] = None) -> ProjectConfig:
        """Carrega o manifesto .fap do disco e define a raiz do projeto automaticamente."""
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo de projeto não encontrado: {file_path}")
            
        # O diretório onde o .fap está é a raiz do projeto
        self.set_project_root(file_path.parent)
            
        if logger_callback:
            logger_callback(f"📂 Lendo manifesto: {file_path.name}")
            
        return ProjectConfig.from_json(file_path)

    # =========================================================================
    # 3. O ORÁCULO DE CAMINHOS (LAZY RESOLUTION)
    # =========================================================================

    def resolve_file(self, file_name: str, file_type: ProjectFileType) -> Optional[Path]:
        """
        Procura um arquivo PRIMEIRO no projeto local (Pendrive).
        Se não encontrar, faz o fallback para a pasta global (Máquina do Perito).
        """
        if not file_name or file_name == "Selecione...":
            return None
            
        # Limpa o nome caso venha da UI com metadados, ex: "BFW (2000 imgs)"
        clean_name = file_name.split(" (")[0]
        
        kwargs = {
            "dataset_name": clean_name,
            "pipeline_name": clean_name,
            "strategy_id": clean_name,
            "config_name": clean_name
        }
        
        # 1. TENTA LOCAL
        if self.current_project_root:
            relative_path = file_type.build_path(**kwargs)
            local_path = self.current_project_root / relative_path
            
            if local_path.exists():
                return local_path

        # 2. FALLBACK GLOBAL
        app_config = self.config_loader.load()
        
        if file_type == ProjectFileType.DATASET_METADATA:
            global_dir = Path(app_config.paths.datasets)
            p_parquet = global_dir / f"{clean_name}.parquet"
            if p_parquet.exists(): return p_parquet
            p_pkl = global_dir / f"{clean_name}.pkl"
            if p_pkl.exists(): return p_pkl
            
        elif file_type == ProjectFileType.PIPELINE_CONFIG:
            global_dir = Path(app_config.paths.pipelines)
            p_json = global_dir / f"{clean_name}.json"
            if p_json.exists(): return p_json
            
        elif file_type == ProjectFileType.MINING_CONFIG:
            global_dir = Path(app_config.paths.mining_configs)
            p_json = global_dir / f"{clean_name}.json"
            if p_json.exists(): return p_json

        # --- [NOVO] Fallback Global para Split Configs ---
        elif file_type == ProjectFileType.SPLIT_CONFIG:
            global_dir = Path(app_config.paths.split_configs)
            p_json = global_dir / f"{clean_name}.json"
            if p_json.exists(): return p_json
            
        elif file_type in [ProjectFileType.TRAINED_STRATEGY]:
            global_dir = Path(app_config.paths.strategies)
            p_model = global_dir / f"{clean_name}{file_type.extension}"
            if p_model.exists(): return p_model

        return None

    # =========================================================================
    # 4. SALVAMENTO E CÓPIAS (SHALLOW COPY / LAZY COPY)
    # =========================================================================

    def save_global_file(self, file_type: ProjectFileType, data: Any, logger_callback: Optional[Callable[[str], None]] = None, **kwargs) -> Path:
        """
        Salva um arquivo (Dataset, Pipeline, Mining Rule, Split Rule) na respectiva 
        pasta GLOBAL do sistema, ignorando a raiz do projeto local.
        """
        app_config = self.config_loader.load()
        
        # Descobre qual é a pasta global baseada no tipo do arquivo
        global_dir = None
        if file_type == ProjectFileType.DATASET_METADATA:
            global_dir = Path(app_config.paths.datasets)
        elif file_type == ProjectFileType.PIPELINE_CONFIG:
            global_dir = Path(app_config.paths.pipelines)
        elif file_type == ProjectFileType.MINING_CONFIG:
            global_dir = Path(app_config.paths.mining_configs)
        # --- [NOVO] ---
        elif file_type == ProjectFileType.SPLIT_CONFIG:
            global_dir = Path(app_config.paths.split_configs)
        else:
            raise ValueError(f"O tipo de arquivo {file_type.name} não possui um diretório global mapeado.")
            
        global_dir.mkdir(parents=True, exist_ok=True)
        
        # Pede para o Enum montar apenas o nome do arquivo, ignorando a pasta local
        try:
            file_name = file_type.name_template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Falta a variável {e} para montar o nome do arquivo {file_type.name}.")
            
        if not file_name.endswith(file_type.extension):
            file_name = f"{file_name}{file_type.extension}"
            
        abs_path = global_dir / file_name
        
        if logger_callback:
            logger_callback(f"🌍 Salvando {file_type.name} no Repositório Global: {file_name}...")

        # Grava os dados dependendo do tipo
        if file_type.extension == ".json":
            payload = data.to_dict() if hasattr(data, 'to_dict') else data
            with open(abs_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
                
        elif file_type.extension in [".pkl", ".ckpt"]:
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            raise NotImplementedError(f"Serialização não implementada para: {file_type.extension}")

        return abs_path

    def save_file(self, file_type: ProjectFileType, data: Any, logger_callback: Optional[Callable[[str], None]] = None, **kwargs) -> Tuple[Path, str]:
        """Salva um objeto no projeto atual delegando a regra de nome ao Enum."""
        if not self.current_project_root:
            raise RuntimeError("Raiz do projeto não definida para salvar arquivo.")
            
        relative_path = file_type.build_path(**kwargs)
        abs_path = self.current_project_root / relative_path
        
        if logger_callback:
            logger_callback(f"💾 Salvando {file_type.name}: {relative_path.name}...")
        
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        if file_type.extension == ".json":
            payload = data.to_dict() if hasattr(data, 'to_dict') else data
            with open(abs_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=4, ensure_ascii=False)
                
        elif file_type.extension in [".pkl", ".ckpt"]:
            with open(abs_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            raise NotImplementedError(f"Serialização não implementada para: {file_type.extension}")

        file_hash = self.hash_service.calculate_file_hash(abs_path)
        return abs_path, file_hash

    def copy_file_to_project(self, src_path: Path, file_type: ProjectFileType, logger_callback: Optional[Callable[[str], None]] = None, **kwargs) -> Tuple[Path, str]:
        """Cópia "Shallow": Copia apenas o metadata/parquet do global para o local."""
        if not self.current_project_root:
            raise RuntimeError("Raiz do projeto não definida para cópia.")
            
        if not src_path.exists():
            raise FileNotFoundError(f"Arquivo origem não encontrado: {src_path}")

        relative_path = file_type.build_path(**kwargs)
        abs_dest_path = self.current_project_root / relative_path
        
        if src_path == abs_dest_path:
            return abs_dest_path, self.hash_service.calculate_file_hash(abs_dest_path)
        
        if logger_callback:
            tamanho_mb = src_path.stat().st_size / (1024 * 1024)
            logger_callback(f"📦 Copiando base {src_path.name} ({tamanho_mb:.1f} MB) para o projeto...")
        
        abs_dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, abs_dest_path)
        
        file_hash = self.hash_service.calculate_file_hash(abs_dest_path)
        return abs_dest_path, file_hash

    # =========================================================================
    # 5. EMPACOTAMENTO DE IMAGENS (DEEP COPY)
    # =========================================================================

    def pack_dataset_images(self, dataset_name: str, logger_callback: Optional[Callable[[str], None]] = None) -> Optional[Tuple[Path, str]]:
        """
        Cópia "Deep": Lê o Parquet local do projeto, copia fisicamente todas as fotografias 
        referenciadas nele para uma subpasta local, e reescreve o Parquet com os novos caminhos.
        Garante portabilidade total do projeto para análise em outros computadores.
        """
        if not self.current_project_root:
            raise RuntimeError("Raiz do projeto não definida.")

        clean_name = dataset_name.split(" (")[0]
        dataset_path = self.resolve_file(clean_name, ProjectFileType.DATASET_METADATA)
        
        if not dataset_path or not dataset_path.exists():
            if logger_callback: logger_callback(f"⚠️ Dataset '{clean_name}' não encontrado para empacotar imagens.")
            return None
            
        if logger_callback: logger_callback(f"📸 Inspecionando imagens do dataset '{clean_name}'...")
        
        try:
            df = pd.read_parquet(dataset_path)
            if 'image_path' not in df.columns:
                if logger_callback: logger_callback(f"   ↳ Dataset não possui coluna de imagens.")
                return dataset_path, self.hash_service.calculate_file_hash(dataset_path)

            # Pasta destino: /MeuProjeto/images/NomeDoDataset/
            images_dir = self.current_project_root / "images" / clean_name
            images_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            skipped = 0
            new_paths = []
            
            for path_str in df['image_path']:
                if pd.isna(path_str) or not path_str:
                    new_paths.append(path_str)
                    continue
                    
                orig_path = Path(path_str)
                
                # Se a imagem já está dentro da pasta do projeto atual, pula a cópia
                if str(self.current_project_root) in str(orig_path):
                    new_paths.append(path_str)
                    skipped += 1
                    continue
                    
                if not orig_path.exists():
                    new_paths.append(path_str) # Mantém o caminho quebrado original para registro
                    continue

                # Evita sobrescrever se houverem duas imagens com o mesmo nome em pastas distintas
                dest_path = images_dir / orig_path.name
                counter = 1
                while dest_path.exists() and dest_path.stat().st_size != orig_path.stat().st_size:
                    dest_path = images_dir / f"{orig_path.stem}_{counter}{orig_path.suffix}"
                    counter += 1
                    
                if not dest_path.exists():
                    shutil.copy2(orig_path, dest_path)
                    copied += 1
                else:
                    skipped += 1
                    
                # Grava o caminho absoluto local (Garante que a UI atual continue funcionando)
                new_paths.append(str(dest_path))

            # Atualiza o DataFrame com os novos caminhos locais e sobrescreve o arquivo
            df['image_path'] = new_paths
            df.to_parquet(dataset_path)
            
            # Recalcula o Hash do Parquet (já que ele foi alterado)
            new_hash = self.hash_service.calculate_file_hash(dataset_path)
            
            if logger_callback: 
                logger_callback(f"   ↳ Concluído: {copied} fotos isoladas no projeto ({skipped} já existiam).")
                
            return dataset_path, new_hash
            
        except Exception as e:
            if logger_callback: logger_callback(f"❌ Erro crítico ao empacotar imagens: {e}")
            return None
    
    def list_available_files(self, file_type: ProjectFileType) -> Tuple[List[str], List[str]]:
        """
        Varre o HD e lista todos os arquivos disponíveis de um determinado tipo.
        Devolve duas listas: (Exclusivos Globais, Locais do Projeto).
        """
        global_files, local_files = set(), set()
        
        # 1. TENTA GLOBAL
        app_config = self.config_loader.load()
        
        # Puxa o caminho global dependendo do tipo solicitado
        global_dir = None
        if file_type == ProjectFileType.DATASET_METADATA:
            global_dir = Path(app_config.paths.datasets)
        elif file_type == ProjectFileType.PIPELINE_CONFIG:
            global_dir = Path(app_config.paths.pipelines)
        elif file_type == ProjectFileType.MINING_CONFIG:
            global_dir = Path(app_config.paths.mining_configs)
        # --- [NOVO] Adicionado diretório global de split configs ---
        elif file_type == ProjectFileType.SPLIT_CONFIG:
            global_dir = Path(app_config.paths.split_configs)
            
        if global_dir and global_dir.exists():
            for p in global_dir.glob(f"*{file_type.extension}"):
                global_files.add(p.stem)
            # Legado do dataset
            if file_type == ProjectFileType.DATASET_METADATA:
                for p in global_dir.glob("*.pkl"):
                    global_files.add(p.stem)

        # 2. TENTA LOCAL
        if self.current_project_root:
            local_dir = self.current_project_root / file_type.folder.value
            if local_dir.exists():
                for p in local_dir.glob(f"*{file_type.extension}"):
                    local_files.add(p.stem)
                # Legado do dataset local
                if file_type == ProjectFileType.DATASET_METADATA:
                    for p in local_dir.glob("*.pkl"):
                        local_files.add(p.stem)

        # 3. FILTRA DUPLICADOS (Prioriza o que está no Pendrive)
        global_exclusive = list(global_files - local_files)
        
        return sorted(global_exclusive), sorted(list(local_files))