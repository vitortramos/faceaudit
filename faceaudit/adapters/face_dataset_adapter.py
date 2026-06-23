import os
from pathlib import Path
from typing import Callable, Optional, Union
import pandas as pd
import numpy as np

from faceaudit.models import FaceDataset, FaceInfo, FFInfo, OFIQInfo, SERFIQInfo
from faceaudit.models.dir_template import BUILTIN_COLUMNS, IGNORE_TOKEN, DirTemplate, split_name
from faceaudit.services.hash_service import HashService

class FaceDatasetAdapter:
    @staticmethod
    def dataset_from_dir(
        dir_path: Union[str, Path], 
        name: Optional[str] = None, 
        recursive: bool = False,
        template: Optional[DirTemplate] = None,
        default_identity: str = "Unknown",
        origin: str = "Local",
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'),
        callback: Optional[Callable[[str, int], None]] = None
    ) -> FaceDataset:
        
        path_obj = Path(dir_path).resolve()
        if not path_obj.exists(): 
            raise FileNotFoundError(f"Diretório inexistente: {path_obj}")

        dataset = FaceDataset(name=name if name else path_obj.name)
        files = list(path_obj.rglob("*") if recursive else path_obj.iterdir())
        image_files = [f for f in files if f.is_file() and f.suffix.lower() in extensions]
        total_images = len(image_files)
        
        for i, file_path in enumerate(image_files):
            face = FaceInfo(image_path=str(file_path.resolve()))
            
            # 1. Atribuições Base (Garante que a Origem definida na UI seja aplicada)
            face.identity = default_identity
            face.origin = origin 
            
            # 2. Aplica o Template de Extração Inteligente
            if template:
                FaceDatasetAdapter._apply_template(face, file_path, path_obj, template)
            
            # 3. Fallback de Segurança (Mudamos de ELIF para IF)
            # Só assume a pasta pai se o template falhou em extrair a identidade
            if recursive and face.identity == "Unknown":
                face.identity = file_path.parent.name
            
            dataset.add(face)

            if callback and i % 50 == 0:
                percent = int(((i + 1) / total_images) * 100)
                callback(f"Lendo: {file_path.name}", percent)

        if len(dataset) > 0:
            if callback: callback("Calculando integridade (Hashes)...", 95)
            dataset = HashService().compute_dataset_hash(dataset)
        
        if callback: callback(f"Concluído: {total_images} imagens carregadas.", 100)
            
        return dataset

    @staticmethod
    def _assign_column(face: FaceInfo, attr: str, val: str, template: DirTemplate) -> None:
        """Roteia o valor extraído para o destino correto:
          - Built-in (identity/gender/ethnicity/origin): atributo direto de FaceInfo.
          - Custom column declarada no template: face.extra_data[col_name].
          - IGNORE_TOKEN ou desconhecido: descarta silenciosamente.
        """
        if not attr:
            return
        attr_clean = attr.lower().strip()
        if not attr_clean or attr_clean == IGNORE_TOKEN:
            return

        if attr_clean in BUILTIN_COLUMNS:
            setattr(face, attr_clean, val)
            return

        if attr_clean in template.custom_columns:
            face.extra_data[attr_clean] = val
            return

        # Coluna referenciada no map mas que não existe no universo
        # declarado: template inconsistente. Não levantamos erro pra
        # não quebrar ingestões em massa — só pulamos. UI bloqueia
        # remoção justamente pra evitar esse estado.

    @staticmethod
    def _apply_template(face: FaceInfo, file_path: Path, root_path: Path, template: DirTemplate):
        """Aplica a lógica de extração sem silenciar erros críticos."""
        try:
            relative_path = file_path.relative_to(root_path)
            folder_parts = relative_path.parent.parts
        except ValueError:
            folder_parts = []

        # 1. Mapeamento de Pastas
        for level_key, attributes in template.folder_map.items():
            level_idx = int(level_key)
            if level_idx < len(folder_parts):
                folder_name = folder_parts[level_idx]

                values = split_name(
                    folder_name,
                    template.folder_separator,
                    template.use_folder_separator,
                    getattr(template, "folder_split_from_right", False),
                    getattr(template, "folder_max_splits", 0),
                )

                for i, attr in enumerate(attributes):
                    val = values[i] if i < len(values) else values[0]
                    FaceDatasetAdapter._assign_column(face, attr, val, template)

        # 2. Mapeamento de Arquivos
        filename_stem = file_path.stem
        file_parts = split_name(
            filename_stem,
            template.filename_separator,
            template.use_filename_separator,
            getattr(template, "filename_split_from_right", False),
            getattr(template, "filename_max_splits", 0),
        )

        for idx_key, attr in template.filename_map.items():
            idx_int = int(idx_key)
            val = file_parts[idx_int] if idx_int < len(file_parts) else file_parts[0]
            FaceDatasetAdapter._assign_column(face, attr, val, template)

        # 3. Força a pasta pai se configurado
        if getattr(template, "identity_from_parent", False):
            face.identity = file_path.parent.name

    @staticmethod
    def dataset_from_old_parquet(
        file_path: Union[str, Path],
        name: Optional[str] = None
    ) -> FaceDataset:
        """
        Realiza a migração de dados de arquivos .parquet do sistema legado.
        """
        path_obj = Path(file_path).resolve()
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo Parquet não encontrado: {path_obj}")

        try:
            df = pd.read_parquet(path_obj)
        except Exception as e:
            raise ValueError(f"Erro ao ler arquivo Parquet: {e}")

        dataset_name = name if name else path_obj.stem
        dataset = FaceDataset(
            name=dataset_name,
            description=f"Importado de Parquet Legado: {path_obj}"
        )

        print(f"Importando {len(df)} registros do Parquet Legado...")

        for _, row in df.iterrows():
            caminho_img = row.get('caminho_completo_imagem')
            if not caminho_img:
                continue

            # 1. Popula FaceInfo (Metadados Básicos)
            face = FaceInfo(
                image_path=str(caminho_img),
                identity=str(row['identidade']) if 'identidade' in row else row.get('identity'),
                ethnicity=str(row['etnia']) if 'etnia' in row else row.get('ethnicity'),
                gender=str(row['sexo']) if 'sexo' in row else row.get('gender'),
                hash=row.get('hash'),
                file_name=row.get('file_name')
            )
            
            # 2. Popula FFInfo
            has_ff_data = any(k in row for k in ['embedding', 'fiqa_score', 'bbox'])
            
            if has_ff_data:
                emb_array = None
                if 'embedding' in row and row['embedding'] is not None:
                    raw_emb = row['embedding']
                    if isinstance(raw_emb, (list, np.ndarray)):
                        emb_array = np.array(raw_emb, dtype=np.float32)

                ff_info = FFInfo(
                    embedding_sepaelv2=emb_array,
                    fiqa_score=row.get('fiqa_score'),
                    det_score=row.get('det_score'),
                    yaw=row.get('yaw'),
                    pitch=row.get('pitch'),
                    roll=row.get('roll'),
                    ipd=row.get('ipd'),
                    bbox=row.get('bbox'),
                    age=row.get('age'),
                    gender=row.get('gender') if 'gender' in row else None
                )
                face.ff_info = ff_info

            # 3. Popula OFIQInfo
            face.ofiq_info = OFIQInfo.from_dict(row.to_dict())

            # 4. Popula SERFIQInfo
            if 'SERFIQ-T10' in row and pd.notna(row['SERFIQ-T10']):
                face.serfiq_info = SERFIQInfo(
                    score=float(row['SERFIQ-T10']),
                    t=10, 
                    alpha=1.5 
                )

            dataset.add(face)

        return dataset