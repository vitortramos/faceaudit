# faceaudit/services/hash_service.py
import hashlib
from typing import Optional, Callable, Union
from pathlib import Path
from tqdm.auto import tqdm
from faceaudit.models import FaceInfo, FaceDataset

class HashService:
    """
    Serviço responsável pela garantia de integridade dos dados via Hash (SHA256).
    
    Este serviço calcula assinaturas digitais únicas para cada imagem individualmente
    e uma assinatura global para o dataset. Isso permite rastrear se houve qualquer
    alteração nos arquivos originais ou corrupção de dados durante o processamento.
    Também utilizado para auditoria de arquivos pesados (Parquet/Pickle) no Workspace.
    """

    def calculate_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Calcula o hash SHA256 de um arquivo de forma eficiente em memória.

        Lê o arquivo em blocos binários de 64kb para evitar o carregamento
        completo de arquivos grandes na memória RAM (Memory Safe).

        Args:
            file_path (Union[str, Path]): Caminho absoluto ou relativo do arquivo.

        Returns:
            Optional[str]: A string hexadecimal representando o hash SHA256 do arquivo,
                           ou None caso o arquivo não exista ou ocorra um erro de leitura.
        """
        try:
            path_obj = Path(file_path)
            if not path_obj.exists() or not path_obj.is_file():
                return None

            # Caminho preferido (Python ≥ 3.11): hashlib.file_digest faz o
            # loop em C, com gestão otimizada de buffer e libera a GIL de
            # forma mais agressiva que o loop Python equivalente.
            if hasattr(hashlib, "file_digest"):
                with open(path_obj, "rb") as f:
                    return hashlib.file_digest(f, "sha256").hexdigest()

            # Fallback para Python < 3.11: loop Python com chunks de 64kb.
            # f.read libera GIL no syscall; sha256.update libera GIL para
            # buffers >= 2048 bytes (caso desses 64kb).
            sha256_hash = hashlib.sha256()
            with open(path_obj, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()

        except Exception as e:
            print(f"Erro ao calcular hash do arquivo {file_path}: {e}")
            return None

    def process_face(
        self,
        face_info: FaceInfo,
        *,
        dataset: Optional[FaceDataset] = None,
    ) -> FaceInfo:
        """
        Processa uma única instância de FaceInfo para calcular seu hash.

        Atualiza o atributo `.hash` do objeto face_info. Em caso de erro
        (ex: arquivo não encontrado), define o hash como None e loga o erro,
        não interrompendo o fluxo.

        Args:
            face_info: face a processar (consome `face_info.image_path`).
            dataset: PR L1b — quando passado, resolve `image_path` via
                `dataset.resolve_image_path` (suporta paths relativos).
                None → usa `face_info.image_path` direto (compat).
        """
        try:
            if face_info.image_path:
                path_to_hash = (
                    str(dataset.resolve_image_path(face_info))
                    if dataset is not None
                    else face_info.image_path
                )
                face_info.hash = self.calculate_file_hash(path_to_hash)
        except Exception as e:
            print(f"Erro ao processar face {face_info.image_path}: {e}")
            face_info.hash = None

        return face_info

    def process_face_dataset(self,
                             dataset: FaceDataset,
                             callback: Optional[Callable[[], None]] = None) -> FaceDataset:
        """
        Itera sobre todas as faces do dataset e força o cálculo do hash para cada uma.
        """
        print(f"Calculando Hashes SHA256 individuais para: {dataset.name}")

        # PR L1b — propaga `dataset` pra resolver paths em modo RELATIVE.
        for face in tqdm(dataset, desc="Hashing Faces", unit="img"):
            self.process_face(face, dataset=dataset)
            if callback:
                callback()

        return dataset

    def compute_dataset_hash(self, 
                             dataset: FaceDataset, 
                             callback: Optional[Callable[[], None]] = None) -> FaceDataset:
        """
        Calcula o hash global de integridade do Dataset (Merkle-like).
        """
        print(f"Verificando integridade e calculando Hash do Dataset: {dataset.name}")
        
        # 1. Garante que todas as faces tenham hash
        faces_to_process = [face for face in dataset if face.hash is None]
        
        if faces_to_process:
            print(f"Encontradas {len(faces_to_process)} faces sem hash. Calculando...")
            for face in tqdm(faces_to_process, desc="Preenchendo Hashes Faltantes"):
                # PR L1b — propaga `dataset` pra resolver paths em modo RELATIVE.
                self.process_face(face, dataset=dataset)
                
                # Aciona o progresso para a GUI
                if callback:
                    callback()
        
        # 2. Concatena os hashes de todas as faces (ordem da lista)
        # Filtra None caso algum arquivo tenha dado erro de leitura
        all_hashes = "".join([face.hash for face in dataset if face.hash is not None])
        
        # 3. Calcula o Hash do Dataset (SHA256 da string concatenada)
        if all_hashes:
            dataset.hash = hashlib.sha256(all_hashes.encode('utf-8')).hexdigest()
            print(f"Hash do Dataset calculado: {dataset.hash}")
        else:
            print("Aviso: Nenhuma face válida com hash para compor o hash do dataset.")
            dataset.hash = None

        return dataset