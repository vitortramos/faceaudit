import os
import shutil
import tempfile
import subprocess
import csv
import platform
from typing import List, Dict, Optional, Callable
from tqdm.auto import tqdm

# ATUALIZAÇÃO: Adicionado FaceDataset aos imports
from faceaudit.models import FaceInfo, OFIQInfo, FaceDataset
from faceaudit.services.ofiq_provenance_service import (
    detect_ofiq_version,
    models_fingerprint,
)

# Modo de chamada gravado em `OFIQInfo.backend`.
BACKEND_EXECUTABLE = "executable"

# O OFIQSampleApp (medido na 1.0.3) escolhe os arquivos da pasta pela
# extensão e só aceita estas três, em minúsculas: `.JPG`, `.jfif`, `.jpe`,
# `.bmp`, `.webp` e `.tif` são pulados sem aviso, e a face fica sem OFIQ.
# O conteúdo não importa para ele — um JPEG chamado `.jfif` passa quando
# chamado `.jpg`.
_EXTENSOES_JPEG = frozenset({".jpg", ".jpeg", ".jpe", ".jfif", ".jfi", ".jif"})


def extensao_para_ofiq(caminho: str):
    """Extensão com que a cópia temporária vai ao OFIQ, e se precisa converter.

    Família JPEG vai como `.jpg` (só o nome muda; os bytes são os mesmos).
    PNG vai como `.png`. O resto é convertido para PNG, que é sem perda:
    renomear um BMP para `.png` funcionaria por acaso de quem decodifica,
    e converter não depende disso.
    """
    ext = os.path.splitext(caminho)[1].lower()
    if ext in _EXTENSOES_JPEG or not ext:
        return ".jpg", False
    if ext == ".png":
        return ".png", False
    return ".png", True


def _gravar_png_sem_perda(origem: str, destino: str) -> None:
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(origem, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"imagem ilegível: {origem}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"falha ao converter para PNG: {origem}")
    buf.tofile(destino)

class OFIQService:
    """
    Serviço wrapper para execução eficiente da biblioteca OFIQ (Open Source Face Image Quality).

    Diferente de bibliotecas Python nativas, o OFIQ roda como um executável C++ externo.
    Para evitar o overhead de iniciar um processo para cada imagem, este serviço gerencia
    uma estratégia de "Batch Processing" com diretórios temporários.

    Workflow:
    1. Agrupa imagens em lotes (chunks).
    2. Copia imagens para uma pasta temporária (com nomes normalizados).
    3. Executa o binário OFIQ apenas uma vez para toda a pasta.
    4. Faz o parse do CSV de saída.
    5. Mapeia os resultados de volta para os objetos FaceInfo originais.
    6. Limpa os arquivos temporários.
    """

    def __init__(self, ofiq_exe_path: Optional[str] = None, ofiq_data_path: Optional[str] = None):
        """
        Inicializa o serviço validando a presença do binário e dos dados de configuração.

        Args:
            ofiq_exe_path (Optional[str], optional): 
                Caminho absoluto para o executável `OFIQSampleApp`. 
                Se None, tenta encontrar automaticamente no PATH ou locais padrão.
            
            ofiq_data_path (Optional[str], optional): 
                Caminho para a pasta `data` contendo os modelos do OFIQ.
                Se None, tenta inferir baseado na localização do executável.

        Raises:
            FileNotFoundError: Se o executável ou a pasta de dados não forem encontrados.
        """
        self.exe_path = self.resolve_exe_path(ofiq_exe_path)
        self.config_path = self.resolve_data_path(ofiq_data_path)
        
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"Executável OFIQ não encontrado em: {self.exe_path}")
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Pasta 'data' do OFIQ não encontrada em: {self.config_path}")

        # Vai para cada face processada. None quando o executável não está
        # numa instalação com `Version.txt` (ex.: achado solto no PATH).
        self.version: Optional[str] = detect_ofiq_version(self.exe_path)
        self.backend: str = BACKEND_EXECUTABLE
        self._models_sha256: Optional[str] = None

    def provenance(self) -> Dict[str, Optional[str]]:
        """Versão, modo e impressão digital dos modelos, para a cadeia de custódia.

        A impressão digital lê os ~420 MB de modelos uma vez por sessão
        (cerca de 1 s) e fica guardada.
        """
        if self._models_sha256 is None:
            self._models_sha256 = models_fingerprint(self.config_path)
        return {
            "ofiq_version": self.version,
            "ofiq_backend": self.backend,
            "ofiq_models_sha256": self._models_sha256,
        }

    def process_face(
        self,
        face: FaceInfo,
        *,
        dataset: Optional[FaceDataset] = None,
    ) -> FaceInfo:
        """
        Processa uma única face isoladamente.

        Atenção: Este método é ineficiente para grandes volumes, pois cria um
        ambiente temporário inteiro para apenas um arquivo. Prefira `process_dataset`.

        Args:
            face (FaceInfo): Objeto a ser processado.
            dataset: PR L1b — propaga pra `process_batch` resolver paths.

        Returns:
            FaceInfo: Objeto enriquecido com métricas OFIQ.
        """
        results = self.process_batch([face], dataset=dataset)
        return results[0]

    def process_batch(
        self,
        faces: List[FaceInfo],
        *,
        dataset: Optional[FaceDataset] = None,
    ) -> List[FaceInfo]:
        """
        Executa o pipeline OFIQ para uma lista de faces em uma única operação de I/O.

        Args:
            faces (List[FaceInfo]): Lista de objetos a serem processados.
            dataset: PR L1b — quando passado, resolve `image_path` de cada
                face via `dataset.resolve_image_path` (suporta paths
                relativos). None → usa `face.image_path` direto (compat).

        Returns:
            List[FaceInfo]: A mesma lista, com os objetos modificados in-place.
        """
        if not faces:
            return []

        # 1. Cria diretório temporário para este lote
        dir_temp = tempfile.mkdtemp(prefix="ofiq_batch_")
        arquivo_csv_out = os.path.join(dir_temp, "output.csv")
        
        # Mapeamento: NomeArquivoTemp -> Objeto FaceInfo original
        mapa_temp_face = {}
        
        try:
            # 2. Copia imagens para o temp com nomes seguros
            imagens_validas = []
            
            for idx, face in enumerate(faces):
                # PR L1b — resolve o path uma vez por face. Quando `dataset`
                # é None mantém o comportamento legado (face.image_path).
                src_path = (
                    str(dataset.resolve_image_path(face))
                    if dataset is not None
                    else face.image_path
                )
                if not os.path.exists(src_path):
                    continue

                # Nome simples para evitar problemas de encoding no C++, e
                # extensão que o OFIQSampleApp aceita (ver
                # `extensao_para_ofiq`): ele escolhe os arquivos da pasta
                # pela extensão e pula os outros sem avisar.
                ext_ofiq, converter = extensao_para_ofiq(src_path)
                nome_temp = f"img_{idx:05d}{ext_ofiq}"
                dest_path = os.path.join(dir_temp, nome_temp)

                try:
                    if converter:
                        _gravar_png_sem_perda(src_path, dest_path)
                    else:
                        shutil.copy2(src_path, dest_path)
                    imagens_validas.append(nome_temp)
                    mapa_temp_face[nome_temp] = face
                except Exception as e:
                    print(f"Erro ao copiar {src_path}: {e}")

            if not imagens_validas:
                return faces

            # 3. Executa o OFIQ
            # Comando: OFIQSampleApp -c <data> -i <input_dir> -o <output.csv>
            cwd = os.path.dirname(self.exe_path) if os.path.isabs(self.exe_path) else None
            cmd = [self.exe_path, '-c', self.config_path, '-i', dir_temp, '-o', arquivo_csv_out]
            
            # Executa silenciosamente (Cross-Platform)
            subprocess.run(
                cmd, 
                cwd=cwd, 
                capture_output=True, 
                check=False,
                text=True, # Usa o encoding padrão do SO automaticamente (UTF-8 no Linux, CP1252/850 no Windows)
                errors='replace'
            )

            # 4. Lê o CSV e popula os objetos
            if os.path.exists(arquivo_csv_out):
                with open(arquivo_csv_out, 'r', encoding='utf-8', errors='replace') as f:
                    # Detecta delimitador (; ou ,)
                    sample = f.read(1024)
                    f.seek(0)
                    delimiter = ';' if ';' in sample else ','
                    
                    reader = csv.DictReader(f, delimiter=delimiter)
                    
                    for row in reader:
                        # O OFIQ retorna o caminho completo ou filename
                        # Pegamos o basename para bater com nossa chave
                        fname = os.path.basename(row.get('Filename', ''))
                        
                        face_original = mapa_temp_face.get(fname)
                        if face_original:
                            # Converte o dict cru do CSV para nosso objeto OFIQInfo
                            info = OFIQInfo.from_dict(row)
                            info.version = self.version
                            info.backend = self.backend
                            face_original.ofiq_info = info

        except Exception as e:
            print(f"Erro crítico no lote OFIQ: {e}")
            
        finally:
            # 5. Limpeza (Sempre remove o diretório temporário)
            shutil.rmtree(dir_temp, ignore_errors=True)
            
        return faces

    def process_dataset(self, 
                        dataset: FaceDataset, 
                        batch_size: int = 50,
                        callback: Optional[Callable[[], None]] = None) -> FaceDataset:
        """
        Processa um FaceDataset completo em chunks otimizados.

        Divide o dataset em lotes de tamanho `batch_size`, processa cada lote
        em um diretório temporário isolado e consolida os resultados.
        Isso evita sobrecarregar o sistema de arquivos ao lidar com milhares de imagens.

        Args:
            dataset (FaceDataset): O dataset contendo as faces.
            batch_size (int, optional): 
                Número de imagens a processar por vez (por chamada do executável).
                Defaults to 50.
            callback (Optional[Callable[[], None]], optional): 
                Função de callback chamada UMA VEZ PARA CADA IMAGEM processada.
                Mesmo processando em lote de 50, o callback será disparado 50 vezes
                em sequência para garantir que a barra de progresso externa flua corretamente.

        Returns:
            FaceDataset: O dataset enriquecido.
        """
        print(f"Iniciando processamento OFIQ do dataset: {dataset.name} ({len(dataset)} imagens)")
        
        # Acessa a lista de faces
        all_faces = dataset.faces
        total = len(all_faces)
        
        if total == 0:
            return dataset

        # Barra de progresso gerenciando o total de imagens
        with tqdm(total=total, desc="OFIQ Processing", unit="img") as pbar:
            # Itera de batch_size em batch_size
            for i in range(0, total, batch_size):
                # Pega uma fatia da lista (chunk)
                chunk = all_faces[i : i + batch_size]
                
                # Processa esse lote (copia, executa, lê CSV, limpa)
                # PR L1b — propaga `dataset` pra `process_batch` resolver paths.
                self.process_batch(chunk, dataset=dataset)
                
                # Atualiza a barra de progresso do TQDM
                chunk_len = len(chunk)
                pbar.update(chunk_len)
                
                # Atualiza o Callback Externo (GUI)
                # Disparamos N vezes para compensar o processamento em lote
                if callback:
                    for _ in range(chunk_len):
                        callback()
        
        return dataset

    # --- Helpers de Resolução de Caminho (Cross-Platform) ---
    #
    # Públicos e estáticos de propósito: a tela de configuração precisa
    # mostrar ONDE o OFIQ será procurado quando o campo está vazio. Sem
    # isso, "campo vazio" parecia defeito, quando significa "resolver
    # automaticamente" — e a UI teria de duplicar a regra para informar.
    @staticmethod
    def resolve_exe_path(path):
        """Tenta localizar o binário OFIQSampleApp no sistema (Windows/Linux)."""
        if path: return path
        
        is_windows = platform.system() == "Windows"
        exe_name = "OFIQSampleApp.exe" if is_windows else "OFIQSampleApp"
        
        found = shutil.which(exe_name)
        if found: return found
        
        if is_windows:
            defaults = [
                r"C:\OFIQ-Project\install_x86_64\Release\bin\OFIQSampleApp.exe",
                r"C:\Program Files\OFIQ\bin\OFIQSampleApp.exe"
            ]
        else:
            defaults = [
                "/usr/local/bin/OFIQSampleApp",
                "/opt/OFIQ/bin/OFIQSampleApp",
                "/usr/bin/OFIQSampleApp"
            ]
            
        for d in defaults:
            if os.path.exists(d): return d
            
        return exe_name

    @staticmethod
    def resolve_data_path(path):
        """Tenta localizar a pasta de dados do OFIQ (Windows/Linux)."""
        if path: return path
        env = os.environ.get("OFIQ_DATA")
        if env: return env
        
        is_windows = platform.system() == "Windows"
        
        if is_windows:
            defaults = [
                r"C:\OFIQ-Project\data",
                r"C:\Program Files\OFIQ\data"
            ]
        else:
            defaults = [
                "/usr/local/share/OFIQ/data",
                "/opt/OFIQ/data",
                "/usr/share/OFIQ/data"
            ]
            
        for d in defaults:
            if os.path.exists(d): return d
            
        return "data"

    # Aliases mantidos para chamadores antigos (o nome privado circulou).
    _resolve_exe = resolve_exe_path
    _resolve_data = resolve_data_path