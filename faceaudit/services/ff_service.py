import numpy as np
import cv2
from typing import List, Optional, Callable
from tqdm.auto import tqdm

# Import absoluto baseada na estrutura do pacote
from faceaudit.models import FaceInfo, FFInfo, FaceDataset

try:
    from forensicface.app import ForensicFace
except ImportError:
    ForensicFace = None 

class FFService:
    """
    Serviço de interface para a biblioteca ForensicFace.

    Responsável por realizar a detecção facial, alinhamento, extração de landmarks
    e cálculo de vetores biométricos (embeddings) utilizando modelos de aprendizado
    profundo (Deep Learning).
    """

    def __init__(self, model: str = 'sepaelv2', det_size: int = 320, use_gpu: bool = False):
        if ForensicFace is None:
            raise ImportError("A biblioteca 'forensicface' não foi encontrada.")
        
        self.model_name = model.lower()
        self.app = ForensicFace(
            model=model,
            det_size=det_size,
            use_gpu=use_gpu,
            extended=True
        )

    def get_embedding_from_array(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrai o vetor biométrico (embedding) de uma imagem em memória (numpy array).
        """
        try:
            res = self.app.process_image(img_array)
            if res and 'embedding' in res:
                return res['embedding']
        except Exception:
            pass
        return None

    def process_face(self, face_info: FaceInfo) -> FaceInfo:
        """
        Processa uma única face, preenchendo o objeto FaceInfo com dados biométricos.
        Os resultados técnicos (predições) são armazenados APENAS em `.ff_info`.
        """
        img_array = self._safe_read_image(face_info.image_path)
        
        if img_array is None:
            return face_info

        try:
            resultado = self.app.process_image(img_array)
            
            if not resultado:
                return face_info

            ff_data = FFInfo(
                ipd=resultado.get('ipd'),
                fiqa_score=resultado.get('fiqa_score'),
                det_score=resultado.get('det_score'),
                gender=resultado.get('gender'), # Predição bruta
                age=resultado.get('age'),       # Predição bruta
                yaw=resultado.get('yaw'),
                pitch=resultado.get('pitch'),
                roll=resultado.get('roll'),
                bbox=resultado.get('bbox'),
                keypoints=resultado.get('keypoints'),
                aligned_face=resultado.get('aligned_face')
            )

            emb = resultado.get('embedding')
            if 'sepaelv2' in self.model_name:
                ff_data.embedding_sepaelv2 = emb
            elif 'sepaelv4' in self.model_name:
                ff_data.embedding_sepaelv4 = emb
            else:
                ff_data.embedding_sepaelv2 = emb

            face_info.ff_info = ff_data

        except Exception as e:
            print(f"Error processing face {face_info.image_path}: {e}")
            
        return face_info

    def process_dataset(self, 
                        dataset: FaceDataset, 
                        progress_callback: Optional[Callable[[str, int], None]] = None) -> FaceDataset:
        """
        Processa um dataset completo, gerando biometria para todas as faces.
        Emite relatórios de progresso em % para a interface gráfica.
        """
        total_faces = len(dataset)
        if total_faces == 0:
            return dataset

        print(f"Iniciando processamento biométrico do dataset: {dataset.name}")
        
        # 1. Extração Biométrica Padrão (ocupa até 95% do progresso)
        for idx, face in enumerate(tqdm(dataset, desc=f"Processando {dataset.name}", unit="face")):
            self.process_face(face)
            
            if progress_callback:
                # Calcula o percentual (de 0 a 95)
                pct = int(((idx + 1) / total_faces) * 95)
                progress_callback(f"Extraindo biometria ({idx + 1}/{total_faces})...", pct)
                
        # 2. Pós-Processamento: Consenso de Gênero por Identidade (últimos 5%)
        if progress_callback:
            progress_callback("Calculando consenso demográfico...", 95)
            
        self._apply_identity_gender_consensus(dataset)
        
        if progress_callback:
            progress_callback("Concluído", 100)
            
        return dataset

    def _apply_identity_gender_consensus(self, dataset: FaceDataset):
        """
        Agrupa todas as faces por identidade. Elege um único gênero para a pessoa 
        usando predições ponderadas pelo FIQA Score (Qualidade).
        """
        print("📊 Calculando consenso demográfico por identidade (ponderado por qualidade)...")
        
        faces_by_id = {}
        for face in dataset:
            if face.identity:
                if face.identity not in faces_by_id:
                    faces_by_id[face.identity] = []
                faces_by_id[face.identity].append(face)

        count_updated = 0

        for identity, faces in faces_by_id.items():
            # --- NOVA LÓGICA ROBUSTA DE CHECAGEM ---
            primeira_face = faces[0]
            
            # Pega o valor, garante que é string, remove espaços e deixa em minúsculo
            gender_atual = str(primeira_face.gender).strip().lower() if primeira_face.gender is not None else ""
            
            # Lista de valores que consideramos como "Célula Vazia/Aguardando IA"
            valores_vazios = ["", "unknown", "none", "nan", "null"]
            
            # Se o campo tiver um gênero real (ex: "male", "female"), o humano manda e a IA pula.
            if gender_atual not in valores_vazios:
                continue
            # ---------------------------------------

            votos = {"Male": 0.0, "Female": 0.0}

            for face in faces:
                if face.ff_info and face.ff_info.gender and face.ff_info.fiqa_score is not None:
                    pred = str(face.ff_info.gender).strip().title()
                    peso = float(face.ff_info.fiqa_score)
                    
                    if pred in votos:
                        votos[pred] += peso
                    else:
                        votos[pred] = peso 

            if sum(votos.values()) == 0:
                continue

            vencedor = max(votos, key=votos.get)

            for face in faces:
                face.gender = vencedor
                
            count_updated += 1

        if count_updated > 0:
            print(f"✅ Gênero estimado unificado para {count_updated} identidades.")

    def process_batch(self, faces: List[FaceInfo]) -> List[FaceInfo]:
        """Processa uma lista de faces em sequência (Wrapper para tqdm)."""
        return [self.process_face(f) for f in tqdm(faces, desc="Extracting Features", unit="img")]

    def _safe_read_image(self, path: str) -> Optional[np.ndarray]:
        """Lê imagens do disco lidando corretamente com caracteres especiais no caminho."""
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
    
    def compare(self, image_path_1: str, image_path_2: str) -> Optional[float]:
        """
        Compara duas imagens faciais utilizando o método nativo do ForensicFace.

        Parâmetros
        ----------
        image_path_1 : str
            Caminho da primeira imagem.

        image_path_2 : str
            Caminho da segunda imagem.

        Retorno
        -------
        Optional[float]
            Escore de similaridade entre os embeddings das faces.
            Retorna None caso a comparação falhe.
        """
        try:
            score = self.app.compare(image_path_1, image_path_2)

            if score is None:
                return None

            return float(score)

        except Exception as e:
            print(f"Error comparing faces: {image_path_1} vs {image_path_2} -> {e}")
            return None