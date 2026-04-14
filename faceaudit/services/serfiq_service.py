import numpy as np
import cv2
from typing import List, Optional, Callable
from tqdm.auto import tqdm

# Import absoluto baseada na estrutura do pacote
from faceaudit.models import FaceInfo, SERFIQInfo, FaceDataset
from .ff_service import FFService

class SERFIQService:
    """
    Serviço para cálculo de Qualidade de Imagem Facial Estocástica (SER-FIQ).
    
    O SER-FIQ (Stochastic Embedding Robustness Face Image Quality) mede a qualidade
    de uma face analisando a estabilidade do seu vetor de características (embedding)
    sob pequenas perturbações.
    
    Conceito:
    Faces de alta qualidade geram embeddings estáveis (agrupados) mesmo se a imagem
    sofrer pequenas alterações. Faces de baixa qualidade geram embeddings dispersos.
    
    Dependências:
    Requer uma instância ativa do `FFService` para gerar os embeddings.
    """

    def __init__(self, ff_service: FFService, T: int = 10, alpha: float = 1.5):
        """
        Inicializa o serviço SER-FIQ com valores padrão.

        Args:
            ff_service (FFService): 
                Instância do serviço de biometria já carregada. Será usada para 
                extrair embeddings das variações da imagem.
            
            T (int, optional): 
                Valor padrão para o número de variações estocásticas (augmentations).
                Pode ser sobrescrito nos métodos de processamento. Defaults to 10.
            
            alpha (float, optional): 
                Valor padrão para o parâmetro de ajuste da função logística.
                Pode ser sobrescrito nos métodos de processamento. Defaults to 1.5.
        """
        self.ff_service = ff_service
        self.default_T = T
        self.default_alpha = alpha

    def process_face(self, face_info: FaceInfo, t: Optional[int] = None, alpha: Optional[float] = None) -> FaceInfo:
        """
        Calcula o score SER-FIQ para uma única face.

        Passos:
        1. Lê a imagem original.
        2. Gera T variações aleatórias (crop, brilho).
        3. Extrai embeddings de todas as variações.
        4. Calcula a dispersão (distância euclidiana média) entre os vetores.
        5. Converte a dispersão em um score de 0 a 100.

        Args:
            face_info (FaceInfo): Objeto contendo o caminho da imagem.
            t (Optional[int]): Sobrescreve o número de variações T para esta execução.
            alpha (Optional[float]): Sobrescreve o parâmetro alpha para esta execução.

        Returns:
            FaceInfo: Objeto enriquecido com os dados do SER-FIQ (.serfiq_info).
        """
        # 1. Determina parâmetros (Override ou Default)
        use_T = t if t is not None else self.default_T
        use_alpha = alpha if alpha is not None else self.default_alpha

        # 2. Read original image using FFService's safe utility
        img_original = self.ff_service._safe_read_image(face_info.image_path)
        
        if img_original is None:
            # Falha na leitura: retorna sem score
            return face_info

        # 3. Calculation Logic
        score = self._calculate_score_core(img_original, T=use_T, alpha=use_alpha)

        # 4. Populate Model
        # Even if score is 0.0 (error), we populate to show it was processed
        face_info.serfiq_info = SERFIQInfo(
            score=score,
            t=use_T,
            alpha=use_alpha
        )

        return face_info

    def process_batch(self, faces: List[FaceInfo], t: Optional[int] = None, alpha: Optional[float] = None) -> List[FaceInfo]:
        """
        Processa uma lista de faces (Wrapper para tqdm).
        
        Args:
            faces (List[FaceInfo]): Lista de faces para processar.
            t (Optional[int]): Sobrescreve T para este lote.
            alpha (Optional[float]): Sobrescreve alpha para este lote.

        Nota: Preferir `process_dataset` para integração completa.
        """
        use_T = t if t is not None else self.default_T

        for face in tqdm(faces, desc=f"Calculating SER-FIQ (T={use_T})", unit="img"):
            self.process_face(face, t=t, alpha=alpha)
        return faces

    def process_dataset(self, 
                        dataset: FaceDataset, 
                        callback: Optional[Callable[[], None]] = None,
                        t: Optional[int] = None, 
                        alpha: Optional[float] = None) -> FaceDataset:
        """
        Processa um FaceDataset completo calculando o score SER-FIQ para todas as faces.

        Este é geralmente o passo mais lento do pipeline, pois cada imagem é processada
        T vezes pela rede neural. O uso de callback é altamente recomendado.

        Args:
            dataset (FaceDataset): O dataset a ser analisado.
            callback (Optional[Callable[[], None]], optional): 
                Função de callback sem argumentos, chamada após a conclusão do cálculo
                para cada imagem original. Defaults to None.
            t (Optional[int]): 
                Sobrescreve o número de iterações T para todo o dataset.
            alpha (Optional[float]): 
                Sobrescreve o parâmetro alpha para todo o dataset.

        Returns:
            FaceDataset: O dataset enriquecido com métricas de robustez.
        """
        use_T = t if t is not None else self.default_T
        
        print(f"Iniciando cálculo SER-FIQ (T={use_T}) para: {dataset.name}")
        
        # Como o SER-FIQ é pesado (roda 10x por face), o feedback visual é crucial
        for face in tqdm(dataset, desc=f"SER-FIQ {dataset.name}", unit="face"):
            self.process_face(face, t=t, alpha=alpha)
            
            # Aciona o progresso para a GUI
            if callback:
                callback()
            
        return dataset

    def _calculate_score_core(self, img_original: np.ndarray, T: int, alpha: float) -> float:
        """
        Núcleo matemático do cálculo do SER-FIQ.

        Lógica:
        1. Extrai embedding da imagem original.
        2. Gera T variações da imagem e extrai seus embeddings.
        3. Normaliza todos os vetores.
        4. Calcula a média dos embeddings.
        5. Mede a distância média de cada variação em relação à média.
        6. Aplica função logística inversa: quanto menor a distância, maior a qualidade.

        Args:
            img_original (np.ndarray): Imagem BGR em memória.
            T (int): Número de variações estocásticas.
            alpha (float): Parâmetro de sensibilidade da função logística.

        Returns:
            float: Score de qualidade entre 0.0 e 100.0.
        """
        embeddings = []

        # A. Original Embedding
        emb_orig = self.ff_service.get_embedding_from_array(img_original)
        if emb_orig is not None:
            embeddings.append(emb_orig)
        else:
            return 0.0 # If original face is not detected, quality is 0

        # B. Augmented Embeddings (T times)
        for _ in range(T):
            img_aug = self._augment_image(img_original)
            emb = self.ff_service.get_embedding_from_array(img_aug)
            
            if emb is not None:
                embeddings.append(emb)
            # If detection fails on augmented image, we just skip it (robustness)

        # Need at least 2 embeddings to calculate variance
        if len(embeddings) < 2:
            return 0.0

        # C. Mathematical Score Calculation
        X = np.array(embeddings, dtype=float)
        
        # L2 Normalize vectors first (Critical for cosine-like behavior in Euclidean space)
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norm + 1e-10)

        # Mean Embedding
        mean_embedding = np.mean(X, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

        # Euclidean distances from the mean
        dists = np.linalg.norm(X - mean_embedding, axis=1)
        mean_dist = np.mean(dists)

        # Logistic calibration
        # Score = 1 / (1 + alpha * dist)
        raw_score = 1.0 / (1.0 + alpha * mean_dist)
        
        return raw_score * 100.0

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """
        Aplica variações estocásticas leves na imagem.

        As variações simulam incertezas na captura:
        1. Random Padding: Simula pequenas variações de escala/enquadramento (Zoom Out).
        2. Random Brightness: Simula pequenas variações de iluminação.

        Args:
            img (np.ndarray): Imagem original.

        Returns:
            np.ndarray: Imagem alterada.
        """
        rows, cols, _ = img.shape
        
        # 1. Random Padding (0% to 5%) - Simulates Zoom Out
        pad_h = int(rows * 0.05)
        pad_w = int(cols * 0.05)
        
        pad_top = np.random.randint(0, pad_h + 1)
        pad_bottom = np.random.randint(0, pad_h + 1)
        pad_left = np.random.randint(0, pad_w + 1)
        pad_right = np.random.randint(0, pad_w + 1)
        
        img_aug = cv2.copyMakeBorder(
            img, 
            pad_top, pad_bottom, pad_left, pad_right, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )

        # 2. Random Brightness (0.95 to 1.05)
        beta = np.random.uniform(0.95, 1.05)
        # convertScaleAbs handles clipping to 0-255 automatically
        img_aug = cv2.convertScaleAbs(img_aug, alpha=beta, beta=0)
        
        return img_aug