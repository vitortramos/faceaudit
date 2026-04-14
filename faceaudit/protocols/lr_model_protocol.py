from typing import Protocol, runtime_checkable, Optional
from faceaudit.models import EmbeddingDataset, EmbeddingInfo

@runtime_checkable
class LRModelProtocol(Protocol):
    """
    Protocolo (Interface) para modelos de cálculo de Likelihood Ratio (LR).
    
    Implementa o padrão Strategy, permitindo que diferentes algoritmos estatísticos
    (PLDA, KDE, Weibull, etc.) sejam usados de forma intercambiável pelo sistema.
    """

    @property
    def name(self) -> str:
        """
        Retorna o identificador único do modelo (ex: 'PLDA_Latent', 'KDE_Cosine').
        Útil para logs e para nomear colunas nos resultados.
        """
        ...
    
    # --- NOVOS ATRIBUTOS ---
    @property
    def requires_train_dataset(self) -> bool:
        """Define se a estratégia precisa de dados para aprendizado de características (ex: PLDA)."""
        ...

    @property
    def requires_calibration_dataset(self) -> bool:
        """Define se a estratégia precisa de dados para ajuste de densidade/probabilidade."""
        ...
    # -----------------------

    def fit(self, 
            train_dataset: Optional[EmbeddingDataset], 
            calibration_dataset: EmbeddingDataset) -> None:
        """
        Treina o modelo estatístico.
        
        A assinatura unificada acomoda diferentes necessidades de dados:
        
        Args:
            train_dataset: 
                Dados para aprendizado de representação (Feature Learning).
                - PLDA: Usa este dataset para aprender a matriz de projeção latente.
                - KDE/Weibull: Ignoram este parâmetro (passar None), pois trabalham apenas com scores.
                
            calibration_dataset: 
                Dados para ajuste de densidade ou calibração (Score-to-Likelihood).
                - Todos os modelos usam este dataset para calibrar suas probabilidades.
        """
        ...

    def predict(self, probe: EmbeddingInfo, reference: EmbeddingInfo) -> float:
        """
        Calcula a Likelihood Ratio (LR) para um par de embeddings.
        
        Args:
            probe: Embedding da imagem questionada.
            reference: Embedding da imagem padrão.
            
        Returns:
            float: O valor da LR. 
                   LR > 1 favorece a hipótese de mesma origem (Hp).
                   LR < 1 favorece a hipótese de origem diferente (Hd).
        """
        ...

    def save(self, path: str) -> None:
        """Persiste o estado interno do modelo em disco."""
        ...

    def load(self, path: str) -> None:
        """Carrega o estado interno do modelo do disco."""
        ...