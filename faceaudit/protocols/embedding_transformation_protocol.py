# faceaudit/protocols/embedding_transformation_protocol.py

from typing import Any, Dict, Protocol, Optional, runtime_checkable
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.enums.transformation_type import TransformationType

@runtime_checkable
class EmbeddingTransformationProtocol(Protocol): # <-- Renomeado para EmbeddingTransformationProtocol
    """
    Protocolo (Interface) para transformadores que realizam operações em embeddings.
    
    Define a interface comum para transformadores que aprendem parâmetros (fit)
    e aplicam transformações (transform) em um ou múltiplos embeddings.
    """

    @property
    def name(self) -> str:
        """
        Retorna o nome único do transformador (e.g., 'CoralTransformation', 'L2Normalization'). # <-- Descrição atualizada
        
        Returns:
            str: O nome do transformador.
        """
        ...
    
    @property
    def transformation_type(self) -> TransformationType:
        """
        Retorna o Tipo Enumerado da transformação.
        Essencial para versionamento, salvamento de checkpoints e sincronia de visualização.
        """
        ...

    @property
    def requires_target_dataset(self) -> bool:
        """
        Define se a transformação *obrigatoriamente* precisa de um dataset alvo
        para funcionar (como o CORAL).
        
        Se True: A UI deve impedir o treino se o usuário não selecionar um Target.
        Se False: A UI deixa o Target como opcional (Whitening usa se tiver, L2 ignora).
        """
        ...

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        Ajusta os parâmetros internos do transformador com base em datasets de treino. # <-- Descrição atualizada
        
        Para transformadores sem estado ou que não requerem um "treino" tradicional
        (como `L2Normalization`), esta função pode simplesmente `pass`.

        Args:
            source_dataset (EmbeddingDataset): O dataset principal de embeddings
                                               a ser usado para ajustar o transformador.
            target_dataset (Optional[EmbeddingDataset]): Um dataset secundário,
                                                          geralmente utilizado por
                                                          transformações como CORAL para
                                                          alinhamento de domínio, onde
                                                          o `source_dataset` é transformado
                                                          ao `target_dataset`. Pode ser `None`.
        """
        ...

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """
        Aplica a transformação do transformador a um `EmbeddingDataset` completo. # <-- Descrição atualizada
        
        Importante: O transformador deve retornar um *novo* `EmbeddingDataset`
        com os embeddings transformados, sem modificar o `dataset` original.

        Args:
            dataset (EmbeddingDataset): O dataset de embeddings a ser transformado.

        Returns:
            EmbeddingDataset: Um novo `EmbeddingDataset` contendo os embeddings
                              já transformados.
        """
        ...

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica a transformação do transformador a um único `EmbeddingInfo`. # <-- Descrição atualizada
        
        Importante: O transformador deve retornar um *novo* `EmbeddingInfo`
        com o embedding transformado, sem modificar o `embedding_info` original.

        Args:
            embedding_info (EmbeddingInfo): O `EmbeddingInfo` a ser transformado.

        Returns:
            EmbeddingInfo: Um novo `EmbeddingInfo` contendo o embedding
                           já transformado.
        """
        ...
    
    # --- NOVO MÉTODO PARA RASTREABILIDADE ---
    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Retorna um dicionário com métricas relevantes da última execução/treinamento.
        
        Exemplos:
        - Whitening: {'explained_variance_ratio': 0.95, 'n_components': 128}
        - CORAL: {'frobenius_norm_diff': 0.02}
        - L2: {'scale': 1.0}
        
        Esses dados serão salvos nos metadados do arquivo Parquet para auditoria.
        """
        ...
    # ----------------------------------------

    def save(self, path: str) -> None:
        """
        Persiste o estado interno (parâmetros aprendidos) do transformador em disco. # <-- Descrição atualizada
        Para transformadores stateless, pode fazer 'pass'.

        Args:
            path (str): Caminho completo para o arquivo onde o estado será salvo.
        """
        ...

    def load(self, path: str) -> None:
        """
        Carrega o estado interno (parâmetros aprendidos) do transformador do disco. # <-- Descrição atualizada
        Para transformadores stateless, pode fazer 'pass'.

        Args:
            path (str): Caminho completo para o arquivo de onde o estado será carregado.
        """
        ...
    
        # ✅ NOVO: estado serializável para shadow-copy seguro (C1.4)
    def get_state(self) -> Dict[str, Any]:
        """
        Retorna um dicionário com o estado treinado/config necessário para replicar
        este step em outro objeto equivalente (ex: pipeline Target).
        Deve conter apenas dados serializáveis (numpy ok) e nunca referências de serviço.
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaura o estado retornado por get_state().
        """
        ...