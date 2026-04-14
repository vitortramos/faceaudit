from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd

# Import relativo
from .embedding_comparison import EmbeddingComparison

@dataclass
class EmbeddingComparisonDataset:
    """
    Coleção de resultados de confrontos biométricos (1:1).
    Útil para auditoria de performance, cálculo de métricas (ROC, EER) 
    e exportação de relatórios de erro.
    """
    name: str
    description: Optional[str] = None
    items: List[EmbeddingComparison] = field(default_factory=list)

    def add(self, comparison: EmbeddingComparison) -> None:
        self.items.append(comparison)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> EmbeddingComparison:
        return self.items[index]

    def __iter__(self):
        return iter(self.items)

    def to_pandas(self) -> pd.DataFrame:
        """
        Converte todos os confrontos em um DataFrame achatado.
        Ideal para análises estatísticas e gráficos.
        """
        if not self.items:
            return pd.DataFrame()
        
        # O método to_dict do EmbeddingComparison já retorna os dados achatados
        rows = [item.to_dict() for item in self.items]
        df = pd.DataFrame(rows)
        
        # Adiciona metadados do dataset como colunas para rastreabilidade
        df.insert(0, 'comparison_dataset_name', self.name)
        return df

    def to_parquet(self, path: Union[str, Path]):
        """Salva os resultados em disco para auditoria posterior."""
        df = self.to_pandas()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"EmbeddingComparisonDataset '{self.name}' salvo em: {path}")

    @property
    def total_genuines(self) -> int:
        """Conta quantos pares são da mesma pessoa (Ground Truth)."""
        return sum(1 for item in self.items if item.same_person is True)

    @property
    def total_impostors(self) -> int:
        """Conta quantos pares são de pessoas diferentes."""
        return sum(1 for item in self.items if item.same_person is False)

    def get_lowest_similarity_genuines(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares que são comprovadamente a mesma pessoa (genuínos), 
        mas que o modelo atribuiu os menores scores de similaridade.
        Útil para analisar casos de 'Dificuldade de Reconhecimento'.
        """
        df = self.to_pandas()
        if df.empty: return df
        genuines = df[df['same_person'] == True]
        return genuines.nsmallest(top_n, 'cosine_similarity')

    def get_highest_similarity_impostors(self, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna os pares que são comprovadamente pessoas diferentes (impostores), 
        mas que o modelo atribuiu os maiores scores de similaridade.
        Útil para analisar casos de 'Risco de Fraude' ou 'Falsa Similaridade'.
        """
        df = self.to_pandas()
        if df.empty: return df
        impostors = df[df['same_person'] == False]
        return impostors.nlargest(top_n, 'cosine_similarity')