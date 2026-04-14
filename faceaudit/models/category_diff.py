from dataclasses import dataclass

@dataclass
class CategoryDiff:
    """
    Representa a comparação de uma categoria específica entre dois datasets.
    Exemplo: Categoria='Male', count_a=100, count_b=120.
    """
    category: str
    count_a: int
    count_b: int

    @property
    def diff(self) -> int:
        """
        Retorna o delta (B - A).
        Positivo significa que o Dataset B tem mais itens nesta categoria.
        """
        return self.count_b - self.count_a
    
    @property
    def diff_str(self) -> str:
        """
        Helper para UI: Retorna o delta formatado com sinal explícito.
        Ex: '+5', '-3', '0'.
        """
        val = self.diff
        return f"+{val}" if val > 0 else str(val)