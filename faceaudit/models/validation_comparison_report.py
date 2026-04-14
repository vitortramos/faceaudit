# faceaudit/models/validation_comparison_report.py
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
import pickle

from faceaudit.models.strategy_validation_result import StrategyValidationResult

@dataclass
class ValidationComparisonReport:
    """
    O 'Laudo de Batalha'. Contém arrays massivos do Numpy.
    É serializado via Pickle para garantir performance e eficiência de armazenamento.
    """
    probe_dataset_name: str
    reference_dataset_name: str
    results: Dict[str, StrategyValidationResult] = field(default_factory=dict)
    
    @property
    def has_results(self) -> bool:
        return len(self.results) > 0

    def get_winner_by_cllr(self) -> Optional[StrategyValidationResult]:
        if not self.has_results: return None
        return min(self.results.values(), key=lambda r: r.metrics.cllr)

    def get_winner_by_eer(self) -> Optional[StrategyValidationResult]:
        if not self.has_results: return None
        return min(self.results.values(), key=lambda r: r.metrics.eer)

    # --- SERIALIZAÇÃO BINÁRIA ---
    def save_to_disk(self, filepath: Path) -> None:
        """Salva o relatório completo em formato binário otimizado."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_disk(cls, filepath: Path) -> Optional['ValidationComparisonReport']:
        """Restaura o relatório binário do disco para a RAM."""
        if not filepath.exists():
            return None
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache de validação: {e}")
            return None