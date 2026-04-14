from dataclasses import dataclass
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class SplitSimulationResult(JSONSerializableMixin):
    """
    Carrega as estatísticas geradas após uma simulação de divisão de dados (Dry-Run).
    Usado pela interface para validar se há dados e combinações suficientes antes de treinar.
    """
    
    # --- DADOS DO TREINO (PLDA Matriz Geométrica) ---
    plda_identities_count: int = 0
    plda_images_count: int = 0
    plda_genuine_pairs: int = 0
    plda_impostor_pairs: int = 0
    
    # --- DADOS DA CALIBRAÇÃO (Regressão Logística) ---
    calib_identities_count: int = 0
    calib_images_count: int = 0
    calib_genuine_pairs: int = 0
    calib_impostor_pairs: int = 0
    
    # --- STATUS GERAL ---
    success: bool = False
    error_message: str = ""
    
    @property
    def total_identities(self) -> int:
        return self.plda_identities_count + self.calib_identities_count
        
    @property
    def total_images(self) -> int:
        return self.plda_images_count + self.calib_images_count
        
    @property
    def plda_total_pairs(self) -> int:
        return self.plda_genuine_pairs + self.plda_impostor_pairs

    @property
    def calib_total_pairs(self) -> int:
        return self.calib_genuine_pairs + self.calib_impostor_pairs