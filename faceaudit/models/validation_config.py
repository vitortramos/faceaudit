# faceaudit/models/validation_config.py
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.project_file import ProjectFile

T = TypeVar("T", bound="ValidationConfig")

@dataclass
class ValidationConfig(JSONSerializableMixin):
    """
    Representa o estado completo da Arena de Validação.
    Define o gabarito, os modelos competidores e o contexto pericial.
    """
    # 1. Definição do Gabarito (Geralmente carregados do manifesto)
    probe_dataset_name: str = "Selecione..."
    reference_dataset_name: str = "Selecione..."
    
    # 2. Estratégias selecionadas para o confronto
    strategy_ids: List[str] = field(default_factory=list)
    
    # 3. Contexto de Saída e Identificação
    report_name: str = "Relatorio_Validacao_01"
    description: str = ""
    
    # 4. Flags de Auditoria (Cache do Gatekeeper)
    # Indica se a combinação atual de datasets/estratégias é segura
    is_audit_passed: bool = False
    audit_messages: List[str] = field(default_factory=list)
    
    # 5. MANIFESTO FÍSICO (Cadeia de Custódia e Deep Copy)
    # Rastreia os datasets de probe e reference usados na Arena.
    project_files: List[ProjectFile] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Checagem básica para habilitar o botão 'Executar' na UI."""
        has_probe = self.probe_dataset_name and self.probe_dataset_name != "Selecione..."
        has_ref = self.reference_dataset_name and self.reference_dataset_name != "Selecione..."
        has_competitors = len(self.strategy_ids) > 0
        return bool(has_probe and has_ref and has_competitors)

    def reset_audit(self):
        """Reseta o status de auditoria quando os datasets ou estratégias mudam."""
        self.is_audit_passed = False
        self.audit_messages = []

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o ValidationConfig a partir do JSON (Arquivo .fap).
        Garante a instanciação correta dos objetos de arquivo (ProjectFile).
        """
        if not data:
            return cls()

        # Reconstrói a lista de arquivos rastreados, se existir
        if 'project_files' in data and isinstance(data['project_files'], list):
            data['project_files'] = [ProjectFile.from_dict(f) for f in data['project_files']]
            
        return cls(**data)