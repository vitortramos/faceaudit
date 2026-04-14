import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any, Type, TypeVar
from pathlib import Path

# Mixin de Serialização
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Imports do Domínio
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.models.pipeline_summary import PipelineSummary

T = TypeVar("T", bound="TransformationPipelineConfig")

@dataclass
class TransformationPipelineConfig(JSONSerializableMixin):
    """
    Define a sequência e a configuração completa de um pipeline de transformação.
    Herdar de JSONSerializableMixin permite que o pipeline seja salvo dentro do JSON do Workspace.
    """
    name: str = field(default="Novo Pipeline", metadata={"help": "Nome único para identificar este pipeline."})
    description: Optional[str] = field(default=None, metadata={"help": "Descrição detalhada."})
    transformators: List[TransformatorConfig] = field(default_factory=list, metadata={"help": "Lista ordenada de etapas."})

    # --- SERIALIZAÇÃO ---

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o pipeline garantindo que cada transformador na lista 
        seja instanciado como o objeto correto (TransformatorConfig).
        """
        if not data:
            return None
            
        if 'transformators' in data and isinstance(data['transformators'], list):
            # Delegamos a reconstrução de cada step para a classe filha, que agora 
            # usará a Fábrica de DTOs internamente.
            data['transformators'] = [
                TransformatorConfig.from_dict(t) if isinstance(t, dict) else t 
                for t in data['transformators']
            ]
            
        return cls(**data)

    # --- PROPRIEDADES DINÂMICAS ---

    @property
    def has_pca(self) -> bool:
        return any(t.type == TransformationType.WHITENING_TRANSFORMATION for t in self.transformators)

    @property
    def has_lda(self) -> bool:
        return any(t.type == TransformationType.LDA_TRANSFORMATION for t in self.transformators)

    @property
    def output_dimension(self) -> Optional[int]:
        for t in reversed(self.transformators):
            # Agora acessamos os atributos fortemente tipados do DTO
            if t.type == TransformationType.WHITENING_TRANSFORMATION and t.config:
                return t.config.n_components 
            if t.type == TransformationType.LDA_TRANSFORMATION and t.config:
                return t.config.n_components
        return None

    # --- GERAÇÃO DE RESUMO UI ---

    def to_summary(self) -> PipelineSummary:
        steps_desc = []
        for step in self.transformators:
            try:
                # O format_summary do Enum agora aceita o DTO diretamente
                desc = step.type.format_summary(step.config)
                steps_desc.append(desc)
            except Exception:
                steps_desc.append(step.name or str(step.type))

        summary_text = " ➔ ".join(steps_desc) if steps_desc else "Pipeline Vazio"
        return PipelineSummary.success(
            text=summary_text,
            has_pca=self.has_pca,
            has_lda=self.has_lda
        )

    # --- UTILITÁRIOS ---

    def set_global_output_dimension(self, dim: int) -> None:
        """
        Força a dimensão de saída em transformadores que suportam redução.
        """
        for t_config in self.transformators:
            supports = getattr(t_config.type, 'supports_dimension_reduction', False)
            type_name = t_config.type.name if hasattr(t_config.type, 'name') else str(t_config.type)
            is_reduction = supports or (type_name in ['WHITENING_TRANSFORMATION', 'LDA_TRANSFORMATION'])

            if is_reduction and t_config.config:
                # Como a retrocompatibilidade de pca_components foi resolvida no from_dict 
                # do WhiteningConfig, aqui só precisamos setar n_components diretamente no DTO.
                t_config.config.n_components = dim
        
    # --- LEGACY SUPPORT (Pickle) ---

    def to_pickle(self, path: Union[str, Path]) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> 'TransformationPipelineConfig':
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def __str__(self):
        try:
            # Substituído params por config
            step_names = [step.type.format_summary(step.config) for step in self.transformators]
        except:
            step_names = [step.name for step in self.transformators]
        return f"Pipeline('{self.name}'): {step_names}"

    def __repr__(self):
        return self.__str__()