from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineSummary:
    """
    Objeto de transporte (Model) que resume um Pipeline para a UI.
    Local: faceaudit/models/pipeline_summary.py
    """
    text: str
    has_pca: bool
    has_lda: bool
    is_valid: bool
    
    @classmethod
    def empty(cls) -> 'PipelineSummary':
        """Estado inicial ou 'Selecione...'"""
        return cls(text="-", has_pca=False, has_lda=False, is_valid=False)

    @classmethod
    def error(cls, msg: str) -> 'PipelineSummary':
        """Estado de erro (arquivo não encontrado, corrompido)"""
        return cls(text=f"❌ {msg}", has_pca=False, has_lda=False, is_valid=False)

    @classmethod
    def success(cls, text: str, has_pca: bool, has_lda: bool) -> 'PipelineSummary':
        """Estado de sucesso com dados carregados"""
        return cls(text=text, has_pca=has_pca, has_lda=has_lda, is_valid=True)