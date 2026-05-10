from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from faceaudit.models.dir_template import DirTemplate


@dataclass
class DatasetIngestionConfig:
    """
    DTO (Data Transfer Object) que contém todas as configurações
    necessárias para criar/enriquecer um dataset via Pipeline.
    """
    # 1. Dados Básicos
    input_path: Path
    dataset_name: str
    description: str = ""

    # 2. Estratégia de Ingestão
    recursive: bool = False
    template: Optional[DirTemplate] = None

    # 3. Metadados Padrão (Fallback ou Force)
    default_identity: str = "Unknown"
    origin: str = "Local"
    default_gender: Optional[str] = None    # "Male", "Female" ou None
    default_ethnicity: Optional[str] = None

    # 4. Flags de etapas (binárias — liga/desliga cada serviço)
    compute_ff: bool = True      # ForensicFace (embedding + detecção)
    compute_ofiq: bool = True    # Qualidade ISO
    compute_serfiq: bool = True  # Qualidade Estocástica

    # 5. Parâmetros operacionais do ForensicFace
    # Lista de modelos a carregar (multi-select). O `FFService` instancia
    # uma `ForensicFace` por modelo — `process_face` usa o primeiro como
    # default e preenche o slot correspondente em FFInfo
    # (`embedding_sepaelv2` ou `embedding_sepaelv4`).
    ff_models: List[str] = field(default_factory=lambda: ["sepaelv2"])
    # Tamanho da imagem de entrada do detector. Comuns: 320 (default),
    # 640 (melhor pra fotos de baixa res, ~2x mais GPU).
    ff_det_size: int = 320
    # Quando True (default), computa FIQA, gender, age, yaw/pitch/roll
    # além do embedding. False = só embedding + bbox + keypoints (mais
    # rápido, dataset mais leve).
    ff_extended: bool = True

    # 6. Parâmetros operacionais do SER-FIQ
    serfiq_t: int = 10        # iterações estocásticas (mais = mais robusto)
    serfiq_alpha: float = 1.5  # fator de escala do score

    # 7. PR 18 — idempotência do enriquecimento.
    # Quando False (default), o `FeatureExtractionPipeline.enrich_existing_dataset`
    # filtra as faces que já têm o enrichment solicitado e pula as demais —
    # útil pra "completar" um dataset parcialmente enriquecido sem reprocessar
    # faces que já passaram (economia de GPU/CPU). True força reprocessamento
    # de tudo, sobrescrevendo os campos *_info existentes.
    force_reprocess: bool = False

    # 8. Marca o dataset como contendo faces já alinhadas (112×112).
    # Quando True, o FaceDataset criado/enriquecido recebe
    # `is_aligned=True` e o FFService usa `process_face_aligned`
    # (pula detect+norm_crop, vai direto pro embedding extraction).
    # Default False — a maioria dos datasets é de fotos originais.
    is_aligned: bool = False

    # 9. Alinhar faces DURANTE a criação — modo "originais → alinhar
    # agora" em 1 passo. Quando True, após o FF rodar nos originais
    # (detect+align+embed), os crops alinhados são salvos como PNG
    # em `aligned_target_dir/images/` e o dataset resultante fica
    # marcado com is_aligned=True, path_mode=RELATIVE, com paths
    # apontando pros novos PNGs. Evita o re-trabalho de
    # New Dataset (com FF) → Image Pipeline (ALIGNED_FACE) → Enrich,
    # que rodaria detect+align 2× no mesmo dataset.
    # Mutuamente exclusivo com is_aligned (face é OU alinhada-fonte
    # OU vai-ser-alinhada).
    align_during_ingestion: bool = False
    # Pasta onde os PNGs alinhados + parquet vão. Estrutura:
    #   <aligned_target_dir>/images/<file>.png
    #   <aligned_target_dir>/<dataset_name>.parquet
    # Obrigatório quando `align_during_ingestion=True`.
    aligned_target_dir: Optional[Path] = None

    def __post_init__(self):
        # Garante que o path seja Path object
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        # Garante que ff_models não fique vazio — sem modelo, FF não roda.
        if not self.ff_models:
            self.ff_models = ["sepaelv2"]