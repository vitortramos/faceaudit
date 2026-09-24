from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from faceaudit.enums.dataset_image_kind import DatasetImageKind
from faceaudit.models.dir_template import DirTemplate


def _default_ff_model_name() -> str:
    """Lazy-resolve do default pra evitar ciclo de imports
    (`SepaelModel` está em `faceaudit.enums`, que pode importar coisas
    de `faceaudit.models` futuramente)."""
    from faceaudit.enums.sepael_model import SepaelModel
    return SepaelModel.default().ff_name


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
    # Metadados embutidos (EXIF/XMP/IPTC) da foto ORIGINAL. Desligado por
    # padrão: só faz sentido em dataset de fotos originais — recorte e
    # alinhamento gravam pixel novo, sem metadado de captura.
    extract_exif: bool = False
    compute_serfiq: bool = True  # Qualidade Estocástica

    # 5. Parâmetros operacionais do ForensicFace
    # Lista de modelos a carregar (multi-select). O `FFService` instancia
    # uma `ForensicFace` por modelo — `process_face` usa o primeiro como
    # default e preenche o slot correspondente em FFInfo
    # (`embedding_sepaelv2`, `embedding_sepaelv3`, ..., `embedding_sepaelv6`).
    # Default vem do `SepaelModel.default()` — único ponto de verdade do
    # codebase pra qual modelo é padrão.
    ff_models: List[str] = field(
        default_factory=lambda: [_default_ff_model_name()]
    )
    # Tamanho da imagem de entrada do detector. Comuns: 320 (default),
    # 640 (melhor pra fotos de baixa res, ~2x mais GPU).
    ff_det_size: int = 320
    # Quando True (default), computa FIQA, gender, age, yaw/pitch/roll
    # além do embedding. False = só embedding + bbox + keypoints (mais
    # rápido, dataset mais leve). O engine calcula esses atributos em
    # BLOCO (tudo-ou-nada); a seleção fina de QUAIS colunas manter vai em
    # `ff_kept_attributes` (filtragem na escrita).
    ff_extended: bool = True

    # Colunas de atributo do ForensicFace a MANTER no dataset. None = manter
    # todas (compat). Chaves: "fiqa", "yaw", "pitch", "roll", "gender",
    # "age", "det_score", "ipd". A UI de Novo Dataset popula isto a partir
    # dos toggles granulares (Pose/Outros/Qualidade); a filtragem efetiva
    # das colunas acontece no pipeline (não gera a coluna desmarcada).
    ff_kept_attributes: Optional[List[str]] = None

    # Override POR extração de capability da máquina (None = usa o default
    # do FFService singleton / Configurações do Sistema; a UI pré-preenche
    # com o valor do sistema e não grava de volta no global).
    use_gpu: Optional[bool] = None
    batch_size: Optional[int] = None

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
    # Pasta onde os PNGs alinhados são gravados, direto na raiz dela
    # (sem subpasta intermediária):
    #   <aligned_target_dir>/<file>.png
    # Obrigatório quando `align_during_ingestion=True`. NÃO determina
    # onde o parquet vai — isso é `dataset_dir`.
    aligned_target_dir: Optional[Path] = None

    # 10. Destino do dataset (parquet) e modo de armazenamento.
    # `dataset_dir=None` → biblioteca global (`pipeline.storage_path`),
    # que é o default e preserva o comportamento histórico.
    dataset_dir: Optional[Path] = None
    # Quando True, as imagens passam a viver junto do parquet em
    # `<dataset_dir>/<dataset_name>/images/` e os paths ficam RELATIVOS —
    # dataset portátil entre máquinas/drives. Quando False, o dataset
    # referencia as imagens onde elas estão, em PathMode.ABSOLUTE.
    self_contained: bool = False
    # Qual FORMA da imagem o dataset autocontido guarda. Só tem efeito com
    # `self_contained=True`; sem ele não há imagem copiada para escolher.
    #   ORIGINAL — a foto como veio.
    #   CROP     — recorte da face com margem, sem rotação. Preserva a
    #              geometria real; é a forma útil para laudo e comparação
    #              visual. NÃO marca `is_aligned` (ver o enum).
    #   ALIGNED  — face 112×112, o que o modelo enxerga.
    image_kind: DatasetImageKind = DatasetImageKind.ORIGINAL
    # Fator de expansão da caixa da face no modo CROP (1.0 = caixa exata,
    # 2.0 = o dobro). Mesma unidade da extração de faces de vídeo.
    crop_margin_factor: float = 2.0

    def __post_init__(self):
        # Garante que o path seja Path object
        if isinstance(self.input_path, str):
            self.input_path = Path(self.input_path)
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir) if self.dataset_dir else None
        self.image_kind = DatasetImageKind.from_str(
            self.image_kind, default=DatasetImageKind.ORIGINAL,
        )
        try:
            self.crop_margin_factor = max(1.0, float(self.crop_margin_factor))
        except (TypeError, ValueError):
            self.crop_margin_factor = 2.0
        # `align_during_ingestion` é a forma antiga de dizer
        # `image_kind=ALIGNED`. Mantida porque há chamadores que ainda a
        # passam; as duas convergem aqui para não divergirem em runtime.
        if self.align_during_ingestion and self.image_kind is DatasetImageKind.ORIGINAL:
            self.image_kind = DatasetImageKind.ALIGNED
        self.align_during_ingestion = self.image_kind is DatasetImageKind.ALIGNED
        # Garante que ff_models não fique vazio — sem modelo, FF não roda.
        if not self.ff_models:
            self.ff_models = [_default_ff_model_name()]