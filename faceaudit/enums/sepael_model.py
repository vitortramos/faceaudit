# faceaudit/enums/sepael_model.py

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional


class SepaelModel(Enum):
    """
    Enum das versões de modelos do ForensicFace conhecidas pelo FaceAudit.

    Cada membro carrega tudo que o código precisa pra coordenar com a
    biblioteca `forensicface`:

    - **ff_name**: literal aceito pelo `ForensicFace(models=[...])` e por
      `_resolve_engine` do FFService (ex: `"sepaelv2"`).
    - **label_pt**: nome amigável pra UI (ex: "Sepael V2").
    - **ffinfo_slot**: nome do atributo correspondente em `FFInfo`
      (ex: `"embedding_sepaelv2"`).
    - **parquet_column**: coluna do parquet quando o embedding desse
      modelo é serializado (ex: `"embedding_v2"`).
    - **embedding_dim**: dimensionalidade do vetor produzido pelo encoder.
    - **encoder_subdir**: subpasta do encoder dentro de
      `~/.forensicface/models/<ff_name>/` (None quando o encoder fica no
      root da pasta).
    - **requires_keypoints**: True quando o modelo é KPRPE (sepaelv6),
      consome `aligned_keypoints` como segundo input ONNX.

    Detecção de disponibilidade em disco via `available_on_disk()` — varre
    `~/.forensicface/models/<ff_name>/` procurando pelos arquivos ONNX
    essenciais (detector det_10g + encoder principal). Retorna apenas
    modelos completos.
    """

    SEPAEL_V2 = (
        "sepaelv2",
        "Sepael V2",
        "embedding_sepaelv2",
        "embedding_v2",
        512,
        "adaface",
        False,
    )
    SEPAEL_V3 = (
        "sepaelv3",
        "Sepael V3",
        "embedding_sepaelv3",
        "embedding_v3",
        512,
        "adaface",
        False,
    )
    SEPAEL_V4 = (
        "sepaelv4",
        "Sepael V4",
        "embedding_sepaelv4",
        "embedding_v4",
        512,
        "adaface_vit_webface4m",
        False,
    )
    SEPAEL_V5 = (
        "sepaelv5",
        "Sepael V5",
        "embedding_sepaelv5",
        "embedding_v5",
        512,
        "topofr_r200_glint360k_face",
        False,
    )
    SEPAEL_V6 = (
        "sepaelv6",
        "Sepael V6 (KPRPE)",
        "embedding_sepaelv6",
        "embedding_v6",
        512,
        None,  # encoder no root ou subdir varia — checa por padrão de nome.
        True,
    )

    def __init__(
        self,
        ff_name: str,
        label_pt: str,
        ffinfo_slot: str,
        parquet_column: str,
        embedding_dim: int,
        encoder_subdir: Optional[str],
        requires_keypoints: bool,
    ) -> None:
        self.ff_name = ff_name
        self.label_pt = label_pt
        self.ffinfo_slot = ffinfo_slot
        self.parquet_column = parquet_column
        self.embedding_dim = embedding_dim
        self.encoder_subdir = encoder_subdir
        self.requires_keypoints = requires_keypoints

    def __str__(self) -> str:
        return self.label_pt

    @classmethod
    def from_ff_name(cls, ff_name: str) -> Optional["SepaelModel"]:
        """Acha o membro pelo nome usado pela lib forensicface
        (`sepaelv2`, `sepaelv3`, ...). Case-insensitive. Retorna None
        quando não reconhecido — caller decide se trata como erro ou
        fallback pro default."""
        if not ff_name:
            return None
        key = ff_name.lower()
        for member in cls:
            if member.ff_name == key:
                return member
        return None

    @classmethod
    def from_parquet_column(cls, column: str) -> Optional["SepaelModel"]:
        """Mapping inverso: dado o nome da coluna parquet (ex: `embedding_v3`),
        retorna o membro correspondente. Útil pra `from_parquet` que vê só
        o schema, sem o `ff_name`."""
        if not column:
            return None
        for member in cls:
            if member.parquet_column == column:
                return member
        return None

    @classmethod
    def default(cls) -> "SepaelModel":
        """Modelo default quando o usuário não escolheu explicitamente —
        sepaelv2 por ser o mais leve (ResNet101) e ter sido o padrão
        histórico do FaceAudit."""
        return cls.SEPAEL_V2

    @classmethod
    def models_root(cls) -> Path:
        """Pasta canônica onde a lib `forensicface` busca os ONNX, fixada
        em `~/.forensicface/models/`. Existe sempre que a lib foi usada
        ao menos uma vez na máquina; pode não existir em primeira execução."""
        return Path.home() / ".forensicface" / "models"

    def installed_on_disk(self, models_root: Optional[Path] = None) -> bool:
        """True quando o detector + encoder principal do modelo estão
        no disco. Suporta os dois layouts que a `forensicface >= 0.7.0`
        aceita:

        - **Shared (novo)**:
          `<models_root>/detection/det_10g.onnx` +
          `<models_root>/recognition/<ff_name>/*face*.onnx`.
        - **Legacy (per-model)**:
          `<models_root>/<ff_name>/det_10g.onnx` +
          `<models_root>/<ff_name>/<encoder_subdir>/*.onnx` (onde
          `encoder_subdir` é qualquer pasta que não seja sidecar
          conhecido).

        A `ForensicFace.__init__` da v0.7.0 tenta o layout shared primeiro
        e cai no legacy. Esse helper imita o mesmo comportamento para que
        `available_on_disk()` reflita o que a lib de fato consegue
        carregar — mesmo depois do usuário rodar
        `python -m forensicface.tools.migrate_shared --apply`.
        """
        root = models_root or self.models_root()

        # Tenta layout shared primeiro: detector compartilhado + encoder
        # em `recognition/<ff_name>/`. Pasta `recognition/` pode existir
        # parcialmente populada — exige tanto o detector quanto pelo
        # menos um arquivo `*face*.onnx` na subpasta do modelo.
        shared_detector = root / "detection" / "det_10g.onnx"
        shared_recog_dir = root / "recognition" / self.ff_name
        if shared_detector.exists() and shared_recog_dir.exists():
            if any(shared_recog_dir.glob("*face*.onnx")):
                return True

        # Fallback layout legacy: detector embutido na pasta do modelo +
        # subdir do encoder.
        legacy_root = root / self.ff_name
        if not legacy_root.exists():
            return False
        if not (legacy_root / "det_10g.onnx").exists():
            return False
        sidecar_stems = {"det_10g", "genderage", "1k3d68", "cr_fiqa", "cr_fiqa_l"}
        for onnx_path in legacy_root.rglob("*.onnx"):
            if onnx_path.stem in sidecar_stems:
                continue
            return True
        return False

    @classmethod
    def available_on_disk(
        cls, models_root: Optional[Path] = None,
    ) -> List["SepaelModel"]:
        """Lista membros (em ordem de declaração) cujos ONNX estão
        instalados em `~/.forensicface/models/`. UI usa pra popular
        dropdowns só com o que de fato roda na máquina."""
        return [m for m in cls if m.installed_on_disk(models_root)]

    @classmethod
    def ffinfo_slot_for(cls, ff_name: str) -> str:
        """Helper pra `FFService._slot_for_model`: ff_name → nome do
        atributo em `FFInfo`. Default pro `SEPAEL_V2.ffinfo_slot` quando
        o ff_name não é reconhecido — preserva o comportamento legado
        do FFService antigo (não levantava em modelos desconhecidos).
        """
        member = cls.from_ff_name(ff_name)
        if member is not None:
            return member.ffinfo_slot
        return cls.SEPAEL_V2.ffinfo_slot
