"""Snapshot reversível de uma ação de sanitização.

Cada operação aplicada ao dataset (excluir, renomear, mesclar) produz um
`AuditActionRecord` que carrega o que precisa ser feito pra reverter. O
controller mantém uma pilha desses records pra undo encadeado.

`image_path` é a chave estável (mesmo conceito do `AuditIssue.locators`)
— sobrevive a deletes e re-loads do dataset, e não exige índices que
ficariam inválidos depois de mutações estruturais.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from faceaudit.models.face_dataset import FaceDataset
    from faceaudit.models.face_info import FaceInfo


@dataclass
class AuditActionRecord:
    """Registro de uma operação reversível.

    Um único record cobre as 3 ações iniciais:
      - rename: muda `identity` de N faces para um novo valor
      - merge: caso especial de rename múltiplo (várias identidades → 1)
      - delete: remove N faces do dataset

    Os 3 campos abaixo são populados conforme o tipo de op:
      * `identity_changes`: image_path → identity ANTIGA (rename/merge)
      * `deleted_faces`: lista das `FaceInfo` removidas (delete)
    """
    op_id: str
    description: str
    identity_changes: Dict[str, str] = field(default_factory=dict)
    deleted_faces: List["FaceInfo"] = field(default_factory=list)

    def apply_undo(self, dataset: "FaceDataset") -> None:
        """Reverte a operação no `dataset`. Idempotente — chamar duas vezes
        é harmless porque a segunda passa não encontra mais nenhum match."""
        # Restaura identidades
        if self.identity_changes:
            paths_to_restore = self.identity_changes
            for face in dataset.faces:
                old = paths_to_restore.get(getattr(face, "image_path", None))
                if old is not None:
                    face.identity = old

        # Re-insere faces deletadas (volta ao final — ordem original não
        # é preservada, mas o usuário não esperava ordem estável).
        if self.deleted_faces:
            dataset.faces.extend(self.deleted_faces)
