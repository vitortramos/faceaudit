"""Enum de operações registradas no audit log forense (PR 25, Tema 5.4).

Cada `AuditLogEntry` carrega uma `AuditOperation` que descreve o tipo
de evento. A lista cobre apenas eventos audit-críticos: criação,
abertura, persistência, alteração de dados/schema, exportação e
verificação de integridade. Filter/sort/visualização não entram —
não alteram dados nem cadeia de custódia.
"""

from __future__ import annotations

from enum import Enum


class AuditOperation(str, Enum):
    """Operações registradas no audit log de um dataset.

    Convenção: nomes em SCREAMING_SNAKE_CASE; values em snake_case
    (estáveis em disco — JSON sidecar usa o `.value`).
    """

    # Lifecycle
    CREATE = "create"
    OPEN = "open"
    SAVE = "save"
    SAVE_AS = "save_as"
    CLOSE = "close"

    # Mutações de dados
    EDIT_LABEL = "edit_label"
    EDIT_SCHEMA = "edit_schema"  # edição global de metadados/specs via diálogo
    ADD_COLUMN = "add_column"
    REMOVE_COLUMN = "remove_column"
    RENAME_COLUMN = "rename_column"

    # Pipelines de processamento
    ENRICH = "enrich"
    MERGE = "merge"
    SPLIT = "split"
    VIDEO_EXTRACT = "video_extract"  # criação de dataset a partir de vídeo

    # Saída de dados
    EXPORT = "export"

    # PR L1d (Tema Localização) — cópia/relativização de imagens
    COPY_IMAGES = "copy_images"

    # Auditoria meta
    INTEGRITY_VERIFIED = "integrity_verified"
    INTEGRITY_FAILED = "integrity_failed"

    @property
    def emoji(self) -> str:
        return {
            "create": "✨",
            "open": "📂",
            "save": "💾",
            "save_as": "💾",
            "close": "🚪",
            "edit_label": "✏️",
            "edit_schema": "🏷️",
            "add_column": "➕",
            "remove_column": "➖",
            "rename_column": "✏️",
            "enrich": "🔬",
            "merge": "🔗",
            "split": "✂️",
            "video_extract": "🎬",
            "export": "📤",
            "copy_images": "🗂",
            "integrity_verified": "✅",
            "integrity_failed": "❌",
        }.get(self.value, "•")

    @property
    def label_pt(self) -> str:
        """Rótulo em português pra UI (usado na coluna 'Operação')."""
        return {
            "create": "Criação",
            "open": "Abertura",
            "save": "Salvamento",
            "save_as": "Salvar como",
            "close": "Fechamento",
            "edit_label": "Edição de rótulo",
            "edit_schema": "Edição de metadados",
            "add_column": "Coluna adicionada",
            "remove_column": "Coluna removida",
            "rename_column": "Coluna renomeada",
            "enrich": "Enriquecimento",
            "merge": "Concatenação",
            "split": "Divisão",
            "video_extract": "Extração de vídeo",
            "export": "Exportação",
            "copy_images": "Localização (cópia/reescrita de imagens)",
            "integrity_verified": "Integridade verificada",
            "integrity_failed": "Falha de integridade",
        }.get(self.value, self.value)
