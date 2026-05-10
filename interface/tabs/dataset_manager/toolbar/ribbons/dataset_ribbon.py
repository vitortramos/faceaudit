"""Ribbon de operações sobre o conteúdo do dataset (PR Ribbon refactor).

Botões: Enriquecer, Juntar, Dividir, Comparar (processamento) +
Estatísticas, Auditar, Relatório, Custódia (análise) + Localizar,
Filtrar, Limpar, Coluna, Galeria (manipulação) + → Estratégia
(exportação).
"""

from PyQt6.QtWidgets import QHBoxLayout

from ..toolbar_widgets import ToolbarButton, ToolbarSeparator
from .responsive_ribbon import ResponsiveRibbon


class DatasetRibbon(ResponsiveRibbon):
    """Ribbon "Dataset" — operações que mexem no conteúdo do dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # ── Grupo: Processamento ──────────────────────────────────────
        self.btn_enrich = ToolbarButton(
            "Enriquecer", "💎",
            tooltip="Adicionar métricas (OFIQ, SER-FIQ) a um dataset existente",
        )
        self.btn_balance = ToolbarButton(
            "Balancear", "⚖",
            tooltip="Balancear amostras por identidade (cap, floor+cap "
                    "ou auto-min) com critério aleatório ou por qualidade",
        )
        self.btn_merge = ToolbarButton("Juntar", "🔗")
        self.btn_split = ToolbarButton("Dividir", "✂️")
        self.btn_compare = ToolbarButton(
            "Comparar", "⚖️",
            tooltip="Comparar este dataset com outro",
        )

        layout.addWidget(self.btn_enrich)
        layout.addWidget(self.btn_balance)
        layout.addWidget(self.btn_merge)
        layout.addWidget(self.btn_split)
        layout.addWidget(self.btn_compare)
        layout.addWidget(ToolbarSeparator())

        # ── Grupo: Análise ────────────────────────────────────────────
        self.btn_stats = ToolbarButton("Estatísticas", "📊")
        self.btn_analyze = ToolbarButton(
            "Auditar", "🔎",
            tooltip="Auditoria interna de integridade e hard-mining biométrico",
        )
        self.btn_schema_report = ToolbarButton(
            "Relatório", "📋",
            tooltip="Relatório de cobertura por coluna (% NaN, distinct, "
                    "min/max). Inclui exportação para CSV/Excel.",
        )
        self.btn_audit_log = ToolbarButton(
            "Custódia", "📜",
            tooltip="Cadeia de custódia forense — histórico de operações "
                    "no dataset com verificação de integridade SHA-256.",
        )

        layout.addWidget(self.btn_stats)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.btn_schema_report)
        layout.addWidget(self.btn_audit_log)
        layout.addWidget(ToolbarSeparator())

        # ── Grupo: Manipulação ────────────────────────────────────────
        self.btn_localize = ToolbarButton(
            "Localizar", "🗂",
            tooltip="Converte paths absolutos em relativos, opcionalmente "
                    "copiando imagens pra pasta autocontida (portátil).",
        )
        self.btn_filter = ToolbarButton("Filtrar", "🔍")
        self.btn_clear_filter = ToolbarButton(
            "Limpar", "🗑️", tooltip="Remover filtros",
        )
        self.btn_clear_filter.setEnabled(False)
        self.btn_add_column = ToolbarButton(
            "Coluna", "➕📋",
            tooltip="Adicionar coluna custom ao dataset (anotação livre)",
        )
        self.btn_toggle_gallery = ToolbarButton(
            "Galeria", "🖼",
            tooltip="Alternar entre visualização de tabela e grid de thumbnails",
        )
        self.btn_toggle_gallery.setCheckable(True)

        layout.addWidget(self.btn_localize)
        layout.addWidget(self.btn_filter)
        layout.addWidget(self.btn_clear_filter)
        layout.addWidget(self.btn_add_column)
        layout.addWidget(self.btn_toggle_gallery)
        layout.addWidget(ToolbarSeparator())

        # ── Grupo: Exportação pra outras abas (PR 17) ─────────────────
        self.btn_send_to_strategy = ToolbarButton(
            "→ Estratégia", "🎯",
            tooltip="Envia o dataset atual para o Strategy Manager V2 — "
                    "switch de aba + import automático no projeto aberto "
                    "(ou prompt pra abrir/criar projeto)",
        )
        layout.addWidget(self.btn_send_to_strategy)

        layout.addStretch()

        # Ativa o comportamento responsivo (alterna full ↔ compact por
        # largura). Tem que vir DEPOIS de montar o layout completo.
        self._install_responsive_behavior()
