"""Ribbon de operações sobre o conteúdo do dataset (PR Ribbon refactor).

Ordem dos botões consolidada com o usuário:
Auditar, Filtrar, Limpar Filtro, Localizar, Enriquecer, Adicionar Coluna,
Juntar, Dividir, Balancear, Comparar, Estatísticas, Custódia.

Movidos pra outras abas: Galeria + Relatório → Exibir;
→ Estratégia → Arquivo.
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

        # ── Auditar (primeiro — fluxo principal de auditoria forense) ──
        self.btn_analyze = ToolbarButton(
            "Auditar", "🔎",
            tooltip="Auditoria interna de integridade e hard-mining biométrico",
        )
        layout.addWidget(self.btn_analyze)
        layout.addWidget(ToolbarSeparator())

        # ── Filtrar / Limpar Filtro ──
        self.btn_filter = ToolbarButton("Filtrar", "🔍")
        self.btn_clear_filter = ToolbarButton(
            "Limpar Filtro", "🗑️", tooltip="Remover filtros",
        )
        self.btn_clear_filter.setEnabled(False)
        layout.addWidget(self.btn_filter)
        layout.addWidget(self.btn_clear_filter)
        layout.addWidget(ToolbarSeparator())

        # ── Localizar (paths absolutos → relativos / cópia portátil) ──
        self.btn_localize = ToolbarButton(
            "Localizar", "🗂",
            tooltip="Converte paths absolutos em relativos, opcionalmente "
                    "copiando imagens pra pasta autocontida (portátil).",
        )
        layout.addWidget(self.btn_localize)
        layout.addWidget(ToolbarSeparator())

        # ── Enriquecer + Adicionar Coluna (mutações que adicionam dados) ──
        self.btn_enrich = ToolbarButton(
            "Enriquecer", "💎",
            tooltip="Adicionar métricas (OFIQ, SER-FIQ) a um dataset existente",
        )
        self.btn_add_column = ToolbarButton(
            "Coluna", "➕📋",
            tooltip="Adicionar coluna custom ao dataset (anotação livre)",
        )
        layout.addWidget(self.btn_enrich)
        layout.addWidget(self.btn_add_column)
        layout.addWidget(ToolbarSeparator())

        # ── Juntar / Dividir / Balancear (mutações estruturais) ──
        self.btn_merge = ToolbarButton("Juntar", "🔗")
        self.btn_split = ToolbarButton("Dividir", "✂️")
        self.btn_balance = ToolbarButton(
            "Balancear", "⚖",
            tooltip="Balancear amostras por identidade (cap, floor+cap "
                    "ou auto-min) com critério aleatório ou por qualidade",
        )
        layout.addWidget(self.btn_merge)
        layout.addWidget(self.btn_split)
        layout.addWidget(self.btn_balance)
        layout.addWidget(ToolbarSeparator())

        # ── Comparar / Estatísticas (análise — não-mutativa) ──
        self.btn_compare = ToolbarButton(
            "Comparar", "⚖️",
            tooltip="Comparar este dataset com outro",
        )
        self.btn_stats = ToolbarButton("Estatísticas", "📊")
        layout.addWidget(self.btn_compare)
        layout.addWidget(self.btn_stats)
        layout.addWidget(ToolbarSeparator())

        # ── Custódia (audit log) ──
        self.btn_audit_log = ToolbarButton(
            "Custódia", "📜",
            tooltip="Cadeia de custódia forense — histórico de operações "
                    "no dataset com verificação de integridade SHA-256.",
        )
        layout.addWidget(self.btn_audit_log)

        layout.addStretch()

        # Ativa o comportamento responsivo (alterna full ↔ compact por
        # largura). Tem que vir DEPOIS de montar o layout completo.
        self._install_responsive_behavior()
