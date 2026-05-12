"""View da Toolbar do Dataset Manager — formato Ribbon (PR Ribbon refactor).

Refator: substitui a toolbar single-row anterior por um `QTabWidget`
com 3 ribbons (Arquivo / Dataset / Imagens) + barra de status dedicada
abaixo, sempre visível.

**Compatibilidade preservada (decisão #5 do usuário)** — a view continua
expondo os mesmos atributos `self.btn_save`, `self.btn_localize`, etc.
via delegação aos ribbons. Controller (`manager_toolbar_controller.py`)
e demais call sites continuam inalterados.

**Decisões consolidadas pelo usuário:**
  1. Salvar na biblioteca → Arquivo; Enviar pra estratégia → Dataset;
     Filtrar/Limpar → Dataset.
  2. Aba Imagens com 2 botões desabilitados como placeholder (opção C).
  3. Default = Arquivo.
  4. Filter status numa barra dedicada abaixo do ribbon (sempre visível).
  5. API pública dos botões mantida via delegação.
"""

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QTabWidget

from .toolbar_widgets import StatusLabel
from .ribbons.arquivo_ribbon import ArquivoRibbon
from .ribbons.dataset_ribbon import DatasetRibbon
from .ribbons.exibir_ribbon import ExibirRibbon
from .ribbons.imagens_ribbon import ImagensRibbon


class ManagerToolbarView(QFrame):
    """Toolbar no formato Ribbon — 3 abas + barra de status dedicada.

    Atalhos `self.btn_xyz` delegam aos ribbons pra preservar API
    pública. Métodos `set_dataset_actions_enabled` e `set_status_text`
    mantidos com mesma assinatura/comportamento.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_style()
        self._init_ui()
        self._wire_button_aliases()

    def _init_style(self):
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Mais alto que a toolbar single-row (90) pra acomodar QTabBar
        # + ribbon row + status bar.
        self.setMaximumHeight(130)
        self.setStyleSheet(
            "QFrame { background-color: #f5f5f5; "
            "border-radius: 5px; border: 1px solid #e0e0e0; }"
        )

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        # ── QTabWidget contendo os 3 ribbons ──
        self.tabs = QTabWidget()
        # Estilo "Office accent bar" — aba selecionada ganha barra azul
        # de 3px no topo + texto azul + bold. Inativas em cinza médio
        # com `border-top transparent` da mesma espessura pra evitar
        # jump visual quando a seleção muda. Hover escurece o fundo
        # da inativa pra dar feedback de mouse-over.
        self.tabs.setStyleSheet(
            # Pane (área dos botões)
            "QTabWidget::pane { "
            "    border: 1px solid #d0d0d0; "
            "    background: white; "
            "    border-radius: 4px; "
            "    top: -1px; "  # encaixa na aba sem gap
            "} "
            # Aba inativa (default)
            "QTabBar::tab { "
            "    background: #e8e8e8; "
            "    color: #555; "
            "    padding: 7px 20px; "
            "    font-size: 11px; "
            "    border: 1px solid #d0d0d0; "
            "    border-top: 3px solid transparent; "  # placeholder pra evitar jump
            "    border-right: none; "
            "    border-top-left-radius: 4px; "
            "    border-top-right-radius: 4px; "
            "    margin-right: 1px; "
            "} "
            # Hover na inativa
            "QTabBar::tab:hover:!selected { "
            "    background: #dcdcdc; "
            "    color: #333; "
            "} "
            # Aba selecionada — accent bar azul + fundo branco contínuo
            "QTabBar::tab:selected { "
            "    background: white; "
            "    color: #1976d2; "
            "    font-weight: bold; "
            "    border-top: 3px solid #1976d2; "
            "    border-bottom: 1px solid white; "  # cobre a borda da pane → continuidade visual
            "} "
            # Última aba não tem margin-right
            "QTabBar::tab:last { margin-right: 0; }"
        )

        self.arquivo_ribbon = ArquivoRibbon()
        self.dataset_ribbon = DatasetRibbon()
        self.exibir_ribbon = ExibirRibbon()
        self.imagens_ribbon = ImagensRibbon()

        self.tabs.addTab(self.arquivo_ribbon, "Arquivo")
        self.tabs.addTab(self.dataset_ribbon, "Dataset")
        self.tabs.addTab(self.exibir_ribbon, "Exibir")
        self.tabs.addTab(self.imagens_ribbon, "Imagens")
        self.tabs.setCurrentIndex(0)  # default: Arquivo (decisão #3)

        main_layout.addWidget(self.tabs)

        # ── Barra de status dedicada (decisão #4 — sempre visível) ──
        self.lbl_status = StatusLabel()
        main_layout.addWidget(self.lbl_status)

    def _wire_button_aliases(self):
        """Cria atalhos `self.btn_<x>` delegando aos ribbons.

        Preserva 100% da API pública anterior — controller e demais
        call sites (PRs 8/15/16/17/24/25/L1d) continuam usando
        `self.view.btn_save` etc. sem mudança.
        """
        # Arquivo
        self.btn_new = self.arquivo_ribbon.btn_new
        self.btn_load = self.arquivo_ribbon.btn_load
        self.btn_save = self.arquivo_ribbon.btn_save
        self.btn_save_as = self.arquivo_ribbon.btn_save_as
        self.btn_close = self.arquivo_ribbon.btn_close
        self.btn_save_to_library = self.arquivo_ribbon.btn_save_to_library
        # Movido de Dataset → Arquivo
        self.btn_send_to_strategy = self.arquivo_ribbon.btn_send_to_strategy

        # Dataset
        self.btn_enrich = self.dataset_ribbon.btn_enrich
        self.btn_balance = self.dataset_ribbon.btn_balance
        self.btn_merge = self.dataset_ribbon.btn_merge
        self.btn_split = self.dataset_ribbon.btn_split
        self.btn_compare = self.dataset_ribbon.btn_compare
        self.btn_stats = self.dataset_ribbon.btn_stats
        self.btn_analyze = self.dataset_ribbon.btn_analyze
        self.btn_audit_log = self.dataset_ribbon.btn_audit_log
        self.btn_localize = self.dataset_ribbon.btn_localize
        self.btn_filter = self.dataset_ribbon.btn_filter
        self.btn_clear_filter = self.dataset_ribbon.btn_clear_filter
        self.btn_add_column = self.dataset_ribbon.btn_add_column

        # Exibir (movidos de Dataset → Exibir)
        self.btn_toggle_gallery = self.exibir_ribbon.btn_toggle_gallery
        self.btn_schema_report = self.exibir_ribbon.btn_schema_report

        # Imagens (placeholder — desabilitados)
        self.btn_image_pipeline_editor = self.imagens_ribbon.btn_image_pipeline_editor
        self.btn_image_pipeline_apply = self.imagens_ribbon.btn_image_pipeline_apply

    # --- API de Manipulação Visual (compatibilidade preservada) ---

    def set_dataset_actions_enabled(self, enabled: bool):
        """Habilita/Desabilita botões que dependem de um dataset
        carregado. Lista mantida estável — qualquer adição futura de
        botão dependente de dataset entra aqui.
        """
        buttons = [
            # Arquivo
            self.btn_close, self.btn_save, self.btn_save_as,
            self.btn_save_to_library,
            # Dataset
            self.btn_enrich, self.btn_balance,
            self.btn_merge, self.btn_split, self.btn_compare,
            self.btn_stats, self.btn_analyze,
            self.btn_schema_report, self.btn_audit_log,
            self.btn_localize, self.btn_filter,
            self.btn_add_column, self.btn_toggle_gallery,
            self.btn_send_to_strategy,
            # Imagens (PR I2 — Editor; PR I5 — Processar)
            self.btn_image_pipeline_editor,
            self.btn_image_pipeline_apply,
        ]
        for btn in buttons:
            btn.setEnabled(enabled)

        # btn_clear_filter tem lógica própria — quando desabilita
        # tudo, força False (mesmo se um filtro estiver ativo).
        if not enabled:
            self.btn_clear_filter.setEnabled(False)

    def set_status_text(self, text: str, is_filtered: bool = False, tooltip: str = ""):
        """Atualiza o texto de status e o estado do botão de filtro.

        Tooltip rich-text (HTML) detalha metadados extras (path, hash,
        descrição, coverage de enriquecimento, métricas detalhadas de
        balanceamento). Hover no status mostra essas infos sem poluir
        a linha principal.
        """
        self.lbl_status.setText(text)
        self.lbl_status.setToolTip(tooltip or text)

        if is_filtered:
            self.btn_clear_filter.setEnabled(True)
            self.btn_filter.set_style("warning")  # Muda cor para amarelo
        else:
            if self.btn_filter.isEnabled():
                self.btn_clear_filter.setEnabled(False)
                self.btn_filter.set_style("default")  # Volta ao normal
