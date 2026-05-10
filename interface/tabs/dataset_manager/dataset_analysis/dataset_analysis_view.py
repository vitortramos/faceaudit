"""View do diálogo de Auditoria & Sanitização do dataset.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │ Auditoria Forense — meu_dataset                                   │
  ├─────────────────────┬───────────────────────────────────────────┤
  │ ▼ Integridade Física│ [emoji] Audit label                        │
  │   🔴 47 Paths órf...│   summary text                            │
  │   🔴 12 Hash conf...│ ┌─────────────────────────────────────────┐│
  │   🔴 3 Sem embe...  │ │ Tabela de issues (path, descrição, ...)  ││
  │ ▼ Lógica Rótulos    │ │                                          ││
  │   🟠 18 Singletons  │ └─────────────────────────────────────────┘│
  │ ▼ Qualidade         │ [✓ Selecionar tudo] [Excluir selecionadas]│
  │   ⚪ 5 sem OFIQ     │                                            │
  │   🟠 8 FIQA<30      │                                            │
  ├─────────────────────┴───────────────────────────────────────────┤
  │ [🔍 Mineração Pesada] [🔄 Reanalisar] [Fechar]                   │
  └──────────────────────────────────────────────────────────────────┘

Toda interação real (filtrar, executar ações) vive no controller. A view
só renderiza estado e emite sinais semânticos.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.audit import AuditFinding, AuditIssue, AuditSeverity

from .balance_metrics_widget import BalanceMetricsWidget
from .boxplot_widget import BoxplotWidget
from .histogram_widget import HistogramWidget
from .hubness_tables_widget import HubnessTablesWidget
from .issue_preview_widget import IssuePreviewWidget
from .scatter_widget import ScatterWidget
from .umap_scatter_widget import UmapScatterWidget


# Hints de UI por métrica de qualidade. Cada entrada define o range, o
# step, o número de decimais, o default e o tooltip do spinbox no
# diálogo de configurações. Métricas não listadas aqui caem num default
# genérico (range [0, 100]).
_QUALITY_METRIC_UI: Dict[str, dict] = {
    "FIQA": {
        "label": "FIQA mínimo (ForensicFace):",
        "range": (0.0, 5.0),
        "step": 0.05,
        "decimals": 2,
        "default": 1.50,
        "tooltip": (
            "Faces com FIQA abaixo deste valor entram em 'baixa qualidade'.\n"
            "FIQA = CR-FIQA bruto da forensicface (ratio CCS/NNCCS, NÃO "
            "normalizado): ruim ≈ 1.44, boa ≈ 2.38; topo raramente passa "
            "de 3.0. Default 1.50."
        ),
    },
    "OFIQ": {
        "label": "OFIQ mínimo (ISO/IEC 29794-5):",
        "range": (0.0, 100.0),
        "step": 1.0,
        "decimals": 1,
        "default": 25.0,
        "tooltip": (
            "Escore unificado OFIQ (padrão ISO). Range nominal 0–100, "
            "valores baixos indicam problemas de pose, oclusão, "
            "iluminação. Default 25 marca faces no quartil inferior."
        ),
    },
    "SERFIQ": {
        "label": "SER-FIQ mínimo (estocástico):",
        "range": (0.0, 100.0),
        "step": 1.0,
        "decimals": 2,
        "default": 50.0,
        "tooltip": (
            "Escore SER-FIQ (similaridade entre passes estocásticos do "
            "embedding). Mais alto = embedding mais robusto. Default 50; "
            "ajuste após observar a distribuição empírica."
        ),
    },
    "EMBEDDING_NORM": {
        "label": "Norma do embedding mínima:",
        "range": (0.0, 50.0),
        "step": 0.5,
        "decimals": 2,
        "default": 17.0,
        "tooltip": (
            "Norma L2 do embedding sepaelv2/AdaFace — proxy de "
            "confidência do modelo. Default 17.0 (AdaFace típico ~20–30)."
        ),
    },
}


def _ui_hints_for(metric_name: str) -> dict:
    """Hints de spinbox + default pra uma métrica. Fallback genérico se a
    métrica não estiver no dict — não bloqueia métricas futuras."""
    return _QUALITY_METRIC_UI.get(metric_name, {
        "label": f"{metric_name} mínimo:",
        "range": (0.0, 100.0),
        "step": 1.0,
        "decimals": 2,
        "default": 0.0,
        "tooltip": "",
    })


class AuditSettingsDialog(QtWidgets.QDialog):
    """Diálogo de configuração dos thresholds das auditorias.

    Permite ajustar:
    - **Thresholds de qualidade** (low_quality): um spinbox por métrica
      disponível no dataset (FIQA/OFIQ/SERFIQ via schema, EMBEDDING_NORM
      sempre se há embeddings). Faces com escore abaixo viram issue.
    - **Min amostras/ID** (singleton): IDs com menos amostras que isso
      são flagged
    - **Cosseno mín. duplicate**: pares de centroides de IDs distintos
      com cosseno >= isso → "mesma pessoa em 2+ labels"
    - **Cosseno máx. mixed (outlier)**: faces dentro do mesmo ID com
      cosseno < isso vs centroide → "anotação contaminada"
    - **Min amostras p/ mixed**: IDs com < amostras não são analisadas
      (centroide pouco confiável)
    """

    def __init__(self, parent=None, *,
                 available_quality_metrics: Optional[List[str]] = None,
                 low_quality_thresholds: Optional[Dict[str, float]] = None,
                 min_samples_singleton: int = 2,
                 duplicate_threshold: float = 0.35,
                 mixed_outlier_threshold: float = 0.20,
                 mixed_min_samples: int = 3):
        super().__init__(parent)
        self.setWindowTitle("Configurações da Auditoria")
        self.setMinimumWidth(460)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        # Thresholds de qualidade — um spinbox por métrica disponível.
        # `_quality_spins[metric_name] = QDoubleSpinBox` permite recoletar
        # os valores em get_settings() sem hardcoding.
        self._quality_spins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        thresholds = low_quality_thresholds or {}
        metrics = list(available_quality_metrics or [])
        if metrics:
            quality_label = QtWidgets.QLabel(
                "<b>Qualidade das amostras</b> — limites por métrica disponível:"
            )
            quality_label.setStyleSheet("color: #1a3d6e; padding-top: 2px;")
            form.addRow(quality_label)
            for metric_name in metrics:
                hints = _ui_hints_for(metric_name)
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(*hints["range"])
                spin.setSingleStep(hints["step"])
                spin.setDecimals(hints["decimals"])
                spin.setValue(float(thresholds.get(
                    metric_name, hints["default"],
                )))
                if hints["tooltip"]:
                    spin.setToolTip(hints["tooltip"])
                form.addRow(hints["label"], spin)
                self._quality_spins[metric_name] = spin

            sep0 = QtWidgets.QFrame()
            sep0.setFrameShape(QtWidgets.QFrame.Shape.HLine)
            form.addRow(sep0)

        self.spin_singleton = QtWidgets.QSpinBox()
        self.spin_singleton.setRange(1, 50)
        self.spin_singleton.setValue(min_samples_singleton)
        self.spin_singleton.setToolTip(
            "Identidades com MENOS amostras que isso aparecem em "
            "'identidades com poucas amostras'."
        )
        form.addRow("Min. amostras/ID (singleton):", self.spin_singleton)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        form.addRow(sep)

        self.spin_dup = QtWidgets.QDoubleSpinBox()
        self.spin_dup.setRange(-1.0, 1.0)
        self.spin_dup.setSingleStep(0.01)
        self.spin_dup.setDecimals(3)
        self.spin_dup.setValue(duplicate_threshold)
        self.spin_dup.setToolTip(
            "Pares de centroides com cosseno >= isso são flaggeados como "
            "'mesma pessoa em IDs distintos'. Aceita toda a faixa [-1, 1] "
            "para permitir varreduras agressivas em datasets ruidosos."
        )
        form.addRow("Cosseno mínimo (identidades duplicadas):", self.spin_dup)

        self.spin_mixed = QtWidgets.QDoubleSpinBox()
        self.spin_mixed.setRange(-1.0, 1.0)
        self.spin_mixed.setSingleStep(0.05)
        self.spin_mixed.setDecimals(3)
        self.spin_mixed.setValue(mixed_outlier_threshold)
        self.spin_mixed.setToolTip(
            "Faces com cosseno ABAIXO disso vs o centroide do próprio ID "
            "são consideradas outliers — provável anotação contaminada. "
            "Default 0.20 (zona conservadora; aumentar pega mais casos "
            "borderline ao custo de mais falsos positivos)."
        )
        form.addRow("Cosseno outlier (identidades misturadas):", self.spin_mixed)

        self.spin_mixed_min = QtWidgets.QSpinBox()
        self.spin_mixed_min.setRange(2, 100)
        self.spin_mixed_min.setValue(mixed_min_samples)
        self.spin_mixed_min.setToolTip(
            "IDs com menos amostras que isso são puladas — centroide "
            "calculado sobre poucas faces é instável."
        )
        form.addRow("Min. amostras p/ análise mixed:", self.spin_mixed_min)

        layout.addLayout(form)

        note = QtWidgets.QLabel(
            "<i>Thresholds de qualidade e min/ID afetam apenas as auditorias "
            "rápidas (recalculadas ao salvar). Os thresholds de identidades "
            "duplicadas/misturadas exigem rerodar 'Auditar identidades' "
            "depois.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555; font-size: 11px; padding-top: 4px;")
        layout.addWidget(note)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setText("Salvar")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_settings(self) -> dict:
        return {
            "low_quality_thresholds": {
                name: spin.value()
                for name, spin in self._quality_spins.items()
            },
            "min_samples_singleton": self.spin_singleton.value(),
            "duplicate_threshold": self.spin_dup.value(),
            "mixed_outlier_threshold": self.spin_mixed.value(),
            "mixed_min_samples": self.spin_mixed_min.value(),
        }


_FINDING_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1
_ISSUE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 2  # AuditIssue inteiro na linha

# Categorias cujo conceito de "total de problemas" não se aplica —
# findings nelas são sempre informativos (histogramas, correlações,
# métricas de balanceamento, diagnóstico de hubness). A coluna "Total"
# da tree fica em branco pra essas em vez de mostrar "0" enganoso.
_INFORMATIONAL_CATEGORIES = frozenset({
    "Histogramas",
    "Correlações",
    "Balanceamento",
    "Hubness",
    "Visualização",
})

# Cap de linhas exibidas na tabela. Acima disso, populamos só as primeiras
# N e exibimos um footer informando o cap. Isso evita travar a main thread
# em datasets onde uma auditoria O(N) acha milhares de issues (ex: orphan
# paths num dataset com paths quebrados em massa). O auditor ainda pode
# usar "Excluir todos" — a ação batch usa as referências internas, não a
# tabela visível.
_TABLE_VISIBLE_CAP = 100

# Texto explicativo por audit_id — mostrado acima da tabela para dar
# contexto ao perito sobre o que aquela auditoria detecta. Auditorias
# sem entrada aqui simplesmente não exibem a caixa de ajuda.
_AUDIT_HELP: Dict[str, str] = {
    "duplicate_identities": (
        "<b>Pessoas com mais de um rótulo</b> — detecta quando <b>duas "
        "identidades distintas</b> no dataset <b>podem ser a mesma pessoa</b>. "
        "O algoritmo calcula o centroide (vetor médio dos embeddings) de "
        "cada identidade e compara os centroides duas a duas. Pares com "
        "cosseno alto entre centroides indicam que os IDs deveriam ser "
        "fundidos. Sinal de problema: <b>cosseno alto</b> entre IDs distintos."
    ),
    "mixed_identities": (
        "<b>Faces com erro de identidade</b> — detecta <b>fotos rotuladas "
        "incorretamente</b> dentro de uma identidade. Pra cada foto X de um "
        "ID, o algoritmo calcula o centroide das <i>outras</i> fotos do mesmo "
        "ID (leave-one-out, excluindo a própria X) e mede o cosseno entre X e "
        "esse centroide. Cosseno baixo significa que X parece ser de outra "
        "pessoa — provável anotação errada. Sinal de problema: "
        "<b>cosseno baixo</b> dentro do mesmo ID."
    ),
    "histogram_fiqa": (
        "<b>Histograma de FIQA</b> — distribuição empírica do escore "
        "de qualidade FIQA (CR-FIQA bruto da forensicface) sobre o "
        "dataset. Útil pra escolher um threshold adequado em "
        "'Configurações → FIQA mínimo' baseado na cauda inferior real "
        "(em vez de chutar)."
    ),
    "histogram_embedding_norm": (
        "<b>Histograma da norma do embedding</b> — distribuição da "
        "norma L2 do embedding sepaelv2/AdaFace por face. A norma "
        "funciona como proxy de confiança do modelo: faces nítidas e "
        "frontais geram normas mais altas; faces problemáticas (pose "
        "extrema, oclusão, baixa nitidez) geram normas mais baixas."
    ),
    "correlation_fiqa_norm": (
        "<b>FIQA × Norma do embedding</b> — relação entre o escore "
        "de qualidade FIQA e a norma do embedding. Esperado: <b>positiva</b> "
        "(FIQA alto → norma alta). Pearson r próximo de 0 ou negativo "
        "sugere que o modelo de embedding e o detector de qualidade "
        "discordam — alguma das duas métricas pode estar comprometida. "
        "O quadrante <b>FIQA alto + norma baixa</b> (canto inferior "
        "direito) é o suspeito: faces que o FIQA aprova mas o embedding "
        "não consegue representar bem."
    ),
    "boxplot_norm_by_fiqa_quartile": (
        "<b>Norma por quartil de FIQA</b> — boxplot da norma do "
        "embedding estratificada nos 4 quartis do FIQA. Útil pra "
        "validar uma correlação aparente: se as <b>medianas</b> dos 4 "
        "quartis formam uma escada monotônica (Q1 → Q4 cresce ou "
        "decresce), a correlação é real e sistemática; se ficam quase "
        "no mesmo nível, a correlação observada no scatter é artefato "
        "de outliers nas caudas."
    ),
    "histogram_samples_per_identity": (
        "<b>Histograma de amostras por identidade</b> — distribuição "
        "do número de fotos por ID no dataset. Cauda esquerda mostra "
        "as 10 identidades com <b>menos</b> amostras (singletons → "
        "candidatas a remoção ou enriquecimento). Cauda direita mostra "
        "as 10 com <b>mais</b> amostras (possíveis 'celebridades' que "
        "dominam o dataset e podem enviesar centroides/calibração)."
    ),
    "balance_metrics": (
        "<b>Métricas de balanceamento</b> — verdict consolidado "
        "(Balanceado / Aceitável / Moderado / Severo) + estatísticas "
        "das contagens por identidade (CV, razão max/min, min/max/"
        "mediana/média/std). Verdict é o pior caso entre CV (≤0.35 "
        "aceitável) e razão max/min (≤4 aceitável)."
    ),
    "umap_embeddings": (
        "<b>UMAP — projeção 2D/3D dos embeddings</b> — visualização "
        "interativa do espaço de embeddings. Em modo <b>por face</b>, "
        "cada ponto é uma foto. Em modo <b>por identidade (centroides)</b>, "
        "cada ponto é uma pessoa (centroide L2-normalizado das fotos "
        "dela), e clicar num ponto abre o painel lateral com as N "
        "identidades mais parecidas medidas por cosseno em 512D — "
        "ferramenta forense pra encontrar lookalikes. Use o dropdown "
        "<b>Colorir por</b> pra alternar entre identidade, sexo, etnia, "
        "etc. — <b>distâncias relativas entre clusters</b> são "
        "razoavelmente confiáveis no UMAP (diferente de t-SNE), mas "
        "magnitudes absolutas seguem sendo arbitrárias. Cada execução "
        "vira um item próprio na tree — pode comparar configurações "
        "sem perder anteriores."
    ),
    "hubness_tables": (
        "<b>Top hubs e anti-hubs (tabela)</b> — listagem das 50 faces "
        "mais hub (Nk alto, atraem vizinhança errada) e 50 mais "
        "anti-hub (Nk=0, faces topologicamente isoladas), com "
        "<b>Identidade / Foto / FIQA / Norma / Nk</b> em cada linha. "
        "Selecione uma linha pra ver a face no preview embaixo. Útil "
        "pra inspecionar visualmente se as hubs são fotos genuinamente "
        "ruins (esperado) ou faces normais (sintoma de problema do "
        "modelo de embedding)."
    ),
    "histogram_hubness_nk": (
        "<b>Histograma de Nk (hubness)</b> — para cada face, conta "
        "quantas vezes ela aparece nos top-K vizinhos de OUTRAS faces "
        "de identidades DIFERENTES (Nk score). <b>Hubs</b> (cauda "
        "direita, Nk alto) atraem vizinhança errada — sintoma clássico "
        "de viés topológico em alta dimensionalidade; tendem a virar "
        "falsos positivos em reconhecimento. <b>Anti-hubs</b> (Nk=0) "
        "são faces topologicamente isoladas — nunca aparecem como "
        "vizinhas, podem ser outliers ou faces de baixa qualidade. "
        "Em dataset bem-comportado, distribuição de Nk é aproximadamente "
        "normal centrada em K; cauda longa à direita indica hubness."
    ),
    "hard_genuines": (
        "<b>Hard genuines</b> — pares de fotos da <i>mesma identidade</i> "
        "com similaridade <b>baixa</b>. Risco de falsos negativos (FN) — "
        "fotos da mesma pessoa que o sistema não reconhece como tal."
    ),
    "hard_impostors": (
        "<b>Hard impostors</b> — pares de fotos de identidades <i>distintas</i> "
        "com similaridade <b>alta</b>. Risco de falsos positivos (FP) — "
        "pessoas diferentes que o sistema poderia confundir."
    ),
}


# Famílias de audit_id cujo prefixo serve pra lookup do help text —
# usado por auditorias appendable que geram audit_ids únicos por
# execução (ex: `umap_embeddings_<uuid>`).
_AUDIT_HELP_PREFIXES = (
    "umap_embeddings",  # cada execução de UMAP tem ID único
)


def _resolve_audit_help(audit_id: str) -> Optional[str]:
    """Resolve o help text pra um audit_id. Match exato primeiro;
    senão tenta prefixos registrados em `_AUDIT_HELP_PREFIXES`."""
    txt = _AUDIT_HELP.get(audit_id)
    if txt is not None:
        return txt
    for prefix in _AUDIT_HELP_PREFIXES:
        if audit_id.startswith(prefix):
            return _AUDIT_HELP.get(prefix)
    return None


class DatasetAnalysisView(QtWidgets.QDialog):
    """View do diálogo de auditoria. Stateless o quanto possível —
    o controller aplica as ações."""

    # Sinais semânticos pro controller
    finding_selected = QtCore.pyqtSignal(object)              # AuditFinding
    delete_issues_requested = QtCore.pyqtSignal(list)         # List[str] de image_paths
    rerun_requested = QtCore.pyqtSignal()
    run_heavy_mining_requested = QtCore.pyqtSignal()
    run_biometric_audit_requested = QtCore.pyqtSignal()       # PR 13
    run_hubness_audit_requested = QtCore.pyqtSignal()         # diagnóstico Nk
    run_umap_audit_requested = QtCore.pyqtSignal()            # projeção UMAP

    # PR 14 — sanitização de identidades + undo
    rename_identity_requested = QtCore.pyqtSignal(list)       # List[str] de image_paths
    merge_identities_requested = QtCore.pyqtSignal(list)      # List[str] de image_paths
    undo_requested = QtCore.pyqtSignal()                       # desfaz último (Ctrl+Z)
    # Desfazer uma ação específica do histórico (não necessariamente a
    # última). Payload: op_id da ação.
    undo_action_at_requested = QtCore.pyqtSignal(str)
    # Settings — emitido quando o user salva o diálogo de configurações.
    # Payload: dict com as chaves fiqa_threshold, min_samples_singleton,
    # duplicate_threshold, mixed_outlier_threshold, mixed_min_samples.
    audit_settings_changed = QtCore.pyqtSignal(dict)

    def __init__(self, dataset_name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Auditoria Forense — {dataset_name}")
        self.resize(1100, 700)
        # Adiciona Minimize/Maximize ao chrome da janela. QDialog default
        # vem só com Close — e em fluxos longos (auditoria + sanitização
        # batch) faz falta poder maximizar a janela.
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )

        self._findings: List[AuditFinding] = []
        self._build_ui()

    # ── Construção da UI ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # Header geral
        self.lbl_header = QtWidgets.QLabel("Carregando auditoria…")
        self.lbl_header.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #1a3d6e;"
        )
        outer.addWidget(self.lbl_header)

        # ── Toolbar (topo) ──
        # Ordem: Configurações, Reanalisar, Auditoria de IDs, Mineração Biométrica.
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_settings = QtWidgets.QPushButton("⚙ Configurações")
        self.btn_settings.setToolTip(
            "Ajusta os thresholds das auditorias (FIQA, min amostras/ID, "
            "duplicates, mixed)."
        )
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        # Cache dos settings atuais — usado pra preencher o diálogo na
        # próxima abertura. Atualizados quando o controller chama
        # `set_audit_settings(...)` ou quando o user salva o dialog.
        # `low_quality_thresholds` é um dict `{metric.name: threshold}`
        # — o controller decide quais métricas estão disponíveis no
        # dataset (via `set_quality_metric_options`) e o diálogo só
        # mostra spinboxes pra essas.
        self._audit_settings: dict = {
            "low_quality_thresholds": {},
            "min_samples_singleton": 2,
            "duplicate_threshold": 0.35,
            "mixed_outlier_threshold": 0.20,
            "mixed_min_samples": 3,
        }
        self._available_quality_metrics: List[str] = []

        self.btn_rerun = QtWidgets.QPushButton("🔄 Reanalisar")
        self.btn_rerun.setToolTip(
            "Reexecuta as auditorias rápidas (após edição manual no dataset)."
        )
        self.btn_rerun.clicked.connect(self.rerun_requested.emit)

        self.btn_run_biometric = QtWidgets.QPushButton("🔬 Auditoria de IDs (pesada)")
        self.btn_run_biometric.setToolTip(
            "Detecta identidades duplicadas (mesma pessoa em 2+ labels) e "
            "identidades misturadas (label cobrindo 2+ pessoas). Usa "
            "centróides de embedding — leva poucos segundos a alguns minutos."
        )
        self.btn_run_biometric.clicked.connect(self.run_biometric_audit_requested.emit)

        self.btn_run_mining = QtWidgets.QPushButton("🔍 Mineração Biométrica (pesada)")
        self.btn_run_mining.setToolTip(
            "Roda hard mining O(N²) — pares de mesma identidade com baixa "
            "similaridade e pares de identidades distintas com alta similaridade. "
            "Pode levar minutos em datasets grandes."
        )
        self.btn_run_mining.clicked.connect(self.run_heavy_mining_requested.emit)

        self.btn_run_hubness = QtWidgets.QPushButton("🌐 Hubness (pesada)")
        self.btn_run_hubness.setToolTip(
            "Análise de hubness: KNN exaustivo O(N²·D) que computa, "
            "para cada face, quantas vezes ela aparece como vizinha "
            "errada de outras faces (Nk score). Identifica hubs (Nk "
            "alto, atraem vizinhança errada) e anti-hubs (Nk=0, faces "
            "topologicamente isoladas). Pode levar segundos a minutos."
        )
        self.btn_run_hubness.clicked.connect(self.run_hubness_audit_requested.emit)

        self.btn_run_umap = QtWidgets.QPushButton("🌌 UMAP (Embeddings)")
        self.btn_run_umap.setToolTip(
            "Projeção UMAP dos embeddings em 2D — agrupa faces "
            "similares no espaço visual pra ver clusters por "
            "identidade, gênero, etnia, origem, FIQA, etc. Abre "
            "diálogo de configuração de hyperparams antes de rodar. "
            "Pode levar segundos a minutos dependendo do dataset."
        )
        self.btn_run_umap.clicked.connect(self.run_umap_audit_requested.emit)

        toolbar.addWidget(self.btn_settings)
        toolbar.addWidget(self.btn_rerun)
        toolbar.addWidget(self.btn_run_biometric)
        toolbar.addWidget(self.btn_run_mining)
        toolbar.addWidget(self.btn_run_hubness)
        toolbar.addWidget(self.btn_run_umap)
        toolbar.addStretch()
        outer.addLayout(toolbar)

        # Atalho global Ctrl+Z pra desfazer última ação. Mantido para
        # compatibilidade com o resto do sistema; a interface preferencial
        # de undo agora é a lista de histórico abaixo da árvore.
        self._undo_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Undo, self
        )
        self._undo_shortcut.activated.connect(self.undo_requested.emit)
        # Atributos legados que outros módulos podem inspecionar — não
        # criamos mais o QPushButton.
        self.btn_undo = None  # placeholder p/ compatibilidade
        self.btn_close = None

        # Splitter principal
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        # ── Esquerda: vertical splitter (tree em cima, histórico embaixo) ──
        left_pane = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(left_pane)

        # Tree de auditorias
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Auditoria", "Total"])
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(True)
        # Coluna 0 (Auditoria) cresce pra absorver o espaço; coluna 1
        # (Total) usa só o necessário pros números curtos. Sem isso, o
        # default fixo de 230 px na coluna 0 corta os rótulos longos
        # tipo "Histograma de amostras por identidade".
        tree_header = self.tree.header()
        tree_header.setStretchLastSection(False)
        tree_header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        tree_header.setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
        )
        # Reserva mínimo razoável pra coluna Total (evita ela colapsar
        # a 0 quando vazia). 50 px = 4–5 dígitos folgados.
        tree_header.setMinimumSectionSize(50)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection)
        left_pane.addWidget(self.tree)

        # Histórico de mudanças no dataset (delete/rename/merge). Cada row
        # tem descrição + botão "↶ Desfazer". Serve como audit trail
        # visual + interface de undo granular (qualquer ação, não só a
        # última como o Ctrl+Z faria).
        history_frame = QtWidgets.QFrame()
        history_layout = QtWidgets.QVBoxLayout(history_frame)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(2)

        lbl_history = QtWidgets.QLabel("📜 Histórico de mudanças")
        lbl_history.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #1a3d6e; padding-top: 4px;"
        )
        history_layout.addWidget(lbl_history)

        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(2)
        self.history_table.setHorizontalHeaderLabels(["Ação", ""])
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self.history_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setShowGrid(False)
        history_header = self.history_table.horizontalHeader()
        history_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        history_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        history_layout.addWidget(self.history_table, 1)

        left_pane.addWidget(history_frame)
        # Tree ocupa ~70% da altura, histórico ~30%. User pode arrastar.
        left_pane.setSizes([400, 200])

        # ── Direita: painel de detalhes ──
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # Header do finding selecionado
        self.lbl_finding_title = QtWidgets.QLabel("Selecione uma auditoria à esquerda.")
        self.lbl_finding_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #333;"
        )
        right_layout.addWidget(self.lbl_finding_title)

        self.lbl_finding_summary = QtWidgets.QLabel("")
        self.lbl_finding_summary.setWordWrap(True)
        self.lbl_finding_summary.setStyleSheet("color: #666; font-size: 11px;")
        right_layout.addWidget(self.lbl_finding_summary)

        # Caixa de explicação contextual (visível só pra auditorias que
        # têm semântica não-óbvia). Atualizada em `_render_finding`
        # baseado no audit_id.
        self.lbl_audit_help = QtWidgets.QLabel("")
        self.lbl_audit_help.setWordWrap(True)
        self.lbl_audit_help.setStyleSheet(
            "color: #1a3d6e; background: #e3f2fd; "
            "border-left: 3px solid #1976d2; "
            "padding: 8px 10px; font-size: 11px;"
        )
        self.lbl_audit_help.hide()
        right_layout.addWidget(self.lbl_audit_help)

        # Stack de painéis à direita: detalhes (tabela+preview) ou
        # histograma. A página exibida muda em `_render_finding` baseada
        # no audit_id selecionado.
        self._right_stack = QtWidgets.QStackedWidget()
        right_layout.addWidget(self._right_stack, 1)

        # Página 0 — Splitter vertical: tabela em cima, preview abaixo.
        details_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._right_stack.addWidget(details_splitter)

        # Página 1 — Histograma (matplotlib FigureCanvas + toolbar)
        self._histogram_widget = HistogramWidget()
        self._right_stack.addWidget(self._histogram_widget)

        # Página 2 — Scatter de correlação entre duas métricas
        self._scatter_widget = ScatterWidget()
        self._right_stack.addWidget(self._scatter_widget)

        # Página 3 — Boxplot estratificado (uma métrica por quartil de outra)
        self._boxplot_widget = BoxplotWidget()
        self._right_stack.addWidget(self._boxplot_widget)

        # Página 4 — Card de métricas de balanceamento (Qt-puro)
        self._balance_metrics_widget = BalanceMetricsWidget()
        self._right_stack.addWidget(self._balance_metrics_widget)

        # Página 5 — Tabelas de hubness (top hubs + anti-hubs + preview)
        self._hubness_tables_widget = HubnessTablesWidget()
        self._right_stack.addWidget(self._hubness_tables_widget)

        # Página 6 — UMAP scatter com combobox de coloração
        self._umap_scatter_widget = UmapScatterWidget()
        self._right_stack.addWidget(self._umap_scatter_widget)

        # Tabela de issues
        self.table_issues = QtWidgets.QTableWidget()
        self.table_issues.setColumnCount(3)
        self.table_issues.setHorizontalHeaderLabels(["Locator", "Descrição", "Detalhes"])
        self.table_issues.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table_issues.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.table_issues.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        header = self.table_issues.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # Quando a seleção muda, atualiza o preview com os paths das
        # linhas selecionadas (deduplicados, com caption de identidade).
        self.table_issues.itemSelectionChanged.connect(self._on_issue_selection_changed)
        details_splitter.addWidget(self.table_issues)

        # Preview: thumbnails das faces da issue selecionada
        self.preview = IssuePreviewWidget()
        # Quando o user clica num thumb, atualiza estados dos botões
        # batch (Renomear/Mesclar) com base nas fotos efetivamente
        # marcadas no preview.
        self.preview.selection_changed.connect(self._on_preview_selection_changed)
        details_splitter.addWidget(self.preview)

        # Tabela em cima ocupa ~55%, preview ~45% — preview precisa
        # acomodar as thumbs de 200px.
        details_splitter.setSizes([280, 280])

        # Barra de ações batch — Sel/Limpar à esquerda, ações destrutivas
        # à direita. Wrap num QWidget pra permitir esconder a barra
        # inteira quando o painel direito é trocado pra histograma
        # (não há ações sobre histogramas).
        self._actions_widget = QtWidgets.QWidget()
        actions_row = QtWidgets.QHBoxLayout(self._actions_widget)
        actions_row.setContentsMargins(0, 0, 0, 0)
        self.btn_select_all = QtWidgets.QPushButton("☑ Sel. tudo")
        self.btn_select_all.setToolTip(
            "Seleciona tudo. Em auditorias par-a-par, seleciona todas as "
            "imagens visíveis no preview; em auditorias normais, todas as "
            "linhas da tabela."
        )
        self.btn_select_all.clicked.connect(self._on_select_all_clicked)

        self.btn_select_none = QtWidgets.QPushButton("☐ Limpar")
        self.btn_select_none.setToolTip(
            "Limpa a seleção corrente (preview ou tabela)."
        )
        self.btn_select_none.clicked.connect(self._on_select_none_clicked)

        # PR 14: ações de sanitização de identidade (estado dinâmico).
        self.btn_rename = QtWidgets.QPushButton("✏️ Renomear ID")
        self.btn_rename.setToolTip(
            "Renomeia em lote TODAS as faces com a identidade selecionada. "
            "Habilitado quando as faces selecionadas têm uma única identidade."
        )
        self.btn_rename.clicked.connect(self._on_rename_clicked)
        self.btn_rename.setEnabled(False)

        self.btn_merge = QtWidgets.QPushButton("🔗 Mesclar IDs")
        self.btn_merge.setToolTip(
            "Funde duas (ou mais) identidades em uma única — útil quando o "
            "auditor de duplicatas detectou que são a mesma pessoa. "
            "Habilitado quando há 2+ identidades distintas na seleção."
        )
        self.btn_merge.clicked.connect(self._on_merge_clicked)
        self.btn_merge.setEnabled(False)

        self.btn_delete = QtWidgets.QPushButton("🗑 Excluir face(s)")
        self.btn_delete.setStyleSheet(
            "background-color: #c62828; color: white; font-weight: bold; padding: 4px 10px;"
        )
        self.btn_delete.clicked.connect(self._on_delete_clicked)
        self.btn_delete.setEnabled(False)

        # Esquerda: seleção. Direita: ações destrutivas/transformação.
        actions_row.addWidget(self.btn_select_all)
        actions_row.addWidget(self.btn_select_none)
        actions_row.addStretch()
        actions_row.addWidget(self.btn_rename)
        actions_row.addWidget(self.btn_merge)
        actions_row.addWidget(self.btn_delete)
        right_layout.addWidget(self._actions_widget)

        splitter.addWidget(right_widget)
        splitter.setSizes([280, 820])

    def set_active_marks(self, deleted: set, renamed: dict) -> None:
        """Propaga ao preview os paths que foram excluídos/renomeados
        no histórico ativo. As cells correspondentes recebem estilo
        diferente — vermelho com banner "EXCLUÍDA" pra deletadas,
        amarelo com "ID: old → new" pra renomeadas. Chamado pelo
        controller após cada delete/rename/merge/undo."""
        self.preview.set_active_marks(deleted, renamed)

    def set_dataset(self, dataset) -> None:
        """Propaga referência ao dataset corrente pros 4 widgets que
        carregam thumbnails — em modo RELATIVE (Image Pipeline output)
        os paths precisam ser resolvidos contra `_parquet_dir/path_root`
        antes do QPixmap. Sem isso, miniaturas/previews ficam vazios."""
        for w in (
            self.preview, self._histogram_widget,
            self._hubness_tables_widget, self._umap_scatter_widget,
        ):
            if hasattr(w, "set_dataset"):
                w.set_dataset(dataset)

    def set_audit_settings(self, settings: dict) -> None:
        """Atualiza o cache local de thresholds — chamado pelo controller
        após resolver os defaults (e após cada Save do diálogo). Faz
        merge raso, com tratamento especial pro dict aninhado de
        `low_quality_thresholds` pra não dropar métricas não tocadas."""
        for key, value in settings.items():
            if key == "low_quality_thresholds" and isinstance(value, dict):
                self._audit_settings.setdefault("low_quality_thresholds", {})
                self._audit_settings["low_quality_thresholds"].update(value)
            else:
                self._audit_settings[key] = value

    def set_quality_metric_options(self, metrics: List[str]) -> None:
        """Define quais métricas de qualidade devem aparecer no diálogo
        de configurações. Lista de `QualityScoreType.name` (e.g.
        `["FIQA", "OFIQ", "EMBEDDING_NORM"]`). Chamado pelo controller
        uma vez na inicialização após inspecionar o dataset."""
        self._available_quality_metrics = list(metrics)

    def _open_settings_dialog(self) -> None:
        dlg = AuditSettingsDialog(
            self,
            available_quality_metrics=self._available_quality_metrics,
            **self._audit_settings,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_settings = dlg.get_settings()
            self.set_audit_settings(new_settings)
            self.audit_settings_changed.emit(dict(self._audit_settings))

    # ── API pública (chamada pelo controller) ────────────────────────────

    def set_findings(self, findings: Sequence[AuditFinding]) -> None:
        """Atualiza a árvore com a lista nova de findings.

        Agrupa por `category` preservando a ordem de inserção (os baratos
        rodam primeiro, os pesados são adicionados depois). Severity é
        propagada do maior pro menor por categoria via emoji do nó pai.
        """
        self._findings = list(findings)
        previous_id = self._currently_selected_audit_id()

        self.tree.blockSignals(True)
        self.tree.clear()

        # Agrupa preservando ordem de aparição
        category_order: List[str] = []
        by_category: dict = {}
        for f in self._findings:
            if f.category not in by_category:
                by_category[f.category] = []
                category_order.append(f.category)
            by_category[f.category].append(f)

        node_to_select: Optional[QtWidgets.QTreeWidgetItem] = None

        for cat in category_order:
            children = by_category[cat]
            cat_item = QtWidgets.QTreeWidgetItem(self.tree)
            cat_item.setText(0, cat)
            # Categorias informativas (Histogramas, Correlações,
            # Balanceamento, Hubness) não têm noção de "total de
            # problemas" — leave em branco em vez de mostrar "0", que
            # confunde o leitor.
            if cat in _INFORMATIONAL_CATEGORIES:
                cat_item.setText(1, "")
            else:
                cat_item.setText(1, str(sum(f.count for f in children)))
            cat_item.setExpanded(True)

            font = cat_item.font(0)
            font.setBold(True)
            cat_item.setFont(0, font)
            cat_item.setFont(1, font)

            for finding in children:
                child = QtWidgets.QTreeWidgetItem(cat_item)
                # Bolinha por estado, não por severidade intrínseca:
                # contagem == 0 → verde (auditoria limpa); > 0 → cor da
                # severidade do tipo de auditoria.
                if finding.count > 0:
                    emoji = finding.severity.emoji
                    child.setForeground(
                        0, QtGui.QColor(_severity_color(finding.severity)),
                    )
                else:
                    emoji = "🟢"
                    child.setForeground(0, QtGui.QColor("#2e7d32"))
                child.setText(0, f"{emoji}  {finding.audit_label}")
                # Coluna Total: mesma lógica do nó-categoria — leave em
                # branco pra findings informativos.
                if finding.category in _INFORMATIONAL_CATEGORIES:
                    child.setText(1, "")
                else:
                    child.setText(1, str(finding.count))
                child.setData(0, _FINDING_ROLE, finding)
                if previous_id is not None and finding.audit_id == previous_id:
                    node_to_select = child

        # Resumo no header
        n_total = sum(f.count for f in self._findings)
        n_critical = sum(
            f.count for f in self._findings
            if f.severity == AuditSeverity.CRITICAL
        )
        self.lbl_header.setText(
            f"Auditoria — {n_total} issue(s) detectada(s) | "
            f"{n_critical} crítica(s)"
        )

        self.tree.blockSignals(False)

        # Seleciona o nó previamente ativo (ou o primeiro com count > 0)
        if node_to_select is None:
            node_to_select = self._first_finding_with_issues()
        if node_to_select is not None:
            self.tree.setCurrentItem(node_to_select)
        else:
            self._render_empty_finding()

    def append_findings(self, extra: Sequence[AuditFinding]) -> None:
        """Adiciona novos findings à lista atual sem perder os existentes.

        Usado quando o hard mining termina e queremos juntar seus resultados
        à árvore sem rerunar as auditorias baratas."""
        self.set_findings(list(self._findings) + list(extra))

    # ── Slots internos ───────────────────────────────────────────────────

    def _on_tree_selection(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            self._render_empty_finding()
            return
        finding = items[0].data(0, _FINDING_ROLE)
        if finding is None:
            self._render_empty_finding()
            return
        self._render_finding(finding)
        self.finding_selected.emit(finding)

    # Auditorias par-a-par cuja unidade de ação é a foto individual
    # (não a row da tabela): para essas, as ações de Excluir / Renomear
    # / Mesclar operam nos thumbs selecionados no preview.
    _PREVIEW_DRIVEN_AUDITS = (
        "duplicate_identities", "mixed_identities",
        "hard_genuines", "hard_impostors",
        # Hash conflicts e duplicatas exatas têm grupo de N≥2 faces.
        # Sem preview-driven, "Excluir face(s)" remove TODAS as do grupo
        # — usuário não consegue escolher 1. Adicionando aqui, a seleção
        # passa pelo preview (clicar thumb da face específica).
        "hash_conflicts", "exact_duplicates",
    )

    def _action_paths(self) -> List[str]:
        """Resolve quais paths a ação atual deve operar.

        Pra auditorias par-a-par (ver `_PREVIEW_DRIVEN_AUDITS`), usa os
        thumbs selecionados no preview — o usuário clica nas fotos que
        quer. Pra demais auditorias (orphan paths, hash conflicts, etc),
        cai no caminho legado de seleção via row da tabela.
        """
        audit_id = self._currently_selected_audit_id()
        if audit_id in self._PREVIEW_DRIVEN_AUDITS:
            return self.preview.selected_paths()
        return self._selected_locators()

    def _on_select_all_clicked(self) -> None:
        """Seleciona tudo. Em auditorias par-a-par, opera no preview
        (todas as imagens visíveis); em auditorias normais, opera na
        tabela (legado)."""
        if self._currently_selected_audit_id() in self._PREVIEW_DRIVEN_AUDITS:
            self.preview.select_all()
        else:
            self.table_issues.selectAll()

    def _on_select_none_clicked(self) -> None:
        """Limpa a seleção atual (preview se par-a-par; tabela caso
        contrário)."""
        if self._currently_selected_audit_id() in self._PREVIEW_DRIVEN_AUDITS:
            self.preview.clear_selection()
            # `clear_selection` não emite sinal — chamamos manual pra
            # manter os botões batch sincronizados.
            self._on_preview_selection_changed()
        else:
            self.table_issues.clearSelection()

    def _on_delete_clicked(self) -> None:
        paths = self._action_paths()
        if not paths:
            QtWidgets.QMessageBox.information(
                self, "Selecione fotos",
                "Selecione no preview as fotos que quer excluir "
                "(clique para selecionar; Ctrl+clique para múltiplas).",
            )
            return
        # Sem confirmação — a operação é reversível via histórico
        # (Ctrl+Z ou botão "↶" na lista) e não persiste no disco até
        # o usuário salvar o dataset. O message box de confirmação só
        # adicionava um clique extra.
        self.delete_issues_requested.emit(paths)

    def _on_rename_clicked(self) -> None:
        """Emite o sinal — controller faz o input dialog + lógica."""
        paths = self._action_paths()
        if paths:
            self.rename_identity_requested.emit(paths)

    def _on_merge_clicked(self) -> None:
        """Emite o sinal — controller faz o input dialog + lógica."""
        paths = self._action_paths()
        if paths:
            self.merge_identities_requested.emit(paths)

    # ── Undo state ───────────────────────────────────────────────────────

    def set_undo_state(self, can_undo: bool, description: Optional[str] = None) -> None:
        """No-op atual — mantido pela retrocompat com chamadas existentes
        do controller. O estado de undo é refletido pela própria lista
        de histórico (`set_action_history`); o botão global foi removido
        em favor do atalho Ctrl+Z."""
        return

    # ── Histórico de mudanças ────────────────────────────────────────────

    def set_action_history(self, records: Sequence) -> None:
        """Popula a tabela de histórico com a pilha de ações reversíveis.

        Cada record vira uma row com a descrição + botão "↶". Ações mais
        recentes aparecem no topo (records reverso). Clicar no botão
        emite `undo_action_at_requested(op_id)` — controller resolve por id.

        A tabela permanece SEMPRE visível, mesmo vazia — manter a forma
        do layout estável evita o "salto" da UI quando todas as ações
        são desfeitas. O label `(nenhuma mudança aplicada ainda)` fica
        visível quando a tabela está vazia (overlay leve), e some
        quando há rows.
        """
        self.history_table.setRowCount(0)
        if not records:
            # Tabela permanece visível (0 rows = só headers). É o estado
            # "inicial / vazio" — sem label adicional pra não criar
            # noise visual.
            return

        # Mais recente em cima (records é uma pilha — fim é a última ação).
        ordered = list(reversed(records))
        self.history_table.setRowCount(len(ordered))
        for row, rec in enumerate(ordered):
            desc = getattr(rec, "description", "") or "(sem descrição)"
            op_id = getattr(rec, "op_id", "")
            cell_desc = QtWidgets.QTableWidgetItem(desc)
            cell_desc.setToolTip(desc)
            self.history_table.setItem(row, 0, cell_desc)

            btn = QtWidgets.QPushButton("↶")
            btn.setToolTip(f"Desfazer: {desc}")
            btn.setStyleSheet("padding: 2px 6px;")
            btn.clicked.connect(
                lambda _checked=False, op=op_id: self.undo_action_at_requested.emit(op)
            )
            self.history_table.setCellWidget(row, 1, btn)
        self.history_table.resizeRowsToContents()

    # ── Action button states (depend on selection) ───────────────────────

    # ── Renderização ────────────────────────────────────────────────────

    def _render_finding(self, finding: AuditFinding) -> None:
        # Bolinha do título espelha a tree: count == 0 → 🟢 (limpa /
        # informativa); count > 0 → cor da severidade do tipo.
        title_emoji = (
            finding.severity.emoji if finding.count > 0 else "🟢"
        )
        self.lbl_finding_title.setText(
            f"{title_emoji}  {finding.audit_label}"
        )
        self.lbl_finding_summary.setText(finding.summary)

        # Mostra/esconde a caixa de explicação conforme a auditoria
        # selecionada — só aparece pras que têm semântica não-óbvia.
        help_text = _resolve_audit_help(finding.audit_id)
        if help_text:
            self.lbl_audit_help.setText(help_text)
            self.lbl_audit_help.show()
        else:
            self.lbl_audit_help.hide()

        # Auditorias do tipo histograma / correlação / boxplot /
        # métricas de balanceamento têm UI completamente diferente —
        # sem tabela, sem preview, sem botões de ação. Switch o stack
        # pra página correspondente e retorna.
        if finding.audit_id.startswith("histogram_"):
            self._render_histogram_finding(finding)
            return
        if finding.audit_id.startswith("correlation_"):
            self._render_correlation_finding(finding)
            return
        if finding.audit_id.startswith("boxplot_"):
            self._render_boxplot_finding(finding)
            return
        if finding.audit_id == "balance_metrics":
            self._render_balance_metrics_finding(finding)
            return
        if finding.audit_id == "hubness_tables":
            self._render_hubness_tables_finding(finding)
            return
        if finding.audit_id.startswith("umap_embeddings"):
            # Cada execução de UMAP gera audit_id único
            # (`umap_embeddings_<uuid>`); o prefix dispatcheia todas.
            self._render_umap_finding(finding)
            return

        # Para todos os outros tipos: garante stack na página de
        # detalhes (tabela + preview) e mostra a barra de ações.
        self._right_stack.setCurrentIndex(0)
        self._actions_widget.show()
        # Limpa os canvases de plot — evita reter dados em memória ao
        # trocar pra um finding tabular.
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()

        # Layout de colunas dispatcheia por audit_id. Auditorias par-a-par
        # (hard mining + duplicate_identities) e leave-one-out (mixed_identities)
        # ganham cabeçalhos próprios mais legíveis pro perito, com larguras
        # iguais redimensionáveis.
        is_hard_mining = finding.audit_id in ("hard_genuines", "hard_impostors")
        is_duplicate_ids = finding.audit_id == "duplicate_identities"
        is_mixed_ids = finding.audit_id == "mixed_identities"
        is_low_quality = finding.audit_id.startswith("low_quality_")
        is_orphan_paths = finding.audit_id == "orphan_paths"
        is_hash_group = finding.audit_id in ("hash_conflicts", "exact_duplicates")
        is_missing_data = (
            finding.audit_id in ("missing_ofiq", "missing_serfiq")
            or finding.audit_id.startswith("missing_embedding_")
        )
        is_singleton = finding.audit_id == "singleton_identities"
        is_custom_layout = (
            is_hard_mining or is_duplicate_ids or is_mixed_ids or is_low_quality
            or is_orphan_paths or is_hash_group or is_missing_data or is_singleton
        )

        if is_hard_mining:
            headers = [
                "Identidade 1", "Imagem 1",
                "Identidade 2", "Imagem 2",
                "Escore de Similaridade",
            ]
        elif is_duplicate_ids:
            headers = [
                "ID 1", "Nº Imagens",
                "ID 2", "Nº Imagens",
                "Similaridade Centroid",
            ]
        elif is_mixed_ids:
            headers = [
                "Identidade", "Foto", "Nº fotos no centroide", "Similaridade",
            ]
        elif is_low_quality:
            metric_label = self._low_quality_metric_label(finding)
            headers = ["Identidade", "Foto", f"Escore ({metric_label})"]
        elif is_orphan_paths:
            headers = ["Identidade", "Foto", "Path"]
        elif is_hash_group:
            headers = ["ID 1", "Foto 1", "ID 2", "Foto 2", "Hash"]
        elif is_missing_data:
            headers = ["Identidade", "Foto"]
        elif is_singleton:
            headers = ["Identidade", "Nº amostras", "Fotos"]
        else:
            headers = ["Locator", "Descrição", "Detalhes"]
        self.table_issues.setColumnCount(len(headers))
        self.table_issues.setHorizontalHeaderLabels(headers)
        header = self.table_issues.horizontalHeader()
        if is_custom_layout:
            # Larguras divididas igualmente, redimensionáveis via mouse.
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
            header.setStretchLastSection(False)
            # `viewport().width()` é 0 antes do layout pass — agendamos
            # a distribuição de larguras pra rodar depois do tick atual.
            QtCore.QTimer.singleShot(0, self._distribute_table_columns_equally)
        else:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        # Cap em N linhas pra evitar travar a main thread quando uma
        # auditoria retorna milhares de issues (ex: orphan paths num
        # dataset com paths quebrados em massa). Os botões batch
        # ("Excluir tudo") operam sobre o finding completo via cache
        # interno, não sobre a tabela — então o cap é puramente visual.
        total_issues = len(finding.issues)
        visible_count = min(total_issues, _TABLE_VISIBLE_CAP)
        truncated = total_issues > _TABLE_VISIBLE_CAP

        if truncated:
            self.lbl_finding_summary.setText(
                f"{finding.summary}  —  exibindo primeiras {_TABLE_VISIBLE_CAP} "
                f"de {total_issues} (use 'Excluir tudo' p/ processar todas)"
            )

        # blockSignals: evita disparar _on_issue_selection_changed pra cada
        # célula durante o populate (causaria N renders do preview).
        # setUpdatesEnabled(False): suspende repaint da tabela durante
        # populate — diminui drasticamente o custo do batch insert no Qt.
        self.table_issues.blockSignals(True)
        self.table_issues.setUpdatesEnabled(False)
        try:
            self.table_issues.setRowCount(visible_count)
            for row in range(visible_count):
                issue = finding.issues[row]
                if is_hard_mining:
                    self._render_hard_mining_row(row, issue, finding.audit_id)
                elif is_duplicate_ids:
                    self._render_duplicate_identities_row(row, issue)
                elif is_mixed_ids:
                    self._render_mixed_identities_row(row, issue)
                elif is_low_quality:
                    self._render_low_quality_row(row, issue)
                elif is_orphan_paths:
                    self._render_orphan_paths_row(row, issue)
                elif is_hash_group:
                    self._render_hash_group_row(row, issue)
                elif is_missing_data:
                    self._render_missing_data_row(row, issue)
                elif is_singleton:
                    self._render_singleton_row(row, issue)
                else:
                    self._render_default_row(row, issue)
        finally:
            self.table_issues.setUpdatesEnabled(True)
            self.table_issues.blockSignals(False)

        self.btn_delete.setEnabled(total_issues > 0)

        # Auto-seleciona a primeira issue pra que o preview já mostre
        # algo útil sem exigir clique adicional. Limpa antes de
        # selecionar — sem isso, se o finding anterior já tinha row 0
        # selecionada, `selectRow(0)` é no-op e o sinal
        # `itemSelectionChanged` não dispara → preview ficaria preso
        # nas imagens da auditoria anterior.
        self.table_issues.clearSelection()
        if len(finding.issues) > 0:
            self.table_issues.selectRow(0)
            # Garante refresh mesmo se o sinal não chegar (defesa em
            # profundidade — alguns paths de signal podem ser drop'ados
            # quando blockSignals foi acionado pertinho).
            self._on_issue_selection_changed()
        else:
            self.preview.set_paths([])
            self._update_action_buttons(identities=set())

    def _render_default_row(self, row: int, issue) -> None:
        """Layout genérico para auditorias não-hard-mining: Locator | Descrição | Detalhes."""
        primary = issue.locators[0] if issue.locators else ""
        extra = (f"  (+{len(issue.locators) - 1})"
                 if len(issue.locators) > 1 else "")
        cell0 = QtWidgets.QTableWidgetItem(primary + extra)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        cell0.setToolTip("\n".join(issue.locators) if issue.locators else "")
        self.table_issues.setItem(row, 0, cell0)

        cell1 = QtWidgets.QTableWidgetItem(issue.description)
        self.table_issues.setItem(row, 1, cell1)

        details_txt = _format_metadata(issue.metadata)
        cell2 = QtWidgets.QTableWidgetItem(details_txt)
        cell2.setToolTip(details_txt)
        self.table_issues.setItem(row, 2, cell2)

    def _render_hard_mining_row(self, row: int, issue, audit_id: str) -> None:
        """Layout dedicado a hard_genuines / hard_impostors:
        Identidade 1 | Imagem 1 | Identidade 2 | Imagem 2 | Escore.
        As metadados (`file1`, `file2`, `cosine`, `identity_1`, `identity_2`)
        são populadas pelo `_df_to_finding` no controller. Em hard_genuines
        identity_1 == identity_2 (mesma pessoa por definição)."""
        meta = issue.metadata or {}
        file1 = str(meta.get("file1", "")) or "?"
        file2 = str(meta.get("file2", "")) or "?"
        id1 = str(meta.get("identity_1", "?"))
        # hard_genuines: a mesma identidade aparece nas duas colunas
        # — o controller só popula identity_2 quando audit_id é
        # hard_impostors. Pro genuine usamos identity_1 nas duas pra
        # consistência visual da tabela (mesmo n.º de colunas).
        id2 = str(meta.get("identity_2", id1))
        cosine = meta.get("cosine")

        # Coluna 0: Identidade 1
        cell0 = QtWidgets.QTableWidgetItem(id1)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        # Coluna 1: Imagem 1
        cell1 = QtWidgets.QTableWidgetItem(file1)
        cell1.setToolTip(issue.locators[0] if issue.locators else "")
        self.table_issues.setItem(row, 1, cell1)

        # Coluna 2: Identidade 2
        cell2 = QtWidgets.QTableWidgetItem(id2)
        self.table_issues.setItem(row, 2, cell2)

        # Coluna 3: Imagem 2
        cell3 = QtWidgets.QTableWidgetItem(file2)
        cell3.setToolTip(issue.locators[1] if len(issue.locators) > 1 else "")
        self.table_issues.setItem(row, 3, cell3)

        # Coluna 4: Escore de Similaridade
        cosine_txt = f"{float(cosine):.4f}" if cosine is not None else "?"
        cell4 = QtWidgets.QTableWidgetItem(cosine_txt)
        cell4.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 4, cell4)

    def _render_duplicate_identities_row(self, row: int, issue) -> None:
        """Layout dedicado a `duplicate_identities`: ID 1 | Nº Imagens |
        ID 2 | Nº Imagens | Similaridade Centroid.

        A metadata da issue (gerada por `audit_duplicate_identities`)
        contém:
            identity_a, identity_b: nomes dos IDs comparados
            similarity: cosseno entre os centróides (já L2-normalizados)
            n_faces_a, n_faces_b: contagem de fotos de cada ID
        """
        meta = issue.metadata or {}
        id_a = str(meta.get("identity_a", "?"))
        id_b = str(meta.get("identity_b", "?"))
        sim = meta.get("similarity")
        n_a = meta.get("n_faces_a")
        n_b = meta.get("n_faces_b")

        # Coluna 0: ID 1
        cell0 = QtWidgets.QTableWidgetItem(id_a)
        # Coluna 0 carrega TODOS os locators (faces de A + faces de B) e o
        # issue inteiro pra que seleção propague pro preview e botões de ação.
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        # Coluna 1: Nº Imagens (ID 1)
        cell1 = QtWidgets.QTableWidgetItem(str(n_a) if n_a is not None else "?")
        cell1.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 1, cell1)

        # Coluna 2: ID 2
        cell2 = QtWidgets.QTableWidgetItem(id_b)
        self.table_issues.setItem(row, 2, cell2)

        # Coluna 3: Nº Imagens (ID 2)
        cell3 = QtWidgets.QTableWidgetItem(str(n_b) if n_b is not None else "?")
        cell3.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 3, cell3)

        # Coluna 4: Similaridade Centroid
        cos_txt = f"{float(sim):.4f}" if sim is not None else "?"
        cell4 = QtWidgets.QTableWidgetItem(cos_txt)
        cell4.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 4, cell4)

    def _render_mixed_identities_row(self, row: int, issue) -> None:
        """Layout dedicado a `mixed_identities` (leave-one-out): cada
        issue é UMA face outlier dentro de um ID. Colunas:
        Identidade | Foto | Nº fotos no centroide | Similaridade."""
        meta = issue.metadata or {}
        identity = str(meta.get("identity", "")) or "?"
        file_name = str(meta.get("file_name", "")) or "?"
        n_others = meta.get("n_others")
        sim = meta.get("similarity")

        # Coluna 0: Identidade. Carrega locators e issue inteiro pra
        # que a seleção propague pro preview e botões de ação.
        cell0 = QtWidgets.QTableWidgetItem(identity)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        # Coluna 1: Foto (nome do arquivo da face suspeita).
        cell1 = QtWidgets.QTableWidgetItem(file_name)
        cell1.setToolTip(meta.get("outlier_path", ""))
        self.table_issues.setItem(row, 1, cell1)

        # Coluna 2: Nº fotos no centroide
        cell2 = QtWidgets.QTableWidgetItem(
            str(n_others) if n_others is not None else "?"
        )
        cell2.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 2, cell2)

        # Coluna 3: Similaridade (cosseno X vs centroide das outras)
        sim_txt = f"{float(sim):.4f}" if sim is not None else "?"
        cell3 = QtWidgets.QTableWidgetItem(sim_txt)
        cell3.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 3, cell3)

    def _render_histogram_finding(self, finding: AuditFinding) -> None:
        """Mostra a página de histograma no stack à direita e plota a
        distribuição da métrica do `finding`. Esconde a barra de ações
        (não há ação possível sobre histograma) e limpa o preview pra
        liberar memória das thumbs anteriores."""
        self._right_stack.setCurrentIndex(1)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()
        # Reseta estado dos botões — em caso de troca rápida pra outra
        # auditoria, evita ficar com botões "habilitados" residuais.
        self._update_action_buttons(identities=set())

        meta = finding.metadata or {}
        values = list(meta.get("values", []))
        image_paths = meta.get("image_paths")
        captions = meta.get("captions")
        metric_label = str(meta.get("metric_display", "")) or finding.audit_label
        self._histogram_widget.render_distribution(
            values, metric_label,
            image_paths=image_paths, captions=captions,
        )

    def _render_correlation_finding(self, finding: AuditFinding) -> None:
        """Mostra a página de scatter no stack à direita e plota a
        correlação entre duas métricas. Mesmo padrão do histograma:
        esconde ações, limpa preview, limpa os outros plot widgets."""
        self._right_stack.setCurrentIndex(2)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._histogram_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()
        self._update_action_buttons(identities=set())

        meta = finding.metadata or {}
        x_values = list(meta.get("x_values", []))
        y_values = list(meta.get("y_values", []))
        x_label = str(meta.get("x_label", "X"))
        y_label = str(meta.get("y_label", "Y"))
        x_threshold = meta.get("x_threshold")
        y_threshold = meta.get("y_threshold")
        self._scatter_widget.render_correlation(
            x_values, y_values, x_label, y_label,
            x_threshold=x_threshold, y_threshold=y_threshold,
        )

    def _render_boxplot_finding(self, finding: AuditFinding) -> None:
        """Mostra a página de boxplot no stack à direita e plota Y por
        quartis de X. Mesmo padrão dos outros plots."""
        self._right_stack.setCurrentIndex(3)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()
        self._update_action_buttons(identities=set())

        meta = finding.metadata or {}
        groups = list(meta.get("groups", []))
        labels = list(meta.get("labels", []))
        x_label = str(meta.get("x_label", "X"))
        y_label = str(meta.get("y_label", "Y"))
        self._boxplot_widget.render_boxplot(groups, labels, x_label, y_label)

    def _render_balance_metrics_finding(self, finding: AuditFinding) -> None:
        """Mostra a página de balance metrics no stack à direita —
        card Qt-puro com verdict + grid de métricas. Mesmo padrão dos
        outros plot-renders: esconde ações, limpa preview/tabela e
        os outros plot widgets."""
        self._right_stack.setCurrentIndex(4)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()
        self._update_action_buttons(identities=set())

        self._balance_metrics_widget.render_metrics(finding.metadata or {})

    def _render_hubness_tables_finding(self, finding: AuditFinding) -> None:
        """Mostra a página de tabelas de hubness no stack à direita —
        duas tabelas (hubs + anti-hubs) com FIQA/norma/Nk + preview de
        imagem ao selecionar uma linha."""
        self._right_stack.setCurrentIndex(5)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._umap_scatter_widget.clear()
        self._update_action_buttons(identities=set())

        meta = finding.metadata or {}
        self._hubness_tables_widget.render_data(
            list(meta.get("hubs", [])),
            list(meta.get("anti_hubs", [])),
        )

    def _render_umap_finding(self, finding: AuditFinding) -> None:
        """Mostra o painel UMAP no stack à direita — scatter 2D com
        combobox pra trocar atributo de coloração."""
        self._right_stack.setCurrentIndex(6)
        self._actions_widget.hide()
        self.preview.set_paths([])
        self.table_issues.setRowCount(0)
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()
        self._update_action_buttons(identities=set())

        meta = finding.metadata or {}
        result = meta.get("result")
        self._umap_scatter_widget.render_result(result)

    def _render_orphan_paths_row(self, row: int, issue) -> None:
        """Layout dedicado a `orphan_paths`: Identidade | Foto | Path.
        Cada issue tem 1 path (face com arquivo ausente). O caso
        agregado de "100% paths vazios" emite finding sem issues, então
        nada chega aqui."""
        from os.path import basename
        meta = issue.metadata or {}
        identity = str(meta.get("identity", "")) or "?"
        path = issue.locators[0] if issue.locators else ""
        file_name = basename(path) if path else "(sem path)"

        cell0 = QtWidgets.QTableWidgetItem(identity)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        cell1 = QtWidgets.QTableWidgetItem(file_name)
        cell1.setToolTip(path)
        self.table_issues.setItem(row, 1, cell1)

        cell2 = QtWidgets.QTableWidgetItem(path)
        cell2.setToolTip(path)
        self.table_issues.setItem(row, 2, cell2)

    def _render_hash_group_row(self, row: int, issue) -> None:
        """Layout dedicado a `hash_conflicts` e `exact_duplicates`:
        ID 1 | Foto 1 | ID 2 | Foto 2 | Hash. Para grupos com >2 faces,
        mostra os 2 primeiros e indica o resto via "(+N)" no rótulo da
        ID 2; o preview lista todas as faces pra o usuário escolher
        qual deletar (preview-driven)."""
        from os.path import basename
        meta = issue.metadata or {}
        identities = list(meta.get("identities", []) or [])
        file_names = list(meta.get("file_names", []) or [])
        h = str(meta.get("hash", "") or "")
        h_short = (h[:12] + "…") if len(h) > 12 else h

        n = len(issue.locators)
        # Preenche com placeholders se metadata estiver incompleto
        while len(identities) < n:
            identities.append("?")
        while len(file_names) < n:
            file_names.append("")

        # Pega face 1 e face 2 — se >2 faces, indica via sufixo
        id1 = identities[0] if n >= 1 else "?"
        id2 = identities[1] if n >= 2 else "?"
        fn1 = file_names[0] if file_names[0] else (
            basename(issue.locators[0]) if n >= 1 else ""
        )
        fn2 = file_names[1] if n >= 2 and file_names[1] else (
            basename(issue.locators[1]) if n >= 2 else ""
        )
        if n > 2:
            id2 = f"{id2}  (+{n - 2})"

        cell0 = QtWidgets.QTableWidgetItem(id1)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        cell1 = QtWidgets.QTableWidgetItem(fn1)
        cell1.setToolTip(issue.locators[0] if n >= 1 else "")
        self.table_issues.setItem(row, 1, cell1)

        cell2 = QtWidgets.QTableWidgetItem(id2)
        self.table_issues.setItem(row, 2, cell2)

        cell3 = QtWidgets.QTableWidgetItem(fn2)
        cell3.setToolTip(issue.locators[1] if n >= 2 else "")
        self.table_issues.setItem(row, 3, cell3)

        cell4 = QtWidgets.QTableWidgetItem(h_short)
        cell4.setToolTip(h)
        self.table_issues.setItem(row, 4, cell4)

    def _render_missing_data_row(self, row: int, issue) -> None:
        """Layout dedicado a `missing_ofiq`/`missing_serfiq`/
        `missing_embedding_*`: Identidade | Foto. Cada issue tem 1
        face — coluna que está faltando vira parte do header da
        finding (não da row)."""
        from os.path import basename
        meta = issue.metadata or {}
        identity = str(meta.get("identity", "")) or "?"
        path = issue.locators[0] if issue.locators else ""
        file_name = basename(path) if path else "(sem path)"

        cell0 = QtWidgets.QTableWidgetItem(identity)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        cell1 = QtWidgets.QTableWidgetItem(file_name)
        cell1.setToolTip(path)
        self.table_issues.setItem(row, 1, cell1)

    def _render_singleton_row(self, row: int, issue) -> None:
        """Layout dedicado a `singleton_identities`: Identidade |
        Nº amostras | Fotos. A row representa a IDENTIDADE inteira;
        as N fotos da identidade estão em locators e aparecem no
        preview pra o usuário escolher quais deletar."""
        from os.path import basename
        meta = issue.metadata or {}
        identity = str(meta.get("identity", "")) or "?"
        n_samples = meta.get("n_samples", len(issue.locators))
        # Lista de filenames (truncada se muito comprida)
        files = [basename(p) for p in issue.locators if p]
        if len(files) <= 3:
            files_text = ", ".join(files)
        else:
            files_text = ", ".join(files[:3]) + f"  (+{len(files) - 3})"

        cell0 = QtWidgets.QTableWidgetItem(identity)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        cell1 = QtWidgets.QTableWidgetItem(str(n_samples))
        cell1.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 1, cell1)

        cell2 = QtWidgets.QTableWidgetItem(files_text)
        cell2.setToolTip("\n".join(files))
        self.table_issues.setItem(row, 2, cell2)

    def _render_low_quality_row(self, row: int, issue) -> None:
        """Layout dedicado a auditorias `low_quality_*` (FIQA, OFIQ, SERFIQ,
        EMBEDDING_NORM): cada issue é UMA face com escore abaixo do
        threshold. Colunas: Identidade | Foto | Escore."""
        meta = issue.metadata or {}
        identity = str(meta.get("identity", "")) or "?"
        primary = issue.locators[0] if issue.locators else ""
        from os.path import basename
        file_name = basename(primary) if primary else "?"
        value = meta.get("value")

        # Coluna 0: Identidade — carrega locators+issue pra que o
        # clique na linha dispare seleção/preview e habilite botões.
        cell0 = QtWidgets.QTableWidgetItem(identity)
        cell0.setData(_PATH_ROLE, list(issue.locators))
        cell0.setData(_ISSUE_ROLE, issue)
        self.table_issues.setItem(row, 0, cell0)

        # Coluna 1: Foto (nome do arquivo, com tooltip do path completo)
        cell1 = QtWidgets.QTableWidgetItem(file_name)
        cell1.setToolTip(primary)
        self.table_issues.setItem(row, 1, cell1)

        # Coluna 2: Escore numérico
        score_txt = f"{float(value):.3f}" if value is not None else "?"
        cell2 = QtWidgets.QTableWidgetItem(score_txt)
        cell2.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        self.table_issues.setItem(row, 2, cell2)

    @staticmethod
    def _low_quality_metric_label(finding) -> str:
        """Retorna um rótulo amigável da métrica pra usar no header da
        tabela. Lê `metric_display` da primeira issue (todas têm a mesma);
        fallback ao audit_id se não houver metadata."""
        for issue in finding.issues:
            label = (issue.metadata or {}).get("metric_display")
            if label:
                return str(label)
        # Fallback: deriva do audit_id (`low_quality_fiqa` → `FIQA`)
        suffix = finding.audit_id.replace("low_quality_", "", 1)
        return suffix.upper() if suffix else "Qualidade"

    def _distribute_table_columns_equally(self) -> None:
        """Divide o viewport da tabela igualmente entre todas as colunas.

        Chamado via QTimer.singleShot pra rodar após o layout pass — o
        viewport().width() é -1 no momento do build mas já válido após
        a primeira pass. Usuário ainda pode arrastar bordas para
        redimensionar (modo Interactive)."""
        n_cols = self.table_issues.columnCount()
        if n_cols <= 0:
            return
        viewport_w = self.table_issues.viewport().width()
        if viewport_w <= 0:
            return
        # Vertical scrollbar pode ocupar alguns px do viewport — descontamos
        # pra não criar barra horizontal.
        col_w = max(60, viewport_w // n_cols)
        for c in range(n_cols):
            self.table_issues.setColumnWidth(c, col_w)

    def _render_empty_finding(self) -> None:
        self.lbl_finding_title.setText("Nenhuma auditoria selecionada.")
        self.lbl_finding_summary.setText("")
        self.table_issues.setRowCount(0)
        self.btn_delete.setEnabled(False)
        self.preview.set_paths([])
        # Volta ao painel default + restaura barra de ações + limpa
        # qualquer plot residual (histograma, scatter, boxplot, card
        # de métricas ou tabelas de hubness).
        self._right_stack.setCurrentIndex(0)
        self._actions_widget.show()
        self._histogram_widget.clear()
        self._scatter_widget.clear()
        self._boxplot_widget.clear()
        self._balance_metrics_widget.clear()
        self._hubness_tables_widget.clear()
        self._umap_scatter_widget.clear()

    def _on_issue_selection_changed(self) -> None:
        """Atualiza o preview + estados dos botões de ação."""
        rows = sorted({i.row() for i in self.table_issues.selectedIndexes()})
        if not rows:
            self.preview.set_paths([])
            self._update_action_buttons(identities=set())
            return

        # Caso especial — exatamente 1 row selecionada com layout par-a-par:
        # duplicate_identities mostra "ID A em cima, ID B embaixo";
        # mixed_identities mostra "foto suspeita em cima, demais embaixo".
        if len(rows) == 1:
            cell0 = self.table_issues.item(rows[0], 0)
            issue = cell0.data(_ISSUE_ROLE) if cell0 else None
            if issue is not None and self._is_duplicate_identities_issue(issue):
                self._render_duplicate_preview(issue)
                meta = issue.metadata or {}
                ids = {
                    str(meta.get("identity_a", "")),
                    str(meta.get("identity_b", "")),
                }
                ids.discard("")
                self._update_action_buttons(ids)
                return
            if issue is not None and self._is_mixed_identities_issue(issue):
                self._render_mixed_preview(issue)
                meta = issue.metadata or {}
                ident = str(meta.get("identity", ""))
                self._update_action_buttons({ident} if ident else set())
                return

        seen: set = set()
        paths: List[str] = []
        captions: List[str] = []
        identities_per_path: Dict[str, str] = {}

        for row in rows:
            cell0 = self.table_issues.item(row, 0)
            if cell0 is None:
                continue
            issue = cell0.data(_ISSUE_ROLE)
            row_paths = cell0.data(_PATH_ROLE) or []
            row_captions = _captions_for_issue(issue, row_paths)

            # Constrói identities_per_path pra alimentar enabling dos botões.
            # Heurística: usa identities posição-alinhadas se presentes;
            # senão, identity única; senão, ignora.
            row_identities = _identities_for_issue(issue, row_paths)

            for p, cap, ident in zip(row_paths, row_captions, row_identities):
                if p and p not in seen:
                    seen.add(p)
                    paths.append(p)
                    captions.append(cap)
                    if ident is not None:
                        identities_per_path[p] = ident

        self.preview.set_paths(paths, captions=captions)
        self._update_action_buttons(set(identities_per_path.values()))

    @staticmethod
    def _is_duplicate_identities_issue(issue) -> bool:
        meta = getattr(issue, "metadata", None) or {}
        # Sinais característicos da issue gerada por audit_duplicate_identities.
        return "identity_a" in meta and "identity_b" in meta and "identities" in meta

    @staticmethod
    def _is_mixed_identities_issue(issue) -> bool:
        meta = getattr(issue, "metadata", None) or {}
        # Sinais característicos da issue gerada por audit_mixed_identities
        # após a refatoração leave-one-out.
        return (
            "outlier_path" in meta
            and "n_others" in meta
            and "other_paths" in meta
        )

    def _render_mixed_preview(self, issue) -> None:
        """Mostra a foto suspeita (linha 1) + todas as outras fotos do
        mesmo ID que formaram o centroide (linha 2). Layout deixa visual-
        mente claro o que foi comparado contra o quê — TODAS as fotos
        pertencem à mesma identidade (mixed = anotação contaminada
        suspeita, não comparação entre 2 IDs)."""
        meta = issue.metadata or {}
        outlier_path = str(meta.get("outlier_path", ""))
        other_paths = list(meta.get("other_paths") or [])
        identity = str(meta.get("identity", "?"))
        sim = meta.get("similarity")

        sim_txt = f"cos={float(sim):.3f}" if sim is not None else ""
        groups = [
            (
                f"Foto comparada — {identity}  ({sim_txt})",
                [outlier_path],
                [""],
            ),
            (
                f"{len(other_paths)} foto(s) que formaram o centroide ({identity})",
                other_paths,
                [""] * len(other_paths),
            ),
        ]
        # Header explícito: 1 identidade, total = 1 (suspeita) + N (outras).
        # Sem override, o set_paired_groups contaria 2 grupos como 2
        # identidades — confunde o usuário porque na verdade é só 1 ID.
        total_faces = 1 + len(other_paths)
        self.preview.set_paired_groups(
            groups,
            header_override=(
                f"{total_faces} face(s) — identidade '{identity}' "
                f"(1 suspeita + {len(other_paths)} centroide)"
            ),
        )

    def _render_duplicate_preview(self, issue) -> None:
        """Quebra os locators do issue em 2 grupos (id_a / id_b) usando
        a lista `identities` posição-alinhada da metadata. Cada grupo
        vira uma linha no preview via `set_paired_groups`."""
        meta = issue.metadata or {}
        id_a = str(meta.get("identity_a", "?"))
        id_b = str(meta.get("identity_b", "?"))
        locs = list(issue.locators or [])
        idents = list(meta.get("identities") or [])

        paths_a: List[str] = []
        paths_b: List[str] = []
        for path, ident in zip(locs, idents):
            if str(ident) == id_a:
                paths_a.append(path)
            elif str(ident) == id_b:
                paths_b.append(path)

        groups = [
            (id_a, paths_a, [""] * len(paths_a)),
            (id_b, paths_b, [""] * len(paths_b)),
        ]
        self.preview.set_paired_groups(groups)

    def _on_preview_selection_changed(self) -> None:
        """Atualiza Renomear/Mesclar com base nas fotos marcadas no
        preview. Mapeia path → identidade via metadata da issue exibida
        (cell selecionada na tabela). Para hard mining, identity_1/_2
        ficam alinhados com locators[0]/[1]; para mixed/duplicate,
        meta.identities[k] alinha com locators[k]."""
        sel_paths = set(self.preview.selected_paths())
        if not sel_paths:
            self._update_action_buttons(identities=set())
            return

        # Acha a issue corrente (do row selecionado na tabela).
        rows = sorted({i.row() for i in self.table_issues.selectedIndexes()})
        if not rows:
            self._update_action_buttons(identities=set())
            return
        cell0 = self.table_issues.item(rows[0], 0)
        issue = cell0.data(_ISSUE_ROLE) if cell0 else None
        if issue is None:
            self._update_action_buttons(identities=set())
            return

        all_paths = list(getattr(issue, "locators", []) or [])
        idents_per_path = _identities_for_issue(issue, all_paths)
        path_to_ident = {
            p: i for p, i in zip(all_paths, idents_per_path) if i is not None
        }
        sel_idents = {
            path_to_ident[p] for p in sel_paths if p in path_to_ident
        }
        self._update_action_buttons(sel_idents)

    def _update_action_buttons(self, identities: set) -> None:
        """Habilita/desabilita Renomear/Mesclar conforme a seleção atual.

        - Renomear: 1 única identidade conhecida
        - Mesclar: 2+ identidades distintas
        - Excluir: qualquer seleção (já tratado no _render_finding)
        """
        n_ids = len(identities)
        self.btn_rename.setEnabled(n_ids == 1)
        self.btn_merge.setEnabled(n_ids >= 2)

    def clear_preview_cache(self) -> None:
        """Limpa o cache de pixmaps do preview. Chamar após bulk-delete
        pra liberar memória de paths que sumiram."""
        self.preview.clear_cache()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _selected_locators(self) -> List[str]:
        """Retorna a lista deduplicada de image_paths das linhas selecionadas."""
        seen = set()
        out: List[str] = []
        for idx in self.table_issues.selectedIndexes():
            if idx.column() != 0:
                continue
            paths = self.table_issues.item(idx.row(), 0).data(_PATH_ROLE) or []
            for p in paths:
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    def _currently_selected_audit_id(self) -> Optional[str]:
        items = self.tree.selectedItems()
        if not items:
            return None
        finding = items[0].data(0, _FINDING_ROLE)
        return getattr(finding, "audit_id", None) if finding else None

    def _first_finding_with_issues(self) -> Optional[QtWidgets.QTreeWidgetItem]:
        """Itera a árvore retornando o primeiro nó-folha com count > 0."""
        for i in range(self.tree.topLevelItemCount()):
            cat = self.tree.topLevelItem(i)
            for j in range(cat.childCount()):
                child = cat.child(j)
                finding = child.data(0, _FINDING_ROLE)
                if finding and finding.count > 0:
                    return child
        # Fallback: primeira folha mesmo que vazia
        if self.tree.topLevelItemCount() > 0:
            cat = self.tree.topLevelItem(0)
            if cat.childCount() > 0:
                return cat.child(0)
        return None


def _severity_color(severity: AuditSeverity) -> str:
    return {
        AuditSeverity.CRITICAL: "#c62828",
        AuditSeverity.WARNING: "#ef6c00",
        AuditSeverity.INFO: "#1565c0",
    }.get(severity, "#333")


def _captions_for_issue(issue, paths: Sequence[str]) -> List[str]:
    """Produz uma caption por path baseado no metadata do issue.

    Heurísticas (em ordem):
      1. `locator_labels` (lista alinhada aos paths) → o caller já
         decidiu — string explícita por locator (ex: "outlier" / "típica")
      2. `identity` (string única) → aplica a todos os paths
      3. `identities` (lista alinhada aos paths) → position-aligned
      4. `value` numérico (audits de qualidade) → "valor: X.XX"
      5. fallback → string vazia (filename já é exibido pelo widget)
    """
    if issue is None:
        return ["" for _ in paths]
    meta = getattr(issue, "metadata", {}) or {}

    # 1. locator_labels explícito do caller
    labels = meta.get("locator_labels")
    if isinstance(labels, (list, tuple)) and len(labels) == len(paths):
        return [str(x) for x in labels]

    # 2. identity única
    identity = meta.get("identity")
    if identity:
        return [str(identity) for _ in paths]

    # 3. identities posição-alinhadas (mesmo len que paths)
    identities = meta.get("identities")
    if isinstance(identities, (list, tuple)) and len(identities) == len(paths):
        return [str(x) for x in identities]

    # 4. valor numérico (low_quality, etc.)
    value = meta.get("value")
    if isinstance(value, (int, float)):
        return [f"valor: {value:.2f}" for _ in paths]

    # 5. fallback
    return ["" for _ in paths]


def _identities_for_issue(issue, paths: Sequence[str]) -> List[Optional[str]]:
    """Retorna a identidade aproximada por path da issue (alinhada com `paths`).

    Usado pra decidir habilitação dos botões Renomear/Mesclar — saber
    quantas identidades distintas estão na seleção. Ordem da heurística:
      1. `identities` (lista alinhada, audit_duplicate_identities)
      2. `identity_1`/`identity_2` (hard_genuines/hard_impostors —
         posicionalmente alinhados com locators[0]/locators[1])
      3. `identity` (única, audit_singleton/orphan/missing/mixed)
      4. None (não dá pra determinar)
    """
    if issue is None:
        return [None for _ in paths]
    meta = getattr(issue, "metadata", {}) or {}

    identities = meta.get("identities")
    if isinstance(identities, (list, tuple)) and len(identities) == len(paths):
        return [str(x) if x is not None else None for x in identities]

    # Hard mining: identity_1 alinhado a locators[0], identity_2 a locators[1].
    # Em hard_genuines o controller só seta identity_1 (são iguais por def);
    # nesse caso aplicamos identity_1 a ambos os paths.
    id1 = meta.get("identity_1")
    if id1 is not None:
        id2 = meta.get("identity_2", id1)
        # Trunca / preenche pra alinhar com `paths`. Em prática hard
        # mining sempre tem 2 paths.
        result: List[Optional[str]] = []
        for idx in range(len(paths)):
            if idx == 0:
                result.append(str(id1))
            elif idx == 1:
                result.append(str(id2) if id2 is not None else None)
            else:
                result.append(None)
        return result

    identity = meta.get("identity")
    if identity:
        return [str(identity) for _ in paths]

    return [None for _ in paths]


def _format_metadata(meta: dict) -> str:
    if not meta:
        return ""
    items = []
    for k, v in meta.items():
        if isinstance(v, (list, tuple)):
            v_str = ", ".join(str(x) for x in v[:5])
            if len(v) > 5:
                v_str += f"… (+{len(v) - 5})"
        else:
            v_str = str(v)
        items.append(f"{k}={v_str}")
    return " | ".join(items)
