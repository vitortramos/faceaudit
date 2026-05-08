"""Diálogo modal pré-execução do UMAP.

Em vez de um botão dedicado de "⚙ Configurações UMAP" separado, o
botão principal "🌌 UMAP (Embeddings)" abre este diálogo onde o
usuário ajusta os hyperparams e clica Executar. Cancelar fecha sem
rodar. A config escolhida fica em cache no controller e é restaurada
na próxima abertura.

Hyperparams do UMAP têm semântica não-trivial — tooltips contextuais
explicam cada um.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.models.umap_config import UmapConfig


class UmapConfigDialog(QtWidgets.QDialog):
    """Diálogo de hyperparams do UMAP. Retorna `UmapConfig` via
    `get_config()` quando aceito; raise via cancel/close."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        config: Optional[UmapConfig] = None,
        n_dataset_faces: int = 0,
    ):
        super().__init__(parent)
        self.setWindowTitle("Configurações — UMAP (Embeddings)")
        self.setMinimumWidth(480)

        cfg = config or UmapConfig()
        self._n_dataset_faces = int(n_dataset_faces)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "<b>UMAP</b> — projeta os embeddings do dataset em 2D/3D "
            "preservando estrutura local e parte da global. Hyperparams "
            "abaixo controlam o balanço entre clusters compactos vs "
            "espalhados e o que o algoritmo considera \"vizinhança\"."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; font-size: 11px; padding-bottom: 4px;")
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        # ── mode (radio): por face vs por identidade (centroides) ──
        mode_widget = QtWidgets.QWidget()
        mode_layout = QtWidgets.QVBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(2)
        self.radio_faces = QtWidgets.QRadioButton(
            "Por face (cada ponto = uma foto)"
        )
        self.radio_centroids = QtWidgets.QRadioButton(
            "Por identidade (centroides) — habilita click pra ver lookalikes"
        )
        is_centroids = (cfg.mode == UmapConfig.MODE_CENTROIDS)
        self.radio_faces.setChecked(not is_centroids)
        self.radio_centroids.setChecked(is_centroids)
        self.radio_faces.setToolTip(
            "Modo padrão: cada foto vira um ponto no scatter. Útil pra "
            "ver clusters por identidade, demografia, FIQA, etc. com a "
            "granularidade máxima."
        )
        self.radio_centroids.setToolTip(
            "Cada identidade é agregada num centroide L2-normalizado "
            "(média dos embeddings). Cada ponto representa UMA pessoa. "
            "Mais limpo visualmente e MUITO mais rápido. Habilita "
            "click numa identidade pra ver as N mais parecidas no "
            "espaço 512D (cosine)."
        )
        mode_layout.addWidget(self.radio_faces)
        mode_layout.addWidget(self.radio_centroids)
        form.addRow("Granularidade:", mode_widget)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        form.addRow(sep)

        # ── n_components ──
        self.spin_n_components = QtWidgets.QSpinBox()
        self.spin_n_components.setRange(2, 3)
        self.spin_n_components.setValue(int(cfg.n_components))
        self.spin_n_components.setToolTip(
            "Dimensão da projeção. <b>2</b> (default) é o mais comum — "
            "scatter plot 2D legível. <b>3</b> permite rotação 3D mas "
            "fica mais difícil de interpretar visualmente."
        )
        form.addRow("Dimensões (n_components):", self.spin_n_components)

        # ── n_neighbors ──
        self.spin_n_neighbors = QtWidgets.QSpinBox()
        self.spin_n_neighbors.setRange(2, 200)
        self.spin_n_neighbors.setValue(int(cfg.n_neighbors))
        self.spin_n_neighbors.setToolTip(
            "Tamanho da vizinhança local considerada pelo UMAP.\n"
            "<b>Baixo (5–10)</b>: preserva estrutura LOCAL, fragmenta "
            "em micro-clusters; bom pra ver outliers individuais.\n"
            "<b>Médio (15–30, default 15)</b>: balanço — clusters por "
            "identidade ficam visíveis, estrutura global razoável.\n"
            "<b>Alto (50–100)</b>: prioriza estrutura GLOBAL; clusters "
            "se fundem em manifolds maiores. Bom pra ver gênero/etnia."
        )
        form.addRow("Vizinhança (n_neighbors):", self.spin_n_neighbors)

        # ── min_dist ──
        self.spin_min_dist = QtWidgets.QDoubleSpinBox()
        self.spin_min_dist.setRange(0.0, 0.99)
        self.spin_min_dist.setSingleStep(0.05)
        self.spin_min_dist.setDecimals(2)
        self.spin_min_dist.setValue(float(cfg.min_dist))
        self.spin_min_dist.setToolTip(
            "Distância mínima entre pontos no espaço 2D — controla "
            "quão agrupados ficam visualmente.\n"
            "<b>0.0–0.1</b>: clusters compactos (clumps densos).\n"
            "<b>0.1 (default)</b>: balanço entre densidade e legibilidade.\n"
            "<b>0.5+</b>: pontos mais espalhados, clusters difusos. "
            "Útil quando há muita sobreposição e precisa enxergar "
            "amostras individuais."
        )
        form.addRow("Compactação (min_dist):", self.spin_min_dist)

        # ── metric ──
        self.combo_metric = QtWidgets.QComboBox()
        self.combo_metric.addItems([
            "cosine", "euclidean", "manhattan", "correlation",
        ])
        idx = self.combo_metric.findText(str(cfg.metric))
        self.combo_metric.setCurrentIndex(idx if idx >= 0 else 0)
        self.combo_metric.setToolTip(
            "Métrica de distância no espaço de embeddings.\n"
            "<b>cosine</b> (default): consistente com a pipeline forense "
            "(cosseno é a métrica do scoring de FR). Recomendado.\n"
            "<b>euclidean</b>: usa a magnitude do embedding também — "
            "AdaFace tem norma significativa, então faz algum sentido.\n"
            "<b>manhattan</b>: L1; menos sensível a outliers.\n"
            "<b>correlation</b>: cosseno após centralizar — útil em "
            "embeddings com viés sistemático."
        )
        form.addRow("Métrica:", self.combo_metric)

        # ── max_samples ──
        self.spin_max_samples = QtWidgets.QSpinBox()
        self.spin_max_samples.setRange(1_000, 1_000_000)
        self.spin_max_samples.setSingleStep(10_000)
        self.spin_max_samples.setValue(int(cfg.max_samples))
        self.spin_max_samples.setToolTip(
            "Limite de amostras enviadas ao UMAP. Datasets maiores são "
            "subsampled reproduzivelmente (seed=42) antes do fit. UMAP "
            "aguenta confortável ~100k em CPU; acima disso o tempo "
            "cresce mais rápido que o ganho de informação visual."
        )
        form.addRow("Máx. amostras (max_samples):", self.spin_max_samples)

        layout.addLayout(form)

        # ── Estimativa ──
        # Atualizada dinamicamente conforme o usuário muda max_samples
        self._estimate_label = QtWidgets.QLabel("")
        self._estimate_label.setWordWrap(True)
        self._estimate_label.setStyleSheet(
            "color: #1a3d6e; font-size: 11px; "
            "background: #e3f2fd; "
            "border-left: 3px solid #1976d2; "
            "padding: 6px 8px; margin-top: 6px;"
        )
        self._update_estimate()
        layout.addWidget(self._estimate_label)

        # Recalcula estimativa quando algum dos hyperparams que afeta
        # o tempo muda
        self.spin_max_samples.valueChanged.connect(self._update_estimate)
        self.spin_n_neighbors.valueChanged.connect(self._update_estimate)
        self.spin_n_components.valueChanged.connect(self._update_estimate)

        # Botões
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText("Executar")
        cancel_btn = btns.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        cancel_btn.setText("Cancelar")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_config(self) -> UmapConfig:
        """Retorna `UmapConfig` montado a partir dos widgets."""
        mode = (
            UmapConfig.MODE_CENTROIDS
            if self.radio_centroids.isChecked()
            else UmapConfig.MODE_FACES
        )
        return UmapConfig(
            n_components=int(self.spin_n_components.value()),
            n_neighbors=int(self.spin_n_neighbors.value()),
            min_dist=float(self.spin_min_dist.value()),
            metric=str(self.combo_metric.currentText()),
            random_state=42,        # fixado pra determinismo
            max_samples=int(self.spin_max_samples.value()),
            subsample_seed=42,
            mode=mode,
            color_attributes=None,  # auto-detect
        )

    # ── Estimativa de custo ──────────────────────────────────────────

    def _update_estimate(self) -> None:
        """Mostra estimativa de tempo + N efetivo após subsample."""
        max_samp = self.spin_max_samples.value()
        n_neigh = self.spin_n_neighbors.value()
        n_comp = self.spin_n_components.value()
        if self._n_dataset_faces > 0:
            n_eff = min(self._n_dataset_faces, max_samp)
        else:
            n_eff = max_samp

        # Estimativa empírica calibrada com BFW (20k → ~50-60s):
        #   secs = base_jit + 1.5s × (N/1000) × (fator de n_neighbors)
        # Onde base_jit cobre o setup + JIT compile do numba na 1ª
        # chamada (~25-30s). n_neighbors aumenta o custo logaritmicamente
        # (fator 1.0 pra 15, ~1.3 pra 30, ~1.6 pra 50). 3D adiciona
        # ~30% sobre 2D.
        neigh_factor = 1.0 + (max(0, n_neigh - 15) / 30.0) * 0.3
        dim_factor = 1.3 if n_comp == 3 else 1.0
        base_jit = 25
        secs = int(
            base_jit + 1.5 * (n_eff / 1000.0) * neigh_factor * dim_factor
        )
        secs = max(secs, 20)

        if secs < 60:
            t_str = f"~{secs}s"
        elif secs < 3600:
            t_str = f"~{secs // 60}min {secs % 60}s"
        else:
            t_str = f"~{secs // 3600}h"

        if self._n_dataset_faces > 0 and self._n_dataset_faces > max_samp:
            sub = (
                f"Dataset tem {self._n_dataset_faces:,} faces — "
                f"subsample reproduzível pra <b>{max_samp:,}</b>."
            )
        else:
            sub = (
                f"Dataset tem {self._n_dataset_faces:,} faces — "
                f"todas serão usadas."
                if self._n_dataset_faces > 0 else
                f"Dataset não medido — assumindo {n_eff:,} amostras."
            )
        self._estimate_label.setText(
            f"<b>Estimativa</b>: {t_str} de processamento. {sub} "
            f"<i>(Base ~25s do JIT compile do numba + fit do UMAP é "
            f"linear-ish em N e cresce com n_neighbors. 3D ~30% mais "
            f"lento que 2D.)</i>"
        )
