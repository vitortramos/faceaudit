"""Diálogo de Localização de Dataset (PR L1d).

Coleta as opções pro `DatasetLocalizationService` antes de abrir o
worker thread. Fica em arquivo único (controller + view juntos)
seguindo padrão de `schema_report_dialog.py` e `audit_log_dialog.py`.

UX:
  - Combo Modo (REWRITE_RELATIVE / COPY_ORIGINALS)
  - Combo Naming (PRESERVE / HASH / BY_IDENTITY) — só ativo em COPY
  - Seletor de pasta destino — default <parquet_dir>/<name>__localized/
  - Checkbox "Pular existentes (idempotente)" — default ON
  - Help text contextual que muda com o modo
  - OK/Cancel
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy


_MODE_OPTIONS = [
    (LocalizationMode.COPY_ORIGINALS,
     "Copiar imagens originais",
     "Copia as imagens pra <pasta destino>/images/. As imagens "
     "originais ficam intocadas. Resultado totalmente portátil — pode "
     "ser movido/zipado e aberto em outra máquina sem quebrar."),
    (LocalizationMode.REWRITE_RELATIVE,
     "Reescrever caminhos (sem cópia)",
     "Apenas reescreve os caminhos no parquet como relativos à pasta "
     "destino. Nenhuma imagem é copiada. Requer que TODAS as imagens "
     "já estejam dentro da pasta destino — caso contrário, falha."),
    (LocalizationMode.REBASE_PREFIX,
     "Atualizar prefixo de caminho (sem cópia)",
     "Substitui um prefixo de path por outro em todas as faces. "
     "Caso de uso típico: dataset criado em outra máquina com paths "
     "absolutos do tipo <code>/home/colega/imgs/...</code> ou "
     "<code>C:\\Users\\alguem\\imgs\\...</code>, e nesta máquina a "
     "mesma estrutura existe em outro caminho (ex: "
     "<code>/Users/eu/data/imgs/...</code>). Não copia nada — só "
     "reescreve os paths. Sanity check: cada arquivo precisa existir "
     "no novo caminho."),
    (LocalizationMode.COPY_ALIGNED,
     "Copiar faces alinhadas (PNG)",
     "Para cada face: detecta + alinha (via ForensicFace) e salva como "
     "PNG (lossless) em <pasta destino>/images/. Resultado: dataset "
     "enxuto com APENAS as faces (já no input do modelo, 112×112). "
     "Operação custosa (~80-150 ms por face) — datasets grandes podem "
     "levar dezenas de minutos. Idempotente: re-execução pula PNGs já "
     "criados. Requer ForensicFace instalado."),
]

_NAMING_OPTIONS = [
    (NamingStrategy.PRESERVE,
     "Preservar nomes (default)",
     "Mantém o nome original. Em colisão entre 'a/img.jpg' e "
     "'b/img.jpg', sufixa: img.jpg + img_1.jpg."),
    (NamingStrategy.HASH,
     "Hash (sha256[:16].ext)",
     "Renomeia pelo hash do conteúdo. Único por construção, ideal "
     "pra datasets grandes onde colisões PRESERVE seriam frequentes."),
    (NamingStrategy.BY_IDENTITY,
     "Por identidade (subpastas)",
     "Cria subpasta <identity>/<file_name>. Agrupa visualmente."),
]


class LocalizeDatasetDialog(QtWidgets.QDialog):
    """Coleta opções de localização. Modal."""

    def __init__(
        self,
        dataset_name: str,
        suggested_target_dir: Path,
        parent: Optional[QtWidgets.QWidget] = None,
        detected_prefix: Optional[str] = None,
    ):
        super().__init__(parent)
        self._dataset_name = dataset_name
        self._suggested_target = suggested_target_dir
        self._detected_prefix = detected_prefix or ""

        self.setWindowTitle(f"Localizar Dataset — {dataset_name}")
        self.setModal(True)
        self.resize(640, 580)
        self._build_ui()
        self._on_mode_changed(0)  # estado inicial coerente

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Localização</b> — converte um dataset com caminhos "
            "absolutos em paths relativos, opcionalmente copiando as "
            "imagens pra uma pasta autocontida. Útil pra portabilidade "
            "(zip/USB/laudo)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Modo ──
        mode_box = QtWidgets.QGroupBox("Modo")
        mode_layout = QtWidgets.QFormLayout(mode_box)
        self.cmb_mode = QtWidgets.QComboBox()
        for mode, label, _ in _MODE_OPTIONS:
            self.cmb_mode.addItem(label, userData=mode)
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addRow("Operação:", self.cmb_mode)

        self.lbl_mode_help = QtWidgets.QLabel()
        self.lbl_mode_help.setWordWrap(True)
        self.lbl_mode_help.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px;"
        )
        mode_layout.addRow("", self.lbl_mode_help)
        layout.addWidget(mode_box)

        # ── Naming (só em COPY) ──
        self.naming_box = QtWidgets.QGroupBox("Estratégia de nome")
        naming_layout = QtWidgets.QFormLayout(self.naming_box)
        self.cmb_naming = QtWidgets.QComboBox()
        for naming, label, _ in _NAMING_OPTIONS:
            self.cmb_naming.addItem(label, userData=naming)
        self.cmb_naming.currentIndexChanged.connect(self._on_naming_changed)
        naming_layout.addRow("Naming:", self.cmb_naming)

        self.lbl_naming_help = QtWidgets.QLabel()
        self.lbl_naming_help.setWordWrap(True)
        self.lbl_naming_help.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px;"
        )
        naming_layout.addRow("", self.lbl_naming_help)
        layout.addWidget(self.naming_box)

        # ── Prefixo antigo (só em REBASE_PREFIX) ──
        self.rebase_box = QtWidgets.QGroupBox("Prefixo antigo (a ser substituído)")
        rebase_layout = QtWidgets.QVBoxLayout(self.rebase_box)
        rebase_inner = QtWidgets.QHBoxLayout()
        self.input_old_prefix = QtWidgets.QLineEdit(self._detected_prefix)
        self.input_old_prefix.setPlaceholderText(
            "ex: /home/colega/datasets/celeb"
        )
        rebase_inner.addWidget(self.input_old_prefix)
        rebase_layout.addLayout(rebase_inner)
        self.lbl_rebase_help = QtWidgets.QLabel()
        self.lbl_rebase_help.setWordWrap(True)
        self.lbl_rebase_help.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px;"
        )
        if self._detected_prefix:
            self.lbl_rebase_help.setText(
                f"<i>Auto-detectado a partir dos paths atuais: "
                f"<code>{self._detected_prefix}</code></i>. "
                "Edite se necessário."
            )
        else:
            self.lbl_rebase_help.setText(
                "<i>Não foi possível auto-detectar — preencha "
                "manualmente o prefixo comum dos paths originais.</i>"
            )
        rebase_layout.addWidget(self.lbl_rebase_help)
        layout.addWidget(self.rebase_box)

        # ── Destino ──
        # O label muda conforme o modo: em REBASE_PREFIX é "Prefixo
        # novo (pasta nesta máquina)", nos demais é "Pasta destino".
        self.target_box = QtWidgets.QGroupBox("Pasta destino")
        target_layout = QtWidgets.QHBoxLayout(self.target_box)
        self.input_target = QtWidgets.QLineEdit(str(self._suggested_target))
        target_layout.addWidget(self.input_target)
        self.btn_browse = QtWidgets.QPushButton("📁 Escolher…")
        self.btn_browse.clicked.connect(self._on_browse)
        target_layout.addWidget(self.btn_browse)
        layout.addWidget(self.target_box)

        # ── Idempotência ──
        self.chk_skip_existing = QtWidgets.QCheckBox(
            "Pular arquivos que já existem com o mesmo conteúdo "
            "(idempotência — permite re-rodar sem dor de cabeça)"
        )
        self.chk_skip_existing.setChecked(True)
        layout.addWidget(self.chk_skip_existing)

        layout.addStretch()

        # ── Botões ──
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)\
            .setText("🗂 Localizar")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # estado inicial dos help texts
        self._on_naming_changed(0)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_mode_changed(self, idx: int):
        mode = self.cmb_mode.currentData()
        # Atualiza help text
        for m, _, h in _MODE_OPTIONS:
            if m == mode:
                self.lbl_mode_help.setText(h)
                break
        # Naming faz sentido nos dois modos de cópia (originais e alinhadas).
        # Em REBASE_PREFIX e REWRITE_RELATIVE não há cópia, sem renomear.
        is_copy = mode in (
            LocalizationMode.COPY_ORIGINALS,
            LocalizationMode.COPY_ALIGNED,
        )
        self.naming_box.setEnabled(is_copy)
        # Idempotência só faz sentido em modos de cópia.
        self.chk_skip_existing.setEnabled(is_copy)
        # Prefixo antigo só em REBASE_PREFIX.
        is_rebase = mode == LocalizationMode.REBASE_PREFIX
        self.rebase_box.setVisible(is_rebase)
        # Em REBASE_PREFIX, o "destino" é o NOVO prefixo (a pasta onde
        # as imagens estão NESTA máquina), não uma pasta de saída
        # autocontida. Renomeamos pra deixar a intenção clara.
        if is_rebase:
            self.target_box.setTitle("Prefixo novo (pasta nesta máquina)")
        else:
            self.target_box.setTitle("Pasta destino")

    def _on_naming_changed(self, idx: int):
        naming = self.cmb_naming.currentData()
        for n, _, h in _NAMING_OPTIONS:
            if n == naming:
                self.lbl_naming_help.setText(h)
                break

    def _on_browse(self):
        current = self.input_target.text().strip() or str(self._suggested_target)
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pasta destino", current,
        )
        if chosen:
            self.input_target.setText(chosen)

    # ── Resultado ──────────────────────────────────────────────────────────

    def get_options(self) -> dict:
        """Devolve as opções escolhidas. Chamar após `exec() == Accepted`.

        Levanta ValueError se a pasta destino estiver vazia, ou se o
        modo é REBASE_PREFIX e o prefixo antigo está vazio.
        """
        target = self.input_target.text().strip()
        if not target:
            raise ValueError("Pasta destino não pode estar vazia.")
        mode = self.cmb_mode.currentData()
        old_prefix: Optional[str] = None
        if mode == LocalizationMode.REBASE_PREFIX:
            old_prefix = self.input_old_prefix.text().strip()
            if not old_prefix:
                raise ValueError(
                    "Modo 'Atualizar prefixo' exige preencher o prefixo "
                    "antigo (a pasta na máquina onde o dataset foi criado)."
                )
        return {
            "mode": mode,
            "naming": self.cmb_naming.currentData(),
            "target_dir": Path(target),
            "skip_existing": self.chk_skip_existing.isChecked(),
            "old_prefix": old_prefix,
        }
