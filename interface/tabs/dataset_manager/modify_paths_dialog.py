"""Diálogo "Modificar Caminhos" (renomeado de "Localizar" — PR L1d).

Unifica todas as operações sobre os **caminhos das imagens** de um
dataset numa única janela:

  - **Corrigir subpasta (`path_root`)** — operação *só de metadados*,
    exclusiva de datasets RELATIVE. Repontar a subpasta onde as imagens
    estão, **sem mover nenhum arquivo**. Validável antes de aplicar.
  - **Reescrever caminhos (relativo)** / **Copiar originais** /
    **Substituir prefixo** — as três migrações *pesadas* do antigo
    Localizar, delegadas ao `DatasetLocalizationService` via worker.

O combo de operação é filtrado pelo `path_mode` do dataset: a correção
de `path_root` só aparece em RELATIVE (não há `path_root` em ABSOLUTE).

Fica em arquivo único (controller + view juntos) seguindo o padrão de
`schema_report_dialog.py` e `audit_log_dialog.py`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.localization_naming import NamingStrategy
from faceaudit.enums.path_mode import PathMode
from faceaudit.services.image_path_resolver import (
    normalize_stored_path,
    resolve_image_path,
)


class PathOp(str, Enum):
    """Operação escolhida na janela (nível de UI).

    `FIX_PATH_ROOT` é metadados-only (não dispara worker); as outras
    três mapeiam 1:1 para `LocalizationMode` e disparam o
    `DatasetLocalizationService`.
    """

    FIX_PATH_ROOT = "fix_path_root"
    REWRITE_RELATIVE = "rewrite_relative"
    COPY_ORIGINALS = "copy_originals"
    REBASE_PREFIX = "rebase_prefix"


# As 3 operações de migração mapeiam direto pro enum de contrato do
# service. FIX_PATH_ROOT não tem `LocalizationMode` — é tratado no
# controller sem worker.
_PATHOP_TO_LOCALIZATION_MODE = {
    PathOp.REWRITE_RELATIVE: LocalizationMode.REWRITE_RELATIVE,
    PathOp.COPY_ORIGINALS: LocalizationMode.COPY_ORIGINALS,
    PathOp.REBASE_PREFIX: LocalizationMode.REBASE_PREFIX,
}

# Rótulo curto por operação (item do combo).
_OP_LABEL = {
    PathOp.FIX_PATH_ROOT:
        "Corrigir subpasta (path_root) — sem mover arquivos",
    PathOp.REWRITE_RELATIVE: "Reescrever caminhos (sem cópia)",
    PathOp.COPY_ORIGINALS: "Copiar imagens originais",
    PathOp.REBASE_PREFIX: "Atualizar prefixo de caminho (sem cópia)",
}

# Texto de ajuda contextual por operação. Os três de migração
# preservam os textos do antigo `_MODE_OPTIONS`.
_OP_HELP = {
    PathOp.FIX_PATH_ROOT:
        "Corrige apenas o ponteiro <code>path_root</code> — a subpasta "
        "(relativa à pasta do parquet) onde as imagens estão. <b>Nenhum "
        "arquivo é movido ou copiado</b>: só o metadado é atualizado. "
        "Use quando as imagens já estão no lugar certo mas o "
        "<code>path_root</code> gravado está errado (ex: dataset copiado "
        "pra outra estrutura de pastas). Clique <b>Validar</b> para "
        "confirmar que os arquivos existem no novo caminho.",
    PathOp.COPY_ORIGINALS:
        "Copia as imagens pra &lt;pasta destino&gt;/&lt;nome do dataset&gt;"
        "_images/. As imagens originais ficam intocadas. Como cada dataset "
        "usa a sua própria subpasta, vários datasets podem ser modificados "
        "na MESMA pasta sem misturar imagens. Resultado totalmente portátil "
        "— pode ser movido/zipado e aberto em outra máquina sem quebrar.",
    PathOp.REWRITE_RELATIVE:
        "Apenas reescreve os caminhos no parquet como relativos à pasta "
        "destino. Nenhuma imagem é copiada. Requer que TODAS as imagens "
        "já estejam dentro da pasta destino — caso contrário, falha.",
    PathOp.REBASE_PREFIX:
        "Substitui um prefixo de path por outro em todas as faces. "
        "Caso de uso típico: dataset criado em outra máquina com paths "
        "absolutos do tipo <code>/home/colega/imgs/...</code> ou "
        "<code>C:\\Users\\alguem\\imgs\\...</code>, e nesta máquina a "
        "mesma estrutura existe em outro caminho (ex: "
        "<code>/Users/eu/data/imgs/...</code>). Não copia nada — só "
        "reescreve os paths. Sanity check: cada arquivo precisa existir "
        "no novo caminho.",
}

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


class ModifyPathsDialog(QtWidgets.QDialog):
    """Coleta a operação de caminhos escolhida. Modal.

    O resultado é lido via `get_result()` após `exec() == Accepted`.
    Para FIX_PATH_ROOT o controller aplica direto (metadados); para as
    demais dispara o worker de localização.
    """

    def __init__(
        self,
        dataset_name: str,
        suggested_target_dir: Path,
        parent: Optional[QtWidgets.QWidget] = None,
        detected_prefix: Optional[str] = None,
        *,
        path_mode: PathMode,
        path_root: Optional[str] = None,
        parquet_dir: Optional[Path] = None,
        dataset: Optional[Any] = None,
    ):
        super().__init__(parent)
        self._dataset_name = dataset_name
        self._suggested_target = suggested_target_dir
        self._detected_prefix = detected_prefix or ""
        self._path_mode = path_mode
        self._path_root = path_root
        self._parquet_dir = Path(parquet_dir) if parquet_dir else None
        self._dataset = dataset

        self.setWindowTitle(f"Modificar Caminhos — {dataset_name}")
        self.setModal(True)
        self.resize(660, 600)
        self._build_ui()
        self._on_op_changed(0)  # estado inicial coerente
        self._on_naming_changed(0)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "<b>Modificar Caminhos</b> — gerencia como o dataset aponta "
            "pras imagens em disco: corrige a subpasta relativa, converte "
            "paths absolutos em relativos, copia as imagens pra uma pasta "
            "autocontida (portátil — zip/USB/laudo) ou substitui um "
            "prefixo de caminho.",
            parent=self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # ── Estado atual (read-only) ──
        if self._path_mode == PathMode.RELATIVE:
            root_disp = self._path_root or "(raiz)"
            state_txt = (
                "<b>Estado atual:</b> Relativo · "
                f"Subpasta (path_root): <code>{root_disp}</code>"
            )
        else:
            state_txt = "<b>Estado atual:</b> Absoluto"
        self.lbl_state = QtWidgets.QLabel(state_txt, parent=self)
        self.lbl_state.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet(
            "color: #2c3e50; background: #eef3f8; border-radius: 4px; "
            "padding: 6px;"
        )
        layout.addWidget(self.lbl_state)

        # ── Operação ──
        op_box = QtWidgets.QGroupBox("Operação", parent=self)
        op_layout = QtWidgets.QFormLayout(op_box)
        self.cmb_op = QtWidgets.QComboBox(parent=op_box)
        for op in self._ops_for_mode():
            self.cmb_op.addItem(_OP_LABEL[op], userData=op)
        self.cmb_op.currentIndexChanged.connect(self._on_op_changed)
        op_layout.addRow("Operação:", self.cmb_op)

        self.lbl_op_help = QtWidgets.QLabel(parent=op_box)
        self.lbl_op_help.setWordWrap(True)
        self.lbl_op_help.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_op_help.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px;"
        )
        op_layout.addRow("", self.lbl_op_help)
        layout.addWidget(op_box)

        # ── Corrigir subpasta (só FIX_PATH_ROOT) ──
        self.pathroot_box = QtWidgets.QGroupBox(
            "Corrigir subpasta (path_root)", parent=self,
        )
        pr_layout = QtWidgets.QVBoxLayout(self.pathroot_box)
        pr_row = QtWidgets.QHBoxLayout()
        self.input_path_root = QtWidgets.QLineEdit(
            self._path_root or "", parent=self.pathroot_box,
        )
        self.input_path_root.setPlaceholderText(
            "subpasta relativa à pasta do parquet — vazio = raiz "
            "(ex: images)"
        )
        pr_row.addWidget(self.input_path_root)
        self.btn_pick_subfolder = QtWidgets.QPushButton(
            "📁 Procurar…", parent=self.pathroot_box,
        )
        self.btn_pick_subfolder.clicked.connect(self._on_pick_subfolder)
        pr_row.addWidget(self.btn_pick_subfolder)
        self.btn_validate_root = QtWidgets.QPushButton(
            "✔ Validar", parent=self.pathroot_box,
        )
        self.btn_validate_root.clicked.connect(self._on_validate)
        pr_row.addWidget(self.btn_validate_root)
        pr_layout.addLayout(pr_row)

        self.lbl_validation = QtWidgets.QLabel("", parent=self.pathroot_box)
        self.lbl_validation.setWordWrap(True)
        self.lbl_validation.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_validation.setStyleSheet("font-size: 11px; padding: 2px;")
        pr_layout.addWidget(self.lbl_validation)

        if self._parquet_dir is None:
            # Sem pasta de parquet (dataset RELATIVE nunca salvo) não dá
            # pra abrir um picker ancorado nem validar existência.
            self.btn_pick_subfolder.setEnabled(False)
            self.btn_pick_subfolder.setToolTip(
                "Dataset ainda não salvo em disco — digite a subpasta "
                "manualmente. Será persistida no próximo 'Salvar'."
            )
        layout.addWidget(self.pathroot_box)

        # ── Prefixo antigo (só em REBASE_PREFIX) ──
        self.rebase_box = QtWidgets.QGroupBox(
            "Prefixo antigo (a ser substituído)", parent=self,
        )
        rebase_layout = QtWidgets.QVBoxLayout(self.rebase_box)
        self.input_old_prefix = QtWidgets.QLineEdit(
            self._detected_prefix, parent=self.rebase_box,
        )
        self.input_old_prefix.setPlaceholderText(
            "ex: /home/colega/datasets/celeb"
        )
        rebase_layout.addWidget(self.input_old_prefix)
        self.lbl_rebase_help = QtWidgets.QLabel(parent=self.rebase_box)
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

        # ── Naming (só em COPY) ──
        self.naming_box = QtWidgets.QGroupBox(
            "Estratégia de nome", parent=self,
        )
        naming_layout = QtWidgets.QFormLayout(self.naming_box)
        self.cmb_naming = QtWidgets.QComboBox(parent=self.naming_box)
        for naming, label, _ in _NAMING_OPTIONS:
            self.cmb_naming.addItem(label, userData=naming)
        self.cmb_naming.currentIndexChanged.connect(self._on_naming_changed)
        naming_layout.addRow("Naming:", self.cmb_naming)

        self.lbl_naming_help = QtWidgets.QLabel(parent=self.naming_box)
        self.lbl_naming_help.setWordWrap(True)
        self.lbl_naming_help.setStyleSheet(
            "color: #555; font-size: 11px; padding: 4px;"
        )
        naming_layout.addRow("", self.lbl_naming_help)
        layout.addWidget(self.naming_box)

        # ── Destino ──
        # O label muda conforme a operação: em REBASE_PREFIX é "Prefixo
        # novo (pasta nesta máquina)", nos demais é "Pasta destino".
        self.target_box = QtWidgets.QGroupBox("Pasta destino", parent=self)
        target_layout = QtWidgets.QHBoxLayout(self.target_box)
        self.input_target = QtWidgets.QLineEdit(
            str(self._suggested_target), parent=self.target_box,
        )
        target_layout.addWidget(self.input_target)
        self.btn_browse = QtWidgets.QPushButton(
            "📁 Escolher…", parent=self.target_box,
        )
        self.btn_browse.clicked.connect(self._on_browse)
        target_layout.addWidget(self.btn_browse)
        layout.addWidget(self.target_box)

        # ── Idempotência ──
        self.chk_skip_existing = QtWidgets.QCheckBox(
            "Pular arquivos que já existem com o mesmo conteúdo "
            "(idempotência — permite re-rodar sem dor de cabeça)",
            parent=self,
        )
        self.chk_skip_existing.setChecked(True)
        layout.addWidget(self.chk_skip_existing)

        layout.addStretch()

        # ── Botões ──
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)\
            .setText("🗂 Modificar caminhos")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _ops_for_mode(self) -> list:
        """Operações oferecidas pro `path_mode` atual, em ordem.

        RELATIVE lidera com a correção leve de `path_root`; ABSOLUTE não
        tem `path_root` — só as migrações (default COPY_ORIGINALS).
        """
        if self._path_mode == PathMode.RELATIVE:
            return [
                PathOp.FIX_PATH_ROOT,
                PathOp.REWRITE_RELATIVE,
                PathOp.COPY_ORIGINALS,
                PathOp.REBASE_PREFIX,
            ]
        return [
            PathOp.COPY_ORIGINALS,
            PathOp.REWRITE_RELATIVE,
            PathOp.REBASE_PREFIX,
        ]

    @staticmethod
    def _relativize_subfolder(
        chosen: Any, parquet_dir: Any,
    ) -> Optional[str]:
        """Converte a pasta escolhida em `path_root` relativo à pasta do
        parquet. Retorna None se `chosen` estiver FORA de `parquet_dir`.
        `""` representa a própria raiz (path_root vazio)."""
        try:
            rel = Path(chosen).resolve().relative_to(
                Path(parquet_dir).resolve()
            )
        except ValueError:
            return None
        s = str(rel)
        if s == ".":
            return ""
        return normalize_stored_path(s)

    def _current_op(self) -> PathOp:
        return self.cmb_op.currentData()

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_op_changed(self, idx: int):
        op = self.cmb_op.currentData()
        if op is None:
            return
        self.lbl_op_help.setText(_OP_HELP.get(op, ""))

        is_fix = op == PathOp.FIX_PATH_ROOT
        is_copy = op == PathOp.COPY_ORIGINALS
        is_rebase = op == PathOp.REBASE_PREFIX

        # FIX_PATH_ROOT é só-metadados → esconde tudo que é migração.
        self.pathroot_box.setVisible(is_fix)
        self.rebase_box.setVisible(is_rebase)
        self.naming_box.setVisible(not is_fix)
        self.naming_box.setEnabled(is_copy)   # renomear só faz sentido em cópia
        self.target_box.setVisible(not is_fix)
        self.chk_skip_existing.setVisible(not is_fix)
        self.chk_skip_existing.setEnabled(is_copy)

        if not is_fix:
            suggested_str = str(self._suggested_target)
            if is_rebase:
                # Em REBASE_PREFIX o "destino" é o NOVO prefixo (a pasta
                # onde as imagens estão NESTA máquina), não uma saída
                # autocontida. Limpa o sugerido default se intacto.
                self.target_box.setTitle("Prefixo novo (pasta nesta máquina)")
                if self.input_target.text().strip() == suggested_str:
                    self.input_target.clear()
                self.input_target.setPlaceholderText(
                    "ex: /Users/perito/data/celeb — clique "
                    "'📁 Escolher…'"
                )
            else:
                self.target_box.setTitle("Pasta destino")
                self.input_target.setPlaceholderText("")
                if not self.input_target.text().strip():
                    self.input_target.setText(suggested_str)

        ok_btn = self.buttons.button(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        ok_btn.setText(
            "✔ Aplicar path_root" if is_fix else "🗂 Modificar caminhos"
        )

    def _on_naming_changed(self, idx: int):
        naming = self.cmb_naming.currentData()
        for n, _, h in _NAMING_OPTIONS:
            if n == naming:
                self.lbl_naming_help.setText(h)
                break

    def _on_pick_subfolder(self):
        if self._parquet_dir is None:
            return
        start = str(self._parquet_dir)
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Subpasta das imagens (dentro da pasta do parquet)",
            start,
        )
        if not chosen:
            return
        rel = self._relativize_subfolder(chosen, self._parquet_dir)
        if rel is None:
            QtWidgets.QMessageBox.warning(
                self, "Fora da pasta do dataset",
                "A subpasta precisa estar DENTRO da pasta do parquet:\n"
                f"{self._parquet_dir}\n\nEscolha uma pasta abaixo dela.",
            )
            return
        self.input_path_root.setText(rel)
        self._on_validate()

    def _on_validate(self):
        candidate = normalize_stored_path(
            self.input_path_root.text().strip()
        )
        ok, msg = self._validate_candidate(candidate)
        color = "#2e7d32" if ok else "#c62828"
        icon = "✅" if ok else "⚠"
        self.lbl_validation.setText(
            f"<span style='color:{color}'>{icon} {msg}</span>"
        )

    def _validate_candidate(self, candidate: str) -> tuple:
        """Valida um `path_root` candidato resolvendo a 1ª face com
        imagem, SEM mutar o dataset. Retorna (ok: bool, msg: str)."""
        if self._dataset is None or self._parquet_dir is None:
            return True, (
                "Validação indisponível (dataset ainda não salvo em "
                "disco)."
            )
        faces = getattr(self._dataset, "faces", None) or []
        sample = next(
            (f for f in faces if getattr(f, "image_path", None)), None,
        )
        if sample is None:
            return True, "Nenhuma face com caminho de imagem para validar."
        resolved = resolve_image_path(
            sample,
            parquet_dir=self._parquet_dir,
            path_mode=PathMode.RELATIVE,
            path_root=candidate,
        )
        if resolved.exists():
            return True, (
                "O arquivo de exemplo existe:<br>"
                f"<code>{resolved}</code>"
            )
        return False, (
            "O arquivo de exemplo NÃO existe:<br>"
            f"<code>{resolved}</code>"
        )

    def _on_browse(self):
        current = self.input_target.text().strip() or str(
            self._suggested_target
        )
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pasta destino", current,
        )
        if chosen:
            self.input_target.setText(chosen)

    # ── Resultado ──────────────────────────────────────────────────────────

    def get_result(self) -> dict:
        """Devolve a operação escolhida. Chamar após `exec() == Accepted`.

        Para FIX_PATH_ROOT retorna `{"op", "path_root"}` (subpasta
        normalizada para POSIX). Para as demais retorna as chaves de
        migração `{"op","mode","naming","target_dir","skip_existing",
        "old_prefix"}` (mesmo shape que o `_iniciar_worker_*` espera).

        Levanta ValueError se a pasta destino estiver vazia (migrações),
        ou se a operação é REBASE_PREFIX e o prefixo antigo está vazio.
        """
        op = self.cmb_op.currentData()

        if op == PathOp.FIX_PATH_ROOT:
            return {
                "op": op,
                "path_root": normalize_stored_path(
                    self.input_path_root.text().strip()
                ),
            }

        target = self.input_target.text().strip()
        if not target:
            raise ValueError("Pasta destino não pode estar vazia.")
        old_prefix: Optional[str] = None
        if op == PathOp.REBASE_PREFIX:
            old_prefix = self.input_old_prefix.text().strip()
            if not old_prefix:
                raise ValueError(
                    "Operação 'Atualizar prefixo' exige preencher o "
                    "prefixo antigo (a pasta na máquina onde o dataset "
                    "foi criado)."
                )
        return {
            "op": op,
            "mode": _PATHOP_TO_LOCALIZATION_MODE[op],
            "naming": self.cmb_naming.currentData(),
            "target_dir": Path(target),
            "skip_existing": self.chk_skip_existing.isChecked(),
            "old_prefix": old_prefix,
        }

    # Alias de compat — callers antigos chamavam `get_options()`.
    get_options = get_result
